#include "benchmarks/benchmark.hpp"
#include "benchmarks/bandwidth/loadBatch.hpp"
#include "utils/util.hpp"

#include <vector>

static constexpr auto WARMUP_REPS = 128;
static_assert(MIN_REPS % MAX_LOADS_PER_BATCH_SINGLE_BLOCK == 0 && WARMUP_REPS % MAX_LOADS_PER_BATCH_SINGLE_BLOCK == 0, "loads per batch must divide the repetitions");


#ifdef __HIP_PLATFORM_NVIDIA__
// Like the NVIDIA L1 batch, but through the texture path (tex1Dfetch).
// All four components are used, otherwise the fetch is narrowed to one component.
template <uint32_t LoadsPerBatch>
__device__ __forceinline__ uint32_t nvidiaTextureLoadBatch(hipTextureObject_t tex, size_t index)
{
    uint32_t dummy = 0;

    #pragma unroll
    for (uint32_t k = 0; k < LoadsPerBatch; ++k)
    {
        const int4 loaded = tex1Dfetch<int4>(tex, static_cast<int>(index + k * (BATCH_STRIDE_BYTES / sizeof(int4))));
        dummy ^= static_cast<uint32_t>(loaded.x ^ loaded.y ^ loaded.z ^ loaded.w);
    }

    return dummy;
}
#endif


template <uint32_t LoadsPerBatch>
__global__ void textureReadBandwidthKernel([[maybe_unused]] hipTextureObject_t tex, uint32_t* __restrict__ dst, uint64_t* __restrict__ timingResult, size_t numElements, size_t reps)
{
    CHECK_LOADS_PER_BATCH(LoadsPerBatch, MAX_LOADS_PER_BATCH_SINGLE_BLOCK);
    const uint32_t blockStride = blockDim.x;

    uint32_t dummy = 0;
    uint64_t start = 0;

    __syncthreads();
    if (threadIdx.x == 0) { start = __globaltimer(); }
    __syncthreads();

    for (size_t rep = 0; rep < reps; rep += LoadsPerBatch)
    {
        for (size_t i = threadIdx.x; i < numElements; i += blockStride)
        {
            #ifdef __HIP_PLATFORM_NVIDIA__
            dummy ^= nvidiaTextureLoadBatch<LoadsPerBatch>(tex, i);
            #endif
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) { *timingResult = __globaltimer() - start; }

    dst[threadIdx.x] = dummy; // prevent dead code elimination
}


// The kernel instance for a number of loads per batch
static auto textureReadBandwidthKernelFor(uint32_t loadsPerBatch)
{
    switch (loadsPerBatch)
    {
        case 1:    return textureReadBandwidthKernel<1>;
        case 2:    return textureReadBandwidthKernel<2>;
        case 4:    return textureReadBandwidthKernel<4>;
        case 8:    return textureReadBandwidthKernel<8>;
        case 16:   return textureReadBandwidthKernel<16>;
        case 32:   return textureReadBandwidthKernel<32>;
        case 64:   return textureReadBandwidthKernel<64>;
        case 128:  return textureReadBandwidthKernel<128>;
    }
    throw std::invalid_argument("Unsupported number of loads per batch: " + std::to_string(loadsPerBatch));
}


// One block, timed inside the kernel with the constant-rate timer, so the launch overhead is not measured
static std::tuple<double, double> textureReadBandwidthLauncher(size_t arraySizeBytes, uint32_t numThreads, uint32_t loadsPerBatch, size_t reps)
{
    const size_t numElements = arraySizeBytes / sizeof(int4);
    const size_t numElementsPadded = (arraySizeBytes + BATCH_PADDING_BYTES) / sizeof(int4);
    const auto kernel = textureReadBandwidthKernelFor(loadsPerBatch);

    int4 *d_srcArr = util::allocateGPUMemory<int4>(numElementsPadded);
    uint32_t *d_dstArr = util::allocateGPUMemory<uint32_t>(numThreads);
    uint64_t *d_timingResult = util::allocateGPUMemory<uint64_t>(1);

    hipTextureObject_t tex = util::createTextureObject<int4>(d_srcArr, numElementsPadded);

    const double timeS = timeSingleBlockKernel(d_timingResult, WARMUP_REPS, reps, 0, [&](size_t r) {
        kernel<<<1, numThreads>>>(tex, d_dstArr, d_timingResult, numElements, r);
    });

    util::hipCheck(hipDestroyTextureObject(tex));
    util::hipCheck(hipFree(d_srcArr));
    util::hipCheck(hipFree(d_dstArr));
    util::hipCheck(hipFree(d_timingResult));

    const double dataGiB = (double) arraySizeBytes * reps / (1 * GiB);

    return {timeS, timeS > 0.0 ? dataGiB / timeS : 0.0};
}


namespace benchmark
{
    namespace nvidia
    {
        CacheBandwidthResult measureTextureReadBandwidthSweep(size_t arraySizeBytes)
        {
            return sweepSingleBlockBandwidth(arraySizeBytes, [&](uint32_t numThreads, uint32_t loadsPerBatch, size_t reps) {
                return textureReadBandwidthLauncher(arraySizeBytes, numThreads, loadsPerBatch, reps);
            });
        }
    }
}
