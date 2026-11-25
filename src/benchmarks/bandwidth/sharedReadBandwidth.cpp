#include "benchmarks/benchmark.hpp"
#include "benchmarks/bandwidth/loadBatch.hpp"
#include "utils/util.hpp"

#include <vector>

static constexpr auto WARMUP_REPS = MAX_LOADS_PER_BATCH_SINGLE_BLOCK;
static_assert(MIN_REPS % MAX_LOADS_PER_BATCH_SINGLE_BLOCK == 0, "loads per batch must divide the repetitions");


template <uint32_t LoadsPerBatch>
__global__ void sharedReadBandwidthKernel(uint32_t* __restrict__ dst, uint64_t* __restrict__ timingResult, uint32_t numElements, size_t reps)
{
    CHECK_LOADS_PER_BATCH(LoadsPerBatch, MAX_LOADS_PER_BATCH_SINGLE_BLOCK);
    const uint32_t blockStride = blockDim.x; // read once, on AMD the "memory" clobber of the asm would reload it in every iteration

    extern __shared__ uint32v4 memory[];

    for (uint32_t i = threadIdx.x; i < numElements + SHARED_BATCH_PADDING_BYTES / sizeof(uint32v4); i += blockStride)
    {
        memory[i] = {0, 0, 0, 0};
    }

    #ifdef __HIP_PLATFORM_AMD__
    const uint32_t base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(memory)); // byte address in the LDS
    #endif

    #ifdef __HIP_PLATFORM_NVIDIA__
    uint64_t base;
    asm volatile ("cvta.to.shared.u64 %0, %1;" : "=l"(base) : "l"(memory));
    #endif

    uint32_t dummy = 0;
    uint64_t start = 0;

    __syncthreads();
    if (threadIdx.x == 0) { start = __globaltimer(); }
    __syncthreads();

    for (size_t rep = 0; rep < reps; rep += LoadsPerBatch)
    {
        for (uint32_t i = threadIdx.x; i < numElements; i += blockStride)
        {
            #ifdef __HIP_PLATFORM_AMD__
            amdSharedLoadBatch<LoadsPerBatch>(base + i * sizeof(uint32v4));
            #endif

            #ifdef __HIP_PLATFORM_NVIDIA__
            dummy ^= nvidiaSharedLoadBatch<LoadsPerBatch>(base + i * sizeof(uint32v4));
            #endif
        }
    }

    __syncthreads();
    if (threadIdx.x == 0) { *timingResult = __globaltimer() - start; }

    dst[threadIdx.x] = dummy; // prevent dead code elimination
}


// The kernel instance for a number of loads per batch
static auto sharedReadBandwidthKernelFor(uint32_t loadsPerBatch)
{
    switch (loadsPerBatch)
    {
        case 1:    return sharedReadBandwidthKernel<1>;
        case 2:    return sharedReadBandwidthKernel<2>;
        case 4:    return sharedReadBandwidthKernel<4>;
        case 8:    return sharedReadBandwidthKernel<8>;
        case 16:   return sharedReadBandwidthKernel<16>;
        case 32:   return sharedReadBandwidthKernel<32>;
        case 64:   return sharedReadBandwidthKernel<64>;
        case 128:  return sharedReadBandwidthKernel<128>;
    }
    throw std::invalid_argument("Unsupported number of loads per batch: " + std::to_string(loadsPerBatch));
}


// One block, timed inside the kernel with the constant-rate timer, so the launch overhead is not measured
static std::tuple<double, double> sharedReadBandwidthLauncher(uint32_t arraySizeBytes, uint32_t numThreads, uint32_t loadsPerBatch, size_t reps)
{
    const uint32_t numElements = arraySizeBytes / sizeof(uint32v4);
    const auto kernel = sharedReadBandwidthKernelFor(loadsPerBatch);

    uint32_t *d_dstArr = util::allocateGPUMemory<uint32_t>(numThreads);
    uint64_t *d_timingResult = util::allocateGPUMemory<uint64_t>(1);

    const double timeS = timeSingleBlockKernel(d_timingResult, WARMUP_REPS, reps, 0, [&](size_t r) {
        kernel<<<1, numThreads, arraySizeBytes + SHARED_BATCH_PADDING_BYTES>>>(d_dstArr, d_timingResult, numElements, r);
    });

    util::hipCheck(hipFree(d_dstArr));
    util::hipCheck(hipFree(d_timingResult));

    const double dataGiB = (double) arraySizeBytes * reps / (1 * GiB);

    return {timeS, timeS > 0.0 ? dataGiB / timeS : 0.0};
}


namespace benchmark
{
    CacheBandwidthResult measureSharedReadBandwidthSweep(uint32_t arraySizeBytes)
    {
        return sweepSingleBlockBandwidth(arraySizeBytes, [&](uint32_t numThreads, uint32_t loadsPerBatch, size_t reps) {
            return sharedReadBandwidthLauncher(arraySizeBytes, numThreads, loadsPerBatch, reps);
        });
    }
}
