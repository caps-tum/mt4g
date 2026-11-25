#include "benchmarks/benchmark.hpp"
#include "benchmarks/bandwidth/loadBatch.hpp"
#include "utils/util.hpp"

#include <vector>

static constexpr auto SIZE = 32 * KiB;
static constexpr auto NUM_ELEMENTS = SIZE / sizeof(uint32v4);
static constexpr auto WARMUP_REPS = MAX_LOADS_PER_BATCH_SINGLE_BLOCK;
static_assert(MIN_REPS % MAX_LOADS_PER_BATCH_SINGLE_BLOCK == 0, "loads per batch must divide the repetitions");


// Same as the dynamic shared read kernel, but with a statically sized array
template <uint32_t LoadsPerBatch>
__global__ void sharedReadBandwidthStaticKernel(uint32_t* __restrict__ dst, uint64_t* __restrict__ timingResult, size_t reps)
{
    CHECK_LOADS_PER_BATCH(LoadsPerBatch, MAX_LOADS_PER_BATCH_SINGLE_BLOCK);
    const uint32_t blockStride = blockDim.x; // read once, on AMD the "memory" clobber of the asm would reload it in every iteration

    __shared__ uint32v4 memory[NUM_ELEMENTS + SHARED_BATCH_PADDING_BYTES / sizeof(uint32v4)];

    for (uint32_t i = threadIdx.x; i < NUM_ELEMENTS + SHARED_BATCH_PADDING_BYTES / sizeof(uint32v4); i += blockStride)
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
        for (uint32_t i = threadIdx.x; i < NUM_ELEMENTS; i += blockStride)
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
static auto sharedReadBandwidthStaticKernelFor(uint32_t loadsPerBatch)
{
    switch (loadsPerBatch)
    {
        case 1:    return sharedReadBandwidthStaticKernel<1>;
        case 2:    return sharedReadBandwidthStaticKernel<2>;
        case 4:    return sharedReadBandwidthStaticKernel<4>;
        case 8:    return sharedReadBandwidthStaticKernel<8>;
        case 16:   return sharedReadBandwidthStaticKernel<16>;
        case 32:   return sharedReadBandwidthStaticKernel<32>;
        case 64:   return sharedReadBandwidthStaticKernel<64>;
        case 128:  return sharedReadBandwidthStaticKernel<128>;
    }
    throw std::invalid_argument("Unsupported number of loads per batch: " + std::to_string(loadsPerBatch));
}


// One block, timed inside the kernel with the constant-rate timer, so the launch overhead is not measured
static std::tuple<double, double> sharedReadBandwidthStaticLauncher(uint32_t numThreads, uint32_t loadsPerBatch, size_t reps)
{
    const auto kernel = sharedReadBandwidthStaticKernelFor(loadsPerBatch);

    uint32_t *d_dstArr = util::allocateGPUMemory<uint32_t>(numThreads);
    uint64_t *d_timingResult = util::allocateGPUMemory<uint64_t>(1);

    const double timeS = timeSingleBlockKernel(d_timingResult, WARMUP_REPS, reps, 0, [&](size_t r) {
        kernel<<<1, numThreads>>>(d_dstArr, d_timingResult, r);
    });

    util::hipCheck(hipFree(d_dstArr));
    util::hipCheck(hipFree(d_timingResult));

    const double dataGiB = (double) SIZE * reps / (1 * GiB);

    return {timeS, timeS > 0.0 ? dataGiB / timeS : 0.0};
}


namespace benchmark
{
    CacheBandwidthResult measureSharedReadBandwidthStaticSweep()
    {
        return sweepSingleBlockBandwidth(SIZE, [&](uint32_t numThreads, uint32_t loadsPerBatch, size_t reps) {
            return sharedReadBandwidthStaticLauncher(numThreads, loadsPerBatch, reps);
        });
    }
}
