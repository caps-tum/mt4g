#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"

#include <vector>
#include <map>
#include <numeric>
#include <optional>

static constexpr auto MS_PER_SECOND = 1000.0; // ms
static constexpr auto ROUNDS = DEFAULT_ROUNDS; // rounds

__global__ void l3WriteBandwidthKernel(uint32v4* __restrict__ dst, size_t n) {
    uint32_t tid = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    uint32_t stride = static_cast<uint32_t>(gridDim.x * blockDim.x);

    uint32v4 dummy = {tid, tid + 1, tid + 2, tid + 3};

    for (size_t j = 0; j < blockDim.x; ++j) {
        tid = (((blockIdx.x + j) * blockDim.x) + threadIdx.x) % stride;

        for (size_t i = tid; i < n; i += stride) {
#ifdef __HIP_PLATFORM_NVIDIA__
            asm volatile(
                "st.global.wb.v4.u32 [%0], {%1,%2,%3,%4};\n"
                :
                : "l"(dst + i), "r"(dummy.x), "r"(dummy.y), "r"(dummy.z), "r"(dummy.w)
            );
#endif
#ifdef __HIP_PLATFORM_AMD__
            asm volatile(
                "flat_store_dwordx4 %0, %1 " GLC_SLC "\n"
                :
                : "v"(dst + i), "v"(dummy)
                : "memory"
            );
#endif
        }
    }
}

double l3WriteBandwidthLauncher(size_t arraySizeBytes) {
    util::hipCheck(hipDeviceReset());

    uint32_t maxThreadsPerBlock = util::getMaxThreadsPerBlock();
    uint32_t maxBlocks = util::getNumberOfComputeUnits();

    uint32v4* d_dstArr = util::allocateGPUMemory<uint32v4>(arraySizeBytes / sizeof(uint32v4));

    l3WriteBandwidthKernel<<<maxBlocks, maxThreadsPerBlock>>>(d_dstArr, arraySizeBytes / sizeof(uint32v4));

    auto start = util::createHipEvent();
    auto end = util::createHipEvent();

    util::hipCheck(hipDeviceSynchronize());
    util::hipCheck(hipEventRecord(start));
    l3WriteBandwidthKernel<<<maxBlocks, maxThreadsPerBlock>>>(d_dstArr, arraySizeBytes / sizeof(uint32v4));
    util::hipCheck(hipEventRecord(end));
    util::hipCheck(hipDeviceSynchronize());

    return util::getElapsedTimeMs(start, end) / maxThreadsPerBlock;
}

namespace benchmark {
    namespace amd {
        double measureL3WriteBandwidth(size_t l3SizeBytes) {
            double testSizeGiB = static_cast<double>(l3SizeBytes) / (1 * GiB);

            std::vector<double> results(ROUNDS);
            for (uint32_t i = 0; i < ROUNDS; ++i) {
                results[i] = l3WriteBandwidthLauncher(l3SizeBytes) / MS_PER_SECOND;
            }

            return testSizeGiB / util::average(results);
        }
    }
}
