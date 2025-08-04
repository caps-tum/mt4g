#include "utils/util.hpp"
#include "benchmarks/base.hpp"   

#include <hip/hip_runtime.h>

__global__ void cuShareScalarL1Kernel(uint32_t *resultBuffer, uint32_t logicalId) {
    resultBuffer[logicalId] = __getPhysicalCUId();
}

namespace util {
    std::vector<uint32_t> getLogicalToPhysicalCUsLUT() {
        uint32_t numCUs = getNumberOfComputeUnits();
        uint32_t *d_resultBuffer = util::allocateGPUMemory(numCUs);

        for (uint32_t i = 0; i < numCUs; ++i) {
            hipStream_t stream = util::createStreamForCU(i);
            cuShareScalarL1Kernel<<<1, 1, 0, stream>>>(d_resultBuffer, i); // Not reliable on CDNA 3...
            // wait for this kernel to finish before destroying the stream
            util::hipCheck(hipStreamSynchronize(stream));
            util::hipCheck(hipStreamDestroy(stream));
        }

        std::vector<uint32_t> resultBuffer = util::copyFromDevice(d_resultBuffer, numCUs);
        util::hipCheck(hipFree(d_resultBuffer));
        return resultBuffer;
    }
} // namespace util
