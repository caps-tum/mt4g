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
            hipStream_t stream = util::createStreamForCU(i); // Not supported on NVIDIA, hence will not work
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


/* IT could work something like that on CDNA 3. However due to only having access to a virtualized GPU we could not verify it
#include "utils/util.hpp"
#include "benchmarks/base.hpp"   

#include <hip/hip_runtime.h>

__global__ void translationKernel(uint32_t *resultBuffer, uint32_t logicalId, uint32_t targetXCCId) {
    uint32_t xcc;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(xcc));
    if (xcc == targetXCCId) {
        uint32_t newId = __smid();
        uint32_t old = atomicCAS(&resultBuffer[logicalId], 0u, newId);
        if (old != 0u) {
            printf("Debug: Slot %u on XCD %u already filled with %u, attempted %u\n", logicalId, targetXCCId, old, newId);
        }
    }
}


namespace util {
    std::vector<uint32_t> getLogicalToPhysicalCUsLUT() {
        std::map<uint32_t, std::vector<uint32_t>> mappingsPerXCD;

        for (uint32_t xcdId = 0; xcdId < util::getNumXCDs(); ++xcdId) {
            uint32_t numCUs = 304;

            uint32_t *d_resultBuffer = util::allocateGPUMemory(numCUs); 
            util::hipCheck(hipMemset(d_resultBuffer, 0, numCUs * sizeof(uint32_t)));

            for (uint32_t cuIdOnXCD = 0; cuIdOnXCD < numCUs; ++cuIdOnXCD) {
                hipStream_t stream = util::createStreamForCU(cuIdOnXCD);
                translationKernel<<<util::getNumXCDs() * 2, 1, 0, stream>>>(d_resultBuffer, cuIdOnXCD, cuIdOnXCD / 8);

                util::hipCheck(hipStreamSynchronize(stream));
                util::hipCheck(hipStreamDestroy(stream));
            }

            std::vector<uint32_t> resultBuffer = util::copyFromDevice(d_resultBuffer, numCUs);
            util::hipCheck(hipFree(d_resultBuffer));
            mappingsPerXCD[xcdId] = resultBuffer;
        }

        util::printMap(mappingsPerXCD);

        return  {};
    }
} // namespace util


*/