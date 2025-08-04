#pragma once

#include <vector>
#include <hip/hip_runtime.h>
#include "utils/errorHandling.hpp"
#include "utils/hip/device.hpp"

namespace util {

inline float getElapsedTimeMs(hipEvent_t start, hipEvent_t stop) {
    util::hipCheck(hipEventSynchronize(start));
    util::hipCheck(hipEventSynchronize(stop));
    float ms = 0.0f;
    util::hipCheck(hipEventElapsedTime(&ms, start, stop));
    return ms;
}

inline hipEvent_t createHipEvent() {
    hipEvent_t event;
    util::hipCheck(hipEventCreate(&event));
    return event;
}

inline bool supportsCooperativeLaunch() {
    static bool supported = []() {
        int device;
        util::hipCheck(hipGetDevice(&device));
        int val = 0;
        util::hipCheck(hipDeviceGetAttribute(
            &val,
            hipDeviceAttributeCooperativeLaunch,
            device));
        return val != 0;
    }();
    return supported;
}

inline hipDeviceProp_t getDeviceProperties() {
    static hipDeviceProp_t deviceProperties = []() {
        int device;
        util::hipCheck(hipGetDevice(&device));
        hipDeviceProp_t props;
        util::hipCheck(hipGetDeviceProperties(&props, device));
        return props;
    }();
    return deviceProperties;
}

inline uint32_t getMaxThreadsPerBlock() {
    static uint32_t maxThreads = [](){
        int32_t device;
        util::hipCheck(hipGetDevice(&device));
        int32_t v;
        util::hipCheck(hipDeviceGetAttribute(&v, hipDeviceAttributeMaxThreadsPerBlock, device));
        return v;
    }();
    return maxThreads;
}

template <typename KernelFunc>
inline uint32_t getMaxActiveBlocks(KernelFunc kernel, uint32_t blockSize, size_t dynamicSharedMemoryBytes = 0U) {
    int32_t device;
    util::hipCheck(hipGetDevice(&device));
    hipDeviceProp_t prop;
    util::hipCheck(hipGetDeviceProperties(&prop, device));
    int32_t blocksPerSM = 0;
    util::hipCheck(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocksPerSM,
        kernel,
        static_cast<int>(blockSize),
        dynamicSharedMemoryBytes));
    return static_cast<uint32_t>(blocksPerSM) * static_cast<uint32_t>(prop.multiProcessorCount);
}

inline hipStream_t createStreamForCU(int32_t cuIdx) {
    #ifdef __HIP_PLATFORM_AMD__
    int32_t dev = 0;
    util::hipCheck(hipGetDevice(&dev));
    int32_t numCUs = 0;
    util::hipCheck(hipDeviceGetAttribute(&numCUs, hipDeviceAttributeMultiprocessorCount, dev));
    assert(cuIdx >= 0 && cuIdx < numCUs);
    const int32_t len = (numCUs + 31) / 32;
    std::vector<uint32_t> mask(len, 0u);
    const int32_t word = cuIdx / 32;
    const int32_t bit  = cuIdx % 32;
    mask[word] = (1u << bit);
    hipStream_t stream;
    util::hipCheck(hipExtStreamCreateWithCUMask(&stream, static_cast<uint32_t>(mask.size()), mask.data()));
    return stream;
    #endif
    #ifdef __HIP_PLATFORM_NVIDIA__
    hipStream_t stream;
    util::hipCheck(hipStreamCreate(&stream));
    return stream;
    #endif
}

inline hipStream_t createStreamForCUs(const std::vector<uint32_t>& cuIdxs) {
    #ifdef __HIP_PLATFORM_AMD__
    int32_t dev = 0;
    util::hipCheck(hipGetDevice(&dev));
    int32_t numCUs = 0;
    util::hipCheck(hipDeviceGetAttribute(&numCUs, hipDeviceAttributeMultiprocessorCount, dev));
    for (int32_t cuIdx : cuIdxs) {
        assert(cuIdx >= 0 && cuIdx < numCUs);
    }
    const int32_t len = (numCUs + 31) / 32;
    std::vector<uint32_t> mask(len, 0u);
    for (int32_t cuIdx : cuIdxs) {
        int32_t word = cuIdx / 32;
        int32_t bit  = cuIdx % 32;
        mask[word] |= (1u << bit);
    }
    hipStream_t stream;
    util::hipCheck(hipExtStreamCreateWithCUMask(&stream, static_cast<uint32_t>(mask.size()), mask.data()));
    return stream;
    #endif
    #ifdef __HIP_PLATFORM_NVIDIA__
    hipStream_t stream;
    util::hipCheck(hipStreamCreate(&stream));
    return stream;
    #endif
}

} // namespace util

