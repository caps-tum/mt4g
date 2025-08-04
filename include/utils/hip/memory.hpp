#pragma once

#include <vector>
#include <hip/hip_runtime.h>

#include "utils/errorHandling.hpp"

namespace util {

hipTextureObject_t createTextureObject(uint32_t *data, size_t dataSize);

template<typename T> T* allocateGPUMemory(const std::vector<T>& data) {
    T* devicePtr = nullptr;
    size_t bytes = data.size() * sizeof(T);
    util::hipCheck(hipMalloc(&devicePtr, bytes));
    util::hipCheck(hipMemcpy(devicePtr, data.data(), bytes, hipMemcpyHostToDevice));
    return devicePtr;
}

template <typename T = uint32_t> T* allocateGPUMemory(size_t numElems) {
    T* devicePtr = nullptr;
    util::hipCheck(hipMalloc(&devicePtr, numElems * sizeof(T)));
    return devicePtr;
}

template<typename T> std::vector<T> copyFromDevice(const T* devicePtr, size_t count) {
    std::vector<T> hostVec(count);
    if (count == 0 || devicePtr == nullptr) return hostVec;
    util::hipCheck(hipMemcpy(hostVec.data(), devicePtr, count * sizeof(T), hipMemcpyDeviceToHost));
    return hostVec;
}

} // namespace util

