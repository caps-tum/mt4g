#pragma once

#include <vector>
#include <hip/hip_runtime.h>

#include "utils/errorHandling.hpp"

namespace util {
/**
 * @brief Create a texture object wrapping linear device memory.
 */
template<typename T> hipTextureObject_t createTextureObject(T* data, size_t elementCount) {
    hipTextureObject_t tex = 0;

    // Resource description: linear memory of T
    hipResourceDesc resDesc {
        .resType = hipResourceTypeLinear,
        .res = {
            .linear = {
                .devPtr       = data,                           // pointer to T
                .desc         = hipCreateChannelDesc<T>(),      // channel for T
                .sizeInBytes  = elementCount * sizeof(T)        // total bytes
            }
        }
    };

    // Texture descriptor: default clamped point sampling
    hipTextureDesc texDesc {
        .addressMode      = { hipAddressModeClamp,
                              hipAddressModeClamp,
                              hipAddressModeClamp },
        .filterMode       = hipFilterModePoint,
        .readMode         = hipReadModeElementType,
        .sRGB             = 0,
        .borderColor      = { 0.f, 0.f, 0.f, 0.f },
        .normalizedCoords = 0,
        .maxAnisotropy    = 0,
        .mipmapFilterMode = hipFilterModePoint,
        .mipmapLevelBias  = 0.f,
        .minMipmapLevelClamp = 0.f,
        .maxMipmapLevelClamp = 0.f
    };

    util::hipCheck(
        hipCreateTextureObject(&tex, &resDesc, &texDesc, nullptr)
    );

    return tex;
}

/**
 * @brief Allocate device memory and copy data from host.
 */
template<typename T> T* allocateGPUMemory(const std::vector<T>& data) {
    T* devicePtr = nullptr;
    size_t bytes = data.size() * sizeof(T);
    util::hipCheck(hipMalloc(&devicePtr, bytes));
    util::hipCheck(hipMemcpy(devicePtr, data.data(), bytes, hipMemcpyHostToDevice));
    return devicePtr;
}

/**
 * @brief Allocate uninitialised device memory for @p numElems elements.
 */
template <typename T = uint32_t> T* allocateGPUMemory(size_t numElems) {
    T* devicePtr = nullptr;
    util::hipCheck(hipMalloc(&devicePtr, numElems * sizeof(T)));
    return devicePtr;
}

/**
 * @brief Copy @p count elements from device to host memory.
 */
template<typename T> std::vector<T> copyFromDevice(const T* devicePtr, size_t count) {
    std::vector<T> hostVec(count);
    if (count == 0 || devicePtr == nullptr) return hostVec;
    util::hipCheck(hipMemcpy(hostVec.data(), devicePtr, count * sizeof(T), hipMemcpyDeviceToHost));
    return hostVec;
}

} // namespace util

