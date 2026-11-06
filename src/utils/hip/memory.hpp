#pragma once

#include <vector>
#include <cstdlib>
#include <new>
#include <hip/hip_runtime.h>

#include "utils/errorHandling.hpp"

namespace util {
    /**
     * @brief Allocator selector for L3/main-memory latency and bandwidth sweeps.
     */
    enum class AllocatorType {
        HipMalloc,         // hipMalloc — baseline device memory
        HipMallocManaged,  // hipMallocManaged — demand-paged unified memory
        HipHostMalloc,     // hipHostMalloc — pinned host memory accessible from GPU
        Malloc             // ::malloc — CPU heap accessible from GPU via HMM on MI300A;
                           // requires XNACK=1
    };

    /**
     * @brief Allocate uninitialised memory using the selected allocator.
     */
    template <typename T = uint32_t> T* allocateMemory(size_t numElems, AllocatorType type) {
        T* ptr = nullptr;
        size_t bytes = numElems * sizeof(T);

        switch (type) {
            case AllocatorType::HipMalloc:
                util::hipCheck(hipMalloc(&ptr, bytes));
                break;
            case AllocatorType::HipMallocManaged:
                util::hipCheck(hipMallocManaged(&ptr, bytes));
                break;
            case AllocatorType::HipHostMalloc:
#ifdef __HIP_PLATFORM_AMD__
                util::hipCheck(hipHostMalloc(&ptr, bytes, hipHostMallocNonCoherent));
#else
                util::hipCheck(hipHostMalloc(&ptr, bytes, hipHostMallocDefault));
#endif
                break;
            case AllocatorType::Malloc:
                ptr = static_cast<T*>(::malloc(bytes));
                if (!ptr) throw std::bad_alloc();
                break;
        }

        return ptr;
    }

    /**
     * @brief Free memory using the matching allocator.
     */
    inline void freeMemory(void* ptr, AllocatorType type) {
        if (!ptr) return;

        switch (type) {
            case AllocatorType::HipMalloc:
            case AllocatorType::HipMallocManaged:
                util::hipCheck(hipFree(ptr));
                break;
            case AllocatorType::HipHostMalloc:
                util::hipCheck(hipHostFree(ptr));
                break;
            case AllocatorType::Malloc:
                ::free(ptr);
                break;
        }
    }

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

