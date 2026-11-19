#pragma once

#include <vector>
#include <algorithm>
#include <cstdlib>
#include <new>
#include <hip/hip_runtime.h>

#include "utils/errorHandling.hpp"

namespace util {
    /**
     * @brief Allocator selector for L3/main-memory latency and bandwidth sweeps.
     */
    enum class AllocatorType {
        HipMalloc,         // device memory
        HipMallocManaged,  // managed (unified) memory
        HipHostMalloc,     // pinned host memory
        Malloc             // pageable host memory, needs XNACK (e.g. MI300A)
    };

    /**
     * @brief Allocate uninitialised memory using the selected allocator.
     */
    // TODO: check that the device can access the memory (managed memory, pageable memory with XNACK)
    // and record the XNACK mode in the results
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

    template <typename T> __global__ void copyMemoryKernel(T* destination, const T* source, size_t count) {
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
            destination[i] = source[i];
        }
    }

    /**
     * @brief Allocate memory with the selected allocator and copy @p data into it from the GPU.
     *
     * The data goes through a temporary device buffer and a copy kernel, so that for every
     * allocator the GPU is the first to touch the memory.
     */
    template <typename T> T* allocateMemory(const std::vector<T>& data, AllocatorType type) {
        T* ptr = allocateMemory<T>(data.size(), type);
        T* staging = allocateGPUMemory(data);

        constexpr uint32_t threads = 256;
        const uint32_t blocks = static_cast<uint32_t>(std::min<size_t>((data.size() + threads - 1) / threads, 65535));
        copyMemoryKernel<<<blocks, threads>>>(ptr, staging, data.size());
        util::hipCheck(hipGetLastError());
        util::hipCheck(hipDeviceSynchronize());

        util::hipCheck(hipFree(staging));
        return ptr;
    }

} // namespace util

