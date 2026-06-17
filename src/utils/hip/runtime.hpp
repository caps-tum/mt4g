#pragma once

#include <vector>
#include <optional>
#include <hip/hip_runtime.h>
#include "utils/errorHandling.hpp"
#include "utils/hip/device.hpp"

namespace util {

    inline hipFuncCache_t cachePreference = hipFuncCachePreferL1;

    /**
     * @brief Set the device cache preference for subsequent kernels.
     */
    inline void setCachePreference(hipFuncCache_t pref) {
        cachePreference = pref;
        util::hipCheck(hipDeviceSetCacheConfig(cachePreference));
    }

    /**
     * @brief Get the current device cache preference.
     */
    inline hipFuncCache_t getCachePreference() {
        return cachePreference;
    }

    /**
     * @brief Reset the current device and restore cache configuration.
     */
    inline void hipDeviceReset() {
        util::hipCheck(::hipDeviceReset());
        util::hipCheck(hipDeviceSetCacheConfig(cachePreference));
    }

    /**
     * @brief Measure elapsed time between two HIP events in milliseconds.
     */
    inline float getElapsedTimeMs(hipEvent_t start, hipEvent_t stop) {
        util::hipCheck(hipEventSynchronize(start));
        util::hipCheck(hipEventSynchronize(stop));
        float ms = 0.0f;
        util::hipCheck(hipEventElapsedTime(&ms, start, stop));
        return ms;
    }

    /**
     * @brief Create a HIP event object.
     */
    inline hipEvent_t createHipEvent() {
        hipEvent_t event;
        util::hipCheck(hipEventCreate(&event));
        return event;
    }

    /**
     * @brief Determine if the device supports cooperative launches.
     */
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

    /**
     * @brief Retrieve and cache the device properties struct.
     */
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

    /**
     * @brief Query the installed HIP driver version.
     */
    inline std::optional<int> getDriverVersion() {
        static std::optional<int> driver = []() -> std::optional<int> {
            int version = 0;
            if (hipDriverGetVersion(&version) == hipSuccess) {
                return version;
            }
            return std::nullopt;
        }();
        return driver;
    }

    /**
     * @brief Query the HIP runtime version.
     */
    inline std::optional<int> getRuntimeVersion() {
        static std::optional<int> runtime = []() -> std::optional<int> {
            int version = 0;
            if (hipRuntimeGetVersion(&version) == hipSuccess) {
                return version;
            }
            return std::nullopt;
        }();
        return runtime;
    }

    /**
     * @brief Maximum number of threads supported per block.
     */
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

    /**
     * @brief Determine the maximum number of active blocks for a kernel.
     */
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

    /**
     * @brief Create a stream bound to a specific compute unit (AMD) or default stream (NVIDIA).
     */
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

    /**
     * @brief Create a stream bound to a set of compute units.
     */
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

