#pragma once

#include <optional>
#include <string>
#include <hip/hip_runtime.h>
#include <vector>
#ifdef __HIP_PLATFORM_AMD__
#include <rocm_smi/rocm_smi.h>
#endif

#include "utils/errorHandling.hpp"
#include "utils/hip/hsa.hpp"

namespace util {

inline bool isAMD() {
#ifdef __HIP_PLATFORM_AMD__
    return true;
#endif
    return false;
}

inline bool isNVIDIA() {
#ifdef __HIP_PLATFORM_NVIDIA__
    return true;
#endif
    return false;
}

inline std::string getVendor() {
#ifdef __HIP_PLATFORM_NVIDIA__
    return "NVIDIA";
#endif
#ifdef __HIP_PLATFORM_AMD__
    return "AMD";
#endif
    return "Unknown";
}

inline double getTheoreticalMaxGlobalMemoryBandwidthGiBs() {
    static double bwGiBs = []() -> double {
        hipDeviceProp_t prop;
        hipCheck(hipGetDeviceProperties(&prop, 0));
        double clkMHz = static_cast<double>(prop.memoryClockRate) / 1000.0;
        double busBytes = static_cast<double>(prop.memoryBusWidth) / 8.0;
        double bytesPerSec = clkMHz * 1e6 * busBytes * 2.0;
        return bytesPerSec / (1024.0 * 1024.0 * 1024.0);
    }();
    return bwGiBs;
}

inline std::optional<size_t> getL1SizeBytes() {
    static std::optional<size_t> v = [](){
#ifdef __HIP_PLATFORM_NVIDIA__
        return std::nullopt;
#endif
#ifdef __HIP_PLATFORM_AMD__
        auto queriedSize = queryCacheLevelBytes(getCurrentHsaAgent(), 1) * 1024;
        return queriedSize > 0 ? std::optional<size_t>(queriedSize) : std::nullopt;
#endif
    }();
    return v;
}

inline std::optional<size_t> getL2SizeBytes() {
    static std::optional<size_t> v = [](){
#ifdef __HIP_PLATFORM_NVIDIA__
        return std::nullopt;
#endif
#ifdef __HIP_PLATFORM_AMD__
        auto queriedSize = queryCacheLevelBytes(getCurrentHsaAgent(), 2) * 1024;
        return queriedSize > 0 ? std::optional<size_t>(queriedSize) : std::nullopt;
#endif
    }();
    return v;
}

inline std::optional<size_t> getL3SizeBytes() {
    static std::optional<size_t> v = [](){
        #ifdef __HIP_PLATFORM_NVIDIA__
        return std::nullopt;
        #endif
        #ifdef __HIP_PLATFORM_AMD__
        auto queriedSize = queryCacheLevelBytes(getCurrentHsaAgent(), 3) * 1024;
        return queriedSize > 0 ? std::optional<size_t>(queriedSize) : std::nullopt;
        #endif
    }();
    return v;
}

inline std::optional<size_t> getL1LineSizeBytes() {
    static std::optional<size_t> v = [](){
#ifdef __HIP_PLATFORM_NVIDIA__
        return std::nullopt;
#endif
#ifdef __HIP_PLATFORM_AMD__
        return getKfdCachelineBytesForLevel(1);
#endif
    }();
    return v;
}

inline std::optional<size_t> getL2LineSizeBytes() {
    static std::optional<size_t> v = [](){
#ifdef __HIP_PLATFORM_NVIDIA__
        return std::nullopt;
#endif
#ifdef __HIP_PLATFORM_AMD__
        return getKfdCachelineBytesForLevel(2);
#endif
    }();
    return v;
}

inline std::optional<size_t> getL3LineSizeBytes() {
    static std::optional<size_t> v = [](){
#ifdef __HIP_PLATFORM_NVIDIA__
        return std::nullopt;
#endif
#ifdef __HIP_PLATFORM_AMD__
        return getKfdCachelineBytesForLevel(3);
#endif
    }();
    return v;
}

inline uint32_t getSIMDsPerCU() {
    static uint32_t xcdCount = [](){
#ifdef __HIP_PLATFORM_NVIDIA__
        return 1;
#endif
#ifdef __HIP_PLATFORM_AMD__
        hsa_init();
        hsa_agent_t agent = getCurrentHsaAgent();
        uint32_t simd = 0;
        hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU), &simd);
        hsa_shut_down();
        return simd;
#endif
    }();
    return xcdCount;
}

inline uint32_t getNumXCDs() {
    static uint32_t xcdCount = [](){
#ifdef __HIP_PLATFORM_NVIDIA__
        return 1;
#endif
#ifdef __HIP_PLATFORM_AMD__
        util::rocmCheck(rsmi_init(0));
        int device;
        util::hipCheck(hipGetDevice(&device));
        uint16_t xcdCounter;
        util::rocmCheck(rsmi_dev_metrics_xcd_counter_get(device, &xcdCounter));
        return static_cast<uint32_t>(xcdCounter);
#endif
    }();
    return xcdCount;
}

inline uint32_t getComputeUnitsPerDie() {
    static uint32_t cusPerDie = []() {
        int device;
        util::hipCheck(hipGetDevice(&device));
        int32_t cuCount;
        util::hipCheck(hipDeviceGetAttribute(
            &cuCount,
            hipDeviceAttributeMultiprocessorCount,
            device));
        return static_cast<uint32_t>(cuCount) / getNumXCDs();
    }();
    return cusPerDie;
}

inline uint32_t getClockRateKHz() {
    int32_t device;
    util::hipCheck(hipGetDevice(&device));
    int32_t rateKHz;
    util::hipCheck(hipDeviceGetAttribute(&rateKHz, hipDeviceAttributeClockRate, device));
    return rateKHz;
}

inline uint32_t getWarpSize() {
    static int32_t warpSize = [](){
        int32_t device;
        util::hipCheck(hipGetDevice(&device));
        int32_t v;
        util::hipCheck(hipDeviceGetAttribute(&v, hipDeviceAttributeWarpSize, device));
        return v;
    }();
    return warpSize;
}

inline uint64_t getGlobalMemorySizeBytes() {
    static uint64_t totalMem = [](){
        int device;
        util::hipCheck(hipGetDevice(&device));
        hipDeviceProp_t props;
        util::hipCheck(hipGetDeviceProperties(&props, device));
        return static_cast<uint64_t>(props.totalGlobalMem);
    }();
    return totalMem;
}

inline uint32_t getNumberOfCoresPerSM() {
#if defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIP_PLATFORM_AMD__)
    int device = 0;
    (void)hipGetDevice(&device);
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, device) == hipSuccess) {
#ifdef __HIP_PLATFORM_NVIDIA__
        const int maj = prop.major;
        const int min = prop.minor;
        if (maj == 2) return 32;
        if (maj == 3) return 192;
        if (maj == 5) return 128;
        if (maj == 6) return (min == 0) ? 64 : 128;
        if (maj == 7) return 64;
        if (maj == 8) {
            if (min == 0) return 64;
            if (min == 6 || min == 9) return 128;
        }
        if (maj == 9) return 128;
        return 0;
#elif defined(__HIP_PLATFORM_AMD__)
        (void)prop;
        return 64;
#endif
    } else {
#ifdef __HIP_PLATFORM_NVIDIA__
        return 128;
#else
        return 64;
#endif
    }
#else
    return 0;
#endif
}

inline uint32_t getNumberOfComputeUnits() {
    static uint32_t cusPerDie = []() {
        int device;
        util::hipCheck(hipGetDevice(&device));
        int32_t cuCount;
        util::hipCheck(hipDeviceGetAttribute(
            &cuCount,
            hipDeviceAttributeMultiprocessorCount,
            device));
        return static_cast<uint32_t>(cuCount);
    }();
    return cusPerDie;
}

} // namespace util

