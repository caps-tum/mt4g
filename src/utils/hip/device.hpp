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
    /**
     * @brief Check whether the build targets the AMD HIP backend.
     */
    inline bool isAMD() {
        #ifdef __HIP_PLATFORM_AMD__
        return true;
        #endif
        return false;
    }

    /**
     * @brief Check whether the build targets the NVIDIA HIP backend.
     */
    inline bool isNVIDIA() {
        #ifdef __HIP_PLATFORM_NVIDIA__
        return true;
        #endif
        return false;
    }

    /**
     * @brief Return a human readable GPU vendor string.
     */
    inline std::string getVendor() {
        #ifdef __HIP_PLATFORM_NVIDIA__
        return "NVIDIA";
        #endif
        #ifdef __HIP_PLATFORM_AMD__
        return "AMD";
        #endif
        return "Unknown";
    }

    /**
     * @brief Compute the theoretical peak global memory bandwidth in GiB/s.
     */
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

    /**
     * @brief Query the L1 cache size in bytes if available.
     */
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

    /**
     * @brief Query the L2 cache size in bytes if available.
     */
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

    /**
     * @brief Query the L3 cache size in bytes if available.
     */
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

    /**
     * @brief Query the L1 cache line size in bytes if available.
     */
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

    /**
     * @brief Query the L2 cache line size in bytes if available.
     */
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

    /**
     * @brief Query the L3 cache line size in bytes if available.
     */
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

    /**
     * @brief Number of L2 caches present on the device.
     */
    inline std::optional<size_t> getL2Amount() {
        static std::optional<size_t> v = [](){
    #ifdef __HIP_PLATFORM_NVIDIA__
            return std::nullopt;
    #endif
    #ifdef __HIP_PLATFORM_AMD__
            return getKfdCacheAmountForLevel(2);
    #endif
        }();
        return v;
    }

    /**
     * @brief Number of L3 caches present on the device.
     */
    inline std::optional<size_t> getL3Amount() {
        static std::optional<size_t> v = [](){
    #ifdef __HIP_PLATFORM_NVIDIA__
            return std::nullopt;
    #endif
    #ifdef __HIP_PLATFORM_AMD__
            return getKfdCacheAmountForLevel(3);
    #endif
        }();
        return v;
    }

    /**
     * @brief Return the number of SIMD units per compute unit.
     */
    inline uint32_t getSIMDsPerCU() {
        static uint32_t simdCount = [](){
            #ifdef __HIP_PLATFORM_NVIDIA__
            return 4; // usually 4 SMSPs
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
        return simdCount;
    }

    /**
     * @brief Query how many XCDs (physical dies) the GPU comprises.
     *
     * Uses RSMI, which reports the true physical die count on discrete GPUs
     * regardless of compute partition mode.  Returns nullopt when RSMI is
     * unavailable or returns unexpected data (e.g. MI300A APU nodes).
     */
    inline std::optional<uint32_t> getNumXCDs() {
        static std::optional<uint32_t> xcdCount = []() -> std::optional<uint32_t> {
            #ifdef __HIP_PLATFORM_NVIDIA__
            return std::nullopt;
            #endif
            #ifdef __HIP_PLATFORM_AMD__
            int device;
            util::hipCheck(hipGetDevice(&device));
            if (rsmi_init(0) == RSMI_STATUS_SUCCESS) {
                uint16_t xcdCounter = 0;
                if (rsmi_dev_metrics_xcd_counter_get(static_cast<uint32_t>(device), &xcdCounter) == RSMI_STATUS_SUCCESS
                    && xcdCounter > 0) {
                    return static_cast<uint32_t>(xcdCounter);
                }
            }
            return std::nullopt;
            #endif
        }();
        return xcdCount;
    }

    /**
     * @brief Query how many XCCs (logical compute clusters) the GPU exposes.
     *
     * Uses hipDeviceAttributeNumberOfXccs, available from ROCm 7.0+.
     * Returns std::nullopt when the HIP attribute is not present (older ROCm).
     * XCC count reflects software-visible partitions and may differ from the
     * physical XCD (die) count depending on the compute partition mode.
     */
    inline std::optional<uint32_t> getNumXCCs() {
        static std::optional<uint32_t> xccCount = []() -> std::optional<uint32_t> {
            #ifdef __HIP_PLATFORM_NVIDIA__
            return std::nullopt;
            #endif
            #if defined(__HIP_PLATFORM_AMD__) && defined(hipDeviceAttributeNumberOfXccs)
            int device;
            util::hipCheck(hipGetDevice(&device));
            int32_t xccAttr = 0;
            hipError_t e = hipDeviceGetAttribute(&xccAttr, hipDeviceAttributeNumberOfXccs, device);
            if (e != hipSuccess) return std::nullopt;
            return static_cast<uint32_t>(xccAttr);
            #else
            return std::nullopt;
            #endif
        }();
        return xccCount;
    }

    /**
     * @brief Compute the number of compute units per die.
     *
     * Divides total CU count by XCD count (physical dies via RSMI) when
     * available, then by XCC count (logical clusters via HIP, equals XCD
     * count in SPX partition mode) as fallback.  Returns nullopt when
     * neither die count is known.
     */
    inline std::optional<uint32_t> getComputeUnitsPerDie() {
        static std::optional<uint32_t> cusPerDie = []() -> std::optional<uint32_t> {
            int device;
            util::hipCheck(hipGetDevice(&device));
            int32_t cuCount;
            util::hipCheck(hipDeviceGetAttribute(
                &cuCount,
                hipDeviceAttributeMultiprocessorCount,
                device));
            if (auto xcd = getNumXCDs()) return static_cast<uint32_t>(cuCount) / *xcd;
            if (auto xcc = getNumXCCs()) return static_cast<uint32_t>(cuCount) / *xcc;
            return std::nullopt;
        }();
        return cusPerDie;
    }

    /**
     * @brief Retrieve the GPU core clock rate in kHz.
     */
    inline uint32_t getClockRateKHz() {
        int32_t device;
        util::hipCheck(hipGetDevice(&device));
        int32_t rateKHz;
        util::hipCheck(hipDeviceGetAttribute(&rateKHz, hipDeviceAttributeClockRate, device));
        return rateKHz;
    }

    /**
     * @brief Return the native warp/wavefront size of the device.
     */
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

    /**
     * @brief Query total global memory size in bytes.
     */
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

    /**
     * @brief Estimate the number of cores per multiprocessor.
     */
    inline uint32_t getNumberOfCoresPerSM() {
        int device = 0;
        (void)hipGetDevice(&device);
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, device) != hipSuccess) {
            #ifdef __HIP_PLATFORM_NVIDIA__
            return 128u;          // safe default for modern NVIDIA (Turing/Ampere/Ada)
            #else
            return 64u;           // AMD CUs have 64 FP32 ALUs per CU
            #endif
        }

        #ifdef __HIP_PLATFORM_NVIDIA__
        const int maj = prop.major;
        const int min = prop.minor;

        // Returns FP32 "CUDA cores" per SM, by compute capability.
        switch (maj) {
            case 1:  return 8u;                                 // Tesla
            case 2:  return (min == 1 ? 48u : 32u);             // Fermi 2.1 vs 2.0
            case 3:  return 192u;                               // Kepler (SMX)
            case 5:  return 128u;                               // Maxwell (SMM) 5.0/5.2/5.3
            case 6:  return (min == 0 ? 64u : 128u);            // Pascal: GP100(6.0)=64, GP10x(6.1/6.2)=128
            case 7:  return 64u;                                // Volta(7.0/7.2)=64, Turing(7.5)=64
            case 8:  return (min == 0 ? 64u : 128u);            // Ampere: GA100(8.0)=64, GA10x/Orin/Ada(8.6/8.7/8.9)=128
            case 9:  return 128u;                               // Hopper (GH100)
            default: return 128u;                               // reasonable default for unknown future parts
        }
        #endif
        
        #ifdef __HIP_PLATFORM_AMD__
        // AMD (GCN/RDNA/CDNA): 64 FP32 ALUs ("stream processors") per CU
        (void)prop;
        return 64u;
        #endif

        return 0u;
    }

    /**
     * @brief Return the total number of compute units on the device.
     */
    inline uint32_t getNumberOfComputeUnits() {
        static uint32_t cus = []() {
            int device;
            util::hipCheck(hipGetDevice(&device));
            int32_t cuCount;
            util::hipCheck(hipDeviceGetAttribute(
                &cuCount,
                hipDeviceAttributeMultiprocessorCount,
                device));
            return static_cast<uint32_t>(cuCount);
        }();
        return cus;
    }

} // namespace util

