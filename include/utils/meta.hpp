#pragma once

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <sys/utsname.h>

#ifdef __HIPCC__
#include <hip/hip_version.h>
#endif

namespace util {

    /**
     * @brief Retrieve the CPU model name of the host system.
     *
     * @return CPU model string or std::nullopt if unavailable.
     */
    inline std::optional<std::string> getHostCpuModel() {
        static std::optional<std::string> cpuModel = []() -> std::optional<std::string> {
            std::ifstream cpuInfo("/proc/cpuinfo");
            if (!cpuInfo.is_open()) return std::nullopt;
            std::string line;
            while (std::getline(cpuInfo, line)) {
                auto pos = line.find("model name");
                if (pos != std::string::npos) {
                    auto colon = line.find(':');
                    if (colon != std::string::npos) {
                        std::string value = line.substr(colon + 1);
                        if (!value.empty() && value[0] == ' ') value.erase(0, 1);
                        return value;
                    }
                }
            }
            return std::nullopt;
        }();
    return cpuModel;
    }

    /**
    * @brief Generate an ISO 8601 UTC timestamp.
    */
    inline std::string getCurrentTimestamp() {
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm{};
        gmtime_r(&t, &tm);
        std::ostringstream oss;
        oss << std::put_time(&tm, "%FT%TZ");
        return oss.str();
    }

    /**
    * @brief Return the version string of the host C/C++ compiler.
    */
    inline std::string getHostCompilerVersion() {
        #ifdef __clang__
        return std::string{"clang "} + std::to_string(__clang_major__) + '.' + std::to_string(__clang_minor__) + '.' + std::to_string(__clang_patchlevel__);
        #endif 

        #ifdef __GNUC__
        return std::string{"gcc "} + std::to_string(__GNUC__) + '.' + std::to_string(__GNUC_MINOR__) + '.' + std::to_string(__GNUC_PATCHLEVEL__);
        #endif
        
        return "unknown";
    }

    /**
    * @brief Determine the version of the GPU compiler used for the build.
    */
    inline std::optional<std::string> getGpuCompilerVersion() {
        #ifdef __CUDACC__
        return std::string{"nvcc "} + std::to_string(__CUDACC_VER_MAJOR__) + '.' + std::to_string(__CUDACC_VER_MINOR__) + '.' + std::to_string(__CUDACC_VER_BUILD__);
        #endif

        #ifdef __HIPCC__
        return std::string{"hipcc "} + std::to_string(HIP_VERSION_MAJOR) + '.' + std::to_string(HIP_VERSION_MINOR) + '.' + std::to_string(HIP_VERSION_PATCH);
        #endif

        return std::nullopt;
    }

    /**
    * @brief Obtain a short description of the operating system.
    */
    inline std::optional<std::string> getOsDescription() {
        static std::optional<std::string> osInfo =
        []() -> std::optional<std::string> {
            struct utsname buf{};
            if (uname(&buf) != 0) return std::nullopt;
            return std::string(buf.sysname) + " " + buf.release;
        }();
        return osInfo;
    }
} // namespace util
