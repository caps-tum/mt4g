#pragma once

#include <hip/hip_runtime.h>
#include <stdexcept>

#ifdef __HIP_PLATFORM_AMD__
#include <rocm_smi/rocm_smi.h>
#endif

namespace util {
    /**
     * @brief Abort on failing HIP API calls.
     *
     * Formats the HIP error message together with the location of the failing
     * call and throws a runtime_error.
     *
     * @param expr  Result returned by a HIP API call.
     * @param file  Source file of the call (use __FILE__).
     * @param line  Source line of the call (use __LINE__).
     * @param func  Function name of the call (use __func__).
     */
    inline void hipCheckImpl( hipError_t expr, const char* file, int line, const char* func ) {
        if (expr != hipSuccess) {
            throw std::runtime_error{
                std::string{"HIP error: "} + hipGetErrorString(expr)
            + " in " + func
            + " at " + file + ":" + std::to_string(line)
            };
        }
    }

    #define hipCheck(expr) hipCheckImpl((expr), __FILE__, __LINE__, __func__)

#ifdef __HIP_PLATFORM_AMD__
    /**
     * @brief Validate ROCm SMI API results.
     *
     * Throws a runtime_error when the provided status code signals failure.
     *
     * @param e Result code returned by the ROCm SMI API.
     */
    inline void rocmCheck(rsmi_status_t e) {
        if (e != RSMI_STATUS_SUCCESS) {
            throw std::runtime_error("RSMI-Fehler");
        }
    }
#endif
}