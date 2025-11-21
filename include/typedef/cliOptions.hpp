#pragma once

#include <hip/hip_runtime.h>
#include <filesystem>
#include <string>

struct CLIOptions {
    std::string fileName;             // Name of output files
    std::filesystem::path location;   // Location of output files
    int  deviceId;                    // GPU ID via -d / --device-id
    bool graphs;                      // Generate graphs if true
    bool rawData;                     // Output raw measurement data
    bool fullReport;                  // Write README with summary and graphs
    bool useStdout;                   // Write final JSON result to stdout
    bool randomize;                   // Randomize P-Chase arrays if true
    bool runSilently;                 // Do not print progress information if true

    // Benchmark groups
    bool runL3;
    bool runL2;
    bool runL1;
    bool runScalar;
    bool runConstant;
    bool runReadOnly;
    bool runTexture;
    bool runSharedMemory;
    bool runMainMemory;
    bool runDepartureDelay;
    bool runResourceSharing;

    hipFuncCache_t cachePreference; // Cache config preference
};
