#pragma once

#include <hip/hip_runtime.h>

struct CLIOptions {
    int  deviceId;          // GPU ID via -d / --device-id
    //std::string outputBase  // Output base directory for graphs / measurements / final json
    bool graphs;            // Generate graphs if true
    bool rawData;           // Output raw measurement data
    bool fullReport;        // Write README with summary and graphs
    bool writeJson;         // Write final JSON result to a file
    bool randomize;         // Randomize P-Chase arrays if true
    bool runSilently;       // Do not print progress information if true

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
