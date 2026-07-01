#pragma once

#include <cstddef>
#include <map>
#include <vector>
#include <cstdint>
#include <nlohmann/json.hpp>

typedef struct CacheBandwidthResult {
    double measuredBandwidth;
    size_t dataBytes;
    uint64_t cycles;
    double time;
    uint32_t numThreads;
    uint32_t numBlocks;
    size_t numReps;
    
    // Raw grid data for visualization (NOT serialized into JSON)
    std::vector<uint32_t> threadsTested;
    std::vector<size_t> repsTested;
    std::vector<uint32_t> blocksTested;
    std::vector<std::vector<double>> bandwidthGridGiBs;
    std::vector<std::vector<std::vector<double>>> bandwidth3D;
    
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(
        CacheBandwidthResult,
        measuredBandwidth, dataBytes, cycles, time, numReps, numThreads, numBlocks
    )
} CacheBandwidthResult;