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
    uint32_t loadsPerBatch = 0; // only set by benchmarks that sweep it

    // Raw grid data for visualization (NOT serialized into JSON)
    std::vector<uint32_t> threadsTested;
    std::vector<size_t> repsTested;
    std::vector<uint32_t> blocksTested;
    std::vector<std::vector<double>> bandwidthGridGiBs;
    std::vector<std::vector<std::vector<double>>> bandwidth3D;
} CacheBandwidthResult;

inline void to_json(nlohmann::json& j, const CacheBandwidthResult& result) {
    j = {
        {"measuredBandwidth", result.measuredBandwidth},
        {"dataBytes", result.dataBytes},
        {"cycles", result.cycles},
        {"time", result.time},
        {"numReps", result.numReps},
        {"numThreads", result.numThreads},
        {"numBlocks", result.numBlocks}
    };
    if (result.loadsPerBatch > 0) {
        j["loadsPerBatch"] = result.loadsPerBatch;
    }
}
