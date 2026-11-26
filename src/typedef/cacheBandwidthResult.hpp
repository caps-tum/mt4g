#pragma once

#include <cstddef>
#include <map>
#include <vector>
#include <cstdint>
#include <limits>
#include <nlohmann/json.hpp>

// Marks grid cells skipped by early termination, unlike a measured 0 GiB/s
inline constexpr double UNMEASURED_BANDWIDTH = std::numeric_limits<double>::quiet_NaN();

typedef struct CacheBandwidthResult {
    double measuredBandwidth = 0.0;
    size_t dataBytes = 0;
    uint64_t cycles = 0;
    double time = 0.0;
    uint32_t numThreads = 0;
    uint32_t numBlocks = 0;
    size_t numReps = 0;
    uint32_t loadsPerBatch = 0; // only set by benchmarks that sweep it

    // Raw grid data for visualization (NOT serialized into JSON)
    std::vector<uint32_t> threadsTested;
    std::vector<size_t> repsTested;
    std::vector<uint32_t> blocksTested;
    std::vector<std::vector<double>> bandwidthGridGiBs;
    std::vector<std::vector<std::vector<double>>> bandwidth3D;

    CacheBandwidthResult() = default;
    CacheBandwidthResult(size_t dataBytes, uint32_t numBlocks) : dataBytes(dataBytes), numBlocks(numBlocks) {}
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
