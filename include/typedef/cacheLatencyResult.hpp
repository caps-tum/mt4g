#pragma once

#include <cstddef>
#include <map>
#include <vector>
#include <cstdint>
#include <nlohmann/json.hpp>

#include "typedef/enums.hpp"

typedef struct CacheLatencyResult {
    std::vector<uint32_t> timings;
    double mean;
    double p50;
    double p95;
    double stdev;
    size_t measurements;
    size_t sampleSize;
    enum MeasureUnit unit;
    enum BenchmarkMethod method;
    
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(
        CacheLatencyResult,
        mean, p50, p95, stdev,
        measurements, sampleSize,
        unit, method
    )
} CacheLatencyResult;