#pragma once

#include <cstddef>
#include <map>
#include <vector>
#include <cstdint>

#include "typedef/enums.hpp"

typedef struct CacheLineSizeResult {
    std::map<size_t, std::map<size_t, std::vector<uint32_t>>> timings;
    size_t size;
    double confidence;
    enum BenchmarkMethod method;
    enum MeasureUnit unit;
    bool randomized;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(
        CacheLineSizeResult,
        size, confidence, method, unit, randomized
    )
} CacheLineSizeResult;
