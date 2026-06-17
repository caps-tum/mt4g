#pragma once

#include <cstddef>
#include "utils/util.hpp"

inline constexpr size_t DEFAULT_SAMPLE_SIZE = 256; // Loads
inline constexpr double DEFAULT_GRACE_FACTOR = 2.0; // Factor
inline constexpr size_t CACHE_SIZE_BENCH_RESOLUTION = 256; // Bytes
inline constexpr double CACHE_MISS_REGION_RELATIVE_DIFFERENCE = 0.02; // Relative difference
inline constexpr size_t CACHE_LINE_SIZE_RESOLUTION_DIVISOR = 2; // Divisor DO NOT CHANGE, Line Size Detection assumes it to be 2. TODO
inline constexpr size_t DEFAULT_ROUNDS = 10; // Bandwidth rounds
inline constexpr size_t DEFAULT_SIZE_DOWN_FACTOR = 4; // Bandwidth benchmark test size reduction factor
inline constexpr double SHARED_THRESHOLD = 2.5; // Threshold divisor to determine wether a measure cache latency counts as evicted or not