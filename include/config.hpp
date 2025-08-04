#pragma once

#include <cstddef>
#include "utils/util.hpp"

inline constexpr size_t DEFAULT_SAMPLE_SIZE = 256; // Loads
inline constexpr double DEFAULT_GRACE_FACTOR = 2.0; // Factor
inline constexpr size_t CACHE_SIZE_BENCH_RESOLUTION = 256; // Bytes
inline constexpr double CACHE_MISS_REGION_RELATIVE_DIFFERENCE = 0.05; // Relative difference
inline constexpr size_t CACHE_LINE_SIZE_RESOLUTION_DIVISOR = 2; // Divisor
inline constexpr size_t DEFAULT_ROUNDS = 10; // Bandwidth rounds
inline constexpr size_t DEFAULT_SIZE_DOWN_FACTOR = 4; // Bandwidth benchmark test size reduction factor