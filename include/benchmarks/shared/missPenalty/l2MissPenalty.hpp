#pragma once
#include <cstddef>
namespace benchmark {
    double measureL2MissPenalty(size_t l2CacheSizeBytes, size_t l2CacheLineSizeBytes, double l2Latency);
}
