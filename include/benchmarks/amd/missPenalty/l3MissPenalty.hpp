#pragma once
#include <cstddef>
#include "benchmarks/benchmark.hpp"
namespace benchmark {
    namespace amd {
        double measureL3MissPenalty(size_t l3CacheSizeBytes, size_t l3FetchGranularityBytes, double l3Latency);
    }
}
