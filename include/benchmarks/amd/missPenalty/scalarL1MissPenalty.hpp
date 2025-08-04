#pragma once
#include <cstddef>
#include "benchmarks/benchmark.hpp"
namespace benchmark {
    namespace amd {
        double measureScalarL1MissPenalty(size_t scalarL1CacheSizeBytes, size_t scalarL1FetchGranularityBytes, double scalarL1Latency);
    }
}
