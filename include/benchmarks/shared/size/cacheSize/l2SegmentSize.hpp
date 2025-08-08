#pragma once

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    /**
     * @brief Measure the size of a single L2 segment.
     *
     * @param l2FetchGranularityBytes Line or sector size used during the test.
     * @return Detected segment size in bytes.
     */
    CacheSizeResult measureL2SegmentSize(size_t l2FullSize, size_t l2FetchGranularityBytes);
}