#pragma once

#include <cstddef>
#include <optional>

#include "benchmarks/benchmark.hpp"

namespace benchmark {
    /**
     * @brief Determine how many cores share an L1 cache.
     *
     * @param l1SizeBytes          Total L1 cache size.
     * @param l1FetchGranularityBytes Cache line or sector size.
     * @param l1MissPenalty        Penalty in cycles for an L1 miss.
     * @return Number of cores per cache.
     */
    std::optional<uint32_t> measureL1Amount(size_t l1SizeBytes, size_t l1FetchGranularityBytes, double l1MissPenalty);
}