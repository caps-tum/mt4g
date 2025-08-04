#pragma once

#include <functional>
#include "typedef/launcherFn.hpp"

namespace util {
    /**
     * @brief Narrow down the region where the latency increases.
     *
     * This helper launches the supplied kernel on progressively smaller
     * intervals until the difference between the returned bounds falls below
     * @p lowerUpperDiff of the initial range.
     *
     * @param launch         Kernel launcher returning per-access timings.
     * @param lowerBytes     Lower bound of the search range in bytes.
     * @param upperBytes     Upper bound of the search range in bytes.
     * @param strideBytes    Stride passed to the kernel launcher.
     * @param lowerUpperDiff Fractional difference between the final bounds.
     * @return Tuple <lowerBytes, upperBytes> surrounding the change point.
     */
    std::tuple<size_t, size_t> findCacheMissRegion(const LauncherFn& launch, size_t lowerBytes, size_t upperBytes, size_t strideBytes, double lowerUpperDiff = 0.05);
}