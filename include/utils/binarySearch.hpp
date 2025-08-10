#pragma once

#include <functional>
#include "typedef/launcherFn.hpp"

namespace util {
    /**
     * @brief Locate the cache miss region using exponential and binary search.
     *
     * The kernel is first launched over exponentially growing sizes to bracket
     * the latency jump. The resulting window is then refined with a binary
     * search until the bounds differ by @p lowerUpperDiff of the original
     * interval.
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