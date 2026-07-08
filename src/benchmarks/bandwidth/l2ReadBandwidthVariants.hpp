#pragma once

#include <cstddef>

namespace benchmark {
    void exploreL2ReadBandwidth(size_t l2SizeBytes);

    /**
     * Generic batched-load bandwidth stress: tries 1/2/4/8/16 dwordx4 loads
     * in-flight per wavefront step and reports the median of EXPLORE_ROUNDS runs.
     *
     * The caller decides the array size for the target memory level:
     *   L2  →  l2CacheSize * numXCDs * 0.9   (fits in L2, 100% hit rate)
     *   L3  →  l3CacheSize * 0.9              (fits in L3, 100% hit rate)
     *   HBM →  very large array               (misses all caches)
     *
     * arraySizeBytes is rounded DOWN to the nearest power of 2 so address
     * wrapping uses a fast bitwise AND.  The optimal batch size (and hence
     * register pressure / occupancy trade-off) varies by GPU; all five are
     * tried so the best can be read from the output.
     */
    void exploreBatchedBandwidth(const char* level, size_t arraySizeBytes);
}
