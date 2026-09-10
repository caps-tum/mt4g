#pragma once

#include <vector>

namespace util {
    /**
     * @brief Test a sequence for near-monotonic behaviour.
     *
     * Small downward noise and occasional spikes are tolerated.
     *
     * @param data      Sequence of samples.
     * @param noiseTol  Allowed downward deviation.
     * @param spikeTol  Threshold for spike detection.
     * @return true if the sequence is considered monotonic.
     */
    bool isAlmostMonotonic(const std::vector<uint32_t>& data, double noiseTol = 30, double spikeTol = 100);

    /**
     * @brief Heuristically check for irregular measurement spikes.
     *
     * @param data Map of sample vectors keyed by size.
     * @return true if a fluke was detected.
     */
    bool hasFlukeOccured(const std::map<size_t, std::vector<uint32_t>>& data);

    /**
     * @brief Compute the segment size closest to @p target that divides @p base.
     *
     * @param base    Base size that will be divided.
     * @param target  Desired segment size.
     * @return Tuple of <divided size, similarity> where similarity is in [0,1].
     */
    std::tuple<size_t, double> tryComputeNearestSegmentSize(size_t base, size_t target);
}