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
     * @brief Estimate a reasonable block count for bandwidth kernels.
     *
     * @param data            Timing measurements used to gauge latency.
     * @param threadBlockSize Block size used by the kernel.
     * @return Suggested thread block count.
     */
    uint32_t tryComputeOptimalBandwidthBlockCount(const std::vector<uint32_t>& data, size_t threadBlockSize);

    /**
     * @brief Choose the most plausible cache line size candidate.
     *
     * @param changePoints Detected change points in bytes.
     * @return Selected candidate size in bytes.
     */
    size_t tryGetMostLikelyLineSizeCandidate(const std::vector<size_t>& changePoints);

    /**
     * @brief Choose the most plausible cache size candidate.
     *
     * @param changePoints Detected cache size candidates.
     * @return Selected candidate size in bytes.
     */
    size_t tryGetMostLikelyCacheSizeCandidate(const std::vector<size_t>& changePoints);

    /**
     * @brief Compute the segment size closest to @p target that divides @p base.
     *
     * @param base    Base size that will be divided.
     * @param target  Desired segment size.
     * @return Tuple of <divided size, similarity> where similarity is in [0,1].
     */
    std::tuple<size_t, double> tryComputeNearestSegmentSize(size_t base, size_t target);
}