#pragma once

#include "typedef/cacheBandwidthResult.hpp"
#include "utils/util.hpp"

#include <cstdint>
#include <cstddef>
#include <vector>

// Fills the block/thread/rep axes of a multi-block sweep and marks all grid cells as unmeasured.
// The sweep runs descending; the axes stay ascending for the CSV layout.
inline void initMultiBlockGrid(CacheBandwidthResult &result, size_t minReps, size_t maxReps)
{
    uint32_t minThreads = util::getWarpSize();
    uint32_t maxThreads = util::getMaxThreadsPerBlock();

    uint32_t minBlocks = util::getNumberOfComputeUnits();
    uint32_t maxBlocks = util::getNumberOfComputeUnits() * util::getMaxBlocksPerMultiProcessor();

    for (uint32_t numBlocks = minBlocks; numBlocks <= maxBlocks; numBlocks *= 2)
    {
        result.blocksTested.push_back(numBlocks);
    }
    for (uint32_t numThreads = minThreads; numThreads <= maxThreads; numThreads *= 2)
    {
        result.threadsTested.push_back(numThreads);
    }
    for (size_t reps = minReps; reps <= maxReps; reps *= 2)
    {
        result.repsTested.push_back(reps);
    }

    result.bandwidth3D.assign(result.blocksTested.size(), std::vector<std::vector<double>>(
        result.threadsTested.size(), std::vector<double>(result.repsTested.size(), UNMEASURED_BANDWIDTH)));
}

// Time in seconds that a kernel measured with __globaltimer() and wrote to d_ticks
inline double globalTimerSeconds(const uint64_t *d_ticks)
{
    const uint64_t ticks = util::copyFromDevice(d_ticks, 1)[0];
    return (double) ticks / (util::getWallClockRateKHz() * 1000.0);
}
