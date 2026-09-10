#pragma once

#include <vector>
#include <map>
#include "typedef/launcherFn.hpp"

namespace util {
    /**
     * @brief Execute a benchmark for a contiguous range of array sizes.
     *
     * @param launcher        Function launching the kernel for a given size.
     * @param begin           Starting array size in bytes.
     * @param end             Last array size in bytes.
     * @param stride          Stride between accesses in bytes.
     * @param arrayIncrease   Increase in array size per step in bytes.
     * @param tag             Label printed during execution.
     * @return Map from array size to measured timing vectors.
     */
    std::map<size_t, std::vector<uint32_t>> runBenchmarkRange(LauncherFn launcher, size_t begin, size_t end, size_t stride, size_t arrayIncrease, const std::string& tag);
}
