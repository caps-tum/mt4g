#include "utils/util.hpp"
#include <vector>
#include <unordered_set>
#include <cstdint>
#include <random>
#include <algorithm>
#include <cstdint>
#include <hip/hip_runtime.h>

// returns next highest-or-equal power of 2 for 32-bit v
static uint32_t nextPowerOfTwo(uint32_t v) {
    if (v == 0) return 1;
    --v;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return ++v;
}

// returns largest power of 2 ≤ v
static uint32_t prevPowerOfTwo(uint32_t v) {
    // if v is already power of two, keep it
    // else, nextP/2 is the previous
    uint32_t nxt = nextPowerOfTwo(v);
    return (nxt == v) ? v : (nxt >> 1);
}

namespace util {   
    std::vector<uint32_t> generateRandomizedPChaseArray(size_t arraySizeBytes, size_t strideBytes) {
        // number of 32-bit words total, and words per stride
        const size_t Nwords = arraySizeBytes / sizeof(uint32_t) - (arraySizeBytes % strideBytes);
        const size_t step   = strideBytes / sizeof(uint32_t);
        const size_t numJumps = Nwords / step;

        // allocate exactly Nwords entries, zero‐initialized
        std::vector<uint32_t> arr(Nwords, 0);

        // build index list [0,1,2,...,numJumps-1]
        std::vector<uint32_t> idx(numJumps);
        std::iota(idx.begin(), idx.end(), 0);

        // Fisher–Yates shuffle, keep idx[0]==0 if you want a fixed start:
        std::random_device rd;
        std::mt19937 gen(rd());
        for (size_t i = numJumps - 1; i > 0; --i) {
            std::uniform_int_distribution<size_t> dist(1, i);
            size_t j = dist(gen);
            std::swap(idx[i], idx[j]);
        }

        // link each shuffled block to the next (wrapping at end)
        for (size_t k = 0; k < numJumps; ++k) {
            uint32_t cur = idx[k]             * step;
            uint32_t nxt = idx[(k + 1) % numJumps] * step;
            arr[cur] = nxt;
        }

        return arr;
    }

    std::vector<uint32_t> generatePChaseArray(size_t arraySizeBytes, size_t strideBytes) {
        const size_t N    = arraySizeBytes / sizeof(uint32_t);
        const size_t step = strideBytes / sizeof(uint32_t);

        std::vector<uint32_t> result(N); 

        for (size_t i = 0; i < N; i += step) {
            result[i] = (i + step) % N;
        }

        return result;
    }

    uint32_t closestPowerOfTwo(uint32_t n) {
        if (n == 0) return 1;

        uint32_t up   = nextPowerOfTwo(n);
        uint32_t down = prevPowerOfTwo(n);

        // choose the nearest; tie → höherer Wert
        if (n - down < up - n) return down;
        return up;
    }

    size_t pickMostTrailingZeros(const std::vector<size_t>& v) {
        if (v.empty()) return 0;

        size_t best = v.front();
        int bestCtz = best == 0 ? sizeof(size_t) * 8 : std::countr_zero(best);

        for (size_t i = 1; i < v.size(); ++i) {
            size_t x = v[i];
            int ctz = x == 0 ? sizeof(size_t) * 8 : std::countr_zero(x);

            if (ctz > bestCtz) {
                best = x;
                bestCtz = ctz;
            }
        }
        return best;
    }

    double convertCyclesToNanoseconds(double cycles) {
        return (cycles / static_cast<double>(getClockRateKHz())) * 1e9;
    }

    std::map<size_t, std::vector<uint32_t>> flattenLineSizeMeasurementsToAverage(const std::map<size_t, std::map<size_t, std::vector<uint32_t>>>& nested_timings) {
        std::map<size_t, std::vector<uint32_t>> result;

        for (const auto& [outer_key, inner_map] : nested_timings) {
            std::vector<uint32_t>& dest_vec = result[outer_key];
            for (const auto& [inner_key, vec] : inner_map) {
                double avg = util::average(vec);
                dest_vec.push_back(static_cast<uint32_t>(avg));
            }
        }

        return result;
    }

    std::tuple<size_t, size_t> adjustKiBBoundaries(size_t begin, size_t end,
                                                   size_t minAllowed,
                                                   size_t maxAllowed,
                                                   bool strictUpper) {
        size_t adjustedBegin = begin;
        size_t adjustedEnd   = end;

        size_t beginMod = adjustedBegin % CACHE_SIZE_BENCH_RESOLUTION;
        if (adjustedBegin - beginMod > minAllowed) {
            adjustedBegin -= beginMod;
        }
        if (adjustedBegin >= minAllowed + CACHE_SIZE_BENCH_RESOLUTION) {
            adjustedBegin -= CACHE_SIZE_BENCH_RESOLUTION;
        }

        size_t endMod = adjustedEnd % CACHE_SIZE_BENCH_RESOLUTION;
        size_t alignOff = endMod ? CACHE_SIZE_BENCH_RESOLUTION - endMod : 0;
        if (strictUpper) {
            if (adjustedEnd + alignOff < maxAllowed) {
                adjustedEnd += alignOff;
            }
        } else {
            adjustedEnd += alignOff;
        }

        if (adjustedEnd + CACHE_SIZE_BENCH_RESOLUTION <= maxAllowed) {
            adjustedEnd += CACHE_SIZE_BENCH_RESOLUTION;
        }

        return { adjustedBegin, adjustedEnd };
    }
}