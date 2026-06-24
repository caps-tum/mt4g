#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <deque>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <tuple>
#include <vector>
#include <map>
#include <vector>
#include <cstddef>
#include <cmath>
#include <cassert>

#include "utils/util.hpp"
#include "typedef/threadPool.hpp"


// Compute D given *sorted* inputs
static inline double statisticSorted(const std::vector<double>& a, const std::vector<double>& b) {
    size_t i = 0, j = 0, na = a.size(), nb = b.size();
    double d = 0.0;
    while (i < na && j < nb) {
        const double x = (a[i] <= b[j]) ? a[i] : b[j];
        while (i < na && a[i] == x) ++i;
        while (j < nb && b[j] == x) ++j;
        d = std::max(d, std::fabs(double(i) / na - double(j) / nb));
    }
    // trailing ends
    return std::max({d, std::fabs(1.0 - double(i) / na), std::fabs(1.0 - double(j) / nb)});
}

// Compute max KS-D over all possible splits for a given sequence of scalars.
static inline double maxDOverAllSplits(const std::vector<double>& seq) {
    const size_t N = seq.size();
    if (N < 2) return 0.0;
    std::vector<double> left;
    std::vector<double> right(seq.begin(), seq.end());
    std::sort(right.begin(), right.end());
    double maxD = 0.0;
    for (size_t split = 0; split + 1 < N; ++split) {
        const double v = seq[split];
        // move v from right -> left while keeping both sorted
        auto it_right = std::lower_bound(right.begin(), right.end(), v);
        if (it_right != right.end()) right.erase(it_right);
        left.insert(std::lower_bound(left.begin(), left.end(), v), v);
        const double D = statisticSorted(left, right);
        if (D > maxD) maxD = D;
    }
    return maxD;
}

// Global permutation p-value for the *maximum* KS-D across all splits (Westfall–Young style).
// No tunable parameters exposed. Internal caps keep runtime controlled.
static inline double globalMaxPvaluePermutation(const std::vector<double>& vals,
                                         std::mt19937 rng = std::mt19937{std::random_device{}()}) {
    const size_t N = vals.size();
    if (N < 2) return 1.0;

    // Observed maximal D
    const double observed_maxD = maxDOverAllSplits(vals);

    // Internal defaults (not exposed): total permutations
    constexpr size_t MIN_REPS = 10'000;
    constexpr size_t MAX_REPS = 100'000;
    size_t reps = (N <= 64 ? 50'000 : 20'000); // automatic heuristic
    reps = std::clamp(reps, MIN_REPS, MAX_REPS);

    auto worker = [&](size_t reps_chunk, uint32_t seed) -> size_t {
        std::mt19937 local_rng(seed);
        std::vector<double> perm = vals;
        size_t exceed = 0;
        for (size_t r = 0; r < reps_chunk; ++r) {
            std::shuffle(perm.begin(), perm.end(), local_rng);
            const double maxD_perm = maxDOverAllSplits(perm);
            if (maxD_perm >= observed_maxD) ++exceed;
        }
        return exceed;
    };

    if (g_in_thread_pool_worker) {
        size_t exceed = worker(reps, rng());
        return (exceed + 1.0) / (reps + 1.0);
    }

    ThreadPool& pool = ThreadPool::instance();
    const size_t hw = std::max<size_t>(1, pool.thread_count());
    const size_t chunks = std::min(reps, hw * 4);
    const size_t base = reps / chunks;
    size_t rem = reps % chunks;

    std::vector<std::future<size_t>> futs; futs.reserve(chunks);
    for (size_t c = 0; c < chunks; ++c) {
        const size_t reps_chunk = base + (c < rem ? 1 : 0);
        const uint32_t seed = rng(); // derive independent seeds
        futs.emplace_back(pool.submit([=]() { return worker(reps_chunk, seed); }));
    }
    size_t exceed = 0;
    for (auto& f : futs) exceed += f.get();
    return (exceed + 1.0) / (reps + 1.0); // add-one smoothing
}

namespace util {
    /**
     * @brief Detect a change point using the Kolmogorov–Smirnov statistic.
     *
     * Each vector in @p data is reduced to a scalar via @p reduce and the
     * maximal KS distance over all possible split positions is evaluated.
     *
     * @tparam Reducer Reduction functor converting a vector to a scalar.
     * @param data        Map of sample vectors keyed by size.
     * @param reduce      Reduction functor applied to each vector.
     * @param returnAfter If true, the returned key is taken after the detected split.
     * @return Tuple of <key, KS distance, p-value>.
     */
    template<typename Reducer> std::tuple<size_t, double, double> detectChangePointKS(const std::map<size_t, std::vector<uint32_t>>& data, Reducer reduce, bool returnAfter) {
        if (data.size() <= 1) {
            return {data.empty() ? 0 : data.begin()->first, 0.0, 1.0};
        }

        // One-time reduction of vectors to scalars
        std::vector<size_t> keys;     keys.reserve(data.size());
        std::vector<double> vals;     vals.reserve(data.size());
        for (const auto& kv : data) {
            keys.push_back(kv.first);
            vals.push_back(static_cast<double>(reduce(kv.second)));
        }

        const size_t N = vals.size();
        const size_t splits = (N > 1) ? (N - 1) : 0;

        // Scan all splits sequentially; keep both sides sorted incrementally.
        size_t bestIdx = 0;
        double bestD = -1.0;
        double bestMeanShift = -1.0; // used only for tie-breaks

        std::vector<double> left, right(vals.begin(), vals.end());
        std::sort(right.begin(), right.end());

        // incremental sums for efficient mean computation
        long double sum_left = 0.0L, sum_right = 0.0L;
        for (double v : right) sum_right += v;

        for (size_t split = 0; split < splits; ++split) {
            const double v = vals[split];

            // move v from right -> left (keep sorted)
            auto it_right = std::lower_bound(right.begin(), right.end(), v);
            if (it_right != right.end()) {
                sum_right -= v;
                right.erase(it_right);
            }
            left.insert(std::lower_bound(left.begin(), left.end(), v), v);
            sum_left += v;

            const double D = statisticSorted(left, right);

            // effect size (used only to break near-equal D)
            const double mean_left  = double(sum_left) / double(left.size());
            const double mean_right = right.empty() ? mean_left : double(sum_right) / double(right.size());
            const double mean_shift = std::fabs(mean_left - mean_right);

            // primary: maximize D; secondary: maximize mean_shift on ties within tolerance
            constexpr double TIE_TOL = 1e-12;
            if (D > bestD + TIE_TOL || (std::fabs(D - bestD) <= TIE_TOL && mean_shift > bestMeanShift)) {
                bestD = D;
                bestIdx = split;
                bestMeanShift = mean_shift;
            }
        }

        // Global (FWER-controlled) p-value via permutation of the maximum D across all splits.
        const double p_global = globalMaxPvaluePermutation(vals);

        const size_t idx = returnAfter ? std::min(bestIdx + 1, keys.size() - 1) : bestIdx;
        return {keys[idx], bestD, p_global};
    }
}
