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
static double statisticSorted(const std::vector<double>& a, const std::vector<double>& b) {
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

// Asymptotic survival function using Kolmogorov's series with Stephens-style finite-sample adjustment.
// lambda = (sqrt(n_eff) + 0.12 + 0.11 / sqrt(n_eff)) * D
static double pAsymptotic(double d, size_t n, size_t m) {
    const double neff   = double(n) * double(m) / double(n + m);
    if (neff <= 0.0) return 1.0;
    const double lambda = (std::sqrt(neff) + 0.12 + 0.11 / std::sqrt(neff)) * d;
    double sum = 0.0;
    for (int j = 1; j < 256; ++j) {
        const double term = std::exp(-2.0 * j * j * lambda * lambda);
        sum += (j & 1) ? term : -term;
        if (term < 1e-14) break; // numeric convergence
    }
    const double p = std::clamp(2.0 * sum, 0.0, 1.0);
    return p;
}

// Exact DP-CDF for n*m ≤ 10_000  (Simard & L'Ecuyer style)
// Kept sequential: wavefront/anti-diagonal parallelization is complex and has high coordination overhead for small sizes; limited benefit.
long double pExact(double d, int n, int m) {
    const int N = n + m;
    // lattice band width (Smirnov/Stephens scaling)
    const int k = int(std::floor(d * std::sqrt(double(n * m) / (n + m)) * N + 1e-12));

    std::vector<long double> prev(m + 1, 0.0L), cur(m + 1, 0.0L);
    prev[0] = 1.0L; // start path
    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= m; ++j) {
            if (!i && !j) continue;
            if (std::abs(i * m - j * n) > k * N) {
                cur[j] = 0.0L;
            } else {
                long double v = 0.0L;
                if (i) v += prev[j];
                if (j) v += cur[j - 1];
                cur[j] = v;
            }
        }
        std::swap(prev, cur);
    }
    // total number of unrestricted paths = binom(n+m, n)
    long double paths = 1.0L;
    for (int i = 1; i <= N; ++i)
        paths = paths * i / (i <= n ? i : (i - n));
    return 1.0L - prev[m] / paths; // survival
}

// Monte-Carlo permutation p-value for *local* two-sample KS (handles ties/discreteness)
// Kept for diagnostics; not used in the final global decision.
static double pPermutationLocal(double D, const std::vector<double>& a, const std::vector<double>& b,
                                size_t reps = 50'000, std::mt19937 rng = std::mt19937{std::random_device{}()}) {
    const size_t n = a.size(), m = b.size();
    if (!n || !m || reps == 0) return 1.0;

    auto worker = [&](size_t reps_chunk, uint32_t seed) -> size_t {
        std::mt19937 local_rng(seed);
        std::vector<double> pool; pool.reserve(n + m);
        pool.insert(pool.end(), a.begin(), a.end());
        pool.insert(pool.end(), b.begin(), b.end());
        std::vector<double> tmp_a(n), tmp_b(m);
        size_t exceed = 0;
        for (size_t r = 0; r < reps_chunk; ++r) {
            std::shuffle(pool.begin(), pool.end(), local_rng);
            std::copy(pool.begin(), pool.begin() + n, tmp_a.begin());
            std::copy(pool.begin() + n, pool.end(), tmp_b.begin());
            std::sort(tmp_a.begin(), tmp_a.end());
            std::sort(tmp_b.begin(), tmp_b.end());
            if (statisticSorted(tmp_a, tmp_b) >= D) ++exceed;
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
        const uint32_t seed = rng();
        futs.emplace_back(pool.submit([=]() { return worker(reps_chunk, seed); }));
    }
    size_t exceed = 0;
    for (auto& f : futs) exceed += f.get();
    return (exceed + 1.0) / (reps + 1.0);
}

// Adaptive wrapper for *local* p-value (kept for completeness)
static double pValueLocal(double D, const std::vector<double>& left, const std::vector<double>& right, bool has_ties) {
    const size_t n = left.size(), m = right.size();
    if (!n || !m) return 1.0;
    if (!has_ties && n * m <= 10'000) return double(pExact(D, int(n), int(m)));
    if (has_ties) return pPermutationLocal(D, left, right);
    return pAsymptotic(D, n, m);
}

// Compute max KS-D over all possible splits for a given sequence of scalars.
static double maxDOverAllSplits(const std::vector<double>& seq) {
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
static double globalMaxPvaluePermutation(const std::vector<double>& vals,
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

// ---------- Change-point detection (KS) ----------
template<typename Reducer>
std::tuple<size_t, double, double> detectChangePointKS(
    const std::map<size_t, std::vector<uint32_t>>& data,
    Reducer reduce,
    bool returnAfter
) {
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

/**
 * @brief Compute the root‑mean‑squared difference between two "graphs"
 *        represented as map<x→vector> by first reducing each vector to
 *        a double via the provided reduction function.
 *
 * @tparam ReductionFunc  Callable that takes const std::vector<uint32_t>&
 *                        and returns a double.
 * @param g1               First graph: x → samples
 * @param g2               Second graph: x → samples
 * @param reduce           Reduction function (e.g. util::average)
 * @return RMSE distance between the two reduced graphs
 */
template<typename ReductionFunc> double computeGraphDistance(const std::map<std::size_t, std::vector<uint32_t>>& g1, const std::map<std::size_t, std::vector<uint32_t>>& g2, ReductionFunc reduce) {
    double sumSq = 0.0;
    std::size_t count = 0;

    auto it1 = g1.begin();
    auto it2 = g2.begin();

    // iterate both maps in parallel (assumes same keys in same order)
    while (it1 != g1.end() && it2 != g2.end()) {
        assert(it1->first == it2->first && "Graphs must share the same x‑axis keys");

        double v1 = reduce(it1->second);
        double v2 = reduce(it2->second);
        double diff = v1 - v2;

        sumSq += diff * diff;
        ++count;
        ++it1; ++it2;
    }

    // if no points, return 0 to indicate "no difference"
    return (count > 0) ? std::sqrt(sumSq / count) : 0.0;
}


// timings: map from line size to map<cache size, vector<uint32_t>> of timings
uint32_t detectLineSizeChange(const std::map<size_t, std::map<size_t, std::vector<uint32_t>>>& timings) {
    // 1. Flatten timings: compute average latency per line size
    std::map<size_t, double> avgPerLine;
    for (const auto& [lineSize, cacheMap] : timings) {
        double sum = 0.0;
        size_t count = 0;
        for (const auto& [cacheSize, samples] : cacheMap) {
            sum += util::average(samples);
            ++count;
        }
        avgPerLine[lineSize] = sum / count;
    }

    // 2. Normalize by smallest line size
    auto sortedSizes = std::vector<size_t>();
    for (const auto& [ls, _] : avgPerLine) sortedSizes.push_back(ls);
    std::sort(sortedSizes.begin(), sortedSizes.end());
    double baseline = avgPerLine[sortedSizes.front()];

    // 3. Compute ratios and first differences
    std::vector<double> ratios;
    for (auto ls : sortedSizes) {
        ratios.push_back(avgPerLine[ls] / baseline);
    }

    // 4. Threshold‑based detection: look for first ratio exceeding a factor (e.g. 1.5),
    //    otherwise choose index with largest jump
    const double THRESH = 1.5;
    size_t changeIdx = 0;
    for (size_t i = 0; i < ratios.size(); ++i) {
        if (ratios[i] > THRESH) { changeIdx = i; break; }
    }
    if (changeIdx == 0) {
        // Fallback: largest difference between successive ratios
        double maxDiff = 0.0;
        for (size_t i = 0; i + 1 < ratios.size(); ++i) {
            double diff = ratios[i + 1] - ratios[i];
            if (diff > maxDiff) { maxDiff = diff; changeIdx = i + 1; }
        }
    }
    return sortedSizes[changeIdx];
}

namespace util {
    // Compute root sum of squared deviations from a given minimum

    double computeRMSFromMin(const std::vector<uint32_t>& vec, double globalMin) {
        if (vec.empty()) return 0.0;
        long double sum = 0.0L;
        for (uint32_t v : vec) {
            const long double d = static_cast<long double>(v) - globalMin;
            sum += d * d;
        }
        return std::sqrt(double(sum / vec.size()));
    }

    // Factory: makes analyzer lambda using computeRMSFromMin (precomputes global min once)
    auto getMagicReductionFunction(const std::map<size_t, std::vector<uint32_t>>& timingsMap) {
        uint32_t globalMinUInt = std::numeric_limits<uint32_t>::max();
        for (const auto& kv : timingsMap) {
            for (uint32_t v : kv.second) if (v < globalMinUInt) globalMinUInt = v;
        }
        const double globalMin = static_cast<double>(globalMinUInt);
        return [globalMin](const std::vector<uint32_t>& vec) -> double {
            return computeRMSFromMin(vec, globalMin);
        };
    }
    
    std::tuple<size_t, double> detectCacheSizeChangePoint(const std::map<size_t, std::vector<uint32_t>>& timingsMap) {
        auto reduction = util::getMagicReductionFunction(timingsMap);

        std::map<size_t, double> reds; 

        for (auto& [size, vec] : timingsMap) {
            reds[size] = reduction(vec);
            std::cout << "Size: " << size << " Reduced to: " << reduction(vec) << std::endl;
        }

        util::pipeMapToPython(reds, "Reduced Values");

        auto [cp, _, p] = detectChangePointKS(timingsMap, reduction, false);
        return {cp, 1-p};
    }

    std::tuple<size_t, double> detectFetchGranularityChangePoint(const std::map<size_t,std::vector<uint32_t>>& timingsMap) {
        auto [cp, _, p] = detectChangePointKS(timingsMap, util::min<uint32_t>, true);
        return {cp, 1-p};
    }

    std::tuple<size_t, double> detectLineSizeChangePoint(const std::map<size_t, std::map<size_t, std::vector<uint32_t>>>& lineSizeToCacheSizeTimings) {
        return {detectLineSizeChange(lineSizeToCacheSizeTimings), 1};
        auto minTimingsMap = lineSizeToCacheSizeTimings.at(util::minKey(lineSizeToCacheSizeTimings));

        std::map<size_t, std::vector<uint32_t>> distances;

        for (auto& [lineSize, timings] : lineSizeToCacheSizeTimings) {
            //std::cout << computeGraphDistance(minTimingsMap, timings, util::average<uint32_t>) << std::endl;

            //util::pipeMapToPython(timings, std::to_string(lineSize));

            distances[lineSize] = {(uint32_t)computeGraphDistance(minTimingsMap, timings, util::average<uint32_t>)};
        }

        auto [lineSize, _, p] = detectChangePointKS(distances, util::average<uint32_t>, false);

        return {lineSize, 1-p};
    }
    

    std::vector<uint32_t> detectShareChangePoint(const std::map<uint32_t, std::tuple<std::vector<uint32_t>, std::vector<uint32_t>>>& timings) {
        if (timings.empty()) {
            return {};
        }
        std::map<uint32_t, uint32_t> largestKeyLUT;

        for (auto& [physicalId, temp] : timings) {
            auto& [a, b] = temp;
            std::vector<uint32_t> result;
            // Build minimum vector, aims to eliminate random flukes
            std::transform(a.begin(), a.end(), b.begin(), std::back_inserter(result), [](uint32_t x, uint32_t y) { return std::min(x, y); });

            largestKeyLUT[(uint32_t)util::average(result)] = physicalId;
        }

        // 50% Increase seems sufficient to assume cache misses. Maybe adjust or replace with parameterless option at a later point
        double acceptanceThreshold =  util::minKey(largestKeyLUT) * 1.5;  

        std::vector<uint32_t> result;
        for (auto& [avg, id] : largestKeyLUT) {
            if (avg > acceptanceThreshold) {
                result.push_back(id);
            }
        } 
        return result; // Return all ids with average latency greater than acceptanceThreshold
    }

    uint32_t detectAmountChangePoint(const std::map<uint32_t, std::tuple<std::vector<uint32_t>, std::vector<uint32_t>>>& timingsMap, double threshold) {
        if (timingsMap.empty()) {
            throw std::invalid_argument("Empty timingsMap");
        }

        // 1) Compute reference sum at the first key
        auto it0 = timingsMap.begin();
        double ref1 = util::average( std::get<0>(it0->second) );
        double ref2 = util::average( std::get<1>(it0->second) );
        double referenceSum = ref1 + ref2;
        double limit = referenceSum - threshold;

        // 2) Scan for earliest drop below limit
        for (auto it = timingsMap.begin(); it != timingsMap.end(); ++it) {
            double a1 = util::average( std::get<0>(it->second) );
            double a2 = util::average( std::get<1>(it->second) );
            if ((a1 + a2) < limit) {
                return it->first;
            }
        }

        // 3) If never below limit, return the largest key
        return timingsMap.rbegin()->first;
    }

    // Calculates the p-th percentile (e.g., 0.95 for 95%)
    double percentile(const std::vector<uint32_t>& data, double p) {
        if (data.empty()) return 0.0;
        std::vector<uint32_t> sorted = data;
        std::sort(sorted.begin(), sorted.end());
        double idx = p * (sorted.size() - 1);
        size_t i = static_cast<size_t>(idx);
        double frac = idx - i;
        if (i + 1 < sorted.size()) {
            return sorted[i] * (1 - frac) + sorted[i + 1] * frac;
        }
        return sorted[i];
    }

    // Calculates the standard deviation
    double stddev(const std::vector<uint32_t>& data) {
        if (data.size() < 2) return 0.0;
        double mean = util::average(data);
        double sq_sum = 0.0;
        for (auto val : data)
            sq_sum += (val - mean) * (val - mean);
        return std::sqrt(sq_sum / (data.size() - 1)); // sample stddev
    }
}