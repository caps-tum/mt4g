#include "utils/util.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

// estimate noiseTol = median(abs diffs),
// spikeTol = max(diff) *oder* percentile(diff>noiseTol)
std::pair<double,double> estimateTols(const std::vector<uint32_t>& data) {
    if (data.size() < 2) return {0.0, 0.0};

    std::vector<double> diffs; diffs.reserve(data.size()-1);
    for (size_t i = 1; i < data.size(); ++i)
        diffs.push_back(std::abs(double(data[i]) - double(data[i-1])));

    // --- keep only *non-zero* diffs ---------------------------
    std::vector<double> nz; nz.reserve(diffs.size());
    std::copy_if(diffs.begin(), diffs.end(),
                 std::back_inserter(nz),
                 [](double d){ return d > 0.0; });

    if (nz.size() < 2)                       // nothing to estimate
        return {0.0, nz.empty() ? 0.0 : nz.front()};

    // --- robust scale: MAD on non-zeros -----------------------
    const auto med = [&](std::vector<double> v){
        size_t m = v.size()/2;
        std::nth_element(v.begin(), v.begin()+m, v.end());
        double res = v[m];
        if (v.size()%2==0) {
            std::nth_element(v.begin(), v.begin()+m-1, v.begin()+m);
            res = (res + v[m-1]) * 0.5;
        }
        return res;
    };
    double mNz = med(nz);
    std::vector<double> absDev; absDev.reserve(nz.size());
    std::transform(nz.begin(), nz.end(), std::back_inserter(absDev),
                   [&](double d){ return std::abs(d - mNz); });

    double noiseTol = 1.4826 * med(absDev);

    std::nth_element(nz.begin(), nz.begin() + nz.size()*95/100, nz.end());
    double spikeTol = nz[nz.size()*95/100];

    return {noiseTol, spikeTol};
}

namespace util {
    bool isAlmostMonotonic(const std::vector<uint32_t>& v, double noiseTol, double spikeTol) {
        if (v.size() < 2) return true;

        double prev = v[0]; // track the last sample, not a global max

        for (size_t i = 1; i < v.size(); ++i) {
            double curr = v[i];

            // 1) allow small noise downward
            if (curr + noiseTol < prev) // compare to prev, not max
                return false;

            // 2) spike: big jump up *and* immediate fall
            bool spikeUp   = (curr > prev + spikeTol);
            bool dropAfter = (i + 1 < v.size() &&
                            v[i + 1] + noiseTol < curr);

            if (spikeUp && dropAfter)
                return false;

            prev = curr; // always advance
        }
        return true;
    }

    bool hasFlukeOccured(const std::map<size_t, std::vector<uint32_t>>& data) {
        return false;
        std::vector<uint32_t> averages;
        averages.reserve(data.size());  
        for (auto& [_, vec] : data) {
            averages.push_back(static_cast<uint32_t>(util::average(vec)));
        }
        auto [noiseTol, spikeTol] = estimateTols(averages);

        noiseTol += 1; // 0 = problems

        return !util::isAlmostMonotonic(averages, noiseTol, spikeTol); 
    }

    size_t tryGetMostLikelyLineSizeCandidate(const std::vector<size_t>& changePoints) {
        if (changePoints.empty()) return 0;

        size_t candidate = changePoints.front();

        size_t nearestPowerOf2 = util::closestPowerOfTwo(candidate);

        // Cache Line Size is most likely a power of 2
        if (std::find(changePoints.begin(), changePoints.end(), nearestPowerOf2) != changePoints.end()) {
            return nearestPowerOf2;
        }

        return candidate;
    }

    size_t tryGetMostLikelyCacheSizeCandidate(const std::vector<size_t>& changePoints) {
        // best bet as a cache size will most likely be (near) a power of two
        return pickMostTrailingZeros(changePoints);
    }

    // Not working
    uint32_t tryComputeOptimalBandwidthBlockCount(const std::vector<uint32_t>& latencyTimingResults, size_t threadBlockSize) {
        // Each CU has util::getSIMDsPerCU() (= 4 in CDNA 2) SIMD Units, each can execute util::getWarpSize() (= 64 in CDNA 2). 
        // Multiply that by the amount of CUs to get the maximum number of threads that can be scheduled concurrently. However, this does not account for
        // latency hiding
        uint32_t maxConcurrentlySchedulableGlobalThreads = util::getNumberOfComputeUnits() * util::getSIMDsPerCU() * util::getWarpSize();

        // No Measurements => we cannot compute "optimal" latency hiding intervals
        if (latencyTimingResults.empty()) return maxConcurrentlySchedulableGlobalThreads / threadBlockSize;

        auto averageRaw = util::average(latencyTimingResults);

        // This supresses TLB Miss outliers that are irrelevant for bandwidth benchmarks
        auto cappedAtAvg = util::cap(latencyTimingResults, (uint32_t)averageRaw);

        // We use this average to predict how many threads the scheduler needs to hide latencies while keeping scheduling overhead minimal
        auto average = util::average(cappedAtAvg);

        // This is how many threads the scheduler should need to handle latency hiding somewhat well while also not having too much overhead
        auto threads = maxConcurrentlySchedulableGlobalThreads * (uint32_t)average;

        // In my testing the optimal value tended to be the nearest power of 2 of the measured threads value
        return util::closestPowerOfTwo(threads / threadBlockSize);
    }

    std::tuple<size_t, double> tryComputeNearestSegmentSize(size_t base, size_t target) {
        if (target == 0) {
            // avoid division by zero
            return { base, 0.0 };
        }
        double raw = static_cast<double>(base) / static_cast<double>(target);
        double rounded = std::round(raw);                // round to nearest integer
        size_t divisor = static_cast<size_t>(rounded);
        if (divisor == 0) divisor = 1;                    // avoid division by zero again

        size_t new_base = base / divisor;

        // similarity: how close divisor is to original target, normalized to [0,1]
        double similarity = 1.0 - std::fabs(static_cast<double>(divisor) - static_cast<double>(target))
                            / static_cast<double>(target);

        if (similarity < 0.0) similarity = 0.0;

        return { new_base, similarity };
    }
}