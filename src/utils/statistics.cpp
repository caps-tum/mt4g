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

namespace util {  
    std::tuple<size_t, double> detectCacheSizeChangePoint(const std::map<size_t, std::vector<uint32_t>>& timingsMap) {
        auto reduction = util::getMagicReductionFunction(timingsMap);
        auto [cp, _, p] = util::detectChangePointKS(timingsMap, reduction, false);
        return {cp, 1 - p};
    }

    std::tuple<size_t, double> detectFetchGranularityChangePoint(const std::map<size_t,std::vector<uint32_t>>& timingsMap) {
        auto [cp, _, p] = util::detectChangePointKS(timingsMap, util::min<uint32_t>, true);
        return {cp, 1-p};
    }

    // What we know: 
    // - The correct answer may only be a multiple of aka the fetch granularity
    // - average timings change point differs from that of the fetch granularity stride => stride > line size
    // - average timings will get faster the farther off the stride is
    // - In an undisturbed measurement the average timings for stride <= correct line size may be slower than at stride = fetch granularity
    // I am aware that this is complicated and could probably be done shorter. It works, however.
    std::tuple<size_t, double> detectLineSizeChangePoint(const std::map<size_t, std::map<size_t, std::vector<uint32_t>>>& lineSizeToCacheSizeTimings) {
        // 1. Compute average timing for each outerKey/innerKey
        std::map<size_t, std::map<size_t, double>> avgData;
        for (const auto& [outerK, innerMap] : lineSizeToCacheSizeTimings) {
            for (const auto& [innerK, vec] : innerMap) {
                double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
                avgData[outerK][innerK] = vec.empty() ? 0.0 : sum / vec.size();
            }
        }

        // 2. Identify boundary pivots: smallest and largest outerKey
        auto fullData = avgData; // keep full copy for penalties & padding
        auto itMin = fullData.begin();
        size_t minKey = itMin->first;
        auto pivotMinMap = itMin->second;
        auto itMax = std::prev(fullData.end());
        size_t maxKey = itMax->first;
        auto pivotMaxMap = itMax->second;

        // 3. Build distance map INCLUDING the pivots (so they get normal scores for padding)
        std::map<size_t, std::vector<std::pair<size_t, double>>> distanceMap;
        for (const auto& [innerK, pivotVal] : pivotMinMap) {
            auto& distVec = distanceMap[innerK];
            for (const auto& [outerK, innerMap] : fullData) {
                // distance vs. pivotMin, only count negative changes
                double diff = innerMap.at(innerK) - pivotVal;
                if (diff > 0.0) diff = 0.0;     // only negative changes
                distVec.emplace_back(outerK, std::abs(diff));
            }
            std::sort(distVec.begin(), distVec.end(),
                    [](auto& a, auto& b){ return a.second < b.second; });
        }

        // 3b) Pre-penalty for mismatch vs. pivotMin change pattern
        // If pivotMin changes between consecutive inner keys but an outerK does NOT,
        // strongly penalize that outerK at the later inner key.
        // Assumes all inner maps share identical keys.
        std::vector<size_t> innerKeys; innerKeys.reserve(pivotMinMap.size());
        for (const auto& [ik, _] : pivotMinMap) innerKeys.push_back(ik);

        for (size_t t = 1; t < innerKeys.size(); ++t) {
            size_t prevK = innerKeys[t-1];
            size_t currK = innerKeys[t];

            double pivotPrev = pivotMinMap.at(prevK);
            double pivotCurr = pivotMinMap.at(currK);
            bool pivotChanged = (pivotCurr != pivotPrev);

            if (!pivotChanged) continue;

            // For each outerK, check whether it failed to change where the pivot did.
            for (const auto& [outerK, innerMap] : fullData) {
                double vPrev = innerMap.at(prevK);
                double vCurr = innerMap.at(currK);
                bool outerChanged = (vCurr != vPrev);
                if (outerChanged) continue;

                // Apply penalty to the entry for (innerK = currK, outerK).
                auto& distVec = distanceMap[currK];
                for (auto& pr : distVec) {
                    if (pr.first == outerK) {
                        pr.second *= pr.second; // HARD penalty
                        break;
                    }
                }
            }
        }

        // Re-sort distance vectors after applying mismatch penalties
        for (auto& [innerK, distVec] : distanceMap) {
            std::sort(distVec.begin(), distVec.end(),
                    [](auto& a, auto& b){ return a.second < b.second; });
        }

        
        // // Print pre-ranking distances
        // for (const auto& [innerK, distVec] : distanceMap) {
        //     std::cout << "Inner key " << innerK << " distances:";
        //     for (const auto& [outerK, dist] : distVec) {
        //         std::cout << " (" << outerK << ", " << dist << ")";
        //     }
        //     std::cout << "\n";
        // }
        

        // 4. Compute penalties using the two largest pivots (maxKey and secondMax)
        auto temp = fullData;
        temp.erase(minKey); // remove min to get two largest to the right
        auto itR = temp.rbegin();
        size_t pivotMax1 = itR->first;
        ++itR;
        size_t pivotMax2 = (itR != temp.rend() ? itR->first : pivotMax1);

        size_t minInner = pivotMinMap.begin()->first;
        size_t maxInner = pivotMinMap.rbegin()->first;
        double midInner  = 0.5 * (minInner + maxInner);
        double halfRange = 0.5 * (maxInner - minInner);

        for (auto& [innerK, distVec] : distanceMap) {
            // precompute max deviations to both pivots USING FULL DATA (pivots included)
            double maxDev1 = 0.0, maxDev2 = 0.0;
            std::vector<double> dev1, dev2;
            dev1.reserve(distVec.size());
            dev2.reserve(distVec.size());
            for (const auto& [outerK, _] : distVec) {
                double v  = fullData[outerK].at(innerK);
                double d1 = std::abs(v - fullData[pivotMax1].at(innerK));
                double d2 = std::abs(v - fullData[pivotMax2].at(innerK));
                dev1.push_back(d1);
                dev2.push_back(d2);
                maxDev1 = std::max(maxDev1, d1);
                maxDev2 = std::max(maxDev2, d2);
            }
            // apply penalty factors in-place
            for (size_t i = 0; i < distVec.size(); ++i) {
                double term2 = (maxDev1 > 0.0 ? 1.0 - dev1[i] / maxDev1 : 0.0);
                double term3 = (maxDev2 > 0.0 ? 1.0 - dev2[i] / maxDev2 : 0.0);
                double weight = 1.0 + 0.5 * (term2 + term3);
                double centrality = (halfRange > 0.0)
                                    ? (1.0 - std::abs(innerK - midInner) / halfRange)
                                    : 0.0;
                distVec[i].second *= weight * (1.0 + centrality);
            }
            std::sort(distVec.begin(), distVec.end(),
                    [](auto& a, auto& b){ return a.second < b.second; });
        }

        
        // // Print penalized distances
        // for (const auto& [innerK, distVec] : distanceMap) {
        //     std::cout << "Penalized distances for inner key " << innerK << ":";
        //     for (const auto& [outerK, dist] : distVec) {
        //         std::cout << " (" << outerK << ", " << dist << ")";
        //     }
        //     std::cout << "\n";
        // }
        

        // 5. Accumulate scores: sum(position * distance) — now includes pivots
        std::map<size_t, double> scores;
        for (auto& [ik, vec] : distanceMap)
            for (size_t p = 0; p < vec.size(); ++p)
                scores[vec[p].first] += p * vec[p].second;

        
        // // print scores before padding & clamping
        // std::cout << "Scores before padding & clamping:\n";
        // for (auto& [k, sc] : scores)
        //     std::cout << "  OuterKey " << k << " -> score " << sc << "\n";
        

        // 6a) pad isolated dips based on neighbors (pivots participate here)
        std::vector<std::pair<size_t,double>> byKey(scores.begin(), scores.end());
        std::sort(byKey.begin(), byKey.end(),
                [](auto& a, auto& b){ return a.first < b.first; });
        size_t n = byKey.size();
        std::vector<double> padded(n);
        for (size_t i = 0; i < n; ++i) {
            double o = byKey[i].second;
            if (n == 1) {
                padded[i] = o;
            }
            else if (i == 0) {
                // left edge uses right neighbor
                padded[i] = (o < byKey[i+1].second) ? byKey[i+1].second : o;
            }
            else if (i+1 == n) {
                // right edge uses left neighbor
                padded[i] = (o < byKey[i-1].second) ? byKey[i-1].second : o;
            }
            else {
                double l = byKey[i-1].second;
                double r = byKey[i+1].second;
                padded[i] = (o < l && o < r) ? std::min(l, r) : o;
            }
        }

        // build padded list & print (still includes pivots)
        std::vector<std::pair<size_t,double>> paddedList(n);
        for (size_t i = 0; i < n; ++i) {
            paddedList[i] = { byKey[i].first, padded[i] };
        }
        
        // std::cout << "Padded scores:\n";
        // for (auto& [k, sc] : paddedList)
        //     std::cout << "  OuterKey " << k << " -> padded " << sc << "\n";
        

        // 6b) AFTER padding, drop pivots from further consideration
        paddedList.erase(
            std::remove_if(paddedList.begin(), paddedList.end(),
                        [&](const auto& pr){ return pr.first == minKey || pr.first == maxKey; }),
            paddedList.end()
        );

        // Safety: if nothing left (or single element), bail gracefully
        if (paddedList.size() < 2) {
            if (paddedList.empty()) {
                // Fallback: return minKey with zero confidence
                return { minKey, 0.0 };
            } else {
                return { paddedList.front().first, 0.0 };
            }
        }

        // 6c) sort by padded score (ascending), tie-break by key
        std::stable_sort(paddedList.begin(), paddedList.end(),
                        [](auto& a, auto& b){
                            if (a.second != b.second) return a.second < b.second;
                            return a.first < b.first;
                        });

        // 6d) downward clamp for monotonicity
        const size_t m = paddedList.size();
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < m; ++j) {
                if (paddedList[j].first > paddedList[i].first
                    && paddedList[j].second < paddedList[i].second)
                {
                    paddedList[i].second = paddedList[j].second;
                }
            }
        }

        
        // // print after clamp
        // std::cout << "Clamped scores (pivots excluded):\n";
        // for (auto& [k, sc] : paddedList)
        //     std::cout << "  OuterKey " << k << " -> clamped " << sc << "\n";
        

        // 6e) final sort: by score asc, then key asc
        std::stable_sort(paddedList.begin(), paddedList.end(),
                        [](auto& a, auto& b){
                            if (a.second != b.second) return a.second < b.second;
                            return a.first < b.first;
                        });

        
        std::cout << "Final adjusted & sorted ranking (pivots excluded):\n";
        for (auto& [k, sc] : paddedList)
            std::cout << "  OuterKey " << k << " -> score " << sc << "\n";
        

        // 7) find change point
        double maxRel = -1.0;
        size_t idx = 0;
        for (size_t i = 0; i + 1 < paddedList.size(); ++i) {
            double c = paddedList[i].second;
            double nsc = paddedList[i+1].second;
            double jump = (nsc + 5.0) / (c + 5.0);
            if (jump > maxRel) {
                maxRel = jump;
                idx = i;
            }
        }
        size_t changeKey = paddedList[idx].first;
        double confidence = 1.0 - 1.0 / maxRel;
        return { changeKey, confidence };
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
    double stdev(const std::vector<uint32_t>& data) {
        if (data.size() < 2) return 0.0;
        double mean = util::average(data);
        double sq_sum = 0.0;
        for (auto val : data)
            sq_sum += (val - mean) * (val - mean);
        return std::sqrt(sq_sum / (data.size() - 1)); // sample stdev
    }
}