#include <vector>
#include <functional>
#include <stdexcept>
#include <algorithm>
#include <utility>
#include <cstddef>
#include <limits>

#include "utils/util.hpp"

// Safe subtraction: returns max(a - b, min_val)
static inline size_t safeSub(size_t a, size_t b, size_t min_val) {
    return (a > b) ? (a - b) : min_val;
}

// Launch kernel at a single size (clamped) and return the maximum latency measurement.
inline uint32_t samplePivot(const LauncherFn& launch, size_t pivotBytes, size_t strideBytes, double rmsMinReference, size_t minBytes, size_t maxBytes) {
    // ensure pivotBytes and strideBytes stay within allowed bounds when used in launch
    const size_t clampedPivot = std::clamp(pivotBytes, minBytes, maxBytes);
    const size_t clampedStride = std::clamp(strideBytes, sizeof(uint32_t), maxBytes); // assuming stride should be at least sizeof(uint32_t)
    return util::computeRMSFromMin(launch(clampedPivot, clampedStride), rmsMinReference);
}

// Iteratively narrow the region where a latency jump occurs.
// The window shrinks until it's smaller than a fraction of the total range.
std::tuple<size_t, size_t> refineRegion(const LauncherFn& launch, size_t lower, size_t upper,
                                        size_t totalLower, size_t totalUpper, size_t strideBytes,
                                        double rmsMinReference, double tol, double lowerUpperDiff) {
    const auto totalRange = totalUpper - totalLower;

    // continue until window is small enough
    while (upper > lower && (upper - lower) > totalRange * lowerUpperDiff) {
        // mid inside [lower, upper], safe from overflow
        const auto mid = lower + (upper - lower) / 2;

        const double timingsLower = samplePivot(launch, lower, strideBytes, rmsMinReference, totalLower, totalUpper);
        const double timingsMid = samplePivot(launch, mid, strideBytes, rmsMinReference, totalLower, totalUpper);

        if (timingsMid - timingsLower > tol) {
            // jump in [lower, mid]
            upper = mid;
        } else {
            // jump in [mid, upper]
            lower = mid;
        }
    }

    // Expand by strideBytes but clamp into total bounds, avoid underflow
    const size_t expandedLower = safeSub(lower, strideBytes, totalLower);
    const size_t expandedUpper = (upper + strideBytes < totalUpper) ? (upper + strideBytes) : totalUpper;

    return { expandedLower, expandedUpper };
}

namespace util {
    constexpr double TOL_FACTOR = 3.0;
    constexpr size_t EXP_SEARCH_BUFFER = 1024;

    // Locate lower and upper bounds for the cache-miss region using
    // exponential search followed by refined binary search.
    // Returns refined [lower, upper] via refineRegion.
    std::tuple<size_t, size_t> findCacheMissRegion(const LauncherFn& launch, size_t lowerBytes, size_t upperBytes, size_t strideBytes, double lowerUpperDiff) {
        if (lowerBytes > upperBytes) {
            throw std::invalid_argument("lowerBytes must be <= upperBytes");
        }

        // Use lowest stride to ensure cache hits at lowest expected size
        const auto rmsMinReference = util::min(launch(lowerBytes, sizeof(uint32_t)));
        const double minTiming = samplePivot(launch, lowerBytes, strideBytes, rmsMinReference, lowerBytes, upperBytes);
        const double maxTiming = samplePivot(launch, upperBytes, strideBytes, rmsMinReference, lowerBytes, upperBytes);
        const double tol = (maxTiming - minTiming) / TOL_FACTOR;

        // std::cout << rmsMinReference << " " << minTiming << " " << maxTiming << " " << tol << std::endl;

        double prev = minTiming;
        size_t n = lowerBytes;
        while (n < upperBytes) {
            const double cur = samplePivot(launch, n, strideBytes, rmsMinReference, lowerBytes, upperBytes);
            if (n != lowerBytes && cur - prev > tol) {
                // initial bounds with buffer, clamped to [lowerBytes, upperBytes]
                const auto half = n / 2;
                const size_t bufferedLower = (half > EXP_SEARCH_BUFFER) ? (half - EXP_SEARCH_BUFFER) : size_t{0};
                const auto newLower = std::max(lowerBytes, bufferedLower);
                const auto newUpper = std::min(upperBytes, n + EXP_SEARCH_BUFFER);

                return refineRegion(launch, newLower, newUpper, lowerBytes, upperBytes,
                                    strideBytes, rmsMinReference, tol, lowerUpperDiff);
            }
            prev = cur;

            // prevent overflow when doubling
            if (n > upperBytes / 2) break;
            n <<= 1;
        }

        // fallback: full range, already within bounds
        return { lowerBytes, upperBytes };
    }
}
