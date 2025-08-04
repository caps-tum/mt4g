#pragma once

#include <optional>
#include <map>
#include <optional>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <vector>
#include <deque>
#include <type_traits>

namespace util {
    /**
     * Applies a sliding window minimum filter to suppress high outliers.
     * Each value is replaced by the minimum in its surrounding window.
     * The first value remains unchanged
     * 
     * @param samples Input vector of numeric values
     * @param windowSize Size of the symmetric window (must be odd)
     */
    template <typename T> std::vector<T> suppressOutliersWithMinFilter(const std::vector<T>& samples, int32_t windowSize) {
        if (windowSize <= 0 || (windowSize % 2) == 0 || samples.empty()) {
            return samples;
        }

        const int32_t n = samples.size();
        const int32_t half = windowSize / 2;
        std::vector<T> result(n);
        std::deque<int32_t> dq;

        for (int32_t i = 0; i < n; ++i) {
            int leftBound = i - half;
            if (leftBound > 0 && !dq.empty() && dq.front() < leftBound) {
                dq.pop_front();
            }

            while (!dq.empty() && samples[dq.back()] >= samples[i]) {
                dq.pop_back();
            }
            dq.push_back(i);

            result[i] = samples[dq.front()];
        }

        return result;
    }

    /**
     * @brief Sum all values in a vector.
     *
     * @tparam T Numeric element type.
     * @param v Input vector.
     * @return Sum of the elements using a widened type.
     */
    template<typename T> auto sum(const std::vector<T>& v) {
        using SumType = std::conditional_t<std::is_integral_v<T>, uint64_t, double>;
        if (v.empty()) 
            return SumType{0};
        return std::accumulate(v.begin(), v.end(), SumType{0});
    }

    /**
     * @brief Compute the arithmetic mean of a vector.
     *
     * @tparam T Numeric element type.
     * @param v Input vector.
     * @return Average value or 0 if the vector is empty.
     */
    template<typename T> double average(const std::vector<T>& v) {
        if (v.empty()) {
            return 0.0;
        }
        return static_cast<double>(sum(v)) / v.size();
    }
    
    /**
     * @brief Return the greater of two values with common type conversion.
     */
    template <typename T1, typename T2> __device__ __host__ inline auto max(T1 a, T2 b) -> typename std::common_type<T1, T2>::type {
        using CT = typename std::common_type<T1, T2>::type;
        return (CT(a) > CT(b)) ? CT(a) : CT(b);
    }

    /**
     * @brief Return the smaller of two values with common type conversion.
     */
    template <typename T1, typename T2> __device__ __host__ inline auto min(T1 a, T2 b) -> typename std::common_type<T1, T2>::type {
        using CT = typename std::common_type<T1, T2>::type;
        return (CT(a) < CT(b)) ? CT(a) : CT(b);
    }

    /**
     * @brief Return the minimum element of a vector.
     *
     * @tparam T Comparable element type.
     * @param v Vector to search (must not be empty).
     * @return Smallest element.
     */
    template <typename T> inline T min(const std::vector<T>& v) {
        if (v.empty()) {
            throw std::invalid_argument("min(std::vector): vector must not be empty");
        }
        return *std::min_element(v.begin(), v.end());
    }

    /**
     * @brief Return the maximum element of a vector.
     *
     * @tparam T Comparable element type.
     * @param v Vector to search (must not be empty).
     * @return Largest element.
     */
    template <typename T> inline T max(const std::vector<T>& v) {
        if (v.empty()) {
            throw std::invalid_argument("max(std::vector): vector must not be empty");
        }
        return *std::max_element(v.begin(), v.end());
    }

    double percentile(const std::vector<uint32_t>& data, double p);
    double stddev(const std::vector<uint32_t>& data);

    /**
     * @brief Compute the Euclidean distance between two vectors.
     *
     * @tparam T Numeric element type.
     * @param a First vector.
     * @param b Second vector.
     * @return Distance between the vectors.
     */
    template <typename T> double euclideanDistance(const std::vector<T>& a, const std::vector<T>& b) {
        static_assert(std::is_arithmetic<T>::value, "Vector type must be numeric");

        if (a.size() != b.size()) {
            throw std::invalid_argument("Vectors must be of equal length");
        }

        double sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            sum += diff * diff;
        }

        return std::sqrt(sum);
    }

    /**
     * @brief Detect the change point that marks a cache size boundary.
     *
     * @param timingsMap Map of array size to timing vectors.
     * @return Detected change points in bytes.
     */
    std::tuple<size_t, double> detectCacheSizeChangePoint(const std::map<size_t, std::vector<uint32_t>>& timingsMap);

    /**
     * @brief Detect change points for fetch granularity.
     *
     * @param timingsMap Map of array size to timing vectors.
     * @return Detected change points in bytes.
     */
    std::tuple<size_t, double> detectFetchGranularityChangePoint(const std::map<size_t, std::vector<uint32_t>>& timingsMap);

    /**
     * @brief Determine the cache line size from timing maps.
     *
     * @param lineSizeToCacheSizeTimings Nested map of line size to cache size to timings.
     * @return Detected change points in bytes.
     */
    std::tuple<size_t, double> detectLineSizeChangePoint(const std::map<size_t, std::map<size_t, std::vector<uint32_t>>>& lineSizeToCacheSizeTimings);

    /**
     * @brief Detect when an amount benchmark drops below a threshold.
     *
     * @param timingsMap Map of amount to pair of timing vectors.
     * @param threshold  Relative drop threshold.
     * @return Detected change point in the amount domain.
     */
    uint32_t detectAmountChangePoint(const std::map<uint32_t, std::tuple<std::vector<uint32_t>, std::vector<uint32_t>>>& timingsMap, double threshold);

    std::vector<uint32_t> detectShareChangePoint(const std::map<uint32_t, std::tuple<std::vector<uint32_t>, std::vector<uint32_t>>>& timings);

    double computeRMSFromMin(const std::vector<uint32_t>& vec, double globalMin);
    auto getMagicReductionFunction(const std::map<size_t, std::vector<uint32_t>>& timingsMap);
}