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
     * @brief Compute the RMS distance from a reference minimum.
     *
     * @tparam T Numeric element type.
     * @param vec       Sample vector.
     * @param globalMin Pre-computed minimum used as reference.
     * @return Root-mean-square distance from @p globalMin.
     */
    template<typename T> inline double computeRMSFromMin(const std::vector<T>& vec, double globalMin) {
        if (vec.empty()) return 0.0;
        long double sum = 0.0L;
        for (T v : vec) {
            // difference to global minimum
            const long double d = static_cast<long double>(v) - globalMin;
            sum += d * d;
        }
        return std::sqrt(double(sum / vec.size()));
    }

    /**
     * @brief Build a reducer computing RMS distances using the global minimum.
     *
     * Scans all vectors in @p timingsMap to determine the smallest element and
     * returns a callable that calculates the RMS distance of any vector to that
     * global minimum.
     *
     * @tparam Map Map type whose mapped value is a vector.
     * @param timingsMap Map of vectors to inspect for the minimum.
     * @return Callable reducing a vector to a double.
     */
    template<typename Map> inline auto getMagicReductionFunction(const Map& timingsMap) {
        using Vec = typename Map::mapped_type;
        using T   = typename Vec::value_type;
        // find smallest element over all vectors
        T globalMinVal = std::numeric_limits<T>::max();
        for (const auto& kv : timingsMap) {
            for (T v : kv.second) {
                if (v < globalMinVal) globalMinVal = v;
            }
        }
        const double globalMin = static_cast<double>(globalMinVal);

        // return lambda that closes over globalMin
        return [globalMin](const Vec& vec) -> double {
            return computeRMSFromMin<T>(vec, globalMin);
        };
    }

    /**
     * @brief Apply a sliding minimum filter to suppress outliers.
     *
     * Each value is replaced by the minimum in its surrounding window while the
     * first element remains unchanged.
     *
     * @param samples    Input vector of numeric values.
     * @param windowSize Size of the symmetric window (must be odd).
     * @return Filtered vector.
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

    /**
     * @brief Compute the p-th percentile of the data.
     *
     * @param data Sample values.
     * @param p    Desired percentile in [0,1].
     * @return Interpolated percentile value.
     */
    double percentile(const std::vector<uint32_t>& data, double p);

    /**
     * @brief Calculate the sample standard deviation of the data.
     *
     * @param data Sample values.
     * @return Standard deviation.
     */
    double stdev(const std::vector<uint32_t>& data);

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

    /**
     * @brief Identify compute units exhibiting increased sharing latency.
     *
     * Averages the paired timing vectors per compute unit and returns all IDs
     * whose average exceeds 1.5 times the minimum.
     */
    std::vector<uint32_t> detectShareChangePoint(const std::map<uint32_t, std::tuple<std::vector<uint32_t>, std::vector<uint32_t>>>& timings);
}