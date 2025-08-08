#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <map>
#include <random>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>
#include <algorithm>

#include <hip/hip_runtime.h>

template <size_t N, typename Map, size_t... Is> auto splitMapImpl(const Map& src, std::index_sequence<Is...>) {
    const size_t total = src.size();
    const size_t base  = total / N;
    const size_t rem   = total % N;

    // build counts: first 'rem' parts get +1
    std::array<size_t, N> counts;
    for (size_t i = 0; i < N; ++i)
        counts[i] = base + (i < rem ? 1 : 0);

    // build iterators bounds[0]=begin, bounds[N]=end
    std::array<typename Map::const_iterator, N+1> bounds;
    bounds[0] = src.begin();
    for (size_t i = 1; i <= N; ++i)
        bounds[i] = std::next(bounds[i-1], counts[i-1]);

    // expand into a tuple of Maps using CTAD (C++17+)
    return std::make_tuple(
        Map(bounds[Is], bounds[Is+1])...
    );
}

namespace util {
    /**
     * @brief Divide a map into @p N equally sized submaps.
     *
     * The input map is split into @p N contiguous sections with sizes
     * differing by at most one element. The resulting submaps preserve
     * the original key ordering.
     *
     * @tparam N   Number of chunks to create.
     * @tparam Map Map type to split.
     * @param src  Source map.
     * @return Tuple containing @p N submaps.
     */
    template <size_t N, typename Map> auto splitMap(const Map& src) {
        return splitMapImpl<N, Map>(
            src, std::make_index_sequence<N>{}
        );
    }

    /**
     * @brief Convenience wrapper returning the smallest map key.
     *
     * @tparam K Key type.
     * @tparam V Mapped type.
     * @param m Map to inspect.
     * @return Minimum key contained in the map.
     */
    template <typename K, typename V> K minKey(const std::map<K, V>& m) {
        return m.begin()->first;
    }

    /**
     * @brief Convenience wrapper returning the largest map key.
     */
    template <typename K, typename V> K maxKey(const std::map<K, V>& m) {
        return std::prev(m.end())->first;
    }

    /**
     * @brief Return the smallest mapped value.
     *
     * Assumes the mapped type is comparable.
     */
    template <typename K, typename V> V minValue(const std::map<K, V>& m) {
        return std::min_element(
            m.begin(), m.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; }
        )->second;
    }

    /**
     * @brief Return the largest mapped value.
     *
     * Assumes the mapped type is comparable.
     */
    template <typename K, typename V> V maxValue(const std::map<K, V>& m) {
        return std::max_element(
            m.begin(), m.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; }
        )->second;
    }

    
    /**
     * @brief Find the next larger key after @p target.
     *
     * @param m      Map to search.
     * @param target Key to compare against.
     * @return Next larger key or std::nullopt.
     */
    template <typename K, typename V> std::optional<K> nextKey(const std::map<K, V>& m, const K& target) {
        auto it = m.upper_bound(target); // first element > target
        if (it == m.end()) {
            return std::nullopt;
        }
        return it->first;
    }

    /**
     * @brief Find the closest smaller key before @p target.
     *
     * @param m      Map to search.
     * @param target Key to compare against.
     * @return Previous smaller key or std::nullopt.
     */
    template <typename K, typename V> std::optional<K> prevKey(const std::map<K, V>& m, const K& target) {
        auto it = m.lower_bound(target); // first element >= target
        if (it == m.begin()) {
            return std::nullopt;
        }
        return std::prev(it)->first;
    }

    /**
     * @brief Apply a reducer to each vector in a map and collect the results.
     *
     * @tparam Key  Key type.
     * @tparam T    Element type inside the vectors.
     * @tparam Func Reduction function accepting a const std::vector<T>&.
     * @return Map with the same keys but reduced values.
     */
    template<typename Key, typename T, typename Func, typename U = std::invoke_result_t<Func, const std::vector<T>&>>
    std::map<Key, U> flatten(const std::map<Key, std::vector<T>>& input, Func f) {
        std::map<Key, U> output;
        for (auto const& [k, vec] : input) {
            output.emplace(k, f(vec));
        }
        return output;
    }

    /**
     * @brief Extract all mapped values into a separate vector.
     */
    template <typename Map> auto values(const Map& m) -> std::vector<typename Map::mapped_type> {
        std::vector<typename Map::mapped_type> result;
        result.reserve(m.size()); 

        for (const auto& [key, value] : m) {
            result.push_back(value);
        }

        return result;
    }

    /**
     * @brief Compare two vectors for element-wise equality.
     */
    template <typename T> bool vectorsEqual(const std::vector<T>& a, const std::vector<T>& b) {
        return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
    }

    /**
     * @brief Generate a randomized pointer chasing pattern.
     *
     * @param arraySizeBytes Size of the array in bytes.
     * @param strideBytes    Stride between elements in bytes.
     * @return Vector describing the pointer chasing sequence.
     */
    std::vector<uint32_t> generateRandomizedPChaseArray(size_t arraySizeBytes, size_t strideBytes);

    /**
     * @brief Generate a sequential pointer chasing pattern.
     *
     * @param arraySizeBytes Size of the array in bytes.
     * @param strideBytes    Stride between elements in bytes.
     * @return Vector describing the pointer chasing sequence.
     */
    std::vector<uint32_t> generatePChaseArray(size_t arraySizeBytes, size_t strideBytes);

    /**
     * @brief Fill a vector with random byte values.
     *
     * @tparam T Element type.
     * @param length Number of elements to generate.
     * @return Vector of random values.
     */
    template<typename T> std::vector<T> generateRandomBytesVector(std::size_t length) {
        static thread_local std::mt19937_64 engine{ std::random_device{}() };  
        static thread_local std::uniform_int_distribution<std::uint64_t> dist64{
            0, UINT64_MAX
        };  

        std::vector<T> vec(length);
        std::uint8_t* bytes = reinterpret_cast<std::uint8_t*>(vec.data());  

        std::size_t total_bytes = length * sizeof(T);
        std::size_t chunks = total_bytes / sizeof(std::uint64_t);
        std::size_t offset = 0;
        for (std::size_t i = 0; i < chunks; ++i) {
            std::uint64_t rnd = dist64(engine);
            std::memcpy(bytes + offset, &rnd, sizeof(rnd));
            offset += sizeof(rnd);
        }

        while (offset < total_bytes) {
            bytes[offset++] = static_cast<std::uint8_t>(engine() & 0xFF);
        }

        return vec;
    }

    /**
     * @brief Extract a slice from the vector using a fixed chunk size.
     *
     * @tparam T Element type.
     * @param original   Source vector.
     * @param sliceSize  Number of elements per slice.
     * @param sliceIndex Index of the slice to retrieve.
     * @return Vector containing the slice.
     */
    template <typename T> std::vector<T> getSlice(const std::vector<T>& original, uint32_t sliceSize, uint32_t sliceIndex) {
        if (sliceSize == 0) {
            return std::vector<T>(0);
        }
        
        size_t steps = original.size() / sliceSize;  // Elemente pro Slice
        auto startIt = original.begin() + static_cast<ptrdiff_t>(sliceIndex * steps);

        std::vector<T> slice(steps);
        std::copy_n(startIt, steps, slice.begin());
        return slice;
    }

    /**
     * @brief Compute the element-wise minimum of two maps with identical keys.
     *
     * @param a First map.
     * @param b Second map.
     * @return Map containing the minimum of each value pair.
     */
    template<typename K, typename V> std::map<K, V> mapMin(const std::map<K, V>& a, const std::map<K, V>& b) {
        std::map<K, V> result;
        for (auto const& [key, va] : a) {
            result.emplace(key, std::min(va, b.at(key)));
        }
        return result;
    }

    /**
     * @brief Cap all vector values at a specified maximum.
     *
     * @tparam T Numeric type.
     * @param v     Input vector.
     * @param capAt Maximum allowed value.
     * @return New vector with capped values.
     */
    template <typename T> std::vector<T> cap(const std::vector<T>& v, T capAt) {
        std::vector<T> capped;
        capped.reserve(v.size());

        std::transform(
            v.begin(), v.end(),
            std::back_inserter(capped),
            [capAt](T e) { return util::min(e, capAt); }
        );
        return capped;
    }
    
    /**
     * @brief Merge two vectors while removing duplicates.
     *
     * @tparam T Element type.
     * @param a First vector.
     * @param b Second vector.
     * @return Union of both vectors with unique entries.
     */
    template<typename T> std::vector<T> unify(const std::vector<T>& a, const std::vector<T>& b) {
        std::unordered_set<T> seen;
        std::vector<T> result;
        result.reserve(a.size() + b.size());

        auto appendUnique = [&](auto const& vec){
            for (auto const& x : vec) {
                if (seen.insert(x).second) {
                    result.push_back(x);
                }
            }
        };

        appendUnique(a);
        appendUnique(b);
        return result;
    }

    /**
     * @brief Convert GPU clock cycles to nanoseconds.
     *
     * @param cycles Number of cycles.
     * @return Time in nanoseconds.
     */
    double convertCyclesToNanoseconds(double cycles);

    /**
     * @brief Return the power of two closest to @p n.
     *
     * @param n Input value.
     * @return Closest power of two.
     */
    uint32_t closestPowerOfTwo(uint32_t n);

    /**
     * @brief Select the value with the most trailing zero bits.
     *
     * @param v Vector of candidate values.
     * @return Element with the largest number of trailing zeros.
     */
    size_t pickMostTrailingZeros(const std::vector<size_t>& v);

    /**
     * @brief Align and expand search boundaries used in cache size benchmarks.
     *
     * The boundaries are rounded to multiples of CACHE_SIZE_BENCH_RESOLUTION. If possible, the lower
     * boundary is decreased by an additional CACHE_SIZE_BENCH_RESOLUTION while staying above
     * @p minAllowed. The upper boundary is increased by an extra CACHE_SIZE_BENCH_RESOLUTION when it
     * does not exceed @p maxAllowed.
     *
     * @param begin       Lower boundary in bytes.
     * @param end         Upper boundary in bytes.
     * @param minAllowed  Minimal allowed boundary value.
     * @param maxAllowed  Maximal allowed boundary value.
     * @param strictUpper If true, the rounded up boundary must remain strictly
     *                    below @p maxAllowed.
     * @return Tuple of adjusted <begin, end> boundaries.
     */
    std::tuple<size_t, size_t> adjustKiBBoundaries(size_t begin, size_t end,
                                                   size_t minAllowed,
                                                   size_t maxAllowed,
                                                   bool strictUpper = false);

    /**
     * @brief Convert nested line-size timings to averages per outer key.
     */
    std::map<size_t, std::vector<uint32_t>>
    flattenLineSizeMeasurementsToAverage(const std::map<size_t, std::map<size_t, std::vector<uint32_t>>>& nested_timings);
}