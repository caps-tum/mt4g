#pragma once

#include <cstdint>
#include <unordered_map>
#include <set>
#include <map>

class DisjointSet {
public:
    using Key = uint32_t;

    void add(Key x);
    bool contains(Key x) const noexcept;

    // find(): auto-creates the element if it does not exist (safer than UB)
    [[nodiscard]] Key find(Key x);

    // unite(): returns true if a merge actually happened
    bool unite(Key a, Key b);

    [[nodiscard]] std::set<std::set<Key>> getEquivalenceClasses() const;

private:
    struct Node {
        Key parent;
        uint32_t size;
    };

    // Using unordered_map for O(1) avg. access
    std::unordered_map<Key, Node> nodes;

    Key findRoot(Key x);              // internal: assumes x exists
    void compressPath(Key x, Key root); // internal: path compression
};
