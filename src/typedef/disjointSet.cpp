#include <utils/util.hpp>

#include <stdexcept>
#include <utility>

void DisjointSet::add(Key x) {
    // Try to insert, but don't overwrite if it exists
    nodes.try_emplace(x, Node{ x, 1 });
}

bool DisjointSet::contains(Key x) const noexcept {
    return nodes.find(x) != nodes.end();
}

DisjointSet::Key DisjointSet::find(Key x) {
    // Auto-register unknown elements instead of triggering UB
    if (!contains(x)) {
        add(x);
        return x;
    }

    Key root = findRoot(x);
    compressPath(x, root);
    return root;
}

bool DisjointSet::unite(Key a, Key b) {
    a = find(a);
    b = find(b);
    if (a == b) return false;

    // Union by size: attach smaller tree under larger tree
    if (nodes[a].size < nodes[b].size) std::swap(a, b);

    nodes[b].parent = a;
    nodes[a].size += nodes[b].size;
    return true;
}

std::set<std::set<DisjointSet::Key>> DisjointSet::getEquivalenceClasses() const {
    using Class = std::set<Key>;

    // 1) Collect elements per root (temporary map keeps it simple)
    std::map<Key, Class> byRoot;

    auto findRootRO = [this](Key x) {
        // Read-only root search (no compression in const method)
        while (nodes.at(x).parent != x) {
            x = nodes.at(x).parent;
        }
        return x;
    };

    for (const auto &kv : nodes) {
        Key x = kv.first;
        byRoot[findRootRO(x)].insert(x);
    }

    // 2) Convert to set<set>
    std::set<Class> result;
    for (auto &kv : byRoot) {
        result.insert(std::move(kv.second)); // values are independent now
    }
    return result;
}

DisjointSet::Key DisjointSet::findRoot(Key x) {
    // Assumes x is present
    Key root = x;
    while (nodes[root].parent != root) {
        root = nodes[root].parent;
    }
    return root;
}

void DisjointSet::compressPath(Key x, Key root) {
    // Iterative path compression to avoid recursion depth issues
    while (nodes[x].parent != x) {
        Key parent = nodes[x].parent;
        nodes[x].parent = root;
        x = parent;
    }
}
