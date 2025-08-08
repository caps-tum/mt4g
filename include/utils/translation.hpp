#pragma once

#include <vector>

namespace util {
    /**
     * @brief Build a lookup table mapping logical to physical compute units.
     *
     * @return Vector where the index is the logical CU and the value the physical CU.
     */
    std::vector<uint32_t> getLogicalToPhysicalCUsLUT();
}