#pragma once

#include <optional>

namespace benchmark {
    /**
     * @brief Determine the fetch granularity of the L2 cache.
     *
     * @return Detected size in bytes.
     */
    CacheSizeResult measureL2FetchGranularity();
}