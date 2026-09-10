#pragma once

#include <tuple>

namespace benchmark {
    /**
     * @brief Measure the delay between memory requests leaving two caches.
     *
     * @return Pair of departure delays for the tested caches.
     */
    std::tuple<double, double> measureDepartureDelay();
}