#pragma once

namespace benchmark {
    namespace nvidia {
        /**
         * @brief Check if global loads are serviced by L1 or L2 on NVIDIA GPUs.
         *
         * @param tolerance Allowed variance for timing comparisons.
         * @return true if L1 is used, false otherwise.
         */
        bool isL1UsedForGlobalLoads(double tolerance);
    }
}