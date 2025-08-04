#pragma once

#include <cstddef>

inline constexpr std::size_t KiB = 1024;
inline constexpr std::size_t MiB = 1024 * KiB;
inline constexpr std::size_t GiB = 1024 * MiB;

#include <map>
#include <vector>
#include <optional>
#include <tuple>

#include "typedef/launcherFn.hpp"
#include "typedef/cliOptions.hpp"
#include "typedef/enums.hpp"
#include "typedef/disjointSet.hpp"

#include "utils/errorHandling.hpp"
#include "utils/statistics.hpp"
#include "utils/tools.hpp"
#include "utils/printing.hpp"
#include "utils/binarySearch.hpp"
#include "utils/runner.hpp"
#include "utils/heuristic.hpp"
#include "utils/translation.hpp"
#include "utils/hip/helpers.hpp"

#include "config.hpp"
