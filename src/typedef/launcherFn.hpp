#pragma once

#include <functional>
#include <cstdint>
#include <vector>

using LauncherFn = std::function<std::vector<uint32_t>(size_t arraySizeBytes, size_t strideBytes)>;