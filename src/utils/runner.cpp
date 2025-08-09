#include <map>

#include "utils/util.hpp"

namespace util {
    std::map<size_t, std::vector<uint32_t>> runBenchmarkRange(LauncherFn launcher, size_t beginBytes, size_t endBytes, size_t strideBytes, size_t arrayIncreaseBytes, const std::string& tag) {
        size_t steps = (endBytes - beginBytes) / arrayIncreaseBytes + 1;

        std::map<size_t, std::vector<uint32_t>> timings;

        std::cout << "[" << tag << "] Start measuring " << steps << " steps\n";

        for (size_t step = 0; step < steps; step += 1) { 
            size_t sizeBytes = beginBytes + step * arrayIncreaseBytes;

            std::vector<uint32_t> t = launcher(sizeBytes, strideBytes);

            timings[sizeBytes] = std::move(t);

            double percent = steps > 1 ? (double(step) / double(steps - 1)) * 100.0 : 100.0; 
            
            std::cout << "\rProgress: " << std::fixed << std::setprecision(2) << percent << '%' << std::flush;
        }


        std::cout << "\n[" << tag << "] Done.\n";
        return timings;
    }
}