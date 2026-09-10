#include <cstddef>

#include "benchmarks/benchmark.hpp"
#include "utils/util.hpp"
#include "const/constArray16384.hpp"

static constexpr auto MIN_EXPECTED_SIZE = 8192;// 8 * KiB
static constexpr auto MAX_ALLOWED_SIZE = MAX_ALLOWED_INDEX * sizeof(uint32_t);// 64 * KiB

//__attribute__((optimize("O0"), noinline))
__global__ void constantL15SizeKernel(uint32_t *timingResults, size_t length, size_t stride) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    __shared__ uint64_t s_timings[MIN_EXPECTED_SIZE / sizeof(uint32_t)]; // sizeof(uint32_t) is correct since we need to store that amount of timing values. 
    __shared__ uint32_t s_index[MIN_EXPECTED_SIZE / sizeof(uint32_t)];

    size_t measureLength = util::min(length / stride, MIN_EXPECTED_SIZE / sizeof(uint32_t));

    uint32_t start, end;
    uint32_t index = 0;

    for (uint32_t k = 0; k < measureLength; k++) {
        s_index[k] = 0;
        s_timings[k] = 0;
    }

    // First round
    for (index = 0; index < length; index += stride) {
        index = arr16384AscStride0[index];
    }

    s_index[0] = index;

    // Second round
    for (index = 0; index < measureLength * stride; index += stride) {
        start = clock();
        index = arr16384AscStride0[index];
        end = clock();
        s_timings[index / stride] = end - start;
    }

    s_index[0] += index;

    for (uint32_t k = 0; k < measureLength; ++k) {
        timingResults[k] = s_timings[k];
    }


    s_index[0] += index;

    timingResults[0] += s_index[0] >> util::min(s_index[0], 32);
}

std::vector<uint32_t> constantL15SizeLauncher(size_t arraySizeBytes, size_t strideBytes) {
    util::hipDeviceReset();

    constexpr size_t CONST_ARRAY_BYTES = CONST_ARRAY_SIZE * sizeof(uint32_t);
    if (arraySizeBytes > MAX_ALLOWED_SIZE) {
        std::cerr << "WARNING: Trying to benchmark constant cache larger than max. allowed constant data (" << arraySizeBytes << "). Continuing with " << MAX_ALLOWED_SIZE << " Bytes" << std::endl;
        arraySizeBytes = MAX_ALLOWED_SIZE;
    }
    if (strideBytes == 0) {
        std::cerr << "WARNING: constantL15SizeLauncher stride of 0 is invalid, capping to "
                  << sizeof(uint32_t) << " Bytes" << std::endl;
        strideBytes = sizeof(uint32_t);
    }
    if (strideBytes > CONST_ARRAY_BYTES) {
        std::cerr << "WARNING: constantL15SizeLauncher stride size capped to "
                  << CONST_ARRAY_BYTES << " Bytes" << std::endl;
        strideBytes = CONST_ARRAY_BYTES;
    }
    if (strideBytes > arraySizeBytes) {
        std::cerr << "WARNING: constantL15SizeLauncher stride larger than array size, capping stride" << std::endl;
        strideBytes = arraySizeBytes;
    }

    // TODO: Find out why all loads after first are 60 cycles despite the constant L1 cache is full at around ~2 * KiB.
    // Same results for cache line size 256 or 64. Maybe old value of 94 cycles was wrong due to it also measuring s_index[k] = j;?
    size_t resultBufferLength = util::min(arraySizeBytes / strideBytes,
                                          MIN_EXPECTED_SIZE / sizeof(uint32_t));
    
    // Allocate GPU VMemory
    uint32_t *d_timingResultBuffer = util::allocateGPUMemory(resultBufferLength);
    

    util::hipCheck(hipDeviceSynchronize());
    constantL15SizeKernel<<<1, util::getMaxThreadsPerBlock()>>>(d_timingResultBuffer, arraySizeBytes / sizeof(uint32_t), strideBytes / sizeof(uint32_t));
    util::hipCheck(hipDeviceSynchronize());

    // Get Results
    std::vector<uint32_t> timingResultBuffer = util::copyFromDevice(d_timingResultBuffer, resultBufferLength);
    

    util::hipDeviceReset();

    return { timingResultBuffer[0] }; // hacky
}


namespace benchmark {
    namespace nvidia {
        CacheSizeResult measureConstantL15Size(size_t cacheFetchGranularityBytes) {
            auto [beginBytes, endBytes] = util::findCacheMissRegion(constantL15SizeLauncher, MIN_EXPECTED_SIZE, MAX_ALLOWED_SIZE, cacheFetchGranularityBytes, CACHE_MISS_REGION_RELATIVE_DIFFERENCE);
        
            // Adjust initial search range to multiples of CACHE_SIZE_BENCH_RESOLUTION and expand when possible
            std::tie(beginBytes, endBytes) =
                util::adjustKiBBoundaries(beginBytes, endBytes,
                                           MIN_EXPECTED_SIZE, MAX_ALLOWED_SIZE,
                                           true);

            std::cout << "[Constant L1.5 Size] Trying Boundaries: " << beginBytes << " - " << endBytes << std::endl;

            std::map<size_t, std::vector<uint32_t>> timings = util::runBenchmarkRange(constantL15SizeLauncher, beginBytes, endBytes, cacheFetchGranularityBytes, CACHE_SIZE_BENCH_RESOLUTION, "Constant L1.5 Size");
            
            auto [changePoint, confidence] = util::detectCacheSizeChangePoint(timings);

            

            CacheSizeResult result = {
                timings,
                confidence == 0 ? CONST_ARRAY_SIZE * sizeof(uint32_t) + 1 : changePoint, // If no change point was detected, we assume CL1.5 > CONST_ARRAY_SIZE * sizeof(uint32_t)
                confidence,
                PCHASE,
                BYTE,
                false
            };

            return result;
        }
    }
}

/*
            bool flukeDetected = false;
            do { // Heuristic: Cache wont get faster with increasing array size, only slower. Thus, you can detect disturbances by checking if the measured timings decreased (significantly) after spiking
                timings 

            util::printMap(timings);
                flukeDetected = util::hasFlukeOccured(timings); // Cache answer times may not decrease again with increasing size, hopefully false most of the time
                if (flukeDetected) { // Restart searching progress with larger area 
                    timings.clear(); 

                    beginBytes -= beginBytes - cacheFetchGranularityBytes > endBytes ? 0 : cacheFetchGranularityBytes; // Prevent underflow
                    endBytes += cacheFetchGranularityBytes;
            
                    std::cout << "Benchmark Constant L1.5 Size measured nonsense, retrying with Boundaries: " << beginBytes << " - " << endBytes << std::endl;
                }
            } while(flukeDetected);
*/