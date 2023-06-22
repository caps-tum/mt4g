//
// Created by nick- on 8/5/2022.
//

#ifndef CUDATEST_L1DATACACHE_CPP
#define CUDATEST_L1DATACACHE_CPP

# include "l1.h"
# include "general_functions.h"
# include "test_functions.h"

/**
 * @brief Measures the L1 DataCache size, cache line size, latency, and bandwidth
 * @return CacheSizeResult
 */
CacheResults executeL1DataCacheChecks(){
    CacheResults result;
    printf("\n\nMeasure L1 DataCache Size\n");
#ifdef IsDebug
    fprintf(out, "\n\nMeasure L1 DataCache Size\n\n");
#endif //IsDebug
    CacheSizeResult L1SizeInBytes = measureSmallCache(1);//measure_L1();
    printf("\n\nMeasure L1 DataCache cache line size\n");
#ifdef IsDebug
    fprintf(out, "\n\nMeasure L1 DataCache cache line size\n\n");
#endif //IsDebug
    unsigned int cacheLineSize = wrapperLineSize(L1SizeInBytes.CacheSize, launchL1KernelBenchmark);
    printf("\n\nMeasure L1 DataCache latencyCycles\n");
#ifdef IsDebug
    fprintf(out, "\n\nMeasure L1 DataCache latencyCycles\n\n");
#endif //IsDebug
    LatencyTuple latency = measureGeneralCacheLatency(200,
                                                      1000,
                                                      1);
    printf("\n\nMeasure L1 DataCache bandwidth\n");
#ifdef IsDebug
    fprintf(out, "\n\nMeasure L1 DataCache bandwidth\n\n");
#endif //IsDebug
    result.CacheSize = L1SizeInBytes;
    result.cacheLineSize = cacheLineSize;
    result.latencyCycles = latency.latencyCycles;
    result.latencyNano = latency.latencyNano;
    result.benchmarked = true;
    return result;
}

#endif //CUDATEST_L1DATACACHE_CPP
