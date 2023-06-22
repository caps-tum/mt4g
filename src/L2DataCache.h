//
// Created by nick- on 8/5/2022.
//

#ifndef CUDATEST_L2DATACACHE_CPP
#define CUDATEST_L2DATACACHE_CPP


# include "LineSize/l2_linesize.h"
#include "general_functions.h"

/**
 * @brief Measures the L2 DataCache size, cache line size, latency, and bandwidth
 * @return CacheSizeResult
 */
CacheResults executeL2DataCacheChecks(unsigned int l2SizeBytes){
    printf("EXECUTE L2 DATACACHE CHECK\n");
#ifdef IsDebug
    fprintf(out, "EXECUTE L2 DATACACHE CHECK\n");
#endif //IsDebug
    CacheResults result;
    printf("\n\nMeasure L2 DataCache Cache Line Size\n");
#ifdef IsDebug
    fprintf(out, "\n\nMeasure L2 DataCache  Cache Line Size\n\n");
#endif //IsDebug
    unsigned int cacheLineSize = measure_L2_LineSize_Alt(l2SizeBytes);
    printf("\n\nMeasure L2 DataCache latencyCycles\n\n");
#ifdef IsDebug
    fprintf(out, "\n\nMeasure L2 DataCache latencyCycles\n\n");
#endif //IsDebug
    int error;
    LatencyTuple latency = launchCacheLatencyBenchmark(200,
                                                       1000,
                                                       1,
                                                       &error);
    result.cacheLineSize = cacheLineSize;
    result.latencyCycles = latency.latencyCycles;
    result.latencyNano = latency.latencyNano;
    result.benchmarked = true;
    return result;
}

#endif //CUDATEST_L2DATACACHE_CPP
