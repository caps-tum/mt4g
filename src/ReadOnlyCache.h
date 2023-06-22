//
// Created by nick- on 8/5/2022.
//

#ifndef CUDATEST_READONLYCACHE_CPP
#define CUDATEST_READONLYCACHE_CPP

# include "ro.h"
# include "ro_lat.h"
# include "test_functions.h"

CacheResults executeReadOnlyCacheChecks(){
    CacheResults result;
    printf("\n\nMeasure ReadOnly Cache Size\n");
#ifdef IsDebug
    fprintf(out, "\n\nMeasure ReadOnly Cache Size\n\n");
#endif //IsDebug
    CacheSizeResult ROSizeInBytes = measureSmallCache(2);
    printf("\n\nMeasure ReadOnly Cache Cache Line Size\n");
#ifdef IsDebug
    fprintf(out, "\n\nMeasure ReadOnly Cache  Cache Line Size\n\n");
#endif //IsDebug

    unsigned int cachelinesize = wrapperLineSize(ROSizeInBytes.CacheSize, launchROBenchmark);

    printf("\n\nMeasure ReadOnly Cache latencyCycles\n");
#ifdef IsDebug
    fprintf(out, "\n\nMeasure ReadOnly Cache latencyCycles\n\n");
#endif //IsDebug
    LatencyTuple latency = measure_RO_Lat();
    result.CacheSize = ROSizeInBytes;
    result.cacheLineSize = cachelinesize;
    result.latencyCycles = latency.latencyCycles;
    result.latencyNano = latency.latencyNano;
    result.benchmarked = true;
    return result;
}

#endif //CUDATEST_READONLYCACHE_CPP
