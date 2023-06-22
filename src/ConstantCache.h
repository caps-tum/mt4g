//
// Created by nick- on 8/5/2022.
//

#ifndef CUDATEST_CONSTANTCACHE_CPP
#define CUDATEST_CONSTANTCACHE_CPP

# include "constCache2.h"
# include "constL1_lat.h"
# include "const15Latency.h"
# include "const1Numbers.h"

Tuple<CacheResults> executeConstantCacheChecks(int deviceID){
    Tuple<CacheResults> results;
    printf("\n\nMeasure Constant Cache Size\n");
#ifdef IsDebug
    fprintf(out, "\n\nMeasure Constant Cache Size\n\n");
#endif //IsDebug
    CacheSizeResult C1Size = ConstCacheCheck(4, 100, 1300);
    CacheSizeResult C15Size = ConstCacheCheck(200, 800, constArrSize);
    printf("\n\nMeasure Constant Cache Cache Line Size\n");
#ifdef IsDebug
    fprintf(out, "\n\nMeasure Constant Cache Line Size\n\n");
#endif //IsDebug
    // last param (upperLimit) - max cache line size, e.g. 256 <-> 256B
    unsigned int c1CacheLineSize = measureConstantCacheLineSize(1,
                                                                256);

    unsigned int c15CacheLineSize = measureConstantCacheLineSize(2,
                                                                 256);
    printf("\n\nMeasure Constant Cache latencyCycles\n");
#ifdef IsDebug
    fprintf(out, "\n\nMeasure Constant Cache latencyCycles\n\n");
#endif //IsDebug
    LatencyTuple latency = measure_ConstL1_Lat();

    results.first.CacheSize = C1Size;
    results.first.cacheLineSize = c1CacheLineSize;
    results.first.latencyCycles = latency.latencyCycles;
    results.first.latencyNano = latency.latencyNano;

    printf("\n\nMeasure Number Of Constant Caches (C1)\n");
#ifdef IsDebug
    fprintf(out, "\n\nMeasure Number Of Constant Caches (C1)\n\n");
#endif //IsDebug
    results.first.numberPerSM = getNumberOfC1(deviceID, C1Size.CacheSize);
    results.first.benchmarked = true;

    LatencyTuple latencyL15 = getC15Latency(deviceID);
    results.second.CacheSize = C15Size;
    results.second.cacheLineSize = c15CacheLineSize;
    results.second.latencyCycles = latencyL15.latencyCycles;
    results.second.latencyNano = latencyL15.latencyNano;
    results.second.benchmarked = true;

    return results;
}

#endif //CUDATEST_CONSTANTCACHE_CPP
