//
// Created by nick- on 8/5/2022.
//

#ifndef CUDATEST_CONSTANTCACHE_CUH
#define CUDATEST_CONSTANTCACHE_CUH

# include "constCache2.cuh"
# include "constL1_lat.cuh"
# include "const15Latency.cuh"
# include "const1Numbers.cuh"
# include "LineSize/c1_linesize.cuh"
# include "LineSize/c15_linesize.cuh"

Tuple<CacheResults> executeConstantCacheChecks(int deviceID){
    Tuple<CacheResults> results;
    printf("\n\nMeasure Constant Cache Size\n");
#ifdef IsDebug
    fprintf(out, "\n\nMeasure Constant Cache Size\n\n");
#endif //IsDebug
    CacheSizeResult C1Size = sizeL1();
    CacheSizeResult C15Size = sizeL15();
    printf("\n\nMeasure Constant Cache Cache Line Size\n");
#ifdef IsDebug
    fprintf(out, "\n\nMeasure Constant Cache Line Size\n\n");
#endif //IsDebug
    unsigned int c1CacheLineSize = measure_C1_LineSize(C1Size.CacheSize);
    unsigned int c15CacheLineSize = measure_C15_LineSize();
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

#endif //CUDATEST_CONSTANTCACHE_CUH
