//
// Created by nick- on 8/5/2022.
//

#ifndef CUDATEST_L1DATACACHE_CUH
#define CUDATEST_L1DATACACHE_CUH

# include "l1.cuh"
# include "l1LatTest.cuh"
# include "l1_lat.cuh"

CacheResults executeL1DataCacheChecks(){
    CacheResults result;
    printf("\n\nMeasure L1 DataCache Size\n");
#ifdef IsDebug
    fprintf(out, "\n\nMeasure L1 DataCache Size\n\n");
#endif //IsDebug
    CacheSizeResult L1SizeInBytes = measure_L1();
    printf("\n\nMeasure L1 DataCache cache line size\n");
#ifdef IsDebug
    fprintf(out, "\n\nMeasure L1 DataCache cache line size\n\n");
#endif //IsDebug
    unsigned int cacheLineSize = wrapperLineSize(L1SizeInBytes.CacheSize, launchL1KernelBenchmark);
    printf("\n\nMeasure L1 DataCache latencyCycles\n");
#ifdef IsDebug
    fprintf(out, "\n\nMeasure L1 DataCache latencyCycles\n\n");
#endif //IsDebug
    measure_L1_LatTest();
    LatencyTuple latency = measure_L1_Lat();
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

#endif //CUDATEST_L1DATACACHE_CUH
