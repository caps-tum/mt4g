//
// Created by nick- on 8/5/2022.
//

#ifndef CUDATEST_READONLYCACHE_CUH
#define CUDATEST_READONLYCACHE_CUH

# include "ro.cuh"
# include "ro_lat.cuh"

CacheResults executeReadOnlyCacheChecks(){
    CacheResults result;
    printf("\n\nMeasure ReadOnly Cache Size\n");
#ifdef IsDebug
    fprintf(out, "\n\nMeasure ReadOnly Cache Size\n\n");
#endif //IsDebug
    CacheSizeResult ROSizeInBytes = measure_ReadOnly();
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

#endif //CUDATEST_READONLYCACHE_CUH
