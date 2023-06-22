//
// Created by nick- on 8/5/2022.
//

#ifndef CUDATEST_MAINMEMORY_CPP
#define CUDATEST_MAINMEMORY_CPP

#include "general_functions.h"

/**
 * @brief Measures the Main Memory latency
 * @param l2SizeInBytes
 * @return CacheResults
 */
CacheResults executeMainMemoryChecks(int l2SizeInBytes){
    CacheResults result;
    printf("Measure Main Memory latencyCycles\n\n");
#ifdef IsDebug
    fprintf(out, "Measure Main Memory latencyCycles\n\n");
#endif //IsDebug
    // stride of 64 means jump over 256 Byte
    // TODO check if i can comment line below out
    // measure_Main(l2SizeInBytes, 64);
    LatencyTuple latency = measureGeneralCacheLatency(l2SizeInBytes,
                                                      1024,
                                                      64);

    result.latencyCycles = latency.latencyCycles * 2;
    result.latencyNano = latency.latencyNano;
    result.benchmarked = true;
    return result;
}

#endif //CUDATEST_MAINMEMORY_CPP
