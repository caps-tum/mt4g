//
// Created by nick- on 8/5/2022.
//

#ifndef CUDATEST_MAINMEMORY_CUH
#define CUDATEST_MAINMEMORY_CUH

#include "main_mem_lat.cuh"
#include "mainMemTest.cuh"

CacheResults executeMainMemoryChecks(int l2SizeInBytes){
    CacheResults result;
    printf("Measure Main Memory latencyCycles\n\n");
#ifdef IsDebug
    fprintf(out, "Measure Main Memory latencyCycles\n\n");
#endif //IsDebug
    // stride of 64 means jump over 256 Byte
    measure_Main(l2SizeInBytes, 64);
    LatencyTuple latency = measure_main_Lat(l2SizeInBytes, 64);
    result.latencyCycles = latency.latencyCycles;
    result.latencyNano = latency.latencyNano;
    result.benchmarked = true;
    return result;
}

#endif //CUDATEST_MAINMEMORY_CUH
