//
// Created by nick- on 8/5/2022.
//

#ifndef CUDATEST_SHAREDMEMORY_CUH
#define CUDATEST_SHAREDMEMORY_CUH

#include "shared_mem_lat.cuh"
#include "sharedMemTest.cuh"

CacheResults executeSharedMemoryChecks(){
    CacheResults result;
    printf("Measure Shared Memory Latency\n\n");
#ifdef IsDebug
    fprintf(out, "Measure Shared Memory Latency\n\n");
#endif //IsDebug
    measure_Shared();
    LatencyTuple latency = measure_shared_Lat();
    result.latencyCycles = latency.latencyCycles;
    result.latencyNano = latency.latencyNano;
    result.benchmarked = true;
    return result;
}

#endif //CUDATEST_SHAREDMEMORY_CUH
