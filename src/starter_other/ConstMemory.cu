//
// Created by nick- on 8/5/2022.
//

#ifndef CUDATEST_CONSTMEMORY_CUH
#define CUDATEST_CONSTMEMORY_CUH

#include "const15MemTest_sep.cuh"
#include "constL1_5_lat_sep.cuh"
#include "../cudaDeviceProperties.cuh"


void executeConstMemoryChecks() {
    measure_Const15();
    LatencyTuple tuple = measure_ConstL1_5_Lat();
    printf("Latency Cycles: %d\n", tuple.latencyCycles);
    printf("Latency NSec: %d\n", tuple.latencyNano);
}

// Separate starter for the C1.5 Latency due to Constant Memory Limit
int main(int argc, char *argv[]) {
    int deviceID = 0;
    if (argc == 2) {
        char* arg = argv[1];
        if (strstr(arg, "-d:") != nullptr) {
            char* id = &arg[strlen("-d:")];
            deviceID = cvtCharArrToInt(id);
        }
    }

    cudaSetDevice(deviceID);

    executeConstMemoryChecks();
    return 0;
}

#endif //CUDATEST_CONSTMEMORY_CUH
