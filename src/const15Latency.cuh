//
// Created by nick- on 9/7/2022.
//

#ifndef CUDATEST_CONST15LATENCY_CUH
#define CUDATEST_CONST15LATENCY_CUH

#include "GPU_resources.cuh"

// Call external helper exe for Constant L1.5 Latency due to Constant Memory limit of 64 KiB
LatencyTuple getC15Latency(int deviceID) {
    unsigned int cycles = 1, nsecs = 1;
    LatencyTuple result;
    char cmd[1024];
#ifdef IsDebug
    fprintf(out, "Executing helper c15\n");
#endif //IsDebug
#ifdef _WIN32
    snprintf(cmd, 1024, "c15.exe -d:%d", deviceID);
#else
    snprintf(cmd, 1024, "./c15 -d:%d", deviceID);
#endif

    FILE *p;
#ifdef _WIN32
    p = _popen(cmd, "r");
#else
    p = popen(cmd, "r");
#endif

    if (p == nullptr) {
        printf("Could not execute command %s!\n", cmd);
    }

    char line[MAX_LINE_LENGTH] = {0};

    fgets(line, MAX_LINE_LENGTH, p);
#ifdef IsDebug
    fprintf(out, "Cycles: %s\n", line);
#endif //IsDebug
    char* ptr = strchr(line, ':');
    ptr = ptr + 1;
    cycles = cvtCharArrToInt(ptr);

    fgets(line, MAX_LINE_LENGTH, p);
#ifdef IsDebug
    fprintf(out, "Nanoseconds: %s\n", line);
#endif //IsDebug
    ptr = strchr(line, ':');
    ptr = ptr + 1;
    nsecs = cvtCharArrToInt(ptr);

#ifdef _WIN32
    _pclose(p);
#else
    pclose(p);
#endif

    result.latencyCycles = cycles;
    result.latencyNano = nsecs;
    return result;
}

#endif //CUDATEST_CONST15LATENCY_CUH
