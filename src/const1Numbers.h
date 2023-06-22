//
// Created by nick- on 9/7/2022.
// Changed by max- on 24/4/2023.
//

#ifndef CUDATEST_CONST1NUMBERS_CUH
#define CUDATEST_CONST1NUMBERS_CUH

#include "GPU_resources.h"
#include "general_functions.h"
#include "ShareChk/chkAllConst.h"
/**
 * Returns the number of Constant L1 Caches Per SM
 * @param deviceID
 * @param C1SizeBytes
 * @return
 */
unsigned int getNumberOfC1(int deviceID, size_t C1SizeBytes) {
    hipError_t error_id;
    error_id = hipSetDevice(deviceID);

    int deviceCount;
    error_id = hipGetDeviceCount(&deviceCount);
    hipDeviceProp_t deviceProp{};

    if (deviceID >= deviceCount) {
        deviceID = 0;
    }

    error_id = hipGetDeviceProperties(&deviceProp, deviceID);
    if (error_id != hipSuccess){
        printf("const1Numbers: Not success!\n");
    }

    int numCores = deviceProp.maxThreadsPerBlock;
    int error = 0;
    unsigned int N = (C1SizeBytes - 50) >> 2; // / 4;

    double *avgRef = (double*) malloc(sizeof(double));
    unsigned int *potMissesRef = (unsigned int*) malloc(sizeof(unsigned int));

    double* avg1 = (double*) malloc(sizeof(double) * numCores);
    double* avg2 = (double*) malloc(sizeof(double) * numCores);
    unsigned int* potMisses1 = (unsigned int*) malloc(sizeof(unsigned int) * numCores);
    unsigned int* potMisses2 = (unsigned int*) malloc(sizeof(unsigned int) * numCores);

    unsigned int *missesFlow = (unsigned int*) malloc(sizeof(unsigned int) * numCores);

    if (avgRef == nullptr || potMissesRef == nullptr || avg1 == nullptr || avg2 == nullptr ||
        potMisses1 == nullptr || potMisses2 == nullptr || missesFlow == nullptr) {
        FreeTestMemory({avgRef, potMissesRef, avg1, avg2, potMisses1,
                        potMisses2, missesFlow}, false);
        error = 1;
        printErrorCodeInformation(error);
        exit(error);
    }

    launchConstantBenchmarkR1((int)N, avgRef, potMissesRef, nullptr, &error);

    for (int i = 1; i < numCores; i++) {
        bool dist = true;
        int n = 5;
        while (dist && n > 0) {
            dist = launchBenchmarkTwoCoreConst(N, &avg1[i - 1], &avg2[i - 1], &potMisses1[i - 1], &potMisses2[i - 1], nullptr, nullptr,
                                        &error, numCores, 0, i);
            if (error != 0) {
                FreeTestMemory({avgRef, potMissesRef, avg1, avg2, potMisses1,
                                potMisses2, missesFlow}, false);
                printErrorCodeInformation(error);
                exit(error);
            }
            --n;
        }
        unsigned int dist1 = potMisses1[i-1] > potMissesRef[0] ? potMisses1[i-1] - potMissesRef[0] : 0;
        unsigned int dist2 = potMisses2[i-1] > potMissesRef[0] ? potMisses2[i-1] - potMissesRef[0] : 0;
        missesFlow[i - 1] = std::max(dist1, dist2) + 5;
    }
#ifdef IsDebug
    char fileName[64];
    snprintf(fileName, 64, "coreFlowConstTest.log");

    FILE * allCoreFile = fopen(fileName, "w");
    if (allCoreFile == nullptr) {
        FreeTestMemory({avgRef, potMissesRef, avg1, avg2, potMisses1,
                        potMisses2, missesFlow}, false);
        printErrorCodeInformation(error);
        exit(error);
    }
#endif //IsDebug

    double avg = computeAvg(missesFlow, numCores-1);
    double thresh = avg / 5.0;
    unsigned int ref = missesFlow[0];
    unsigned int tol = ref / 3;
    unsigned int countSameCache = 1;
    for (int i = 0; i < numCores-1; i++) {
#ifdef IsDebug
        fprintf(allCoreFile, "%d:%d\n", i + 1, missesFlow[i]);
#endif //IsDebug
        if (missesFlow[i] >= ref - tol || missesFlow[i] >= avg - thresh) {
            ++countSameCache;
        }
    }
#ifdef IsDebug
    fclose(allCoreFile);
#endif //IsDebug

    unsigned int nCaches = numCores / countSameCache;

    FreeTestMemory({avgRef, potMissesRef, avg1, avg2, potMisses1,
                    potMisses2, missesFlow}, false);
    return nCaches;
}

#endif //CUDATEST_CONST1NUMBERS_CUH
