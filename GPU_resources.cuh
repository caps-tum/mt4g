//
// Created by nick- on 6/25/2022.
//

#ifndef CUDATEST_GPU_RESOURCES_CUH
#define CUDATEST_GPU_RESOURCES_CUH

#include "ErrorHandler.h"
#include <map>

#define measureSize 2048
#define lessSize 1024
#define LineMeasureSize 256

__shared__ long long s_tvalue[measureSize];
__shared__ unsigned int s_index[measureSize];

typedef struct CacheSizeResult {
    size_t CacheSize = 1; //byte size
    bool realCP = false;
    unsigned int maxSizeBenchmarked = 1; //byte size
} CacheSizeResult;

typedef struct CacheResults {
    CacheSizeResult CacheSize = {};
    unsigned int cacheLineSize = 1;
    unsigned int latencyCycles = 1;
    unsigned int latencyNano = 1;
    unsigned long long bandwidth = 1;
    unsigned int numberPerSM = 0; // only measured for L1 Data, Texture & ReadOnly
    bool benchmarked = false;
} CacheResults;

typedef struct LatencyTuple {
    unsigned int latencyCycles;
    unsigned int latencyNano;
} LatencyTuple;

template <typename T>
struct Triple {
    T first;
    T second;
    T third;
};
typedef Triple<double> dTriple;
typedef Triple<unsigned int> uIntTriple;

template <typename T>
struct Tuple {
    T first;
    T second;
};
typedef Tuple<double> dTuple;

#define FreeWrapBenchmarkLaunchResources()  \
for (int i = 0; i < sizeFlow; i++) {        \
    if (time[i] != nullptr) {               \
        free(time[i]);                      \
    }                                       \
}                                           \
free(time);                                 \
free(avgFlow);                              \
free(potMissesFlow);                        \

/**
 * Wrapper function calling the p-chase benchmark for various caches
 * @param launcher
 * @param begin Begin of the region
 * @param end  End of the region
 * @param stride Stride for the p-chase array, mostly 1
 * @param arrayIncrease How much the array increases each iteration, mostly 1
 * @param type string for the output file
 * @return
 */
int wrapBenchmarkLaunch(bool (*launcher)(int, int, double*, unsigned int*, unsigned int**, int*), int begin, int end, int stride, int arrayIncrease, const char* type) {
    int sizeFlow = 1 + (end-begin);
    double* avgFlow = (double*) malloc(sizeof(double) * sizeFlow);
    unsigned int *potMissesFlow = (unsigned int*) malloc(sizeof(unsigned int) * sizeFlow);
    unsigned int** time = (unsigned int**) malloc(sizeof(unsigned int*) * sizeFlow);
    for (int i = 0; i < sizeFlow; ++i) {
        time[i] = nullptr;
    }

    if (avgFlow == nullptr || potMissesFlow == nullptr || time == nullptr) {
        FreeWrapBenchmarkLaunchResources()
        printErrorCodeInformation(1);
        exit(1);
    }

    double progress = 0.0;
    printf("====================================================================================================\n");
    for (int N = begin; N <= end; N+=arrayIncrease) {
        bool dist = true;
        int count = 5;
        int index = (N - begin) / arrayIncrease;

        while(dist && count > 0) {
            if ((float)(N - begin) / (float)(end-begin) > progress) {
                progress = progress + 0.01;
                printf("=");
                fflush(stdout);
            }
            int error = 0;
            dist = launcher(N, stride, &avgFlow[index], &potMissesFlow[index], &time[index], &error);
            if (error != 0) {
                FreeWrapBenchmarkLaunchResources()
                printErrorCodeInformation(error);
                exit(error);
            }
            --count;
        }
    }

    printAvgFlow(avgFlow, sizeFlow, begin, stride, type);
    printMissesFlow(potMissesFlow, sizeFlow, begin, stride);

    int result = detectChangePoint(time, sizeFlow, measureSize);

    FreeWrapBenchmarkLaunchResources()

    return result;
}

/**
 * Returns the value, which is the most present in the given array
 * @param array
 * @param arraySize
 * @return
 */
unsigned int getMostValueInArray(unsigned int* array, int arraySize) {
    std::map<unsigned int,unsigned int> mostCtrMap;
    for (int i = 0; i < arraySize; ++i) {
        if (array[i] == 0) {
            break;
        }
        if (mostCtrMap.find(array[i]) != mostCtrMap.end()) {
            unsigned int current = mostCtrMap[array[i]];
            mostCtrMap[array[i]] = current + 1;
        } else {
            mostCtrMap[array[i]] = 1;
        }
    }

    unsigned int most = 0;
    unsigned int mostVal = 0;
    for (auto& x: mostCtrMap) {
        if (x.second > most) {
            mostVal = x.first;
            most = x.second;
        }
    }

    return mostVal;
}

/**
 * Wrapper function for the line size
 * @param cacheSizeBytes The size of the cache to be measured
 * @param launcher the p-chase function for the caches
 * @return
 */
unsigned int wrapperLineSize(unsigned int cacheSizeBytes, bool (*launcher)(int, int, double*, unsigned int*, unsigned int**, int*)) {
    unsigned int cacheSizeInts = cacheSizeBytes / 4;

    unsigned int **time = (unsigned int**) malloc(sizeof(unsigned int*));
    if (time == nullptr) {
        printErrorCodeInformation(1);
        exit(1);
    }
    unsigned int lineSize = 0;

    bool dist = true;
    int n = 5;
    while(dist && n > 0) {
        int error = 0;
        dist = launcher((int) cacheSizeInts * 2, 1, nullptr, nullptr, time, &error);
        if (error != 0) {
            printErrorCodeInformation(error);
            free(time[0]);
            free(time);
            exit(1);
        }
        --n;
    }

    unsigned int *h_missIndex = (unsigned int*) calloc(measureSize, sizeof(unsigned int));
    if (h_missIndex == nullptr) {
        free(time[0]);
        free(time);
        printErrorCodeInformation(1);
        exit(1);
    }
    unsigned long long ref = 0;
    for (int i = 0; i < measureSize; ++i) {
        ref = ref + time[0][i];
    }
    ref = ref / measureSize;

    int lastMissIndex = 0;
    int missPtr = 0;
    int tol = 50;

    for (int i = 1; i < measureSize; ++i) {
        if (time[0][i] > ref + tol) {
            h_missIndex[missPtr] = i - lastMissIndex;
            lastMissIndex = i;
            ++missPtr;
        }
    }

    cudaDeviceSynchronize();
    lineSize = getMostValueInArray(h_missIndex, measureSize) * 4;

    free(h_missIndex);
    free(time[0]);
    free(time);

    return lineSize;
}


#endif //CUDATEST_GPU_RESOURCES_CUH
