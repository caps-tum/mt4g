#include "hip/hip_runtime.h"
//
// Created by nick- on 6/25/2022.
//

#ifndef CUDATEST_GPU_RESOURCES_CPP
#define CUDATEST_GPU_RESOURCES_CPP

#include "ErrorHandler.h"
#include <map>
#include <unordered_map>

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
    //unsigned long long bandwidth = 1; -- was before

    unsigned int numberPerSM = 0; // only measured for L1 Data, Texture & ReadOnly
    bool benchmarked = false;

    //// extra values ////
    float bandwidth = 1;
    float throughput = 1;
    bool bw_tested = false;
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

    //printf("\n\nGPUres/119\tDetecting change point..\nAlgorithm performs some heavy computations, stand by..\n");
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
 // TODO test prev implementation and current one
unsigned int getMostValueInArray(unsigned int* array, int arraySize) {
    // Create a hash table to keep track of the frequency of each value in the array
    std::unordered_map<unsigned int, int> frequencyMap;

    // Iterate through the array and increment the frequency of each value in the hash table
    for (int i = 0; i < arraySize; i++) {
        if (array[i] != 0)
            frequencyMap[array[i]]++;
    }

    // Initialize variables to keep track of the most frequent value and its frequency
    unsigned int mostFrequentValue = 0;
    int highestFrequency = 0;

    // Iterate through the hash table and find the value with the highest frequency
    for (auto const& pair : frequencyMap) {
        if (pair.second > highestFrequency) {
            mostFrequentValue = pair.first;
            highestFrequency = pair.second;
        }
    }

    // Return the most frequent value
    return mostFrequentValue;
}


/**
 * Wrapper function for the cache line size
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
        // quick for outliers >20x of the neighbours value
        if (i < measureSize-1 && time[0][i] > time[0][i+1]*20)
            time[0][i] = time[0][i+1];

        ref = ref + time[0][i];
    }
    ref = ref / measureSize;

    int lastMissIndex = 0;
    int missPtr = 0;

    // TODO comment why 1.3 ( = +30%)
    for (int i = 1; i < measureSize; ++i) {
        //printf("gpures229\t%d\t%llu\t%d\n", time[0][i], ref, tol);
        //if (time[0][i] > ref + tol) {
        if (time[0][i] > 1.3*ref) {
            h_missIndex[missPtr] = i - lastMissIndex;
            lastMissIndex = i;
            ++missPtr;
        }
    }

    hipError_t result = hipDeviceSynchronize();
    lineSize = getMostValueInArray(h_missIndex, measureSize) * 4;
    free(h_missIndex);
    free(time[0]);
    free(time);

    return lineSize;
}


#endif //CUDATEST_GPU_RESOURCES_CPP
