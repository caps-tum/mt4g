//
// Created by nick- on 8/31/2022.
//

#ifndef CUDATEST_BINARYSEARCH_H
#define CUDATEST_BINARYSEARCH_H

# include "eval.h"
# include "ErrorHandler.h"


#define FreeBinarySearchBoundariesResources()   \
free(logAvgLower);                              \
if (logAvgMid != nullptr) {                     \
    free(logAvgMid);                            \
}                                               \
if (logAvgUpper != nullptr) {                   \
    free(logAvgUpper);                          \
}                                               \
for (int k = 0; k < searchWindowSize; ++k) {    \
    if (time[k] != nullptr) {                   \
    free(time[k]);                              \
    }                                           \
}                                               \
free(time);                                     \

void finalizeBinarySearchNoCacheMiss(bool (*benchmark)(int, int, double*, unsigned int*, unsigned int**, int*), int* bounds, int searchWindowSize, int totalLowerBound, int totalUpperBound, double tol = 0.5) {
    int lower = bounds[0], upper = bounds[1];

    double* logAvgLower = (double*) malloc(sizeof(double) * searchWindowSize);
    double* logAvgMid = nullptr;
    double* logAvgUpper = nullptr;
    unsigned int ** time = (unsigned int**)malloc(sizeof(unsigned int*) * searchWindowSize);
    for (int i = 0; i < searchWindowSize; ++i) {
        time[i] = nullptr;
    }

    for (int i = 0; i < searchWindowSize; i++) {
        bool b = true;
        int tolCount = 3;
        while (b && tolCount > 0) {
            int error = 0;
            b = benchmark(lower, 1, &logAvgLower[i], nullptr, time, &error);
            if (error != 0) {
                FreeBinarySearchBoundariesResources()
                errorStatus = error;
                return;
            }
            --tolCount;
        }
    }
    postProcessing(logAvgLower, searchWindowSize, 3);
    double windowAvgLower = computeAvg(logAvgLower, searchWindowSize);
#ifdef IsDebug
    fprintf(out, "Avg Lower Window: %f\n", windowAvgLower);
#endif //IsDebug

    logAvgUpper = (double*) malloc(sizeof(double) * searchWindowSize);

    for (int i = 0; i < searchWindowSize; i++) {
        bool b = true;
        int tolCount = 3;
        while (b && tolCount > 0) {
            int error = 0;
            b = (*benchmark)(upper, 1, &logAvgUpper[i], nullptr, time, &error);
            if (error != 0) {
                FreeBinarySearchBoundariesResources()
                errorStatus = error;
                return;
            }
            --tolCount;
        }
    }
    postProcessing(logAvgUpper, searchWindowSize, 3);
    double windowAvgUpper = computeAvg(logAvgUpper, searchWindowSize);
#ifdef IsDebug
    fprintf(out, "Avg Upper Window: %f\n", windowAvgUpper);
#endif //IsDebug
    FreeBinarySearchBoundariesResources()

    if (abs(windowAvgUpper - windowAvgLower) > tol) {
        return;
    } else {
        bounds[0] = totalLowerBound, bounds[1] = totalUpperBound;
    }

}

/**
 * RECURSIVE function to narrow the region for cache miss start down
 * @param benchmark
 * @param bounds
 * @param searchWindowSize
 * @param tol
 */
void binarySearchBoundaries(bool (*benchmark)(int, int, double*, unsigned int*, unsigned int**, int*), int* bounds, int searchWindowSize, int totalLowerBound, int totalUpperBound, double tol = 0.5) {
    // Check latency at lower boundary, upper boundary and the middle and decide where to go further
    double* logAvgLower = (double*) malloc(sizeof(double) * searchWindowSize);
    double* logAvgMid = nullptr;
    double* logAvgUpper = nullptr;
    unsigned int ** time = (unsigned int**)malloc(sizeof(unsigned int*) * searchWindowSize);
    for (int i = 0; i < searchWindowSize; ++i) {
        time[i] = nullptr;
    }

    int lower = bounds[0], upper = bounds[1];
#ifdef IsDebug
    fprintf(out, "Execute binary search mit indices %d - %d\n", lower, upper);
#endif //IsDebug
    if (upper - lower < 750) {
        return;
    }
    int midIndex = lower + ((upper - lower) >> 1); // / 2
#ifdef IsDebug
    fprintf(out, "Mid index: %d\n", midIndex);
#endif //IsDebug

    int shiftToLeft = (searchWindowSize + (2-1)) / 2;
    for (int i = 0; i < searchWindowSize; i++) {
        int size = lower - shiftToLeft + i;
        bool b = true;
        int tolCount = 3;
        while(b && tolCount > 0) {
            int error = 0;
            b = benchmark(size, 1, &logAvgLower[i], nullptr, time, &error);
            if (error != 0) {
                FreeBinarySearchBoundariesResources()
                errorStatus = error;
                return;
            }
            --tolCount;
        }
    }
    postProcessing(logAvgLower, searchWindowSize, 3);
    double windowAvgLower = computeAvg(logAvgLower, searchWindowSize);
#ifdef IsDebug
    fprintf(out, "Avg Lower Window: %f\n", windowAvgLower);
#endif //IsDebug

    logAvgMid = (double*) malloc(sizeof(double) * searchWindowSize);

    for (int i = 0; i < searchWindowSize; i++) {
        int size = midIndex - shiftToLeft + i;
        bool b = true;
        int tolCount = 3;
        while(b && tolCount > 0) {
            int error = 0;
            b = (*benchmark)(size, 1, &logAvgMid[i], nullptr, time, &error);
            if (error != 0) {
                FreeBinarySearchBoundariesResources()
                errorStatus = error;
                return;
            }
            --tolCount;
        }
    }
    postProcessing(logAvgMid, searchWindowSize, 3);
    double windowAvgMid = computeAvg(logAvgMid, searchWindowSize);
    if (windowAvgMid < 0.) {
        printf("error");
    }
#ifdef IsDebug
    fprintf(out, "Avg Mid Window: %f\n", windowAvgMid);
#endif //IsDebug

    if (abs(windowAvgMid - windowAvgLower) > tol) {
        bounds[1] = midIndex + 8;
        FreeBinarySearchBoundariesResources()
        binarySearchBoundaries(benchmark, &bounds[0], searchWindowSize, totalLowerBound, totalUpperBound);
        return;
    }

    logAvgUpper = (double*) malloc(sizeof(double) * searchWindowSize);
    for (int i = 0; i < searchWindowSize; i++) {
        int size = upper - shiftToLeft + i;
        bool b = true;
        int tolCount = 3;
        while(b && tolCount > 0) {
            int error = 0;
            b = (*benchmark)(size, 1, &logAvgUpper[i], nullptr, time, &error);
            if (error != 0) {
                FreeBinarySearchBoundariesResources()
                errorStatus = error;
                return;
            }
            --tolCount;
        }
    }
    postProcessing(logAvgUpper, searchWindowSize, 3);
    double windowAvgUpper = computeAvg(logAvgUpper, searchWindowSize);
#ifdef IsDebug
    fprintf(out, "Avg Upper Window: %f\n", windowAvgUpper);
#endif //IsDebug

    FreeBinarySearchBoundariesResources()

    if (abs(windowAvgUpper - windowAvgMid) > tol) {
        bounds[0] = midIndex - ((upper-lower) >> 4); // / 16
        binarySearchBoundaries(benchmark, &bounds[0], searchWindowSize, totalLowerBound, totalUpperBound);
        return;
    }

    printf("[BinarySearch.h] - ERROR: no cache miss in region |[%d...%d]\n", lower, upper);
    printf("[BinarySearch.h] - Widen Out Search\n");
#ifdef IsDebug
    fprintf(out, "[BinarySeach.h] - ERROR: no cache miss in region |[%d...%d]\n", lower, upper);
    fprintf(out, "[BinarySeach.h] - Widen Out Search\n");
#endif //IsDebug
    // If no cache miss is detected in the whole region, widen out the region boundaries and finalize search
    bounds[0] = std::max(lower - (upper - lower), totalLowerBound);
    bounds[1] = std::min(upper + (upper - lower), totalUpperBound);
    printf("Finalizing with | [%d...%d]\n", bounds[0], bounds[1]);
    finalizeBinarySearchNoCacheMiss(benchmark, &bounds[0], searchWindowSize, totalLowerBound, totalUpperBound);
    printf("Finalized with | [%d...%d]\n", bounds[0], bounds[1]);
}

// Start region search for cache size
void getBoundaries (bool (*benchmark)(int, int, double*, unsigned int*, unsigned int**, int*), int boundaries[2], int searchWindowSize, double tol = 1.23) {
    int absoluteLowerBound = boundaries[0] / (int)sizeof(int);
    int absoluteUpperBound = boundaries[1] / (int)sizeof(int);

    double currentWindowAvg;
    // Doubling the size in each step
    for (int N = absoluteLowerBound; N < absoluteUpperBound; N = N << 1) {
        int shiftToLeft = (searchWindowSize + (2-1)) / 2;

        // Not only one size but a small number of sizes surrounding N
        double* logAvg = (double*) malloc(sizeof(double) * searchWindowSize);
        for (int i = 0; i < searchWindowSize; i++) {
            int size = N - shiftToLeft + i;
            bool dist = true;
            int count = 5;
            while(dist && count > 0) {
                int error = 0;
                dist = (*benchmark)(size, 1, &logAvg[i], nullptr, nullptr, &error);
                if (error != 0) {
                    free(logAvg);
                    printErrorCodeInformation(error);
                    exit(-1);
                }
                --count;
            }
        }
#ifdef IsDebug
        fprintf(out, "\nPrint Average flow around the array size of %d\n", N);
        for (int i = 0; i < searchWindowSize; i++) {
            fprintf(out, "Average for Array Size %d is %f\n", N - shiftToLeft + i, logAvg[i]);
        }
#endif //IsDebug

        postProcessing(logAvg, searchWindowSize, 3);

#ifdef IsDebug
        fprintf(out, "\nPrint Average flow around the array size of %d\n", N);
        for (int i = 0; i < searchWindowSize; i++) {
            fprintf(out, "Average for Array Size %d is %f\n", N - shiftToLeft + i, logAvg[i]);
        }
#endif //IsDebug

        double windowAvg = computeAvg(logAvg, searchWindowSize);
        // If the avg loading time is significantly higher than the iteration before
        if (N != absoluteLowerBound && windowAvg - currentWindowAvg > tol) {
#ifdef IsDebug
            fprintf(out, "Cache miss jump in search detected\n");
#endif //IsDebug
            boundaries[0] = std::max(absoluteLowerBound, (N/2) - (1024/(int)sizeof(int)));
            boundaries[1] = std::min(absoluteUpperBound, N + (1024/(int)sizeof(int)));
            // Start actual binary search within the region
            binarySearchBoundaries(benchmark, &boundaries[0], searchWindowSize, absoluteLowerBound, absoluteUpperBound, tol);

            if (errorStatus != 0) {
                free(logAvg);
                printErrorCodeInformation(errorStatus);
                exit(-1);
            }
            return;
        }
        currentWindowAvg = windowAvg;
        free(logAvg);
    }
}

#endif //CUDATEST_BINARYSEARCH_H
