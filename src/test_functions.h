#ifndef MT4G_TEST_FUNCTIONS_H
#define MT4G_TEST_FUNCTIONS_H

#include "ro.h"
#include "texture.h"
#include "l1.h"
# include <cstdio>
# include <cstdint>
# include <cstring>

// small = l1, ro, txt
typedef bool (*BenchmarkFunctionPointer)(int, int, double *, unsigned int *, unsigned int **, int *);

// type: 1 - L1, 2 - RO, 3 - TXT
CacheSizeResult measureSmallCache(int cache_type) {
    // switching depending on cache type
    BenchmarkFunctionPointer functionPtr;

    std::string name;
    switch (cache_type) {
        case 1: // L1
            functionPtr = &launchL1KernelBenchmark;
            name = "L1";
            break;
        case 2: // RO
            functionPtr = &launchROBenchmark;
            name = "RO";
            break;
        case 3: // TXT
            functionPtr = &launchTextureBenchmark;
            name = "Texture";
            break;
    }

    int absoluteLowerBoundary = 1024;
    int absoluteUpperBoundary = 1024 << 10; // 1024 * 1024
    int widenBounds = 8;

    int bounds[2] = {absoluteLowerBoundary, absoluteUpperBoundary};

    getBoundaries(functionPtr, bounds, 5);

    printf("Got Boundaries: %d...%d\n", bounds[0], bounds[1]);

    int cp = -1;
    int begin = bounds[0] - widenBounds;
    int end = bounds[1] + widenBounds;
    int stride = 1;
    int arrayIncrease = 1;

    while (cp == -1 && begin >= absoluteLowerBoundary / sizeof(int) - widenBounds &&
           end <= absoluteUpperBoundary / sizeof(int) + widenBounds &&
           begin > 0 && end <= absoluteUpperBoundary) {

        cp = wrapBenchmarkLaunch(functionPtr, begin, end,
                                 stride, arrayIncrease, name.c_str());

        if (cp == -1) {
            begin = begin - (end - begin);
            end = end + (end - begin);
#ifdef IsDebug
            fprintf(out, "\nGot Boundaries: %d...%d\n", begin, end);
#endif //IsDebug
            printf("\nGot Boundaries: %d...%d\n", begin, end);
        }
    }

    CacheSizeResult result;
    int cacheSizeInInt = (begin + cp * arrayIncrease);
    result.CacheSize = (cacheSizeInInt << 2); // * 4);
    result.realCP = cp > 0;
    result.maxSizeBenchmarked = end << 2; // * 4;
    return result;
}

#endif //MT4G_TEST_FUNCTIONS_H