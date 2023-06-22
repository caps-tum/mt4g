#include "hip/hip_runtime.h"

#ifndef CUDATEST_L1
#define CUDATEST_L1

# include <cstdio>

# include "binarySearch.h"
# include "hip/hip_runtime.h"
# include "eval.h"
# include "GPU_resources.h"
# include "l1_latency_size.h"
# include "general_functions.h"

#define HARDCODED_THRESHOLD_FOR_L1 2000

bool launchL1KernelBenchmark(int N, int stride, double *avgOut,
                             unsigned int *potMissesOut,
                             unsigned int **time,
                             int *error) {
    thresholdAndPrefix t1{};
    t1.prefix   = "L1_";

    t1.threshold = HARDCODED_THRESHOLD_FOR_L1;
    t1.type_of_cache = 1;
    return launchFullCacheKernelBenchmark(N, stride, avgOut, potMissesOut, time, error, t1);
}

#endif //CUDATEST_L1

