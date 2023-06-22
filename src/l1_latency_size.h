//
// Created by max on 12.04.23.
//
#include "hip/hip_runtime.h"

#ifndef CUDATEST_L1BOTHTEST
#define CUDATEST_L1BOTHTEST

# include <cstdio>

# include "binarySearch.h"
# include "hip/hip_runtime.h"
# include "eval.h"
# include "GPU_resources.h"
# include "general_functions.h"

__global__ void
cache_test(unsigned int *my_array, int array_length, unsigned int *duration,
           unsigned int *index, bool *isDisturbed, thresholdAndPrefix t1) {
    unsigned int start_time, end_time;
    bool dist = false;
    unsigned int j = 0;

    for (int k = 0; k < measureSize; k++) {
        s_index[k] = 0;
        s_tvalue[k] = 0;
    }

    if (t1.type_of_cache == 1) {
        // L1
        for (int k = 0; k < array_length; k++) {
            NON_TEMPORAL_LOAD_CA(j, my_array + j);
        }

        for (int k = 0; k < measureSize; k++) {
            start_time = clock();
            NON_TEMPORAL_LOAD_CA(j, my_array + j);
            end_time = clock();
            s_tvalue[k] = end_time - start_time;
        }
    } else {
        // L2
        // First round
        for (int k = 0; k < array_length; k++) {
            NON_TEMPORAL_LOAD_CG(j, my_array + j);
        }

        // Second round
        for (int k = 0; k < measureSize; k++) {
            start_time = clock();
            NON_TEMPORAL_LOAD_CG(j, my_array + j);
            end_time = clock();
            s_tvalue[k] = end_time - start_time;
        }
    }

    for (int k = 0; k < measureSize; k++) {
        if (s_tvalue[k] > t1.threshold) {
            dist = true;
        }
        index[k] = s_index[k];
        duration[k] = s_tvalue[k];
    }

    *isDisturbed = dist;
}

bool launchFullCacheKernelBenchmark(int N, int stride, double *avgOut, unsigned int *potMissesOut,
                                    unsigned int **time, int *error,
                                    thresholdAndPrefix t1) {

    hipError_t error_id;

    unsigned int *h_a = nullptr, *h_index = nullptr, *h_timeinfo = nullptr,
            *d_a = nullptr, *duration = nullptr, *d_index = nullptr;
    bool *disturb = nullptr, *d_disturb = nullptr;

    do {
        // Allocate Memory on Host
        h_a = (unsigned int *) mallocAndCheck("l1lat/110", sizeof(unsigned int) * (N),
                                              "h_a", error);

        h_index = (unsigned int *) mallocAndCheck("l1lat/113", sizeof(unsigned int) * measureSize,
                                                  "h_index", error);

        h_timeinfo = (unsigned int *) mallocAndCheck("l1lat/116", sizeof(unsigned int) * measureSize,
                                                     "h_timeinfo", error);

        disturb = (bool *) mallocAndCheck("l1lat/119", sizeof(bool),
                                          "disturb", error);

        // Allocate Memory on GPU
        if (hipMallocAndCheck("l1lat/123", (void **) &d_a, sizeof(unsigned int) * N,
                              "d_a", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("l1lat/127", (void **) &duration, sizeof(unsigned int) * measureSize,
                              "duration", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("l1lat/131", (void **) &d_index, sizeof(unsigned int) * measureSize,
                              "d_index", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("l1lat/135", (void **) &d_disturb, sizeof(bool),
                              "d_disturb", error) != hipSuccess)
            break;

        // Initialize p-chase array
        for (int i = 0; i < N; i++) {
            h_a[i] = (i + stride) % N;
        }

        // Copy array from Host to GPU
        if (hipMemcpyAndCheck("l1lat/145", d_a, h_a, sizeof(unsigned int) * N,
                              "h_a -> d_a", error, false) != hipSuccess)
            break;
        error_id = hipDeviceSynchronize();

        // Launch Kernel function
        dim3 Db = dim3(1);
        dim3 Dg = dim3(1, 1, 1);

        // last value - max value of test result; any value above will trigger error
        hipLaunchKernelGGL(cache_test, Dg, Db, 0, 0, d_a, N, duration, d_index, d_disturb, t1);

        error_id = hipDeviceSynchronize();

        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[L1.CUH]: Kernel launch/execution Error: %s\n", hipGetErrorString(error_id));
            *error = 5;
            break;
        }
        error_id = hipDeviceSynchronize();

        // Copy results from GPU to Host
        if (hipMemcpyAndCheck("l1lat/168", h_timeinfo, duration, sizeof(unsigned int) * measureSize,
                              "duration -> h_timeinfo", error, true) != hipSuccess)
            break;

        if (hipMemcpyAndCheck("l1lat/172", h_index, d_index, sizeof(unsigned int) * measureSize,
                              "d_index -> h_index", error, true) != hipSuccess)
            break;

        if (hipMemcpyAndCheck("l1lat/176", disturb, d_disturb, sizeof(bool),
                              "d_disturb -> disturb", error, true) != hipSuccess)
            break;

        error_id = hipDeviceSynchronize();

        if (!*disturb)
            createOutputFile(N, measureSize, h_index, h_timeinfo, avgOut, potMissesOut, t1.prefix);

    } while (false);

    // Free Memory on GPU
    FreeTestMemory({d_a, d_index, duration, d_disturb}, true);

    bool ret = false;
    if (disturb != nullptr) {
        ret = *disturb;
        free(disturb);
    }

    // Free Memory on Host
    FreeTestMemory({h_a, h_index}, false);

    SET_PART_OF_2D(time, h_timeinfo);

    hipError_t result = hipDeviceReset();
    return ret;

}

#endif //CUDATEST_L1BOTHTEST