#include "hip/hip_runtime.h"
//
// Created by nick- on 6/29/2022.
//

#ifndef CUDATEST_L1_L2_DIFF_CPP
#define CUDATEST_L1_L2_DIFF_CPP

# include <cstdio>
# include "hip/hip_runtime.h"
# include "general_functions.h"

#define diffSize 100

__global__ void l1_differ(unsigned int *my_array, unsigned int *durationL1, unsigned int *indexL1) {
    unsigned int start_time, end_time;

    __shared__ ALIGN(16) long long s_tvalue_l1[diffSize];
    __shared__ ALIGN(16) unsigned int s_index_l1[diffSize];

    for (int k = 0; k < diffSize; k++) {
        s_index_l1[k] = 0;
        s_tvalue_l1[k] = 0;
    }

    unsigned int j = 0;

    // First round
    for (int k = 0; k < diffSize; k++) {
        NON_TEMPORAL_LOAD_CA(j, my_array + j);
    }

    // Second round
    for (int k = 0; k < diffSize; k++) {
        LOCAL_CLOCK(start_time);
        NON_TEMPORAL_LOAD_CA(j, my_array + j);
        s_index_l1[k] = j;
        LOCAL_CLOCK(end_time);
        s_tvalue_l1[k] = end_time - start_time;
    }

    for (int k = 0; k < diffSize; k++) {
        indexL1[k] = s_index_l1[k];
        durationL1[k] = s_tvalue_l1[k];
    }
}

__global__ void l2_differ(unsigned int *my_array, unsigned int *durationL2, unsigned int *indexL2) {
    unsigned int start_time, end_time;

    __shared__ ALIGN(16) long long s_tvalue_l2[diffSize];
    __shared__ ALIGN(16) unsigned int s_index_l2[diffSize];

    for (int k = 0; k < diffSize; k++) {
        s_index_l2[k] = 0;
        s_tvalue_l2[k] = 0;
    }

    unsigned int j = 0;

    ////////////////////////
    // First round
    for (int k = 0; k < diffSize; k++) {
        NON_TEMPORAL_LOAD_CG(j, my_array + j);
    }

    // Second round
    for (int k = 0; k < diffSize; k++) {
        LOCAL_CLOCK(start_time);
        NON_TEMPORAL_LOAD_CG(j, my_array + j);
        s_index_l2[k] = j;
        LOCAL_CLOCK(end_time);
        s_tvalue_l2[k] = end_time - start_time;
    }

    for (int k = 0; k < diffSize; k++) {
        indexL2[k] = s_index_l2[k];
        durationL2[k] = s_tvalue_l2[k];
    }
}

bool measureL1_L2_difference(double tol) {
    int error = 0;
    unsigned int *h_a = nullptr, *h_indexL1 = nullptr, *h_timeinfoL1 = nullptr, *h_indexL2 = nullptr, *h_timeinfoL2 = nullptr,
            *d_a = nullptr, *durationL1 = nullptr, *durationL2 = nullptr, *d_indexL1 = nullptr, *d_indexL2 = nullptr;
    double absDistance = 0.;
    hipError_t error_id;
    error_id = hipDeviceReset();

    do {
        // Allocate Memory on Host
        h_a = (unsigned int *) mallocAndCheck("l1_l2_diff", sizeof(unsigned int) * (diffSize),
                                              "h_a", &error);

        h_indexL1 = (unsigned int *) mallocAndCheck("l1_l2_diff", sizeof(unsigned int) * (diffSize),
                                                    "h_indexL1", &error);

        h_timeinfoL1 = (unsigned int *) mallocAndCheck("l1_l2_diff", sizeof(unsigned int) * (diffSize),
                                                       "h_timeinfoL1", &error);

        h_indexL2 = (unsigned int *) mallocAndCheck("l1_l2_diff", sizeof(unsigned int) * (diffSize),
                                                    "h_indexL2", &error);

        h_timeinfoL2 = (unsigned int *) mallocAndCheck("l1_l2_diff", sizeof(unsigned int) * (diffSize),
                                                       "h_timeinfoL2", &error);

        // Allocate Memory on GPU
        if (hipMallocAndCheck("l1_l2_diff", (void **) &d_a, sizeof(unsigned int) * diffSize,
                              "d_a", &error) != hipSuccess)
            break;
        if (hipMallocAndCheck("l1_l2_diff", (void **) &durationL1, sizeof(unsigned int) * diffSize,
                              "durationL1", &error) != hipSuccess)
            break;
        if (hipMallocAndCheck("l1_l2_diff", (void **) &durationL2, sizeof(unsigned int) * diffSize,
                              "durationL2", &error) != hipSuccess)
            break;

        if (hipMallocAndCheck("l1_l2_diff", (void **) &d_indexL1, sizeof(unsigned int) * diffSize,
                              "d_indexL1", &error) != hipSuccess)
            break;
        if (hipMallocAndCheck("l1_l2_diff", (void **) &d_indexL2, sizeof(unsigned int) * diffSize,
                              "d_indexL2", &error) != hipSuccess)
            break;


        // Initialize p-chase array
        for (int i = 0; i < diffSize; i++) {
            h_a[i] = (i + 1) % diffSize;
        }

        // Copy array from Host to GPU
        if (hipMemcpyAndCheck("l1_l2_diff", d_a, h_a, sizeof(unsigned int) * diffSize,
                              "h_a -> d_a", &error, false) != hipSuccess)
            break;

        // Launch L1 Kernel function
        dim3 Db = dim3(1);
        dim3 Dg = dim3(1, 1, 1);
        hipLaunchKernelGGL(l1_differ, Dg, Db, 0, 0, d_a, durationL1, d_indexL1);
        error_id = hipDeviceSynchronize();
        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[L1_L2_DIFF.CPP]: Kernel launch/execution L1 Error: %s\n", hipGetErrorString(error_id));
            error = 5;
            break;
        }
        error_id = hipDeviceSynchronize();

        // Launch L2 Kernel function
        hipLaunchKernelGGL(l2_differ, Dg, Db, 0, 0, d_a, durationL2, d_indexL2);
        error_id = hipDeviceSynchronize();
        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[L1_L2_DIFF.CPP] Kernel launch/execution L2 Error: %s\n", hipGetErrorString(error_id));
            error = 5;
            break;
        }
        error_id = hipDeviceSynchronize();

        // Copy results from GPU to Host
        if (hipMemcpyAndCheck("l1_l2_diff", h_timeinfoL1, durationL1, sizeof(unsigned int) * diffSize,
                              "durationL1 -> h_timeinfoL1", &error, true) != hipSuccess)
            break;
        if (hipMemcpyAndCheck("l1_l2_diff", h_indexL1, d_indexL1, sizeof(unsigned int) * diffSize,
                              "d_indexL1 -> h_indexL1", &error, true) != hipSuccess)
            break;
        if (hipMemcpyAndCheck("l1_l2_diff", h_timeinfoL2, durationL2, sizeof(unsigned int) * diffSize,
                              "durationL2 -> h_timeinfoL2", &error, true) != hipSuccess)
            break;

        if (hipMemcpyAndCheck("l1_l2_diff", h_indexL2, d_indexL2, sizeof(unsigned int) * diffSize,
                              "d_indexL2 -> h_indexL2", &error, true) != hipSuccess)
            break;

#ifdef IsDebug
        for (int i = 0; i < diffSize; i++) {
            fprintf(out, "[%d]: L1=%d, L2=%d\n", h_indexL1[i], h_timeinfoL1[i], h_timeinfoL2[i]);
        }
#endif //IsDebug
        error_id = hipDeviceSynchronize();

        double avgL1 = 0.;
        double avgL2 = 0.;
        createOutputFile(diffSize, diffSize, h_indexL1, h_timeinfoL1, &avgL1, nullptr, "L1Differ_");
        createOutputFile(diffSize, diffSize, h_indexL2, h_timeinfoL2, &avgL2, nullptr, "L2Differ_");

        absDistance = abs(avgL2 - avgL1);
        printf("[L1_L2_DIFF.CPP]: Compare average values: L1 %f <<>> L2 %f, compute absolute distance: %f\n", avgL1,
               avgL2, absDistance);
    } while (false);

    // Free Memory on GPU
    FreeTestMemory({d_a, d_indexL1, d_indexL2, durationL1, durationL2}, true);


    // Free Memory on Host
    FreeTestMemory({h_a, h_indexL1, h_indexL2, h_timeinfoL1, h_timeinfoL2}, false);

    if (error != 0) {
        printErrorCodeInformation(error);
        exit(error);
    }

    error_id = hipDeviceReset();
    return absDistance >= 5.0;
    //return absDistance >= tol;
}

#endif //CUDATEST_L1_L2_DIFF_CPP