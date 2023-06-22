#include "hip/hip_runtime.h"

#ifndef CUDATEST_RO
#define CUDATEST_RO

# include <cstdio>
# include <cstdint>

# include "binarySearch.h"
# include "hip/hip_runtime.h"
# include "eval.h"
# include "GPU_resources.h"

__global__ void
RO_size(const unsigned int *__restrict__ my_array, int array_length, unsigned int *duration, unsigned int *index,
        bool *isDisturbed) {
    unsigned int start_time, end_time;
    unsigned int j = 0;

    bool dist = false;

    for (int k = 0; k < measureSize; k++) {
        s_index[k] = 0;
        s_tvalue[k] = 0;
    }

    // First round
    for (int k = 0; k < array_length; k++)
        j = __ldg(&my_array[j]);

    // Second round
    for (int k = 0; k < measureSize; k++) {
        int l = k % array_length;
        start_time = clock();
        j = __ldg(&my_array[l]);
        s_index[k] = j;
        end_time = clock();
        s_tvalue[k] = end_time - start_time;
    }

    for (int k = 0; k < measureSize; k++) {
        if (s_tvalue[k] > 2000) {
            dist = true;
        }
        index[k] = s_index[k];
        duration[k] = s_tvalue[k];

        *isDisturbed = dist;
    }
}


bool launchROBenchmark(int N, int stride, double *avgOut, unsigned int *potMissesOut,
                       unsigned int **time, int *error) {
    hipError_t error_id;
    error_id = hipDeviceReset();

    unsigned int *h_a = nullptr, *h_index = nullptr, *h_timeinfo = nullptr,
            *d_a = nullptr, *duration = nullptr, *d_index = nullptr;
    bool *disturb = nullptr, *d_disturb = nullptr;

    do {
        // Allocate Memory on Host
        h_a = (unsigned int *) mallocAndCheck("ro.h", sizeof(unsigned int) * (N),
                                              "h_a", error);

        h_index = (unsigned int *) mallocAndCheck("ro.h", sizeof(unsigned int) * measureSize,
                                                  "h_index", error);

        h_timeinfo = (unsigned int *) mallocAndCheck("ro.h", sizeof(unsigned int) * measureSize,
                                                     "h_timeinfo", error);

        disturb = (bool *) mallocAndCheck("ro.h", sizeof(bool),
                                          "disturb", error);

        // Allocate Memory on GPU
        if (hipMallocAndCheck("ro.h", (void **) &d_a, sizeof(unsigned int) * N,
                              "d_a", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("ro.h", (void **) &duration, sizeof(unsigned int) * measureSize,
                              "duration", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("ro.h", (void **) &d_index, sizeof(unsigned int) * measureSize,
                              "d_index", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("ro.h", (void **) &d_disturb, sizeof(bool),
                              "d_disturb", error) != hipSuccess)
            break;

        // Initialize p-chase array
        for (int i = 0; i < N; i++) {
            //original:
            h_a[i] = (i + stride) % N;
        }

        // Copy array from Host to GPU
        if (hipMemcpyAndCheck("ro.h", d_a, h_a, sizeof(unsigned int) * N,
                              "h_a -> d_a", error, false) != hipSuccess)
            break;

        error_id = hipDeviceSynchronize();

        // Launch Kernel function
        dim3 Db = dim3(1);
        dim3 Dg = dim3(1, 1, 1);
        hipLaunchKernelGGL(RO_size, Dg, Db, 0, 0, d_a, N, duration, d_index, d_disturb);

        error_id = hipDeviceSynchronize();

        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[RO.CPP]: Kernel launch/execution Error: %s\n", hipGetErrorString(error_id));
            *error = 5;
            break;
        }
        error_id = hipDeviceSynchronize();

        // Copy results from GPU to Host
        if (hipMemcpyAndCheck("ro.h", h_timeinfo, duration, sizeof(unsigned int) * measureSize,
                              "duration -> h_timeinfo", error, true) != hipSuccess)
            break;

        if (hipMemcpyAndCheck("ro.h", h_index, d_index, sizeof(unsigned int) * measureSize,
                              "d_index -> h_index", error, true) != hipSuccess)
            break;

        if (hipMemcpyAndCheck("ro.h", disturb, d_disturb, sizeof(bool),
                              "d_disturb -> disturb", error, true) != hipSuccess)
            break;

        error_id = hipDeviceSynchronize();

        if (!*disturb)
            createOutputFile(N, measureSize, h_index, h_timeinfo, avgOut, potMissesOut, "RO_");
    } while (false);

    // Free Memory on GPU
    FreeTestMemory({d_a, d_index, duration, d_disturb}, true);

    // Free Memory on Host
    bool ret = false;
    if (disturb != nullptr) {
        ret = *disturb;
        free(disturb);
    }

    // Free Memory on Host
    FreeTestMemory({h_a, h_index}, false);

    if (h_timeinfo != nullptr) {
        if (time != nullptr) {
            time[0] = h_timeinfo;
        } else {
            free(h_timeinfo);
        }
    }

    error_id = hipDeviceReset();
    return ret;
}

#endif //CUDATEST_RO


