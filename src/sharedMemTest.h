#include "hip/hip_runtime.h"

#ifndef CUDATEST_SHARED
#define CUDATEST_SHARED

# include <cstdio>

# include "hip/hip_runtime.h"
# include "eval.h"
# include "GPU_resources.h"
#include "general_functions.h"

#define sharedTestSize 400

__global__ void shared_test (unsigned int * duration, unsigned int *index, bool* isDisturbed);

bool launchSharedKernelBenchmark(double *avgOut, unsigned int* potMissesOut, unsigned int** time, int* error);

#define FreeMeasureShared() \
free(avg);                  \
free(misses);               \
free(time);                 \


CacheResults measure_Shared() {
    double *avg = (double*) malloc(sizeof(double));
    unsigned int* misses = (unsigned int*) malloc(sizeof(unsigned int));
    unsigned int** time = (unsigned int**) malloc(sizeof(unsigned int*));
    if (avg == nullptr || misses == nullptr || time == nullptr) {
        FreeMeasureShared()
        printErrorCodeInformation(1);
        exit(1);
    }

    int error = 0;
    bool dist = true;
    int count = 5;

    while(dist && count > 0) {
        dist =  launchSharedKernelBenchmark(avg, misses, time, &error);
        --count;
    }

    free(time[0]);
    FreeMeasureShared()

    if (error != 0) {
        printErrorCodeInformation(error);
        exit(error);
    }

    return CacheResults{};
}

#define SHARED_MEM_TEST "sharedMemTest.h"

bool launchSharedKernelBenchmark(double *avgOut, unsigned int* potMissesOut, unsigned int** time, int* error) {
    hipError_t error_id;

    unsigned int* h_index = nullptr, *h_timeinfo = nullptr, *duration = nullptr, *d_index = nullptr;
    bool* disturb = nullptr, *d_disturb = nullptr;

    do {
        // Allocate Memory on Host
        h_index = (unsigned int *) mallocAndCheck(SHARED_MEM_TEST, sizeof(unsigned int) * lessSize,
                                                  "h_index", error);

        h_timeinfo = (unsigned int *) mallocAndCheck(SHARED_MEM_TEST, sizeof(unsigned int) * lessSize,
                                                     "h_timeinfo", error);

        disturb = (bool *) mallocAndCheck(SHARED_MEM_TEST, sizeof(bool),
                                          "disturb", error);

        // Allocate Memory on GPU
        if (hipMallocAndCheck(SHARED_MEM_TEST, (void **) &duration, sizeof(unsigned int) * lessSize,
                              "duration", error) != hipSuccess)
            break;

        if (hipMallocAndCheck(SHARED_MEM_TEST, (void **) &d_index, sizeof(unsigned int) * lessSize,
                              "d_index", error) != hipSuccess)
            break;

        if (hipMallocAndCheck(SHARED_MEM_TEST, (void **) &d_disturb, sizeof(bool),
                              "d_disturb", error) != hipSuccess)
            break;

        error_id = hipDeviceSynchronize();

        // Launch Kernel function
        dim3 Db = dim3(1);
        dim3 Dg = dim3(1, 1, 1);
        hipLaunchKernelGGL(shared_test, Dg, Db, 0, 0, duration, d_index, d_disturb);

        error_id = hipDeviceSynchronize();

        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[SHAREDMEMTEST.CPP]: Kernel launch/execution Error: %s\n", hipGetErrorString(error_id));
            *error = 5;
            break;
        }
        error_id = hipDeviceSynchronize();

        // Copy results from GPU to Host
        if (hipMemcpyAndCheck(SHARED_MEM_TEST, h_timeinfo, duration, sizeof(unsigned int) * lessSize,
                              "duration -> h_timeinfo", error, true) != hipSuccess)
            break;

        if (hipMemcpyAndCheck(SHARED_MEM_TEST, h_index, d_index, sizeof(unsigned int) * lessSize,
                              "d_index -> h_index", error, true) != hipSuccess)
            break;

        if (hipMemcpyAndCheck(SHARED_MEM_TEST, disturb, d_disturb, sizeof(bool),
                              "d_disturb -> disturb", error, true) != hipSuccess)
            break;
        error_id = hipDeviceSynchronize();

        createOutputFile(sharedTestSize, lessSize, h_index, h_timeinfo, avgOut, potMissesOut, "Shared_");
    } while(false);

    // Free Memory on GPU
    FreeTestMemory({d_index, duration, d_disturb}, true);

    // Free Memory on Host
    bool ret = false;
    if (disturb != nullptr) {
        ret = *disturb;
        free(disturb);
    }

    FreeTestMemory({h_index}, false);

    SET_PART_OF_2D(time, h_timeinfo);

    error_id = hipDeviceReset();
    return ret;
}

__global__ void shared_test (unsigned int * duration, unsigned int *index, bool* isDisturbed) {

    unsigned int start_time, end_time;
    __shared__ ALIGN(16) unsigned int s_array[sharedTestSize];

    // Creation of array needs to be done on kernel
    for (int i = 0; i < sharedTestSize; i++) {
        s_array[i] = (i + 1) % sharedTestSize;
    }

    __shared__ ALIGN(16) long long shared_tvalue[lessSize];
    __shared__ ALIGN(16) unsigned int shared_index[lessSize];

    bool dist = false;
    unsigned int j = 0;

    for(int k=0; k<lessSize; k++){
        shared_index[k] = 0;
        shared_tvalue[k] = 0;
    }

    // No first round required
    for (int k = 0; k < lessSize; k++) {
        start_time = clock();
        j = s_array[j];
        shared_index[k] = j;
        end_time = clock();
        shared_tvalue[k] = end_time-start_time;
    }

    for(int k=0; k<lessSize; k++){
        if (shared_tvalue[k] > 1200) {
            dist = true;
        }
        index[k]= shared_index[k];
        duration[k] = shared_tvalue[k];
    }
    *isDisturbed = dist;
}

#endif //CUDATEST_SHARED

