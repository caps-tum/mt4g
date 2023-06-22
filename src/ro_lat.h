#include "hip/hip_runtime.h"

#ifndef CUDATEST_RO_LAT
#define CUDATEST_RO_LAT

# include <cstdio>

# include "hip/hip_runtime.h"
# include "eval.h"
# include "GPU_resources.h"

__global__ void ro_lat_test(const unsigned int* __restrict__ my_array, int array_length, struct dataForTest d1);

LatencyTuple launchROLatKernelBenchmark(int N, int stride, int* error);

LatencyTuple measure_RO_Lat() {
    int stride = 1;
    int error = 0;
    LatencyTuple lat = launchROLatKernelBenchmark(200, stride, &error);
    if (error != 0) {
        printErrorCodeInformation(error);
        exit(error);
    }
    return lat;
}


LatencyTuple launchROLatKernelBenchmark(int N, int stride, int* error) {
    LatencyTuple result;
    hipError_t error_id;

    unsigned int *h_a = nullptr, *h_time = nullptr, *d_a = nullptr, *d_time = nullptr;

    do {
        // Allocate Memory on Host
        h_a = (unsigned int *) malloc(sizeof(unsigned int) * (N));
        if (h_a == nullptr) {
            printf("[RO_LAT.H]: malloc h_a Error\n");
            *error = 1;
            break;
        }

        h_time = (unsigned int *) malloc(sizeof(unsigned int));
        if (h_time == nullptr) {
            printf("[RO_LAT.H]: malloc h_time Error\n");
            *error = 1;
            break;
        }

        // Allocate Memory on GPU
        error_id = hipMalloc((void **) &d_a, sizeof(unsigned int) * (N));
        if (error_id != hipSuccess) {
            printf("[RO_LAT.H]: hipMalloc d_a Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = hipMalloc((void **) &d_time, sizeof(unsigned int));
        if (error_id != hipSuccess) {
            printf("[RO_LAT.H]: hipMalloc d_time Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        // Initialize p-chase array
        for (int i = 0; i < N; i++) {
            //original:
            h_a[i] = (i + stride) % N;
        }

        // Copy array from Host to GPU
        error_id = hipMemcpy(d_a, h_a, sizeof(unsigned int) * N, hipMemcpyHostToDevice);
        if (error_id != hipSuccess) {
            printf("[RO_LAT.H]: hipMemcpy d_a Error: %s\n", hipGetErrorString(error_id));
            *error = 3;
            break;
        }

        error_id = hipDeviceSynchronize();

        // Launch Kernel function with clock function
        dim3 Db = dim3(1);
        dim3 Dg = dim3(1, 1, 1);
        struct dataForTest d1{};
        d1.type_of_clock = 0;
        d1.d_time = d_time;
        hipLaunchKernelGGL(ro_lat_test, Dg, Db, 0, 0, d_a, N, d1);

        error_id = hipDeviceSynchronize();

        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[RO_LAT.H]: Kernel launch/execution with clock Error:%s\n", hipGetErrorString(error_id));
            *error = 5;
            break;
        }
        error_id = hipDeviceSynchronize();

        // Copy results from GPU to Host
        error_id = hipMemcpy((void *) h_time, (void *) d_time, sizeof(unsigned int), hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[RO_LAT.H]: hipMemcpy d_time Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }
        error_id = hipDeviceSynchronize();

        unsigned int lat = h_time[0];
#ifdef IsDebug
        fprintf(out, "Measured Read-Only avg latencyCycles is %d cycles (add-lat %d and %d)\n", lat, h_time[1], h_time[2]);
#endif //IsDebug
        result.latencyCycles = lat;

        error_id = hipDeviceSynchronize();

        // Launch Kernel function with globaltimer
        d1.type_of_clock = 1;
        hipLaunchKernelGGL(ro_lat_test, Dg, Db, 0, 0, d_a, N, d1);

        error_id = hipDeviceSynchronize();

        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[RO_LAT.H]: Kernel launch/execution with globaltimer Error: %s\n", hipGetErrorString(error_id));
            *error = 5;
            break;
        }
        error_id = hipDeviceSynchronize();

        // Copy results from GPU to Host
        error_id = hipMemcpy((void *) h_time, (void *) d_time, sizeof(unsigned int), hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[RO_LAT.H]: hipMemcpy d_time Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }
        error_id = hipDeviceSynchronize();

        lat = h_time[0];
#ifdef IsDebug
        fprintf(out, "Measured Read-Only avg latencyCycles is %d nanoseconds\n", lat);
#endif //IsDebug
        result.latencyNano = lat;
    } while (false);

    // Free Memory on GPU
    FreeTestMemory({d_a, d_time}, true);

    // Free Memory on Host
    FreeTestMemory({h_a, h_time}, false);

    error_id = hipDeviceReset();
    return result;
}

__global__ void ro_lat_test(const unsigned int* __restrict__ my_array, int array_length, struct dataForTest d1) {
    int iter = 10000;
    // unsigned int *time,
    //                            bool isGlobalTimer

    unsigned long long start_time, end_time;
    unsigned int j = 0;

    // First round
    for (int k = 0; k < array_length; k++) {
        j = __ldg(&my_array[j]);
    }

    // Second round
    if (d1.type_of_clock == 1) {
        GLOBAL_CLOCK(start_time);
        for (int k = 0; k < iter; k++) {
            j = __ldg(&my_array[j]);
        }
        s_index[0] = j;
        GLOBAL_CLOCK(end_time);
    } else {
        LOCAL_CLOCK(start_time);
        for (int k = 0; k < iter; k++) {
            j = __ldg(&my_array[j]);
        }
        s_index[0] = j;
        LOCAL_CLOCK(end_time);
    }

    unsigned int diff = (unsigned int) (end_time - start_time);

    d1.d_time[0] = diff / iter;
}
#endif //CUDATEST_RO_LAT

