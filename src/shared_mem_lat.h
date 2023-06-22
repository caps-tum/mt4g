#include "hip/hip_runtime.h"

#ifndef CUDATEST_SHAREDMEM_LAT
#define CUDATEST_SHAREDMEM_LAT

# include <cstdio>

# include "hip/hip_runtime.h"
# include "eval.h"
# include "GPU_resources.h"
#include "general_functions.h"

#define sharedTestSize 400

__global__ void shared_lat(unsigned int *time);

__global__ void shared_lat_globaltimer(unsigned int *time);

LatencyTuple launchSharedLatBenchmark(int *error);

LatencyTuple measure_shared_Lat() {
    int error = 0;
    LatencyTuple lat = launchSharedLatBenchmark(&error);
    if (error != 0) {
        printErrorCodeInformation(error);
        exit(error);
    }
    return lat;
}


#define SHARED_MEM_LAT "shared_mem_lat.H"

LatencyTuple launchSharedLatBenchmark(int *error) {
    LatencyTuple result;
    hipError_t error_id;
    unsigned int *h_time = nullptr, *d_time = nullptr;

    do {
        // Allocate Memory on Host
        h_time = (unsigned int *) mallocAndCheck(SHARED_MEM_LAT, sizeof(unsigned int),
                                                 "h_time", error);

        // Allocate Memory on GPU
        if (hipMallocAndCheck(SHARED_MEM_LAT, (void **) &d_time, sizeof(unsigned int),
                              "d_time", error) != hipSuccess) {
            break;
        }

        error_id = hipDeviceSynchronize();

        // Launch Kernel function with clock function
        dim3 Db = dim3(1);
        dim3 Dg = dim3(1, 1, 1);
        hipLaunchKernelGGL(shared_lat, Dg, Db, 0, 0, d_time);

        error_id = hipDeviceSynchronize();

        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[SHARED_MEM_LAT.H]: Kernel launch/execution with clock function Error: %s\n",
                   hipGetErrorString(error_id));
            *error = 5;
            break;
        }
        error_id = hipDeviceSynchronize();

        // Copy results from GPU to Host
        if (hipMemcpyAndCheck(SHARED_MEM_LAT, h_time, d_time, sizeof(unsigned int),
                              "d_time -> h_time", error, true) != hipSuccess) {
            break;
        }

        error_id = hipDeviceSynchronize();

        unsigned int lat = h_time[0];
#ifdef IsDebug
        fprintf(out, "Measured Shared avg latencyCycles is %d cycles\n", lat);
#endif //IsDebug
        result.latencyCycles = lat;

        // Execute Kernel function with globaltimer
        hipLaunchKernelGGL(shared_lat_globaltimer, Dg, Db, 0, 0, d_time);
        error_id = hipDeviceSynchronize();

        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[SHARED_MEM_LAT.H]: Kernel launch/execution with globaltimer Error: %s\n",
                   hipGetErrorString(error_id));
            *error = 5;
            break;
        }
        error_id = hipDeviceSynchronize();

        // Copy results from GPU to Host
        if (hipMemcpyAndCheck(SHARED_MEM_LAT, h_time, d_time, sizeof(unsigned int),
                              "d_time -> h_time", error, true) != hipSuccess) {
            break;
        }
        error_id = hipDeviceSynchronize();

        lat = h_time[0];
#ifdef IsDebug
        fprintf(out, "Measured Shared avg latencyCycles is %d nanoseconds\n", lat);
#endif //IsDebug
        result.latencyNano = lat;
    } while (false);

    // Free Memory on GPU
    FreeTestMemory({d_time}, true);

    // Free Memory on Host
    FreeTestMemory({h_time}, false);

    error_id = hipDeviceReset();
    return result;
}

__global__ void shared_lat_globaltimer(unsigned int *time) {
    int iter = 10000;

    unsigned long long start_time, end_time;
    __shared__ ALIGN(16) unsigned int s_array[sharedTestSize];

    // Creation of array needs to be done on kernel
    for (int i = 0; i < sharedTestSize; i++) {
        s_array[i] = (i + 1) % sharedTestSize;
    }

    unsigned int j = 0;

    // No first round required
    GLOBAL_CLOCK(start_time);
    for (int k = 0; k < iter; k++) {
        j = s_array[j];
    }
    s_index[0] = j;
    GLOBAL_CLOCK(end_time);

    unsigned int diff = (unsigned int) (end_time - start_time);

    time[0] = diff / iter;
}

__global__ void shared_lat(unsigned int *time) {
    int iter = 10000;

    unsigned int start_time, end_time;
    __shared__ ALIGN(16) unsigned int s_array[sharedTestSize];

    // Creation of array needs to be done on kernel
    for (int i = 0; i < sharedTestSize; i++) {
        s_array[i] = (i + 1) % sharedTestSize;
    }

    unsigned int j = 0;

    // No first round required
    start_time = clock();
    for (int k = 0; k < iter; k++) {
        j = s_array[j];
    }
    s_index[0] = j;
    end_time = clock();

    unsigned int diff = end_time - start_time;

    time[0] = diff / iter;
}

#endif //CUDATEST_SHAREDMEM_LAT

