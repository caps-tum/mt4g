
#ifndef CUDATEST_SHAREDMEM_LAT
#define CUDATEST_SHAREDMEM_LAT

# include <cstdio>

# include "cuda.h"
# include "eval.h"
# include "GPU_resources.cuh"

#define sharedTestSize 400

__global__ void shared_lat (unsigned int* time);
__global__ void shared_lat_globaltimer (unsigned int* time);

LatencyTuple launchSharedLatBenchmark(int* error);

LatencyTuple measure_shared_Lat() {
    int error = 0;
    LatencyTuple lat = launchSharedLatBenchmark(&error);
    if (error != 0) {
        printErrorCodeInformation(error);
        exit(error);
    }
    return lat;
}


LatencyTuple launchSharedLatBenchmark(int* error) {
    LatencyTuple result;
    cudaError_t error_id;
    unsigned int *h_time = nullptr, *d_time = nullptr;

    do {
        // Allocate Memory on Host
        h_time = (unsigned int *) malloc(sizeof(unsigned int));
        if (h_time == nullptr) {
            printf("[SHARED_MEM_LAT.CUH]: malloc h_time Error\n");
            *error = 1;
            break;
        }

        // Allocate Memory on GPU
        error_id = cudaMalloc((void **) &d_time, sizeof(unsigned int));
        if (error_id != cudaSuccess) {
            printf("[SHARED_MEM_LAT.CUH]: cudaMalloc d_time Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        cudaDeviceSynchronize();

        // Launch Kernel function with clock function
        dim3 Db = dim3(1);
        dim3 Dg = dim3(1, 1, 1);
        shared_lat <<<Dg, Db>>>(d_time);

        cudaDeviceSynchronize();

        error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
            printf("[SHARED_MEM_LAT.CUH]: Kernel launch/execution with clock function Error: %s\n", cudaGetErrorString(error_id));
            *error = 5;
            break;
        }
        cudaDeviceSynchronize();

        // Copy results from GPU to Host
        error_id = cudaMemcpy((void *) h_time, (void *) d_time, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[SHARED_MEM_LAT.CUH]: cudaMemcpy d_time Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }
        cudaDeviceSynchronize();

        unsigned int lat = h_time[0];
#ifdef IsDebug
        fprintf(out, "Measured Shared avg latencyCycles is %d cycles\n", lat);
#endif //IsDebug
        result.latencyCycles = lat;

        // Execute Kernel function with globaltimer
        shared_lat_globaltimer<<<Dg, Db>>>(d_time);
        cudaDeviceSynchronize();

        error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
            printf("[SHARED_MEM_LAT.CUH]: Kernel launch/execution with globaltimer Error: %s\n", cudaGetErrorString(error_id));
            *error = 5;
            break;
        }
        cudaDeviceSynchronize();

        // Copy results from GPU to Host
        error_id = cudaMemcpy((void *) h_time, (void *) d_time, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[SHARED_MEM_LAT.CUH]: cudaMemcpy d_time Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }
        cudaDeviceSynchronize();

        lat = h_time[0];
#ifdef IsDebug
        fprintf(out, "Measured Shared avg latencyCycles is %d nanoseconds\n", lat);
#endif //IsDebug
        result.latencyNano = lat;
    } while(false);

    // Free Memory on GPU
    if (d_time != nullptr) {
        cudaFree(d_time);
    }

    // Free Memory on Host
    if (h_time != nullptr) {
        free(h_time);
    }

    cudaDeviceReset();
    return result;
}

__global__ void shared_lat_globaltimer (unsigned int * time) {
    int iter = 10000;

    unsigned long long start_time, end_time;
    __shared__ unsigned int s_array[sharedTestSize];

    // Creation of array needs to be done on kernel
    for (int i = 0; i < sharedTestSize; i++) {
        s_array[i] = (i+1) % sharedTestSize;
    }

    unsigned int j = 0;

    // No first round required
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start_time));
    for (int k = 0; k < iter; k++) {
        j = s_array[j];
    }
    s_index[0] = j;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(end_time));

    unsigned int diff = (unsigned int) (end_time - start_time);

    time[0] = diff / iter;
}

__global__ void shared_lat (unsigned int * time) {
    int iter = 10000;

    unsigned int start_time, end_time;
    __shared__ unsigned int s_array[sharedTestSize];

    // Creation of array needs to be done on kernel
    for (int i = 0; i < sharedTestSize; i++) {
        s_array[i] = (i+1) % sharedTestSize;
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

