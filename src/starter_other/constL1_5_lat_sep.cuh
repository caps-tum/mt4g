
#ifndef CUDATEST_CONSTL1_5_LAT
#define CUDATEST_CONSTL1_5_LAT
//# define isDebug
# include <cstdio>

# include "cuda.h"
# include "../eval.h"
# include "../GPU_resources.cuh"

__global__ void constL1_5_lat(unsigned int * time);
__global__ void constL1_5_lat_globaltimer(unsigned int * time);

LatencyTuple launchConstL1_5LatKernelBenchmark(int* error);

LatencyTuple measure_ConstL1_5_Lat() {
    int error = 0;
    LatencyTuple lat = launchConstL1_5LatKernelBenchmark(&error);
    if (error != 0) {
        printErrorCodeInformation(error);
        exit(error);
    }
    return lat;
}

LatencyTuple launchConstL1_5LatKernelBenchmark(int* error) {
    LatencyTuple result;
    cudaError_t error_id;
#ifdef IsDebug
    FILE* c15Out = fopen("c15Out.log", "w");
#endif //IsDebug

    unsigned int* h_time = nullptr, *d_time = nullptr;

    do {

        // Allocate memory on host
        h_time = (unsigned int *) malloc(sizeof(unsigned int));
        if (h_time == nullptr) {
            printf("[CONSTL1_5_LAT_SEP.CUH]: malloc h_time Error\n");
            *error = 1;
            break;
        }

        // Allocate memory on GPU
        error_id = cudaMalloc((void **) &d_time, sizeof(unsigned int));
        if (error_id != cudaSuccess) {
            printf("[CONSTL1_5_LAT_SEP.CUH]: cudaMalloc d_time Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        cudaDeviceSynchronize();

        // Launch kernel function using clock function
        dim3 Db = dim3(1);
        dim3 Dg = dim3(1, 1, 1);
        constL1_5_lat<<<Dg, Db>>>(d_time);

        cudaDeviceSynchronize();
        error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
            printf("[CONSTL1_5_LAT_SEP.CUH]: Kernel launch/execution with clock Error: %s\n", cudaGetErrorString(error_id));
            *error = 5;
            break;
        }
        cudaDeviceSynchronize();

        // Copy results from GPU to Host
        error_id = cudaMemcpy((void *) h_time, (void *) d_time, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CONSTL1_5_LAT_SEP.CUH]: cudaMemcpy d_time Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }
        cudaDeviceSynchronize();


        unsigned int lat = h_time[0];
#ifdef IsDebug
        fprintf(c15Out, "Measured Const L1.5 avg latencyCycles is %d cycles\n", lat);
#endif //IsDebug
        result.latencyCycles = lat;

        // Launch kernel function using globaltimer
        constL1_5_lat_globaltimer<<<Dg, Db>>>(d_time);
        cudaDeviceSynchronize();

        error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
            printf("[CONSTL1_5_LAT_SEP.CUH]: Kernel launch/execution with globaltimer Error: %s\n", cudaGetErrorString(error_id));
            *error = 5;
            break;
        }
        cudaDeviceSynchronize();

        // Copy results from GPU to Host
        error_id = cudaMemcpy((void *) h_time, (void *) d_time, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CONSTL1_5_LAT_SEP.CUH]: cudaMemcpy d_time Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }
        cudaDeviceSynchronize();

        lat = h_time[0];
#ifdef IsDebug
        fprintf(c15Out, "Measured Const L1.5 avg latencyCycles is %d nanoseconds\n", lat);
#endif //IsDebug
        result.latencyNano = lat;
        cudaDeviceSynchronize();
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
#ifdef IsDebug
    fclose(c15Out);
#endif //IsDebug
    return result;
}

__global__ void constL1_5_lat_globaltimer (unsigned int *time) {
    unsigned long long start_time, end_time;
    unsigned int j = 0;

     // First round
    for (int k = 0; k < constArrSize; k++) {
        s_index[0] += arr[k];
    }

    // Second round
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start_time));
    for (int k = 0; k < MEASURE_SIZE; k++) {
        j = arr[j];
    }
    s_index[1] = j;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(end_time));

    unsigned int diff = (unsigned int) (end_time - start_time);

    time[0] = diff / MEASURE_SIZE;
}

__global__ void constL1_5_lat (unsigned int *time) {
    unsigned int start_time, end_time;
    unsigned int j = 0;

    // First round
    for (int k = 0; k < constArrSize; k++) {
        s_index[0] += arr[k];
    }

    // Second round
    start_time = clock();
    for (int k = 0; k < MEASURE_SIZE; k++) {
        j = arr[j];
    }
    s_index[1] = j;
    end_time = clock();

    unsigned int diff = end_time - start_time;

    time[0] = diff / MEASURE_SIZE;
}

#endif //CUDATEST_CONSTL1_5_LAT

