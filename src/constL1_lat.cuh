
#ifndef CUDATEST_CONSTL1_LAT
#define CUDATEST_CONSTL1_LAT

# include <cstdio>

# include "cuda.h"
# include "eval.h"
# include "GPU_resources.cuh"

static __device__ __constant__ unsigned int arrLat[200] = {1,2,3,4,5,6,7,8,9,10,
                                                           11,12,13,14,15,16,17,18,19,20,
                                                           21,22,23,24,25,26,27,28,29,30,
                                                           31,32,33,34,35,36,37,38,39,40,
                                                           41,42,43,44,45,46,47,48,49,50,
                                                           51,52,53,54,55,56,57,58,59,60,
                                                           61,62,63,64,65,66,67,68,69,70,
                                                           71,72,73,74,75,76,77,78,79,80,
                                                           81,82,83,84,85,86,87,88,89,90,
                                                           91,92,93,94,95,96,97,98,99,100,
                                                           101,102,103,104,105,106,107,108,109,110,
                                                           111,112,113,114,115,116,117,118,119,120,
                                                           121,122,123,124,125,126,127,128,129,130,
                                                           131,132,133,134,135,136,137,138,139,140,
                                                           141,142,143,144,145,146,147,148,149,150,
                                                           151,152,153,154,155,156,157,158,159,160,
                                                           161,162,163,164,165,166,167,168,169,170,
                                                           171,172,173,174,175,176,177,178,179,180,
                                                           181,182,183,184,185,186,187,188,189,190,
                                                           191,192,193,194,195,196,197,198,199, 0};

__global__ void constL1_lat(int array_length, unsigned int * time);
__global__ void constL1_lat_globaltimer(int array_length, unsigned int * time);

LatencyTuple launchConstL1LatKernelBenchmark(int N, int* error);

LatencyTuple measure_ConstL1_Lat() {
    int error = 0;
    LatencyTuple lat = launchConstL1LatKernelBenchmark(200, &error);
    if (error != 0) {
        printErrorCodeInformation(error);
        exit(error);
    }
    return lat;
}


LatencyTuple launchConstL1LatKernelBenchmark(int N, int* error) {
    LatencyTuple result;
    cudaError_t error_id;

    unsigned int *h_time = nullptr, *d_time = nullptr;

    do {
        // Allocate Memory on Host
        h_time = (unsigned int *) malloc(sizeof(unsigned int));
        if (h_time == nullptr) {
            printf("[CONSTL1_LAT.CUH]: malloc h_time Error\n");
            *error = 1;
            break;
        }

        // Allocate Memory on GPU
        error_id = cudaMalloc((void **) &d_time, sizeof(unsigned int));
        if (error_id != cudaSuccess) {
            printf("[CONSTL1_LAT.CUH]: cudaMalloc d_time Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }
        cudaDeviceSynchronize();

        // Launch Kernel function with clock function
        dim3 Db = dim3(1);
        dim3 Dg = dim3(1, 1, 1);
        constL1_lat <<<Dg, Db>>>(N, d_time);

        cudaDeviceSynchronize();
        error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
            printf("[CONSTL1_LAT.CUH]: Kernel launch/execution with clock Error: %s\n", cudaGetErrorString(error_id));
            *error = 5;
            break;
        }
        cudaDeviceSynchronize();

        // Copy results from GPU to Host
        error_id = cudaMemcpy((void *) h_time, (void *) d_time, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CONSTL1_LAT.CUH]: cudaMemcpy d_time Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }
        cudaDeviceSynchronize();

        unsigned int lat = h_time[0];
#ifdef IsDebug
        fprintf(out, "Measured Const L1 avg latencyCycles is %d cycles\n", lat);
#endif //IsDebug
        result.latencyCycles = lat;
        cudaDeviceSynchronize();

        // Launch Kernel function with globaltimer
        constL1_lat_globaltimer<<<Dg, Db>>>(N, d_time);

        cudaDeviceSynchronize();
        error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
            printf("[CONSTL1_LAT.CUH]: Kernel launch/execution with globaltimer Error: %s\n", cudaGetErrorString(error_id));
            *error = 5;
            break;
        }
        cudaDeviceSynchronize();

        // Copy results from GPU to Host
        error_id = cudaMemcpy((void *) h_time, (void *) d_time, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CONSTL1_LAT.CUH]: cudaMemcpy d_time Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        cudaDeviceSynchronize();
        lat = h_time[0];
#ifdef IsDebug
        fprintf(out, "Measured Const L1 avg latencyCycles is %d nanoseconds\n", lat);
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

__global__ void constL1_lat_globaltimer (int array_length, unsigned int *time) {
    int iter = 10000;

    unsigned long long start_time, end_time;
    unsigned int j = 0;

    // First round
    for (int k = 0; k < array_length + 1; k++) {
        j = arrLat[j];
    }

    // Second round
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start_time));
    for (int k = 0; k < iter; k++) {
        j = arrLat[j];
    }
    s_index[0] = j;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(end_time));

    unsigned int diff = (unsigned int) (end_time - start_time);

    time[0] = diff / iter;
}

__global__ void constL1_lat (int array_length, unsigned int *time) {
    int iter = 10000;

    unsigned int start_time, end_time;
    unsigned int j = 0;

    // First round
	for (int k = 0; k < array_length+1; k++) {
        j = arrLat[j];
    }

    // Second round
    start_time = clock();
    for (int k = 0; k < iter; k++) {
        j = arrLat[j];
    }
    s_index[0] = j;
    end_time = clock();

    unsigned int diff = end_time - start_time;

    time[0] = diff / iter;
}

#endif //CUDATEST_CONSTL1_LAT

