
#ifndef CUDATEST_RO_LAT
#define CUDATEST_RO_LAT

# include <cstdio>

# include "cuda.h"
# include "eval.h"
# include "GPU_resources.cuh"

__global__ void ro_lat(const unsigned int* __restrict__ my_array, int array_length, unsigned int * time);
__global__ void ro_lat_globaltimer(const unsigned int* __restrict__ my_array, int array_length, unsigned int * time);

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
    cudaError_t error_id;

    unsigned int *h_a = nullptr, *h_time = nullptr, *d_a = nullptr, *d_time = nullptr;

    do {
        // Allocate Memory on Host
        h_a = (unsigned int *) malloc(sizeof(unsigned int) * (N));
        if (h_a == nullptr) {
            printf("[RO_LAT.CUH]: malloc h_a Error\n");
            *error = 1;
            break;
        }

        h_time = (unsigned int *) malloc(sizeof(unsigned int));
        if (h_time == nullptr) {
            printf("[RO_LAT.CUH]: malloc h_time Error\n");
            *error = 1;
            break;
        }

        // Allocate Memory on GPU
        error_id = cudaMalloc((void **) &d_a, sizeof(unsigned int) * (N));
        if (error_id != cudaSuccess) {
            printf("[RO_LAT.CUH]: cudaMalloc d_a Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_time, sizeof(unsigned int));
        if (error_id != cudaSuccess) {
            printf("[RO_LAT.CUH]: cudaMalloc d_time Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        // Initialize p-chase array
        for (int i = 0; i < N; i++) {
            //original:
            h_a[i] = (i + stride) % N;
        }

        // Copy array from Host to GPU
        error_id = cudaMemcpy(d_a, h_a, sizeof(unsigned int) * N, cudaMemcpyHostToDevice);
        if (error_id != cudaSuccess) {
            printf("[RO_LAT.CUH]: cudaMemcpy d_a Error: %s\n", cudaGetErrorString(error_id));
            *error = 3;
            break;
        }

        cudaDeviceSynchronize();

        // Launch Kernel function with clock function
        dim3 Db = dim3(1);
        dim3 Dg = dim3(1, 1, 1);
        ro_lat <<<Dg, Db>>>(d_a, N, d_time);

        cudaDeviceSynchronize();

        error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
            printf("[RO_LAT.CUH]: Kernel launch/execution with clock Error:%s\n", cudaGetErrorString(error_id));
            *error = 5;
            break;
        }
        cudaDeviceSynchronize();

        // Copy results from GPU to Host
        error_id = cudaMemcpy((void *) h_time, (void *) d_time, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[RO_LAT.CUH]: cudaMemcpy d_time Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }
        cudaDeviceSynchronize();

        unsigned int lat = h_time[0];
#ifdef IsDebug
        fprintf(out, "Measured Read-Only avg latencyCycles is %d cycles (add-lat %d and %d)\n", lat, h_time[1], h_time[2]);
#endif //IsDebug
        result.latencyCycles = lat;

        cudaDeviceSynchronize();

        // Launch Kernel function with globaltimer
        ro_lat_globaltimer<<<Dg, Db>>>(d_a, N, d_time);

        cudaDeviceSynchronize();

        error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
            printf("[RO_LAT.CUH]: Kernel launch/execution with globaltimer Error: %s\n", cudaGetErrorString(error_id));
            *error = 5;
            break;
        }
        cudaDeviceSynchronize();

        // Copy results from GPU to Host
        error_id = cudaMemcpy((void *) h_time, (void *) d_time, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[RO_LAT.CUH]: cudaMemcpy d_time Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }
        cudaDeviceSynchronize();

        lat = h_time[0];
#ifdef IsDebug
        fprintf(out, "Measured Read-Only avg latencyCycles is %d nanoseconds\n", lat);
#endif //IsDebug
        result.latencyNano = lat;
    } while (false);

    // Free Memory on GPU
    if (d_a != nullptr) {
        cudaFree(d_a);
    }

    if (d_time != nullptr) {
        cudaFree(d_time);
    }

    // Free Memory on Host
    if (h_a != nullptr) {
        free(h_a);
    }

    if (h_time != nullptr) {
        free(h_time);
    }

    cudaDeviceReset();
    return result;
}

__global__ void ro_lat_globaltimer (const unsigned int* __restrict__ my_array, int array_length, unsigned int *time) {
    int iter = 10000;

    unsigned long long start_time, end_time;
    unsigned int j = 0;

    // First round
    for (int k = 0; k < array_length; k++) {
        j = __ldg(&my_array[j]);
    }

    // Second round
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start_time));
    for (int k = 0; k < iter; k++) {
        j = __ldg(&my_array[j]);
    }
    s_index[0] = j;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(end_time));

    unsigned int diff = (unsigned int) (end_time - start_time);

    time[0] = diff / iter;
}

__global__ void ro_lat (const unsigned int* __restrict__ my_array, int array_length, unsigned int *time) {
    int iter = 10000;

    unsigned int start_time, end_time;
    unsigned int j = 0;

    // First round
	for (int k = 0; k < array_length; k++) {
        j = __ldg(&my_array[j]);
    }

    // Second round
    start_time = clock();
    for (int k = 0; k < iter; k++) {
        j = __ldg(&my_array[j]);
    }
    s_index[0] = j;
    end_time = clock();

    unsigned int diff = end_time - start_time;

    time[0] = diff / iter;
}

#endif //CUDATEST_RO_LAT

