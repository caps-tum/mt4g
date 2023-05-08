
#ifndef CUDATEST_SHARED
#define CUDATEST_SHARED

# include <cstdio>

# include "cuda.h"
# include "eval.h"
# include "GPU_resources.cuh"

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


bool launchSharedKernelBenchmark(double *avgOut, unsigned int* potMissesOut, unsigned int** time, int* error) {
    cudaError_t error_id;

    unsigned int* h_index = nullptr, *h_timeinfo = nullptr, *duration = nullptr, *d_index = nullptr;
    bool* disturb = nullptr, *d_disturb = nullptr;

    do {
        // Allocate Memory on Host
        h_index = (unsigned int *) malloc(sizeof(unsigned int) * LESS_SIZE);
        if (h_index == nullptr) {
            printf("[SHAREDMEMTEST.CUH]: malloc h_index Error\n");
            *error = 1;
            break;
        }

        h_timeinfo = (unsigned int *) malloc(sizeof(unsigned int) * LESS_SIZE);
        if (h_timeinfo == nullptr) {
            printf("[SHAREDMEMTEST.CUH]: malloc h_timeinfo Error\n");
            *error = 1;
            break;
        }

        disturb = (bool *) malloc(sizeof(bool));
        if (disturb == nullptr) {
            printf("[SHAREDMEMTEST.CUH]: malloc disturb Error\n");
            *error = 1;
            break;
        }

        // Allocate Memory on GPU
        error_id = cudaMalloc((void **) &duration, sizeof(unsigned int) * LESS_SIZE);
        if (error_id != cudaSuccess) {
            printf("[SHAREDMEMTEST.CUH]: cudaMalloc duration Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_index, sizeof(unsigned int) * LESS_SIZE);
        if (error_id != cudaSuccess) {
            printf("[SHAREDMEMTEST.CUH]: cudaMalloc d_index Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_disturb, sizeof(bool));
        if (error_id != cudaSuccess) {
            printf("[SHAREDMEMTEST.CUH]: cudaMalloc d_disturb Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        cudaDeviceSynchronize();

        // Launch Kernel function
        dim3 Db = dim3(1);
        dim3 Dg = dim3(1, 1, 1);
        shared_test <<<Dg, Db>>>(duration, d_index, d_disturb);

        cudaDeviceSynchronize();

        error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
            printf("[SHAREDMEMTEST.CUH]: Kernel launch/execution Error: %s\n", cudaGetErrorString(error_id));
            *error = 5;
            break;
        }
        cudaDeviceSynchronize();

        // Copy results from GPU to Host
        error_id = cudaMemcpy((void *) h_timeinfo, (void *) duration, sizeof(unsigned int) * LESS_SIZE,cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[SHAREDMEMTEST.CUH]: cudaMemcpy duration Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) h_index, (void *) d_index, sizeof(unsigned int) * LESS_SIZE,cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[SHAREDMEMTEST.CUH]: cudaMemcpy d_index Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) disturb, (void *) d_disturb, sizeof(bool), cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[SHAREDMEMTEST.CUH]: cudaMemcpy d_disturb Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }
        cudaDeviceSynchronize();

        createOutputFile(sharedTestSize, LESS_SIZE, h_index, h_timeinfo, avgOut, potMissesOut, "Shared_");
    } while(false);

    // Free Memory on GPU
    if (d_index != nullptr) {
        cudaFree(d_index);
    }

    if (duration != nullptr) {
        cudaFree(duration);
    }

    if (d_disturb != nullptr) {
        cudaFree(d_disturb);
    }

    // Free Memory on Host
    bool ret = false;
    if (disturb != nullptr) {
        ret = *disturb;
        free(disturb);
    }

    if (h_index != nullptr) {
        free(h_index);
    }

    if (h_timeinfo != nullptr) {
        if (time != nullptr) {
            time[0] = h_timeinfo;
        } else {
            free(h_timeinfo);
        }
    }

    cudaDeviceReset();
    return ret;
}

__global__ void shared_test (unsigned int * duration, unsigned int *index, bool* isDisturbed) {

    unsigned int start_time, end_time;
    __shared__ unsigned int s_array[sharedTestSize];

    // Creation of array needs to be done on kernel
    for (int i = 0; i < sharedTestSize; i++) {
        s_array[i] = (i + 1) % sharedTestSize;
    }

    __shared__ long long shared_tvalue[LESS_SIZE];
    __shared__ unsigned int shared_index[LESS_SIZE];

    bool dist = false;
    unsigned int j = 0;

    for(int k=0; k<LESS_SIZE; k++){
        shared_index[k] = 0;
        shared_tvalue[k] = 0;
    }

    // No first round required
    for (int k = 0; k < LESS_SIZE; k++) {
        start_time = clock();
        j = s_array[j];
        shared_index[k] = j;
        end_time = clock();
        shared_tvalue[k] = end_time-start_time;
    }

    for(int k=0; k<LESS_SIZE; k++){
        if (shared_tvalue[k] > 1200) {
            dist = true;
        }
        index[k]= shared_index[k];
        duration[k] = shared_tvalue[k];
    }
    *isDisturbed = dist;
}

#endif //CUDATEST_SHARED

