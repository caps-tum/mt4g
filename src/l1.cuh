
#ifndef CUDATEST_L1
#define CUDATEST_L1

# include <cstdio>

# include "binarySearch.h"
# include "cuda.h"
# include "eval.h"
# include "GPU_resources.cuh"

__global__ void l1_size (unsigned int * my_array, int array_length, unsigned int * duration, unsigned int *index, bool* isDisturbed);

bool launchL1KernelBenchmark(int N, int stride, double *avgOut, unsigned int* potMissesOut, unsigned int** time, int* error);

CacheSizeResult measure_L1() {
// 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
    int absoluteLowerBoundary = 1024;
    int absoluteUpperBoundary = 1024 << 10; // 1024 * 1024
    int widenBounds = 8;

    //Start with 1K integers until 1000K integers
    int bounds[2] = {absoluteLowerBoundary, absoluteUpperBoundary};
    getBoundaries(launchL1KernelBenchmark, bounds, 5);
#ifdef IsDebug
    fprintf(out, "Got Boundaries: %d...%d\n", bounds[0], bounds[1]);
#endif //IsDebug
    printf("Got Boundaries: %d...%d\n", bounds[0], bounds[1]);

    int cp = -1;
    int begin = bounds[0] - widenBounds;
    int end = bounds[1] + widenBounds;
    int stride = 1;
    int arrayIncrease = 1;

    while (cp == -1 && begin >= absoluteLowerBoundary / sizeof(int) - widenBounds && end <= absoluteUpperBoundary / sizeof(int) + widenBounds) {
        cp = wrapBenchmarkLaunch(launchL1KernelBenchmark, begin, end, stride, arrayIncrease, "L1");

        if (cp == -1) {
            begin = begin - (end - begin);
            end = end + (end - begin);
#ifdef IsDebug
            fprintf(out, "\nGot Boundaries: %d...%d\n", begin, end);
#endif //IsDebug
            printf("\nGot Boundaries: %d...%d\n", begin, end);
        }
    }

    CacheSizeResult result;
    int cacheSizeInInt = (begin + cp * arrayIncrease);
    result.CacheSize = (cacheSizeInInt << 2); // * 4);
    result.realCP = cp > 0;
    result.maxSizeBenchmarked = end << 2; // * 4;
    return result;
}


bool launchL1KernelBenchmark(int N, int stride, double *avgOut, unsigned int* potMissesOut, unsigned int** time, int* error) {
    //cudaDeviceReset();
    cudaError_t error_id;

    unsigned int *h_a = nullptr, *h_index = nullptr, *h_timeinfo = nullptr,
    *d_a = nullptr, *duration = nullptr, *d_index = nullptr;
    bool *disturb = nullptr, *d_disturb = nullptr;

    do {
        // Allocate Memory on Host
        h_a = (unsigned int *) malloc(sizeof(unsigned int) * (N));
        if (h_a == nullptr) {
            printf("[L1.CUH]: malloc h_a Error\n");
            *error = 1;
            break;
        }

        h_index = (unsigned int *) malloc(sizeof(unsigned int) * MEASURE_SIZE);
        if (h_index == nullptr) {
            printf("[L1.CUH]: malloc h_index Error\n");
            *error = 1;
            break;
        }

        h_timeinfo = (unsigned int *) malloc(sizeof(unsigned int) * MEASURE_SIZE);
        if (h_timeinfo == nullptr) {
            printf("[L1.CUH]: malloc h_timeinfo Error\n");
            *error = 1;
            break;
        }

        disturb = (bool *) malloc(sizeof(bool));
        if (disturb == nullptr) {
            printf("[L1.CUH]: malloc disturb Error\n");
            *error = 1;
            break;
        }

        // Allocate Memory on GPU
        error_id = cudaMalloc((void **) &d_a, sizeof(unsigned int) * (N));
        if (error_id != cudaSuccess) {
            printf("[L1.CUH]: cudaMalloc d_a Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &duration, sizeof(unsigned int) * MEASURE_SIZE);
        if (error_id != cudaSuccess) {
            printf("[L1.CUH]: cudaMalloc duration Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_index, sizeof(unsigned int) * MEASURE_SIZE);
        if (error_id != cudaSuccess) {
            printf("[L1.CUH]: cudaMalloc d_index Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_disturb, sizeof(bool));
        if (error_id != cudaSuccess) {
            printf("[L1.CUH]: cudaMalloc disturb Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        // Initialize p-chase array
        for (int i = 0; i < N; i++) {
            h_a[i] = (i + stride) % N;
        }

        // Copy array from Host to GPU
        error_id = cudaMemcpy(d_a, h_a, sizeof(unsigned int) * N, cudaMemcpyHostToDevice);
        if (error_id != cudaSuccess) {
            printf("[L1.CUH]: cudaMemcpy d_a Error: %s\n", cudaGetErrorString(error_id));
            *error = 3;
            break;
        }
        cudaDeviceSynchronize();

        // Launch Kernel function
        dim3 Db = dim3(1);
        dim3 Dg = dim3(1, 1, 1);
        l1_size <<<Dg, Db>>>(d_a, N, duration, d_index, d_disturb);

        cudaDeviceSynchronize();

        error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
            printf("[L1.CUH]: Kernel launch/execution Error: %s\n", cudaGetErrorString(error_id));
            *error = 5;
            break;
        }
        cudaDeviceSynchronize();

        // Copy results from GPU to Host
        error_id = cudaMemcpy((void *) h_timeinfo, (void *) duration, sizeof(unsigned int) * MEASURE_SIZE,cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[L1.CUH]: cudaMemcpy duration Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) h_index, (void *) d_index, sizeof(unsigned int) * MEASURE_SIZE,cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[L1.CUH]: cudaMemcpy d_index Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) disturb, (void *) d_disturb, sizeof(bool), cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[L1.CUH]: cudaMemcpy disturb Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        cudaDeviceSynchronize();

        if (!*disturb)
            createOutputFile(N, MEASURE_SIZE, h_index, h_timeinfo, avgOut, potMissesOut, "L1_");

    } while(false);

    // Free Memory on GPU
    if (d_a != nullptr) {
        cudaFree(d_a);
    }

    if (d_index != nullptr) {
        cudaFree(d_index);
    }

    if (duration != nullptr) {
        cudaFree(duration);
    }

    if (d_disturb != nullptr) {
        cudaFree(d_disturb);
    }

    bool ret = false;
    if (disturb != nullptr) {
        ret = *disturb;
        free(disturb);
    }

    // Free Memory on Host
    if (h_a != nullptr) {
        free(h_a);
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

__global__ void l1_size (unsigned int * my_array, int array_length, unsigned int * duration, unsigned int *index, bool* isDisturbed) {

    unsigned int start_time, end_time;
    bool dist = false;
    unsigned int j = 0;

    for(int k=0; k<MEASURE_SIZE; k++){
        s_index[k] = 0;
        s_tvalue[k] = 0;
    }

    // First round
    unsigned int* ptr;
	for (int k = 0; k < array_length; k++) {
        ptr = my_array + j;
        asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(j) : "l"(ptr) : "memory");
	    //j = my_array[j];
	}

    // Second round
    asm volatile(" .reg .u64 smem_ptr64;\n\t"
                 " cvta.to.shared.u64 smem_ptr64, %0;\n\t" :: "l"(s_index));
    for (int k = 0; k < MEASURE_SIZE; k++) {
        ptr = my_array + j;
        //start_time = clock();
        asm volatile ("mov.u32 %0, %%clock;\n\t"
                      "ld.global.ca.u32 %1, [%3];\n\t"
                      "st.shared.u32 [smem_ptr64], %1;"
                      "mov.u32 %2, %%clock;\n\t"
                      "add.u64 smem_ptr64, smem_ptr64, 4;" : "=r"(start_time), "=r"(j), "=r"(end_time) : "l"(ptr) : "memory");
            //start_time = clock();
            //j = my_array[j];
            //s_index[k] = j;
            //end_time = clock();
            s_tvalue[k] = end_time-start_time;
    }

    for(int k=0; k<MEASURE_SIZE; k++){
        if (s_tvalue[k] > 2000) {
            dist = true;
        }
        index[k]= s_index[k];
        duration[k] = s_tvalue[k];
    }
    *isDisturbed = dist;
}

#endif //CUDATEST_L1
