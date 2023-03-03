
#ifndef CUDATEST_L2LATTEST
#define CUDATEST_L2LATTEST

# include <cstdio>

# include "cuda.h"
# include "eval.h"
# include "GPU_resources.cuh"

__global__ void l2_lat_test (unsigned int * my_array, int array_length, unsigned int * duration, unsigned int* index, bool* isDisturbed);

bool launchL2LatTestKernelBenchmark(int N, int stride, double *avgOut, unsigned int* potMissesOut, unsigned int** time, int* error);


#define FreeMeasureL2LatTest()  \
free(avg);                      \
free(misses);                   \
free(time);                     \

void measure_L2LatTest() {
    double *avg = (double*) malloc(sizeof(double));
    unsigned int* misses = (unsigned int*) malloc(sizeof(unsigned int));
    unsigned int** time = (unsigned int**) malloc(sizeof(unsigned int*));
    if (avg == nullptr || misses == nullptr || time == nullptr) {
        FreeMeasureL2LatTest()
        printErrorCodeInformation(1);
        exit(1);
    }

    int stride = 1;
    int error = 0;
    bool dist = true;
    int count = 5;

    while (dist && count > 0) {
        dist = launchL2LatTestKernelBenchmark(200, stride, avg, misses, time, &error);
        --count;
    }

    free(time[0]);
    FreeMeasureL2LatTest()

    if (error != 0) {
        printErrorCodeInformation(error);
        exit(error);
    }
}


bool launchL2LatTestKernelBenchmark(int N, int stride, double *avgOut, unsigned int* potMissesOut, unsigned int** time, int *error) {
    cudaError_t error_id;

    unsigned int *h_a = nullptr, *h_index = nullptr, *h_timeinfo = nullptr,
    *d_a = nullptr, *d_index = nullptr, *duration = nullptr;
    bool *disturb = nullptr, *d_disturb = nullptr;

    do {
        // Allocate Memory on Host
        h_a = (unsigned int *) malloc(sizeof(unsigned int) * (N));
        if (h_a == nullptr) {
            printf("[L2LATTEST.CUH]: malloc h_a Error\n");
            *error = 1;
            break;
        }

        h_index = (unsigned int *) malloc(sizeof(unsigned int) * MEASURE_SIZE);
        if (h_index == nullptr) {
            printf("[L2LATTEST.CUH]: malloc h_index Error\n");
            *error = 1;
            break;
        }

        h_timeinfo = (unsigned int *) malloc(sizeof(unsigned int) * MEASURE_SIZE);
        if (h_timeinfo == nullptr) {
            printf("[L2LATTEST.CUH]: malloc h_timeinfo Error\n");
            *error = 1;
            break;
        }

        disturb = (bool *) malloc(sizeof(bool));
        if (disturb == nullptr) {
            printf("[L2LATTEST.CUH]: malloc disturb Error\n");
            *error = 1;
            break;
        }

        // Allocate Memory on GPU
        error_id = cudaMalloc((void **) &d_a, sizeof(unsigned int) * (N));
        if (error_id != cudaSuccess) {
            printf("[L2LATTEST.CUH]: cudaMalloc d_a Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_index, sizeof(unsigned int) * MEASURE_SIZE);
        if (error_id != cudaSuccess) {
            printf("[L2LATTEST.CUH]: cudaMalloc d_index Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &duration, sizeof(unsigned int) * MEASURE_SIZE);
        if (error_id != cudaSuccess) {
            printf("[L2LATTEST.CUH]: cudaMalloc duration Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_disturb, sizeof(bool));
        if (error_id != cudaSuccess) {
            printf("[L2LATTEST.CUH]: cudaMalloc d_disturb Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        // Initialize p-chase array
        for (int i = 0; i < N; i++) {
            //original:
            h_a[i] = (i + stride) % N;
        }

        // Copy results from Host to GPU
        error_id = cudaMemcpy(d_a, h_a, sizeof(unsigned int) * N, cudaMemcpyHostToDevice);
        if (error_id != cudaSuccess) {
            printf("[L2LATTEST.CUH]: cudaMemcpy d_a Error: %s\n", cudaGetErrorString(error_id));
            *error = 3;
            break;
        }
        cudaDeviceSynchronize();

        // Launch Kernel function with clock function
        dim3 Db = dim3(1);
        dim3 Dg = dim3(1, 1, 1);
        l2_lat_test <<<Dg, Db>>>(d_a, N, duration, d_index, d_disturb);

        cudaDeviceSynchronize();

        error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
            printf("[L2LATTEST.CUH]: Kernel launch/execution with clock Error: %s\n", cudaGetErrorString(error_id));
            *error = 5;
            break;
        }
        cudaDeviceSynchronize();

        // Copy results from GPU to Host
        error_id = cudaMemcpy((void *) h_timeinfo, (void *) duration, sizeof(unsigned int) * MEASURE_SIZE,cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[L2LATTEST.CUH]: cudaMemcpy duration Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) h_index, (void *) d_index, sizeof(unsigned int) * MEASURE_SIZE,cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[L2LATTEST.CUH]: cudaMemcpy d_index Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) disturb, (void *) d_disturb, sizeof(bool), cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[L2LATTEST.CUH]: cudaMemcpy d_disturb Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }
        cudaDeviceSynchronize();

        createOutputFile(N, MEASURE_SIZE, h_index, h_timeinfo, avgOut, potMissesOut, "L2Lat_");

    } while (false);


    // Free Memory on GPU
    if (d_a != nullptr) {
        cudaFree(d_a);
    }

    if (duration != nullptr) {
        cudaFree(duration);
    }

    if (d_index != nullptr) {
        cudaFree(d_index);
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

    if (h_a != nullptr) {
        free(h_a);
    }

    if (h_timeinfo != nullptr) {
        if (time != nullptr) {
            time[0] = h_timeinfo;
        } else {
            free(h_timeinfo);
        }
    }

    if (h_index != nullptr) {
        free(h_index);
    }

    cudaDeviceReset();
    return ret;
}

__global__ void l2_lat_test (unsigned int * my_array, int array_length, unsigned int * duration, unsigned int* index, bool* isDisturbed) {
    unsigned int start_time, end_time;
    bool dist = false;
    unsigned int j = 0;

    for(int k=0; k<MEASURE_SIZE; k++){
        s_index[k] = 0;
        s_tvalue[k] = 0;
    }

    // First round
	for (int k = 0; k < array_length; k++) {
        j = my_array[j];
    }

    // Second round
    for (int k = 0; k < MEASURE_SIZE; k++) {
        start_time = clock();
        asm volatile(
            "ld.global.cg.u32 %0, [%1];\n\t" : "=r"(j) : "l"(my_array+j) : "memory"
        );
        s_index[k] = j;
        end_time = clock();
        s_tvalue[k] = end_time - start_time;
    }

    for(int k=0; k<MEASURE_SIZE; k++){
        if (s_tvalue[k] > 1200) {
            dist = true;
        }
        index[k]= s_index[k];
        duration[k] = s_tvalue[k];
    }

    *isDisturbed = dist;
}

#endif //CUDATEST_L2LATTEST

