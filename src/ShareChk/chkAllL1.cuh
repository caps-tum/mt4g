
#ifndef CUDATEST_ALL1
#define CUDATEST_ALL1

# include <cstdio>
# include <cstdint>
# include "cuda.h"

/**
 * See launchBenchmarkTwoCoreTexture
 * @param N
 * @param duration1
 * @param duration2
 * @param index1
 * @param index2
 * @param isDisturbed
 * @param baseCore
 * @param testCore
 */
__global__ void chkTwoCoreL1(unsigned int N, unsigned int* array1, unsigned int* array2, unsigned int * duration1, unsigned int * duration2, unsigned int *index1, unsigned int *index2,
                             bool* isDisturbed, unsigned int baseCore, unsigned int testCore) {
    *isDisturbed = false;

    unsigned int start_time, end_time;
    unsigned int j = 0;
    __shared__ long long s_tvalue1[LESS_SIZE];
    __shared__ unsigned int s_index1[LESS_SIZE];
    __shared__ long long s_tvalue2[LESS_SIZE];
    __shared__ unsigned int s_index2[LESS_SIZE];

    __syncthreads();

    if (threadIdx.x == baseCore) {
        for (int k = 0; k < LESS_SIZE; k++) {
            s_index1[k] = 0;
            s_tvalue1[k] = 0;
        }
    }

    __syncthreads();

    if (threadIdx.x == testCore) {
        for (int k = 0; k < LESS_SIZE; k++) {
            s_index2[k] = 0;
            s_tvalue2[k] = 0;
        }
    }

    unsigned int* ptr;

    __syncthreads();

    if (threadIdx.x == baseCore) {
        for (int k = 0; k < N; k++) {
            ptr = array1 + j;
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(j) : "l"(ptr) : "memory");
        }
    }

    __syncthreads();

    if (threadIdx.x == testCore) {
        for (int k = 0; k < N; k++) {
            ptr = array2 + j;
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(j) : "l"(ptr) : "memory");
        }
    }

    __syncthreads();

    if (threadIdx.x == baseCore) {
        //second round
        for (int k = 0; k < LESS_SIZE; k++) {
            ptr = array1 + j;
            start_time = clock();
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(j) : "l"(ptr) : "memory");
            s_index1[k] = j;
            end_time = clock();
            s_tvalue1[k] = (end_time - start_time);
        }
    }

    __syncthreads();

    if (threadIdx.x == testCore) {
        for (int k = 0; k < LESS_SIZE; k++) {
            ptr = array2 + j;
            start_time = clock();
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(j) : "l"(ptr) : "memory");
            s_index2[k] = j;
            end_time = clock();
            s_tvalue2[k] = (end_time - start_time);
        }
    }

    __syncthreads();

    if (threadIdx.x == baseCore) {
        for (int k = 0; k < LESS_SIZE; k++) {
            index1[k] = s_index1[k];
            duration1[k] = s_tvalue1[k];
            if (duration1[k] > 3000) {
                *isDisturbed = true;
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == testCore){
        for (int k = 0; k < LESS_SIZE; k++) {
            index2[k] = s_index2[k];
            duration2[k] = s_tvalue2[k];
            if (duration2[k] > 3000) {
                *isDisturbed = true;
            }
        }
    }
}

/**
 * launches the two core share texture cache kernel benchmark
 * @param arraySize arraySize of the Array
 * @param avgOut1 pointer for storing the avg value for first thread
 * @param avgOut2 pointer for storing the avg value for second thread
 * @param potMissesOut1 pointer for storing potential misses for first thread
 * @param potMissesOut2 pointer for storing potential misses for second thread
 * @param time1 pointer for storing time for first thread
 * @param time2 pointer for storing time for second thread
 * @param numberOfCores number of Cores per SM
 * @param baseCore the baseCore
 * @param testCore the testCore
 * @return bool value if benchmark was somehow disturbed, e.g on some rare occasions
 * the loading time for only one value is VERY high
 */
bool launchBenchmarkTwoCoreL1(unsigned int arraySize, double *avgOut1, double* avgOut2, unsigned int* potMissesOut1, unsigned int* potMissesOut2, unsigned int **time1, unsigned int **time2, int* error,
                              unsigned int numberOfCores, unsigned int baseCore, unsigned int testCore) {
    cudaDeviceReset();
    cudaError_t error_id;

    unsigned int *h_index1 = nullptr, *h_index2 = nullptr, *h_timeinfo1 = nullptr, *h_timeinfo2 = nullptr, *h_a = nullptr,
    *duration1 = nullptr, *duration2 = nullptr, *d_index1 = nullptr, *d_index2 = nullptr, *d_a1 = nullptr, *d_a2 = nullptr;
    bool* disturb = nullptr, *d_disturb = nullptr;

    do {
        // Allocate Memory on Host
        h_index1 = (unsigned int *) malloc(sizeof(unsigned int) * LESS_SIZE);
        if (h_index1 == nullptr) {
            printf("[CHKALLL1.CUH]: malloc h_index1 Error\n");
            *error = 1;
            break;
        }

        h_index2 = (unsigned int *) malloc(sizeof(unsigned int) * LESS_SIZE);
        if (h_index2 == nullptr) {
            printf("[CHKALLL1.CUH]: malloc h_index2 Error\n");
            *error = 1;
            break;
        }

        h_timeinfo1 = (unsigned int *) malloc(sizeof(unsigned int) * LESS_SIZE);
        if (h_timeinfo1 == nullptr) {
            printf("[CHKALLL1.CUH]: malloc h_timeinfo1 Error\n");
            *error = 1;
            break;
        }

        h_timeinfo2 = (unsigned int *) malloc(sizeof(unsigned int) * LESS_SIZE);
        if (h_timeinfo2 == nullptr) {
            printf("[CHKALLL1.CUH]: malloc h_timeinfo2 Error\n");
            *error = 1;
            break;
        }

        disturb = (bool *) malloc(sizeof(bool));
        if (disturb == nullptr) {
            printf("[CHKALLL1.CUH]: malloc disturb Error\n");
            *error = 1;
            break;
        }

        h_a = (unsigned int *) malloc(sizeof(unsigned int) * (arraySize));
        if (h_a == nullptr) {
            printf("[CHKALLL1.CUH]: malloc h_a Error\n");
            *error = 1;
            break;
        }

        // Allocate Memory on GPU
        error_id = cudaMalloc((void **) &duration1, sizeof(unsigned int) * LESS_SIZE);
        if (error_id != cudaSuccess) {
            printf("[CHKALLL1.CUH]: cudaMalloc duration1 Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &duration2, sizeof(unsigned int) * LESS_SIZE);
        if (error_id != cudaSuccess) {
            printf("[CHKALLL1.CUH]: cudaMalloc duration2 Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_index1, sizeof(unsigned int) * LESS_SIZE);
        if (error_id != cudaSuccess) {
            printf("[CHKALLL1.CUH]: cudaMalloc d_indextxt1 Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_index2, sizeof(unsigned int) * LESS_SIZE);
        if (error_id != cudaSuccess) {
            printf("[CHKALLL1.CUH]: cudaMalloc d_index2 Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_disturb, sizeof(bool));
        if (error_id != cudaSuccess) {
            printf("[CHKALLL1.CUH]: cudaMalloc disturb Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_a1, sizeof(unsigned int) * (arraySize));
        if (error_id != cudaSuccess) {
            printf("[CHKALLL1.CUH]: cudaMalloc d_a1 Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_a2, sizeof(unsigned int) * (arraySize));
        if (error_id != cudaSuccess) {
            printf("[CHKALLL1.CUH]: cudaMalloc d_a2 Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        // Initialize p-chase array
        for (int i = 0; i < arraySize; i++) {
            h_a[i] = (i + 1) % arraySize;
        }

        // Copy array from Host to GPU
        error_id = cudaMemcpy(d_a1, h_a, sizeof(unsigned int) * arraySize, cudaMemcpyHostToDevice);
        if (error_id != cudaSuccess) {
            printf("[CHKALLL1.CUH]: cudaMemcpy d_a1 Error: %s\n", cudaGetErrorString(error_id));
            *error = 3;
            break;
        }

        error_id = cudaMemcpy(d_a2, h_a, sizeof(unsigned int) * arraySize, cudaMemcpyHostToDevice);
        if (error_id != cudaSuccess) {
            printf("[CHKALLL1.CUH]: cudaMemcpy d_a2 Error: %s\n", cudaGetErrorString(error_id));
            *error = 3;
            break;
        }
        cudaDeviceSynchronize();

        error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
            printf("[CHKALLL1.CUH]: Error 2 is %s\n", cudaGetErrorString(error_id));
            *error = 99;
            break;
        }
        cudaDeviceSynchronize();

        // Launch Kernel function
        dim3 Db = dim3(numberOfCores);
        dim3 Dg = dim3(1, 1, 1);
        chkTwoCoreL1<<<Dg, Db>>>(arraySize, d_a1, d_a2, duration1, duration2, d_index1, d_index2, d_disturb, baseCore, testCore);

        cudaDeviceSynchronize();
        error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
            printf("[CHKALLL1.CUH]: Kernel launch/execution Error: %s\n", cudaGetErrorString(error_id));
            *error = 5;
            break;
        }

        // Copy results from GPU to Host
        error_id = cudaMemcpy((void *) h_timeinfo1, (void *) duration1, sizeof(unsigned int) * LESS_SIZE, cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKALLL1.CUH]: cudaMemcpy duration1 Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) h_timeinfo2, (void *) duration2, sizeof(unsigned int) * LESS_SIZE, cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKALLL1.CUH]: cudaMemcpy duration2 Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) h_index1, (void *) d_index1, sizeof(unsigned int) * LESS_SIZE, cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKALLL1.CUH]: cudaMemcpy d_index1 Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) h_index2, (void *) d_index2, sizeof(unsigned int) * LESS_SIZE, cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKALLL1.CUH]: cudaMemcpy d_index2 Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) disturb, (void *) d_disturb, sizeof(bool), cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKALLL1.CUH]: cudaMemcpy disturb Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        char prefix1[64], prefix2[64];
        snprintf(prefix1, 64, "AllL1_T1_%d_%d", baseCore, testCore);
        snprintf(prefix2, 64, "AllL1_T2_%d_%d", baseCore, testCore);

        createOutputFile((int) arraySize, LESS_SIZE, h_index1, h_timeinfo1, avgOut1, potMissesOut1, prefix1);
        createOutputFile((int) arraySize, LESS_SIZE, h_index2, h_timeinfo2, avgOut2, potMissesOut2, prefix2);
    } while(false);

    bool ret = false;
    if (disturb != nullptr) {
        ret = *disturb;
        free(disturb);
    }

    // Free Memory on GPU
    if (d_index1 != nullptr) {
        cudaFree(d_index1);
    }

    if (d_index2 != nullptr) {
        cudaFree(d_index2);
    }

    if (duration1 != nullptr) {
        cudaFree(duration1);
    }

    if (duration2 != nullptr) {
        cudaFree(duration2);
    }

    if (d_a1 != nullptr) {
        cudaFree(d_a1);
    }

    if (d_disturb != nullptr) {
        cudaFree(d_disturb);
    }

    // Free Memory on Host
    if (h_index1 != nullptr) {
        free(h_index1);
    }

    if (h_index2 != nullptr) {
        free(h_index2);
    }

    if (h_a != nullptr) {
        free(h_a);
    }

    if (h_timeinfo1 != nullptr) {
        if (time1 != nullptr) {
            time1[0] = h_timeinfo1;
        } else {
            free(h_timeinfo1);
        }
    }

    if (h_timeinfo2 != nullptr) {
        if (time2 != nullptr) {
            time2[0] = h_timeinfo2;
        } else {
            free(h_timeinfo2);
        }
    }

    cudaDeviceReset();
    return ret;
}


#endif //CUDATEST_ALL1