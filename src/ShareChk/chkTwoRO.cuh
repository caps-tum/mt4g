
#ifndef CUDATEST_TWORO
#define CUDATEST_TWORO

# include <cstdio>
# include <cstdint>

# include "cuda.h"

__global__ void chkTwoRO(unsigned int N, const unsigned int* __restrict__ arrayRO1, const unsigned int* __restrict__ arrayRO2, unsigned int *durationRO1, unsigned int * durationRO2, unsigned int *indexRO1, unsigned int *indexRO2,
                         bool* isDisturbed) {
    *isDisturbed = false;

    unsigned int start_time, end_time;
    unsigned int j = 0;
    __shared__ long long s_tvalueRO1[LESS_SIZE];
    __shared__ unsigned int s_indexRO1[LESS_SIZE];
    __shared__ long long s_tvalueRO2[LESS_SIZE];
    __shared__ unsigned int s_indexRO2[LESS_SIZE];

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < LESS_SIZE; k++) {
            s_indexRO1[k] = 0;
            s_tvalueRO1[k] = 0;
        }
    }

    __syncthreads();

    if (threadIdx.x == 1){
        for (int k = 0; k < LESS_SIZE; k++) {
            s_indexRO2[k] = 0;
            s_tvalueRO2[k] = 0;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < N; k++) {
            j = __ldg(&arrayRO1[j]);
        }
    }

    __syncthreads();

    if (threadIdx.x == 1){
        for (int k = 0; k < N; k++) {
            j = __ldg(&arrayRO2[j]);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        //second round
        for (int k = 0; k < LESS_SIZE; k++) {
            start_time = clock();
            j = __ldg(&arrayRO1[j]);
            s_indexRO1[k] = j;
            end_time = clock();
            s_tvalueRO1[k] = (end_time - start_time);
        }
    }

    __syncthreads();

    if (threadIdx.x == 1){
        for (int k = 0; k < LESS_SIZE; k++) {
            start_time = clock();
            j = __ldg(&arrayRO2[j]);
            s_indexRO2[k] = j;
            end_time = clock();
            s_tvalueRO2[k] = (end_time - start_time);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < LESS_SIZE; k++) {
            indexRO1[k] = s_indexRO1[k];
            durationRO1[k] = s_tvalueRO1[k];
            if (durationRO1[k] > 3000) {
                *isDisturbed = true;
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == 1){
        for (int k = 0; k < LESS_SIZE; k++) {
            indexRO2[k] = s_indexRO2[k];
            durationRO2[k] = s_tvalueRO2[k];
            if (durationRO2[k] > 3000) {
                *isDisturbed = true;
            }
        }
    }
}

bool launchBenchmarkTwoRO(unsigned int N, double *avgOut1, double* avgOut2, unsigned int* potMissesOut1, unsigned int* potMissesOut2, unsigned int **time1, unsigned int **time2, int* error) {
    cudaDeviceReset();
    cudaError_t error_id;

    unsigned int *h_indexReadOnly1 = nullptr, *h_indexReadOnly2 = nullptr, *h_timeinfoReadOnly1 = nullptr, *h_timeinfoReadOnly2 = nullptr, *h_aReadOnly = nullptr,
    *durationRO1 = nullptr, *durationRO2 = nullptr, *d_indexRO1 = nullptr, *d_indexRO2 = nullptr, *d_aReadOnly1 = nullptr, *d_aReadOnly2 = nullptr;
    bool *disturb = nullptr, *d_disturb = nullptr;

    do {
        // Allocate Memory on Host
        h_indexReadOnly1 = (unsigned int *) malloc(sizeof(unsigned int) * LESS_SIZE);
        if (h_indexReadOnly1 == nullptr) {
            printf("[CHKTWORO.CUH]: cudaMalloc h_indexReadOnly1 Error\n");
            *error = 1;
            break;
        }

        h_indexReadOnly2 = (unsigned int *) malloc(sizeof(unsigned int) * LESS_SIZE);
        if (h_indexReadOnly2 == nullptr) {
            printf("[CHKTWORO.CUH]: cudaMalloc h_indexReadOnly2 Error\n");
            *error = 1;
            break;
        }

        h_timeinfoReadOnly1 = (unsigned int *) malloc(sizeof(unsigned int) * LESS_SIZE);
        if (h_timeinfoReadOnly1 == nullptr) {
            printf("[CHKTWORO.CUH]: cudaMalloc h_timeinfoReadOnly1 Error\n");
            *error = 1;
            break;
        }

        h_timeinfoReadOnly2 = (unsigned int *) malloc(sizeof(unsigned int) * LESS_SIZE);
        if (h_timeinfoReadOnly2 == nullptr) {
            printf("[CHKTWORO.CUH]: cudaMalloc h_timeinfoReadOnly2 Error\n");
            *error = 1;
            break;
        }

        disturb = (bool *) malloc(sizeof(bool));
        if (disturb == nullptr) {
            printf("[CHKTWORO.CUH]: cudaMalloc disturb Error\n");
            *error = 1;
            break;
        }

        h_aReadOnly = (unsigned int *) malloc(sizeof(unsigned int) * (N));
        if (h_aReadOnly == nullptr) {
            printf("[CHKTWORO.CUH]: cudaMalloc h_aReadOnly Error\n");
            *error = 1;
            break;
        }

        // Allocate Memory on GPU
        error_id = cudaMalloc((void **) &durationRO1, sizeof(unsigned int) * LESS_SIZE);
        if (error_id != cudaSuccess) {
            printf("[CHKTWORO.CUH]: cudaMalloc durationRO1 Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &durationRO2, sizeof(unsigned int) * LESS_SIZE);
        if (error_id != cudaSuccess) {
            printf("[CHKTWORO.CUH]: cudaMalloc durationRO2 Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_indexRO1, sizeof(unsigned int) * LESS_SIZE);
        if (error_id != cudaSuccess) {
            printf("[CHKTWORO.CUH]: cudaMalloc d_indexRO1 Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_indexRO2, sizeof(unsigned int) * LESS_SIZE);
        if (error_id != cudaSuccess) {
            printf("[CHKTWORO.CUH]: cudaMalloc d_indexRO2 Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_disturb, sizeof(bool));
        if (error_id != cudaSuccess) {
            printf("[CHKTWORO.CUH]: cudaMalloc disturb Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_aReadOnly1, sizeof(unsigned int) * (N));
        if (error_id != cudaSuccess) {
            printf("[CHKTWORO.CUH]: cudaMalloc d_aReadOnly1 Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }
        error_id = cudaMalloc((void **) &d_aReadOnly2, sizeof(unsigned int) * (N));
        if (error_id != cudaSuccess) {
            printf("[CHKTWORO.CUH]: cudaMalloc d_aReadOnly2 Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        // Initialize p-chase array
        for (int i = 0; i < N; i++) {
            h_aReadOnly[i] = (i + 1) % N;
        }

        // Copy array from Host to GPU
        error_id = cudaMemcpy(d_aReadOnly1, h_aReadOnly, sizeof(unsigned int) * N, cudaMemcpyHostToDevice);
        if (error_id != cudaSuccess) {
            printf("[CHKTWORO.CUH]: cudaMemcpy d_aReadOnly1 Error: %s\n", cudaGetErrorString(error_id));
            *error = 3;
            break;
        }

        error_id = cudaMemcpy(d_aReadOnly2, h_aReadOnly, sizeof(unsigned int) * N, cudaMemcpyHostToDevice);
        if (error_id != cudaSuccess) {
            printf("[CHKTWORO.CUH]: cudaMemcpy d_aReadOnly2 Error: %s\n", cudaGetErrorString(error_id));
            *error = 3;
            break;
        }

        cudaDeviceSynchronize();

        error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
            printf("[CHKTWORO.CUH]: cudaDeviceSynchronize Error: %s\n", cudaGetErrorString(error_id));
            *error = 99;
            break;
        }
        cudaDeviceSynchronize();

        // Launch Kernel function
        dim3 Db = dim3(2);
        dim3 Dg = dim3(1, 1, 1);
        chkTwoRO<<<Dg, Db>>>(N, d_aReadOnly1, d_aReadOnly2, durationRO1, durationRO2, d_indexRO1, d_indexRO2, d_disturb);

        cudaDeviceSynchronize();
        error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
            printf("[CHKTWORO.CUH]: Kernel launch/execution Error: %s\n", cudaGetErrorString(error_id));
            *error = 5;
            break;
        }

        // Copy results from GPU to Host
        error_id = cudaMemcpy((void *) h_timeinfoReadOnly1, (void *) durationRO1, sizeof(unsigned int) * LESS_SIZE,cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKTWORO.CUH]: cudaMemcpy durationRO1 Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) h_timeinfoReadOnly2, (void *) durationRO2, sizeof(unsigned int) * LESS_SIZE,cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKTWORO.CUH]: cudaMemcpy durationRO2 Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) h_indexReadOnly1, (void *) d_indexRO1, sizeof(unsigned int) * LESS_SIZE,cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKTWORO.CUH]: cudaMemcpy d_indexRO1 Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) h_indexReadOnly2, (void *) d_indexRO2, sizeof(unsigned int) * LESS_SIZE,cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKTWORO.CUH]: cudaMemcpy d_indexRO2 Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) disturb, (void *) d_disturb, sizeof(bool), cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKTWORO.CUH]: cudaMemcpy disturb Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        createOutputFile((int) N, LESS_SIZE, h_indexReadOnly1, h_timeinfoReadOnly1, avgOut1, potMissesOut1, "TwoRO1_");
        createOutputFile((int) N, LESS_SIZE, h_indexReadOnly2, h_timeinfoReadOnly2, avgOut2, potMissesOut2, "TwoRO2_");
    } while(false);

    bool ret = false;
    if (disturb) {
        ret = *disturb;
        free(disturb);
    }

    // Free Memory on GPU
    if (d_indexRO1 != nullptr) {
        cudaFree(d_indexRO1);
    }

    if (d_indexRO2 != nullptr) {
        cudaFree(d_indexRO2);
    }

    if (durationRO1 != nullptr) {
        cudaFree(durationRO1);
    }

    if (durationRO2 != nullptr) {
        cudaFree(durationRO2);
    }

    if (d_aReadOnly1 != nullptr) {
        cudaFree(d_aReadOnly1);
    }

    if (d_aReadOnly2 != nullptr) {
        cudaFree(d_aReadOnly2);
    }

    if (d_disturb != nullptr) {
        cudaFree(d_disturb);
    }

    // Free Memory on Host
    if (h_indexReadOnly1 != nullptr) {
        free(h_indexReadOnly1);
    }

    if (h_indexReadOnly2 != nullptr) {
        free(h_indexReadOnly2);
    }

    if (h_aReadOnly != nullptr) {
        free(h_aReadOnly);
    }

    if (h_timeinfoReadOnly1 != nullptr) {
        if (time1 != nullptr) {
            time1[0] = h_timeinfoReadOnly1;
        } else {
            free(h_timeinfoReadOnly1);
        }
    }

    if (h_timeinfoReadOnly2 != nullptr) {
        if (time2 != nullptr) {
            time2[0] = h_timeinfoReadOnly2;
        } else {
            free(h_timeinfoReadOnly2);
        }
    }

    cudaDeviceReset();
    return ret;
}

#define FreeMeasureTwoROResOnlyPtr()        \
free(time);                                 \
free(timeRef);                              \
free(avgFlow);                              \
free(potMissesFlow);                        \
free(avgFlowRef);                           \
free(potMissesFlowRef);                     \

#define FreeMeasureTwoROResources()         \
if (time[0] != nullptr) {                   \
    free(time[0]);                          \
}                                           \
if (time[1] != nullptr) {                   \
    free(time[1]);                          \
}                                           \
if (timeRef[0] != nullptr) {                \
    free(timeRef[0]);                       \
}                                           \
free(time);                                 \
free(timeRef);                              \
free(avgFlow);                              \
free(potMissesFlow);                        \
free(avgFlowRef);                           \
free(potMissesFlowRef);                     \

double measure_TwoRO(unsigned int measuredSizeCache, unsigned int sub) {
    unsigned int CacheSizeInInt = (measuredSizeCache - sub) / 4;

    double* avgFlowRef = (double*) malloc(sizeof(double));
    unsigned int *potMissesFlowRef = (unsigned int*) malloc(sizeof(unsigned int));
    unsigned int** timeRef = (unsigned int**) malloc(sizeof(unsigned int*));

    double* avgFlow = (double*) malloc(sizeof(double)  * 2);
    unsigned int *potMissesFlow = (unsigned int*) malloc(sizeof(unsigned int) * 2);
    unsigned int** time = (unsigned int**) malloc(sizeof(unsigned int*) * 2);
    if (avgFlowRef == nullptr || potMissesFlowRef == nullptr || timeRef == nullptr ||
        avgFlow == nullptr || potMissesFlow == nullptr || time == nullptr) {
        FreeMeasureTwoROResOnlyPtr()
        printErrorCodeInformation(1);
        exit(1);
    }
    timeRef[0] = time[0] = time[1] = nullptr;

    bool dist = true; int n = 5;
    while(dist && n > 0) {
        int error = 0;
        dist = launchROBenchmarkReferenceValue((int) CacheSizeInInt, 1, avgFlowRef, potMissesFlowRef, timeRef, &error);
        if (error != 0) {
            FreeMeasureTwoROResources()
            printErrorCodeInformation(error);
            exit(error);
        }
        --n;
    }

    dist = true;
    n = 5;
    while(dist && n > 0) {
        int error = 0;
        dist = launchBenchmarkTwoRO(CacheSizeInInt, &avgFlow[0], &avgFlow[1], &potMissesFlow[0], &potMissesFlow[1], &time[0],
                                    &time[1], &error);
        if (error != 0) {
            FreeMeasureTwoROResources()
            printErrorCodeInformation(error);
            exit(error);
        }
        --n;
    }

    dTriple result;
    result.first = avgFlowRef[0];
    result.second = avgFlow[0];
    result.third = avgFlow[1];
#ifdef IsDebug
    fprintf(out, "Measured RO Avg in clean execution: %f\n", avgFlowRef[0]);

    fprintf(out, "Measured RO1 Avg While Shared With RO2:  %f\n", avgFlow[0]);
    fprintf(out, "Measured RO1 Pot Misses While Shared With RO2:  %u\n", potMissesFlow[0]);

    fprintf(out, "Measured RO2 Avg While Shared With RO1:  %f\n", avgFlow[1]);
    fprintf(out, "Measured RO2 Pot Misses While Shared With RO1:  %u\n", potMissesFlow[1]);
#endif //IsDebug

    FreeMeasureTwoROResources()

    return std::max(std::abs(result.second - result.first), std::abs(result.third - result.first));
}


#endif //CUDATEST_TWORO