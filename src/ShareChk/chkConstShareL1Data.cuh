
#ifndef CUDATEST_CONSTSHAREDATA
#define CUDATEST_CONSTSHAREDATA

# include <cstdio>
# include <cstdint>

# include "cuda.h"

__global__ void chkConstShareData(unsigned int ConstN, unsigned int DataN, unsigned int * my_array, unsigned int * durationConst, unsigned int * durationData, unsigned int *indexConst, unsigned int *indexData,
                                  bool* isDisturbed) {
    *isDisturbed = false;

    unsigned int start_time, end_time;
    unsigned int j = 0;
    __shared__ long long s_tvalueConst[LESS_SIZE];
    __shared__ unsigned int s_indexConst[LESS_SIZE];
    __shared__ long long s_tvalueData[LESS_SIZE];
    __shared__ unsigned int s_indexData[LESS_SIZE];

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < LESS_SIZE; k++) {
            s_indexConst[k] = 0;
            s_tvalueConst[k] = 0;
        }
    }

    __syncthreads();

    if (threadIdx.x == 1) {
        for(int k=0; k<LESS_SIZE; k++){
            s_indexData[k] = 0;
            s_tvalueData[k] = 0;
        }
    }

    unsigned int* ptr;
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < ConstN; k++) {
            j = arr[j];
            j = j % ConstN;
        }
    }

    __syncthreads();

    if (threadIdx.x == 1) {
        for (int k = 0; k < DataN; k++) {
            ptr = my_array + j;
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(j) : "l"(ptr) : "memory");
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < LESS_SIZE; k++) {
            start_time = clock();
            j = arr[j];
            s_indexConst[k] = j;
            end_time = clock();
            j = j % ConstN;
            s_tvalueConst[k] = end_time - start_time;
        }
    }

    __syncthreads();

    if (threadIdx.x == 1) {
        asm volatile(" .reg .u64 smem_ptr64;\n\t"
                     " cvta.to.shared.u64 smem_ptr64, %0;\n\t" :: "l"(s_indexData));
        for (int k = 0; k < LESS_SIZE; k++) {
            ptr = my_array + j;
            asm volatile ("mov.u32 %0, %%clock;\n\t"
                          "ld.global.ca.u32 %1, [%3];\n\t"
                          "st.shared.u32 [smem_ptr64], %1;"
                          "mov.u32 %2, %%clock;\n\t"
                          "add.u64 smem_ptr64, smem_ptr64, 4;" : "=r"(start_time), "=r"(j), "=r"(end_time) : "l"(ptr) : "memory");
            s_tvalueData[k] = end_time-start_time;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < LESS_SIZE; k++) {
            indexConst[k] = s_indexConst[k];
            durationConst[k] = s_tvalueConst[k];
            if (durationConst[k] > 3000) {
                *isDisturbed = true;
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == 1) {
        for(int k=0; k<LESS_SIZE; k++){
            indexData[k]= s_indexData[k];
            durationData[k] = s_tvalueData[k];
            if (durationData[k] > 3000) {
                *isDisturbed = true;
            }
        }
    };
}

bool launchL1DataBenchmarkReferenceValue(int N, int stride, double *avgOut, unsigned int* potMissesOut, unsigned int** time, int* error) {
    return launchL1KernelBenchmark(N, stride, avgOut, potMissesOut, time, error);
}

bool launchConstBenchmarkReferenceValue(int N, double *avgOut, unsigned int* potMissesOut, unsigned int** time, int* error) {
    return launchConstantBenchmarkR1(N, avgOut, potMissesOut, time, error);
}

bool launchBenchmarkChkConstShareData(unsigned int ConstN, unsigned int DataN, double *avgOutConst, double* avgOutData, unsigned int* potMissesOutConst,
                                      unsigned int* potMissesOutData, unsigned int **timeConst, unsigned int **timeData, int* error) {
    cudaDeviceReset();
    cudaError_t error_id;

    unsigned int *h_indexConst = nullptr, *h_indexData = nullptr, *h_timeinfoConst = nullptr, *h_timeinfoData = nullptr, *h_a = nullptr,
    *durationConst = nullptr, *durationData = nullptr, *d_indexConst = nullptr, *d_indexData = nullptr, *d_a = nullptr;
    bool *disturb = nullptr, *d_disturb = nullptr;

    do {
        // Allocate Memory on Host
        h_indexConst = (unsigned int *) malloc(sizeof(unsigned int) * LESS_SIZE);
        if (h_indexConst == nullptr) {
            printf("[CHKCONSTSHAREL1DATA.CUH]: malloc h_indexConst Error\n");
            *error = 1;
            break;
        }

        h_indexData = (unsigned int *) malloc(sizeof(unsigned int) * LESS_SIZE);
        if (h_indexData == nullptr) {
            printf("[CHKCONSTSHAREL1DATA.CUH]: malloc h_indexData Error\n");
            *error = 1;
            break;
        }

        h_timeinfoConst = (unsigned int *) malloc(sizeof(unsigned int) * LESS_SIZE);
        if (h_timeinfoConst == nullptr) {
            printf("[CHKCONSTSHAREL1DATA.CUH]: malloc h_timeinfoConst Error\n");
            *error = 1;
            break;
        }

        h_timeinfoData = (unsigned int *) malloc(sizeof(unsigned int) * LESS_SIZE);
        if (h_timeinfoData == nullptr) {
            printf("[CHKCONSTSHAREL1DATA.CUH]: malloc h_timeinfoData Error\n");
            *error = 1;
            break;
        }

        disturb = (bool *) malloc(sizeof(bool));
        if (disturb == nullptr) {
            printf("[CHKCONSTSHAREL1DATA.CUH]: malloc disturb Error\n");
            *error = 1;
            break;
        }

        h_a = (unsigned int *) malloc(sizeof(unsigned int) * (DataN));
        if (h_a == nullptr) {
            printf("[CHKCONSTSHAREL1DATA.CUH]: malloc h_a Error\n");
            *error = 1;
            break;
        }

        // Allocate Memory on GPU
        error_id = cudaMalloc((void **) &durationConst, sizeof(unsigned int) * LESS_SIZE);
        if (error_id != cudaSuccess) {
            printf("[CHKCONSTSHAREL1DATA.CUH]: cudaMalloc durationConst Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &durationData, sizeof(unsigned int) * LESS_SIZE);
        if (error_id != cudaSuccess) {
            printf("[CHKCONSTSHAREL1DATA.CUH]: cudaMalloc durationData Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_indexConst, sizeof(unsigned int) * LESS_SIZE);
        if (error_id != cudaSuccess) {
            printf("[CHKCONSTSHAREL1DATA.CUH]: cudaMalloc d_indexConst Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_indexData, sizeof(unsigned int) * LESS_SIZE);
        if (error_id != cudaSuccess) {
            printf("[CHKCONSTSHAREL1DATA.CUH]: cudaMalloc d_indexData Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_disturb, sizeof(bool));
        if (error_id != cudaSuccess) {
            printf("[CHKCONSTSHAREL1DATA.CUH]: cudaMalloc disturb Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_a, sizeof(unsigned int) * (DataN));
        if (error_id != cudaSuccess) {
            printf("[CHKCONSTSHAREL1DATA.CUH]: cudaMalloc d_a Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        // Initialize p-chase array
        for (int i = 0; i < DataN; i++) {
            h_a[i] = (i + 1) % DataN;
        }

        // Copy array from Host to GPU
        error_id = cudaMemcpy(d_a, h_a, sizeof(unsigned int) * DataN, cudaMemcpyHostToDevice);
        if (error_id != cudaSuccess) {
            printf("[CHKCONSTSHAREL1DATA.CUH]: cudaMemcpy d_a Error: %s\n", cudaGetErrorString(error_id));
            *error = 3;
            break;
        }
        cudaDeviceSynchronize();

        // Launch Kernel function
        dim3 Db = dim3(2);
        dim3 Dg = dim3(1, 1, 1);
        chkConstShareData<<<Dg, Db>>>(ConstN, DataN, d_a, durationConst, durationData, d_indexConst, d_indexData, d_disturb);

        cudaDeviceSynchronize();
        error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
            printf("[CHKCONSTSHAREL1DATA.CUH]: Kernel launch/execution Error: %s\n", cudaGetErrorString(error_id));
            *error = 5;
            break;
        }

        // Copy results from GPU to Host
        error_id = cudaMemcpy((void *) h_timeinfoConst, (void *) durationConst, sizeof(unsigned int) * LESS_SIZE,
                              cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKCONSTSHAREL1DATA.CUH]: cudaMemcpy durationConst Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) h_timeinfoData, (void *) durationData, sizeof(unsigned int) * LESS_SIZE,
                              cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKCONSTSHAREL1DATA.CUH]: cudaMemcpy durationData Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) h_indexConst, (void *) d_indexConst, sizeof(unsigned int) * LESS_SIZE,
                              cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKCONSTSHAREL1DATA.CUH]: cudaMemcpy d_indexConst Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) h_indexData, (void *) d_indexData, sizeof(unsigned int) * LESS_SIZE,
                              cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKCONSTSHAREL1DATA.CUH]: cudaMemcpy d_indexData Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) disturb, (void *) d_disturb, sizeof(bool), cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKCONSTSHAREL1DATA.CUH]: cudaMemcpy disturb Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        createOutputFile((int) ConstN, LESS_SIZE, h_indexConst, h_timeinfoConst, avgOutConst, potMissesOutConst,"ShareConstDataConst_");
        createOutputFile((int) DataN, LESS_SIZE, h_indexData, h_timeinfoData, avgOutData, potMissesOutData, "ShareConstDataData_");
    } while(false);

    bool ret = false;
    if (disturb != nullptr) {
        ret = *disturb;
        free(disturb);
    }

    // Free Memory on GPU
    if (d_indexConst != nullptr) {
        cudaFree(d_indexConst);
    }

    if (d_indexData != nullptr) {
        cudaFree(d_indexData);
    }

    if (durationConst != nullptr) {
        cudaFree(durationConst);
    }

    if (durationData != nullptr) {
        cudaFree(durationData);
    }

    if (d_disturb != nullptr) {
        cudaFree(d_disturb);
    }

    // Free Memory on Host
    if (h_indexConst != nullptr) {
        free(h_indexConst);
    }

    if (h_indexData != nullptr) {
        free(h_indexData);
    }

    if (h_timeinfoConst != nullptr) {
        if (timeConst != nullptr) {
            timeConst[0] = h_timeinfoConst;
        } else {
            free(h_timeinfoConst);
        }
    }

    if (h_timeinfoData != nullptr) {
        if (timeData != nullptr) {
            timeData[0] = h_timeinfoData;
        } else {
            free(h_timeinfoData);
        }
    }

    cudaDeviceReset();
    return ret;
}

#define FreeMeasureConstL1ResOnlyPtr()      \
free(time);                                 \
free(avgFlow);                              \
free(potMissesFlow);                        \
free(timeRefL1);                            \
free(avgFlowRefL1);                         \
free(potMissesFlowRefL1);                   \
free(timeRefConst);                         \
free(avgFlowRefConst);                      \
free(potMissesFlowRefConst);                \

#define FreeMeasureConstL1Resources()       \
if (time[0] != nullptr) {                   \
    free(time[0]);                          \
}                                           \
if (time[1] != nullptr) {                   \
    free(time[1]);                          \
}                                           \
if (timeRefL1[0] != nullptr) {              \
    free(timeRefL1[0]);                     \
}                                           \
if (timeRefConst[0] != nullptr) {           \
    free(timeRefConst[0]);                  \
}                                           \
free(time);                                 \
free(avgFlow);                              \
free(potMissesFlow);                        \
free(timeRefL1);                            \
free(avgFlowRefL1);                         \
free(potMissesFlowRefL1);                   \
free(timeRefConst);                         \
free(avgFlowRefConst);                      \
free(potMissesFlowRefConst);                \

dTuple measure_ConstShareData(unsigned int measuredSizeConstL1, unsigned int measuredSizeDataL1, unsigned int sub) {
    unsigned int ConstSizeInInt = (measuredSizeConstL1-sub) >> 2; // / 4;
    unsigned int L1DataSizeInInt = (measuredSizeDataL1-sub) >> 2; // / 4;

    double* avgFlowRefConst = (double*) malloc(sizeof(double));
    unsigned int *potMissesFlowRefConst = (unsigned int*) malloc(sizeof(unsigned int));
    unsigned int** timeRefConst = (unsigned int**) malloc(sizeof(unsigned int*));

    double* avgFlowRefL1 = (double*) malloc(sizeof(double));
    unsigned int *potMissesFlowRefL1 = (unsigned int*) malloc(sizeof(unsigned int));
    unsigned int** timeRefL1 = (unsigned int**) malloc(sizeof(unsigned int*));

    double* avgFlow = (double*) malloc(sizeof(double)  * 2);
    unsigned int *potMissesFlow = (unsigned int*) malloc(sizeof(unsigned int) * 2);
    unsigned int** time = (unsigned int**) malloc(sizeof(unsigned int*) * 2);
    if (avgFlowRefConst == nullptr || potMissesFlowRefConst == nullptr || timeRefConst == nullptr ||
        avgFlowRefL1 == nullptr ||potMissesFlowRefL1 == nullptr || timeRefL1 == nullptr ||
        avgFlow == nullptr || potMissesFlow == nullptr || time == nullptr) {
        FreeMeasureConstL1ResOnlyPtr()
        printErrorCodeInformation(1);
        exit(1);
    }

    timeRefConst[0] = timeRefL1[0] = time[0] = time[1] = nullptr;

    bool dist = true;
    int n = 5;
    while (dist && n > 0) {
        int error = 0;
        dist = launchConstBenchmarkReferenceValue((int) ConstSizeInInt, avgFlowRefConst, potMissesFlowRefConst, timeRefConst, &error);
        if (error != 0) {
            FreeMeasureConstL1Resources()
            printErrorCodeInformation(error);
            exit(error);
        }
        --n;
    }

    dist = true;
    n = 5;
    while (dist && n > 0) {
        int error = 0;
        dist = launchL1DataBenchmarkReferenceValue((int) L1DataSizeInInt, 1, avgFlowRefL1, potMissesFlowRefL1, timeRefL1, &error);
        if (error != 0) {
            FreeMeasureConstL1Resources()
            printErrorCodeInformation(error);
            exit(error);
        }
        --n;
    }

    dist = true;
    n = 5;
    while(dist && n > 0) {
        int error = 0;
        dist = launchBenchmarkChkConstShareData(ConstSizeInInt, L1DataSizeInInt, &avgFlow[0], &avgFlow[1], &potMissesFlow[0],
                                         &potMissesFlow[1], &time[0], &time[1], &error);
        if (error != 0) {
            FreeMeasureConstL1Resources()
            printErrorCodeInformation(error);
            exit(error);
        }
        --n;
    }
#ifdef IsDebug
    fprintf(out, "Measured Const Avg in clean execution: %f\n", avgFlowRefConst[0]);
    fprintf(out, "Measured L1 Data Avg in clean execution: %f\n", avgFlowRefL1[0]);

    fprintf(out, "Measured Const Avg While Shared With L1Data:  %f\n", avgFlow[0]);
    fprintf(out, "Measured Const Pot Misses While Shared With L1Data:  %u\n", potMissesFlow[0]);

    fprintf(out, "Measured L1 Data Avg While Shared With Const:  %f\n", avgFlow[1]);
    fprintf(out, "Measured L1 Data Pot Misses While Shared With Const:  %u\n", potMissesFlow[1]);
#endif //IsDebug

    dTuple result;
    result.first = (double) potMissesFlow[0]; //Constant check over measured misses
    result.second = std::abs(avgFlow[1] - avgFlowRefL1[0]);

    FreeMeasureConstL1Resources()

    return result;
}

#endif //CUDATEST_CONSTSHAREDATA