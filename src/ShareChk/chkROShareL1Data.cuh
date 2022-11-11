
#ifndef CUDATEST_ROSHAREDATA
#define CUDATEST_ROSHAREDATA

# include <cstdio>
# include <cstdint>

# include "cuda.h"

__global__ void chkROShareL1Data(unsigned int RON, unsigned int DataN, const unsigned int* __restrict__ myArrayReadOnly, unsigned int* my_array,
                                  unsigned int * durationRO, unsigned int * durationData, unsigned int *indexRO, unsigned int *indexData,
                                  bool* isDisturbed) {
    *isDisturbed = false;

    unsigned int start_time, end_time;
    unsigned int j = 0;
    __shared__ long long s_tvalueRO[lessSize];
    __shared__ unsigned int s_indexRO[lessSize];
    __shared__ long long s_tvalueData[lessSize];
    __shared__ unsigned int s_indexData[lessSize];

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < lessSize; k++) {
            s_indexRO[k] = 0;
            s_tvalueRO[k] = 0;
        }
    }

    __syncthreads();

    if (threadIdx.x == 1) {
        for(int k=0; k<lessSize; k++){
            s_indexData[k] = 0;
            s_tvalueData[k] = 0;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < RON; k++)
            j = __ldg(&myArrayReadOnly[j]);
    }

    unsigned int* ptr;
    __syncthreads();

    if (threadIdx.x == 1) {
        for (int k = 0; k < DataN; k++) {
            ptr = my_array + j;
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(j) : "l"(ptr) : "memory");
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        //second round
        for (int k = 0; k < lessSize; k++) {
            start_time = clock();
            j = __ldg(&myArrayReadOnly[j]);
            s_indexRO[k] = j;
            end_time = clock();
            s_tvalueRO[k] = end_time - start_time;
        }
    }

    __syncthreads();

    if (threadIdx.x == 1) {
        asm volatile(" .reg .u64 smem_ptr64;\n\t"
                     " cvta.to.shared.u64 smem_ptr64, %0;\n\t" :: "l"(s_indexData));
        for (int k = 0; k < lessSize; k++) {
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
        for (int k = 0; k < lessSize; k++) {
            indexRO[k] = s_indexRO[k];
            durationRO[k] = s_tvalueRO[k];
            if (durationRO[k] > 3000) {
                *isDisturbed = true;
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == 1) {
        for(int k=0; k<lessSize; k++){
            indexData[k]= s_indexData[k];
            durationData[k] = s_tvalueData[k];
            if (durationData[k] > 3000) {
                *isDisturbed = true;
            }
        }
    }
}

bool launchBenchmarkChkROShareL1Data(unsigned int RO_N, unsigned int DataN, double *avgOutRO, double* avgOutData, unsigned int* potMissesOutRO,
                                     unsigned int* potMissesOutData, unsigned int **timeRO, unsigned int **timeData, int* error) {
    cudaDeviceReset();
    cudaError_t error_id;

    unsigned int *h_indexRO = nullptr, *h_indexData = nullptr, *h_timeinfoRO = nullptr, *h_timeinfoData = nullptr, *h_aData = nullptr, *h_aRO = nullptr,
    *durationRO = nullptr, *durationData = nullptr, *d_indexRO = nullptr, *d_indexData = nullptr, *d_aRO = nullptr, *d_aData = nullptr;
    bool *disturb = nullptr, *d_disturb = nullptr;

    do {
        // Allocate Memory on Host
        h_indexRO = (unsigned int *) malloc(sizeof(unsigned int) * lessSize);
        if (h_indexRO == nullptr) {
            printf("[CHKROSHAREL1DATA.CUH]: malloc h_indexRO Error\n");
            *error = 1;
            break;
        }

        h_indexData = (unsigned int *) malloc(sizeof(unsigned int) * lessSize);
        if (h_indexData == nullptr) {
            printf("[CHKROSHAREL1DATA.CUH]: malloc h_indexData Error\n");
            *error = 1;
            break;
        }

        h_timeinfoRO = (unsigned int *) malloc(sizeof(unsigned int) * lessSize);
        if (h_timeinfoRO == nullptr) {
            printf("[CHKROSHAREL1DATA.CUH]: malloc h_timeinfoRO Error\n");
            *error = 1;
            break;
        }

        h_timeinfoData = (unsigned int *) malloc(sizeof(unsigned int) * lessSize);
        if (h_timeinfoData == nullptr) {
            printf("[CHKROSHAREL1DATA.CUH]: malloc h_timeinfoData Error\n");
            *error = 1;
            break;
        }

        disturb = (bool *) malloc(sizeof(bool));
        if (disturb == nullptr) {
            printf("[CHKROSHAREL1DATA.CUH]: malloc disturb Error\n");
            *error = 1;
            break;
        }

        h_aData = (unsigned int *) malloc(sizeof(unsigned int) * (DataN));
        if (h_aData == nullptr) {
            printf("[CHKROSHAREL1DATA.CUH]: malloc h_aData Error\n");
            *error = 1;
            break;
        }

        h_aRO = (unsigned int *) malloc(sizeof(unsigned int) * (RO_N));
        if (h_aRO == nullptr) {
            printf("[CHKROSHAREL1DATA.CUH]: malloc h_aRO Error\n");
            *error = 1;
            break;
        }

        // Allocate Memory on GPU
        error_id = cudaMalloc((void **) &durationRO, sizeof(unsigned int) * lessSize);
        if (error_id != cudaSuccess) {
            printf("[CHKROSHAREL1DATA.CUH]: cudaMalloc durationRO Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &durationData, sizeof(unsigned int) * lessSize);
        if (error_id != cudaSuccess) {
            printf("[CHKROSHAREL1DATA.CUH]: cudaMalloc durationData Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_indexRO, sizeof(unsigned int) * lessSize);
        if (error_id != cudaSuccess) {
            printf("[CHKROSHAREL1DATA.CUH]: cudaMalloc d_indexRO Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_indexData, sizeof(unsigned int) * lessSize);
        if (error_id != cudaSuccess) {
            printf("[CHKROSHAREL1DATA.CUH]: cudaMalloc d_indexData Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_disturb, sizeof(bool));
        if (error_id != cudaSuccess) {
            printf("[CHKROSHAREL1DATA.CUH]: cudaMalloc disturb Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_aData, sizeof(unsigned int) * (DataN));
        if (error_id != cudaSuccess) {
            printf("[CHKROSHAREL1DATA.CUH]: cudaMalloc d_aData Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_aRO, sizeof(unsigned int) * (RO_N));
        if (error_id != cudaSuccess) {
            printf("[CHKROSHAREL1DATA.CUH]: cudaMalloc d_aRO Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        // Initialize p-chase arrays
        for (int i = 0; i < DataN; i++) {
            h_aData[i] = (i + 1) % DataN;
        }

        for (int i = 0; i < RO_N; i++) {
            h_aRO[i] = (i + 1) % RO_N;
        }

        // Copy arrays from Host to GPU
        error_id = cudaMemcpy(d_aData, h_aData, sizeof(unsigned int) * DataN, cudaMemcpyHostToDevice);
        if (error_id != cudaSuccess) {
            printf("[CHKROSHAREL1DATA.CUH]: cudaMemcpy d_aData Error: %s\n", cudaGetErrorString(error_id));
            *error = 3;
            break;
        }

        error_id = cudaMemcpy(d_aRO, h_aRO, sizeof(unsigned int) * RO_N, cudaMemcpyHostToDevice);
        if (error_id != cudaSuccess) {
            printf("[CHKROSHAREL1DATA.CUH]: cudaMemcpy d_aRO Error: %s\n", cudaGetErrorString(error_id));
            *error = 3;
            break;
        }

        cudaDeviceSynchronize();

        error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
            printf("[CHKROSHAREL1DATA.CUH]: cudaDeviceSynchronize Error: %s\n", cudaGetErrorString(error_id));
            *error = 99;
            break;
        }
        cudaDeviceSynchronize();

        // Launch Kernel function
        dim3 Db = dim3(2);
        dim3 Dg = dim3(1, 1, 1);
        chkROShareL1Data<<<Dg, Db>>>(RO_N, DataN, d_aRO, d_aData, durationRO, durationData, d_indexRO, d_indexData, d_disturb);

        cudaDeviceSynchronize();
        error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
            printf("[CHKROSHAREL1DATA.CUH]: Kernel launch/execution Error: %s\n", cudaGetErrorString(error_id));
            *error = 5;
            break;
        }

        // Copy results from GPU to Host
        error_id = cudaMemcpy((void *) h_timeinfoRO, (void *) durationRO, sizeof(unsigned int) * lessSize,
                              cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKROSHAREL1DATA.CUH]: cudaMemcpy durationRO Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) h_timeinfoData, (void *) durationData, sizeof(unsigned int) * lessSize,
                              cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKROSHAREL1DATA.CUH]: cudaMemcpy durationData Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) h_indexRO, (void *) d_indexRO, sizeof(unsigned int) * lessSize, cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKROSHAREL1DATA.CUH]: cudaMemcpy d_indexRO Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) h_indexData, (void *) d_indexData, sizeof(unsigned int) * lessSize,
                              cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKROSHAREL1DATA.CUH]: cudaMemcpy d_indexData Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) disturb, (void *) d_disturb, sizeof(bool), cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKROSHAREL1DATA.CUH]: cudaMemcpy disturb Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        createOutputFile((int) RO_N, lessSize, h_indexRO, h_timeinfoRO, avgOutRO, potMissesOutRO, "ShareRODataRO_");
        createOutputFile((int) DataN, lessSize, h_indexData, h_timeinfoData, avgOutData, potMissesOutData, "ShareRODataData_");
    } while(false);

    bool ret = false;
    if (disturb != nullptr) {
        ret = *disturb;
        free(disturb);
    }

    // Free Memory on GPU
    if (d_indexRO != nullptr) {
        cudaFree(d_indexRO);
    }

    if (d_indexData != nullptr) {
        cudaFree(d_indexData);
    }

    if (durationRO != nullptr) {
        cudaFree(durationRO);
    }

    if (durationData != nullptr) {
        cudaFree(durationData);
    }

    if (d_aData != nullptr) {
        cudaFree(d_aData);
    }

    if (d_aRO != nullptr) {
        cudaFree(d_aRO);
    }

    if (d_disturb != nullptr) {
        cudaFree(d_disturb);
    }

    // Free Memory on Host
    if (h_indexRO != nullptr) {
        free(h_indexRO);
    }

    if (h_indexData != nullptr) {
        free(h_indexData);
    }

    if (h_aRO != nullptr) {
        free(h_aRO);
    }

    if (h_aData != nullptr) {
        free(h_aData);
    }

    if (h_timeinfoRO != nullptr) {
        if (timeRO != nullptr) {
            timeRO[0] = h_timeinfoRO;
        } else {
            free(h_timeinfoRO);
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

#define FreeMeasureROL1ResOnlyPtr() \
free(time);                         \
free(avgFlow);                      \
free(potMissesFlow);                \
free(timeRefRO);                    \
free(avgFlowRefRO);                 \
free(potMissesFlowRefRO);           \
free(timeRefL1);                    \
free(avgFlowRefL1);                 \
free(potMissesFlowRefL1);           \

#define FreeMeasureROL1Resources()          \
if (time[0] != nullptr) {                   \
    free(time[0]);                          \
}                                           \
if (time[1] != nullptr) {                   \
    free(time[1]);                          \
}                                           \
if (timeRefRO[0] != nullptr) {              \
    free(timeRefRO[0]);                     \
}                                           \
if (timeRefL1[0] != nullptr) {              \
    free(timeRefL1[0]);                     \
}                                           \
free(time);                                 \
free(avgFlow);                              \
free(potMissesFlow);                        \
free(timeRefRO);                            \
free(avgFlowRefRO);                         \
free(potMissesFlowRefRO);                   \
free(timeRefL1);                            \
free(avgFlowRefL1);                         \
free(potMissesFlowRefL1);                   \

dTuple measure_ROShareL1Data(unsigned int measuredSizeRO, unsigned int measuredSizeData, unsigned int sub) {
    unsigned int ROSizeInInt = (measuredSizeRO-sub) >> 2; // / 4;
    unsigned int L1DataSizeInInt = (measuredSizeData - sub) >> 2; // / 4;

    double* avgFlowRefRO = (double*) malloc(sizeof(double));
    unsigned int *potMissesFlowRefRO = (unsigned int*) malloc(sizeof(unsigned int));
    unsigned int** timeRefRO = (unsigned int**) malloc(sizeof(unsigned int*));

    double* avgFlowRefL1 = (double*) malloc(sizeof(double));
    unsigned int *potMissesFlowRefL1 = (unsigned int*) malloc(sizeof(unsigned int));
    unsigned int** timeRefL1 = (unsigned int**) malloc(sizeof(unsigned int*));

    double* avgFlow = (double*) malloc(sizeof(double)  * 2);
    unsigned int *potMissesFlow = (unsigned int*) malloc(sizeof(unsigned int) * 2);
    unsigned int** time = (unsigned int**) malloc(sizeof(unsigned int*) * 2);
    if (avgFlowRefRO == nullptr || potMissesFlowRefRO == nullptr || timeRefRO == nullptr ||
        avgFlowRefL1 == nullptr || potMissesFlowRefL1 == nullptr || timeRefL1 == nullptr ||
        avgFlow == nullptr || potMissesFlow == nullptr || time == nullptr) {
        FreeMeasureROL1ResOnlyPtr()
        printErrorCodeInformation(1);
        exit(1);
    }
    timeRefRO[0] = timeRefL1[0] = time[0] = time[1] = nullptr;

    bool dist = true;
    int n = 5;
    while (dist && n > 0) {
        int error = 0;
        dist = launchROBenchmarkReferenceValue((int) ROSizeInInt, 1, avgFlowRefRO, potMissesFlowRefRO, timeRefRO, &error);
        if (error != 0) {
            FreeMeasureROL1Resources()
            printErrorCodeInformation(error);
            exit(error);
        }
        --n;
    }

    dist = true;
    n = 5;
    while(dist && n > 0) {
        int error = 0;
        dist = launchL1DataBenchmarkReferenceValue((int) L1DataSizeInInt, 1, avgFlowRefL1, potMissesFlowRefL1, timeRefL1, &error);
        if (error != 0) {
            FreeMeasureROL1Resources()
            printErrorCodeInformation(error);
            exit(error);
        }
        --n;
    }

    dist = true;
    n = 5;
    while(dist && n > 0) {
        int error = 0;
        dist = launchBenchmarkChkROShareL1Data(ROSizeInInt, L1DataSizeInInt, &avgFlow[0], &avgFlow[1], &potMissesFlow[0],
                                        &potMissesFlow[1], &time[0], &time[1], &error);
        if (error != 0) {
            FreeMeasureROL1Resources()
            printErrorCodeInformation(error);
            exit(-error);
        }
        --n;
    }
#ifdef IsDebug
    fprintf(out ,"Measured RO Avg in clean execution: %f\n", avgFlowRefRO[0]);
    fprintf(out, "Measured L1 Data Avg in clean execution: %f\n", avgFlowRefL1[0]);

    fprintf(out, "Measured RO Avg While Shared With L1 Data:  %f\n", avgFlow[0]);
    fprintf(out, "Measured RO Pot Misses While Shared With L1 Data:  %u\n", potMissesFlow[0]);

    fprintf(out, "Measured L1 Data Avg While Shared With RO:  %f\n", avgFlow[1]);
    fprintf(out, "Measured L1 Data Pot Misses While Shared With RO:  %u\n", potMissesFlow[1]);
#endif //IsDebug

    dTuple result;
    result.first = std::abs(avgFlow[0] - avgFlowRefRO[0]);
    result.second = std::abs(avgFlow[1] - avgFlowRefL1[0]);

    FreeMeasureROL1Resources()

    return result;
}


#endif //CUDATEST_ROSHAREDATA