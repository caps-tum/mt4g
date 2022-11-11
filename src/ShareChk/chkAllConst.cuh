
#ifndef CUDATEST_ALLCONST
#define CUDATEST_ALLCONST

# include <cstdio>
# include <cstdint>
# include "cuda.h"

/**
 * See launchBenchmarkTwoCoreTexture
 * @param TextureN
 * @param durationTxt1
 * @param durationTxt2
 * @param indexTxt1
 * @param indexTxt2
 * @param isDisturbed
 * @param baseCore
 * @param testCore
 */
__global__ void chkTwoCoreConst(unsigned int ConstN, unsigned int * durationTxt1, unsigned int * durationTxt2, unsigned int *indexTxt1,
                                unsigned int *indexTxt2, bool* isDisturbed, unsigned int baseCore, unsigned int testCore) {
    *isDisturbed = false;

    unsigned int start_time, end_time;
    unsigned int j = 0;
    __shared__ long long s_tvalueTxt1[lessSize];
    __shared__ unsigned int s_indexTxt1[lessSize];
    __shared__ long long s_tvalueTxt2[lessSize];
    __shared__ unsigned int s_indexTxt2[lessSize];

    __syncthreads();

    if (threadIdx.x == baseCore) {
        for (int k = 0; k < lessSize; k++) {
            s_indexTxt1[k] = 0;
            s_tvalueTxt1[k] = 0;
        }
        j = 0;
    }

    __syncthreads();

    if (threadIdx.x == testCore) {
        for (int k = 0; k < lessSize; k++) {
            s_indexTxt2[k] = 0;
            s_tvalueTxt2[k] = 0;
        }
        j = ConstN;
    }

    __syncthreads();

    if (threadIdx.x == baseCore) {
        for (int k = 0; k < ConstN; k++) {
            j = arr[j];
            j = j % ConstN;
        }
    }

    __syncthreads();

    if (threadIdx.x == testCore) {
        j = constArrSize - ConstN;
        for (int k = 0; k < ConstN; k++) {
            j = arr[j];
            j = (j % ConstN) + ConstN;
        }
    }

    __syncthreads();

    if (threadIdx.x == baseCore) {
        //second round
        for (int k = 0; k < lessSize; k++) {
            start_time = clock();
            j = arr[j];
            s_indexTxt1[k] = j;
            end_time = clock();
            j = j % ConstN;
            s_tvalueTxt1[k] = (end_time - start_time);
        }
    }

    __syncthreads();

    if (threadIdx.x == testCore) {
        for (int k = 0; k < lessSize; k++) {
            start_time = clock();
            j = arr[j];
            s_indexTxt2[k] = j;
            end_time = clock();
            j = (j % ConstN) + ConstN;
            s_tvalueTxt2[k] = (end_time - start_time);
        }
    }

    __syncthreads();

    if (threadIdx.x == baseCore) {
        for (int k = 0; k < lessSize; k++) {
            indexTxt1[k] = s_indexTxt1[k];
            durationTxt1[k] = s_tvalueTxt1[k];
            if (durationTxt1[k] > 3000) {
                *isDisturbed = true;
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == testCore){
        for (int k = 0; k < lessSize; k++) {
            indexTxt2[k] = s_indexTxt2[k];
            durationTxt2[k] = s_tvalueTxt2[k];
            if (durationTxt2[k] > 3000) {
                *isDisturbed = true;
            }
        }
    }
}

/**
 * launches the two core share texture cache kernel benchmark
 * @param ConstN size of the Array
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
bool launchBenchmarkTwoCoreConst(unsigned int ConstN, double *avgOut1, double* avgOut2, unsigned int* potMissesOut1,
                                 unsigned int* potMissesOut2, unsigned int **time1, unsigned int **time2, int* error,
                                 unsigned int numberOfCores, unsigned int baseCore, unsigned int testCore) {
    cudaDeviceReset();
    cudaError_t error_id;

    unsigned int *h_indexTexture1 = nullptr, *h_indexTexture2 = nullptr, *h_timeinfoTexture1 = nullptr, *h_timeinfoTexture2 = nullptr,
    *durationTxt1 = nullptr, *durationTxt2 = nullptr, *d_indexTxt1 = nullptr, *d_indexTxt2 = nullptr;
    bool *disturb = nullptr, *d_disturb = nullptr;

    do {
        // Allocate Memory on Host
        h_indexTexture1 = (unsigned int *) malloc(sizeof(unsigned int) * lessSize);
        if (h_indexTexture1 == nullptr) {
            printf("[CHKALLCONST.CUH]: malloc h_indexTexture1 Error\n");
            *error = 1;
            break;
        }

        h_indexTexture2 = (unsigned int *) malloc(sizeof(unsigned int) * lessSize);
        if (h_indexTexture2 == nullptr) {
            printf("[CHKALLCONST.CUH]: malloc h_indexTexture2 Error\n");
            *error = 1;
            break;
        }

        h_timeinfoTexture1 = (unsigned int *) malloc(sizeof(unsigned int) * lessSize);
        if (h_timeinfoTexture1 == nullptr) {
            printf("[CHKALLCONST.CUH]: malloc h_timeinfoTexture1 Error\n");
            *error = 1;
            break;
        }

        h_timeinfoTexture2 = (unsigned int *) malloc(sizeof(unsigned int) * lessSize);
        if (h_timeinfoTexture2 == nullptr) {
            printf("[CHKALLCONST.CUH]: malloc h_timeinfoTexture2 Error\n");
            *error = 1;
            break;
        }

        disturb = (bool *) malloc(sizeof(bool));
        if (disturb == nullptr) {
            printf("[CHKALLCONST.CUH]: malloc disturb Error\n");
            *error = 1;
            break;
        }

        // Allocate Memory on GPU
        error_id = cudaMalloc((void **) &durationTxt1, sizeof(unsigned int) * lessSize);
        if (error_id != cudaSuccess) {
            printf("[CHKALLCONST.CUH]: cudaMalloc durationTxt1 Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &durationTxt2, sizeof(unsigned int) * lessSize);
        if (error_id != cudaSuccess) {
            printf("[CHKALLCONST.CUH]: cudaMalloc durationTxt2 Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_indexTxt1, sizeof(unsigned int) * lessSize);
        if (error_id != cudaSuccess) {
            printf("[CHKALLCONST.CUH]: cudaMalloc d_indextxt1 Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_indexTxt2, sizeof(unsigned int) * lessSize);
        if (error_id != cudaSuccess) {
            printf("[CHKALLCONST.CUH]: cudaMalloc d_indexTxt2 Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_disturb, sizeof(bool));
        if (error_id != cudaSuccess) {
            printf("[CHKALLCONST.CUH]: cudaMalloc disturb Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }
        cudaDeviceSynchronize();

        // Launch Kernel function
        dim3 Db = dim3(numberOfCores);
        dim3 Dg = dim3(1, 1, 1);
        chkTwoCoreConst<<<Dg, Db>>>( ConstN, durationTxt1, durationTxt2, d_indexTxt1,
                                     d_indexTxt2,d_disturb, baseCore, testCore);

        cudaDeviceSynchronize();
        error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
            printf("[CHKALLCONST.CUH]: Kernel launch/execution Error: %s\n", cudaGetErrorString(error_id));
            *error = 5;
            break;
        }

        // Copy results from GPU to Hosst
        error_id = cudaMemcpy((void *) h_timeinfoTexture1, (void *) durationTxt1, sizeof(unsigned int) * lessSize,
                              cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKALLCONST.CUH]: cudaMemcpy durationTxt1 Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) h_timeinfoTexture2, (void *) durationTxt2, sizeof(unsigned int) * lessSize,
                              cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKALLCONST.CUH]: cudaMemcpy durationTxt2 Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) h_indexTexture1, (void *) d_indexTxt1, sizeof(unsigned int) * lessSize,
                              cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKALLCONST.CUH]: cudaMemcpy d_indexTxt1 Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) h_indexTexture2, (void *) d_indexTxt2, sizeof(unsigned int) * lessSize,
                              cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKALLCONST.CUH]: cudaMemcpy d_indexTxt2 Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) disturb, (void *) d_disturb, sizeof(bool), cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKALLCONST.CUH]: cudaMemcpy disturb Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        char prefix1[64], prefix2[64];
        snprintf(prefix1, 64, "AllConst_T1_%d_%d", baseCore, testCore);
        snprintf(prefix2, 64, "AllConst_T2_%d_%d", baseCore, testCore);

        createOutputFile((int) ConstN, lessSize, h_indexTexture1, h_timeinfoTexture1, avgOut1, potMissesOut1, prefix1);
        createOutputFile((int) ConstN, lessSize, h_indexTexture2, h_timeinfoTexture2, avgOut2, potMissesOut2, prefix2);
    } while(false);

    bool ret = false;
    if (disturb != nullptr) {
        ret = *disturb;
        free(disturb);
    }

    // Free Memory on GPU
    if (d_indexTxt1 != nullptr) {
        cudaFree(d_indexTxt1);
    }

    if (d_indexTxt2 != nullptr) {
        cudaFree(d_indexTxt2);
    }

    if (durationTxt1 != nullptr) {
        cudaFree(durationTxt1);
    }

    if (durationTxt2 != nullptr) {
        cudaFree(durationTxt2);
    }

    if (d_disturb != nullptr) {
        cudaFree(d_disturb);
    }

    // Free Memory on Host
    if (h_indexTexture1 != nullptr) {
        free(h_indexTexture1);
    }

    if (h_indexTexture2 != nullptr) {
        free(h_indexTexture2);
    }

    if (time1 != nullptr) {
        time1[0] = h_timeinfoTexture1;
    } else {
        free(h_timeinfoTexture1);
    }

    if (time2 != nullptr) {
        time2[0] = h_timeinfoTexture2;
    } else {
        free(h_timeinfoTexture2);
    }

    cudaDeviceReset();
    return ret;
}

#endif //CUDATEST_ALLCONST