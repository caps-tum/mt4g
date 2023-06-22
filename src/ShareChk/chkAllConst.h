#include "hip/hip_runtime.h"

#ifndef CUDATEST_ALLCONST
#define CUDATEST_ALLCONST

# include <cstdio>
# include <cstdint>
# include "hip/hip_runtime.h"
# include "../general_functions.h"
/*
 * Threshold where each value above is considered as an error
 * in AMD:      [~100..~1500] vs 3000
 * in Nvidia:   [~40..~105] vs 3000
 */
#define HARDCODED_3000 3000

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
__global__ void
chkTwoCoreConst(unsigned int ConstN, unsigned int *durationTxt1, unsigned int *durationTxt2, unsigned int *indexTxt1,
                unsigned int *indexTxt2, bool *isDisturbed, unsigned int baseCore, unsigned int testCore) {
    *isDisturbed = false;

    unsigned int start_time, end_time;
    unsigned int j = 0;
    __shared__ ALIGN(16) long long s_tvalueTxt1[lessSize];
    __shared__ ALIGN(16) unsigned int s_indexTxt1[lessSize];
    __shared__ ALIGN(16) long long s_tvalueTxt2[lessSize];
    __shared__ ALIGN(16) unsigned int s_indexTxt2[lessSize];

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
            if (durationTxt1[k] > HARDCODED_3000) {
                *isDisturbed = true;
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == testCore) {
        for (int k = 0; k < lessSize; k++) {
            indexTxt2[k] = s_indexTxt2[k];
            durationTxt2[k] = s_tvalueTxt2[k];
            if (durationTxt2[k] > HARDCODED_3000) {
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
 * @warning the loading time for only one value is VERY high
 *
 */
bool launchBenchmarkTwoCoreConst(unsigned int ConstN, double *avgOut1, double *avgOut2, unsigned int *potMissesOut1,
                                 unsigned int *potMissesOut2, unsigned int **time1, unsigned int **time2, int *error,
                                 unsigned int numberOfCores, unsigned int baseCore, unsigned int testCore) {

    resetDeviceAndCheck("chkAllCost.h", error);

    hipError_t error_id;
    hipError_t result;

    unsigned int *h_indexTexture1 = nullptr, *h_indexTexture2 = nullptr, *h_timeinfoTexture1 = nullptr, *h_timeinfoTexture2 = nullptr,
            *durationTxt1 = nullptr, *durationTxt2 = nullptr, *d_indexTxt1 = nullptr, *d_indexTxt2 = nullptr;
    bool *disturb = nullptr, *d_disturb = nullptr;

    do {
        // Allocate Memory on Host
        h_indexTexture1 = (unsigned int *) mallocAndCheck("chkAllCost.h", sizeof(unsigned int) * lessSize,
                                                          "h_indexTexture1", error);

        h_indexTexture2 = (unsigned int *) mallocAndCheck("chkAllCost.h", sizeof(unsigned int) * lessSize,
                                                          "h_indexTexture2", error);

        h_timeinfoTexture1 = (unsigned int *) mallocAndCheck("chkAllCost.h", sizeof(unsigned int) * lessSize,
                                                             "h_timeinfoTexture1", error);

        h_timeinfoTexture2 = (unsigned int *) mallocAndCheck("chkAllCost.h", sizeof(unsigned int) * lessSize,
                                                             "h_timeinfoTexture2", error);

        disturb = (bool *) mallocAndCheck("chkAllCost.h", sizeof(bool), "disturb", error);

        // Allocate Memory on GPU
        error_id = hipMallocAndCheck("chkAllConst.h", (void **) &durationTxt1,
                                     sizeof(unsigned int) * lessSize,
                                     "durationTxt1", error);

        error_id = hipMallocAndCheck("chkAllConst.h", (void **) &durationTxt2,
                                     sizeof(unsigned int) * lessSize,
                                     "durationTxt2", error);

        error_id = hipMallocAndCheck("chkAllConst.h", (void **) &d_indexTxt1,
                                     sizeof(unsigned int) * lessSize,
                                     "d_indexTxt1", error);

        error_id = hipMallocAndCheck("chkAllConst.h", (void **) &d_indexTxt2,
                                     sizeof(unsigned int) * lessSize,
                                     "d_indexTxt2", error);

        error_id = hipMallocAndCheck("chkAllConst.h", (void **) &d_disturb,
                                     sizeof(bool),
                                     "d_disturb", error);

        error_id = hipDeviceSynchronize();

        // Launch Kernel function
        dim3 Db = dim3(numberOfCores);
        dim3 Dg = dim3(1, 1, 1);
        hipLaunchKernelGGL(chkTwoCoreConst, Dg, Db, 0, 0, ConstN, durationTxt1, durationTxt2, d_indexTxt1,
                           d_indexTxt2, d_disturb, baseCore, testCore);

        error_id = hipDeviceSynchronize();

        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[CHKALLCONST.H]: Kernel launch/execution Error: %s\n", hipGetErrorString(error_id));
            *error = 5;
            break;
        }

        // Copy results from GPU to Host
        if (hipMemcpyAndCheck("chkAllConst.h", h_timeinfoTexture1, d_indexTxt1,
                              sizeof(unsigned int) * lessSize, "d_indexTxt1 -> h_timeinfoTexture1", error, true) !=
            hipSuccess) {
            break;
        }

        if (hipMemcpyAndCheck("chkAllConst.h", h_timeinfoTexture2, durationTxt2,
                              sizeof(unsigned int) * lessSize, "durationTxt2 -> h_timeinfoTexture2", error, true) !=
            hipSuccess) {
            break;
        }

        if (hipMemcpyAndCheck("chkAllConst.h", h_indexTexture1, d_indexTxt1,
                              sizeof(unsigned int) * lessSize, "d_indexTxt1 -> h_indexTexture1", error, true) != hipSuccess) {
            break;
        }
        if (hipMemcpyAndCheck("chkAllConst.h", h_indexTexture2, d_indexTxt2,
                              sizeof(unsigned int) * lessSize, "d_indexTxt2 -> h_indexTexture2", error, true) != hipSuccess) {
            break;
        }

        if (hipMemcpyAndCheck("chkAllConst.h", disturb, d_disturb,
                              sizeof(bool), "disturb -> d_disturb", error, true) != hipSuccess) {
            break;
        }

        char prefix1[64], prefix2[64];
        snprintf(prefix1, 64, "AllConst_T1_%d_%d", baseCore, testCore);
        snprintf(prefix2, 64, "AllConst_T2_%d_%d", baseCore, testCore);

        createOutputFile((int) ConstN, lessSize, h_indexTexture1, h_timeinfoTexture1, avgOut1, potMissesOut1, prefix1);
        createOutputFile((int) ConstN, lessSize, h_indexTexture2, h_timeinfoTexture2, avgOut2, potMissesOut2, prefix2);
    } while (false);

    bool ret = false;
    if (disturb != nullptr) {
        ret = *disturb;
        free(disturb);
    }

    // Free Memory on GPU
    FreeTestMemory({d_indexTxt1, d_indexTxt2, durationTxt1, durationTxt2, d_disturb}, true);

    // Free Memory on Host
    FreeTestMemory({h_indexTexture1, h_indexTexture2}, false);

    SET_PART_OF_2D(time1, h_timeinfoTexture1);
    SET_PART_OF_2D(time2, h_timeinfoTexture2);

    result = hipDeviceReset();
    if (result != hipSuccess) {
        std::cout << "chkAllConst.h\tError resetting device: " << hipGetErrorString(result) << std::endl;
    }
    return ret;
}

#endif //CUDATEST_ALLCONST
