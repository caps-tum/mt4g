#include "hip/hip_runtime.h"

#ifndef CUDATEST_ALLRO
#define CUDATEST_ALLRO

# include <cstdio>
# include <cstdint>
# include "hip/hip_runtime.h"
# include "../general_functions.h"

#define HARDCODED_3000 3000

/**
 * See launchBenchmarkTwoCoreTexture
 * @param RO_N
 * @param durationRO1
 * @param durationRO2
 * @param indexRO1
 * @param indexRO2
 * @param isDisturbed
 * @param baseCore
 * @param testCore
 */
__global__ void
chkTwoCoreRO(unsigned int RO_N, const unsigned int *__restrict__ arrayRO1, const unsigned int *__restrict__ arrayRO2,
             unsigned int *durationRO1, unsigned int *durationRO2, unsigned int *indexRO1, unsigned int *indexRO2,
             bool *isDisturbed, unsigned int baseCore, unsigned int testCore) {
    *isDisturbed = false;

    unsigned int start_time, end_time;
    unsigned int j = 0;
    __shared__ ALIGN(16) long long s_tvalueRO1[lessSize];
    __shared__ ALIGN(16) unsigned int s_indexRO1[lessSize];
    __shared__ ALIGN(16) long long s_tvalueRO2[lessSize];
    __shared__ ALIGN(16) unsigned int s_indexRO2[lessSize];

    __syncthreads();

    if (threadIdx.x == baseCore) {
        for (int k = 0; k < lessSize; k++) {
            s_indexRO1[k] = 0;
            s_tvalueRO1[k] = 0;
        }
    }

    __syncthreads();

    if (threadIdx.x == testCore) {
        for (int k = 0; k < lessSize; k++) {
            s_indexRO2[k] = 0;
            s_tvalueRO2[k] = 0;
        }
    }

    __syncthreads();

    if (threadIdx.x == baseCore) {
        for (int k = 0; k < RO_N; k++) {
            j = __ldg(&arrayRO1[j]);
        }
    }

    __syncthreads();

    if (threadIdx.x == testCore) {
        for (int k = 0; k < RO_N; k++) {
            j = __ldg(&arrayRO2[j]);
        }
    }

    __syncthreads();

    if (threadIdx.x == baseCore) {
        //second round
        for (int k = 0; k < lessSize; k++) {
            start_time = clock();
            j = __ldg(&arrayRO1[j]);
            s_indexRO1[k] = j;
            end_time = clock();
            s_tvalueRO1[k] = (end_time - start_time);
        }
    }

    __syncthreads();

    if (threadIdx.x == testCore) {
        for (int k = 0; k < lessSize; k++) {
            start_time = clock();
            j = __ldg(&arrayRO2[j]);
            s_indexRO2[k] = j;
            end_time = clock();
            s_tvalueRO2[k] = (end_time - start_time);
        }
    }

    __syncthreads();

    if (threadIdx.x == baseCore) {
        for (int k = 0; k < lessSize; k++) {
            indexRO1[k] = s_indexRO1[k];
            durationRO1[k] = s_tvalueRO1[k];
            if (durationRO1[k] > HARDCODED_3000) {
                *isDisturbed = true;
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == testCore) {
        for (int k = 0; k < lessSize; k++) {
            indexRO2[k] = s_indexRO2[k];
            durationRO2[k] = s_tvalueRO2[k];
            if (durationRO2[k] > HARDCODED_3000) {
                *isDisturbed = true;
            }
        }
    }
}

/**
 * launches the two core share read-only cache kernel benchmark
 * @param RO_N size of the Array
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
bool launchBenchmarkTwoCoreRO(unsigned int RO_N, double *avgOut1, double *avgOut2, unsigned int *potMissesOut1,
                              unsigned int *potMissesOut2, unsigned int **time1, unsigned int **time2, int *error,
                              unsigned int numberOfCores, unsigned int baseCore, unsigned int testCore) {
    resetDeviceAndCheck("[chkAllRO/138]", error);

    hipError_t result;
    hipError_t error_id;

    unsigned int *h_index1 = nullptr, *h_index2 = nullptr, *h_timeinfo1 = nullptr, *h_timeinfo2 = nullptr, *h_a = nullptr,
            *duration1 = nullptr, *duration2 = nullptr, *d_index1 = nullptr, *d_index2 = nullptr, *d_a1 = nullptr, *d_a2 = nullptr;
    bool *disturb = nullptr, *d_disturb = nullptr;

    do {
        h_index1 = (unsigned int *) mallocAndCheck("chkAllRO/151", sizeof(unsigned int) * lessSize,
                                                   "h_index1", error);

        h_index2 = (unsigned int *) mallocAndCheck("chkAllRO/156", sizeof(unsigned int) * lessSize,
                                                   "h_index2", error);

        h_timeinfo1 = (unsigned int *) mallocAndCheck("chkAllRO/161", sizeof(unsigned int) * lessSize,
                                                      "h_timeinfo1", error);

        h_timeinfo2 = (unsigned int *) mallocAndCheck("chkAllRO/166", sizeof(unsigned int) * lessSize,
                                                      "h_timeinfo2", error);

        disturb = (bool *) mallocAndCheck("chkAllRO/171", sizeof(bool), "disturb", error);

        h_a = (unsigned int *) mallocAndCheck("chkAllRO/175", sizeof(unsigned int) * RO_N, "h_a", error);


        // Allocate Memory on GPU
        if (hipMallocAndCheck("chkAllRO/182", (void **) &duration1,
                              sizeof(unsigned int) * lessSize,
                              "duration1", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("chkAllRO/188", (void **) &duration2,
                              sizeof(unsigned int) * lessSize,
                              "duration2", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("chkAllRO/194", (void **) &d_index1,
                                         sizeof(unsigned int) * lessSize,
                                         "d_index1", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("chkAllRO/200", (void **) &d_index2,
                              sizeof(unsigned int) * lessSize,
                              "d_index2", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("chkAllRO/206", (void **) &d_disturb,
                              sizeof(bool),
                              "disturb", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("chkAllRO/212", (void **) &d_a1,
                              sizeof(unsigned int) * RO_N,
                              "d_a1", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("chkAllRO/218", (void **) &d_a2,
                              sizeof(unsigned int) * RO_N,
                              "d_a2", error) != hipSuccess)
            break;

        // Initialize p-chase array
        for (int i = 0; i < RO_N; i++) {
            //original:
            h_a[i] = (i + 1) % RO_N;
        }

        // Copy array from Host to GPU
        if (hipMemcpyAndCheck("roAll", d_a1, h_a, sizeof(unsigned int) * RO_N,
                              "h_a -> d_a1", error, false) != hipSuccess)
            break;
        if (hipMemcpyAndCheck("roAll", d_a2, h_a, sizeof(unsigned int) * RO_N,
                              "h_a -> d_a2", error, false) != hipSuccess)
            break;

        result = hipDeviceSynchronize();
        if (result != hipSuccess) {
            std::cerr << "chkAllRO/261\tError synchronizing device: " << hipGetErrorString(result) << std::endl;
        }

        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[CHKALLRO.CPP]: Error 2 is %s\n", hipGetErrorString(error_id));
            *error = 99;
            break;
        }
        result = hipDeviceSynchronize();
        if (result != hipSuccess) {
            std::cerr << "chkAllRO/272\tError synchronizing device: " << hipGetErrorString(result) << std::endl;
            *error = 3;
        }

        // Launch Kernel function
        dim3 Db = dim3(numberOfCores);
        dim3 Dg = dim3(1, 1, 1);
        hipLaunchKernelGGL(chkTwoCoreRO, Dg, Db, 0, 0, RO_N, d_a1, d_a2, duration1, duration2, d_index1, d_index2,
                           d_disturb, baseCore,
                           testCore);

        result = hipDeviceSynchronize();
        if (result != hipSuccess) {
            std::cerr << "chkAllRO/283\tError synchronizing device: " << hipGetErrorString(result) << std::endl;
            *error = 3;
        }

        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[CHKALLRO.CPP]: Kernel launch/execution Error: %s\n", hipGetErrorString(error_id));
            *error = 5;
            break;
        }

        // Copy results from GPU to Host

        if (hipMemcpyAndCheck("chkAllRO/284", h_timeinfo1, duration1, sizeof(unsigned int) * lessSize,
                              "duration1 -> h_timeinfo1", error, true) != hipSuccess)
            break;

        if (hipMemcpyAndCheck("chkAllRO/287", h_timeinfo2, duration2, sizeof(unsigned int) * lessSize,
                              "duration2 -> h_timeinfo2", error, true) != hipSuccess)
            break;

        if (hipMemcpyAndCheck("chkAllRO/294", h_index1, d_index1, sizeof(unsigned int) * lessSize,
                              "d_index1 -> h_index1", error, true) != hipSuccess)
            break;

        if (hipMemcpyAndCheck("chkAllRO/299", h_index2, d_index2, sizeof(unsigned int) * lessSize,
                              "d_index2 -> h_index2", error, true) != hipSuccess)
            break;

        if (hipMemcpyAndCheck("chkAllRO/304", disturb, d_disturb, sizeof(bool),
                              "d_disturb -> disturb", error, true) != hipSuccess)
            break;

        char prefix1[64], prefix2[64];
        snprintf(prefix1, 64, "AllRO_T1_%d_%d", baseCore, testCore);
        snprintf(prefix2, 64, "AllRO_T2_%d_%d", baseCore, testCore);

        createOutputFile((int) RO_N, lessSize, h_index1, h_timeinfo1, avgOut1, potMissesOut1, prefix1);
        createOutputFile((int) RO_N, lessSize, h_index2, h_timeinfo2, avgOut2, potMissesOut2, prefix2);
    } while (false);

    bool ret = false;
    if (disturb != nullptr) {
        ret = *disturb;
        free(disturb);
    }

    // Free Memory on GPU
    FreeTestMemory({d_index1, d_index2, duration1, duration2, d_a1, d_disturb}, true);

    // Free Memory on Host
    FreeTestMemory({h_index1, h_index2, h_a}, false);

    SET_PART_OF_2D(time1, h_timeinfo1);
    SET_PART_OF_2D(time2, h_timeinfo2);

    result = hipDeviceReset();
    if (result != hipSuccess) {
        std::cerr << "chkAllRO/423\tError resetting device: " << hipGetErrorString(result) << std::endl;
        *error = 3;
    }
    return ret;
}

#endif //CUDATEST_ALLRO
