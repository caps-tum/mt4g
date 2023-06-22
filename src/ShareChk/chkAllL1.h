#include "hip/hip_runtime.h"

#ifndef CUDATEST_ALL1
#define CUDATEST_ALL1

# include <cstdio>
# include <cstdint>
# include "hip/hip_runtime.h"
# include "../general_functions.h"

#define HARDCODED_3000 3000
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
    __shared__ ALIGN(16) long long s_tvalue1[lessSize];
    __shared__ ALIGN(16) unsigned int s_index1[lessSize];
    __shared__ ALIGN(16) long long s_tvalue2[lessSize];
    __shared__ ALIGN(16) unsigned int s_index2[lessSize];

    __syncthreads();

    if (threadIdx.x == baseCore) {
        for (int k = 0; k < lessSize; k++) {
            s_index1[k] = 0;
            s_tvalue1[k] = 0;
        }
    }

    __syncthreads();

    if (threadIdx.x == testCore) {
        for (int k = 0; k < lessSize; k++) {
            s_index2[k] = 0;
            s_tvalue2[k] = 0;
        }
    }

    unsigned int* ptr;

    __syncthreads();

    if (threadIdx.x == baseCore) {
        for (int k = 0; k < N; k++) {
            NON_TEMPORAL_LOAD_CA(j, array1 + j);
        }
    }

    __syncthreads();

    if (threadIdx.x == testCore) {
        for (int k = 0; k < N; k++) {
            //ptr = array2 + j;
            NON_TEMPORAL_LOAD_CA(j, array2 + j);
        }
    }

    __syncthreads();

    if (threadIdx.x == baseCore) {
        //second round
        for (int k = 0; k < lessSize; k++) {
            ptr = array1 + j;
            LOCAL_CLOCK(start_time);
            NON_TEMPORAL_LOAD_CA(j, ptr);
            s_index1[k] = j;
            LOCAL_CLOCK(end_time);
            s_tvalue1[k] = end_time - start_time;
        }
    }

    __syncthreads();

    if (threadIdx.x == testCore) {
        for (int k = 0; k < lessSize; k++) {
            ptr = array2 + j;
            LOCAL_CLOCK(start_time);
            NON_TEMPORAL_LOAD_CA(j, ptr);
            s_index2[k] = j;
            LOCAL_CLOCK(end_time);
            s_tvalue2[k] = end_time - start_time;
        }
    }

    __syncthreads();

    if (threadIdx.x == baseCore) {
        for (int k = 0; k < lessSize; k++) {
            index1[k] = s_index1[k];
            duration1[k] = s_tvalue1[k];
            if (duration1[k] > HARDCODED_3000) {
                *isDisturbed = true;
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == testCore){
        for (int k = 0; k < lessSize; k++) {
            index2[k] = s_index2[k];
            duration2[k] = s_tvalue2[k];
            if (duration2[k] > HARDCODED_3000) {
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
    hipError_t result = hipDeviceReset();
    if (result != hipSuccess) {
        std::cerr << "ChkAllL1/175\tError resetting device: " << hipGetErrorString(result) << std::endl;
    }
    hipError_t error_id;

    unsigned int *h_index1 = nullptr, *h_index2 = nullptr, *h_timeinfo1 = nullptr, *h_timeinfo2 = nullptr, *h_a = nullptr,
    *duration1 = nullptr, *duration2 = nullptr, *d_index1 = nullptr, *d_index2 = nullptr, *d_a1 = nullptr, *d_a2 = nullptr;
    bool* disturb = nullptr, *d_disturb = nullptr;

    do {
        // Allocate Memory on Host
        h_index1 = (unsigned int *) mallocAndCheck("chkAllL1", sizeof(unsigned int) * lessSize,
                                                   "h_index1", error);

        h_index2 = (unsigned int *) mallocAndCheck("chkAllL1", sizeof(unsigned int) * lessSize,
                                                   "h_index2", error);

        h_timeinfo1 = (unsigned int *) mallocAndCheck("chkAllL1", sizeof(unsigned int) * lessSize,
                                                      "h_timeinfo1", error);

        h_timeinfo2 = (unsigned int *) mallocAndCheck("chkAllL1", sizeof(unsigned int) * lessSize,
                                                      "h_timeinfo2", error);

        disturb = (bool *) mallocAndCheck("chkAllL1", sizeof(bool), "disturb", error);

        h_a = (unsigned int *) mallocAndCheck("chkAllL1", sizeof(unsigned int) * arraySize, "h_a", error);


        // Allocate Memory on GPU
        if (hipMallocAndCheck("chkAllL1", (void **) &duration1,
                              sizeof(unsigned int) * lessSize,
                              "duration1", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("chkAllL1", (void **) &duration2,
                              sizeof(unsigned int) * lessSize,
                              "duration2", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("chkAllL1", (void **) &d_index1,
                              sizeof(unsigned int) * lessSize,
                              "d_index1", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("chkAllL1", (void **) &d_index2,
                              sizeof(unsigned int) * lessSize,
                              "d_index2", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("chkAllL1", (void **) &d_disturb,
                              sizeof(bool),
                              "disturb", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("chkAllRO/212", (void **) &d_a1,
                              sizeof(unsigned int) * arraySize,
                              "d_a1", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("chkAllRO/218", (void **) &d_a2,
                              sizeof(unsigned int) * arraySize,
                              "d_a2", error) != hipSuccess)
            break;

        // Initialize p-chase array
        for (int i = 0; i < arraySize; i++) {
            h_a[i] = (i + 1) % arraySize;
        }

        // Copy array from Host to GPU
        if (hipMemcpyAndCheck("allL1", d_a1, h_a, sizeof(unsigned int) * arraySize,
                              "h_a -> d_a1", error, false) != hipSuccess)
            break;

        if (hipMemcpyAndCheck("roAll", d_a2, h_a, sizeof(unsigned int) * arraySize,
                              "h_a -> d_a2", error, false) != hipSuccess)
            break;
        error_id = hipDeviceSynchronize();

        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[CHKALLL1.CPP]: Error 2 is %s\n", hipGetErrorString(error_id));
            *error = 99;
            break;
        }
        error_id = hipDeviceSynchronize();

        // Launch Kernel function
        dim3 Db = dim3(numberOfCores);
        dim3 Dg = dim3(1, 1, 1);
        hipLaunchKernelGGL(chkTwoCoreL1, Dg, Db, 0, 0, arraySize, d_a1, d_a2, duration1, duration2, d_index1, d_index2, d_disturb, baseCore, testCore);

        error_id = hipDeviceSynchronize();
        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[CHKALLL1.CPP]: Kernel launch/execution Error: %s\n", hipGetErrorString(error_id));
            *error = 5;
            break;
        }

        // Copy results from GPU to Host
        error_id = hipMemcpy((void *) h_timeinfo1, (void *) duration1, sizeof(unsigned int) * lessSize, hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[CHKALLL1.CPP]: hipMemcpy duration1 Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = hipMemcpy((void *) h_timeinfo2, (void *) duration2, sizeof(unsigned int) * lessSize, hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[CHKALLL1.CPP]: hipMemcpy duration2 Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = hipMemcpy((void *) h_index1, (void *) d_index1, sizeof(unsigned int) * lessSize, hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[CHKALLL1.CPP]: hipMemcpy d_index1 Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = hipMemcpy((void *) h_index2, (void *) d_index2, sizeof(unsigned int) * lessSize, hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[CHKALLL1.CPP]: hipMemcpy d_index2 Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = hipMemcpy((void *) disturb, (void *) d_disturb, sizeof(bool), hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[CHKALLL1.CPP]: hipMemcpy disturb Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        char prefix1[64], prefix2[64];
        snprintf(prefix1, 64, "AllL1_T1_%d_%d", baseCore, testCore);
        snprintf(prefix2, 64, "AllL1_T2_%d_%d", baseCore, testCore);

        createOutputFile((int) arraySize, lessSize, h_index1, h_timeinfo1, avgOut1, potMissesOut1, prefix1);
        createOutputFile((int) arraySize, lessSize, h_index2, h_timeinfo2, avgOut2, potMissesOut2, prefix2);
    } while(false);

    bool ret = false;
    if (disturb != nullptr) {
        ret = *disturb;
        free(disturb);
    }

    // Free Memory on GPU
    FreeTestMemory({d_index1, d_index2, duration1, duration2, d_a1, d_disturb}, true); // on GPU


    // Free Memory on Host
    FreeTestMemory({h_index1, h_index2, h_a}, false); // on CPU

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

    result = hipDeviceReset();
    if (result != hipSuccess) {
        std::cerr << "ChkAllL1\tError resetting device: " << hipGetErrorString(result) << std::endl;
    }
    return ret;
}


#endif //CUDATEST_ALL1
