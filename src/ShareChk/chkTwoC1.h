#include "hip/hip_runtime.h"

#ifndef CUDATEST_TWOCONST1
#define CUDATEST_TWOCONST1

# include <cstdio>
# include <cstdint>

# include "hip/hip_runtime.h"
#include "../general_functions.h"
__global__ void chkTwoC1(unsigned int N, unsigned int *durationC1_1, unsigned int * durationC1_2, unsigned int *indexC1_1, unsigned int *indexC1_2,
                         bool* isDisturbed) {
    *isDisturbed = false;

    unsigned int start_time, end_time;
    unsigned int j = 0;
    __shared__ ALIGN(16) long long s_tvalueC1_1[lessSize];
    __shared__ ALIGN(16) unsigned int s_indexC1_1[lessSize];
    __shared__ ALIGN(16) long long s_tvalueC1_2[lessSize];
    __shared__ ALIGN(16) unsigned int s_indexC1_2[lessSize];

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < lessSize; k++) {
            s_indexC1_1[k] = 0;
            s_tvalueC1_1[k] = 0;
        }
        j = 0;
    }

    __syncthreads();

    if (threadIdx.x == 1){
        for (int k = 0; k < lessSize; k++) {
            s_indexC1_2[k] = 0;
            s_tvalueC1_2[k] = 0;
        }
        j = N;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < N; k++) {
            j = arr[j];
            j = j % N;
        }
    }

    __syncthreads();

    if (threadIdx.x == 1) {
        for (int k = 0; k < N; k++) {
            j = arr[j];
            j = (j % N) + N;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        //second round
        for (int k = 0; k < lessSize; k++) {
            start_time = clock();
            j = arr[j];
            s_indexC1_1[k] = j;
            end_time = clock();
            j = j % N;
            s_tvalueC1_1[k] = (end_time - start_time);
        }
    }

    __syncthreads();

    if (threadIdx.x == 1) {
        for (int k = 0; k < lessSize; k++) {
            start_time = clock();
            j = arr[j];
            s_indexC1_2[k] = j;
            end_time = clock();
            j = (j % N) + N;
            s_tvalueC1_2[k] = (end_time - start_time);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < lessSize; k++) {
            indexC1_1[k] = s_indexC1_1[k];
            durationC1_1[k] = s_tvalueC1_1[k];
            if (durationC1_1[k] > 3000) {
                *isDisturbed = true;
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == 1) {
        for (int k = 0; k < lessSize; k++) {
            indexC1_2[k] = s_indexC1_2[k];
            durationC1_2[k] = s_tvalueC1_2[k];
            if (durationC1_2[k] > 3000) {
                *isDisturbed = true;
            }
        }
    }
}

bool launchBenchmarkTwoC1(unsigned int N, double *avgOut1, double* avgOut2, unsigned int* potMissesOut1, unsigned int* potMissesOut2,
                          unsigned int **time1, unsigned int **time2, int* error) {
    hipError_t error_id;
    error_id = hipDeviceReset();

    unsigned int *h_indexC1_1 = nullptr, *h_indexC1_2 = nullptr, *h_timeinfoC1_1 = nullptr, *h_timeinfoC1_2 = nullptr,
    *durationC1_1 = nullptr, *durationC1_2 = nullptr, *d_indexC1_1 = nullptr, *d_indexC1_2 = nullptr;
    bool *disturb = nullptr, *d_disturb = nullptr;

    do {
        // Allocate Memory on Host
        h_indexC1_1 = (unsigned int *) malloc(sizeof(unsigned int) * lessSize);
        if (h_indexC1_1 == nullptr) {
            printf("[CHKTWOC1.CPP]: malloc h_indexC1_1 Error\n");
            *error = 1;
            break;
        }

        h_indexC1_2 = (unsigned int *) malloc(sizeof(unsigned int) * lessSize);
        if (h_indexC1_2 == nullptr) {
            printf("[CHKTWOC1.CPP]: malloc h_indexC1_2 Error\n");
            *error = 1;
            break;
        }

        h_timeinfoC1_1 = (unsigned int *) malloc(sizeof(unsigned int) * lessSize);
        if (h_timeinfoC1_1 == nullptr) {
            printf("[CHKTWOC1.CPP]: malloc h_timeinfoC1_1 Error\n");
            *error = 1;
            break;
        }

        h_timeinfoC1_2 = (unsigned int *) malloc(sizeof(unsigned int) * lessSize);
        if (h_timeinfoC1_2 == nullptr) {
            printf("[CHKTWOC1.CPP]: malloc h_timeinfoC1_2 Error\n");
            *error = 1;
            break;
        }

        disturb = (bool *) malloc(sizeof(bool));
        if (disturb == nullptr) {
            printf("[CHKTWOC1.CPP]: malloc disturb Error\n");
            *error = 1;
            break;
        }

        // Allocate Memory on GPU
        error_id = hipMalloc((void **) &durationC1_1, sizeof(unsigned int) * lessSize);
        if (error_id != hipSuccess) {
            printf("[CHKTWOC1.CPP]: hipMalloc durationC1_1 Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = hipMalloc((void **) &durationC1_2, sizeof(unsigned int) * lessSize);
        if (error_id != hipSuccess) {
            printf("[CHKTWOC1.CPP]: hipMalloc durationC1_2 Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = hipMalloc((void **) &d_indexC1_1, sizeof(unsigned int) * lessSize);
        if (error_id != hipSuccess) {
            printf("[CHKTWOC1.CPP]: hipMalloc d_indexC1_1 Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = hipMalloc((void **) &d_indexC1_2, sizeof(unsigned int) * lessSize);
        if (error_id != hipSuccess) {
            printf("[CHKTWOC1.CPP]: hipMalloc d_indexC1_2 Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = hipMalloc((void **) &d_disturb, sizeof(bool));
        if (error_id != hipSuccess) {
            printf("[CHKTWOC1.CPP]: hipMalloc disturb Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = hipDeviceSynchronize();

        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[CHKTWOC1.CPP]: hipDeviceSynchronize Error: %s\n", hipGetErrorString(error_id));
            *error = 99;
            break;
        }
        error_id = hipDeviceSynchronize();

        // Launch Kernel function
        dim3 Db = dim3(2);
        dim3 Dg = dim3(1, 1, 1);
        hipLaunchKernelGGL(chkTwoC1, Dg, Db, 0, 0, N, durationC1_1, durationC1_2, d_indexC1_1, d_indexC1_2, d_disturb);

        error_id = hipDeviceSynchronize();
        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[CHKTWOC1.CPP]: Kernel launch/execution Error: %s\n", hipGetErrorString(error_id));
            *error = 5;
            break;
        }

        // Copy results from GPU to Host
        error_id = hipMemcpy((void *) h_timeinfoC1_1, (void *) durationC1_1, sizeof(unsigned int) * lessSize,
                              hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[CHKTWOC1.CPP]: hipMemcpy durationC1_1 Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = hipMemcpy((void *) h_timeinfoC1_2, (void *) durationC1_2, sizeof(unsigned int) * lessSize,
                              hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[CHKTWOC1.CPP]: hipMemcpy durationC1_2 Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = hipMemcpy((void *) h_indexC1_1, (void *) d_indexC1_1, sizeof(unsigned int) * lessSize,
                              hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[CHKTWOC1.CPP]: hipMemcpy d_indexC1_1 Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = hipMemcpy((void *) h_indexC1_2, (void *) d_indexC1_2, sizeof(unsigned int) * lessSize,
                              hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[CHKTWOC1.CPP]: hipMemcpy d_indexC1_2 Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = hipMemcpy((void *) disturb, (void *) d_disturb, sizeof(bool), hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[CHKTWOC1.CPP]: hipMemcpy disturb Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        createOutputFile((int) N, lessSize, h_indexC1_1, h_timeinfoC1_1, avgOut1, potMissesOut1, "TWOC1_1_");
        createOutputFile((int) N, lessSize, h_indexC1_2, h_timeinfoC1_2, avgOut2, potMissesOut2, "TWOC1_2_");
    } while(false);

    bool ret = false;
    if (disturb != nullptr) {
        ret = *disturb;
        free(disturb);
    }

    // Free Memory on GPU
    FreeTestMemory({durationC1_1, durationC1_2, d_indexC1_1, d_indexC1_2, d_disturb}, true);

    // Free Memory on Host
    FreeTestMemory({h_indexC1_1, h_indexC1_2}, false);

    SET_PART_OF_2D(time1, h_timeinfoC1_1);
    SET_PART_OF_2D(time2, h_timeinfoC1_2);

    error_id = hipDeviceReset();
    return ret;
}

#define FreeMeasureTwoC1ResOnlyPtr() \
free(time);                         \
free(timeRef);                      \
free(avgFlow);                      \
free(potMissesFlow);                \
free(avgFlowRef);                   \
free(potMissesFlowRef);             \

#define FreeMeasureTwoC1Resources() \
if (time[0] != nullptr) {           \
    free(time[0]);                  \
}                                   \
if (time[1] != nullptr) {           \
    free(time[1]);                  \
}                                   \
if (timeRef[0] != nullptr) {        \
    free(timeRef[0]);               \
}                                   \
free(time);                         \
free(timeRef);                      \
free(avgFlow);                      \
free(potMissesFlow);                \
free(avgFlowRef);                   \
free(potMissesFlowRef);             \

unsigned int measure_TwoC1(unsigned int measuredSizeC1, unsigned int sub) {
    unsigned int C1SizeInInt = (measuredSizeC1 - sub) / 4;

    double* avgFlowRef = (double*) malloc(sizeof(double));
    unsigned int *potMissesFlowRef = (unsigned int*) malloc(sizeof(unsigned int));
    unsigned int** timeRef = (unsigned int**) malloc(sizeof(unsigned int*));

    double* avgFlow = (double*) malloc(sizeof(double)  * 2);
    unsigned int *potMissesFlow = (unsigned int*) malloc(sizeof(unsigned int) * 2);
    unsigned int** time = (unsigned int**) malloc(sizeof(unsigned int*) * 2);
    if (avgFlowRef == nullptr || potMissesFlowRef == nullptr || timeRef == nullptr ||
        avgFlow == nullptr || potMissesFlow == nullptr || time == nullptr) {
        FreeMeasureTwoC1ResOnlyPtr()
        printErrorCodeInformation(1);
        exit(1);
    }

    timeRef[0] = time[0] = time[1] = nullptr;

    bool dist = true; int n = 5;
    while(dist && n > 0) {
        int error = 0;
        dist = launchConstBenchmarkReferenceValue((int) C1SizeInInt, avgFlowRef, potMissesFlowRef, timeRef, &error);
        if (error != 0) {
            FreeMeasureTwoC1Resources()
            printErrorCodeInformation(error);
            exit(error);
        }
        --n;
    }

    dist = true;
    n = 5;
    while(dist && n > 0) {
        int error = 0;
        dist = launchBenchmarkTwoC1(C1SizeInInt, &avgFlow[0], &avgFlow[1], &potMissesFlow[0], &potMissesFlow[1], &time[0],
                                    &time[1], &error);
        if (error != 0) {
            FreeMeasureTwoC1Resources()
            printErrorCodeInformation(error);
            exit(error);
        }
        --n;
    }
#ifdef IsDebug
    fprintf(out, "Measured C1 Avg in clean execution: %f\n", avgFlowRef[0]);

    fprintf(out, "Measured C1 Avg While Shared With C1_2:  %f\n", avgFlow[0]);
    fprintf(out, "Measured C1 Pot Misses While Shared With C1_2:  %u\n", potMissesFlow[0]);

    fprintf(out, "Measured C1_2 Avg While Shared With C1:  %f\n", avgFlow[1]);
    fprintf(out, "Measured C1_2 Pot Misses While Shared With C1:  %u\n", potMissesFlow[1]);
#endif //IsDebug

    unsigned int result = std::max(potMissesFlow[0], potMissesFlow[1]);;

    FreeMeasureTwoC1Resources()

    return result;
}


#endif //CUDATEST_TWOCONST1
