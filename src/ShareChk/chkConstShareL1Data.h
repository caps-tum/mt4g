#include "hip/hip_runtime.h"

#ifndef CUDATEST_CONSTSHAREDATA
#define CUDATEST_CONSTSHAREDATA

# include <cstdio>
# include <cstdint>

# include "hip/hip_runtime.h"
# include "../general_functions.h"

#define HARDCODED_3000 3000

__global__ void
chkConstShareData(unsigned int ConstN, unsigned int DataN, unsigned int *my_array, unsigned int *durationConst,
                  unsigned int *durationData, unsigned int *indexConst, unsigned int *indexData,
                  bool *isDisturbed) {
    *isDisturbed = false;

    unsigned int start_time, end_time;
    unsigned int j = 0;
    __shared__ ALIGN(16) long long s_tvalueConst[lessSize];
    __shared__ ALIGN(16) unsigned int s_indexConst[lessSize];
    __shared__ ALIGN(16) long long s_tvalueData[lessSize];
    __shared__ ALIGN(16) unsigned int s_indexData[lessSize];

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < lessSize; k++) {
            s_indexConst[k] = 0;
            s_tvalueConst[k] = 0;
        }
    }

    __syncthreads();

    if (threadIdx.x == 1) {
        for (int k = 0; k < lessSize; k++) {
            s_indexData[k] = 0;
            s_tvalueData[k] = 0;
        }
    }

    unsigned int *ptr;
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
            NON_TEMPORAL_LOAD_CA(j, ptr);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < lessSize; k++) {
            LOCAL_CLOCK(start_time);
            j = arr[j];
            s_indexConst[k] = j;
            LOCAL_CLOCK(end_time);
            j = j % ConstN;
            s_tvalueConst[k] = end_time - start_time;
        }
    }

    __syncthreads();

    if (threadIdx.x == 1) {
        for (int k = 0; k < lessSize; k++) {
            ptr = my_array + j;
            LOCAL_CLOCK(start_time);
            NON_TEMPORAL_LOAD_CA(j, ptr);
            LOCAL_CLOCK(end_time);
            s_tvalueData[k] = end_time-start_time;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < lessSize; k++) {
            indexConst[k] = s_indexConst[k];
            durationConst[k] = s_tvalueConst[k];
            if (durationConst[k] > HARDCODED_3000) {
                *isDisturbed = true;
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == 1) {
        for (int k = 0; k < lessSize; k++) {
            indexData[k] = s_indexData[k];
            durationData[k] = s_tvalueData[k];
            if (durationData[k] > HARDCODED_3000) {
                *isDisturbed = true;
            }
        }
    };
}

bool
launchL1DataBenchmarkReferenceValue(int N, int stride, double *avgOut, unsigned int *potMissesOut, unsigned int **time,
                                    int *error) {
    return launchL1KernelBenchmark(N, stride, avgOut, potMissesOut, time, error);
}

bool
launchConstBenchmarkReferenceValue(int N, double *avgOut, unsigned int *potMissesOut, unsigned int **time, int *error) {
    return launchConstantBenchmarkR1(N, avgOut, potMissesOut, time, error);
}

bool launchBenchmarkChkConstShareData(unsigned int ConstN, unsigned int DataN, double *avgOutConst, double *avgOutData,
                                      unsigned int *potMissesOutConst,
                                      unsigned int *potMissesOutData, unsigned int **timeConst, unsigned int **timeData,
                                      int *error) {
    hipError_t error_id;
    error_id = hipDeviceReset();

    unsigned int *h_indexConst = nullptr, *h_indexData = nullptr, *h_timeinfoConst = nullptr, *h_timeinfoData = nullptr, *h_a = nullptr,
            *durationConst = nullptr, *durationData = nullptr, *d_indexConst = nullptr, *d_indexData = nullptr, *d_a = nullptr;
    bool *disturb = nullptr, *d_disturb = nullptr;

    do {
        // Allocate Memory on Host
        h_indexConst = (unsigned int *) mallocAndCheck("chkConstShareL1Data", sizeof(unsigned int) * lessSize,
                                                       "h_indexConst", error);

        h_indexData = (unsigned int *) mallocAndCheck("chkConstShareL1Data", sizeof(unsigned int) * lessSize,
                                                      "h_indexData", error);

        h_timeinfoConst = (unsigned int *) mallocAndCheck("chkConstShareL1Data", sizeof(unsigned int) * lessSize,
                                                          "h_timeinfoConst", error);

        h_timeinfoData = (unsigned int *) mallocAndCheck("chkConstShareL1Data", sizeof(unsigned int) * lessSize,
                                                         "h_timeinfoData", error);

        disturb = (bool *) mallocAndCheck("chkConstShareL1Data", sizeof(bool), "disturb", error);

        h_a = (unsigned int *) mallocAndCheck("chkConstShareL1Data", sizeof(unsigned int) *
        (DataN), "h_a", error);

        // Allocate Memory on GPU
        if (hipMallocAndCheck("chkConstShareL1Data", (void **) &durationConst,
                              sizeof(unsigned int) * lessSize,
                              "durationConst", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("chkConstShareL1Data", (void **) &durationData,
                              sizeof(unsigned int) * lessSize,
                              "durationData", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("chkConstShareL1Data", (void **) &d_indexConst,
                              sizeof(unsigned int) * lessSize,
                              "d_indexConst", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("chkConstShareL1Data", (void **) &d_indexData,
                              sizeof(unsigned int) * lessSize,
                              "d_indexData", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("chkConstShareL1Data", (void **) &d_disturb,
                              sizeof(bool),
                              "d_disturb", error) != hipSuccess)
            break;
        if (hipMallocAndCheck("chkConstShareL1Data", (void **) &d_a,
                              sizeof(unsigned int) * (DataN),
                              "d_a", error) != hipSuccess)
            break;

        // Initialize p-chase array
        for (int i = 0; i < DataN; i++) {
            h_a[i] = (i + 1) % DataN;
        }

        // Copy array from Host to GPU
        if (hipMemcpyAndCheck("chkConstShareL1Data", d_a, h_a, sizeof(unsigned int) * DataN,
                              "h_a -> d_a", error, false) != hipSuccess)
            break;

        error_id = hipDeviceSynchronize();

        // Launch Kernel function
        dim3 Db = dim3(2);
        dim3 Dg = dim3(1, 1, 1);
        hipLaunchKernelGGL(chkConstShareData, Dg, Db, 0, 0, ConstN, DataN, d_a, durationConst, durationData,
                           d_indexConst, d_indexData, d_disturb);

        error_id = hipDeviceSynchronize();
        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[CHKCONSTSHAREL1DATA.CPP]: Kernel launch/execution Error: %s\n", hipGetErrorString(error_id));
            *error = 5;
            break;
        }

        // Copy results from GPU to Host
        if (hipMemcpyAndCheck("chkConstShareL1Data", h_timeinfoConst, durationConst, sizeof(unsigned int) * lessSize,
                              "durationConst -> h_timeinfoConst", error, true) != hipSuccess)
            break;
        if (hipMemcpyAndCheck("chkConstShareL1Data", h_timeinfoData, durationData, sizeof(unsigned int) * lessSize,
                              "durationData -> h_timeinfoData", error, true) != hipSuccess)
            break;

        if (hipMemcpyAndCheck("chkConstShareL1Data", h_indexConst, d_indexConst, sizeof(unsigned int) * lessSize,
                              "d_indexConst -> h_indexConst", error, true) != hipSuccess)
            break;

        if (hipMemcpyAndCheck("chkConstShareL1Data", h_indexData, d_indexData, sizeof(unsigned int) * lessSize,
                              "d_indexData -> h_indexData", error, true) != hipSuccess)
            break;

        if (hipMemcpyAndCheck("chkConstShareL1Data", disturb, d_disturb, sizeof(bool), "d_disturb -> disturb", error,
                              true) != hipSuccess)



        createOutputFile((int) ConstN, lessSize, h_indexConst, h_timeinfoConst, avgOutConst, potMissesOutConst,
                         "ShareConstDataConst_");
        createOutputFile((int) DataN, lessSize, h_indexData, h_timeinfoData, avgOutData, potMissesOutData,
                         "ShareConstDataData_");
    } while (false);

    bool ret = false;
    if (disturb != nullptr) {
        ret = *disturb;
        free(disturb);
    }

    // Free Memory on GPU
    FreeTestMemory({d_a, d_indexConst, d_indexData, durationConst, durationData, d_disturb}, true);

    // Free Memory on Host
    FreeTestMemory({h_indexConst, h_indexData}, false);

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

    error_id = hipDeviceReset();
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

#define HARDCODED_NUMBER_OF_REPEATS 3

dTuple measure_ConstShareData(unsigned int measuredSizeConstL1, unsigned int measuredSizeDataL1, unsigned int sub) {
    unsigned int ConstSizeInInt = (measuredSizeConstL1 - sub) >> 2; // / 4;
    unsigned int L1DataSizeInInt = (measuredSizeDataL1 - sub) >> 2; // / 4;

    double *avgFlowRefConst = (double *) malloc(sizeof(double));
    unsigned int *potMissesFlowRefConst = (unsigned int *) malloc(sizeof(unsigned int));
    unsigned int **timeRefConst = (unsigned int **) malloc(sizeof(unsigned int *));

    double *avgFlowRefL1 = (double *) malloc(sizeof(double));
    unsigned int *potMissesFlowRefL1 = (unsigned int *) malloc(sizeof(unsigned int));
    unsigned int **timeRefL1 = (unsigned int **) malloc(sizeof(unsigned int *));

    double *avgFlow = (double *) malloc(sizeof(double) * 2);
    unsigned int *potMissesFlow = (unsigned int *) malloc(sizeof(unsigned int) * 2);
    unsigned int **time = (unsigned int **) malloc(sizeof(unsigned int *) * 2);
    if (avgFlowRefConst == nullptr || potMissesFlowRefConst == nullptr || timeRefConst == nullptr ||
        avgFlowRefL1 == nullptr || potMissesFlowRefL1 == nullptr || timeRefL1 == nullptr ||
        avgFlow == nullptr || potMissesFlow == nullptr || time == nullptr) {
        FreeMeasureConstL1ResOnlyPtr()
        printErrorCodeInformation(1);
        exit(1);
    }

    timeRefConst[0] = timeRefL1[0] = time[0] = time[1] = nullptr;

    bool dist = true;
    int n = HARDCODED_NUMBER_OF_REPEATS;
    while (dist && n > 0) {
        int error = 0;
        dist = launchConstBenchmarkReferenceValue((int) ConstSizeInInt, avgFlowRefConst, potMissesFlowRefConst,
                                                  timeRefConst, &error);
        if (error != 0) {
            FreeMeasureConstL1Resources()
            printErrorCodeInformation(error);
            exit(error);
        }
        --n;
    }

    dist = true;
    n = HARDCODED_NUMBER_OF_REPEATS;
    while (dist && n > 0) {
        int error = 0;
        dist = launchL1DataBenchmarkReferenceValue((int) L1DataSizeInInt, 1, avgFlowRefL1, potMissesFlowRefL1,
                                                   timeRefL1, &error);
        if (error != 0) {
            FreeMeasureConstL1Resources()
            printErrorCodeInformation(error);
            exit(error);
        }
        --n;
    }

    dist = true;
    n = HARDCODED_NUMBER_OF_REPEATS;
    while (dist && n > 0) {
        int error = 0;
        dist = launchBenchmarkChkConstShareData(ConstSizeInInt, L1DataSizeInInt, &avgFlow[0], &avgFlow[1],
                                                &potMissesFlow[0],
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
