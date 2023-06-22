#include "hip/hip_runtime.h"

#ifndef CUDATEST_ROSHAREDATA
#define CUDATEST_ROSHAREDATA

# include <cstdio>
# include <cstdint>

# include "hip/hip_runtime.h"
# include "../general_functions.h"

#define HARDCODED_3000 3000

__global__ void chkROShareL1Data(unsigned int RON, unsigned int DataN, const unsigned int *__restrict__ myArrayReadOnly,
                                 unsigned int *my_array,
                                 unsigned int *durationRO, unsigned int *durationData, unsigned int *indexRO,
                                 unsigned int *indexData,
                                 bool *isDisturbed) {
    *isDisturbed = false;

    unsigned int start_time, end_time;
    unsigned int j = 0;
    __shared__ ALIGN(16) long long s_tvalueRO[lessSize];
    __shared__ ALIGN(16) unsigned int s_indexRO[lessSize];
    __shared__ ALIGN(16) long long s_tvalueData[lessSize];
    __shared__ ALIGN(16) unsigned int s_indexData[lessSize];

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < lessSize; k++) {
            s_indexRO[k] = 0;
            s_tvalueRO[k] = 0;
        }
    }

    __syncthreads();

    if (threadIdx.x == 1) {
        for (int k = 0; k < lessSize; k++) {
            s_indexData[k] = 0;
            s_tvalueData[k] = 0;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < RON; k++)
            j = __ldg(&myArrayReadOnly[j]);
    }

    unsigned int *ptr;
    __syncthreads();

    if (threadIdx.x == 1) {
        for (int k = 0; k < DataN; k++) {
            ptr = my_array + j;
            NON_TEMPORAL_LOAD_CA(j, ptr);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        //second round
        for (int k = 0; k < lessSize; k++) {
            LOCAL_CLOCK(start_time);
            j = __ldg(&myArrayReadOnly[j]);
            s_indexRO[k] = j;
            LOCAL_CLOCK(end_time);
            s_tvalueRO[k] = end_time - start_time;
        }
    }

    __syncthreads();

    if (threadIdx.x == 1) {
        for (int k = 0; k < lessSize; k++) {
            ptr = my_array + j;
            LOCAL_CLOCK(start_time);
            NON_TEMPORAL_LOAD_CA(j, ptr);
            LOCAL_CLOCK(end_time);
            s_tvalueData[k] = end_time - start_time;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < lessSize; k++) {
            indexRO[k] = s_indexRO[k];
            durationRO[k] = s_tvalueRO[k];
            if (durationRO[k] > HARDCODED_3000) {
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
    }
}

bool launchBenchmarkChkROShareL1Data(unsigned int RO_N, unsigned int DataN, double *avgOutRO, double *avgOutData,
                                     unsigned int *potMissesOutRO,
                                     unsigned int *potMissesOutData, unsigned int **timeRO, unsigned int **timeData,
                                     int *error) {
    hipError_t error_id;
    error_id = hipDeviceReset();

    unsigned int *h_indexRO = nullptr, *h_indexData = nullptr, *h_timeinfoRO = nullptr, *h_timeinfoData = nullptr, *h_aData = nullptr, *h_aRO = nullptr,
            *durationRO = nullptr, *durationData = nullptr, *d_indexRO = nullptr, *d_indexData = nullptr, *d_aRO = nullptr, *d_aData = nullptr;
    bool *disturb = nullptr, *d_disturb = nullptr;

    do {
        // Allocate Memory on Host
        h_indexRO = (unsigned int *) mallocAndCheck("chkROShareL1Data", sizeof(unsigned int) * lessSize,
                                                    "h_indexRO", error);

        h_indexData = (unsigned int *) mallocAndCheck("chkROShareL1Data", sizeof(unsigned int) * lessSize,
                                                      "h_indexData", error);

        h_timeinfoRO = (unsigned int *) mallocAndCheck("chkROShareL1Data", sizeof(unsigned int) * lessSize,
                                                        "h_timeinfoRO", error);

        h_timeinfoData = (unsigned int *) mallocAndCheck("chkROShareL1Data", sizeof(unsigned int) * lessSize,
                                                         "h_timeinfoData", error);

        disturb = (bool *) mallocAndCheck("chkROShareL1Data", sizeof(bool), "disturb", error);


        h_aData = (unsigned int *) mallocAndCheck("chkROShareL1Data", sizeof(unsigned int) * DataN,
                                                  "h_aData", error);

        h_aRO = (unsigned int *) mallocAndCheck("chkROShareL1Data", sizeof(unsigned int) * RO_N,
                                                "h_aRO", error);

        // Allocate Memory on GPU
        if (hipMallocAndCheck("chkROShareL1Data", (void **) &durationRO,
                              sizeof(unsigned int) * lessSize,
                              "durationRO", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("chkROShareL1Data", (void **) &durationData,
                              sizeof(unsigned int) * lessSize,
                              "durationData", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("chkROShareL1Data", (void **) &d_indexRO,
                              sizeof(unsigned int) * lessSize,
                              "d_indexRO", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("chkROShareL1Data", (void **) &d_indexData,
                              sizeof(unsigned int) * lessSize,
                              "d_indexData", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("chkROShareL1Data", (void **) &d_disturb,
                              sizeof(bool),
                              "d_disturb", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("chkROShareL1Data", (void **) &d_aData,
                              sizeof(unsigned int) * (DataN),
                              "d_aData", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("chkROShareL1Data", (void **) &d_aRO,
                              sizeof(unsigned int) * (RO_N),
                              "d_aRO", error) != hipSuccess)
            break;

        // Initialize p-chase arrays
        for (int i = 0; i < DataN; i++) {
            h_aData[i] = (i + 1) % DataN;
        }

        for (int i = 0; i < RO_N; i++) {
            h_aRO[i] = (i + 1) % RO_N;
        }

        // Copy arrays from Host to GPU
        if (hipMemcpyAndCheck("chkROShareL1Data", d_aData, h_aData, sizeof(unsigned int) * DataN,
                              "h_aData -> d_aData", error, false) != hipSuccess)
            break;

        if (hipMemcpyAndCheck("chkROShareL1Data", d_aRO, h_aRO, sizeof(unsigned int) * RO_N,
                              "h_aRO -> d_aRO", error, false) != hipSuccess)
            break;

        error_id = hipDeviceSynchronize();

        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[CHKROSHAREL1DATA.CPP]: hipDeviceSynchronize Error: %s\n", hipGetErrorString(error_id));
            *error = 99;
            break;
        }
        error_id = hipDeviceSynchronize();

        // Launch Kernel function
        dim3 Db = dim3(2);
        dim3 Dg = dim3(1, 1, 1);
        hipLaunchKernelGGL(chkROShareL1Data, Dg, Db, 0, 0, RO_N, DataN, d_aRO, d_aData, durationRO, durationData,
                           d_indexRO, d_indexData, d_disturb);

        error_id = hipDeviceSynchronize();
        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[CHKROSHAREL1DATA.CPP]: Kernel launch/execution Error: %s\n", hipGetErrorString(error_id));
            *error = 5;
            break;
        }

        // Copy results from GPU to Host
        if (hipMemcpyAndCheck("chkROShareL1Data", h_timeinfoRO, durationRO, sizeof(unsigned int) * lessSize,
                              "durationRO -> h_timeinfoRO", error, true) != hipSuccess)
            break;

        if (hipMemcpyAndCheck("chkROShareL1Data", h_timeinfoData, durationData, sizeof(unsigned int) * lessSize,
                              "durationData -> h_timeinfoData", error, true) != hipSuccess)
            break;

        if (hipMemcpyAndCheck("chkROShareL1Data", h_indexRO, d_indexRO, sizeof(unsigned int) * lessSize,
                              "d_indexRO -> h_indexRO", error, true) != hipSuccess)
            break;

        if (hipMemcpyAndCheck("chkROShareL1Data", h_indexData, d_indexData, sizeof(unsigned int) * lessSize,
                              "d_indexData -> h_indexRO", error, true) != hipSuccess)
            break;

        if (hipMemcpyAndCheck("chkROShareL1Data", disturb, d_disturb, sizeof(bool),
                              "d_disturb -> disturb", error, true) != hipSuccess)
            break;


        createOutputFile((int) RO_N, lessSize, h_indexRO, h_timeinfoRO, avgOutRO, potMissesOutRO, "ShareRODataRO_");
        createOutputFile((int) DataN, lessSize, h_indexData, h_timeinfoData, avgOutData, potMissesOutData,
                         "ShareRODataData_");
    } while (false);

    bool ret = false;
    if (disturb != nullptr) {
        ret = *disturb;
        free(disturb);
    }

    // Free Memory on GPU
    FreeTestMemory({d_indexRO, d_indexData, durationRO, durationData, d_aData, d_aRO, d_disturb}, true);

    // Free Memory on Host
    FreeTestMemory({h_indexRO, h_indexData, h_aData, h_aRO}, false);

    SET_PART_OF_2D(timeRO, h_timeinfoRO);
    SET_PART_OF_2D(timeData, h_timeinfoData);

    error_id = hipDeviceReset();
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
    unsigned int ROSizeInInt = (measuredSizeRO - sub) >> 2; // / 4;
    unsigned int L1DataSizeInInt = (measuredSizeData - sub) >> 2; // / 4;

    double *avgFlowRefRO = (double *) malloc(sizeof(double));
    unsigned int *potMissesFlowRefRO = (unsigned int *) malloc(sizeof(unsigned int));
    unsigned int **timeRefRO = (unsigned int **) malloc(sizeof(unsigned int *));

    double *avgFlowRefL1 = (double *) malloc(sizeof(double));
    unsigned int *potMissesFlowRefL1 = (unsigned int *) malloc(sizeof(unsigned int));
    unsigned int **timeRefL1 = (unsigned int **) malloc(sizeof(unsigned int *));

    double *avgFlow = (double *) malloc(sizeof(double) * 2);
    unsigned int *potMissesFlow = (unsigned int *) malloc(sizeof(unsigned int) * 2);
    unsigned int **time = (unsigned int **) malloc(sizeof(unsigned int *) * 2);
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
        dist = launchROBenchmarkReferenceValue((int) ROSizeInInt, 1, avgFlowRefRO, potMissesFlowRefRO, timeRefRO,
                                               &error);
        if (error != 0) {
            FreeMeasureROL1Resources()
            printErrorCodeInformation(error);
            exit(error);
        }
        --n;
    }

    dist = true;
    n = 5;
    while (dist && n > 0) {
        int error = 0;
        dist = launchL1DataBenchmarkReferenceValue((int) L1DataSizeInInt, 1, avgFlowRefL1, potMissesFlowRefL1,
                                                   timeRefL1, &error);
        if (error != 0) {
            FreeMeasureROL1Resources()
            printErrorCodeInformation(error);
            exit(error);
        }
        --n;
    }

    dist = true;
    n = 5;
    while (dist && n > 0) {
        int error = 0;
        dist = launchBenchmarkChkROShareL1Data(ROSizeInInt, L1DataSizeInInt, &avgFlow[0], &avgFlow[1],
                                               &potMissesFlow[0],
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
