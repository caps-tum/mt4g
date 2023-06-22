#include "hip/hip_runtime.h"

#ifndef CUDATEST_ROSHARETEXTURE
#define CUDATEST_ROSHARETEXTURE

# include <cstdio>
# include <cstdint>

# include "hip/hip_runtime.h"
#include "../general_functions.h"

__global__ void chkROShareTexture(hipTextureObject_t tex, unsigned int RON, unsigned int TextureN,
                                  const unsigned int *__restrict__ myArrayReadOnly,
                                  unsigned int *durationRO, unsigned int *durationTexture, unsigned int *indexRO,
                                  unsigned int *indexTexture,
                                  bool *isDisturbed) {
    *isDisturbed = false;

    unsigned int start_time, end_time;
    unsigned int j = 0;
    int j2 = 0;
    __shared__ ALIGN(16) long long s_tvalueRO[lessSize];
    __shared__ ALIGN(16) unsigned int s_indexRO[lessSize];
    __shared__ ALIGN(16) long long s_tvalueTexture[lessSize];
    __shared__ ALIGN(16) unsigned int s_indexTexture[lessSize];

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
            s_indexTexture[k] = 0;
            s_tvalueTexture[k] = 0;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < RON; k++)
            j = __ldg(&myArrayReadOnly[j]);
    }

    __syncthreads();

    if (threadIdx.x == 1) {
        for (int k = 0; k < TextureN; k++) {
            j2 = tex1Dfetch<int>(tex, j2);
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
        for (int k = 0; k < lessSize; k++) {
            start_time = clock();
            j2 = tex1Dfetch<int>(tex, j2);
            s_indexTexture[k] = j2;
            end_time = clock();
            s_tvalueTexture[k] = (end_time - start_time);
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
        for (int k = 0; k < lessSize; k++) {
            indexTexture[k] = s_indexTexture[k];
            durationTexture[k] = s_tvalueTexture[k];
            if (durationTexture[k] > 3000) {
                *isDisturbed = true;
            }
        }
    }
}

bool launchROBenchmarkReferenceValue(int N, int stride, double *avgOut, unsigned int *potMissesOut, unsigned int **time,
                                     int *error) {
    return launchROBenchmark(N, stride, avgOut, potMissesOut, time, error);
}

bool
launchTextureBenchmarkReferenceValue(int N, int stride, double *avgOut, unsigned int *potMissesOut, unsigned int **time,
                                     int *error) {
    return launchTextureBenchmark(N, stride, avgOut, potMissesOut, time, error);
}

bool launchBenchmarkChkROShareTexture(unsigned int RO_N, unsigned int TextureN, double *avgOutRO, double *avgOutTexture,
                                      unsigned int *potMissesOutRO,
                                      unsigned int *potMissesOutTexture, unsigned int **timeRO,
                                      unsigned int **timeTexture, int *error) {
    hipError_t error_id;
    error_id = hipDeviceReset();

    int *h_aTexture = nullptr, *d_aTexture = nullptr;
    unsigned int *h_indexRO = nullptr, *h_indexTexture = nullptr, *h_timeinfoRO = nullptr, *h_timeinfoTexture = nullptr, *h_aRO = nullptr,
            *durationRO = nullptr, *durationTexture = nullptr, *d_indexRO = nullptr, *d_indexTexture = nullptr, *d_aRO = nullptr;
    bool *disturb = nullptr, *d_disturb = nullptr;
    bool bindedTexture = false;
    hipTextureObject_t tex = 0;

    do {
        // Allocate Memory on Host
        h_indexRO = (unsigned int *) mallocAndCheck("chkROShareTexture", sizeof(unsigned int) * lessSize,
                                                    "h_indexRO", error);

        h_indexTexture = (unsigned int *) mallocAndCheck("chkROShareTexture", sizeof(unsigned int) * lessSize,
                                                         "h_indexTexture", error);

        h_timeinfoRO = (unsigned int *) mallocAndCheck("chkROShareTexture", sizeof(unsigned int) * lessSize,
                                                       "h_timeinfoRO", error);

        h_timeinfoTexture = (unsigned int *) mallocAndCheck("chkROShareTexture", sizeof(unsigned int) * lessSize,
                                                            "h_timeinfoTexture", error);

        disturb = (bool *) mallocAndCheck("chkROShareTexture", sizeof(bool), "disturb", error);

        h_aTexture = (int *) mallocAndCheck("chkROShareTexture", sizeof(unsigned int) * TextureN,
                                            "h_aTexture", error);

        h_aRO = (unsigned int *) mallocAndCheck("chkAllRO/175", sizeof(unsigned int) * RO_N, "h_aRO", error);


        // Allocate Memory on GPU
        if (hipMallocAndCheck("chkROShareTexture", (void **) &durationRO,
                              sizeof(unsigned int) * lessSize,
                              "durationRO", error) != hipSuccess)
            break;


        if (hipMallocAndCheck("chkROShareTexture", (void **) &durationTexture,
                              sizeof(unsigned int) * lessSize,
                              "durationTexture", error) != hipSuccess)
            break;
        if (hipMallocAndCheck("chkROShareTexture", (void **) &d_indexRO,
                              sizeof(unsigned int) * lessSize,
                              "d_indexRO", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("chkROShareTexture", (void **) &d_indexTexture,
                              sizeof(unsigned int) * lessSize,
                              "d_indexTexture", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("chkROShareTexture", (void **) &d_disturb,
                              sizeof(bool),
                              "d_disturb", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("chkROShareTexture", (void **) &d_aTexture,
                              sizeof(int) * (TextureN),
                              "d_aTexture", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("chkROShareTexture", (void **) &d_aRO,
                              sizeof(int) * (RO_N),
                              "d_aRO", error) != hipSuccess)
            break;

        // Initialize p-chase arrays
        for (int i = 0; i < TextureN; i++) {
            h_aTexture[i] = (i + 1) % (int) TextureN;
        }

        for (int i = 0; i < RO_N; i++) {
            h_aRO[i] = (i + 1) % RO_N;
        }

        // Copy results from GPU to Host
        if (hipMemcpyAndCheck("chkROShareTexture", d_aTexture, h_aTexture, sizeof(unsigned int) * TextureN,
                              "d_aTexture -> h_aTexture", error, false) != hipSuccess)
            break;


        /* copy array elements from CPU to GPU */
        if (hipMemcpyAndCheck("chkROShareTexture", d_aRO, h_aRO, sizeof(unsigned int) * RO_N,
                              "d_aRO -> h_aRO", error, false) != hipSuccess)
            break;

        // Create Texture Object
        hipResourceDesc resDesc = {};
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = hipResourceTypeLinear;
        resDesc.res.linear.devPtr = d_aTexture;
        resDesc.res.linear.desc.f = hipChannelFormatKindSigned;
        resDesc.res.linear.desc.x = 32; // bits per channel
        resDesc.res.linear.sizeInBytes = TextureN * sizeof(int);

        hipTextureDesc texDesc = {};
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = hipReadModeElementType;

        error_id = hipCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);
        bindedTexture = true;

        error_id = hipDeviceSynchronize();

        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[CHKROSHARETEXTURE.CPP]: hipCreateTextureObject Error: %s\n", hipGetErrorString(error_id));
            *error = 4;
            bindedTexture = false;
            break;
        }
        error_id = hipDeviceSynchronize();

        // Launch Kernel function
        dim3 Db = dim3(2);
        dim3 Dg = dim3(1, 1, 1);
        hipLaunchKernelGGL(chkROShareTexture, Dg, Db, 0, 0, tex, RO_N, TextureN, d_aRO, durationRO, durationTexture,
                           d_indexRO, d_indexTexture, d_disturb);

        error_id = hipDeviceSynchronize();
        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[CHKROSHARETEXTURE.CPP]: Kernel launch/execution Error: %s\n", hipGetErrorString(error_id));
            *error = 5;
            break;
        }

        // Copy results from GPU to Host
        if (hipMemcpyAndCheck("chkROShareTexture", h_timeinfoRO, durationRO, sizeof(unsigned int) * lessSize,
                              "durationRO -> h_timeinfoRO", error, true) != hipSuccess)
            break;

        if (hipMemcpyAndCheck("chkROShareTexture", h_timeinfoTexture, durationTexture, sizeof(unsigned int) * lessSize,
                              "durationTexture -> h_timeinfoTexture", error, true) != hipSuccess)
            break;

        if (hipMemcpyAndCheck("chkROShareTexture", h_indexRO, d_indexRO, sizeof(unsigned int) * lessSize,
                              "d_indexRO -> h_indexRO", error, true) != hipSuccess)
            break;

        if (hipMemcpyAndCheck("chkROShareTexture", h_indexTexture, d_indexTexture, sizeof(unsigned int) * lessSize,
                              "d_indexTexture -> h_indexTexture", error, true) != hipSuccess)
            break;

        if (hipMemcpyAndCheck("chkROShareTexture", disturb, d_disturb, sizeof(bool),
                              "d_disturb -> disturb", error, true) != hipSuccess)
            break;

        createOutputFile((int) RO_N, lessSize, h_indexRO, h_timeinfoRO, avgOutRO, potMissesOutRO, "ShareROTextureRO_");
        createOutputFile((int) TextureN, lessSize, h_indexTexture, h_timeinfoTexture, avgOutTexture,
                         potMissesOutTexture,
                         "ShareROTextureTexture_");
    } while (false);

    // Free Texture Object
    if (bindedTexture) {
        error_id = hipDestroyTextureObject(tex);
    }

    bool ret = false;
    if (disturb != nullptr) {
        ret = *disturb;
        free(disturb);
    }

    // Free Memory on GPU
    FreeTestMemory({d_indexRO, d_indexTexture, durationRO, durationTexture, d_aTexture, d_aRO, d_disturb}, true);


    // Free Memory on Host
    FreeTestMemory({h_indexRO, h_indexTexture, h_aRO, h_aTexture}, false);

    SET_PART_OF_2D(timeRO, h_timeinfoRO);
    SET_PART_OF_2D(timeTexture, h_timeinfoTexture);

    error_id = hipDeviceReset();
    return ret;
}

#define FreeMeasureRoTxtResOnlyPtr() \
free(time);                          \
free(avgFlow);                       \
free(potMissesFlow);                 \
free(timeRefRO);                     \
free(avgFlowRefRO);                  \
free(potMissesFlowRefRO);            \
free(timeRefTxt);                    \
free(avgFlowRefTxt);                 \
free(potMissesFlowRefTxt);           \

#define FreeMeasureROTxtResources()         \
if (time[0] != nullptr) {                   \
    free(time[0]);                          \
}                                           \
if (time[1] != nullptr) {                   \
    free(time[1]);                          \
}                                           \
if (timeRefRO[0] != nullptr) {              \
    free(timeRefRO[0]);                     \
}                                           \
if (timeRefTxt[0] != nullptr) {             \
    free(timeRefTxt[0]);                    \
}                                           \
free(time);                                 \
free(avgFlow);                              \
free(potMissesFlow);                        \
free(timeRefRO);                            \
free(avgFlowRefRO);                         \
free(potMissesFlowRefRO);                   \
free(timeRefTxt);                           \
free(avgFlowRefTxt);                        \
free(potMissesFlowRefTxt);                  \


dTuple measure_ROShareTexture(unsigned int measuredSizeRO, unsigned int measuredSizeTexture, unsigned int sub) {
    unsigned int ROSizeInInt = (measuredSizeRO - sub) >> 2; // / 4;
    unsigned int TextureSizeInInt = (measuredSizeTexture - sub) >> 2; // / 4;

    double *avgFlowRefRO = (double *) malloc(sizeof(double));
    unsigned int *potMissesFlowRefRO = (unsigned int *) malloc(sizeof(unsigned int));
    unsigned int **timeRefRO = (unsigned int **) malloc(sizeof(unsigned int *));

    double *avgFlowRefTxt = (double *) malloc(sizeof(double));
    unsigned int *potMissesFlowRefTxt = (unsigned int *) malloc(sizeof(unsigned int));
    unsigned int **timeRefTxt = (unsigned int **) malloc(sizeof(unsigned int *));

    double *avgFlow = (double *) malloc(sizeof(double) * 2);
    unsigned int *potMissesFlow = (unsigned int *) malloc(sizeof(unsigned int) * 2);
    unsigned int **time = (unsigned int **) malloc(sizeof(unsigned int *) * 2);
    if (avgFlowRefRO == nullptr || potMissesFlowRefRO == nullptr || timeRefRO == nullptr ||
        avgFlowRefTxt == nullptr || potMissesFlowRefTxt == nullptr || timeRefTxt == nullptr ||
        avgFlow == nullptr || potMissesFlow == nullptr || time == nullptr) {
        FreeMeasureRoTxtResOnlyPtr()
        printErrorCodeInformation(1);
        exit(1);
    }

    timeRefRO[0] = timeRefTxt[0] = time[0] = time[1] = nullptr;

    bool dist = true;
    int n = 5;
    while (dist && n > 0) {
        int error = 0;
        dist = launchROBenchmarkReferenceValue((int) ROSizeInInt, 1, avgFlowRefRO, potMissesFlowRefRO, timeRefRO,
                                               &error);
        if (error != 0) {
            FreeMeasureROTxtResources()
            printErrorCodeInformation(error);
            exit(error);
        }
        --n;
    }

    dist = true;
    n = 5;
    while (dist && n > 0) {
        int error = 0;
        dist = launchTextureBenchmarkReferenceValue((int) TextureSizeInInt, 1, avgFlowRefTxt, potMissesFlowRefTxt,
                                                    timeRefTxt, &error);
        if (error != 0) {
            FreeMeasureROTxtResources()
            printErrorCodeInformation(error);
            exit(error);
        }
        --n;
    }

    dist = true;
    n = 5;
    while (dist && n > 0) {
        int error = 0;
        dist = launchBenchmarkChkROShareTexture(ROSizeInInt, TextureSizeInInt, &avgFlow[0], &avgFlow[1],
                                                &potMissesFlow[0],
                                                &potMissesFlow[1], &time[0], &time[1], &error);
        if (error != 0) {
            FreeMeasureROTxtResources()
            printErrorCodeInformation(error);
            exit(error);
        }
        --n;
    }
#ifdef IsDebug
    fprintf(out, "Measured RO Avg in clean execution: %f\n", avgFlowRefRO[0]);
    fprintf(out, "Measured Texture Avg in clean execution: %f\n", avgFlowRefTxt[0]);

    fprintf(out, "Measured RO Avg While Shared With Texture:  %f\n", avgFlow[0]);
    fprintf(out, "Measured RO Pot Misses While Shared With Texture:  %u\n", potMissesFlow[0]);

    fprintf(out, "Measured Texture Avg While Shared With RO:  %f\n", avgFlow[1]);
    fprintf(out, "Measured Texture Pot Misses While Shared With RO:  %u\n", potMissesFlow[1]);
#endif //IsDebug

    dTuple result;
    result.first = std::abs(avgFlow[0] - avgFlowRefRO[0]);
    result.second = std::abs(avgFlow[1] - avgFlowRefTxt[0]);

    FreeMeasureROTxtResources()

    return result;
}


#endif //CUDATEST_ROSHARETEXTURE
