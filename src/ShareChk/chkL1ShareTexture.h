#include "hip/hip_runtime.h"

#ifndef CUDATEST_L1SHARETEXTURE
#define CUDATEST_L1SHARETEXTURE

# include <cstdio>
# include <cstdint>

# include "hip/hip_runtime.h"
# include "../general_functions.h"

#define HARDCODED_3000 3000

__global__ void
chkL1ShareTexture(hipTextureObject_t tex, unsigned int L1_N, unsigned int TextureN, unsigned int *myArray,
                  unsigned int *durationL1, unsigned int *durationTexture, unsigned int *indexL1,
                  unsigned int *indexTexture,
                  bool *isDisturbed) {
    *isDisturbed = false;

    unsigned int start_time, end_time;
    unsigned int j = 0;
    int j2 = 0;
    __shared__ ALIGN(16) long long s_tvalueL1[lessSize];
    __shared__ ALIGN(16) unsigned int s_indexL1[lessSize];
    __shared__ ALIGN(16) long long s_tvalueTexture[lessSize];
    __shared__ ALIGN(16) unsigned int s_indexTexture[lessSize];

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < lessSize; k++) {
            s_indexL1[k] = 0;
            s_tvalueL1[k] = 0;
        }
    }

    __syncthreads();

    if (threadIdx.x == 1) {
        for (int k = 0; k < lessSize; k++) {
            s_indexTexture[k] = 0;
            s_tvalueTexture[k] = 0;
        }
    }

    unsigned int *ptr;
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < L1_N; k++) {
            ptr = myArray + j;
            NON_TEMPORAL_LOAD_CA(j, ptr);
        }
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
            ptr = myArray + j;
            LOCAL_CLOCK(start_time);
            NON_TEMPORAL_LOAD_CA(j, ptr);
            s_indexL1[k] = j;
            LOCAL_CLOCK(end_time);
            s_tvalueL1[k] = end_time - start_time;
        }
    }

    __syncthreads();

    if (threadIdx.x == 1) {
        for (int k = 0; k < lessSize; k++) {
            LOCAL_CLOCK(start_time);
            j2=tex1Dfetch<int>(tex, j2);
            s_indexTexture[k] = j2;
            LOCAL_CLOCK(end_time);
            s_tvalueTexture[k] = (end_time - start_time);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < lessSize; k++) {
            indexL1[k] = s_indexL1[k];
            durationL1[k] = s_tvalueL1[k];
            if (durationL1[k] > HARDCODED_3000) {
                *isDisturbed = true;
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == 1) {
        for (int k = 0; k < lessSize; k++) {
            indexTexture[k] = s_indexTexture[k];
            durationTexture[k] = s_tvalueTexture[k];
            if (durationTexture[k] > HARDCODED_3000) {
                *isDisturbed = true;
            }
        }
    }
}

bool launchL1BenchmarkReferenceValue(int N, int stride, double *avgOut, unsigned int *potMissesOut, unsigned int **time,
                                     int *error) {
    return launchL1KernelBenchmark(N, stride, avgOut, potMissesOut, time, error);
}

bool launchBenchmarkChkL1ShareTexture(unsigned int L1Data_N, unsigned int TextureN, double *avgOutL1Data,
                                      double *avgOutTexture, unsigned int *potMissesOutL1Data,
                                      unsigned int *potMissesOutTexture, unsigned int **timeL1Data,
                                      unsigned int **timeTexture, int *error) {
    hipError_t error_id;
    error_id = hipDeviceReset();

    int *h_aTexture = nullptr, *d_aTexture = nullptr;
    unsigned int *h_indexL1Data = nullptr, *h_indexTexture = nullptr, *h_timeinfoL1Data = nullptr, *h_timeinfoTexture = nullptr, *h_aL1 = nullptr,
            *durationL1 = nullptr, *durationTexture = nullptr, *d_indexL1 = nullptr, *d_indexTexture = nullptr, *d_aL1 = nullptr;
    bool *disturb = nullptr, *d_disturb = nullptr;
    bool bindedTexture = false;
    hipTextureObject_t tex = 0;

    do {
        // Allocate Memory on Host
        h_indexL1Data = (unsigned int *) mallocAndCheck("chkL1ShareTexture", sizeof(unsigned int) * lessSize,
                                                        "h_indexL1Data", error);

        h_indexTexture = (unsigned int *) mallocAndCheck("chkL1ShareTexture", sizeof(unsigned int) * lessSize,
                                                         "h_indexTexture", error);

        h_timeinfoL1Data = (unsigned int *) mallocAndCheck("chkL1ShareTexture", sizeof(unsigned int) * lessSize,
                                                           "h_timeinfoL1Data", error);

        h_timeinfoTexture = (unsigned int *) mallocAndCheck("chkL1ShareTexture", sizeof(unsigned int) * lessSize,
                                                            "h_timeinfoTexture", error);

        disturb = (bool *) mallocAndCheck("chkL1ShareTexture", sizeof(bool), "disturb", error);


        h_aTexture = (int *)mallocAndCheck("chkL1ShareTexture", sizeof(unsigned int) * TextureN,
                                           "h_aTexture", error);

        h_aL1 = (unsigned int *) mallocAndCheck("chkL1ShareTexture", sizeof(unsigned int) * L1Data_N,
                                                "h_aL1", error);

        // Allocate Memory on GPU
        if (hipMallocAndCheck("chkL1ShareTexture", (void **) &durationL1,
                              sizeof(unsigned int) * lessSize,
                              "durationL1", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("chkL1ShareTexture", (void **) &durationTexture,
                              sizeof(unsigned int) * lessSize,
                              "durationTexture", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("chkL1ShareTexture", (void **) &d_indexL1,
                              sizeof(unsigned int) * lessSize,
                              "d_indexL1", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("chkL1ShareTexture", (void **) &d_indexTexture,
                              sizeof(unsigned int) * lessSize,
                              "d_indexTexture", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("chkL1ShareTexture", (void **) &d_disturb,
                              sizeof(bool),
                              "d_disturb", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("chkL1ShareTexture", (void **) &d_aTexture,
                              sizeof(unsigned int) * TextureN,
                              "d_aTexture", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("chkL1ShareTexture", (void **) &d_aL1,
                              sizeof(unsigned int) * L1Data_N,
                              "d_aL1", error) != hipSuccess)
            break;

        // Initialize p-chase arrays
        for (int i = 0; i < TextureN; i++) {
            h_aTexture[i] = (i + 1) % (int) TextureN;
        }

        for (int i = 0; i < L1Data_N; i++) {
            //original:
            h_aL1[i] = (i + 1) % L1Data_N;
        }

        // Copy arrays from Host to GPU
        if (hipMemcpyAndCheck("chkL1ShareTexture", d_aTexture, h_aTexture, sizeof(unsigned int) * TextureN,
                              "h_aTexture -> d_aTexture", error, false) != hipSuccess)
            break;

        if (hipMemcpyAndCheck("chkL1ShareTexture", d_aL1, h_aL1, sizeof(unsigned int) * L1Data_N,
                              "h_aL1 -> d_aL1", error, false) != hipSuccess)
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
            printf("[CHKL1SHARETEXTURE.CPP]: hipCreateTextureObject Error: %s\n", hipGetErrorString(error_id));
            *error = 4;
            bindedTexture = false;
            break;
        }
        error_id = hipDeviceSynchronize();

        // Launch Kernel function
        dim3 Db = dim3(2);
        dim3 Dg = dim3(1, 1, 1);
        hipLaunchKernelGGL(chkL1ShareTexture, Dg, Db, 0, 0, tex, L1Data_N, TextureN, d_aL1, durationL1, durationTexture,
                           d_indexL1, d_indexTexture, d_disturb);

        error_id = hipDeviceSynchronize();
        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[CHKL1SHARETEXTURE.CPP]: Kernel launch/execution Error: %s\n", hipGetErrorString(error_id));
            *error = 5;
            break;
        }

        // Copy results from GPU to Host
        if (hipMemcpyAndCheck("chkL1ShareTexture", h_timeinfoL1Data, durationL1, sizeof(unsigned int) * lessSize,
                              "durationL1 -> h_timeinfoL1Data", error, true) != hipSuccess)
            break;



        error_id = hipMemcpy((void *) h_timeinfoTexture, (void *) durationTexture, sizeof(unsigned int) * lessSize,
                             hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[CHKL1SHARETEXTURE.CPP]: hipMemcpy durationTexture Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = hipMemcpy((void *) h_indexL1Data, (void *) d_indexL1, sizeof(unsigned int) * lessSize,
                             hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[CHKL1SHARETEXTURE.CPP]: hipMemcpy d_indexL1 Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = hipMemcpy((void *) h_indexTexture, (void *) d_indexTexture, sizeof(unsigned int) * lessSize,
                             hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[CHKL1SHARETEXTURE.CPP]: hipMemcpy d_indexTexture Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = hipMemcpy((void *) disturb, (void *) d_disturb, sizeof(bool), hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[CHKL1SHARETEXTURE.CPP]: hipMemcpy disturb Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        createOutputFile((int) L1Data_N, lessSize, h_indexL1Data, h_timeinfoL1Data, avgOutL1Data, potMissesOutL1Data,
                         "ShareL1TextureRO_");
        createOutputFile((int) TextureN, lessSize, h_indexTexture, h_timeinfoTexture, avgOutTexture,
                         potMissesOutTexture,
                         "ShareL1TextureTexture_");
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
    FreeTestMemory({d_indexL1, d_indexTexture, durationL1, durationTexture, d_aTexture, d_aL1, d_disturb}, true);

    // Free Memory on Host
    FreeTestMemory({h_indexL1Data, h_indexTexture, h_aL1, h_aTexture}, false);

    SET_PART_OF_2D(timeL1Data, h_timeinfoL1Data);
    SET_PART_OF_2D(timeTexture, h_timeinfoTexture);

    error_id = hipDeviceReset();
    return ret;
}

#define FreeMeasureL1TxtResOnlyPtr() \
free(time);                          \
free(avgFlow);                       \
free(potMissesFlow);                 \
free(timeRefL1);                     \
free(avgFlowRefL1);                  \
free(potMissesFlowRefL1);            \
free(timeRefTxt);                    \
free(avgFlowRefTxt);                 \
free(potMissesFlowRefTxt);           \

#define FreeMeasureL1TxtResources()         \
if (time[0] != nullptr) {                   \
    free(time[0]);                          \
}                                           \
if (time[1] != nullptr) {                   \
    free(time[1]);                          \
}                                           \
if (timeRefL1[0] != nullptr) {              \
    free(timeRefL1[0]);                     \
}                                           \
if (timeRefTxt[0] != nullptr) {             \
    free(timeRefTxt[0]);                    \
}                                           \
free(time);                                 \
free(avgFlow);                              \
free(potMissesFlow);                        \
free(timeRefL1);                            \
free(avgFlowRefL1);                         \
free(potMissesFlowRefL1);                   \
free(timeRefTxt);                           \
free(avgFlowRefTxt);                        \
free(potMissesFlowRefTxt);                  \


dTuple measure_L1ShareTexture(unsigned int measuredSizeL1, unsigned int measuredSizeTexture, unsigned int sub) {
    unsigned int L1SizeInInt = (measuredSizeL1 - sub) >> 2;// / 4;
    unsigned int TextureSizeInInt = (measuredSizeTexture - sub) >> 2;// / 4;

    double *avgFlowRefL1 = (double *) malloc(sizeof(double));
    unsigned int *potMissesFlowRefL1 = (unsigned int *) malloc(sizeof(unsigned int));
    unsigned int **timeRefL1 = (unsigned int **) malloc(sizeof(unsigned int *));

    double *avgFlowRefTxt = (double *) malloc(sizeof(double));
    unsigned int *potMissesFlowRefTxt = (unsigned int *) malloc(sizeof(unsigned int));
    unsigned int **timeRefTxt = (unsigned int **) malloc(sizeof(unsigned int *));

    double *avgFlow = (double *) malloc(sizeof(double) * 2);
    unsigned int *potMissesFlow = (unsigned int *) malloc(sizeof(unsigned int) * 2);
    unsigned int **time = (unsigned int **) malloc(sizeof(unsigned int *) * 2);
    if (avgFlowRefL1 == nullptr || potMissesFlowRefL1 == nullptr || timeRefL1 == nullptr ||
        avgFlowRefTxt == nullptr || potMissesFlowRefTxt == nullptr || timeRefTxt == nullptr ||
        avgFlow == nullptr || potMissesFlow == nullptr || time == nullptr) {
        FreeMeasureL1TxtResOnlyPtr()
        printErrorCodeInformation(1);
        exit(1);
    }
    timeRefL1[0] = timeRefTxt[0] = time[0] = time[1] = nullptr;

    bool dist = true;
    int n = 5;
    while (dist && n > 0) {
        int error = 0;
        dist = launchL1BenchmarkReferenceValue((int) L1SizeInInt, 1, avgFlowRefL1, potMissesFlowRefL1, timeRefL1,
                                               &error);
        if (error != 0) {
            FreeMeasureL1TxtResources()
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
            FreeMeasureL1TxtResources()
            printErrorCodeInformation(error);
            exit(error);
        }
        --n;
    }

    dist = true;
    n = 5;
    while (dist && n > 0) {
        int error = 0;
        dist = launchBenchmarkChkL1ShareTexture(L1SizeInInt, TextureSizeInInt, &avgFlow[0], &avgFlow[1],
                                                &potMissesFlow[0],
                                                &potMissesFlow[1], &time[0], &time[1], &error);
        if (error != 0) {
            FreeMeasureL1TxtResources()
            printErrorCodeInformation(error);
            exit(error);
        }
        --n;
    }

#ifdef IsDebug
    fprintf(out, "Measured L1 Avg in clean execution: %f\n", avgFlowRefL1[0]);
    fprintf(out, "Measured Texture Avg in clean execution: %f\n", avgFlowRefTxt[0]);

    fprintf(out, "Measured L1 Avg While Shared With Texture:  %f\n", avgFlow[0]);
    fprintf(out, "Measured L1 Pot Misses While Shared With Texture:  %u\n", potMissesFlow[0]);

    fprintf(out, "Measured Texture Avg While Shared With L1:  %f\n", avgFlow[1]);
    fprintf(out, "Measured Texture Pot Misses While Shared With L1:  %u\n", potMissesFlow[1]);
#endif //IsDebug

    dTuple result;
    result.first = std::abs(avgFlow[0] - avgFlowRefL1[0]);
    result.second = std::abs(avgFlow[1] - avgFlowRefTxt[0]);

    FreeMeasureL1TxtResources()

    return result;
}


#endif //CUDATEST_L1SHARETEXTURE
