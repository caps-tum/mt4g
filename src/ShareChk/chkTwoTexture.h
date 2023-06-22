#include "hip/hip_runtime.h"

#ifndef CUDATEST_TWOTEXTURE
#define CUDATEST_TWOTEXTURE

# include <cstdio>
# include <cstdint>

# include "hip/hip_runtime.h"
# include "../general_functions.h"
__global__ void chkTwoTexture(hipTextureObject_t tex1, hipTextureObject_t tex2, unsigned int TextureN, unsigned int * durationTxt1, unsigned int * durationTxt2, unsigned int *indexTxt1, unsigned int *indexTxt2,
                              bool* isDisturbed) {
    *isDisturbed = false;

    unsigned int start_time, end_time;
    int j = 0;
    __shared__ ALIGN(16) long long s_tvalueTxt1[lessSize];
    __shared__ ALIGN(16) unsigned int s_indexTxt1[lessSize];
    __shared__ ALIGN(16) long long s_tvalueTxt2[lessSize];
    __shared__ ALIGN(16) unsigned int s_indexTxt2[lessSize];

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < lessSize; k++) {
            s_indexTxt1[k] = 0;
            s_tvalueTxt1[k] = 0;
        }
    }

    __syncthreads();

    if (threadIdx.x == 1){
        for (int k = 0; k < lessSize; k++) {
            s_indexTxt2[k] = 0;
            s_tvalueTxt2[k] = 0;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < TextureN; k++) {
            j = tex1Dfetch<int>(tex1, j);
        }
    }

    __syncthreads();

    if (threadIdx.x == 1){
        for (int k = 0; k < TextureN; k++) {
            j = tex1Dfetch<int>(tex2, j);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        //second round
        for (int k = 0; k < lessSize; k++) {
            start_time = clock();
            j = tex1Dfetch<int>(tex1, j);
            s_indexTxt1[k] = j;
            end_time = clock();
            s_tvalueTxt1[k] = (end_time - start_time);
        }
    }

    __syncthreads();

    if (threadIdx.x == 1){
        for (int k = 0; k < lessSize; k++) {
            start_time = clock();
            j = tex1Dfetch<int>(tex2, j);
            s_indexTxt2[k] = j;
            end_time = clock();
            s_tvalueTxt2[k] = (end_time - start_time);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < lessSize; k++) {
            indexTxt1[k] = s_indexTxt1[k];
            durationTxt1[k] = s_tvalueTxt1[k];
            if (durationTxt1[k] > 3000) {
                *isDisturbed = true;
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == 1){
        for (int k = 0; k < lessSize; k++) {
            indexTxt2[k] = s_indexTxt2[k];
            durationTxt2[k] = s_tvalueTxt2[k];
            if (durationTxt2[k] > 3000) {
                *isDisturbed = true;
            }
        }
    }
}

bool launchBenchmarkTwoTexture(unsigned int TextureN, double *avgOutTxt1, double* avgOutTxt2, unsigned int* potMissesOutTxt1, unsigned int* potMissesOutTxt2, unsigned int **timeTxt1, unsigned int **timeTxt2, int* error) {
    hipError_t error_id;
    error_id = hipDeviceReset();

    int* h_aTexture = nullptr, *d_aTexture = nullptr;
    unsigned int *h_indexTexture1 = nullptr, *h_indexTexture2 = nullptr, *h_timeinfoTexture1 = nullptr, *h_timeinfoTexture2 = nullptr,
    *durationTxt1 = nullptr, *durationTxt2 = nullptr, *d_indexTxt1 = nullptr, *d_indexTxt2 = nullptr;
    bool *disturb = nullptr, *d_disturb = nullptr;
    bool bindedTexture = false;
    hipTextureObject_t tex1 = 0, tex2 = 0;

    do {
        // Allocate Memory on Host
        h_indexTexture1 = (unsigned int *) malloc(sizeof(unsigned int) * lessSize);
        if (h_indexTexture1 == nullptr) {
            printf("[CHKTWOTEXTURE.CPP]: malloc h_indexTexture1 Error\n");
            *error = 1;
            break;
        }

        h_indexTexture2 = (unsigned int *) malloc(sizeof(unsigned int) * lessSize);
        if (h_indexTexture2 == nullptr) {
            printf("[CHKTWOTEXTURE.CPP]: malloc h_indexTexture2 Error\n");
            *error = 1;
            break;
        }

        h_timeinfoTexture1 = (unsigned int *) malloc(sizeof(unsigned int) * lessSize);
        if (h_timeinfoTexture1 == nullptr) {
            printf("[CHKTWOTEXTURE.CPP]: malloc h_timeinfoTexture1 Error\n");
            *error = 1;
            break;
        }

        h_timeinfoTexture2 = (unsigned int *) malloc(sizeof(unsigned int) * lessSize);
        if (h_timeinfoTexture2 == nullptr) {
            printf("[CHKTWOTEXTURE.CPP]: malloc h_timeinfoTexture2 Error\n");
            *error = 1;
            break;
        }

        disturb = (bool *) malloc(sizeof(bool));
        if (disturb == nullptr) {
            printf("[CHKTWOTEXTURE.CPP]: malloc disturb Error\n");
            *error = 1;
            break;
        }

        h_aTexture = (int*) malloc(sizeof(int) * (TextureN));
        if (h_aTexture == nullptr) {
            printf("[CHKTWOTEXTURE.CPP]: malloc h_aTexture Error\n");
            *error = 1;
            break;
        }

        // Allocate Memory on GPU
        error_id = hipMalloc((void **) &durationTxt1, sizeof(unsigned int) * lessSize);
        if (error_id != hipSuccess) {
            printf("[CHKTWOTEXTURE.CPP]: hipMalloc durationTxt1 Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = hipMalloc((void **) &durationTxt2, sizeof(unsigned int) * lessSize);
        if (error_id != hipSuccess) {
            printf("[CHKTWOTEXTURE.CPP]: hipMalloc durationTxt2 Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = hipMalloc((void **) &d_indexTxt1, sizeof(unsigned int) * lessSize);
        if (error_id != hipSuccess) {
            printf("[CHKTWOTEXTURE.CPP]: hipMalloc d_indextxt1 Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = hipMalloc((void **) &d_indexTxt2, sizeof(unsigned int) * lessSize);
        if (error_id != hipSuccess) {
            printf("[CHKTWOTEXTURE.CPP]: hipMalloc d_indexTxt2 Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = hipMalloc((void **) &d_disturb, sizeof(bool));
        if (error_id != hipSuccess) {
            printf("[CHKTWOTEXTURE.CPP]: hipMalloc disturb Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = hipMalloc((void **) &d_aTexture, sizeof(int) * (TextureN));
        if (error_id != hipSuccess) {
            printf("[CHKTWOTEXTURE.CPP]: hipMalloc d_aTexture Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        // Initialize p-chase array
        for (int i = 0; i < TextureN; i++) {
            h_aTexture[i] = (i + 1) % (int)TextureN;
        }

        // Copy array from Host to GPU
        error_id = hipMemcpy(d_aTexture, h_aTexture, sizeof(int) * TextureN, hipMemcpyHostToDevice);
        if (error_id != hipSuccess) {
            printf("[CHKTWOTEXTURE.CPP]: hipMemcpy d_aTexture Error: %s\n", hipGetErrorString(error_id));
            *error = 3;
            break;
        }

        // Create Texture Object
        hipResourceDesc resDesc = {};
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = hipResourceTypeLinear;
        resDesc.res.linear.devPtr = d_aTexture;
        resDesc.res.linear.desc.f = hipChannelFormatKindSigned;
        resDesc.res.linear.desc.x = 32; // bits per channel
        resDesc.res.linear.sizeInBytes = TextureN*sizeof(int);

        hipTextureDesc texDesc = {};
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = hipReadModeElementType;

        error_id = hipCreateTextureObject(&tex1, &resDesc, &texDesc, nullptr);
        error_id = hipCreateTextureObject(&tex2, &resDesc, &texDesc, nullptr);
        bindedTexture = true;

        error_id = hipDeviceSynchronize();

        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[CHKTWOTEXTURE.CPP]: hipCreateTextureObject Error: %s\n", hipGetErrorString(error_id));
            *error = 4;
            bindedTexture = false;
            break;
        }
        error_id = hipDeviceSynchronize();

        // Launch Kernel function
        dim3 Db = dim3(2);
        dim3 Dg = dim3(1, 1, 1);
        hipLaunchKernelGGL(chkTwoTexture, Dg, Db, 0, 0, tex1, tex2, TextureN, durationTxt1, durationTxt2, d_indexTxt1, d_indexTxt2, d_disturb);

        error_id = hipDeviceSynchronize();
        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[CHKTWOTEXTURE.CPP]: Kernel launch/execution Error: %s\n", hipGetErrorString(error_id));
            *error = 5;
            break;
        }

        // Copy results from GPU to Host
        error_id = hipMemcpy((void *) h_timeinfoTexture1, (void *) durationTxt1, sizeof(unsigned int) * lessSize,hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[CHKTWOTEXTURE.CPP]: hipMemcpy durationTxt1 Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = hipMemcpy((void *) h_timeinfoTexture2, (void *) durationTxt2, sizeof(unsigned int) * lessSize,hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[CHKTWOTEXTURE.CPP]: hipMemcpy durationTxt2 Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = hipMemcpy((void *) h_indexTexture1, (void *) d_indexTxt1, sizeof(unsigned int) * lessSize,hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[CHKTWOTEXTURE.CPP]: hipMemcpy d_indexTxt1 Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = hipMemcpy((void *) h_indexTexture2, (void *) d_indexTxt2, sizeof(unsigned int) * lessSize,hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[CHKTWOTEXTURE.CPP]: hipMemcpy d_indexTxt2 Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = hipMemcpy((void *) disturb, (void *) d_disturb, sizeof(bool), hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[CHKTWOTEXTURE.CPP]: hipMemcpy disturb Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        createOutputFile((int) TextureN, lessSize, h_indexTexture1, h_timeinfoTexture1, avgOutTxt1, potMissesOutTxt1, "TwoTexture1_");
        createOutputFile((int) TextureN, lessSize, h_indexTexture2, h_timeinfoTexture2, avgOutTxt2, potMissesOutTxt2, "TwoTexture2_");

    } while(false);

    // Free Texture Object
    if (bindedTexture) {
        error_id = hipDestroyTextureObject(tex1);
        error_id = hipDestroyTextureObject(tex2);
    }

    bool ret = false;
    if (disturb != nullptr) {
        ret = *disturb;
        free(disturb);
    }

    // Free Memory on GPU
    FreeTestMemory({durationTxt1, durationTxt2, d_indexTxt1, d_indexTxt2, d_aTexture, d_disturb}, true);

    // Free Memory on Host
    FreeTestMemory({h_indexTexture1, h_indexTexture2, h_aTexture}, false);

    SET_PART_OF_2D(timeTxt1, h_timeinfoTexture1);
    SET_PART_OF_2D(timeTxt2, h_timeinfoTexture2);

    error_id = hipDeviceReset();
    return ret;
}

#define FreeMeasureTwoTextureResOnlyPtr()   \
free(time);                                 \
free(timeRef);                              \
free(avgFlow);                              \
free(potMissesFlow);                        \
free(avgFlowRef);                           \
free(potMissesFlowRef);                     \

#define FreeMeasureTwoTextureResources()    \
if (time[0] != nullptr) {                   \
    free(time[0]);                          \
}                                           \
if (time[1] != nullptr) {                   \
    free(time[1]);                          \
}                                           \
if (timeRef[0] != nullptr) {                \
    free(timeRef[0]);                       \
}                                           \
free(time);                                 \
free(timeRef);                              \
free(avgFlow);                              \
free(potMissesFlow);                        \
free(avgFlowRef);                           \
free(potMissesFlowRef);                     \


double measure_TwoTexture(unsigned int measuredSizeCache, unsigned int sub) {
    unsigned int CacheSizeInInt = (measuredSizeCache - sub) / 4;

    double* avgFlowRef = (double*) malloc(sizeof(double));
    unsigned int *potMissesFlowRef = (unsigned int*) malloc(sizeof(unsigned int));
    unsigned int** timeRef = (unsigned int**) malloc(sizeof(unsigned int*));

    double* avgFlow = (double*) malloc(sizeof(double)  * 2);
    unsigned int *potMissesFlow = (unsigned int*) malloc(sizeof(unsigned int) * 2);
    unsigned int** time = (unsigned int**) malloc(sizeof(unsigned int*) * 2);
    if (avgFlowRef == nullptr || potMissesFlowRef == nullptr || timeRef == nullptr ||
        avgFlow == nullptr || potMissesFlow == nullptr || time == nullptr) {
        FreeMeasureTwoTextureResOnlyPtr()
        printErrorCodeInformation(1);
        exit(1);
    }
    timeRef[0] = time[0] = time[1] = nullptr;

    bool dist = true;
    int n = 5;
    while(dist && n > 0) {
        int error = 0;
        dist = launchTextureBenchmarkReferenceValue((int) CacheSizeInInt, 1, avgFlowRef, potMissesFlowRef, timeRef, &error);
        if (error != 0) {
            FreeMeasureTwoTextureResources()
            printErrorCodeInformation(error);
            exit(error);
        }
        --n;
    }

    dist = true;
    n = 5;
    while(dist && n > 0) {
        int error = 0;
        dist = launchBenchmarkTwoTexture(CacheSizeInInt, &avgFlow[0], &avgFlow[1], &potMissesFlow[0], &potMissesFlow[1],
                                         &time[0], &time[1], &error);
        if (error != 0) {
            FreeMeasureTwoTextureResources()
            printErrorCodeInformation(error);
            exit(error);
        }
        --n;
    }

    dTriple result;
    result.first = avgFlowRef[0];
    result.second = avgFlow[0];
    result.third = avgFlow[1];
#ifdef IsDebug
    fprintf(out, "Measured Txt Avg in clean execution: %f\n", avgFlowRef[0]);

    fprintf(out, "Measured Txt1 Avg While Shared With Txt2:  %f\n", avgFlow[0]);
    fprintf(out, "Measured Txt1 Pot Misses While Shared With Txt2:  %u\n", potMissesFlow[0]);

    fprintf(out, "Measured Txt2 Avg While Shared With Txt1:  %f\n", avgFlow[1]);
    fprintf(out, "Measured Txt2 Pot Misses While Shared With Txt1:  %u\n", potMissesFlow[1]);
#endif //IsDebug

    FreeMeasureTwoTextureResources()

    return std::max(std::abs(result.second - result.first), std::abs(result.third - result.first));
}


#endif //CUDATEST_TWOTEXTURE
