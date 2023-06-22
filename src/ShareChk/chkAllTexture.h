#include "hip/hip_runtime.h"

#ifndef CUDATEST_ALLTEXTURE
#define CUDATEST_ALLTEXTURE

# include <cstdio>
# include <cstdint>
# include "hip/hip_runtime.h"

#define HARDCODED_3000 3000
/**
 * See launchBenchmarkTwoCoreTexture
 * @param TextureN
 * @param duration1
 * @param duration2
 * @param index1
 * @param index2
 * @param isDisturbed
 * @param baseCore
 * @param testCore
 */
__global__ void chkTwoCoreTexture(hipTextureObject_t tex1, hipTextureObject_t tex2, unsigned int TextureN, unsigned int * duration1,
                                  unsigned int * duration2, unsigned int *index1, unsigned int *index2, bool* isDisturbed, unsigned int baseCore, unsigned int testCore) {
    *isDisturbed = false;

    unsigned int start_time, end_time;
    int j = 0;
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
    }

    __syncthreads();

    if (threadIdx.x == testCore) {
        for (int k = 0; k < lessSize; k++) {
            s_indexTxt2[k] = 0;
            s_tvalueTxt2[k] = 0;
        }
    }

    __syncthreads();

    if (threadIdx.x == baseCore) {
        for (int k = 0; k < TextureN; k++) {
            j = tex1Dfetch<int>(tex1, j);
        }
    }

    __syncthreads();

    if (threadIdx.x == testCore) {
        for (int k = 0; k < TextureN; k++) {
            j = tex1Dfetch<int>(tex2, j);
        }
    }

    __syncthreads();

    if (threadIdx.x == baseCore) {
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

    if (threadIdx.x == testCore) {
        for (int k = 0; k < lessSize; k++) {
            start_time = clock();
            j = tex1Dfetch<int>(tex2, j);
            s_indexTxt2[k] = j;
            end_time = clock();
            s_tvalueTxt2[k] = (end_time - start_time);
        }
    }

    __syncthreads();

    if (threadIdx.x == baseCore) {
        for (int k = 0; k < lessSize; k++) {
            index1[k] = s_indexTxt1[k];
            duration1[k] = s_tvalueTxt1[k];
            if (duration1[k] > HARDCODED_3000) {
                *isDisturbed = true;
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == testCore){
        for (int k = 0; k < lessSize; k++) {
            index2[k] = s_indexTxt2[k];
            duration2[k] = s_tvalueTxt2[k];
            if (duration2[k] > HARDCODED_3000) {
                *isDisturbed = true;
            }
        }
    }
}

/**
 * launches the two core share texture cache kernel benchmark
 * @param TextureN size of the Texture Array
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
bool launchBenchmarkTwoCoreTexture(unsigned int TextureN, double *avgOut1, double* avgOut2, unsigned int* potMissesOut1, unsigned int* potMissesOut2,
                                   unsigned int **time1, unsigned int **time2, int* error, unsigned int numberOfCores, unsigned int baseCore, unsigned int testCore) {
    hipError_t error_id;
    error_id = hipDeviceReset();

    int *h_a = nullptr, *d_a = nullptr;
    unsigned int *h_index1 = nullptr, *h_index2 = nullptr, *h_timeinfo1 = nullptr, *h_timeinfo2 = nullptr,
    *duration1 = nullptr, *duration2 = nullptr, *d_index1 = nullptr, *d_index2 = nullptr;
    bool *disturb = nullptr, *d_disturb = nullptr;
    bool textureBinded = false;
    hipTextureObject_t tex1 = 0, tex2 = 0;

    do {
        // Allocate Memory on Host
        h_index1 = (unsigned int *) malloc(sizeof(unsigned int) * lessSize);
        if (h_index1 == nullptr) {
            printf("[CHKALLTEXTURE.CPP]: malloc h_index1 Error\n");
            *error = 1;
            break;
        }

        h_index2 = (unsigned int *) malloc(sizeof(unsigned int) * lessSize);
        if (h_index2 == nullptr) {
            printf("[CHKALLTEXTURE.CPP]: malloc h_index2 Error\n");
            *error = 1;
            break;
        }

        h_timeinfo1 = (unsigned int *) malloc(sizeof(unsigned int) * lessSize);
        if (h_timeinfo1 == nullptr) {
            printf("[CHKALLTEXTURE.CPP]: malloc h_timeinfo1 Error\n");
            *error = 1;
            break;
        }

        h_timeinfo2 = (unsigned int *) malloc(sizeof(unsigned int) * lessSize);
        if (h_timeinfo2 == nullptr) {
            printf("[CHKALLTEXTURE.CPP]: malloc h_timeinfo2 Error\n");
            *error = 1;
            break;
        }

        disturb = (bool *) malloc(sizeof(bool));
        if (disturb == nullptr) {
            printf("[CHKALLTEXTURE.CPP]: malloc disturb Error\n");
            *error = 1;
            break;
        }

        h_a = (int*) malloc(sizeof(int) * (TextureN));
        if (h_a == nullptr) {
            printf("[CHKALLTEXTURE.CPP]: malloc h_a Error\n");
            *error = 1;
            break;
        }

        // Allocate Memory on GPU
        error_id = hipMalloc((void **) &duration1, sizeof(unsigned int) * lessSize);
        if (error_id != hipSuccess) {
            printf("[CHKALLTEXTURE.CPP]: hipMalloc duration1 Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = hipMalloc((void **) &duration2, sizeof(unsigned int) * lessSize);
        if (error_id != hipSuccess) {
            printf("[CHKALLTEXTURE.CPP]: hipMalloc duration2 Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = hipMalloc((void **) &d_index1, sizeof(unsigned int) * lessSize);
        if (error_id != hipSuccess) {
            printf("[CHKALLTEXTURE.CPP]: hipMalloc d_indextxt1 Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = hipMalloc((void **) &d_index2, sizeof(unsigned int) * lessSize);
        if (error_id != hipSuccess) {
            printf("[CHKALLTEXTURE.CPP]: hipMalloc d_index2 Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = hipMalloc((void **) &d_disturb, sizeof(bool));
        if (error_id != hipSuccess) {
            printf("[CHKALLTEXTURE.CPP]: hipMalloc disturb Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = hipMalloc((void **) &d_a, sizeof(int) * (TextureN));
        if (error_id != hipSuccess) {
            printf("[CHKALLTEXTURE.CPP]: hipMalloc d_a Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        // Initialize p-chase array
        for (int i = 0; i < TextureN; i++) {
            //original:
            h_a[i] = (i + 1) % (int)TextureN;
        }

        // Copy array from Host to GPU
        error_id = hipMemcpy(d_a, h_a, sizeof(int) * TextureN, hipMemcpyHostToDevice);
        if (error_id != hipSuccess) {
            printf("[CHKALLTEXTURE.CPP]: hipMemcpy d_a Error: %s\n", hipGetErrorString(error_id));
            *error = 3;
            break;
        }

        // Create Texture Objects
        hipResourceDesc resDesc = {};
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = hipResourceTypeLinear;
        resDesc.res.linear.devPtr = d_a;
        resDesc.res.linear.desc.f = hipChannelFormatKindSigned;
        resDesc.res.linear.desc.x = 32; // bits per channel
        resDesc.res.linear.sizeInBytes = TextureN*sizeof(int);

        hipTextureDesc texDesc = {};
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = hipReadModeElementType;

        error_id = hipCreateTextureObject(&tex1, &resDesc, &texDesc, nullptr);
        error_id = hipCreateTextureObject(&tex2, &resDesc, &texDesc, nullptr);
        textureBinded = true;

        error_id = hipDeviceSynchronize();

        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[CHKALLTEXTURE.CPP]: hipCreateTextureObject Error: %s\n", hipGetErrorString(error_id));
            *error = 4;
            textureBinded = false;
            break;
        }
        error_id = hipDeviceSynchronize();

        // Launch Kernel function
        dim3 Db = dim3(numberOfCores);
        dim3 Dg = dim3(1, 1, 1);
        hipLaunchKernelGGL(chkTwoCoreTexture, Dg, Db, 0, 0, tex1, tex2, TextureN, duration1, duration2, d_index1, d_index2, d_disturb, baseCore,
                                      testCore);

        error_id = hipDeviceSynchronize();
        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[CHKALLTEXTURE.CPP]: Kernel launch/execution Error: %s\n", hipGetErrorString(error_id));
            *error = 5;
            break;
        }

        // Copy results from GPU to Host
        error_id = hipMemcpy((void *) h_timeinfo1, (void *) duration1, sizeof(unsigned int) * lessSize,
                              hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[CHKALLTEXTURE.CPP]: hipMemcpy duration1 Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = hipMemcpy((void *) h_timeinfo2, (void *) duration2, sizeof(unsigned int) * lessSize,
                              hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[CHKALLTEXTURE.CPP]: hipMemcpy duration2 Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = hipMemcpy((void *) h_index1, (void *) d_index1, sizeof(unsigned int) * lessSize,
                              hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[CHKALLTEXTURE.CPP]: hipMemcpy d_index1 Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = hipMemcpy((void *) h_index2, (void *) d_index2, sizeof(unsigned int) * lessSize,
                              hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[CHKALLTEXTURE.CPP]: hipMemcpy d_index2 Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = hipMemcpy((void *) disturb, (void *) d_disturb, sizeof(bool), hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[CHKALLTEXTURE.CPP]: hipMemcpy disturb Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }

        char prefix1[64], prefix2[64];
        snprintf(prefix1, 64, "AllTexture_T1_%d_%d", baseCore, testCore);
        snprintf(prefix2, 64, "AllTexture_T2_%d_%d", baseCore, testCore);

        createOutputFile((int) TextureN, lessSize, h_index1, h_timeinfo1, avgOut1, potMissesOut1, prefix1);
        createOutputFile((int) TextureN, lessSize, h_index2, h_timeinfo2, avgOut2, potMissesOut2, prefix2);
    } while(false);

    // Free Texture Objects
    if (textureBinded) {
        error_id = hipDestroyTextureObject(tex1);
        error_id = hipDestroyTextureObject(tex2);
    }

    bool ret = false;
    if (disturb != nullptr) {
        ret = *disturb;
        free(disturb);
    }

    // Free Memory on GPU
    if (d_index1 != nullptr) {
        error_id = hipFree(d_index1);
    }

    if (d_index2 != nullptr) {
        error_id = hipFree(d_index2);
    }

    if (duration1 != nullptr) {
        error_id = hipFree(duration1);
    }

    if (duration2 != nullptr) {
        error_id = hipFree(duration2);
    }

    if (d_a != nullptr) {
        error_id = hipFree(d_a);
    }

    if (d_disturb != nullptr) {
        error_id = hipFree(d_disturb);
    }

    // Free Memory on Host
    if (h_index1 != nullptr) {
        free(h_index1);
    }

    if (h_index2 != nullptr) {
        free(h_index2);
    }

    if (h_a != nullptr) {
        free(h_a);
    }

    if (time1 != nullptr) {
        time1[0] = h_timeinfo1;
    } else {
        free(h_timeinfo1);
    }

    if (time2 != nullptr) {
        time2[0] = h_timeinfo2;
    } else {
        free(h_timeinfo2);
    }

    error_id = hipDeviceReset();
    return ret;
}

#endif //CUDATEST_ALLTEXTURE
