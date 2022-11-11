
#ifndef CUDATEST_TWOTEXTURE
#define CUDATEST_TWOTEXTURE

# include <cstdio>
# include <cstdint>

# include "cuda.h"

__global__ void chkTwoTexture(cudaTextureObject_t tex1, cudaTextureObject_t tex2, unsigned int TextureN, unsigned int * durationTxt1, unsigned int * durationTxt2, unsigned int *indexTxt1, unsigned int *indexTxt2,
                              bool* isDisturbed) {
    *isDisturbed = false;

    unsigned int start_time, end_time;
    int j = 0;
    __shared__ long long s_tvalueTxt1[lessSize];
    __shared__ unsigned int s_indexTxt1[lessSize];
    __shared__ long long s_tvalueTxt2[lessSize];
    __shared__ unsigned int s_indexTxt2[lessSize];

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
    cudaDeviceReset();
    cudaError_t error_id;

    int* h_aTexture = nullptr, *d_aTexture = nullptr;
    unsigned int *h_indexTexture1 = nullptr, *h_indexTexture2 = nullptr, *h_timeinfoTexture1 = nullptr, *h_timeinfoTexture2 = nullptr,
    *durationTxt1 = nullptr, *durationTxt2 = nullptr, *d_indexTxt1 = nullptr, *d_indexTxt2 = nullptr;
    bool *disturb = nullptr, *d_disturb = nullptr;
    bool bindedTexture = false;
    cudaTextureObject_t tex1 = 0, tex2 = 0;

    do {
        // Allocate Memory on Host
        h_indexTexture1 = (unsigned int *) malloc(sizeof(unsigned int) * lessSize);
        if (h_indexTexture1 == nullptr) {
            printf("[CHKTWOTEXTURE.CUH]: malloc h_indexTexture1 Error\n");
            *error = 1;
            break;
        }

        h_indexTexture2 = (unsigned int *) malloc(sizeof(unsigned int) * lessSize);
        if (h_indexTexture2 == nullptr) {
            printf("[CHKTWOTEXTURE.CUH]: malloc h_indexTexture2 Error\n");
            *error = 1;
            break;
        }

        h_timeinfoTexture1 = (unsigned int *) malloc(sizeof(unsigned int) * lessSize);
        if (h_timeinfoTexture1 == nullptr) {
            printf("[CHKTWOTEXTURE.CUH]: malloc h_timeinfoTexture1 Error\n");
            *error = 1;
            break;
        }

        h_timeinfoTexture2 = (unsigned int *) malloc(sizeof(unsigned int) * lessSize);
        if (h_timeinfoTexture2 == nullptr) {
            printf("[CHKTWOTEXTURE.CUH]: malloc h_timeinfoTexture2 Error\n");
            *error = 1;
            break;
        }

        disturb = (bool *) malloc(sizeof(bool));
        if (disturb == nullptr) {
            printf("[CHKTWOTEXTURE.CUH]: malloc disturb Error\n");
            *error = 1;
            break;
        }

        h_aTexture = (int*) malloc(sizeof(int) * (TextureN));
        if (h_aTexture == nullptr) {
            printf("[CHKTWOTEXTURE.CUH]: malloc h_aTexture Error\n");
            *error = 1;
            break;
        }

        // Allocate Memory on GPU
        error_id = cudaMalloc((void **) &durationTxt1, sizeof(unsigned int) * lessSize);
        if (error_id != cudaSuccess) {
            printf("[CHKTWOTEXTURE.CUH]: cudaMalloc durationTxt1 Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &durationTxt2, sizeof(unsigned int) * lessSize);
        if (error_id != cudaSuccess) {
            printf("[CHKTWOTEXTURE.CUH]: cudaMalloc durationTxt2 Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_indexTxt1, sizeof(unsigned int) * lessSize);
        if (error_id != cudaSuccess) {
            printf("[CHKTWOTEXTURE.CUH]: cudaMalloc d_indextxt1 Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_indexTxt2, sizeof(unsigned int) * lessSize);
        if (error_id != cudaSuccess) {
            printf("[CHKTWOTEXTURE.CUH]: cudaMalloc d_indexTxt2 Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_disturb, sizeof(bool));
        if (error_id != cudaSuccess) {
            printf("[CHKTWOTEXTURE.CUH]: cudaMalloc disturb Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_aTexture, sizeof(int) * (TextureN));
        if (error_id != cudaSuccess) {
            printf("[CHKTWOTEXTURE.CUH]: cudaMalloc d_aTexture Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        // Initialize p-chase array
        for (int i = 0; i < TextureN; i++) {
            h_aTexture[i] = (i + 1) % (int)TextureN;
        }

        // Copy array from Host to GPU
        error_id = cudaMemcpy(d_aTexture, h_aTexture, sizeof(int) * TextureN, cudaMemcpyHostToDevice);
        if (error_id != cudaSuccess) {
            printf("[CHKTWOTEXTURE.CUH]: cudaMemcpy d_aTexture Error: %s\n", cudaGetErrorString(error_id));
            *error = 3;
            break;
        }

        // Create Texture Object
        cudaResourceDesc resDesc = {};
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeLinear;
        resDesc.res.linear.devPtr = d_aTexture;
        resDesc.res.linear.desc.f = cudaChannelFormatKindSigned;
        resDesc.res.linear.desc.x = 32; // bits per channel
        resDesc.res.linear.sizeInBytes = TextureN*sizeof(int);

        cudaTextureDesc texDesc = {};
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = cudaReadModeElementType;

        cudaCreateTextureObject(&tex1, &resDesc, &texDesc, nullptr);
        cudaCreateTextureObject(&tex2, &resDesc, &texDesc, nullptr);
        bindedTexture = true;

        cudaDeviceSynchronize();

        error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
            printf("[CHKTWOTEXTURE.CUH]: cudaCreateTextureObject Error: %s\n", cudaGetErrorString(error_id));
            *error = 4;
            bindedTexture = false;
            break;
        }
        cudaDeviceSynchronize();

        // Launch Kernel function
        dim3 Db = dim3(2);
        dim3 Dg = dim3(1, 1, 1);
        chkTwoTexture<<<Dg, Db>>>(tex1, tex2, TextureN, durationTxt1, durationTxt2, d_indexTxt1, d_indexTxt2, d_disturb);

        cudaDeviceSynchronize();
        error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
            printf("[CHKTWOTEXTURE.CUH]: Kernel launch/execution Error: %s\n", cudaGetErrorString(error_id));
            *error = 5;
            break;
        }

        // Copy results from GPU to Host
        error_id = cudaMemcpy((void *) h_timeinfoTexture1, (void *) durationTxt1, sizeof(unsigned int) * lessSize,cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKTWOTEXTURE.CUH]: cudaMemcpy durationTxt1 Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) h_timeinfoTexture2, (void *) durationTxt2, sizeof(unsigned int) * lessSize,cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKTWOTEXTURE.CUH]: cudaMemcpy durationTxt2 Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) h_indexTexture1, (void *) d_indexTxt1, sizeof(unsigned int) * lessSize,cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKTWOTEXTURE.CUH]: cudaMemcpy d_indexTxt1 Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) h_indexTexture2, (void *) d_indexTxt2, sizeof(unsigned int) * lessSize,cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKTWOTEXTURE.CUH]: cudaMemcpy d_indexTxt2 Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) disturb, (void *) d_disturb, sizeof(bool), cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKTWOTEXTURE.CUH]: cudaMemcpy disturb Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        createOutputFile((int) TextureN, lessSize, h_indexTexture1, h_timeinfoTexture1, avgOutTxt1, potMissesOutTxt1, "TwoTexture1_");
        createOutputFile((int) TextureN, lessSize, h_indexTexture2, h_timeinfoTexture2, avgOutTxt2, potMissesOutTxt2, "TwoTexture2_");

    } while(false);

    // Free Texture Object
    if (bindedTexture) {
        cudaDestroyTextureObject(tex1);
        cudaDestroyTextureObject(tex2);
    }

    bool ret = false;
    if (disturb != nullptr) {
        ret = *disturb;
        free(disturb);
    }

    // Free Memory on GPU
    if (d_indexTxt1 != nullptr) {
        cudaFree(d_indexTxt1);
    }

    if (d_indexTxt2 != nullptr) {
        cudaFree(d_indexTxt2);
    }

    if (durationTxt1 != nullptr) {
        cudaFree(durationTxt1);
    }

    if (durationTxt2 != nullptr) {
        cudaFree(durationTxt2);
    }

    if (d_aTexture != nullptr) {
        cudaFree(d_aTexture);
    }

    if (d_disturb != nullptr) {
        cudaFree(d_disturb);
    }

    // Free Memory on Host
    if (h_indexTexture1 != nullptr) {
        free(h_indexTexture1);
    }

    if (h_indexTexture2 != nullptr) {
        free(h_indexTexture2);
    }

    if (h_aTexture != nullptr) {
        free(h_aTexture);
    }

    if (h_timeinfoTexture1 != nullptr) {
        if (timeTxt1 != nullptr) {
            timeTxt1[0] = h_timeinfoTexture1;
        } else {
            free(h_timeinfoTexture1);
        }
    }

    if (h_timeinfoTexture2) {
        if (timeTxt2 != nullptr) {
            timeTxt2[0] = h_timeinfoTexture2;
        } else {
            free(h_timeinfoTexture2);
        }
    }

    cudaDeviceReset();
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