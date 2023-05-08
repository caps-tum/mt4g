
#ifndef CUDATEST_L1SHARETEXTURE
#define CUDATEST_L1SHARETEXTURE

# include <cstdio>
# include <cstdint>

# include "cuda.h"

__global__ void chkL1ShareTexture(cudaTextureObject_t tex, unsigned int L1_N, unsigned int TextureN, unsigned int* myArray,
                                  unsigned int * durationL1, unsigned int * durationTexture, unsigned int *indexL1, unsigned int *indexTexture,
                                  bool* isDisturbed) {
    *isDisturbed = false;

    unsigned int start_time, end_time;
    unsigned int j = 0;
    int j2 = 0;
    __shared__ long long s_tvalueL1[LESS_SIZE];
    __shared__ unsigned int s_indexL1[LESS_SIZE];
    __shared__ long long s_tvalueTexture[LESS_SIZE];
    __shared__ unsigned int s_indexTexture[LESS_SIZE];

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < LESS_SIZE; k++) {
            s_indexL1[k] = 0;
            s_tvalueL1[k] = 0;
        }
    }

    __syncthreads();

    if (threadIdx.x == 1) {
        for(int k=0; k < LESS_SIZE; k++) {
            s_indexTexture[k] = 0;
            s_tvalueTexture[k] = 0;
        }
    }

    unsigned int* ptr;
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < L1_N; k++) {
            ptr = myArray + j;
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(j) : "l"(ptr) : "memory");
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
        for (int k = 0; k < LESS_SIZE; k++) {
            ptr = myArray + j;
            start_time = clock();
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(j) : "l"(ptr) : "memory");
            s_indexL1[k] = j;
            end_time = clock();
            s_tvalueL1[k] = end_time - start_time;
        }
    }

    __syncthreads();

    if (threadIdx.x == 1) {
        for (int k = 0; k < LESS_SIZE; k++) {
            start_time=clock();
            j2=tex1Dfetch<int>(tex, j2);
            s_indexTexture[k] = j2;
            end_time=clock();
            s_tvalueTexture[k] = (end_time - start_time);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < LESS_SIZE; k++) {
            indexL1[k] = s_indexL1[k];
            durationL1[k] = s_tvalueL1[k];
            if (durationL1[k] > 3000) {
                *isDisturbed = true;
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == 1) {
        for(int k=0; k<LESS_SIZE; k++){
            indexTexture[k]= s_indexTexture[k];
            durationTexture[k] = s_tvalueTexture[k];
            if (durationTexture[k] > 3000) {
                *isDisturbed = true;
            }
        }
    }
}

bool launchL1BenchmarkReferenceValue(int N, int stride, double *avgOut, unsigned int* potMissesOut, unsigned int** time, int* error) {
    return launchL1KernelBenchmark(N, stride, avgOut, potMissesOut, time, error);
}

bool launchBenchmarkChkL1ShareTexture(unsigned int L1Data_N, unsigned int TextureN, double *avgOutL1Data, double* avgOutTexture, unsigned int* potMissesOutL1Data,
                                      unsigned int* potMissesOutTexture, unsigned int **timeL1Data, unsigned int **timeTexture, int* error) {
    cudaDeviceReset();
    cudaError_t error_id;

    int *h_aTexture = nullptr, *d_aTexture = nullptr;
    unsigned int *h_indexL1Data = nullptr, *h_indexTexture = nullptr, *h_timeinfoL1Data = nullptr, *h_timeinfoTexture = nullptr, *h_aL1 = nullptr,
    *durationL1 = nullptr, *durationTexture = nullptr, *d_indexL1 = nullptr, *d_indexTexture = nullptr, *d_aL1 = nullptr;
    bool *disturb = nullptr, *d_disturb = nullptr;
    bool bindedTexture = false;
    cudaTextureObject_t  tex = 0;

    do {
        // Allocate Memory on Host
        h_indexL1Data = (unsigned int *) malloc(sizeof(unsigned int) * LESS_SIZE);
        if (h_indexL1Data == nullptr) {
            printf("[CHKL1SHARETEXTURE.CUH]: malloc h_indexL1Data Error\n");
            *error = 1;
            break;
        }

        h_indexTexture = (unsigned int *) malloc(sizeof(unsigned int) * LESS_SIZE);
        if (h_indexTexture == nullptr) {
            printf("[CHKL1SHARETEXTURE.CUH]: malloc h_indexTexture Error\n");
            *error = 1;
            break;
        }

        h_timeinfoL1Data = (unsigned int *) malloc(sizeof(unsigned int) * LESS_SIZE);
        if (h_timeinfoL1Data == nullptr) {
            printf("[CHKL1SHARETEXTURE.CUH]: malloc h_timeinfoL1Data Error\n");
            *error = 1;
            break;
        }

        h_timeinfoTexture = (unsigned int *) malloc(sizeof(unsigned int) * LESS_SIZE);
        if (h_timeinfoTexture == nullptr) {
            printf("[CHKL1SHARETEXTURE.CUH]: malloc h_timeinfoTexture Error\n");
            *error = 1;
            break;
        }

        disturb = (bool *) malloc(sizeof(bool));
        if (disturb == nullptr){
            printf("[CHKL1SHARETEXTURE.CUH]: malloc disturb Error\n");
            *error = 1;
            break;
        }

        h_aTexture = (int *) malloc(sizeof(int) * (TextureN));
        if (h_aTexture == nullptr) {
            printf("[CHKL1SHARETEXTURE.CUH]: malloc h_aTexture Error\n");
            *error = 1;
            break;
        }

        h_aL1 = (unsigned int *) malloc(sizeof(unsigned int) * (L1Data_N));
        if (h_aL1 == nullptr) {
            printf("[CHKL1SHARETEXTURE.CUH]: malloc h_aL1 Error\n");
            *error = 1;
            break;
        }

        // Allocate Memory on GPU
        error_id = cudaMalloc((void **) &durationL1, sizeof(unsigned int) * LESS_SIZE);
        if (error_id != cudaSuccess) {
            printf("[CHKL1SHARETEXTURE.CUH]: cudaMalloc durationL1 Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &durationTexture, sizeof(unsigned int) * LESS_SIZE);
        if (error_id != cudaSuccess) {
            printf("[CHKL1SHARETEXTURE.CUH]: cudaMalloc durationTexture Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_indexL1, sizeof(unsigned int) * LESS_SIZE);
        if (error_id != cudaSuccess) {
            printf("[CHKL1SHARETEXTURE.CUH]: cudaMalloc d_indexL1 Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_indexTexture, sizeof(unsigned int) * LESS_SIZE);
        if (error_id != cudaSuccess) {
            printf("[CHKL1SHARETEXTURE.CUH]: cudaMalloc d_indexTexture Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_disturb, sizeof(bool));
        if (error_id != cudaSuccess) {
            printf("[CHKL1SHARETEXTURE.CUH]: cudaMalloc disturb Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_aTexture, sizeof(unsigned int) * (TextureN));
        if (error_id != cudaSuccess) {
            printf("[CHKL1SHARETEXTURE.CUH]: cudaMalloc d_aTexture Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_aL1, sizeof(unsigned int) * (L1Data_N));
        if (error_id != cudaSuccess) {
            printf("[CHKL1SHARETEXTURE.CUH]: cudaMalloc d_aL1 Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        // Initialize p-chase arrays
        for (int i = 0; i < TextureN; i++) {
            h_aTexture[i] = (i + 1) % (int)TextureN;
        }

        for (int i = 0; i < L1Data_N; i++) {
            //original:
            h_aL1[i] = (i + 1) % L1Data_N;
        }

        // Copy arrays from Host to GPU
        error_id = cudaMemcpy(d_aTexture, h_aTexture, sizeof(int) * TextureN, cudaMemcpyHostToDevice);
        if (error_id != cudaSuccess) {
            printf("[CHKL1SHARETEXTURE.CUH]: cudaMemcpy d_aTexture Error: %s\n", cudaGetErrorString(error_id));
            *error = 3;
            break;
        }

        error_id = cudaMemcpy(d_aL1, h_aL1, sizeof(unsigned int) * L1Data_N, cudaMemcpyHostToDevice);
        if (error_id != cudaSuccess) {
            printf("[CHKL1SHARETEXTURE.CUH]: cudaMemcpy d_aL1 Error: %s\n", cudaGetErrorString(error_id));
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

        cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);
        bindedTexture = true;

        cudaDeviceSynchronize();
        error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
            printf("[CHKL1SHARETEXTURE.CUH]: cudaCreateTextureObject Error: %s\n", cudaGetErrorString(error_id));
            *error = 4;
            bindedTexture = false;
            break;
        }
        cudaDeviceSynchronize();

        // Launch Kernel function
        dim3 Db = dim3(2);
        dim3 Dg = dim3(1, 1, 1);
        chkL1ShareTexture<<<Dg, Db>>>(tex, L1Data_N, TextureN, d_aL1, durationL1, durationTexture,
                                      d_indexL1, d_indexTexture,d_disturb);

        cudaDeviceSynchronize();
        error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
            printf("[CHKL1SHARETEXTURE.CUH]: Kernel launch/execution Error: %s\n", cudaGetErrorString(error_id));
            *error = 5;
            break;
        }

        // Copy results from GPU to Host
        error_id = cudaMemcpy((void *) h_timeinfoL1Data, (void *) durationL1, sizeof(unsigned int) * LESS_SIZE,
                              cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKL1SHARETEXTURE.CUH]: cudaMemcpy durationL1 Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) h_timeinfoTexture, (void *) durationTexture, sizeof(unsigned int) * LESS_SIZE,
                              cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKL1SHARETEXTURE.CUH]: cudaMemcpy durationTexture Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) h_indexL1Data, (void *) d_indexL1, sizeof(unsigned int) * LESS_SIZE,
                              cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKL1SHARETEXTURE.CUH]: cudaMemcpy d_indexL1 Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) h_indexTexture, (void *) d_indexTexture, sizeof(unsigned int) * LESS_SIZE,
                              cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKL1SHARETEXTURE.CUH]: cudaMemcpy d_indexTexture Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) disturb, (void *) d_disturb, sizeof(bool), cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKL1SHARETEXTURE.CUH]: cudaMemcpy disturb Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        createOutputFile((int) L1Data_N, LESS_SIZE, h_indexL1Data, h_timeinfoL1Data, avgOutL1Data, potMissesOutL1Data,
                         "ShareL1TextureRO_");
        createOutputFile((int) TextureN, LESS_SIZE, h_indexTexture, h_timeinfoTexture, avgOutTexture, potMissesOutTexture,
                         "ShareL1TextureTexture_");
    } while(false);

    // Free Texture Object
    if (bindedTexture) {
        cudaDestroyTextureObject(tex);
    }

    bool ret = false;
    if (disturb != nullptr) {
        ret = *disturb;
        free(disturb);
    }

    // Free Memory on GPU
    if (d_indexL1 != nullptr) {
        cudaFree(d_indexL1);
    }

    if (d_indexTexture != nullptr) {
        cudaFree(d_indexTexture);
    }

    if (durationL1 != nullptr) {
        cudaFree(durationL1);
    }

    if (durationTexture != nullptr) {
        cudaFree(durationTexture);
    }

    if (d_aTexture != nullptr) {
        cudaFree(d_aTexture);
    }

    if (d_aL1 != nullptr) {
        cudaFree(d_aL1);
    }

    if (d_disturb != nullptr) {
        cudaFree(d_disturb);
    }

    // Free Memory on Host
    if (h_indexL1Data != nullptr) {
        free(h_indexL1Data);
    }

    if (h_indexTexture != nullptr) {
        free(h_indexTexture);
    }

    if (h_aL1 != nullptr) {
        free(h_aL1);
    }

    if (h_aTexture != nullptr) {
        free(h_aTexture);
    }

    if (timeL1Data != nullptr) {
        timeL1Data[0] = h_timeinfoL1Data;
    } else {
        free(h_timeinfoL1Data);
    }

    if (timeTexture != nullptr) {
        timeTexture[0] = h_timeinfoTexture;
    } else {
        free(h_timeinfoTexture);
    }

    cudaDeviceReset();
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
    unsigned int TextureSizeInInt = (measuredSizeTexture-sub) >> 2;// / 4;

    double* avgFlowRefL1 = (double*) malloc(sizeof(double));
    unsigned int *potMissesFlowRefL1 = (unsigned int*) malloc(sizeof(unsigned int));
    unsigned int** timeRefL1 = (unsigned int**) malloc(sizeof(unsigned int*));

    double* avgFlowRefTxt = (double*) malloc(sizeof(double));
    unsigned int *potMissesFlowRefTxt = (unsigned int*) malloc(sizeof(unsigned int));
    unsigned int** timeRefTxt = (unsigned int**) malloc(sizeof(unsigned int*));

    double* avgFlow = (double*) malloc(sizeof(double)  * 2);
    unsigned int *potMissesFlow = (unsigned int*) malloc(sizeof(unsigned int) * 2);
    unsigned int** time = (unsigned int**) malloc(sizeof(unsigned int*) * 2);
    if (avgFlowRefL1 == nullptr || potMissesFlowRefL1 == nullptr || timeRefL1 == nullptr ||
        avgFlowRefTxt == nullptr || potMissesFlowRefTxt == nullptr ||timeRefTxt == nullptr ||
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
        dist = launchL1BenchmarkReferenceValue((int) L1SizeInInt, 1, avgFlowRefL1, potMissesFlowRefL1, timeRefL1, &error);
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
        dist = launchTextureBenchmarkReferenceValue((int) TextureSizeInInt, 1, avgFlowRefTxt, potMissesFlowRefTxt, timeRefTxt, &error);
        if (error != 0) {
            FreeMeasureL1TxtResources()
            printErrorCodeInformation(error);
            exit(error);
        }
        --n;
    }

    dist = true;
    n = 5;
    while(dist && n > 0) {
        int error = 0;
        dist = launchBenchmarkChkL1ShareTexture(L1SizeInInt, TextureSizeInInt, &avgFlow[0], &avgFlow[1], &potMissesFlow[0],
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