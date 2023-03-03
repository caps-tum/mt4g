
#ifndef CUDATEST_ROSHARETEXTURE
#define CUDATEST_ROSHARETEXTURE

# include <cstdio>
# include <cstdint>

# include "cuda.h"

__global__ void chkROShareTexture(cudaTextureObject_t tex, unsigned int RON, unsigned int TextureN, const unsigned int* __restrict__ myArrayReadOnly,
                                  unsigned int * durationRO, unsigned int * durationTexture, unsigned int *indexRO, unsigned int *indexTexture,
                                  bool* isDisturbed) {
    *isDisturbed = false;

    unsigned int start_time, end_time;
    unsigned int j = 0;
    int j2 = 0;
    __shared__ long long s_tvalueRO[LESS_SIZE];
    __shared__ unsigned int s_indexRO[LESS_SIZE];
    __shared__ long long s_tvalueTexture[LESS_SIZE];
    __shared__ unsigned int s_indexTexture[LESS_SIZE];

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int k = 0; k < LESS_SIZE; k++) {
            s_indexRO[k] = 0;
            s_tvalueRO[k] = 0;
        }
    }

    __syncthreads();

    if (threadIdx.x == 1) {
        for(int k=0; k < LESS_SIZE; k++) {
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

    if (threadIdx.x == 1){
        for (int k = 0; k < TextureN; k++) {
            j2 = tex1Dfetch<int>(tex, j2);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        //second round
        for (int k = 0; k < LESS_SIZE; k++) {
            start_time = clock();
            j = __ldg(&myArrayReadOnly[j]);
            s_indexRO[k] = j;
            end_time = clock();
            s_tvalueRO[k] = end_time - start_time;
        }
    }

    __syncthreads();

    if (threadIdx.x == 1){
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
            indexRO[k] = s_indexRO[k];
            durationRO[k] = s_tvalueRO[k];
            if (durationRO[k] > 3000) {
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

bool launchROBenchmarkReferenceValue(int N, int stride, double *avgOut, unsigned int* potMissesOut, unsigned int** time, int* error) {
    return launchROBenchmark(N, stride, avgOut, potMissesOut, time, error);
}

bool launchTextureBenchmarkReferenceValue(int N, int stride, double *avgOut, unsigned int* potMissesOut, unsigned int** time, int* error) {
    return launchTextureBenchmark(N, stride, avgOut, potMissesOut, time, error);
}

bool launchBenchmarkChkROShareTexture(unsigned int RO_N, unsigned int TextureN, double *avgOutRO, double* avgOutTexture, unsigned int* potMissesOutRO,
                                      unsigned int* potMissesOutTexture, unsigned int **timeRO, unsigned int **timeTexture, int* error) {
    cudaDeviceReset();
    cudaError_t error_id;

    int *h_aTexture = nullptr, *d_aTexture = nullptr;
    unsigned int *h_indexRO = nullptr, *h_indexTexture = nullptr, *h_timeinfoRO = nullptr, *h_timeinfoTexture = nullptr, *h_aRO = nullptr,
    *durationRO = nullptr, *durationTexture = nullptr, *d_indexRO = nullptr, *d_indexTexture = nullptr, *d_aRO = nullptr;
    bool *disturb = nullptr, *d_disturb = nullptr;
    bool bindedTexture = false;
    cudaTextureObject_t  tex = 0;

    do {
        // Allocate Memory on Host
        h_indexRO = (unsigned int *) malloc(sizeof(unsigned int) * LESS_SIZE);
        if (h_indexRO == nullptr) {
            printf("[CHKROSHARETEXTURE.CUH]: malloc h_indexRO Error\n");
            *error = 1;
            break;
        }

        h_indexTexture = (unsigned int *) malloc(sizeof(unsigned int) * LESS_SIZE);
        if (h_indexTexture == nullptr) {
            printf("[CHKROSHARETEXTURE.CUH]: malloc h_indexTexture Error\n");
            *error = 1;
            break;
        }

        h_timeinfoRO = (unsigned int *) malloc(sizeof(unsigned int) * LESS_SIZE);
        if (h_timeinfoRO == nullptr) {
            printf("[CHKROSHARETEXTURE.CUH]: malloc h_timeinfoRO Error\n");
            *error = 1;
            break;
        }

        h_timeinfoTexture = (unsigned int *) malloc(sizeof(unsigned int) * LESS_SIZE);
        if (h_timeinfoTexture == nullptr) {
            printf("[CHKROSHARETEXTURE.CUH]: malloc h_timeinfoTexture Error\n");
            *error = 1;
            break;
        }

        disturb = (bool *) malloc(sizeof(bool));
        if (disturb == nullptr) {
            printf("[CHKROSHARETEXTURE.CUH]: malloc disturb Error\n");
            *error = 1;
            break;
        }

        h_aTexture = (int *) malloc(sizeof(int) * (TextureN));
        if (h_aTexture == nullptr) {
            printf("[CHKROSHARETEXTURE.CUH]: malloc h_aTexture Error\n");
            *error = 1;
            break;
        }

        h_aRO = (unsigned int *) malloc(sizeof(unsigned int) * (RO_N));
        if (h_aRO == nullptr) {
            printf("[CHKROSHARETEXTURE.CUH]: malloc h_aRO Error\n");
            *error = 1;
            break;
        }

        // Allocate Memory on GPU
        error_id = cudaMalloc((void **) &durationRO, sizeof(unsigned int) * LESS_SIZE);
        if (error_id != cudaSuccess) {
            printf("[CHKROSHARETEXTURE.CUH]: cudaMalloc durationRO Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &durationTexture, sizeof(unsigned int) * LESS_SIZE);
        if (error_id != cudaSuccess) {
            printf("[CHKROSHARETEXTURE.CUH]: cudaMalloc durationTexture Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_indexRO, sizeof(unsigned int) * LESS_SIZE);
        if (error_id != cudaSuccess) {
            printf("[CHKROSHARETEXTURE.CUH]: cudaMalloc d_indexRO Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_indexTexture, sizeof(unsigned int) * LESS_SIZE);
        if (error_id != cudaSuccess) {
            printf("[CHKROSHARETEXTURE.CUH]: cudaMalloc d_indexTexture Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_disturb, sizeof(bool));
        if (error_id != cudaSuccess) {
            printf("[CHKROSHARETEXTURE.CUH]: cudaMalloc disturb Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_aTexture, sizeof(int) * (TextureN));
        if (error_id != cudaSuccess) {
            printf("[CHKROSHARETEXTURE.CUH]: cudaMalloc d_aTexture Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_aRO, sizeof(unsigned int) * (RO_N));
        if (error_id != cudaSuccess) {
            printf("[CHKROSHARETEXTURE.CUH]: cudaMalloc d_aRO Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        // Initialize p-chase arrays
        for (int i = 0; i < TextureN; i++) {
            h_aTexture[i] = (i + 1) % (int)TextureN;
        }

        for (int i = 0; i < RO_N; i++) {
            h_aRO[i] = (i + 1) % RO_N;
        }

        // Copy arrays from GPU to Host
        error_id = cudaMemcpy(d_aTexture, h_aTexture, sizeof(int) * TextureN, cudaMemcpyHostToDevice);
        if (error_id != cudaSuccess) {
            printf("[CHKROSHARETEXTURE.CUH]: cudaMemcpy d_aTexture Error: %s\n", cudaGetErrorString(error_id));
            *error = 3;
            break;
        }

        /* copy array elements from CPU to GPU */
        error_id = cudaMemcpy(d_aRO, h_aRO, sizeof(unsigned int) * RO_N, cudaMemcpyHostToDevice);
        if (error_id != cudaSuccess) {
            printf("[CHKROSHARETEXTURE.CUH]: cudaMemcpy d_aRO Error: %s\n", cudaGetErrorString(error_id));
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
            printf("[CHKROSHARETEXTURE.CUH]: cudaCreateTextureObject Error: %s\n", cudaGetErrorString(error_id));
            *error = 4;
            bindedTexture = false;
            break;
        }
        cudaDeviceSynchronize();

        // Launch Kernel function
        dim3 Db = dim3(2);
        dim3 Dg = dim3(1, 1, 1);
        chkROShareTexture<<<Dg, Db>>>(tex, RO_N, TextureN, d_aRO, durationRO, durationTexture, d_indexRO, d_indexTexture, d_disturb);

        cudaDeviceSynchronize();
        error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
            printf("[CHKROSHARETEXTURE.CUH]: Kernel launch/execution Error: %s\n", cudaGetErrorString(error_id));
            *error = 5;
            break;
        }

        // Copy results from GPU to Host
        error_id = cudaMemcpy((void *) h_timeinfoRO, (void *) durationRO, sizeof(unsigned int) * LESS_SIZE,
                              cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKROSHARETEXTURE.CUH]: cudaMemcpy durationRO Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) h_timeinfoTexture, (void *) durationTexture, sizeof(unsigned int) * LESS_SIZE,
                              cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKROSHARETEXTURE.CUH]: cudaMemcpy durationTexture Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) h_indexRO, (void *) d_indexRO, sizeof(unsigned int) * LESS_SIZE, cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKROSHARETEXTURE.CUH]: cudaMemcpy d_indexRO Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) h_indexTexture, (void *) d_indexTexture, sizeof(unsigned int) * LESS_SIZE,
                              cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKROSHARETEXTURE.CUH]: cudaMemcpy d_indexTexture Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        error_id = cudaMemcpy((void *) disturb, (void *) d_disturb, sizeof(bool), cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[CHKROSHARETEXTURE.CUH]: cudaMemcpy disturb Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        createOutputFile((int) RO_N, LESS_SIZE, h_indexRO, h_timeinfoRO, avgOutRO, potMissesOutRO, "ShareROTextureRO_");
        createOutputFile((int) TextureN, LESS_SIZE, h_indexTexture, h_timeinfoTexture, avgOutTexture, potMissesOutTexture,
                         "ShareROTextureTexture_");
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
    if (d_indexRO != nullptr) {
        cudaFree(d_indexRO);
    }

    if (d_indexTexture != nullptr) {
        cudaFree(d_indexTexture);
    }

    if (durationRO != nullptr) {
        cudaFree(durationRO);
    }

    if (durationTexture != nullptr) {
        cudaFree(durationTexture);
    }

    if (d_aTexture != nullptr) {
        cudaFree(d_aTexture);
    }

    if (d_aRO != nullptr) {
        cudaFree(d_aRO);
    }

    if (d_disturb != nullptr) {
        cudaFree(d_disturb);
    }

    // Free Memory on Host
    if (h_indexRO != nullptr) {
        free(h_indexRO);
    }

    if (h_indexTexture != nullptr) {
        free(h_indexTexture);
    }

    if (h_aRO != nullptr) {
        free(h_aRO);
    }

    if (h_aTexture != nullptr) {
        free(h_aTexture);
    }

    if (h_timeinfoRO != nullptr) {
        if (timeRO != nullptr) {
            timeRO[0] = h_timeinfoRO;
        } else {
            free(h_timeinfoRO);
        }
    }

    if (h_timeinfoTexture != nullptr) {
        if (timeTexture != nullptr) {
            timeTexture[0] = h_timeinfoTexture;
        } else {
            free(h_timeinfoTexture);
        }
    }

    cudaDeviceReset();
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
    unsigned int ROSizeInInt = (measuredSizeRO-sub) >> 2; // / 4;
    unsigned int TextureSizeInInt = (measuredSizeTexture-sub) >> 2; // / 4;

    double* avgFlowRefRO = (double*) malloc(sizeof(double));
    unsigned int *potMissesFlowRefRO = (unsigned int*) malloc(sizeof(unsigned int));
    unsigned int** timeRefRO = (unsigned int**) malloc(sizeof(unsigned int*));

    double* avgFlowRefTxt = (double*) malloc(sizeof(double));
    unsigned int *potMissesFlowRefTxt = (unsigned int*) malloc(sizeof(unsigned int));
    unsigned int** timeRefTxt = (unsigned int**) malloc(sizeof(unsigned int*));

    double* avgFlow = (double*) malloc(sizeof(double)  * 2);
    unsigned int *potMissesFlow = (unsigned int*) malloc(sizeof(unsigned int) * 2);
    unsigned int** time = (unsigned int**) malloc(sizeof(unsigned int*) * 2);
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
        dist = launchROBenchmarkReferenceValue((int) ROSizeInInt, 1, avgFlowRefRO, potMissesFlowRefRO, timeRefRO, &error);
        if (error != 0) {
            FreeMeasureROTxtResources()
            printErrorCodeInformation(error);
            exit(error);
        }
        --n;
    }

    dist = true;
    n = 5;
    while(dist && n > 0) {
        int error = 0;
        dist = launchTextureBenchmarkReferenceValue((int) TextureSizeInInt, 1, avgFlowRefTxt, potMissesFlowRefTxt, timeRefTxt, &error);
        if (error != 0) {
            FreeMeasureROTxtResources()
            printErrorCodeInformation(error);
            exit(error);
        }
        --n;
    }

    dist = true;
    n = 5;
    while(dist && n > 0) {
        int error = 0;
        dist = launchBenchmarkChkROShareTexture(ROSizeInInt, TextureSizeInInt, &avgFlow[0], &avgFlow[1], &potMissesFlow[0],
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