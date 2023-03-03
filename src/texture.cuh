
#ifndef CUDATEST_TEXT
#define CUDATEST_TEXT

#include <cstdio>

# include "eval.h"
# include "GPU_resources.cuh"
# include "binarySearch.h"

__global__ void texture_size (cudaTextureObject_t tex, unsigned int size, unsigned int *duration, unsigned int *index, bool* isDisturbed);

bool launchTextureBenchmark(int N, int stride, double* avgOut, unsigned int* potMissesOut, unsigned int** time, int* error);

CacheSizeResult measure_texture() {
    int absoluteLowerBoundary = 1024;
    int absoluteUpperBoundary = 1024 << 10; // 1024 * 1024
    int widenBounds = 8;

    int bounds[2] = {absoluteLowerBoundary, absoluteUpperBoundary};
    getBoundaries(launchTextureBenchmark, bounds, 5);
#ifdef IsDebug
    fprintf(out, "Got Boundaries: %d...%d\n", bounds[0], bounds[1]);
#endif //IsDebug
    printf("Got Boundaries: %d...%d\n", bounds[0], bounds[1]);

    int cp = -1;
    int begin = bounds[0] - widenBounds;
    int end = bounds[1] + widenBounds;
    int stride = 1;
    int arrayIncrease = 1;

    while (cp == -1 && begin >= absoluteLowerBoundary / sizeof(int) - widenBounds && end <= absoluteUpperBoundary / sizeof(int) + widenBounds) {
        cp = wrapBenchmarkLaunch(launchTextureBenchmark, begin, end, stride, arrayIncrease, "Texture");

        // If region really did not contain a change point, widen region further
        if (cp == -1) {
            begin = begin - (end - begin);
            end = end + (end - begin);
#ifdef IsDebug
            fprintf(out, "\nGot Boundaries: %d...%d\n", begin, end);
#endif //IsDebug
            printf("\nGot Boundaries: %d...%d\n", begin, end);
        }
    }

    CacheSizeResult result;
    int cacheSizeInInt = (begin + cp * arrayIncrease);
    result.CacheSize = (cacheSizeInInt << 2); // * 4);
    result.realCP = cp > 0;
    result.maxSizeBenchmarked = end << 2; // * 4;
    return result;
}

bool launchTextureBenchmark(int N, int stride, double* avgOut, unsigned int* potMissesOut, unsigned int** time, int* error) {
    cudaError_t error_id;

    int* h_a = nullptr, *d_a = nullptr;
    unsigned int *h_duration = nullptr, *h_index = nullptr, *d_index = nullptr, *d_duration = nullptr;
    bool* disturb = nullptr, *d_disturb = nullptr;
    unsigned int size =  N * sizeof(int);
    cudaTextureObject_t  tex = 0;
    bool bindedTexture = false;

    do {
        // Allocate Memory On Host
        h_a = (int *) malloc(size);
        if (h_a == nullptr) {
            printf("[TEXTURE.CUH]: malloc h_a Error\n");
            *error = 1;
            break;
        }

        h_duration = (unsigned int *) malloc(MEASURE_SIZE * sizeof(unsigned int));
        if (h_duration == nullptr) {
            printf("[TEXTURE.CUH]: malloc h_duration Error\n");
            *error = 1;
            break;
        }

        h_index = (unsigned int *) malloc(MEASURE_SIZE * sizeof(unsigned int));
        if (h_index == nullptr) {
            printf("[TEXTURE.CUH]: malloc h_index Error\n");
            *error = 1;
            break;
        }

        disturb = (bool *) malloc(sizeof(bool));
        if (disturb == nullptr) {
            printf("[TEXTURE.CUH]: malloc disturb Error\n");
            *error = 1;
            break;
        }

        // Allocate Memory on GPU
        error_id = cudaMalloc((void **) &d_a, size);
        if (error_id != cudaSuccess) {
            printf("[TEXTURE.CUH]: cudaMalloc d_a Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc(&d_index, MEASURE_SIZE * sizeof(unsigned int));
        if (error_id != cudaSuccess) {
            printf("[TEXTURE.CUH]: cudaMalloc d_index Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc(&d_duration, MEASURE_SIZE * sizeof(unsigned int));
        if (error_id != cudaSuccess) {
            printf("[TEXTURE.CUH]: cudaMalloc duration Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_disturb, sizeof(bool));
        if (error_id != cudaSuccess) {
            printf("[TEXTURE.CUH]: cudaMalloc disturb Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        // Initialize p-chase array
        for (int i = 0; i < N; i++) {
            h_a[i] = (i + stride) % N;
        }

        // Copy array to GPU
        error_id = cudaMemcpy((void *) d_a, (void *) h_a, size, cudaMemcpyHostToDevice);
        if (error_id != cudaSuccess) {
            printf("[TEXTURE.CUH]: cudaMemcpy d_a Error: %s\n", cudaGetErrorString(error_id));
            *error = 3;
            break;
        }

        // Create Texture Object
        cudaResourceDesc resDesc = {};
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeLinear;
        resDesc.res.linear.devPtr = d_a;
        resDesc.res.linear.desc.f = cudaChannelFormatKindSigned;
        resDesc.res.linear.desc.x = 32; // bits per channel
        resDesc.res.linear.sizeInBytes = N*sizeof(int);

        cudaTextureDesc texDesc = {};
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = cudaReadModeElementType;

        cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);
        bindedTexture = true;

        cudaDeviceSynchronize();

        error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
            printf("[TEXTURE.CUH]: cudaCreateTextureObject Error: %s\n", cudaGetErrorString(error_id));
            *error = 4;
            bindedTexture = false;
            break;
        }

        // Launch Kernel function
        dim3 Db = dim3(1);
        dim3 Dg = dim3(1, 1, 1);
        texture_size <<<Dg, Db>>>(tex, size, d_duration, d_index, d_disturb);

        cudaDeviceSynchronize();

        error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
            printf("[TEXTURE.CUH]: Kernel launch/execution Error: %s\n", cudaGetErrorString(error_id));
            *error = 5;
            break;
        }

        cudaDeviceSynchronize();

        // Copy results from GPU to Host
        error_id = cudaMemcpy((void *) h_index, (void *) d_index, MEASURE_SIZE * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[TEXTURE.CUH]: cudaMemcpy d_index Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }
        error_id = cudaMemcpy((void *) h_duration, (void *) d_duration, MEASURE_SIZE * sizeof(unsigned int),cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[TEXTURE.CUH]: cudaMemcpy duration Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }
        error_id = cudaMemcpy((void *) disturb, (void *) d_disturb, sizeof(bool), cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[TEXTURE.CUH]: cudaMemcpy disturb Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }

        if (!*disturb)
            createOutputFile(N, MEASURE_SIZE, h_index, h_duration, avgOut, potMissesOut, "Texture_");

    } while(false);

    // Free texture
    if (bindedTexture) {
        cudaDestroyTextureObject(tex);
    }

    cudaDeviceSynchronize();

    // Free Memory on GPU
    if (d_a != nullptr) {
        cudaFree(d_a);
    }

    if (d_duration != nullptr) {
        cudaFree(d_duration);
    }

    if (d_index != nullptr) {
        cudaFree(d_index);
    }

    if (d_disturb != nullptr) {
        cudaFree(d_disturb);
    }

    // Free Memory on Host
    bool ret = false;
    if (disturb != nullptr) {
        ret = *disturb;
        free(disturb);
    }

    if (h_a != nullptr) {
        free(h_a);
    }

    if (h_duration != nullptr) {
        if (time != nullptr) {
            time[0] = h_duration;
        } else {
            free(h_duration);
        }
    }

    if (h_index != nullptr) {
        free(h_index);
    }

    return ret;
}

__global__ void texture_size (cudaTextureObject_t tex, unsigned int size, unsigned int *duration, unsigned int *index, bool* isDisturbed) {

    bool dist = false;

   unsigned int start, end;
   int j = 0;

	for (int k=0; k< MEASURE_SIZE; k++) {
        s_index[k] = 0;
        s_tvalue[k] = 0;
    }

    // First round
    for (int k = 0; k < size; k++) {
        j=tex1Dfetch<int>(tex, j);
    }

    // Second round
	for (int k=0; k < MEASURE_SIZE; k++) {
        start=clock();
        j=tex1Dfetch<int>(tex, j);
        s_index[k] = j;
        end=clock();
        s_tvalue[k] = (end -start);
    }

    for (int k=0; k < MEASURE_SIZE; k++){
        if (s_tvalue[k] > 2000) {
            dist = true;
        }
        duration[k] = s_tvalue[k];
        index[k] = s_index[k];
    }
    *isDisturbed = dist;
}

#endif //CUDATEST_TEXT