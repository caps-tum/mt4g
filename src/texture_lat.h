#include "hip/hip_runtime.h"

#ifndef CUDATEST_TEXTURE_LAT
#define CUDATEST_TEXTURE_LAT

# include <cstdio>

# include "hip/hip_runtime.h"
# include "eval.h"
# include "GPU_resources.h"

//texture<int, 1, hipReadModeElementType> tex_ref;

__global__ void texture_lat (hipTextureObject_t tex, int * my_array, int array_length, unsigned int * time);
__global__ void texture_lat_globaltimer (hipTextureObject_t tex, int * my_array, int array_length, unsigned int * time);

LatencyTuple launchTextureLatKernelBenchmark(int N, int stride, int* error);

LatencyTuple measure_Texture_Lat() {
    int stride = 1;
    int error = 0;
    LatencyTuple lat = launchTextureLatKernelBenchmark(200, stride, &error);
    if (error != 0) {
        printErrorCodeInformation(error);
        exit(error);
    }
    return lat;
}

LatencyTuple launchTextureLatKernelBenchmark(int N, int stride, int* error) {
    LatencyTuple result;
    hipError_t error_id;

    int *h_a = nullptr, *d_a = nullptr;
    unsigned int *h_time = nullptr, *d_time = nullptr;
    bool bindedTexture = false;
    hipTextureObject_t  tex = 0;

    do {
        // Allocate Memory on Host
        h_a = (int *) malloc(sizeof(int) * (N));
        if (h_a == nullptr) {
            printf("[TEXTURE_LAT.H]: malloc h_a Error\n");
            *error = 1;
            break;
        }

        h_time = (unsigned int *) malloc(sizeof(unsigned int));
        if (h_time == nullptr) {
            printf("[TEXTURE_LAT.H]: malloc h_time Error\n");
            *error = 1;
            break;
        }

        // Allocate Memory on GPU
        error_id = hipMalloc((void **) &d_a, sizeof(int) * (N));
        if (error_id != hipSuccess) {
            printf("[TEXTURE_LAT.H]: hipMalloc d_a Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = hipMalloc((void **) &d_time, sizeof(unsigned int));
        if (error_id != hipSuccess) {
            printf("[TEXTURE_LAT.H]: hipMalloc d_time Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        // Initialize p-chase array
        for (int i = 0; i < N; i++) {
            //original:
            h_a[i] = (i + stride) % N;
        }

        // Copy array from Host to GPU
        error_id = hipMemcpy(d_a, h_a, sizeof(int) * N, hipMemcpyHostToDevice);
        if (error_id != hipSuccess) {
            printf("[TEXTURE_LAT.H]: hipMemcpy d_a Error: %s\n", hipGetErrorString(error_id));
            *error = 3;
            break;
        }

        // Create Texture Object
        hipResourceDesc resDesc = {};
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = hipResourceTypeLinear;
        resDesc.res.linear.devPtr = d_a;
        resDesc.res.linear.desc.f = hipChannelFormatKindSigned;
        resDesc.res.linear.desc.x = 32; // bits per channel
        resDesc.res.linear.sizeInBytes = N*sizeof(int);

        hipTextureDesc texDesc = {};
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = hipReadModeElementType;

        error_id = hipCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);
        bindedTexture = true;

        error_id = hipDeviceSynchronize();

        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[TEXTURE_LAT.H]: hipCreateTextureObject Error: %s\n", hipGetErrorString(error_id));
            *error = 4;
            bindedTexture = false;
            break;
        }

        error_id = hipDeviceSynchronize();

        // Launch Kernel function with clock function
        dim3 Db = dim3(1);
        dim3 Dg = dim3(1, 1, 1);
        hipLaunchKernelGGL(texture_lat, Dg, Db, 0, 0, tex, d_a, N, d_time);

        error_id = hipDeviceSynchronize();

        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[TEXTURE_LAT.H]: Kernel launch/execution with clock Error: %s\n", hipGetErrorString(error_id));
            *error = 5;
            break;
        }
        error_id = hipDeviceSynchronize();

        // Copy results from GPU to Host
        error_id = hipMemcpy((void *) h_time, (void *) d_time, sizeof(unsigned int), hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[TEXTURE_LAT.H]: hipMemcpy d_time Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }
        error_id = hipDeviceSynchronize();

        unsigned int lat = h_time[0];
#ifdef IsDebug
        fprintf(out, "Measured Texture avg latencyCycles is %d cycles\n", lat);
#endif //IsDebug
        result.latencyCycles = lat;

        // Launch kernel function with globaltimer
        hipLaunchKernelGGL(texture_lat_globaltimer, Dg, Db, 0, 0, tex, d_a, N, d_time);

        error_id = hipDeviceSynchronize();

        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[TEXTURE_LAT.H]: Kernel launch/execution with globaltimer Error: %s\n", hipGetErrorString(error_id));
            *error = 5;
            break;
        }
        error_id = hipDeviceSynchronize();

        // Copy results from GPU to Host
        error_id = hipMemcpy((void *) h_time, (void *) d_time, sizeof(unsigned int), hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[TEXTURE_LAT.H]: hipMemcpy d_time Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }
        error_id = hipDeviceSynchronize();

        lat = h_time[0];
#ifdef IsDebug
        fprintf(out, "Measured Texture avg latencyCycles is %d nanoseconds\n", lat);
#endif //IsDebug
        result.latencyNano = lat;
    } while(false);

    if (bindedTexture) {
        // Free Texture Object
        error_id = hipDestroyTextureObject(tex);
    }

    // Free Memory on GPU
    if (d_a != nullptr) {
        error_id = hipFree(d_a);
    }

    if (d_time != nullptr) {
        error_id = hipFree(d_time);
    }

    // Free Memory on Host
    if (h_a != nullptr) {
        free(h_a);
    }

    if (h_time != nullptr) {
        free(h_time);
    }

    error_id = hipDeviceReset();

    return result;
}

__global__ void texture_lat_globaltimer (hipTextureObject_t tex, int* my_array, int array_length, unsigned int *time) {
    int iter = 10000;

    unsigned long long start_time, end_time;
    int j = 0;

    // First round
    for (int k = 0; k < array_length; k++) {
        j = tex1Dfetch<int>(tex, j);
    }

    // Second round
    start_time = clock();
    for (int k = 0; k < iter; k++) {
        j = tex1Dfetch<int>(tex, j);
    }
    s_index[0] = j;
    end_time = clock();

    unsigned int diff = (unsigned int) (end_time - start_time);

    time[0] = diff / iter;
}

__global__ void texture_lat (hipTextureObject_t tex, int* my_array, int array_length, unsigned int *time) {
    int iter = 10000;

    unsigned int start_time, end_time;
    int j = 0;

    // First round
	for (int k = 0; k < array_length; k++) {
        j=tex1Dfetch<int>(tex, j);
    }

    // Second round
    start_time = clock();
    for (int k = 0; k < iter; k++) {
        j=tex1Dfetch<int>(tex, j);
    }
    s_index[0] = j;
    end_time = clock();

    unsigned int diff = end_time - start_time;

    time[0] = diff / iter;
}

#endif //CUDATEST_TEXTURE_LAT

