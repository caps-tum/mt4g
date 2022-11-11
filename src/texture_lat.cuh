
#ifndef CUDATEST_TEXTURE_LAT
#define CUDATEST_TEXTURE_LAT

# include <cstdio>

# include "cuda.h"
# include "eval.h"
# include "GPU_resources.cuh"

//texture<int, 1, cudaReadModeElementType> tex_ref;

__global__ void texture_lat (cudaTextureObject_t tex, int * my_array, int array_length, unsigned int * time);
__global__ void texture_lat_globaltimer (cudaTextureObject_t tex, int * my_array, int array_length, unsigned int * time);

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
    cudaError_t error_id;

    int *h_a = nullptr, *d_a = nullptr;
    unsigned int *h_time = nullptr, *d_time = nullptr;
    bool bindedTexture = false;
    cudaTextureObject_t  tex = 0;

    do {
        // Allocate Memory on Host
        h_a = (int *) malloc(sizeof(int) * (N));
        if (h_a == nullptr) {
            printf("[TEXTURE_LAT.CUH]: malloc h_a Error\n");
            *error = 1;
            break;
        }

        h_time = (unsigned int *) malloc(sizeof(unsigned int));
        if (h_time == nullptr) {
            printf("[TEXTURE_LAT.CUH]: malloc h_time Error\n");
            *error = 1;
            break;
        }

        // Allocate Memory on GPU
        error_id = cudaMalloc((void **) &d_a, sizeof(int) * (N));
        if (error_id != cudaSuccess) {
            printf("[TEXTURE_LAT.CUH]: cudaMalloc d_a Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        error_id = cudaMalloc((void **) &d_time, sizeof(unsigned int));
        if (error_id != cudaSuccess) {
            printf("[TEXTURE_LAT.CUH]: cudaMalloc d_time Error: %s\n", cudaGetErrorString(error_id));
            *error = 2;
            break;
        }

        // Initialize p-chase array
        for (int i = 0; i < N; i++) {
            //original:
            h_a[i] = (i + stride) % N;
        }

        // Copy array from Host to GPU
        error_id = cudaMemcpy(d_a, h_a, sizeof(int) * N, cudaMemcpyHostToDevice);
        if (error_id != cudaSuccess) {
            printf("[TEXTURE_LAT.CUH]: cudaMemcpy d_a Error: %s\n", cudaGetErrorString(error_id));
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
            printf("[TEXTURE_LAT.CUH]: cudaCreateTextureObject Error: %s\n", cudaGetErrorString(error_id));
            *error = 4;
            bindedTexture = false;
            break;
        }

        cudaDeviceSynchronize();

        // Launch Kernel function with clock function
        dim3 Db = dim3(1);
        dim3 Dg = dim3(1, 1, 1);
        texture_lat<<<Dg, Db>>>(tex, d_a, N, d_time);

        cudaDeviceSynchronize();

        error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
            printf("[TEXTURE_LAT.CUH]: Kernel launch/execution with clock Error: %s\n", cudaGetErrorString(error_id));
            *error = 5;
            break;
        }
        cudaDeviceSynchronize();

        // Copy results from GPU to Host
        error_id = cudaMemcpy((void *) h_time, (void *) d_time, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[TEXTURE_LAT.CUH]: cudaMemcpy d_time Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }
        cudaDeviceSynchronize();

        unsigned int lat = h_time[0];
#ifdef IsDebug
        fprintf(out, "Measured Texture avg latencyCycles is %d cycles\n", lat);
#endif //IsDebug
        result.latencyCycles = lat;

        // Launch kernel function with globaltimer
        texture_lat_globaltimer<<<Dg, Db>>>(tex, d_a, N, d_time);

        cudaDeviceSynchronize();

        error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
            printf("[TEXTURE_LAT.CUH]: Kernel launch/execution with globaltimer Error: %s\n", cudaGetErrorString(error_id));
            *error = 5;
            break;
        }
        cudaDeviceSynchronize();

        // Copy results from GPU to Host
        error_id = cudaMemcpy((void *) h_time, (void *) d_time, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if (error_id != cudaSuccess) {
            printf("[TEXTURE_LAT.CUH]: cudaMemcpy d_time Error: %s\n", cudaGetErrorString(error_id));
            *error = 6;
            break;
        }
        cudaDeviceSynchronize();

        lat = h_time[0];
#ifdef IsDebug
        fprintf(out, "Measured Texture avg latencyCycles is %d nanoseconds\n", lat);
#endif //IsDebug
        result.latencyNano = lat;
    } while(false);

    if (bindedTexture) {
        // Free Texture Object
        cudaDestroyTextureObject(tex);
    }

    // Free Memory on GPU
    if (d_a != nullptr) {
        cudaFree(d_a);
    }

    if (d_time != nullptr) {
        cudaFree(d_time);
    }

    // Free Memory on Host
    if (h_a != nullptr) {
        free(h_a);
    }

    if (h_time != nullptr) {
        free(h_time);
    }

    cudaDeviceReset();

    return result;
}

__global__ void texture_lat_globaltimer (cudaTextureObject_t tex, int* my_array, int array_length, unsigned int *time) {
    int iter = 10000;

    unsigned long long start_time, end_time;
    int j = 0;

    // First round
    for (int k = 0; k < array_length; k++) {
        j = tex1Dfetch<int>(tex, j);
        /*
        int4 values;
        asm volatile(
            "tex.1d.v4.u32.s32 {%0, %1, %2, %3}, [tex_ref, {%4}];" : "=r"(values.x), "=r"(values.y), "=r"(values.z), "=r"(values.w) : "r"(j)
        );
        j = values.x;
         */
    }

    // Second round
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start_time));
    for (int k = 0; k < iter; k++) {
        j = tex1Dfetch<int>(tex, j);
        /*
        int4 values;
        asm volatile(
                "tex.1d.v4.u32.s32 {%0, %1, %2, %3}, [tex_ref, {%4}];" : "=r"(values.x), "=r"(values.y), "=r"(values.z), "=r"(values.w) : "r"(j)
                );
        j = values.x;
         */
    }
    s_index[0] = j;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(end_time));

    unsigned int diff = (unsigned int) (end_time - start_time);

    time[0] = diff / iter;
}

__global__ void texture_lat (cudaTextureObject_t tex, int* my_array, int array_length, unsigned int *time) {
    int iter = 10000;

    unsigned int start_time, end_time;
    int j = 0;

    // First round
	for (int k = 0; k < array_length; k++) {
        j=tex1Dfetch<int>(tex, j);
        /*
        int4 values;
        asm volatile(
                "tex.1d.v4.u32.s32 {%0, %1, %2, %3}, [tex_ref, {%4}];" : "=r"(values.x), "=r"(values.y), "=r"(values.z), "=r"(values.w) : "r"(j)
                );
        j = values.x;
         */
    }

    // Second round
    start_time = clock();
    for (int k = 0; k < iter; k++) {
        j=tex1Dfetch<int>(tex, j);
        /*
        int4 values;
        asm volatile(
                "tex.1d.v4.u32.s32 {%0, %1, %2, %3}, [tex_ref, {%4}];" : "=r"(values.x), "=r"(values.y), "=r"(values.z), "=r"(values.w) : "r"(j)
                );
        j = values.x;
         */
    }
    s_index[0] = j;
    end_time = clock();

    unsigned int diff = end_time - start_time;

    time[0] = diff / iter;
}

#endif //CUDATEST_TEXTURE_LAT

