#include "hip/hip_runtime.h"

#ifndef CUDATEST_TEXT
#define CUDATEST_TEXT

#include <cstdio>

# include "eval.h"
# include "GPU_resources.h"
# include "binarySearch.h"

__global__ void texture_size(hipTextureObject_t tex, unsigned int size, unsigned int *duration, unsigned int *index,
                             bool *isDisturbed) {

    bool dist = false;

    unsigned int start, end;
    int j = 0;

    for (int k = 0; k < measureSize; k++) {
        s_index[k] = 0;
        s_tvalue[k] = 0;
    }

    // First round
    for (int k = 0; k < size; k++) {
        j = tex1Dfetch<int>(tex, j);
    }

    // Second round
    for (int k = 0; k < measureSize; k++) {
        start = clock();
        j = tex1Dfetch<int>(tex, j);
        s_index[k] = j;
        end = clock();
        s_tvalue[k] = (end - start);
    }

    for (int k = 0; k < measureSize; k++) {
        if (s_tvalue[k] > 2000) {
            dist = true;
        }
        duration[k] = s_tvalue[k];
        index[k] = s_index[k];
    }
    *isDisturbed = dist;
}

bool
launchTextureBenchmark(int N, int stride, double *avgOut, unsigned int *potMissesOut,
                       unsigned int **time, int *error) {
    hipError_t error_id;

    int *h_a = nullptr, *d_a = nullptr;
    unsigned int *h_timeinfo = nullptr, *h_index = nullptr, *d_index = nullptr, *duration = nullptr;
    bool *disturb = nullptr, *d_disturb = nullptr;

    unsigned int size = N * sizeof(int);
    hipTextureObject_t tex = 0;
    bool bindedTexture = false;

    //do-while - to break from simple execution to free memory
    do {
        // Allocate Memory On Host
        h_a = (int *) mallocAndCheck("texture.h", sizeof(int) * (N),
                                              "h_a", error);

        h_index = (unsigned int *) mallocAndCheck("texture.h", sizeof(unsigned int) * measureSize,
                                                  "h_index", error);

        h_timeinfo = (unsigned int *) mallocAndCheck("texture.h", sizeof(unsigned int) * measureSize,
                                                     "h_timeinfo", error);

        disturb = (bool *) mallocAndCheck("texture.h", sizeof(bool),
                                          "disturb", error);

        // Allocate Memory on GPU
        if (hipMallocAndCheck("texture.h", (void **) &d_a, sizeof(unsigned int) * N,
                              "d_a", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("texture.h", (void **) &duration, sizeof(unsigned int) * measureSize,
                              "duration", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("texture.h", (void **) &d_index, sizeof(unsigned int) * measureSize,
                              "d_index", error) != hipSuccess)
            break;


        if (hipMallocAndCheck("texture.h", (void **) &d_disturb, sizeof(bool),
                              "d_disturb", error) != hipSuccess)
            break;

        // Initialize p-chase array
        for (int i = 0; i < N; i++) {
            h_a[i] = (i + stride) % N;
        }

        // Copy array to GPU
        if (hipMemcpyAndCheck("texture.h", d_a, h_a, sizeof(unsigned int) * N,
                              "h_a -> d_a", error, false) != hipSuccess)
            break;

        // Create Texture Object
        hipResourceDesc resDesc = {};
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = hipResourceTypeLinear;
        resDesc.res.linear.devPtr = d_a;
        resDesc.res.linear.desc.f = hipChannelFormatKindSigned;
        resDesc.res.linear.desc.x = 32; // bits per channel
        resDesc.res.linear.sizeInBytes = N * sizeof(int);

        hipTextureDesc texDesc = {};
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = hipReadModeElementType;

        error_id = hipCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);
        bindedTexture = true;

        error_id = hipDeviceSynchronize();

        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[TEXTURE.H]: hipCreateTextureObject Error: %s\n", hipGetErrorString(error_id));
            *error = 4;
            bindedTexture = false;
            break;
        }

        // Launch Kernel function
        dim3 Db = dim3(1);
        dim3 Dg = dim3(1, 1, 1);
        hipLaunchKernelGGL(texture_size, Dg, Db, 0, 0, tex, size, duration, d_index, d_disturb);

        error_id = hipDeviceSynchronize();

        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[TEXTURE.H]: Kernel launch/execution Error: %s\n", hipGetErrorString(error_id));
            *error = 5;
            break;
        }

        error_id = hipDeviceSynchronize();

        // Copy results from GPU to Host
        if (hipMemcpyAndCheck("texture.h", h_timeinfo, duration, sizeof(unsigned int) * measureSize,
                              "duration -> h_timeinfo", error, true) != hipSuccess)
            break;

        if (hipMemcpyAndCheck("texture.h", h_index, d_index, sizeof(unsigned int) * measureSize,
                              "d_index -> h_index", error, true) != hipSuccess)
            break;

        if (hipMemcpyAndCheck("texture.h", disturb, d_disturb, sizeof(bool),
                              "d_disturb -> disturb", error, true) != hipSuccess)
            break;

        if (!*disturb)
            createOutputFile(N, measureSize, h_index, h_timeinfo, avgOut, potMissesOut, "Texture_");

    } while (false);

    // Free texture
    if (bindedTexture) {
        error_id = hipDestroyTextureObject(tex);
    }

    error_id = hipDeviceSynchronize();

    // Free Memory on GPU
    FreeTestMemory({d_a, d_index, duration, d_disturb}, true);

    // Free Memory on Host
    bool ret = false;
    if (disturb != nullptr) {
        ret = *disturb;
        free(disturb);
    }

    FreeTestMemory({h_a, h_index}, false);

    if (h_timeinfo != nullptr) {
        if (time != nullptr) {
            time[0] = h_timeinfo;
        } else {
            free(h_timeinfo);
        }
    }

    error_id = hipDeviceReset();
    return ret;
}


#endif //CUDATEST_TEXT
