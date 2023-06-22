#include "hip/hip_runtime.h"

#ifndef CUDATEST_L2_LINESIZEALT
#define CUDATEST_L2_LINESIZEALT

# include <cstdio>

# include "hip/hip_runtime.h"
# include "../eval.h"
# include "../GPU_resources.h"
# include "../general_functions.h"

__global__ void l2_lineSize (unsigned int N, unsigned int * my_array, unsigned int *missIndex);

unsigned int launchL2LineSizeAltKernelBenchmark(unsigned int N, int stride, int* error);

unsigned int measure_L2_LineSize_Alt(unsigned int l2SizeBytes) {
    unsigned int l2SizeInts = l2SizeBytes >> 2; // / 4;
    int error = 0;

    unsigned int lineSize = 0;
    // Doubling the size of N such that the whole array is not already cached in L2 after copy from Host to GPU
    lineSize = launchL2LineSizeAltKernelBenchmark(l2SizeInts * 2, 1, &error);
    if (error != 0) {
        printErrorCodeInformation(error);
        exit(error);
    }
    return lineSize;
}

unsigned int launchL2LineSizeAltKernelBenchmark(unsigned int N, int stride, int* error) {
    unsigned int lineSize = 0;
    hipError_t error_id;
    unsigned int *h_a = nullptr, *h_missIndex = nullptr,
    *d_a = nullptr, *d_missIndex = nullptr;

    do {
        // Allocate Memory on Host Memory
        h_a = (unsigned int *) mallocAndCheck("l2_linesize", sizeof(unsigned int) * (N),
                                              "h_a", error);

        h_missIndex = (unsigned int*) calloc(LineMeasureSize, sizeof(unsigned int));
        if (h_missIndex == nullptr) {
            printf("[L2_LINESIZE.CPP]: malloc h_missIndex Error\n");
            *error = 1;
            break;
        }

        // Allocate Memory on GPU Memory
        if (hipMallocAndCheck("l2_linesize", (void **) &d_a, sizeof(unsigned int) * N,
                              "d_a", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("l2_linesize", (void **) &d_missIndex, sizeof(unsigned int) * LineMeasureSize,
                              "d_missIndex", error) != hipSuccess)
            break;

        // Initialize p-chase array
        for (int i = 0; i < N; i++) {
            //original:
            h_a[i] = (i + stride) % N;
        }

        // Copy elements from Host to GPU
        if (hipMemcpyAndCheck("l2_linesize", d_a, h_a, sizeof(unsigned int) * N,
                              "h_a -> d_a", error, false) != hipSuccess)
            break;

        // Copy zeroes to GPU array
        if (hipMemcpyAndCheck("l2_linesize", d_missIndex, h_missIndex, sizeof(unsigned int) * LineMeasureSize,
                              "h_missIndex -> d_missIndex", error, false) != hipSuccess)
            break;

        error_id = hipDeviceSynchronize();

        // Launch Kernel function
        dim3 Db = dim3(1);
        dim3 Dg = dim3(1, 1, 1);
        hipLaunchKernelGGL(l2_lineSize, Dg, Db, 0, 0, N, d_a, d_missIndex);
        error_id = hipDeviceSynchronize();

        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[L2_LINESIZE.CPP]: Kernel launch/execution with clock Error: %s\n", hipGetErrorString(error_id));
            *error = 5;
            break;
        }
        error_id = hipDeviceSynchronize();

        // Copy results from GPU to Host
        error_id = hipMemcpy((void *) h_missIndex, (void *) d_missIndex, sizeof(unsigned int) * LineMeasureSize, hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[L2_LINESIZE.CPP]: hipMemcpy d_missIndex Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }
        error_id = hipDeviceSynchronize();

        lineSize = getMostValueInArray(h_missIndex, LineMeasureSize) * 4;
        error_id = hipDeviceSynchronize();
    } while(false);

    // Free Memory on GPU
    FreeTestMemory({d_a, d_missIndex}, true);

    // Free Memory on Host
    FreeTestMemory({h_a, h_missIndex}, false);

    error_id = hipDeviceReset();

    return lineSize;
}

__global__ void l2_lineSize (unsigned int N, unsigned int* my_array, unsigned int *missIndex) {
    unsigned int start_time, end_time;
    unsigned int j = 0;
    int tol = 50;
    unsigned long long ref = 0;

    // Using cold cache misses for this cache
    for (int k = 0; k < LineMeasureSize; k++) {
        LOCAL_CLOCK(start_time);
        NON_TEMPORAL_LOAD_CG(j, my_array + j);
        s_index[k] = j;
        LOCAL_CLOCK(end_time);
        s_tvalue[k] = end_time - start_time;

        // capturing average time
        ref += s_tvalue[k];
    }

    ref /= LineMeasureSize;

    int lastMissIndex = 0;
    int missPtr = 0;

    for (int i = 1; i < LineMeasureSize; ++i) {
        if (s_tvalue[i] > ref + tol) {
            missIndex[missPtr] = i - lastMissIndex;
            lastMissIndex = i;
            ++missPtr;
        }
    }
}

#endif //CUDATEST_L2_LINESIZEALT

