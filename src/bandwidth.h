//
// Created by max on 09.06.23.
//

#ifndef MT4G_WITH_HIP_BANDWIDTH_THROUGHPUT_H
#define MT4G_WITH_HIP_BANDWIDTH_THROUGHPUT_H

# include "general_functions.h"

#define ERROR_PAIR std::pair<float, float>(-1, -1)
#define N 20 * (1 << 20) // 20M elements (80 MB)


// taken legacy code from https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
// vanilla saxpy for general GPU bandwidth test
__global__ void main_memory_stress_test(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

__global__ void l1_stress_test(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0) y[i] = a * x[i] + y[i-1]; // sequential access
}

__global__ void l2_stress_test(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[(i + clock()) & (n - 1)]; // random access,
    // "& (n - 1)" = "/n", but faster
}

std::pair<float, float> executeBandwidthThroughputChecks(std::string who_calls) {
    std::cout << "Measure " << who_calls << " Memory Bandwidth" << std::endl;
#ifdef IsDebug
    fprintf(out, "Measure Bandwidth & Throughput\n\n");
#endif //IsDebug

    int error = 0; // For checking errors

    float *x, *y, *d_x, *d_y;
    hipEvent_t start, stop;
    float milliseconds = 0;

    float bestBandwidth = 0.0f;

    for (int threadsPerBlock = 32; threadsPerBlock <= 1024; threadsPerBlock *= 2) {
        do {
            x = (float *)mallocAndCheck("bandwidth.h", sizeof(float) * N, "x", &error);
            y = (float *)mallocAndCheck("bandwidth.h", sizeof(float) * N, "y", &error);

            if (hipMallocAndCheck("bandwidth.h", (void **)&d_x, sizeof(float) * N, "d_x", &error) != hipSuccess)
                break;
            if (hipMallocAndCheck("bandwidth.h", (void **)&d_y, sizeof(float) * N, "d_y", &error) != hipSuccess)
                break;

            for (int i = 0; i < N; i++) {
                x[i] = 1.0f;
                y[i] = 2.0f;
            }

            if (hipEventCreate(&start) != hipSuccess)
                break;
            if (hipEventCreate(&stop) != hipSuccess)
                break;

            if (hipMemcpyAndCheck("bandwidth.h", d_x, x, sizeof(float) * N, "x -> d_x", &error, false) != hipSuccess)
                break;
            if (hipMemcpyAndCheck("bandwidth.h", d_y, y, sizeof(float) * N, "y -> d_y", &error, false) != hipSuccess)
                break;

            int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
            if (who_calls == "MAIN") {
                if (hipEventRecord(start) != hipSuccess)
                    break;

                main_memory_stress_test<<<blocksPerGrid, threadsPerBlock>>>(N, 2.0f, d_x, d_y);

                if (hipEventRecord(stop) != hipSuccess)
                    break;
            } else if (who_calls == "L2") {
                if (hipEventRecord(start) != hipSuccess)
                    break;

                l2_stress_test<<<blocksPerGrid, threadsPerBlock>>>(N, 2.0f, d_x, d_y);

                if (hipEventRecord(stop) != hipSuccess)
                    break;
            } else if (who_calls == "L1") {
                if (hipEventRecord(start) != hipSuccess)
                    break;

                l1_stress_test<<<blocksPerGrid, threadsPerBlock>>>(N, 2.0f, d_x, d_y);

                if (hipEventRecord(stop) != hipSuccess)
                    break;
            } else {
                printf("Unknown caller parameter\n");
                break;
            }

            if (hipMemcpyAndCheck("bandwidth.h", y, d_y, sizeof(float) * N, "d_y -> y", &error, true) != hipSuccess)
                break;

            if (hipEventSynchronize(stop) != hipSuccess)
                break;

            if (hipEventElapsedTime(&milliseconds, start, stop) != hipSuccess)
                break;

#ifdef IsDebug
            fprintf(out, "Elapsed time: %f ms\n", milliseconds);
            float maxError = 0.0f;
            for (int i = 0; i < N; i++) {
                maxError = max(maxError, std::abs(y[i] - 4.0f));
            }
            printf("Type: %d\tMax error: %f\n", type, maxError);
#endif //IsDebug
        } while (false);

        // freeing part //
        FreeTestMemory({d_x, d_y}, true);   // on GPU
        FreeTestMemory({x, y}, false);      // on CPU

        if (error != 0)
            return ERROR_PAIR;

        float bandwidth = N * 4 * 3 / milliseconds / 1e6;

        if (bandwidth > bestBandwidth) {
            bestBandwidth = bandwidth;
        }
    }

    std::pair<float, float> result(bestBandwidth, -1); // second value is not used, was designed for throughput
    return result;
}


#endif //MT4G_WITH_HIP_BANDWIDTH_THROUGHPUT_H
