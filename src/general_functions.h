//
// Created by max on 06.03.23.
//

#ifndef MT4G_GENERAL_FUNCTIONS_H
#define MT4G_GENERAL_FUNCTIONS_H

// Here are the template function for hipXX,
// i.e. hipDeviceReset or hipFree are collected.
# include "hip/hip_runtime.h"
# include "eval.h"
# include "GPU_resources.h"
# include "general_functions.h"
# include "l1_latency_size.h"
# include "constCache2.h"

// Memory alignment macro
// src - https://stackoverflow.com/questions.h/cuda-memory-alignment
#if defined(__CUDACC__) // NVCC
#define ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
#define ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
#define ALIGN(n) __declspec(align(n))
#elif defined(__clang__) // Clang
#else // Otherwise
#define ALIGN(n) __attribute__((aligned(n)))
#error "Please provide a definition for ALIGN macro for your host compiler!"
#endif

// Platform-specific clocks
/**
 * CA - cache at all levels
 * CD - cache at L2 only (no L1)
 * ---
 * NVI: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators
 * AMD: loads to both L1 and L2 by default, hence same instructions
 */
#ifdef __HIP_PLATFORM_AMD__
#define NON_TEMPORAL_LOAD_CA(j, ptr) j = *ptr;
#define NON_TEMPORAL_LOAD_CG(j, ptr) j = *ptr;
#define GLOBAL_CLOCK(time) time = __builtin_readcyclecounter();
#else
#define NON_TEMPORAL_LOAD_CA(j, ptr) j = __ldca(ptr);
#define NON_TEMPORAL_LOAD_CG(j, ptr)  j = __ldcg(ptr);
#define GLOBAL_CLOCK(time) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time));
#endif

#define LOCAL_CLOCK(time) time = clock();


struct dataForTest {
    unsigned int *d_a;      // device data block
    unsigned int *d_time;   // device time block
    int type_of_clock;      // 0 - local, 1 - global
    int warmup_iterations;  // number of iterations in Stage #1 - warm-up
    int test_iterations;    // number of iterations in Stage #2 - test
};

struct thresholdAndPrefix {
    int threshold;
    const char *prefix;      // prefix for output files, is read-only
    int type_of_cache;       // 1 - L1, 2 - L2, 3 - RO, 4 - TXT
};

void freeAndCheck(const char *where, void *ptr, const char *name, int *error) {
    if (ptr != nullptr) {
        hipError_t result = hipFree(ptr);
        if (result != hipSuccess) {
            std::cerr << where << "\tError freeing " << name << ": " << hipGetErrorString(result) << std::endl;
            *error = 3;
        }
    }
}

void resetDeviceAndCheck(const char *where, int *error) {
    hipError_t result = hipDeviceReset();
    if (result != hipSuccess) {
        std::cerr << where << "\tError resetting device: " << hipGetErrorString(result) << std::endl;
        *error = 3;
    }
}

void *mallocAndCheck(const char *where, size_t size, const char *name, int *error) {
    void *ptr = malloc(size);
    if (ptr == nullptr) {
        std::cerr << where << "\tmalloc " << name << " Error!" << std::endl;
        *error = 1;
        return nullptr;
    }
    return ptr;
}

/**
 * @brief function for ((aligned)) memory allocation on GPU side
 * more info: https://rocm-developer-tools.github.io/HIP/group__Memory.html#ga805c7320498926e444616fe090c727ee
 * @param where funciton (or file)-caller
 * @param ptr pointer to allocated memory
 * @param size size of allocated memory in bytes
 * @param name name of allocated memory [for error description]
 * @param error 0 if OK, non-zero = error
 * @return 0 if OK, non-zero = error
 */
hipError_t hipMallocAndCheck(const char *where, void **ptr, size_t size, const char *name, int *error) {
    hipError_t result = hipMallocPitch(ptr, &size, size, 1);//hipMalloc(ptr, size);
    if (result != hipSuccess || *ptr == nullptr) {
        printf("%s\thipMalloc %s Error: %s\n", where, name, hipGetErrorString(result));
        *error = 2;
    }
    return result;
}

// Freeing part of 2D array
#define SET_PART_OF_2D(time, h_time) \
    if (h_time != nullptr && time != nullptr)        \
        if (time != nullptr) { \
                time[0] = h_time; \
            } else { \
                free(h_time); \
            } \


/**
 * @brief copying data from device to host or vice versa
 * @param isDeviceToHost false -> host to device, true -> device to host
 * @param where function (or file)-caller
 * @param dst where to copy
 * @param src where to copy from
 * @param size size of data in bytes
 * @param name where is called (names of src/dest) for error message
 * @param error type of error
 * @return 0 if OK, non-zero - if error
 */
hipError_t hipMemcpyAndCheck(const char *where, void *dst, const void *src,
                             size_t size, const char *name, int *error,
                             bool isDeviceToHost) {
    hipError_t result;
    if (isDeviceToHost) {
        result = hipMemcpy(dst, src, size, hipMemcpyDeviceToHost);
    } else {
        result = hipMemcpy(dst, src, size, hipMemcpyHostToDevice);
    }

    if (result != hipSuccess) {
        printf("%s\thipMemcpy %s Error: %s\n", where, name, hipGetErrorString(result));
        *error = 6;
    }
    return result;
}

__global__ void latency_test(dataForTest d1) {

    unsigned long long start_time, end_time;
    unsigned int j = 0;

    // Warming up
    for (int k = 0; k < d1.warmup_iterations; k++) {
        j = d1.d_a[j];
    }

    // No first round required
    if (d1.type_of_clock == 1) {
        GLOBAL_CLOCK(start_time);
        for (int k = 0; k < d1.test_iterations; k++) {
            NON_TEMPORAL_LOAD_CG(j, d1.d_a + j);
        }
        s_index[0] = j;
        GLOBAL_CLOCK(end_time);
    } else {
        LOCAL_CLOCK(start_time);
        for (int k = 0; k < d1.test_iterations; k++) {
            NON_TEMPORAL_LOAD_CG(j, d1.d_a + j);
        }
        s_index[0] = j;
        LOCAL_CLOCK(end_time);
    }

    unsigned int diff = (unsigned int) (end_time - start_time);

    d1.d_time[0] = diff / d1.test_iterations;
}

LatencyTuple launchCacheLatencyBenchmark(int warmup_iterations, int test_iterations,
                                         int stride, int *error) {
    LatencyTuple result;
    hipError_t error_id;
    unsigned int *h_a = nullptr, *h_time = nullptr, *d_a = nullptr, *d_time = nullptr;

    do {
        // Allocate Memory on Host
        h_a = (unsigned int *) mallocAndCheck("general_functions/cacheLatency",
                                              sizeof(unsigned int) * warmup_iterations,
                                              "h_a", error);

        h_time = (unsigned int *) mallocAndCheck("general_functions/cacheLatency", sizeof(unsigned int),
                                                 "h_time", error);

        // Allocate Memory on GPU
        if (hipMallocAndCheck("general_functions/cacheLatency", (void **) &d_a,
                              sizeof(unsigned int) * warmup_iterations,
                              "d_a", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("general_functions/cacheLatency", (void **) &d_time, sizeof(unsigned int),
                              "d_time", error) != hipSuccess)
            break;

        // Initialize p-chase array
        for (int i = 0; i < warmup_iterations; i++) {
            //original:
            h_a[i] = (i + stride) % warmup_iterations;
        }

        // Copy array from Host to GPU
        if (hipMemcpyAndCheck("general_functions/cacheLatency", d_a, h_a, sizeof(unsigned int) * warmup_iterations,
                              "h_a -> d_a", error, false) != hipSuccess)
            break;
        error_id = hipDeviceSynchronize();

        // Launch Kernel function with clock function
        dim3 Db = dim3(1);
        dim3 Dg = dim3(1, 1, 1);
        dataForTest d1{};
        d1.d_a = d_a;
        d1.d_time = d_time;
        d1.type_of_clock = 0;
        d1.warmup_iterations = warmup_iterations;
        d1.test_iterations = test_iterations;

        hipLaunchKernelGGL(latency_test, Dg, Db, 0, 0, d1);

        error_id = hipDeviceSynchronize();

        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[general_functions/cacheLatency]: Kernel launch/execution with clock Error:%s\n",
                   hipGetErrorString(error_id));
            *error = 5;
            break;
        }
        error_id = hipDeviceSynchronize();

        // Copy results from GPU to Host
        if (hipMemcpyAndCheck("general_functions/cacheLatency", h_time, d_time, sizeof(unsigned int),
                              "d_time -> h_time", error, true) != hipSuccess)
            break;

        error_id = hipDeviceSynchronize();

        unsigned int lat = h_time[0];
#ifdef IsDebug
    fprintf(out, "Measured Main avg latencyCycles is %d cycles\n", lat);
#endif //IsDebug
        result.latencyCycles = lat;
        error_id = hipDeviceSynchronize();

        // Launch Kernel function with global timer
        d1.type_of_clock = 1;
        hipLaunchKernelGGL(latency_test, Dg, Db, 0, 0, d1);

        error_id = hipDeviceSynchronize();
        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[general_functions/cacheLatency]: Kernel launch/execution with clock Error:%s\n",
                   hipGetErrorString(error_id));
            *error = 5;
            break;
        }

        error_id = hipDeviceSynchronize();

        // Copy results from Host to GPU
        if (hipMemcpyAndCheck("general_functions/cacheLatency", d_time, h_time, sizeof(unsigned int),
                              "h_time -> d_time", error, false) != hipSuccess)
            break;
        error_id = hipDeviceSynchronize();
        lat = h_time[0];
#ifdef IsDebug
        fprintf(out, "Measured Main avg latencyCycles is %d nanoseconds\n", lat);
#endif //IsDebug
        result.latencyNano = lat;
    } while (false);

    // Free Memory on GPU
    FreeTestMemory({d_a, d_time}, true);

    // Free Memory on Host
    FreeTestMemory({h_a, h_time}, false);

    error_id = hipDeviceReset();
    return result;
}

__global__ void constantCacheLineSize(int cache_type, unsigned int upperLimit, unsigned int *lineSize) {
    unsigned int start_time, end_time;
    unsigned int j = 0;
    unsigned int modulo = (cache_type == 1 ? 999999 : 1200); // C1 : C1.5

    // Using cold cache misses for this cache
    for (int k = 0; k < upperLimit; k++) {
        start_time = clock();
        j = arr[j];
        s_index[k] = j;
        end_time = clock();
        j = j % modulo;
        s_tvalue[k] = end_time - start_time;
    }

    // cache checks
    unsigned long long ref = (s_tvalue[14] + s_tvalue[15] + s_tvalue[16]) / 3;
    int firstIndexMiss = 16;
    int secondIndexMiss = firstIndexMiss;
    if (cache_type == 1) { // C1
        while (s_tvalue[firstIndexMiss] <= ref + 25) { // tolerance
            firstIndexMiss++;
        }

        secondIndexMiss = firstIndexMiss + 1;
        while (s_tvalue[secondIndexMiss] < ref + 25) {
            secondIndexMiss++;
        }
    } else { // C1.5
        while (s_tvalue[firstIndexMiss] < ref + 100 && firstIndexMiss < 256) { // tolerance
            firstIndexMiss++;
        }

        secondIndexMiss = firstIndexMiss + 1;
        while (s_tvalue[secondIndexMiss] < ref + 100 && secondIndexMiss < 256) {
            secondIndexMiss++;
        }
    }

    lineSize[0] = ((unsigned int) secondIndexMiss - (unsigned int) firstIndexMiss) * 4;
}


unsigned int constantCacheLineSizeBenchmark(int cache, int upperLimit, int *error) {
    unsigned int lineSize = 0;
    hipError_t error_id;
    unsigned int *h_lineSize = nullptr, *d_lineSize = nullptr;

    do {
        // Allocation on Host Memory
        h_lineSize = (unsigned int *) malloc(sizeof(unsigned int));
        if (h_lineSize == nullptr) {
            printf("[C15_LINESIZE.CPP]: malloc h_lineSize Error\n");
            *error = 1;
            break;
        }
        // Allocation on GPU Memory
        error_id = hipMalloc((void **) &d_lineSize, sizeof(unsigned int));
        if (error_id != hipSuccess) {
            printf("[C15_LINESIZE.CPP]: hipMalloc d_lineSize Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }
        error_id = hipDeviceSynchronize();

        // Launch Kernel function
        dim3 Db = dim3(1);
        dim3 Dg = dim3(1, 1, 1);
        hipLaunchKernelGGL(constantCacheLineSize, Dg, Db, 0, 0, cache, upperLimit, d_lineSize);
        error_id = hipDeviceSynchronize();
        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[C15_LINESIZE.CPP]: Kernel launch/execution with clock Error: %s\n", hipGetErrorString(error_id));
            *error = 5;
            break;
        }
        /* copy results from GPU to CPU */
        error_id = hipDeviceSynchronize();
        error_id = hipMemcpy((void *) h_lineSize, (void *) d_lineSize, sizeof(unsigned int), hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[C15_LINESIZE.CPP]: hipMemcpy d_lineSize Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }
        error_id = hipDeviceSynchronize();

        lineSize = h_lineSize[0];
        error_id = hipDeviceSynchronize();
    } while (false);

    if (d_lineSize != nullptr) {
        error_id = hipFree(d_lineSize);
    }

    if (h_lineSize != nullptr) {
        free(h_lineSize);
    }

    error_id = hipDeviceReset();

    return lineSize;
}

unsigned int measureConstantCacheLineSize(int cache,
                                          int upperLimit) {
    int error = 0;
    int lineSize = constantCacheLineSizeBenchmark(cache,
                                                  upperLimit,
                                                  &error);

    if (error != 0) {
        printErrorCodeInformation(error);
        exit(error);
    }
    return lineSize;
}

__global__ void main_size_test(unsigned int *index, bool *isDisturbed, dataForTest d1) {
    unsigned int start_time, end_time;

    bool dist = false;
    unsigned int j = 0;

    for (int k = 0; k < measureSize; k++) {
        s_index[k] = 0;
        s_tvalue[k] = 0;
    }

    // Warming up, filling TLP and PT
    for (int k = 0; k < 32; k++) {
        j = d1.d_a[j];
    }

    // No real first round required
    for (int k = 0; k < measureSize; k++) {
        LOCAL_CLOCK(start_time);
        NON_TEMPORAL_LOAD_CG(j, d1.d_a + j);
        s_index[k] = j;
        LOCAL_CLOCK(end_time);
        s_tvalue[k] = end_time - start_time;
    }

    for (int k = 0; k < measureSize; k++) {
        if (s_tvalue[k] > 1200) {
            dist = true;
        }
        index[k] = s_index[k];
        d1.d_time[k] = s_tvalue[k];
    }
    *isDisturbed = dist;
}

bool launchMainKernelBenchmark(int N, int stride, double *avgOut, unsigned int *potMissesOut, unsigned int **time,
                               int *error) {
    hipError_t error_id;

    unsigned int *h_a = nullptr, *h_index = nullptr, *h_timeinfo = nullptr,
            *d_a = nullptr, *d_index = nullptr, *duration = nullptr;
    bool *disturb = nullptr, *d_disturb = nullptr;

    do {
        // Allocate Memory on Host
        h_a = (unsigned int *) mallocAndCheck("mainmemtest.h", sizeof(unsigned int) * (N),
                                              "h_a", error);

        h_index = (unsigned int *) mallocAndCheck("mainmemtest.h", sizeof(unsigned int) * measureSize,
                                                  "h_index", error);

        h_timeinfo = (unsigned int *) mallocAndCheck("mainmemtest.h", sizeof(unsigned int) * measureSize,
                                                     "h_timeinfo", error);

        disturb = (bool *) mallocAndCheck("mainmemtest.h", sizeof(bool),
                                          "disturb", error);

        // Allocate Memory on GPU
        if (hipMallocAndCheck("mainmemtest.h", (void **) &d_a, sizeof(unsigned int) * N,
                              "d_a", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("mainmemtest.h", (void **) &duration, sizeof(unsigned int) * measureSize,
                              "duration", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("mainmemtest.h", (void **) &d_index, sizeof(unsigned int) * measureSize,
                              "d_index", error) != hipSuccess)
            break;

        if (hipMallocAndCheck("mainmemtest.h", (void **) &d_disturb, sizeof(bool),
                              "d_disturb", error) != hipSuccess)
            break;

        // Initialize p-chase array
        for (int i = 0; i < N; i++) {
            //original:
            h_a[i] = (i + stride) % N;
        }

        // Copy array from Host to GPU
        if (hipMemcpyAndCheck("mainmemtest.h", d_a, h_a, sizeof(unsigned int) * N,
                              "h_a -> d_a", error, false) != hipSuccess)
            break;
        error_id = hipDeviceSynchronize();

        // Launch Kernel function
        dim3 Db = dim3(1);
        dim3 Dg = dim3(1, 1, 1);

        dataForTest d1{};
        d1.warmup_iterations = 32;
        d1.test_iterations = measureSize;
        d1.d_a = d_a;
        d1.d_time = duration;

        hipLaunchKernelGGL(main_size_test, Dg, Db, 0, 0, d_index, d_disturb, d1);

        error_id = hipDeviceSynchronize();

        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[MAINMEMTEST.H]: Kernel launch/execution with clock Error:%s\n", hipGetErrorString(error_id));
            *error = 5;
            break;
        }
        error_id = hipDeviceSynchronize();

        // Copy results from GPU to Host
        if (hipMemcpyAndCheck("mainmemtest.h", h_timeinfo, duration, sizeof(unsigned int) * measureSize,
                              "duration -> h_timeinfo", error, false) != hipSuccess)
            break;

        if (hipMemcpyAndCheck("mainmemtest.h", h_index, d_index, sizeof(unsigned int) * measureSize,
                              "d_index -> h_index", error, false) != hipSuccess)
            break;

        if (hipMemcpyAndCheck("mainmemtest.h", disturb, d_disturb, sizeof(bool),
                              "d_disturb -> disturb", error, false) != hipSuccess)
            break;

        error_id = hipDeviceSynchronize();

        createOutputFile(N, measureSize, h_index, h_timeinfo, avgOut, potMissesOut, "Main_");
    } while (false);

    // Free Memory on GPU
    FreeTestMemory({d_a, d_index, duration, d_disturb}, true);

    // Free Memory on Host
    bool ret = false;
    if (disturb != nullptr) {
        ret = *disturb;
        free(disturb);
    }

    // Free Memory on Host
    FreeTestMemory({h_a, h_index}, false);

    SET_PART_OF_2D(time, h_timeinfo);

    error_id = hipDeviceReset();
    return ret;
}

LatencyTuple measureGeneralCacheLatency(int arrSize, int test_iterations, int stride) {
    int error = 0;
    LatencyTuple lat = launchCacheLatencyBenchmark(arrSize, test_iterations,
                                                   stride, &error);
    if (error != 0) {
        printErrorCodeInformation(error);
        exit(error);
    }
    return lat;
}


CacheResults measureCacheResults(int arrSize, int test_iterations, int stride, thresholdAndPrefix t1) {
    double *avg = (double *) malloc(sizeof(double));
    unsigned int *misses = (unsigned int *) malloc(sizeof(unsigned int));
    unsigned int **time = (unsigned int **) malloc(sizeof(unsigned int *));
    if (avg == nullptr || misses == nullptr || time == nullptr) {
        FreeTestMemory({avg, misses, time}, false);
        printErrorCodeInformation(1);
        exit(1);
    }

    int error = 0;
    bool dist = true;
    int count = 5;

    while (dist && count > 0) {
        dist = launchMainKernelBenchmark(arrSize, stride, avg, misses, time, &error);
        --count;
    }

    FreeTestMemory({time[0], avg, misses, time}, false);
    if (error != 0) {
        printErrorCodeInformation(error);
        exit(error);
    }

    return CacheResults{};
}


#endif //MT4G_GENERAL_FUNCTIONS_H
