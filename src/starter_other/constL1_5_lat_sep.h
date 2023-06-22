#include "hip/hip_runtime.h"

#ifndef CUDATEST_CONSTL1_5_LAT
#define CUDATEST_CONSTL1_5_LAT
//# define isDebug
# include <cstdio>

# include "hip/hip_runtime.h"
# include "../eval.h"
# include "../GPU_resources.h"

#ifdef __HIP_PLATFORM_AMD__
#define GLOBAL_CLOCK(time) time = __builtin_readcyclecounter();
#else
#define GLOBAL_CLOCK(time) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time));
#endif

__global__ void constL1_5_lat(unsigned int * time);
__global__ void constL1_5_lat_globaltimer(unsigned int * time);

LatencyTuple launchConstL1_5LatKernelBenchmark(int* error);

LatencyTuple measure_ConstL1_5_Lat() {
    int error = 0;
    LatencyTuple lat = launchConstL1_5LatKernelBenchmark(&error);
    if (error != 0) {
        printErrorCodeInformation(error);
        exit(error);
    }
    return lat;
}

LatencyTuple launchConstL1_5LatKernelBenchmark(int* error) {
    LatencyTuple result;
    hipError_t error_id;
#ifdef IsDebug
    FILE* c15Out = fopen("c15Out.log", "w");
#endif //IsDebug

    unsigned int* h_time = nullptr, *d_time = nullptr;

    do {

        // Allocate memory on host
        h_time = (unsigned int *) malloc(sizeof(unsigned int));
        if (h_time == nullptr) {
            printf("[CONSTL1_5_LAT_SEP.CPP]: malloc h_time Error\n");
            *error = 1;
            break;
        }

        // Allocate memory on GPU
        error_id = hipMalloc((void **) &d_time, sizeof(unsigned int));
        if (error_id != hipSuccess) {
            printf("[CONSTL1_5_LAT_SEP.CPP]: hipMalloc d_time Error: %s\n", hipGetErrorString(error_id));
            *error = 2;
            break;
        }

        int x = hipDeviceSynchronize();
        if (x != hipSuccess){
        	*error = 1;
        	printf("[CONSTL1_5_LAT_SEP.CPP]: hipDeviceSynchronize return error: %s\n", hipGetErrorString(error_id));
        	break;
        }

        // Launch kernel function using clock function
        dim3 Db = dim3(1);
        dim3 Dg = dim3(1, 1, 1);
        hipLaunchKernelGGL(constL1_5_lat, Dg, Db, 0, 0, d_time);

        x = hipDeviceSynchronize();
        if (x != hipSuccess){
        	*error = 1;
        	printf("[CONSTL1_5_LAT_SEP.CPP]: hipDeviceSynchronize return error: %s\n", hipGetErrorString(error_id));
        	break;
        }
        
        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[CONSTL1_5_LAT_SEP.CPP]: Kernel launch/execution with clock Error: %s\n", hipGetErrorString(error_id));
            *error = 5;
            break;
        }
        x = hipDeviceSynchronize();
        if (x != hipSuccess){
        	*error = 1;
        	printf("[CONSTL1_5_LAT_SEP.CPP]: hipDeviceSynchronize return error: %s\n", hipGetErrorString(error_id));
        	break;
        }

        // Copy results from GPU to Host
        error_id = hipMemcpy((void *) h_time, (void *) d_time, sizeof(unsigned int), hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[CONSTL1_5_LAT_SEP.CPP]: hipMemcpy d_time Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }
        x = hipDeviceSynchronize();
        if (x != hipSuccess){
        	*error = 1;
        	printf("[CONSTL1_5_LAT_SEP.CPP]: hipDeviceSynchronize return error: %s\n", hipGetErrorString(error_id));
        	break;
        }


        unsigned int lat = h_time[0];
#ifdef IsDebug
        fprintf(c15Out, "Measured Const L1.5 avg latencyCycles is %d cycles\n", lat);
#endif //IsDebug
        result.latencyCycles = lat;

        // Launch kernel function using globaltimer
        hipLaunchKernelGGL(constL1_5_lat_globaltimer, Dg, Db, 0, 0, d_time);
        x = hipDeviceSynchronize();
        if (x != hipSuccess){
        	*error = 1;
        	printf("[CONSTL1_5_LAT_SEP.CPP]: hipDeviceSynchronize return error: %s\n", hipGetErrorString(error_id));
        	break;
        }

        error_id = hipGetLastError();
        if (error_id != hipSuccess) {
            printf("[CONSTL1_5_LAT_SEP.CPP]: Kernel launch/execution with globaltimer Error: %s\n", hipGetErrorString(error_id));
            *error = 5;
            break;
        }
        x = hipDeviceSynchronize();
        if (x != hipSuccess){
        	*error = 1;
        	printf("[CONSTL1_5_LAT_SEP.CPP]: hipDeviceSynchronize return error: %s\n", hipGetErrorString(error_id));
        	break;
        }

        // Copy results from GPU to Host
        error_id = hipMemcpy((void *) h_time, (void *) d_time, sizeof(unsigned int), hipMemcpyDeviceToHost);
        if (error_id != hipSuccess) {
            printf("[CONSTL1_5_LAT_SEP.CPP]: hipMemcpy d_time Error: %s\n", hipGetErrorString(error_id));
            *error = 6;
            break;
        }
        x = hipDeviceSynchronize();
        if (x != hipSuccess){
        	*error = 1;
        	printf("[CONSTL1_5_LAT_SEP.CPP]: hipDeviceSynchronize return error: %s\n", hipGetErrorString(error_id));
        	break;
        }

        lat = h_time[0];
#ifdef IsDebug
        fprintf(c15Out, "Measured Const L1.5 avg latencyCycles is %d nanoseconds\n", lat);
#endif //IsDebug
        result.latencyNano = lat;
        x = hipDeviceSynchronize();
        if (x != hipSuccess){
        	*error = 1;
        	printf("[CONSTL1_5_LAT_SEP.CPP]: hipDeviceSynchronize return error: %s\n", hipGetErrorString(error_id));
        	break;
        }
    } while(false);

    // Free Memory on GPU
    if (d_time != nullptr) {
        int x = hipFree(d_time);
        if (x != hipSuccess){
        	*error = 1;
        	printf("[CONSTL1_5_LAT_SEP.CPP]: hipFree return error: %s\n", hipGetErrorString(error_id));
        }
    }
    // Free Memory on Host
    if (h_time != nullptr) {
        free(h_time);
    }
    int x = hipDeviceReset();
    if (x != hipSuccess){
        printf("L1_5/170\tCant reset device!\n");
    }
#ifdef IsDebug
    fclose(c15Out);
#endif //IsDebug
    return result;
}

__global__ void constL1_5_lat_globaltimer (unsigned int *time) {
    unsigned long long start_time, end_time;
    unsigned int j = 0;

     // First round
    for (int k = 0; k < constArrSize; k++) {
        s_index[0] += arr[k];
    }

    // Second round
    GLOBAL_CLOCK(start_time);
    for (int k = 0; k < measureSize; k++) {
        s_index[0] += arr[k];
    }
    s_index[1] = j;
    GLOBAL_CLOCK(end_time);


    unsigned int diff = (unsigned int) (end_time - start_time);

    time[0] = diff / measureSize;
}

__global__ void constL1_5_lat (unsigned int *time) {
    unsigned int start_time, end_time;
    unsigned int j = 0;

    // First round
    for (int k = 0; k < constArrSize; k++) {
        s_index[0] += arr[k];
    }

    // Second round
    start_time = clock();
    for (int k = 0; k < measureSize; k++) {
        j = arr[j];
    }
    s_index[1] = j;
    end_time = clock();

    unsigned int diff = end_time - start_time;

    time[0] = diff / measureSize;
}

#endif //CUDATEST_CONSTL1_5_LAT

