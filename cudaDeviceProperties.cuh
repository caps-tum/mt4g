//
// Created by nick- on 6/26/2022.
//

#ifndef CUDATEST_CUDADEVICEPROPERTIES_CUH
#define CUDATEST_CUDADEVICEPROPERTIES_CUH

//TODO: compile switch
#define USE_HELPER_CUDA_DEFINITION

#include "eval.h"
#include <cuda.h>
#define __STDC_WANT_LIB_EXT1__ 1
#include <cstring>
#include <cstdlib>
#include <cstdio>
#ifdef USE_HELPER_CUDA_DEFINITION
#include "CUDASAMPLES/Common/helper_cuda.h"
#endif

#define MAX_LINE_LENGTH 1024

/**
 * Converts a int-string to an int
 * @param start
 * @return
 */
int cvtCharArrToInt(char* start) {
    char* cvtPtr;
    int num = strtol(start, &cvtPtr, 10);

    if (start == cvtPtr) {
#ifdef IsDebug
        fprintf(out, "Char* is not an Int - Conversion failed!\n");
#endif
        return 0;
    } else if (*cvtPtr != '\0') {
#ifdef IsDebug
        fprintf(out, "Non-Int rest in Char* after Conversion - Conversion Warning!\n");
        fprintf(out, "Rest char* starts with character with ascii value: %d\n", int(cvtPtr[0]));
#endif
    } else if (errno != 0 && num == 0) {
#ifdef IsDebug
        fprintf(out, "Conversion failed!\n");
#endif
        return 0;
    }
    return num;
}

/**
 * Parses line for deviceQuery executable
 * @param line
 * @return
 */
int parseCoreLine(char* line) {
    char *ptr = strstr(line, "MP");
    if (ptr == nullptr) {
#ifdef IsDebug
        fprintf(out, "Output has unknown format!\n");
#endif
        return 0;
    }
    ptr = ptr + strlen("MP");
    if (strlen(ptr) < 10 || ptr[0] != ':' || ptr[1] != ' ' ||  ptr[2] != ' ' ||  ptr[3] != ' ' ||
            ptr[4] != ' ' || ptr[5] != ' ') {
#ifdef IsDebug
        fprintf(out, "Output has unknown format!\n");
#endif
        return 0;
    }
    ptr = ptr + 6;
    char* start = ptr;
    while(isdigit(ptr[0])) {
        ++ptr;
    }
    ptr[0] = '\0';
    return cvtCharArrToInt(start);
}

/**
 * Fetches the number of cores with a various number of options
 * @param cmd
 * @return
 */
int getCoreNumber(char* cmd) {
    printf("Execute command to get number of cores: %s\n", cmd);
#ifdef _WIN32
    if (strstr(cmd, "nvidia-settings") != nullptr) {
        printf("nvidia-settings does not work for windows\n");
        return 0;
    } else if(strstr(cmd, "deviceQuery.exe") == nullptr) {
        printf("It is required to use deviceQuery.exe\n");
        return 0;
    }
#else
    if (strstr(cmd, "nvidia-settings") != nullptr && strstr(cmd, "deviceQuery") != nullptr)  {
        printf("Nvidia-settings or deviceQuery not in command!\n");
        return 0;
    }
#endif
    FILE *p;
#ifdef _WIN32
    p = _popen(cmd, "r");
#else
    p = popen(cmd, "r");
#endif
    if (p == nullptr) {
        printf("Could not execute command %s!\n", cmd);
    }

    int totalNumOfCores;
    if (strstr(cmd, "deviceQuery") != nullptr) {
        printf("Using deviceQuery option for number of cores\n");
        char line[MAX_LINE_LENGTH] = {0};

        while (fgets(line, MAX_LINE_LENGTH, p)) {
            if (strstr(line, "core") || strstr(line, "Core")) {
                totalNumOfCores = parseCoreLine(line);
                break;
            }
        }
    } else {
        printf("Using nvidia-settings option for number of cores\n");
        char num[16] = {0};
        fgets(num, 16, p);
        totalNumOfCores = cvtCharArrToInt(num);
    }

#ifdef _WIN32
    _pclose(p);
#else
    pclose(p);
#endif
    return totalNumOfCores;
}

typedef struct CudaDeviceInfo {
    /* the name of the GPU device */
    char GPUname[256];
    /* the cuda compute capability of the device */
    float cudaVersion;
    /* the number of Streaming Multiprocessors */
    int numberOfSMs;
    /* the number of cores per SM*/
    int numberOfCores;
    /* the amount of shared memory per thread block */
    size_t sharedMemPerThreadBlock;
    /* the amount of shared memory per streaming multiprocessor in bytes */
    size_t sharedMemPerSM;
    /* the amount of registers per thread block in bytes */
    int registersPerThreadBlock;
    /* the amount of registers per streaming multiprocessors */
    int registersPerSM;
    /* the amount of global memory available in bytes */
    size_t cudaMaxGlobalMem;
    /* the amount of constant memory available in bytes */
    size_t  cudaMaxConstMem;
    /* the size of the L2 Cache in bytes */
    int L2CacheSize;
    /* memory clock frequency in Kilohertz */
    int memClockRate;
    /* memory bus width in bits */
    int memBusWidth;
    /* GPU clock frequency in Kilohertz */
    int GPUClockRate;
    /* Maximum Number of threads in a thread block */
    int maxThreadsPerBlock;
} CudaDeviceInfo;

/**
 * Get every possible information from CUDA
 * @param nviCoreCmd
 * @param coreSwitch
 * @param deviceID
 * @return
 */
CudaDeviceInfo getDeviceProperties(char* nviCoreCmd, int coreSwitch, int deviceID) {
    CudaDeviceInfo info;

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cudaDeviceProp deviceProp{};

    if (deviceID >= deviceCount) {
        deviceID = 0;
    }

    cudaGetDeviceProperties(&deviceProp, deviceID);
#ifdef _WIN32
    strcpy_s(info.GPUname, deviceProp.name);
#else
    strcpy(info.GPUname, deviceProp.name);
#endif
    info.cudaVersion = (float)deviceProp.major + (float)((float)deviceProp.minor / 10.);
    info.sharedMemPerThreadBlock = deviceProp.sharedMemPerBlock;
    info.sharedMemPerSM = deviceProp.sharedMemPerMultiprocessor;
    info.numberOfSMs = deviceProp.multiProcessorCount;
    info.registersPerThreadBlock = deviceProp.regsPerBlock;
    info.registersPerSM = deviceProp.regsPerMultiprocessor;
    info.cudaMaxGlobalMem = deviceProp.totalGlobalMem;
    info.cudaMaxConstMem = deviceProp.totalConstMem;
    info.L2CacheSize = deviceProp.l2CacheSize;
    info.memClockRate = deviceProp.memoryClockRate;
    info.memBusWidth = deviceProp.memoryBusWidth;
    info.GPUClockRate = deviceProp.clockRate;
    info.maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;

    if (coreSwitch == 0) {
        printf("Using helper_cuda option for number of cores\n");
        info.numberOfCores = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * info.numberOfSMs;
    } else {
        info.numberOfCores = getCoreNumber(nviCoreCmd);
    }
    return info;
}

#endif //CUDATEST_CUDADEVICEPROPERTIES_CUH
