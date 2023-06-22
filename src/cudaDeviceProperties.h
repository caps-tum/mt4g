//
// Created by nick- on 6/26/2022.
//

#ifndef CUDATEST_CUDADEVICEPROPERTIES_CPP
#define CUDATEST_CUDADEVICEPROPERTIES_CPP

//TODO: compile switch
#define USE_HELPER_CUDA_DEFINITION

#include "eval.h"
#include <hip/hip_runtime.h>
#define __STDC_WANT_LIB_EXT1__ 1
#define HARDCODED_NUMBER_OF_CORES_IN_AMD_CU 64
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <array>
#include <regex>
#include <unistd.h>

#ifdef USE_HELPER_CUDA_DEFINITION
//#include "cuda-samples/Common/helper_cuda.h"
#endif

#define MAX_LINE_LENGTH 1024

/**
 * Converts a int-string to an int
 * @param start
 * @return
 */
int cvtCharArrToInt(char* start) {
    int num;
    sscanf(start, "%d", &num);
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
 * @param cmd - command to execute
 * @return number of cores
 */
int getCoreNumber(const std::string& cmd) {
    std::cout << "Execute command to get number of cores: " << cmd << std::endl;

#ifdef _WIN32
    if (cmd.find("nvidia-settings") != std::string::npos) {
        std::cout << "nvidia-settings does not work for windows" << std::endl;
        return 0;
    } else if(cmd.find("deviceQuery.exe") == std::string::npos) {
        std::cout << "It is required to use deviceQuery.exe" << std::endl;
        return 0;
    }
#else
    if (cmd.find("nvidia-settings") != std::string::npos && cmd.find("deviceQuery") != std::string::npos) {
        std::cout << "Nvidia-settings or deviceQuery not in command!" << std::endl;
        return 0;
    }
#endif

    FILE *p;
#ifdef _WIN32
    p = _popen(cmd.c_str(), "r");
#else
    p = popen(cmd.c_str(), "r");
#endif
    if (p == nullptr) {
        std::cout << "Could not execute command " << cmd << "!" << std::endl;
    }

    int totalNumOfCores;
    if (cmd.find("deviceQuery") != std::string::npos) {
        std::cout << "Using deviceQuery option for number of cores" << std::endl;
        char line[MAX_LINE_LENGTH] = {0};

        while (fgets(line, MAX_LINE_LENGTH, p)) {
            if (strstr(line, "core") || strstr(line, "Core")) {
                totalNumOfCores = parseCoreLine(line);
                break;
            }
        }
    } else {
        std::cout << "Using nvidia-settings option for number of cores" << std::endl;
        char num[16] = {0};
        fgets(num, 16, p);
        totalNumOfCores = std::stoi(num);
    }

#ifdef _WIN32
    _pclose(p);
#else
    pclose(p);
#endif
    return totalNumOfCores;
}

/**
 * Gets number of compute units for amd gpu
 * @return number of compute units
 */
int getComputeUnits() {
    std::array<char, 128> buffer{};
    std::string result;
    std::regex cuRegex("Compute Unit\\:[ ]+(\\d+)");
    std::smatch match;

    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen("rocminfo | grep 'Compute Unit'", "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("Error opening pipe");
    }

    std::string lastLine;
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        lastLine = buffer.data();
    }

    if (std::regex_search(lastLine, match, cuRegex)) {
        return std::stoi(match[1]);
    }

    throw std::runtime_error("Compute units not found");
}

/**
 * Gets number of work groups for amd gpu
 * @return number of work groups
 */
int getWorkGroups() {
    std::array<char, 128> buffer{};
    std::string result;
    std::regex cuRegex("Workgroup Max Size\\:[ ]+(\\d+)[0-9x()]+");
    std::smatch match;

    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen("rocminfo | grep 'Workgroup Max Size:'", "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("Error opening pipe");
    }

    std::string lastLine;
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        lastLine = buffer.data();
    }

    if (std::regex_search(lastLine, match, cuRegex)) {
        return std::stoi(match[1]);
    }

    throw std::runtime_error("Work groups not found");
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
CudaDeviceInfo getDeviceProperties(const std::string& nviCoreCmd, int coreSwitch, int deviceID) {
    CudaDeviceInfo info;

    int deviceCount;
    hipError_t result = hipGetDeviceCount(&deviceCount);
    if (result != hipSuccess) {
        std::cout << "cudaDevice/247\tget device count\t" << hipGetErrorString(result) << std::endl;
    }
    hipDeviceProp_t deviceProp{};

    if (deviceID >= deviceCount) {
        deviceID = 0;
    }

    result = hipGetDeviceProperties(&deviceProp, deviceID);
    if (result != hipSuccess) {
        std::cout << "cudaDevice/257\tget device properties\t" << hipGetErrorString(result) << std::endl;
    }

#ifdef _WIN32
    strcpy_s(info.GPUname, deviceProp.name);
#else
    strcpy(info.GPUname, deviceProp.name);
#endif
    info.cudaVersion = (float) deviceProp.major + (float) ((float) deviceProp.minor / 10.);
    info.sharedMemPerThreadBlock = deviceProp.sharedMemPerBlock;
    info.sharedMemPerSM = deviceProp.maxSharedMemoryPerMultiProcessor;

#ifdef __HIP_PLATFORM_AMD__
    info.numberOfSMs = getComputeUnits();
#else
    info.numberOfSMs = deviceProp.multiProcessorCount;
#endif

    // regsPerBlock = total number of VGPRs/CU
    //  thus - get number of workgroups and find number of VGPRs/workgroup
#ifdef __HIP_PLATFORM_AMD__
    int groups = getWorkGroups();
    info.registersPerThreadBlock = deviceProp.regsPerBlock / groups;
    info.registersPerSM = info.registersPerThreadBlock * deviceProp.warpSize;
#else
    info.registersPerThreadBlock = deviceProp.regsPerBlock;
    info.registersPerSM = info.registersPerThreadBlock;
#endif

    info.cudaMaxGlobalMem = deviceProp.totalGlobalMem;
    info.cudaMaxConstMem = deviceProp.totalConstMem;
    info.L2CacheSize = deviceProp.l2CacheSize;
    info.memClockRate = deviceProp.memoryClockRate;
    info.memBusWidth = deviceProp.memoryBusWidth;
    info.GPUClockRate = deviceProp.clockRate;
    info.maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;

#ifdef __HIP_PLATFORM_AMD__
    info.numberOfCores = info.numberOfSMs * HARDCODED_NUMBER_OF_CORES_IN_AMD_CU;
#else
	info.numberOfCores = getCoreNumber(nviCoreCmd);
#endif
    return info;
}

#endif //CUDATEST_CUDADEVICEPROPERTIES_CPP
