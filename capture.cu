//
// Created by nick- on 6/22/2022.
//
#ifndef CAPTURE
#define CAPTURE

# include "eval.h"
# include "cudaDeviceProperties.cuh"
# include "L1DataCache.cuh"
# include "ReadOnlyCache.cuh"
# include "TextureCache.cuh"
# include "ConstantCache.cuh"
# include "L2DataCache.cuh"
# include "l1_l2_diff.cuh"
# include "ShareChk/chkConstShareL1Data.cuh"
# include "ShareChk/chkROShareTexture.cuh"
# include "ShareChk/chkROShareL1Data.cuh"
# include "ShareChk/chkL1ShareTexture.cuh"
# include "ShareChk/chkTwoTexture.cuh"
# include "ShareChk/chkTwoRO.cuh"
# include "ShareChk/chkTwoL1.cuh"
# include "ShareChk/chkTwoC1.cuh"
# include "MainMemory.cuh"
# include "SharedMemory.cuh"
# include "ShareChk/chkAllCore.cuh"

# include <climits>

bool L1ShareConst = false;
bool ROShareL1Data = false;
bool ROShareTexture = false;
bool L1ShareTexture = false;

enum Caches {
    L2, Texture, RO, Const1, Const2, L1, MAIN, SHARED
};

char unitsByte[4][4] = {"B", "KiB", "MiB", "GiB"};
char unitsHz[3][4] = {"KHz", "MHz", "GHz"};
char shared_where [8][4] = {"GPU", "SM", "SM", "SM", "SM", "SM", "GPU", "SM"};

CacheResults overallResults[8];

// Returns for a given byte size the best fitting unit and its value (KiB, MiB, GiB)
const char* getSizeNiceFormatByte(double* val, size_t original) {
    int unitIndex = 0;

    if (original > 1024 * 1024 * 1024) {
        original = original >> 10;
        ++unitIndex;
    }

    double result = (double) original;

    if (result > 1000.) {
        result = result / 1024.;
        ++unitIndex;
    }

    if (result > 1000.) {
        result = result / 1024.;
        ++unitIndex;
    }

    const char* unit = unitsByte[unitIndex];
    *val = result;
    return unit;
}

// Returns for a given byte size the best fitting unit and its value (KHz, MHz, GHz)
const char* getSizeNiceFormatHertz(double* val, unsigned int original) {
    int unitIndex = 0;

    if (original > 1000 * 1000 * 1000) {
        original = original  / 1000;
        ++unitIndex;
    }

    double result = (double) original;

    if (unitIndex == 0) {
        if (result > 1000.) {
            result = result / 1000.;
            ++unitIndex;
        }
    }

    if (result > 1000.) {
        result = result / 1000.;
        ++unitIndex;
    }

    const char* unit = unitsHz[unitIndex];
    *val = result;
    return unit;
}

void printOverallBenchmarkResults(CacheResults* result, bool L1_global_load_enabled, CudaDeviceInfo cudaInfo) {
    printf("\n\n**************************************************\n");
    printf("\tPRINT GPU BENCHMARK RESULT\n");
    printf("**************************************************\n\n");

    char outputCSV[] = "GPU_Memory_Topology.csv";
    FILE *csv = fopen(outputCSV, "w");
    if (csv == nullptr) {
        printf("[WARNING]: Cannot open file for writing - close csv file if currently open\n");
        csv = stdout;
    }

    printf("GPU name: %s\n\n", cudaInfo.GPUname);
    fprintf(csv, "GPU_INFORMATION; GPU_vendor; \"Nvidia\"; GPU_name; \"%s\"\n", cudaInfo.GPUname);

    printf("PRINT COMPUTE RESOURCE INFORMATION:\n");
    fprintf(csv, "COMPUTE_RESOURCE_INFORMATION; ");
    printf("CUDA compute capability: %.2f\n", cudaInfo.cudaVersion);
    fprintf(csv, "CUDA_compute_capability; \"%.2f\"; ", cudaInfo.cudaVersion);
    printf("Number Of streaming multiprocessors: %d\n", cudaInfo.numberOfSMs);
    fprintf(csv, "Number_of_streaming_multiprocessors; %d; ", cudaInfo.numberOfSMs);
    printf("Number Of Cores in GPU: %d\n", cudaInfo.numberOfCores);
    fprintf(csv, "Number_of_cores_in_GPU; %d; ", cudaInfo.numberOfCores);
    printf("Number Of Cores/SM in GPU: %d\n\n", cudaInfo.numberOfCores / cudaInfo.numberOfSMs);
    fprintf(csv, "Number_of_cores_per_SM; %d\n", cudaInfo.numberOfCores / cudaInfo.numberOfSMs);

    printf("PRINT REGISTER INFORMATION:\n");
    fprintf(csv, "REGISTER_INFORMATION; ");
    printf("Registers per thread block: %d 32-bit registers\n", cudaInfo.registersPerThreadBlock);
    fprintf(csv, "Registers_per_thread_block; %d; \"32-bit registers\"; ", cudaInfo.registersPerThreadBlock);
    printf("Registers per SM: %d 32-bit registers\n\n", cudaInfo.registersPerSM);
    fprintf(csv, "Registers_per_SM; %d; \"32-bit registers\"\n", cudaInfo.registersPerSM);

    printf("PRINT ADDITIONAL INFORMATION:\n");
    fprintf(csv, "ADDITIONAL_INFORMATION; ");

    double val;
    unsigned int originalFrequency = cudaInfo.memClockRate;
    const char* MemClockFreqUnit = getSizeNiceFormatHertz(&val, originalFrequency);
    printf("Memory Clock Frequency: %.3f %s\n", val, MemClockFreqUnit);
    fprintf(csv, "Memory_Clock_Frequency; %.3f; \"%s\"; ", val, MemClockFreqUnit);
    printf("Memory Bus Width: %d bits\n", cudaInfo.memBusWidth);
    fprintf(csv, "Memory_Bus_Width; %d; \"bit\"; ", cudaInfo.memBusWidth);

    originalFrequency = cudaInfo.GPUClockRate;
    const char* GPUClockFreqUnit = getSizeNiceFormatHertz(&val, originalFrequency);
    printf("GPU Clock rate: %.3f %s\n\n", val, GPUClockFreqUnit);
    fprintf(csv, "GPU_Clock_Rate; %.3f; \"%s\"\n", val, GPUClockFreqUnit);

    fprintf(csv, "L1_DATA_CACHE; ");
    if (!L1_global_load_enabled) {
        printf("L1 DATA CACHE INFORMATION missing: GPU does not allow caching of global loads in L1\n");
        fprintf(csv, "\"N/A\"\n");
    } else {
        if (result[L1].benchmarked) {
            printf("PRINT L1 DATA CACHE INFORMATION:\n");

            if (result[L1].CacheSize.realCP) {
                double size;
                size_t original = result[L1].CacheSize.CacheSize;
                const char* unit = getSizeNiceFormatByte(&size, original);
                printf("Detected L1 Data Cache Size: %f %s\n", size, unit);
                fprintf(csv, "Size; %f; %s; \"%c\"; ", size, unit, '=');
            } else {
                double size;
                size_t original = result[L1].CacheSize.maxSizeBenchmarked;
                const char* unit = getSizeNiceFormatByte(&size, original);
                printf("Detected L1 Data Cache Size: >= %f %s\n", size, unit);
                fprintf(csv, "Size; %f; %s; \"%s\"; ", size, unit, ">=");
            }
            printf("Detected L1 Data Cache Line Size: %d B\n", result[L1].cacheLineSize);
            fprintf(csv, "Cache_Line_Size; %d; \"B\"; ", result[L1].cacheLineSize);
            printf("Detected L1 Data Cache Load Latency: %d cycles\n", result[L1].latencyCycles);
            fprintf(csv, "Load_Latency; %d; \"cycles\"; ", result[L1].latencyCycles);
            printf("Detected L1 Data Cache Load Latency: %d nanoseconds\n", result[L1].latencyNano);
            fprintf(csv, "Load_Latency; %d; \"nanoseconds\"; ", result[L1].latencyNano);
            printf("L1 Data Cache Is Shared On %s-level\n", shared_where[L1]);
            fprintf(csv, "Shared_On; \"%s-level\"; ", shared_where[L1]);
            printf("Does L1 Data Cache Share the physical cache with the Texture Cache? %s\n", L1ShareTexture ? "Yes" : "No");
            fprintf(csv, "Share_Cache_With_Texture; %d; ", L1ShareTexture);
            printf("Does L1 Data Cache Share the physical cache with the Read-Only Cache? %s\n", ROShareL1Data ? "Yes" : "No");
            fprintf(csv, "Share_Cache_With_Read-Only; %d; ", ROShareL1Data);
            printf("Does L1 Data Cache Share the physical cache with the Constant L1 Cache? %s\n", L1ShareConst ? "Yes" : "No");
            fprintf(csv, "Share_Cache_With_ConstantL1; %d; ", L1ShareConst);
            //printf("Detected L1 Cache Load Bandwidth: %llu MB / s\n\n", result[L1].bandwidth);
            printf("Detected L1 Data Caches Per SM: %d\n\n", result[L1].numberPerSM);
            fprintf(csv, "Caches_Per_SM; %d\n", result[L1].numberPerSM);
        } else {
            printf("L1 Data CACHE WAS NOT BENCHMARKED!\n\n");
            fprintf(csv, "\"N/A\"\n");
        }
    }

    fprintf(csv, "L2_DATA_CACHE; ");
    if (result[L2].benchmarked) {
        printf("PRINT L2 CACHE INFORMATION:\n");
        if (result[L2].CacheSize.realCP) {
            double size;
            size_t original = result[L2].CacheSize.CacheSize;
            const char* unit = getSizeNiceFormatByte(&size, original);
            printf("Detected L2 Cache Size: %.3f %s\n", size, unit);
            fprintf(csv, "Size; %.3f; %s; \"%c\"; ", size, unit, '=');
        } else {
            double size;
            size_t original = result[L2].CacheSize.maxSizeBenchmarked;
            const char* unit = getSizeNiceFormatByte(&size, original);
            printf("Detected L2 Cache Size: >= %.3f %s\n", size, unit);
            fprintf(csv, "Size; %.3f; %s; \"%s\"; ", size, unit, ">=");
        }
        printf("Detected L2 Cache Line Size: %d B\n", result[L2].cacheLineSize);
        fprintf(csv, "Cache_Line_Size; %d; \"B\"; ", result[L2].cacheLineSize);
        printf("Detected L2 Cache Load Latency: %d cycles\n", result[L2].latencyCycles);
        fprintf(csv, "Load_Latency; %d; \"cycles\"; ", result[L2].latencyCycles);
        printf("Detected L2 Cache Load Latency: %d nanoseconds\n", result[L2].latencyNano);
        fprintf(csv, "Load_Latency; %d; \"nanoseconds\"; ", result[L2].latencyNano);
        printf("L2 Cache Is Shared On %s-level\n\n", shared_where[L2]);
        fprintf(csv, "Shared_On; \"%s-level\"\n", shared_where[L2]);
    } else {
        printf("L2 CACHE WAS NOT BENCHMARKED!\n\n");
        fprintf(csv, "\"N/A\"\n");
    }

    fprintf(csv, "TEXTURE_CACHE; ");
    if (result[Texture].benchmarked) {
        printf("PRINT TEXTURE CACHE INFORMATION:\n");
        if (result[Texture].CacheSize.realCP) {
            double size;
            size_t original = result[Texture].CacheSize.CacheSize;
            const char* unit = getSizeNiceFormatByte(&size, original);
            printf("Detected Texture Cache Size: %f %s\n", size, unit);
            fprintf(csv, "Size; %f; %s; \"%c\"; ", size, unit, '=');
        } else {
            double size;
            size_t original = result[Texture].CacheSize.maxSizeBenchmarked;
            const char* unit = getSizeNiceFormatByte(&size, original);
            printf("Detected Texture Cache Size: >= %f %s\n", size, unit);
            fprintf(csv, "Size; %f; %s; \"%s\"; ", size, unit, ">=");
        }
        printf("Detected Texture Cache Line Size: %d B\n", result[Texture].cacheLineSize);
        fprintf(csv, "Cache_Line_Size; %d; \"B\"; ", result[Texture].cacheLineSize);
        printf("Detected Texture Cache Load Latency: %d cycles\n", result[Texture].latencyCycles);
        fprintf(csv, "Load_Latency; %d; \"cycles\"; ", result[Texture].latencyCycles);
        printf("Detected Texture Cache Load Latency: %d nanoseconds\n", result[Texture].latencyNano);
        fprintf(csv, "Load_Latency; %d; \"nanoseconds\"; ", result[Texture].latencyNano);
        printf("Texture Cache Is Shared On %s-level\n", shared_where[Texture]);
        fprintf(csv, "Shared_On; \"%s-level\"; ", shared_where[Texture]);
        printf("Does Texture Cache Share the physical cache with the L1 Data Cache? %s\n", L1ShareTexture ? "Yes" : "No");
        fprintf(csv, "Share_Cache_With_L1_Data; %d; ", L1ShareTexture);
        printf("Does Texture Cache Share the physical cache with the Read-Only Cache? %s\n", ROShareTexture ? "Yes" : "No");
        fprintf(csv, "Share_Cache_With_Read-Only; %d; ", ROShareTexture);
        printf("Detected Texture Caches Per SM: %d\n\n", result[Texture].numberPerSM);
        fprintf(csv, "Caches_Per_SM; %d\n", result[Texture].numberPerSM);
    } else {
        printf("TEXTURE CACHE WAS NOT BENCHMARKED!\n\n");
        fprintf(csv, "\"N/A\"\n");
    }

    fprintf(csv, "READ-ONLY_CACHE; ");
    if (result[RO].benchmarked) {
        printf("PRINT Read-Only CACHE INFORMATION:\n");
        if (result[RO].CacheSize.realCP) {
            double size;
            size_t original = result[RO].CacheSize.CacheSize;
            const char* unit = getSizeNiceFormatByte(&size, original);
            printf("Detected Read-Only Cache Size: %f %s\n", size, unit);
            fprintf(csv, "Size; %f; %s; \"%c\"; ", size, unit, '=');
        } else {
            double size;
            size_t original = result[RO].CacheSize.maxSizeBenchmarked;
            const char* unit = getSizeNiceFormatByte(&size, original);
            printf("Detected Read-Only Cache Size: >= %f %s\n", size, unit);
            fprintf(csv, "Size; %f; %s; \"%s\"; ", size, unit, ">=");
        }
        printf("Detected Read-Only Cache Line Size: %d B\n", result[RO].cacheLineSize);
        fprintf(csv, "Cache_Line_Size; %d; \"B\"; ", result[RO].cacheLineSize);
        printf("Detected Read-Only Cache Load Latency: %d cycles\n", result[RO].latencyCycles);
        fprintf(csv, "Load_Latency; %d; \"cycles\"; ", result[RO].latencyCycles);
        printf("Detected Read-Only Cache Load Latency: %d nanoseconds\n", result[RO].latencyNano);
        fprintf(csv, "Load_Latency; %d; \"nanoseconds\"; ", result[RO].latencyNano);
        printf("Read-Only Cache Is Shared On %s-level\n", shared_where[RO]);
        fprintf(csv, "Shared_On; \"%s-level\"; ", shared_where[RO]);
        printf("Does Read-Only Cache Share the physical cache with the L1 Data Cache? %s\n", ROShareL1Data ? "Yes" : "No");
        fprintf(csv, "Share_Cache_With_L1_Data; %d; ", ROShareL1Data);
        printf("Does Read-Only Cache Share the physical cache with the Texture Cache? %s\n", ROShareTexture ? "Yes" : "No");
        fprintf(csv, "Share_Cache_With_Texture; %d; ", ROShareTexture);
        printf("Detected Read-Only Caches Per SM: %d\n\n", result[RO].numberPerSM);
        fprintf(csv, "Caches_Per_SM; %d\n", result[RO].numberPerSM);
    } else {
        printf("READ-ONLY CACHE WAS NOT BENCHMARKED!\n\n");
        fprintf(csv, "\"N/A\"\n");
    }

    fprintf(csv, "CONSTANT_L1_CACHE; ");
    if (result[Const1].benchmarked) {
        printf("PRINT CONSTANT CACHE L1 INFORMATION:\n");
        if (result[Const1].CacheSize.realCP) {
            double size;
            size_t original = result[Const1].CacheSize.CacheSize;
            const char* unit = getSizeNiceFormatByte(&size, original);
            printf("Detected Constant L1 Cache Size: %f %s\n", size, unit);
            fprintf(csv, "Size; %f; %s; \"%c\"; ", size, unit, '=');
        } else {
            double size;
            size_t original = result[Const1].CacheSize.maxSizeBenchmarked;
            const char* unit = getSizeNiceFormatByte(&size, original);
            printf("Detected Constant L1 Cache Size: >= %f %s\n", size, unit);
            fprintf(csv, "Size; %f; %s; \"%s\"; ", size, unit, ">=");
        }
        printf("Detected Constant L1 Cache Line Size: %d B\n", result[Const1].cacheLineSize);
        fprintf(csv, "Cache_Line_Size; %d; \"B\"; ", result[Const1].cacheLineSize);
        printf("Detected Constant L1 Cache Load Latency: %d cycles\n", result[Const1].latencyCycles);
        fprintf(csv, "Load_Latency; %d; \"cycles\"; ", result[Const1].latencyCycles);
        printf("Detected Constant L1 Cache Load Latency: %d nanoseconds\n", result[Const1].latencyNano);
        fprintf(csv, "Load_Latency; %d; \"nanoseconds\"; ", result[Const1].latencyNano);
        printf("Constant L1 Cache Is Shared On %s-level\n", shared_where[Const1]);
        fprintf(csv, "Shared_On; \"%s-level\"; ", shared_where[Const1]);
        printf("Does Constant L1 Cache Share the physical cache with the L1 Data Cache? %s\n\n", L1ShareConst ? "Yes" : "No");
        fprintf(csv, "Share_Cache_With_L1_Data; %d; ", L1ShareConst);
        printf("Detected Constant L1 Caches Per SM: %d\n\n", result[Const1].numberPerSM);
        fprintf(csv, "Caches_Per_SM; %d\n", result[Const1].numberPerSM);
    } else {
        printf("CONSTANT CACHE L1 WAS NOT BENCHMARKED!\n");
        fprintf(csv, "\"N/A\"\n");
    }

    fprintf(csv, "CONST_L1_5_CACHE; ");
    if (result[Const2].benchmarked) {
        printf("PRINT CONSTANT L1.5 CACHE INFORMATION:\n");
        if (result[Const2].CacheSize.realCP) {
            double size;
            size_t original = result[Const2].CacheSize.CacheSize;
            const char* unit = getSizeNiceFormatByte(&size, original);
            printf("Detected Constant L1.5 Cache Size: %f %s\n", size, unit);
            fprintf(csv, "Size; %f; %s; \"%c\"; ", size, unit, '=');
        } else {
            double size;
            size_t original = result[Const2].CacheSize.maxSizeBenchmarked;
            const char* unit = getSizeNiceFormatByte(&size, original);
            printf("Detected Constant L1.5 Cache Size: >= %f %s\n", size, unit);
            fprintf(csv, "Size; %f; %s; \"%s\"; ", size, unit, ">=");
        }
        printf("Detected Constant L1.5 Cache Line Size: %d B\n", result[Const2].cacheLineSize);
        fprintf(csv, "Cache_Line_Size; %d; \"B\"; ", result[Const2].cacheLineSize);
        printf("Detected Constant L1.5 Cache Load Latency: %d cycles\n", result[Const2].latencyCycles);
        fprintf(csv, "Load_Latency; %d; \"cycles\"; ", result[Const2].latencyCycles);
        printf("Detected Constant L1.5 Cache Load Latency: %d nanoseconds\n", result[Const2].latencyNano);
        fprintf(csv, "Load_Latency; %d; \"nanoseconds\"; ", result[Const2].latencyNano);
        printf("Const L1.5 Cache Is Shared On %s-level\n\n", shared_where[Const2]);
        fprintf(csv, "Shared_On; \"%s-level\"\n", shared_where[Const2]);
    } else {
        printf("CONSTANT CACHE L1.5 WAS NOT BENCHMARKED!\n");
        fprintf(csv, "\"N/A\"\n");
    }

    fprintf(csv, "MAIN_MEMORY; ");
    if (result[MAIN].benchmarked) {
        printf("PRINT MAIN MEMORY INFORMATION:\n");
        if (result[MAIN].CacheSize.realCP) {
            double size;
            size_t original = result[MAIN].CacheSize.CacheSize;
            const char* unit = getSizeNiceFormatByte(&size, original);
            printf("Detected Main Memory Size: %f %s\n", size, unit);
            fprintf(csv, "Size; %f; %s; \"%c\"; ", size, unit, '=');
        } else {
            double size;
            size_t original = result[MAIN].CacheSize.maxSizeBenchmarked;
            const char* unit = getSizeNiceFormatByte(&size, original);
            printf("Detected Main Memory Size: >= %f %s\n", size, unit);
            fprintf(csv, "Size; %f; %s; \"%s\"; ", size, unit, ">=");
        }
        printf("Detected Main Memory Load Latency: %d cycles\n", result[MAIN].latencyCycles);
        fprintf(csv, "Load_Latency; %d; \"cycles\"; ", result[MAIN].latencyCycles);
        printf("Detected Main Memory Load Latency: %d nanoseconds\n", result[MAIN].latencyNano);
        fprintf(csv, "Load_Latency; %d; \"nanoseconds\"; ", result[MAIN].latencyNano);
        printf("Main Memory Is Shared On %s-level\n\n", shared_where[MAIN]);
        fprintf(csv, "Shared_On; \"%s-level\"\n", shared_where[MAIN]);
    } else {
        printf("MAIN MEMORY WAS NOT BENCHMARKED!\n");
        fprintf(csv, "\"N/A\"\n");
    }

    fprintf(csv, "SHARED_MEMORY; ");
    if (result[SHARED].benchmarked) {
        printf("PRINT SHARED MEMORY INFORMATION:\n");
        if (result[SHARED].CacheSize.realCP) {
            double size;
            size_t original = result[SHARED].CacheSize.CacheSize;
            const char* unit = getSizeNiceFormatByte(&size, original);
            printf("Detected Shared Memory Size: %f %s\n", size, unit);
            fprintf(csv, "Size; %f; %s; \"%c\"; ", size, unit, '=');
        } else {
            double size;
            size_t original = result[SHARED].CacheSize.maxSizeBenchmarked;
            const char* unit = getSizeNiceFormatByte(&size, original);
            printf("Detected Shared Memory Size: >= %f %s\n", size, unit);
            fprintf(csv, "Size; %f; %s; \"%s\"; ", size, unit, ">=");
        }
        printf("Detected Shared Memory Load Latency: %d cycles\n", result[SHARED].latencyCycles);
        fprintf(csv, "Load_Latency; %d; \"cycles\"; ", result[SHARED].latencyCycles);
        printf("Detected Shared Memory Load Latency: %d nanoseconds\n", result[SHARED].latencyNano);
        fprintf(csv, "Load_Latency; %d; \"nanoseconds\"; ", result[SHARED].latencyNano);
        printf("Shared Memory Is Shared On %s-level\n\n", shared_where[SHARED]);
        fprintf(csv, "Shared_On; \"%s-level\"\n", shared_where[SHARED]);
    } else {
        printf("SHARED MEMORY WAS NOT BENCHMARKED!\n");
        fprintf(csv, "\"N/A\"\n");
    }
    fclose(csv);
}

// Fills results with additional CUDA information
void fillWithCUDAInfo(CudaDeviceInfo cudaInfo, size_t totalMem) {
    overallResults[SHARED].CacheSize.CacheSize = cudaInfo.sharedMemPerSM;
    overallResults[SHARED].CacheSize.realCP = true;
    overallResults[SHARED].CacheSize.maxSizeBenchmarked = cudaInfo.sharedMemPerSM;

    overallResults[L2].CacheSize.CacheSize = cudaInfo.L2CacheSize;
    overallResults[L2].CacheSize.realCP = true;
    overallResults[L2].CacheSize.maxSizeBenchmarked = cudaInfo.L2CacheSize;

    overallResults[MAIN].CacheSize.CacheSize = totalMem;
    overallResults[MAIN].CacheSize.realCP = true;
    overallResults[MAIN].CacheSize.maxSizeBenchmarked = totalMem;
}

// command line args, if true, corresponding size/latency benchmark is executed
bool l1 = false;
bool l2 = false;
bool ro = false;
bool txt = false;
bool constant = false;

//0 = cuda_helper.h
//1 = deviceQuery
//2 = nvidia-settings
int coreSwitch = 0;

int deviceID = 0;

#define coreQuerySize 1024
char cudaCoreQueryPath[coreQuerySize];

void parseArgs(int argc, char *argv[]) {
#ifdef _WIN32
    printf("Usage: MemTop.exe [OPTIONS]\n"
#else
    printf("Usage: ./MemTop [OPTIONS]\n"
#endif
           "\nOPTIONS\n=============================================\n"
           "\n-p:<path>: \n\tOverwrites the source of information for the number of Cuda Cores\n\t<path> specifies the path to the directory, that contains the \'deviceQuery\' executable"
           "\n-p: Overwrites the source of information for the number of Cuda Cores, uses \'nvidia-settings\'"
           "\n-d:<id> Sets the device, that will be benchmarked"
           "\n-l1: Turns on benchmark for l1 data cache"
           "\n-l2: Turns on benchmark for l2 data cache"
           "\n-txt: Turns on benchmark for texture cache"
           "\n-ro: Turns on benchmark for read-only cache"
           "\n-c: Turns on benchmark for constant cache"
           "\n\nIf none of the benchmark switches is used, every benchmark is executed!\n");

    bool bFlags = false;

    for (int i = 1; i < argc; i++) {
        char* arg = argv[i];
        if (strcmp(arg, "-l1") == 0) {
            printf("Will execute L1 Data Check\n");
            l1 = bFlags = true;
        } else if (strcmp(arg, "-l2") == 0) {
            l2 = bFlags = true;
            printf("Will execute L2 Check\n");
        } else if (strcmp(arg, "-ro") == 0) {
            ro = bFlags = true;
            printf("Will execute RO Check\n");
        } else if (strcmp(arg, "-txt") == 0) {
            txt = bFlags = true;
            printf("Will execute Texture Check\n");
        } else if (strcmp(arg, "-c") == 0) {
            constant = bFlags = true;
            printf("Will execute Constant Check\n");
        } else if (strcmp(arg, "-p") == 0) {
#ifdef _WIN32
            printf("nvidia-settings is only available on linux plattforms\n");
#else
            coreSwitch = 1;
            snprintf(cudaCoreQueryPath, coreQuerySize, "nvidia-settings -q CUDACores -t");
#endif
        } else if (strstr(arg, "-p:") != nullptr) {
            coreSwitch = 2;
            char* path = &arg[strlen("-p:")];
#ifdef _WIN32
            if (strlen(path) + strlen("/deviceQuery.exe") > coreQuerySize) {
                printf("Path to \'deviceQuery\' is too long (> %d)\n", coreQuerySize);
                coreSwitch = 0;
            } else {
                snprintf(cudaCoreQueryPath, coreQuerySize, "\"%s%cdeviceQuery.exe\"", path, separator());
            }
#else
            if (strlen(path) + strlen("/deviceQuery") > coreQuerySize) {
                printf("Path to \'deviceQuery\' is too long (> %d)\n", coreQuerySize);
                coreSwitch = 0;
            } else {
                snprintf(cudaCoreQueryPath, coreQuerySize, "\"%s%cdeviceQuery\"", path, separator());
            }
#endif
        } else if (strstr(arg, "-d:") != nullptr) {
            char* id = &arg[strlen("-d:")];
            deviceID = cvtCharArrToInt(id);
            printf("Specified deviceID %d\n", deviceID);
        } else {
            printf("Unknown parameter \"%s\"\n", arg);
        }
    }

    if (!bFlags) {
        l1 = l2 = ro = txt = constant = true;
        printf("Will execute All Checks\n");
    }
}
int main(int argc, char *argv[]){
    parseArgs(argc, argv);

    cleanupOutput();

    // Use first device (in case of multi-GPU machine)
    int numberGPUs;
    cudaGetDeviceCount(&numberGPUs);
    if (deviceID >= numberGPUs) {
        printf("Specified device ID %d >= %d(number of installed GPUs) - will use default GPU 0!\n"
               "Use \'nvidia-smi\' to see the ID of the desired GPU device!\n", deviceID, numberGPUs);
        deviceID = 0;
    }
    cudaSetDevice(deviceID);

#ifdef IsDebug
    out = fopen("GPUlog.log", "w");
#endif //IsDebug

    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);

    CudaDeviceInfo cudaInfo = getDeviceProperties(cudaCoreQueryPath, coreSwitch, deviceID);

    CacheResults L1_results;
    CacheResults L2_results;
    CacheResults textureResults;
    CacheResults ReadOnlyResults;

    printf( "\n\nMeasure if L1 is used for caching global loads\n");
    bool L1_used_for_global_loads = measureL1_L2_difference(25.);

#ifdef IsDebug
    if (L1_used_for_global_loads) {
        fprintf(out, "--L1 Cache is used for global loads\n");
    } else {
        fprintf(out, "--L1 Cache is not used for global loads\n");
    }
#endif //IsDebug

    if (l2) {
        // L2 Data Cache Checks
        L2_results = executeL2DataCacheChecks(cudaInfo.L2CacheSize);
        overallResults[L2] = L2_results;

        cudaDeviceReset();
    }

    if (L1_used_for_global_loads && l1) {
        // L1 Data Cache Checks
        L1_results = executeL1DataCacheChecks();
#ifdef IsDebug
        fprintf(out, "Detected L1 Cache Size: %f KiB\n", (double)L1_results.CacheSize.CacheSize / 1024.);
        fprintf(out, "Detected L1 Latency in cycles: %d\n", L1_results.latencyCycles);
#endif //IsDebug
        overallResults[L1] = L1_results;

        cudaDeviceReset();
    }

    if (txt) {
        // Texture Cache Checks
        textureResults = executeTextureCacheChecks();
#ifdef IsDebug
        fprintf(out, "Detected Texture Cache Size: %f KiB\n", (double) textureResults.CacheSize.CacheSize / 1024.);
#endif //IsDebug
        overallResults[Texture] = textureResults;

        cudaDeviceReset();
    }

    if (ro) {
        // Read Only Cache Checks
        ReadOnlyResults = executeReadOnlyCacheChecks();
#ifdef IsDebug
        fprintf(out, "Detected Read-Only Cache Size: %f KiB\n", (double) ReadOnlyResults.CacheSize.CacheSize / 1024.);
#endif //IsDebug
        overallResults[RO] = ReadOnlyResults;

        cudaDeviceReset();
    }

    if (constant) {
        // Constant Cache L1 & L1.5 Checks
        Tuple<CacheResults> results = executeConstantCacheChecks(deviceID);
        overallResults[Const1] = results.first;
        overallResults[Const2] = results.second;
        cudaDeviceReset();
    }

    double TxtDistance = 0.;
    if (textureResults.benchmarked) {
        // Reference Value for interference if two threads/cores access texture cache
        printf("\n\nCheck two tex Share check\n");
        TxtDistance = measure_TwoTexture(textureResults.CacheSize.CacheSize, 200);
        // Now check how many texture caches are in one SM
    }

    double RODistance = 0.;
    if (ReadOnlyResults.benchmarked) {
        // Reference Value for interference if two threads/cores access ro cache
        printf("\n\nCheck two RO Share check\n");
        RODistance = measure_TwoRO(ReadOnlyResults.CacheSize.CacheSize, 200);
    }

    double C1Distance = 0.;
    if (overallResults[Const1].benchmarked) {
        // Reference Value for interference if two threads/cores access constant cache
        printf("\n\nCheck two C1 Share check\n");
        C1Distance = (double)measure_TwoC1(overallResults[Const1].CacheSize.CacheSize, 10);
    }

    dTuple constDataShareResult = {0., 0.};
    dTuple roTextureShareResult = {0., 0.};
    dTuple roDataShareResult = {0., 0.};
    dTuple dataTextureShareResult = {0., 0.};
    double shareThresholdConst = C1Distance / 3;
    double shareThresholdTxt = TxtDistance / 3;
    double shareThresholdRo = RODistance / 3;
    double shareThresholdData = 10;
#ifdef IsDebug
    fprintf(out, "Measured distances: Txt = %f, RO = %f, C1 = %f\n", TxtDistance, RODistance, C1Distance);
#endif

    // Check if L1, Texture and/or Readonly Cache are the same physical cache
    if (L1_used_for_global_loads && L1_results.benchmarked) {

        printf("\n\nCheck two L1 Share check\n");
        double L1Distance = measure_TwoL1(L1_results.CacheSize.CacheSize, 200);
        shareThresholdData = L1Distance / 3;
#ifdef IsDebug
        fprintf(out, "Measured distances: L1D = %f\n", L1Distance);
#endif

        if (overallResults[Const1].benchmarked) {
            printf("\n\nCheck if Const L1 Cache and Data L1 Cache share the same Cache Hardware physically\n");
            constDataShareResult = measure_ConstShareData(overallResults[Const1].CacheSize.CacheSize,
                                                          L1_results.CacheSize.CacheSize, 800);
            if ((C1Distance - shareThresholdConst) < constDataShareResult.first ||
                constDataShareResult.second > shareThresholdData || (L1Distance - shareThresholdData) < constDataShareResult.second) {
#ifdef IsDebug
                printf("L1 Data Cache and Constant Cache share the same physical space\n");
                fprintf(out, "L1 Data Cache and Constant Cache share the same physical space\n");
#endif
                L1ShareConst = true;
            } else {
#ifdef IsDebug
                printf("L1 Data Cache and Constant Cache have separate physical spaces\n");
                fprintf(out, "L1 Data Cache and Constant Cache have separate physical spaces\n");
#endif //IsDebug
                L1ShareConst = false;
            }

            cudaDeviceReset();
        }

        if (ReadOnlyResults.benchmarked) {
            printf("\n\nCheck if Read Only Cache and L1 Data Cache share the same Cache Hardware physically\n");
            roDataShareResult = measure_ROShareL1Data(ReadOnlyResults.CacheSize.CacheSize,
                                                      L1_results.CacheSize.CacheSize, 800);
            if (roDataShareResult.first > shareThresholdRo || (RODistance - shareThresholdRo) < roDataShareResult.first ||
                roDataShareResult.second > shareThresholdData || (L1Distance - shareThresholdData) < roDataShareResult.second) {
#ifdef IsDebug
                printf("Read-Only Cache and L1 Data Cache share the same physical space\n");
                fprintf(out, "Read-Only Cache and L1 Data Cache share the same physical space\n");
#endif //IsDebug
                ROShareL1Data = true;
            } else {
#ifdef IsDebug
                printf("Read-Only Cache and L1 Data Cache have separate physical spaces\n");
                fprintf(out, "Read-Only Cache and L1 Data Cache have separate physical spaces\n");
#endif //IsDebug
                ROShareL1Data = false;
            }

            cudaDeviceReset();
        }

        if (textureResults.benchmarked) {
            printf("\n\nCheck if L1 Data Cache and Texture Cache share the same Cache Hardware physically\n");
            dataTextureShareResult  = measure_L1ShareTexture(L1_results.CacheSize.CacheSize, textureResults.CacheSize.CacheSize, 800);

            if (dataTextureShareResult.first > shareThresholdData || (L1Distance - shareThresholdData) < dataTextureShareResult.first ||
                dataTextureShareResult.second > shareThresholdTxt || (TxtDistance - shareThresholdTxt) < dataTextureShareResult.second) {
#ifdef IsDebug
                printf("L1 Data Cache and Texture Cache share the same physical space\n");
                fprintf(out, "L1 Data Cache and Texture Cache share the same physical space\n");
#endif //IsDebug
                L1ShareTexture = true;
            } else {
#ifdef IsDebug
                printf("L1 Data Cache and Texture Cache have separate physical spaces\n");
                fprintf(out, "L1 Data Cache and Texture Cache have separate physical spaces\n");
#endif //IsDebug
                L1ShareTexture = false;
            }
        }
    }

    if (ReadOnlyResults.benchmarked && textureResults.benchmarked) {
        printf("\n\nCheck if Read Only Cache and Texture Cache share the same Cache Hardware physically\n");

        roTextureShareResult = measure_ROShareTexture(ReadOnlyResults.CacheSize.CacheSize,
                                                      textureResults.CacheSize.CacheSize, 800);
        if (roTextureShareResult.first > shareThresholdRo || (RODistance - shareThresholdRo) < roTextureShareResult.first ||
            roTextureShareResult.second > shareThresholdTxt || (TxtDistance - shareThresholdTxt) < roTextureShareResult.second) {
#ifdef IsDebug
            printf("Read-Only Cache and Texture Cache share the same physical space\n");
            fprintf(out, "Read-Only Cache and Texture Cache share the same physical space\n");
#endif //IsDebug
            ROShareTexture = true;
        } else {
#ifdef IsDebug
            printf("Read-Only Cache and Texture Cache have separate physical spaces\n");
            fprintf(out, "Read-Only Cache and Texture Cache have separate physical spaces\n");
#endif //IsDebug
            ROShareTexture = false;
        }

        cudaDeviceReset();
    }

#ifdef IsDebug
    fprintf(out, "Print Result of Share Checks:\n");
    fprintf(out, "ConstantShareData: constDistance = %f, dataDistance = %f\n", constDataShareResult.first, constDataShareResult.second);
    fprintf(out, "ROShareData: roDistance = %f, dataDistance = %f\n", roDataShareResult.first, roDataShareResult.second);
    fprintf(out, "ROShareTexture: roDistance = %f, textDistance = %f\n", roTextureShareResult.first, roTextureShareResult.second);
    fprintf(out, "DataShareTexture: DataDistance = %f, textDistance = %f\n", dataTextureShareResult.first, dataTextureShareResult.second);
#endif

    // Number of caches per SM checks
    uIntTriple numberOfCachesPerSM = checkNumberOfCachesPerSM(textureResults, ReadOnlyResults, L1_results, cudaInfo, ROShareTexture, ROShareL1Data, L1ShareTexture);
    overallResults[Texture].numberPerSM = numberOfCachesPerSM.first;
    overallResults[RO].numberPerSM = numberOfCachesPerSM.second;
    overallResults[L1].numberPerSM = numberOfCachesPerSM.third;

    // Main memory Checks
    CacheResults mainResult = executeMainMemoryChecks(cudaInfo.L2CacheSize);
    overallResults[MAIN] = mainResult;

    // Shared memory Checks
    CacheResults sharedResult = executeSharedMemoryChecks();
    overallResults[SHARED] = sharedResult;

    // Add general CudaDeviceInfo information
    fillWithCUDAInfo(cudaInfo, totalMem);

    printOverallBenchmarkResults(overallResults, L1_used_for_global_loads, cudaInfo);
    return 0;
}

#endif //CAPTURE
