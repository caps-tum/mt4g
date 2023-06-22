//
// Created by nick- on 6/22/2022.
//
#ifndef CAPTURE
#define CAPTURE

# include "eval.h"
# include "cudaDeviceProperties.h"
# include "L1DataCache.h"
# include "ReadOnlyCache.h"
# include "TextureCache.h"
# include "ConstantCache.h"
# include "L2DataCache.h"
# include "l1_l2_diff.h"
# include "ShareChk/chkConstShareL1Data.h"
# include "ShareChk/chkROShareTexture.h"
# include "ShareChk/chkROShareL1Data.h"
# include "ShareChk/chkL1ShareTexture.h"
# include "ShareChk/chkTwoTexture.h"
# include "ShareChk/chkTwoRO.h"
# include "ShareChk/chkTwoL1.h"
# include "ShareChk/chkTwoC1.h"
# include "MainMemory.h"
# include "SharedMemory.h"
# include "ShareChk/chkAllCore.h"
#include "bandwidth.h"

# include <climits>

// for writing to file
#include <iostream>
#include <fstream>
#include <iomanip>

bool L1ShareConst = false;
bool ROShareL1Data = false;
bool ROShareTexture = false;
bool L1ShareTexture = false;

enum Caches {
    L2, Texture, RO, Const1, Const2, L1, MAIN, SHARED
};

char unitsByte[4][4] = {"B", "KiB", "MiB", "GiB"};
char unitsHz[3][4] = {"KHz", "MHz", "GHz"};

// Platform specific names
#ifdef __HIP_PLATFORM_AMD__
char shared_where[8][4] = {"GPU", "CU", "CU", "CU", "CU", "CU", "GPU", "CU"};
#else
char shared_where[8][4] = {"GPU", "SM", "SM", "SM", "SM", "SM", "GPU", "SM"};
#endif

CacheResults overallResults[8];

// Returns for a given byte size the best fitting unit and its value (KiB, MiB, GiB)
const char *getSizeNiceFormatByte(double *val, size_t original) {
    int unitIndex = 0;  // Initialize the unit index to zero, which will represent the byte unit

    // If the original size is greater than one gigabyte (1024 * 1024 * 1024 bytes), divide it by 1024
    if (original > 1024 * 1024 * 1024) {
        original = original >> 10;
        ++unitIndex;
    }

    double result = (double) original;  // Convert the original size to a double for precision

    // If the result is greater than 1000, divide it by 1024 to represent the next unit (megabyte -> kilobyte)
    // and increment the unit index to represent the next unit (megabyte -> kilobyte)
    if (result > 1000.) {
        result = result / 1024.;
        ++unitIndex;
    }

    // If the result is still greater than 1000, divide it by 1024 to represent the next unit (kilobyte -> byte)
    // and increment the unit index to represent the next unit (kilobyte -> byte)
    if (result > 1000.) {
        result = result / 1024.;
        ++unitIndex;
    }

    const char *unit = unitsByte[unitIndex];  // Get the unit string associated with the unit index
    *val = result;  // Set the value pointer to the final calculated result
    return unit;  // Return the unit string
}

// Returns for a given byte size the best fitting unit and its value (KHz, MHz, GHz)
const char *getSizeNiceFormatHertz(double *val, unsigned int original) {
    int unitIndex = 0;

    if (original > 1000 * 1000 * 1000) {
        original = original / 1000;
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

    const char *unit = unitsHz[unitIndex];
    *val = result;
    return unit;
}

/**
 * Function for printing results to both console and .csv file
 * @param result - measured cache results
 * @param L1_global_load_enabled - flag whether L1 global cache is enabled
 * @param cudaInfo - CUDA/HCC info about GPU
 */
void printOverallBenchmarkResults(CacheResults *result, bool L1_global_load_enabled, CudaDeviceInfo cudaInfo) {
    std::cout << "\n\n**************************************************\n"
                 "\tPRINT GPU BENCHMARK RESULT\n"
                 "**************************************************\n\n";

    // Name of file with results
    std::string outputCSV = "GPU_Memory_Topology.csv";
    std::ofstream csv(outputCSV);

    if (!csv) {
        std::cerr << "[WARNING]: Cannot open file for writing - close csv file if currently open\n";
        csv.copyfmt(std::cout);
    }

    // Predefining some params to increase readability
    // Most data taken from:
    //  https://www.olcf.ornl.gov/wp-content/uploads/2019/10/ORNL_Application_Readiness_Workshop-AMD_GPU_Basics.pdf

    std::string vendorName;
    std::string computeCapability;
    // CU [Compute Unit] - AMD, SM [Streaming Multiprocessor] - Nvidia
    std::string CUorSM;
    // Threadblock - CUDA, Workgroup - HIP
    std::string blockGroup;

    // Setting vendor specific names
#ifdef __HIP_PLATFORM_AMD__
    vendorName = "AMD";
    computeCapability = "HCC";
    CUorSM = "CU";
    blockGroup = "workgroup";
#else
    vendorName = "Nvidia";
    computeCapability = "CUDA";
    CUorSM = "SM";
    blockGroup = "threadblock";
#endif

    std::cout << "GPU name: " << cudaInfo.GPUname << "\n\n";
    csv << "GPU_INFORMATION; GPU_vendor; \"" << vendorName << "\"; GPU_name; \"" << cudaInfo.GPUname << "\"\n";


    // Print compute resource information to stdout and write to CSV
    std::cout << "PRINT COMPUTE RESOURCE INFORMATION:\n";
    csv << "COMPUTE_RESOURCE_INFORMATION; ";

    std::cout << computeCapability << " compute capability: " << cudaInfo.cudaVersion << '\n';
    csv << computeCapability << "_compute_capability; \"" << cudaInfo.cudaVersion << "\"; ";


    // Print number of CUs or SMs
    std::cout << "Number Of " << CUorSM << ": " << cudaInfo.numberOfSMs << "\n";
    csv << "Number_of_" << CUorSM << "; " << cudaInfo.numberOfSMs << "; ";

    // Print number of cores in GPU and cores per CU or SM
    std::cout << "Number Of Cores in GPU: " << cudaInfo.numberOfCores << "\n";
    csv << "Number_of_cores_in_GPU; " << cudaInfo.numberOfCores << "; ";

    std::cout << "Number Of Cores per " << CUorSM << " in GPU: " << cudaInfo.numberOfCores / cudaInfo.numberOfSMs
              << "\n\n";
    csv << "Number_of_cores_per_" << CUorSM << "; " << cudaInfo.numberOfCores / cudaInfo.numberOfSMs << "\n";

    // Print register information
    std::cout << "PRINT REGISTER INFORMATION:\n";
    csv << "REGISTER_INFORMATION; ";

    std::cout << "Registers per " << blockGroup << ": " << cudaInfo.registersPerThreadBlock << " 32-bit registers\n";
    csv << "Registers_per_" << blockGroup << "; " << cudaInfo.registersPerThreadBlock << "; \"32-bit registers\"; ";

    std::cout << "Registers per " << CUorSM << ": " << cudaInfo.registersPerSM << " 32-bit registers\n\n";
    csv << "Registers_per_" << CUorSM << "; " << cudaInfo.registersPerSM << "; \"32-bit registers\"\n";

    // Print additional information
    std::cout << "PRINT ADDITIONAL INFORMATION:\n";
    csv << "ADDITIONAL_INFORMATION; ";

    double val;
    unsigned int originalFrequency = cudaInfo.memClockRate;
    const char *MemClockFreqUnit = getSizeNiceFormatHertz(&val, originalFrequency);
    std::cout << "Memory Clock Frequency: " << std::fixed << std::setprecision(3) << val << ' ' << MemClockFreqUnit
              << '\n';
    csv << "Memory_Clock_Frequency; " << std::fixed << std::setprecision(3) << val << "; \"" << MemClockFreqUnit
        << "\"; ";

    std::cout << "Memory Bus Width: " << cudaInfo.memBusWidth << " bits\n";
    csv << "Memory_Bus_Width; " << cudaInfo.memBusWidth << "; \"bit\"; ";

    originalFrequency = cudaInfo.GPUClockRate;
    const char *GPUClockFreqUnit = getSizeNiceFormatHertz(&val, originalFrequency);
    std::cout << "GPU Clock rate: " << std::fixed << std::setprecision(3) << val << ' ' << GPUClockFreqUnit << "\n\n";
    csv << "GPU_Clock_Rate; " << std::fixed << std::setprecision(3) << val << "; \"" << GPUClockFreqUnit << "\"\n";

    csv << "L1_DATA_CACHE; ";
    if (!L1_global_load_enabled) {
        std::cout << "L1 DATA CACHE INFORMATION missing: GPU does not allow caching of global loads in L1\n";
        csv << "\"N/A\"\n";
    } else {
        if (result[L1].benchmarked) {
            std::cout << "PRINT L1 DATA CACHE INFORMATION:\n";

            if (result[L1].CacheSize.realCP) {
                double size;
                size_t original = result[L1].CacheSize.CacheSize;
                const char *unit = getSizeNiceFormatByte(&size, original);
                std::cout << "Detected L1 Data Cache Size: " << size << " " << unit << "\n";
                csv << "Size; " << size << "; " << unit << "; \"" << '=' << "\"; ";
            } else {
                double size;
                size_t original = result[L1].CacheSize.maxSizeBenchmarked;
                const char *unit = getSizeNiceFormatByte(&size, original);
                std::cout << "Benchmarked L1 Data Cache Size: >= " << size << " " << unit << "\n";
                csv << "Size; " << size << "; " << unit << "; \"" << ">=" << "\"; ";
            }

            std::cout << "Detected L1 Data Cache Line Size: " << result[L1].cacheLineSize << " B\n";
            csv << "Cache_Line_Size; " << result[L1].cacheLineSize << "; \"" << "B" << "\"; ";
            std::cout << "Detected L1 Data Cache Load Latency: " << result[L1].latencyCycles << " cycles\n";
            csv << "Load_Latency; " << result[L1].latencyCycles << "; \"" << "cycles" << "\"; ";
            std::cout << "Detected L1 Data Cache Load Latency: " << result[L1].latencyNano << " nanoseconds\n";
            csv << "Load_Latency; " << result[L1].latencyNano << "; \"" << "nanoseconds" << "\"; ";
            std::cout << "L1 Data Cache Is Shared On " << shared_where[L1] << "-level\n";
            csv << "Shared_On; \"" << shared_where[L1] << "-level\"; ";
            std::cout << "Does L1 Data Cache Share the physical cache with the Texture Cache? "
                      << (L1ShareTexture ? "Yes" : "No") << "\n";
            csv << "Share_Cache_With_Texture; " << L1ShareTexture << "; ";
            std::cout << "Does L1 Data Cache Share the physical cache with the Read-Only Cache? "
                      << (ROShareL1Data ? "Yes" : "No") << "\n";
            csv << "Share_Cache_With_Read-Only; " << ROShareL1Data << "; ";
            std::cout << "Does L1 Data Cache Share the physical cache with the Constant L1 Cache? "
                      << (L1ShareConst ? "Yes" : "No") << "\n";
            csv << "Share_Cache_With_ConstantL1; " << L1ShareConst << "; ";
            std::cout << "Detected L1 Data Caches Per SM: " << result[L1].numberPerSM << "\n";
            csv << "Caches_Per_SM; " << result[L1].numberPerSM << "; ";

            if (result[L1].bw_tested) {
                std::cout << "Detected L1 Data Cache Bandwidth: " << result[L1].bandwidth << " GB/s\n";
                csv << "Bandwidth; " << result[L1].bandwidth << "; \"GB/s\"\n";
            }
        } else {
            std::cout << "L1 Data CACHE WAS NOT BENCHMARKED!\n\n";
            csv << "\"N/A\"\n";
        }
    }

    csv << "L2_DATA_CACHE; ";
    if (result[L2].benchmarked) {
        std::cout << "PRINT L2 CACHE INFORMATION:\n";
        if (result[L2].CacheSize.realCP) {
            double size;
            size_t original = result[L2].CacheSize.CacheSize;
            const char *unit = getSizeNiceFormatByte(&size, original);
            std::cout << "Detected L2 Cache Size: " << std::fixed << std::setprecision(3) << size << " " << unit
                      << std::endl;
            csv << "Size; " << std::fixed << std::setprecision(3) << size << "; " << unit << "; \"=\"; ";
        } else {
            double size;
            size_t original = result[L2].CacheSize.maxSizeBenchmarked;
            const char *unit = getSizeNiceFormatByte(&size, original);
            std::cout << "Detected L2 Cache Size: >= " << std::fixed << std::setprecision(3) << size << " " << unit
                      << std::endl;
            csv << "Size; " << std::fixed << std::setprecision(3) << size << "; " << unit << "; \">=\"; ";
        }
        std::cout << "Detected L2 Cache Line Size: " << result[L2].cacheLineSize << " B" << std::endl;
        csv << "Cache_Line_Size; " << result[L2].cacheLineSize << "; \"B\"; ";
        std::cout << "Detected L2 Cache Load Latency: " << result[L2].latencyCycles << " cycles" << std::endl;
        csv << "Load_Latency; " << result[L2].latencyCycles << "; \"cycles\"; ";
        std::cout << "Detected L2 Cache Load Latency: " << result[L2].latencyNano << " nanoseconds" << std::endl;
        csv << "Load_Latency; " << result[L2].latencyNano << "; \"nanoseconds\"; ";
        std::cout << "L2 Cache Is Shared On " << shared_where[L2] << "-level\n";
        csv << "Shared_On; \"" << shared_where[L2] << "-level\"; ";

        if (result[L2].bw_tested) {
            std::cout << "Detected L2 Cache Bandwidth: " << result[L2].bandwidth << " GB/s\n\n";
            csv << "Bandwidth; " << result[L2].bandwidth << "; \"GB/s\"\n";
        }
    } else {
        std::cout << "L2 CACHE WAS NOT BENCHMARKED!\n\n";
        csv << "\"N/A\"\n";
    }

    csv << "TEXTURE_CACHE; ";
    if (result[Texture].benchmarked) {
        std::cout << "PRINT TEXTURE CACHE INFORMATION:\n";
        if (result[Texture].CacheSize.realCP) {
            double size;
            size_t original = result[Texture].CacheSize.CacheSize;
            const char *unit = getSizeNiceFormatByte(&size, original);
            std::cout << "Detected Texture Cache Size: " << size << ' ' << unit << '\n';
            csv << "Size; " << size << "; " << unit << "; \"=\"; ";
        } else {
            double size;
            size_t original = result[Texture].CacheSize.maxSizeBenchmarked;
            const char *unit = getSizeNiceFormatByte(&size, original);
            std::cout << "Detected Texture Cache Size: >= " << size << ' ' << unit << '\n';
            csv << "Size; " << size << "; " << unit << "; \">=\"; ";
        }
        std::cout << "Detected Texture Cache Line Size: " << result[Texture].cacheLineSize << " B\n";
        csv << "Cache_Line_Size; " << result[Texture].cacheLineSize << "; \"B\"; ";
        std::cout << "Detected Texture Cache Load Latency: " << result[Texture].latencyCycles << " cycles\n";
        csv << "Load_Latency; " << result[Texture].latencyCycles << "; \"cycles\"; ";
        std::cout << "Detected Texture Cache Load Latency: " << result[Texture].latencyNano << " nanoseconds\n";
        csv << "Load_Latency; " << result[Texture].latencyNano << "; \"nanoseconds\"; ";
        std::cout << "Texture Cache Is Shared On " << shared_where[Texture] << "-level\n";
        csv << "Shared_On; \"" << shared_where[Texture] << "-level\"; ";
        std::cout << "Does Texture Cache Share the physical cache with the L1 Data Cache? "
                  << (L1ShareTexture ? "Yes" : "No") << '\n';
        csv << "Share_Cache_With_L1_Data; " << L1ShareTexture << "; ";
        std::cout << "Does Texture Cache Share the physical cache with the Read-Only Cache? "
                  << (ROShareTexture ? "Yes" : "No") << '\n';
        csv << "Share_Cache_With_Read-Only; " << ROShareTexture << "; ";
        std::cout << "Detected Texture Caches Per SM: " << result[Texture].numberPerSM << "\n\n";
        csv << "Caches_Per_SM; " << result[Texture].numberPerSM << '\n';
    } else {
        std::cout << "TEXTURE CACHE WAS NOT BENCHMARKED!\n\n";
        csv << "\"N/A\"\n";
    }

    csv << "READ-ONLY_CACHE; ";
    if (result[RO].benchmarked) {
        std::cout << "PRINT Read-Only CACHE INFORMATION:\n";
        if (result[RO].CacheSize.realCP) {
            double size;
            size_t original = result[RO].CacheSize.CacheSize;
            const char *unit = getSizeNiceFormatByte(&size, original);
            std::cout << "Detected Read-Only Cache Size: " << size << " " << unit << std::endl;
            csv << "Size; " << size << "; " << unit << "; \"=\"; ";
        } else {
            double size;
            size_t original = result[RO].CacheSize.maxSizeBenchmarked;
            const char *unit = getSizeNiceFormatByte(&size, original);
            std::cout << "Detected Read-Only Cache Size: >= " << size << " " << unit << std::endl;
            csv << "Size; " << size << "; " << unit << "; \">=\"; ";
        }
        std::cout << "Detected Read-Only Cache Line Size: " << result[RO].cacheLineSize << " B" << std::endl;
        csv << "Cache_Line_Size; " << result[RO].cacheLineSize << "; \"B\"; ";
        std::cout << "Detected Read-Only Cache Load Latency: " << result[RO].latencyCycles << " cycles" << std::endl;
        csv << "Load_Latency; " << result[RO].latencyCycles << "; \"cycles\"; ";
        std::cout << "Detected Read-Only Cache Load Latency: " << result[RO].latencyNano << " nanoseconds" << std::endl;
        csv << "Load_Latency; " << result[RO].latencyNano << "; \"nanoseconds\"; ";
        std::cout << "Read-Only Cache Is Shared On " << shared_where[RO] << "-level" << std::endl;
        csv << "Shared_On; \"" << shared_where[RO] << "-level\"; ";
        std::cout << "Does Read-Only Cache Share the physical cache with the L1 Data Cache? "
                  << (ROShareL1Data ? "Yes" : "No") << std::endl;
        csv << "Share_Cache_With_L1_Data; " << ROShareL1Data << "; ";
        std::cout << "Does Read-Only Cache Share the physical cache with the Texture Cache? "
                  << (ROShareTexture ? "Yes" : "No") << std::endl;
        csv << "Share_Cache_With_Texture; " << ROShareTexture << "; ";
        std::cout << "Detected Read-Only Caches Per SM: " << result[RO].numberPerSM << std::endl << std::endl;
        csv << "Caches_Per_SM; " << result[RO].numberPerSM << std::endl;
    } else {
        std::cout << "READ-ONLY CACHE WAS NOT BENCHMARKED!" << std::endl << std::endl;
        csv << "\"N/A\"\n";
    }

    csv << "CONSTANT_L1_CACHE; ";
    if (result[Const1].benchmarked) {
        std::cout << "PRINT CONSTANT CACHE L1 INFORMATION:\n";
        if (result[Const1].CacheSize.realCP) {
            double size;
            size_t original = result[Const1].CacheSize.CacheSize;
            const char *unit = getSizeNiceFormatByte(&size, original);
            std::cout << "Detected Constant L1 Cache Size: " << size << " " << unit << std::endl;
            csv << "Size; " << size << "; " << unit << "; \"=\"; ";
        } else {
            double size;
            size_t original = result[Const1].CacheSize.maxSizeBenchmarked;
            const char *unit = getSizeNiceFormatByte(&size, original);
            std::cout << "Detected Constant L1 Cache Size: >= " << size << " " << unit << std::endl;
            csv << "Size; " << size << "; " << unit << "; \">=\"; ";
        }
        std::cout << "Detected Constant L1 Cache Line Size: " << result[Const1].cacheLineSize << " B" << std::endl;
        csv << "Cache_Line_Size; " << result[Const1].cacheLineSize << "; \"B\"; ";
        std::cout << "Detected Constant L1 Cache Load Latency: " << result[Const1].latencyCycles << " cycles"
                  << std::endl;
        csv << "Load_Latency; " << result[Const1].latencyCycles << "; \"cycles\"; ";
        std::cout << "Detected Constant L1 Cache Load Latency: " << result[Const1].latencyNano << " nanoseconds"
                  << std::endl;
        csv << "Load_Latency; " << result[Const1].latencyNano << "; \"nanoseconds\"; ";
        std::cout << "Constant L1 Cache Is Shared On " << shared_where[Const1] << "-level" << std::endl;
        csv << "Shared_On; \"" << shared_where[Const1] << "-level\"; ";
        std::cout << "Does Constant L1 Cache Share the physical cache with the L1 Data Cache? "
                  << (L1ShareConst ? "Yes" : "No") << std::endl << std::endl;
        csv << "Share_Cache_With_L1_Data; " << L1ShareConst << "; ";
        std::cout << "Detected Constant L1 Caches Per SM: " << result[Const1].numberPerSM << std::endl << std::endl;
        csv << "Caches_Per_SM; " << result[Const1].numberPerSM << std::endl;
    } else {
        std::cout << "CONSTANT CACHE L1 WAS NOT BENCHMARKED!\n";
        csv << "\"N/A\"\n";
    }


    csv << "CONST_L1_5_CACHE; ";
    if (result[Const2].benchmarked) {
        std::cout << "PRINT CONSTANT L1.5 CACHE INFORMATION:\n";
        if (result[Const2].CacheSize.realCP) {
            double size;
            size_t original = result[Const2].CacheSize.CacheSize;
            const char *unit = getSizeNiceFormatByte(&size, original);
            std::cout << "Detected Constant L1.5 Cache Size: " << size << " " << unit << "\n";
            csv << "Size; " << size << "; " << unit << "; \"" << '=' << "\"; ";
        } else {
            double size;
            size_t original = result[Const2].CacheSize.maxSizeBenchmarked;
            const char *unit = getSizeNiceFormatByte(&size, original);
            std::cout << "Detected Constant L1.5 Cache Size: >= " << size << " " << unit << "\n";
            csv << "Size; " << size << "; " << unit << "; \"" << ">=" << "\"; ";
        }
        std::cout << "Detected Constant L1.5 Cache Line Size: " << result[Const2].cacheLineSize << " B\n";
        csv << "Cache_Line_Size; " << result[Const2].cacheLineSize << "; \"B\"; ";
        std::cout << "Detected Constant L1.5 Cache Load Latency: " << result[Const2].latencyCycles << " cycles\n";
        csv << "Load_Latency; " << result[Const2].latencyCycles << "; \"cycles\"; ";
        std::cout << "Detected Constant L1.5 Cache Load Latency: " << result[Const2].latencyNano << " nanoseconds\n";
        csv << "Load_Latency; " << result[Const2].latencyNano << "; \"nanoseconds\"; ";
        std::cout << "Const L1.5 Cache Is Shared On " << shared_where[Const2] << "-level\n\n";
        csv << "Shared_On; \"" << shared_where[Const2] << "-level\"\n";
    } else {
        std::cout << "CONSTANT CACHE L1.5 WAS NOT BENCHMARKED!\n";
        csv << "\"N/A\"\n";
    }

    csv << "MAIN_MEMORY; ";
    if (result[MAIN].benchmarked) {
        std::cout << "PRINT MAIN MEMORY INFORMATION:\n";
        if (result[MAIN].CacheSize.realCP) {
            double size;
            size_t original = result[MAIN].CacheSize.CacheSize;
            const char *unit = getSizeNiceFormatByte(&size, original);
            std::cout << "Detected Main Memory Size: " << size << " " << unit << std::endl;
            csv << "Size; " << size << "; " << unit << "; \"" << '=' << "\"; ";
        } else {
            double size;
            size_t original = result[MAIN].CacheSize.maxSizeBenchmarked;
            const char *unit = getSizeNiceFormatByte(&size, original);
            std::cout << "Detected Main Memory Size: >= " << size << " " << unit << std::endl;
            csv << "Size; " << size << "; " << unit << "; \"" << ">=" << "\"; ";
        }
        std::cout << "Detected Main Memory Load Latency: " << result[MAIN].latencyCycles << " cycles" << std::endl;
        csv << "Load_Latency; " << result[MAIN].latencyCycles << "; \"cycles\"; ";
        std::cout << "Detected Main Memory Load Latency: " << result[MAIN].latencyNano << " nanoseconds" << std::endl;
        csv << "Load_Latency; " << result[MAIN].latencyNano << "; \"nanoseconds\"; ";
        std::cout << "Main Memory Is Shared On " << shared_where[MAIN] << "-level\n";
        csv << "Shared_On; \"" << shared_where[MAIN] << "-level\"; ";
        if (result[MAIN].bw_tested) {
            std::cout << "Detected Main Memory Bandwidth: " << result[MAIN].bandwidth << " GB/s\n\n";
            csv << "Bandwidth; " << result[MAIN].bandwidth << "; \"" << "GB/s" << "\"\n";
        }
    } else {
        std::cout << "MAIN MEMORY WAS NOT BENCHMARKED!\n";
        csv << "\"N/A\"\n";
    }

    csv << "SHARED_MEMORY; ";
    if (result[SHARED].benchmarked) {
        std::cout << "PRINT SHARED MEMORY INFORMATION:\n";
        if (result[SHARED].CacheSize.realCP) {
            double size;
            size_t original = result[SHARED].CacheSize.CacheSize;
            const char *unit = getSizeNiceFormatByte(&size, original);
            std::cout << "Detected Shared Memory Size: " << size << " " << unit << std::endl;
            csv << "Size; " << size << "; " << unit << "; \"" << '=' << "\"; ";
        } else {
            double size;
            size_t original = result[SHARED].CacheSize.maxSizeBenchmarked;
            const char *unit = getSizeNiceFormatByte(&size, original);
            std::cout << "Detected Shared Memory Size: >= " << size << " " << unit << std::endl;
            csv << "Size; " << size << "; " << unit << "; \"" << ">=" << "\"; ";
        }
        std::cout << "Detected Shared Memory Load Latency: " << result[SHARED].latencyCycles << " cycles" << std::endl;
        csv << "Load_Latency; " << result[SHARED].latencyCycles << "; \"cycles\"; ";
        std::cout << "Detected Shared Memory Load Latency: " << result[SHARED].latencyNano << " nanoseconds"
                  << std::endl;
        csv << "Load_Latency; " << result[SHARED].latencyNano << "; \"nanoseconds\"; ";
        std::cout << "Shared Memory Is Shared On " << shared_where[SHARED] << "-level" << std::endl << std::endl;
        csv << "Shared_On; \"" << shared_where[SHARED] << "-level\"\n";
    } else {
        std::cout << "SHARED MEMORY WAS NOT BENCHMARKED!\n";
        csv << "\"N/A\"\n";
    }

    csv.close();
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
//char cudaCoreQueryPath[coreQuerySize];
std::string cudaCoreQueryPath = "nvidia-settings -q CUDACores -t";

void parseArgs(int argc, char *argv[]) {
#ifdef _WIN32
    printf("Usage: MemTop.exe [OPTIONS]\n"
#else
    printf("Usage: ./mt4g [OPTIONS]\n"
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
        char *arg = argv[i];
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
            std::cout << "nvidia-settings is only available on linux platforms\n";
#else
            coreSwitch = 1;
            cudaCoreQueryPath = "nvidia-settings -q CUDACores -t";
#endif

        } else if (strstr(arg, "-p:") != nullptr) {
            coreSwitch = 2;
            std::string path = arg + strlen("-p:");

#ifdef _WIN32
            if (path.length() + strlen("/deviceQuery.exe") > coreQuerySize) {
                std::cout << "Path to 'deviceQuery' is too long (> " << coreQuerySize << ")\n";
                coreSwitch = 0;
            } else {
                cudaCoreQueryPath = "\"" + path + separator() + "deviceQuery.exe\"";
            }
#else
            if (path.length() + strlen("/deviceQuery") > coreQuerySize) {
                std::cout << "Path to 'deviceQuery' is too long (> " << coreQuerySize << ")\n";
                coreSwitch = 0;
            } else {
                cudaCoreQueryPath = "\"" + path + separator() + "deviceQuery\"";
            }
#endif
        } else if (strstr(arg, "-d:") != nullptr) {
            char *id = &arg[strlen("-d:")];
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

int main(int argc, char *argv[]) {
    parseArgs(argc, argv);

    cleanupOutput();
    hipError_t result;

    // Use first device (in case of multi-GPU machine)
    int numberGPUs;
    result = hipGetDeviceCount(&numberGPUs);
    if (deviceID >= numberGPUs) {
        printf("Specified device ID %d >= %d(number of installed GPUs) - will use default GPU 0!\n"
               "Use \'nvidia-smi\' to see the ID of the desired GPU device!\n", deviceID, numberGPUs);
        deviceID = 0;
    }

    result = hipSetDevice(deviceID);
    if (result != hipSuccess) {
        std::cout << "Capture/610\tError setting device: " << hipGetErrorString(result) << std::endl;
        std::cerr << "Aborting.." << std::endl;
        return -1;
    }

#ifdef IsDebug
    out = fopen("GPUlog.log", "w");
#endif //IsDebug

    size_t freeMem, totalMem;
    result = hipMemGetInfo(&freeMem, &totalMem);

    CudaDeviceInfo cudaInfo = getDeviceProperties(cudaCoreQueryPath, coreSwitch, deviceID);

    CacheResults L1_results;
    CacheResults L2_results;
    CacheResults textureResults;
    CacheResults ReadOnlyResults;

    printf("\n\nMeasure if L1 is used for caching global loads\n");
    // TODO tolerance
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

        // TODO call
        std::pair<float, float> bw_th_results = executeBandwidthThroughputChecks("L2"); // throughput not implemented
        overallResults[L2].bandwidth = bw_th_results.first;
        overallResults[L2].bw_tested = true;

        result = hipDeviceReset();
        if (result != hipSuccess) {
            std::cout << "Capture/641\tError resetting device: " << hipGetErrorString(result) << std::endl;
        }
    }

    if (L1_used_for_global_loads && l1) {
        // L1 Data Cache Checks
        L1_results = executeL1DataCacheChecks();
#ifdef IsDebug
        fprintf(out, "Detected L1 Cache Size: %f KiB\n", (double)L1_results.CacheSize.CacheSize / 1024.);
        fprintf(out, "Detected L1 Latency in cycles: %d\n", L1_results.latencyCycles);
#endif //IsDebug
        overallResults[L1] = L1_results;

        // TODO call
        std::pair<float, float> bw_th_results = executeBandwidthThroughputChecks("L1"); // throughput not implemented
        overallResults[L1].bandwidth = bw_th_results.first;
        overallResults[L1].bw_tested = true;

        result = hipDeviceReset();
        if (result != hipSuccess) {
            std::cout << "Capture/653\tError resetting device: " << hipGetErrorString(result) << std::endl;
        }
    }

    if (txt) {
        // Texture Cache Checks
        textureResults = executeTextureCacheChecks();
#ifdef IsDebug
        fprintf(out, "Detected Texture Cache Size: %f KiB\n", (double) textureResults.CacheSize.CacheSize / 1024.);
#endif //IsDebug
        overallResults[Texture] = textureResults;

        result = hipDeviceReset();
        if (result != hipSuccess) {
            std::cout << "Capture/664\tError resetting device: " << hipGetErrorString(result) << std::endl;
        }
    }

    if (ro) {
        // Read Only Cache Checks
        ReadOnlyResults = executeReadOnlyCacheChecks();
#ifdef IsDebug
        fprintf(out, "Detected Read-Only Cache Size: %f KiB\n", (double) ReadOnlyResults.CacheSize.CacheSize / 1024.);
#endif //IsDebug
        overallResults[RO] = ReadOnlyResults;

        result = hipDeviceReset();
        if (result != hipSuccess) {
            std::cout << "Capture/675\tError resetting device: " << hipGetErrorString(result) << std::endl;
        }
    }

    if (constant) {
        // Constant Cache L1 & L1.5 Checks
        Tuple<CacheResults> results = executeConstantCacheChecks(deviceID);
        overallResults[Const1] = results.first;
        overallResults[Const2] = results.second;
        result = hipDeviceReset();
        if (result != hipSuccess) {
            std::cout << "Capture/683\tError resetting device: " << hipGetErrorString(result) << std::endl;
        }
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
        C1Distance = (double) measure_TwoC1(overallResults[Const1].CacheSize.CacheSize, 10);
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

        // TODO takes a lot of time
        if (overallResults[Const1].benchmarked) {
            printf("\n\nCheck if Const L1 Cache and Data L1 Cache share the same Cache Hardware physically\n");
            constDataShareResult = measure_ConstShareData(overallResults[Const1].CacheSize.CacheSize,
                                                          L1_results.CacheSize.CacheSize, 800);
            if ((C1Distance - shareThresholdConst) < constDataShareResult.first ||
                constDataShareResult.second > shareThresholdData ||
                (L1Distance - shareThresholdData) < constDataShareResult.second) {
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

            result = hipDeviceReset();
            if (result != hipSuccess) {
                std::cout << "Capture/751\tError resetting device: " << hipGetErrorString(result) << std::endl;
            }
        }

        if (ReadOnlyResults.benchmarked) {
            printf("\n\nCheck if Read Only Cache and L1 Data Cache share the same Cache Hardware physically\n");
            roDataShareResult = measure_ROShareL1Data(ReadOnlyResults.CacheSize.CacheSize,
                                                      L1_results.CacheSize.CacheSize, 800);
            if (roDataShareResult.first > shareThresholdRo ||
                (RODistance - shareThresholdRo) < roDataShareResult.first ||
                roDataShareResult.second > shareThresholdData ||
                (L1Distance - shareThresholdData) < roDataShareResult.second) {
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

            result = hipDeviceReset();
            if (result != hipSuccess) {
                std::cout << "Capture/775\tError resetting device: " << hipGetErrorString(result) << std::endl;
            }
        }

        if (textureResults.benchmarked) {
            printf("\n\nCheck if L1 Data Cache and Texture Cache share the same Cache Hardware physically\n");
            dataTextureShareResult = measure_L1ShareTexture(L1_results.CacheSize.CacheSize,
                                                            textureResults.CacheSize.CacheSize, 800);

            if (dataTextureShareResult.first > shareThresholdData ||
                (L1Distance - shareThresholdData) < dataTextureShareResult.first ||
                dataTextureShareResult.second > shareThresholdTxt ||
                (TxtDistance - shareThresholdTxt) < dataTextureShareResult.second) {
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
        if (roTextureShareResult.first > shareThresholdRo ||
            (RODistance - shareThresholdRo) < roTextureShareResult.first ||
            roTextureShareResult.second > shareThresholdTxt ||
            (TxtDistance - shareThresholdTxt) < roTextureShareResult.second) {
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

        result = hipDeviceReset();
        if (result != hipSuccess) {
            std::cout << "Capture/845\tError resetting device: " << hipGetErrorString(result) << std::endl;
        }
    }

#ifdef IsDebug
    fprintf(out, "Print Result of Share Checks:\n");
    fprintf(out, "ConstantShareData: constDistance = %f, dataDistance = %f\n", constDataShareResult.first, constDataShareResult.second);
    fprintf(out, "ROShareData: roDistance = %f, dataDistance = %f\n", roDataShareResult.first, roDataShareResult.second);
    fprintf(out, "ROShareTexture: roDistance = %f, textDistance = %f\n", roTextureShareResult.first, roTextureShareResult.second);
    fprintf(out, "DataShareTexture: DataDistance = %f, textDistance = %f\n", dataTextureShareResult.first, dataTextureShareResult.second);
#endif

    // Number of caches per SM checks
    uIntTriple numberOfCachesPerSM = checkNumberOfCachesPerSM(textureResults, ReadOnlyResults, L1_results, cudaInfo,
                                                              ROShareTexture, ROShareL1Data, L1ShareTexture);
    overallResults[Texture].numberPerSM = numberOfCachesPerSM.first;
    overallResults[RO].numberPerSM = numberOfCachesPerSM.second;
    overallResults[L1].numberPerSM = numberOfCachesPerSM.third;

    // Main memory Checks
    CacheResults mainResult = executeMainMemoryChecks(cudaInfo.L2CacheSize);
    overallResults[MAIN] = mainResult;

    // Main memory Bandwidth-ONLY Checks
    std::pair<float, float> bw_th_results = executeBandwidthThroughputChecks("MAIN"); // throughput is not implemented
    overallResults[MAIN].bandwidth = bw_th_results.first;
    overallResults[MAIN].bw_tested = true;

    // Shared memory Checks
    CacheResults sharedResult = executeSharedMemoryChecks();
    overallResults[SHARED] = sharedResult;

    // Add general CudaDeviceInfo information
    fillWithCUDAInfo(cudaInfo, totalMem);

    printOverallBenchmarkResults(overallResults, L1_used_for_global_loads, cudaInfo);
    return 0;
}

#endif //CAPTURE
