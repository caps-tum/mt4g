//
// Created by nick- on 9/7/2022.
//

#ifndef CUDATEST_CONST15LATENCY_CPP
#define CUDATEST_CONST15LATENCY_CPP

#include "GPU_resources.h"

#ifndef __has_include
static_assert(false, "__has_include not supported");
#else
#  if __cplusplus >= 201703L && __has_include(<filesystem>)
#    include <filesystem>
namespace fs = std::filesystem;
#  elif __has_include(<experimental/filesystem>)
#    include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#  elif __has_include(<boost/filesystem.hpp>)
#    include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#  endif
#endif

// Call external helper exe for Constant L1.5 Latency due to Constant Memory limit of 64 KiB
LatencyTuple getC15Latency(int deviceID) {
    unsigned int cycles = 1, nsecs = 1;
    LatencyTuple result;
    char cmd[1024];
#ifdef IsDebug
    fprintf(out, "Executing helper c15\n");
#endif //IsDebug

#ifdef _WIN32
    //TODO test this
    std::cout << "Executing c15.exe - if it does not exist in the current directory, it may crash. " << std::endl;
    snprintf(cmd, 1024, "c15.exe -d:%d", deviceID);
#else
    fs::path path = fs::canonical("/proc/self/exe").parent_path();
    fs::path file("c15");
    path = path / file;
    if (fs::exists(path)) {
        // TODO investigate wtf
        std::cerr << "LINUX: Calling support tool at: " << path << std::endl;
        snprintf(cmd, 1024, "%s -d:%d", path.u8string().c_str(), deviceID);
    } else
        std::cerr << "ERROR: path " << path << "does not exist. Skipping..." << std::endl;
#endif

    FILE *p;
#ifdef _WIN32
    p = _popen(cmd, "r");
#else
    p = popen(cmd, "r");
#endif

    if (p == nullptr) {
        std::cerr << "Could not execute command: " << cmd << std::endl;
    }

    char line[MAX_LINE_LENGTH] = {0};

    fgets(line, MAX_LINE_LENGTH, p);
#ifdef IsDebug
    fprintf(out, "Cycles: %s\n", line);
#endif //IsDebug
    char *ptr = strchr(line, ':');
    ptr = ptr + 1;
    cycles = cvtCharArrToInt(ptr);

    fgets(line, MAX_LINE_LENGTH, p);
#ifdef IsDebug
    fprintf(out, "Nanoseconds: %s\n", line);
#endif //IsDebug
    ptr = strchr(line, ':');
    ptr = ptr + 1;
    nsecs = cvtCharArrToInt(ptr);

#ifdef _WIN32
    _pclose(p);
#else
    pclose(p);
#endif

    // These lines take data from c15(.exe) and pass them to result struct
    result.latencyCycles = cycles;
    result.latencyNano = nsecs;
    return result;
}

#endif //CUDATEST_CONST15LATENCY_CPP
