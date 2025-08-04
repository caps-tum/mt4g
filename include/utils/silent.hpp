#pragma once

#include <fstream>
#include <iostream>
#include <unistd.h>
#include <fcntl.h>

namespace util {

/**
 * @brief Temporarily redirect stdout and stderr to /dev/null.
 */
class SilentMode {
    std::ofstream nullStream;
    std::streambuf* oldCout;
    std::streambuf* oldCerr;
    int oldStdoutFd;
    int oldStderrFd;
public:
    SilentMode()
        : nullStream("/dev/null"),
          oldCout(std::cout.rdbuf(nullStream.rdbuf())),
          oldCerr(std::cerr.rdbuf(nullStream.rdbuf())) {
        fflush(stdout);
        fflush(stderr);
        oldStdoutFd = dup(STDOUT_FILENO);
        oldStderrFd = dup(STDERR_FILENO);
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull != -1) {
            dup2(devnull, STDOUT_FILENO);
            dup2(devnull, STDERR_FILENO);
            close(devnull);
        }
    }

    ~SilentMode() {
        fflush(stdout);
        fflush(stderr);
        if (oldStdoutFd != -1) {
            dup2(oldStdoutFd, STDOUT_FILENO);
            close(oldStdoutFd);
        }
        if (oldStderrFd != -1) {
            dup2(oldStderrFd, STDERR_FILENO);
            close(oldStderrFd);
        }
        std::cout.rdbuf(oldCout);
        std::cerr.rdbuf(oldCerr);
    }
};
}
