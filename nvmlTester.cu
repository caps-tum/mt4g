//
// Created by nick- on 10/6/2022.
//
// max: hipify doesnt change it

#include <nvml.h>
#include <cstdio>

#ifndef NVML_RT_CALL
#define NVML_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<nvmlReturn_t>( call );                                                               \
        if ( status != NVML_SUCCESS )                                                                                  \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA NVML call \"%s\" in line %d of file %s failed "                                      \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     nvmlErrorString( status ),                                                                        \
                     status );                                                                                         \
    }
#endif  // NVML_RT_CALL

int main() {

    NVML_RT_CALL( nvmlInit() );

    unsigned int numCores = 0;
    nvmlDevice_t dev;
    nvmlDeviceArchitecture_t arch;

    NVML_RT_CALL(nvmlDeviceGetHandleByIndex_v2(0, &dev))

    NVML_RT_CALL(nvmlDeviceGetNumGpuCores(dev, &numCores) )

    NVML_RT_CALL(nvmlDeviceGetArchitecture ( dev, &arch))

    printf("NumCores: %d\nArch: %d", numCores, arch);
    NVML_RT_CALL( nvmlShutdown( ) );

    return -1;
}
