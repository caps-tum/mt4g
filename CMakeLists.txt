# initial version: 3.22, was tested also under 3.10
cmake_minimum_required(VERSION 3.22)
project(mt4g)
enable_language(CXX)


####################################################################################
### define paths for root dirs if not predefined ###
# noodles' src: https://github.com/ROCm-Developer-Tools/HIP/issues/1029
#[[if(NOT DEFINED HIP_PATH)
    message("hip path not defined!")
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()

if(NOT DEFINED ROCM_PATH)
    message("rocm path not defined!")
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which ROCM has been installed")
    elseif(DEFINED ENV{HIP_PATH})
        set(ROCM_PATH "$ENV{HIP_PATH}/.." CACHE PATH "Path to which ROCM has been installed")
    else()
        set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to which ROCM has been installed")
    endif()
endif()

if(NOT DEFINED HCC_PATH)
    message("hcc path not defined!")
    if(DEFINED ENV{HCC_PATH})
        set(HCC_PATH $ENV{HCC_PATH} CACHE PATH "Path to which HCC has been installed")
    else()
        set(HCC_PATH "${ROCM_PATH}/hcc" CACHE PATH "Path to which HCC has been installed")
    endif()
    set(HCC_HOME "${HCC_PATH}")
endif()

if(NOT DEFINED HIP_CLANG_PATH)
    message("hip-clang path not defined!")
    if(NOT DEFINED ENV{HIP_CLANG_PATH})
        set(HIP_CLANG_PATH "${ROCM_PATH}/llvm/bin" CACHE PATH "Path to which HIP compatible clang binaries have been installed")
    else()
        set(HIP_CLANG_PATH $ENV{HIP_CLANG_PATH} CACHE PATH "Path to which HIP compatible clang binaries have been installed")
    endif()
endif()]]


# first one connects hip::device / hip::host
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" "${HIP_PATH}/lib/cmake/hip" ${CMAKE_MODULE_PATH})
list(APPEND CMAKE_PREFIX_PATH
    "${HIP_PATH}/cmake"
    "${HIP_PATH}/lib/cmake"
    "${HIP_PATH}/../lib/cmake" # hopefully catches all extra HIP dependencies
)
# message(STATUS "MAIN: cmake module path: ${CMAKE_MODULE_PATH}")
# message(STATUS "MAIN: cmake prefix path: ${CMAKE_PREFIX_PATH}")

find_package(HIP REQUIRED)
# message(STATUS "MAIN HIP libraries: ${HIP_LIBRARIES}")

set(CMAKE_HIP_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_HIP_LINKER   ${HIP_HIPCC_EXECUTABLE})


find_program(NVIDIA_SMI nvidia-smi)
find_program(ROCM_SMI rocm-smi)

if(NVIDIA_SMI)
    message(STATUS "Chosen architecture -> Nvidia")
    # https://stackoverflow.com/questions/40695455/what-utility-binary-can-i-call-to-determine-an-nvidia-gpus-compute-capability

    # Run the nvidia-smi command and extract the compute capability
    # returns "compute_cap/n7.5" (7.5 - depending on GPU)
    execute_process(
            COMMAND nvidia-smi --query-gpu=compute_cap --format=csv
            OUTPUT_VARIABLE NVIDIA_SMI_OUTPUT
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # Via regex takes "7" and "5" from "7.5"
    string(REGEX MATCH "([0-9]+)\\.([0-9]+)" COMPUTE_CAP ${NVIDIA_SMI_OUTPUT})

    # Concats to "sm_75"
    set(GPU_NAME "sm_${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
    
    # different flags
    # Set the HIPCC_FLAGS variable to include the device name
	set(HIPCC_FLAGS "--offload-arch=${GPU_NAME}")

elseif(ROCM_SMI)
    # https://docs.amd.com/bundle/HIP-Programming-Guide-v5.0/page/Transitioning_from_CUDA_to_HIP.html
    message(STATUS "Chosen architecture -> AMD")

    # Run rocminfo -> grep full name of GPU, i.e. "amdgcn-amd-amdhsa--gfx1012:xnack-" -> take "gfx1012:xnack-"
    execute_process(
            # if take "gfx1012:xnack-" -> rocminfo | grep 'Name:[ ]*amdgcn-amd-amdhsa--' | awk -F'--' '{print $2}'
            COMMAND sh -c "rocminfo | grep -m 1 -E gfx[^0]{1}.{2} | awk '{print $2}'"
            OUTPUT_VARIABLE GPU_NAME
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    # different flags
    # Set the HIPCC_FLAGS variable to include the device name
	set(HIPCC_FLAGS "-arch=${GPU_NAME}")
else()
    message(STATUS "Chosen architecture -> N/A")

    # expected behavior -> as in "-march=native"
    set(GPU_NAME "native")
endif()

# Print the name of the current GPU
message(STATUS "Current GPU name: ${GPU_NAME}")

# Print the HIPCC_FLAGS variable to verify it includes the device name
message(STATUS "HIPCC_FLAGS: ${HIPCC_FLAGS}")

# Top-level build just includes subdirectories.
add_subdirectory(src)
