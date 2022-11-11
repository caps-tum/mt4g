cmake_minimum_required(VERSION 3.22)
project(mt4g CUDA)
enable_language(CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA 11 REQUIRED)

string(TOUPPER ${NVIDIA_UARCH} NVIDIA_UARCH)
if(NVIDIA_UARCH STREQUAL "KEPLER")
    set(COMPUTE_CAP_COMPILE_OPTIONS "PRIVATE" "$<$<COMPILE_LANGUAGE:CUDA>:-gencode" "arch=compute_35,code=sm_35>")
elseif(NVIDIA_UARCH STREQUAL "MAXWELL")
    set(COMPUTE_CAP_COMPILE_OPTIONS "PRIVATE" "$<$<COMPILE_LANGUAGE:CUDA>:-gencode" "arch=compute_50,code=sm_50>")
elseif(NVIDIA_UARCH STREQUAL "PASCAL")
    set(COMPUTE_CAP_COMPILE_OPTIONS "PRIVATE" "$<$<COMPILE_LANGUAGE:CUDA>:-gencode" "arch=compute_60,code=sm_60>")
elseif(NVIDIA_UARCH STREQUAL "VOLTA")
    set(COMPUTE_CAP_COMPILE_OPTIONS "PRIVATE" "$<$<COMPILE_LANGUAGE:CUDA>:-gencode" "arch=compute_70,code=sm_70>")
elseif(NVIDIA_UARCH STREQUAL "TURING")
    set(COMPUTE_CAP_COMPILE_OPTIONS "PRIVATE" "$<$<COMPILE_LANGUAGE:CUDA>:-gencode" "arch=compute_75,code=sm_75>")
elseif(NVIDIA_UARCH STREQUAL "AMPERE")
    set(COMPUTE_CAP_COMPILE_OPTIONS "PRIVATE" "$<$<COMPILE_LANGUAGE:CUDA>:-gencode" "arch=compute_86,code=sm_86>")
endif()
# see: https://dewiki.de/Lexikon/CUDA#Unterstützte_GPUs
# TODO: automate this: https://stackoverflow.com/questions/68223398/how-can-i-get-cmake-to-automatically-detect-the-value-for-cuda-architectures

#TODO: IF(WIN32)....
#set(CMAKE_THREAD_LIBS_INIT "-lpthread")
#set(CMAKE_HAVE_THREADS_LIBRARY 1)
#set(CMAKE_USE_WIN32_THREADS_INIT 0)
#set(CMAKE_USE_PTHREADS_INIT 1)
#set(THREADS_PREFER_PTHREAD_FLAG ON)


# Top-level build just includes subdirectories.
add_subdirectory(src)
