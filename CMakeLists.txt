cmake_minimum_required(VERSION 3.22)
project(MemTop CUDA)
enable_language(CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)

# see: https://dewiki.de/Lexikon/CUDA#Unterstützte_GPUs
# TODO: automate this: https://stackoverflow.com/questions/68223398/how-can-i-get-cmake-to-automatically-detect-the-value-for-cuda-architectures
# Kepler
#set(COMPUTE_CAP_COMPILE_OPTIONS "PRIVATE" "$<$<COMPILE_LANGUAGE:CUDA>:-gencode" "arch=compute_35,code=sm_35>")
# Maxwell
#set(COMPUTE_CAP_COMPILE_OPTIONS "PRIVATE" "$<$<COMPILE_LANGUAGE:CUDA>:-gencode" "arch=compute_50,code=sm_50>")
# Pascal
#set(COMPUTE_CAP_COMPILE_OPTIONS "PRIVATE" "$<$<COMPILE_LANGUAGE:CUDA>:-gencode" "arch=compute_60,code=sm_60>")
# Volta
#set(COMPUTE_CAP_COMPILE_OPTIONS "PRIVATE" "$<$<COMPILE_LANGUAGE:CUDA>:-gencode" "arch=compute_70,code=sm_70>")
# Turing
#set(COMPUTE_CAP_COMPILE_OPTIONS "PRIVATE" "$<$<COMPILE_LANGUAGE:CUDA>:-gencode" "arch=compute_75,code=sm_75>")
# Ampere
#set(COMPUTE_CAP_COMPILE_OPTIONS "PRIVATE" "$<$<COMPILE_LANGUAGE:CUDA>:-gencode" "arch=compute_86,code=sm_86>")

#TODO: IF(WIN32)....
set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(CUDA)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(CUDASAMPLES/Common)
link_directories(${CMAKE_CURRENT_BINARY_DIR})


add_executable(c15 starter_other/ConstMemory.cu)
add_executable(MemTop capture.cu)
#add_executable(MemTop nvmlTester.cu)

set_property(TARGET MemTop PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(MemTop PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(MemTop PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set_property(TARGET c15 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(c15 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(c15 PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_link_options("-lcuda")
add_link_options("-lnvidia-ml")

message(${CUDA_TOOLKIT_ROOT_DIR})

IF(UNIX)
target_link_libraries( MemTop ${CUDA_LIBRARIES} ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/libnvidia-ml.so "stdc++fs" )
target_link_libraries( c15 ${CUDA_LIBRARIES} "stdc++fs")
ENDIF()

IF(WIN32)
target_link_libraries( MemTop "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib")
target_link_libraries( MemTop "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/nvml.lib")
target_link_libraries( c15 "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib")
ENDIF()

IF(DEFINED COMPUTE_CAP_COMPILE_OPTIONS)
target_compile_options(MemTop ${COMPUTE_CAP_COMPILE_OPTIONS})
target_compile_options(c15 ${COMPUTE_CAP_COMPILE_OPTIONS})
ENDIF()
target_compile_options(MemTop PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
target_compile_options(c15 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)

if ("${IsDebug}" STREQUAL "1")
add_definitions(-DIsDebug)
endif()
