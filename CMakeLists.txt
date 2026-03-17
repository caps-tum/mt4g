cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_COMPILER hipcc)

project(mt4g VERSION 1.2.0 LANGUAGES CXX)

set(GPU_TARGET_ARCH "" CACHE STRING "The target GPU architecture")
if(GPU_TARGET_ARCH STREQUAL "")
    message(FATAL_ERROR "Please set GPU_TARGET_ARCH, e.g. -DGPU_TARGET_ARCH=sm_70 or -DGPU_TARGET_ARCH=gfx90a")
endif()
if(NOT DEFINED ENV{HIP_PATH})
    message(FATAL_ERROR "HIP_PATH is not set in environment")
endif()

find_package(nlohmann_json 3.11.2 REQUIRED)
find_package(cxxopts 3.2.0 REQUIRED)

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS ${CMAKE_SOURCE_DIR}/.git)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --dirty --always
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_DESCRIBE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NOT GIT_DESCRIBE STREQUAL "")
        string(REGEX REPLACE "^v" "" MT4G_VERSION ${GIT_DESCRIBE})
    else()
        set(MT4G_VERSION ${CMAKE_PROJECT_VERSION})
    endif()
else()
    set(MT4G_VERSION ${CMAKE_PROJECT_VERSION})
endif()
configure_file(${CMAKE_SOURCE_DIR}/include/version.hpp.in ${CMAKE_SOURCE_DIR}/include/version.hpp)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(SYSTEM $ENV{HIP_PATH}/include)
link_libraries(nlohmann_json::nlohmann_json)
link_libraries(cxxopts::cxxopts)

if(GPU_TARGET_ARCH MATCHES "^sm_")
    if(NOT DEFINED ENV{CUDA_PATH})
        message(FATAL_ERROR "CUDA_PATH is not set in environment")
    endif()
    set(ENV{HIP_PLATFORM} nvidia)

    add_compile_options(--gpu-architecture=${GPU_TARGET_ARCH} -x cu -Wno-deprecated-gpu-targets)
    include_directories(SYSTEM $ENV{CUDA_PATH}/include)
    link_directories($ENV{CUDA_PATH}/lib64)
    link_libraries(cudart)
elseif(GPU_TARGET_ARCH MATCHES "^gfx")
    set(ENV{HIP_PLATFORM} amd)

    add_compile_options(--offload-arch=${GPU_TARGET_ARCH} -x hip -Wall -Wextra -Wpedantic -Wnull-dereference)
    include_directories(SYSTEM /opt/rocm/include)
    link_directories(/opt/rocm/lib)
    link_libraries(rocm_smi64 hsa-runtime64 pthread)
else()
    message(FATAL_ERROR "Invalid GPU_TARGET_ARCH '${GPU_TARGET_ARCH}'. Must start with sm_ or gfx")
endif()

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cpp")

add_executable(mt4g ${SOURCES})

install(TARGETS mt4g DESTINATION bin)
