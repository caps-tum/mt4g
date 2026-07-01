cmake_minimum_required(VERSION 3.21)

# CMake 3.21+ supports HIP as a first-class language, which handles compiler
# and HIP path discovery automatically. If CMake cannot find HIP, set HIP_PATH
# and call find_package(hip REQUIRED CONFIG) manually as a fallback.
project(mt4g VERSION 1.2.0 LANGUAGES CXX HIP)

set(GPU_TARGET_ARCH "" CACHE STRING "The target GPU architecture")
if(GPU_TARGET_ARCH STREQUAL "")
    message(FATAL_ERROR "Please set GPU_TARGET_ARCH, e.g. -DGPU_TARGET_ARCH=sm_70 or -DGPU_TARGET_ARCH=gfx90a")
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
        set(MT4G_VERSION "${CMAKE_PROJECT_VERSION}-cmake")
    endif()
else()
    set(MT4G_VERSION "${CMAKE_PROJECT_VERSION}-cmake")
endif()
configure_file(${CMAKE_SOURCE_DIR}/src/version.hpp.in ${CMAKE_SOURCE_DIR}/src/version.hpp)

set(CMAKE_CXX_EXTENSIONS OFF)

if(GPU_TARGET_ARCH MATCHES "^sm_")
    set(ENV{HIP_PLATFORM} nvidia)
    add_compile_options(--gpu-architecture=${GPU_TARGET_ARCH} -x cu -Wno-deprecated-gpu-targets)
    find_package(CUDAToolkit REQUIRED)
elseif(GPU_TARGET_ARCH MATCHES "^gfx")
    set(ENV{HIP_PLATFORM} amd)
    add_compile_options(--offload-arch=${GPU_TARGET_ARCH} -x hip -Wall -Wextra -Wpedantic -Wnull-dereference)
else()
    message(FATAL_ERROR "Invalid GPU_TARGET_ARCH '${GPU_TARGET_ARCH}'. Must start with sm_ or gfx")
endif()

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cpp")

add_executable(mt4g ${SOURCES})

target_include_directories(mt4g PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_features(mt4g PUBLIC cxx_std_20)
target_link_libraries(mt4g PRIVATE
    nlohmann_json::nlohmann_json
    cxxopts::cxxopts
    hip::host
)

if(GPU_TARGET_ARCH MATCHES "^sm_")
    target_link_libraries(mt4g PRIVATE CUDA::cudart)
elseif(GPU_TARGET_ARCH MATCHES "^gfx")
    target_link_directories(mt4g PRIVATE /opt/rocm/lib)
    target_link_libraries(mt4g PRIVATE rocm_smi64 hsa-runtime64 pthread)
endif()

install(TARGETS mt4g DESTINATION bin)
