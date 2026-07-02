cmake_minimum_required(VERSION 3.21)

set(GPU_TARGET_ARCH "" CACHE STRING "The target GPU architecture")

# Lock CMake's HIP arch to GPU_TARGET_ARCH before project() so the build
# machine's GPU is never auto-detected. For NVIDIA (sm_*) targets the HIP
# arch variable is not used; set it empty to avoid contamination.
if(GPU_TARGET_ARCH MATCHES "^gfx")
    set(CMAKE_HIP_ARCHITECTURES "${GPU_TARGET_ARCH}" CACHE STRING "" FORCE)
else()
    set(CMAKE_HIP_ARCHITECTURES "" CACHE STRING "" FORCE)
endif()

# CMake 3.21+ supports HIP as a first-class language.
project(mt4g VERSION 1.2.0 LANGUAGES CXX HIP)

if(GPU_TARGET_ARCH STREQUAL "")
    message(FATAL_ERROR "Please set GPU_TARGET_ARCH, e.g. -DGPU_TARGET_ARCH=sm_70 or -DGPU_TARGET_ARCH=gfx90a")
endif()

find_package(hip REQUIRED CONFIG)
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
    find_package(hsa-runtime64 REQUIRED CONFIG)
    get_filename_component(_hip_bin "${CMAKE_HIP_COMPILER}" DIRECTORY)
    get_filename_component(_rocm_root "${_hip_bin}/../../.." ABSOLUTE)
    find_library(ROCM_SMI_LIB NAMES rocm_smi64 REQUIRED HINTS "${_rocm_root}/lib")

    # ROCm >= 7.x: rocm_smi.h transitively includes <libdrm/drm.h> via kfd_ioctl.h.
    # Requires libdrm-devel (dnf/apt) or libdrm loaded via spack.
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(LIBDRM QUIET libdrm)
    endif()
else()
    message(FATAL_ERROR "Invalid GPU_TARGET_ARCH '${GPU_TARGET_ARCH}'. Must start with sm_ or gfx")
endif()

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cpp")
# Force all .cpp sources through the HIP compiler (clang++) instead of the
# default CXX compiler (GCC), which does not understand --offload-arch / -x hip.
set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE HIP)

add_executable(mt4g ${SOURCES})

target_include_directories(mt4g PRIVATE ${CMAKE_SOURCE_DIR}/src ${LIBDRM_INCLUDE_DIRS})
set_target_properties(mt4g PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON HIP_STANDARD 20 HIP_STANDARD_REQUIRED ON)
target_link_libraries(mt4g PRIVATE
    nlohmann_json::nlohmann_json
    cxxopts::cxxopts
    hip::host
)

if(GPU_TARGET_ARCH MATCHES "^sm_")
    target_link_libraries(mt4g PRIVATE CUDA::cudart)
elseif(GPU_TARGET_ARCH MATCHES "^gfx")
    target_link_libraries(mt4g PRIVATE hsa-runtime64::hsa-runtime64 ${ROCM_SMI_LIB} pthread)
endif()

install(TARGETS mt4g DESTINATION bin)
