# is 3.22, was tested also under 3.10
cmake_minimum_required(VERSION 3.22)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
link_directories(${CMAKE_CURRENT_BINARY_DIR})

# src: https://github.com/GPUOpen-ProfessionalCompute-Libraries/rpp/blob/ab75f4d90bf8ebccd10dd4292983a6638807c4c1/utilities/rpp-unittests/HIP_NEW/CMakeLists.txt

if(NOT "${HIP_PATH}")
    set(HIP_PATH "/opt/rocm/hip")
endif()

if(NOT "${CMAKE_MODULE_PATH}")
    set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" "${HIP_PATH}/lib/cmake/hip" ${CMAKE_MODULE_PATH})
endif()

if(NOT "${CMAKE_PREFIX_PATH}")
    list(APPEND CMAKE_PREFIX_PATH
            "${HIP_PATH}/cmake"
            "${HIP_PATH}/lib/cmake"
            "${HIP_PATH}/../lib/cmake" # hopefully catches all extra HIP dependencies
            )
endif()

message(STATUS "SRC module path:   ${CMAKE_MODULE_PATH}")
message(STATUS "SRC prefix path:   ${CMAKE_PREFIX_PATH}")

# Set compiler and linker
find_package(HIP REQUIRED HINTS
        "${HIP_PATH}/cmake"
        "${HIP_PATH}"
        "opt/rocm/hip")
message(STATUS "HIP libraries: ${HIP_LIBRARIES}")


set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})


if (HIP_FOUND)

    # Create the executable
    add_executable(c15 starter_other/ConstMemory.cpp)
    add_executable(mt4g capture.cpp general_functions.h)

    # Link with HIP
    target_include_directories(c15 PRIVATE ${ROCM_PATH}/hsa/include)
    target_include_directories(mt4g PRIVATE ${ROCM_PATH}/hsa/include)

    IF(UNIX)
        target_link_libraries( mt4g ${HIP_LIBRARIES} "stdc++fs")

        target_link_libraries( c15 ${HIP_LIBRARIES} "stdc++fs")
    ENDIF()

    if ("${IsDebug}" STREQUAL "1")
        add_definitions(-DIsDebug)
    endif()



    install(TARGETS mt4g c15 DESTINATION bin)
else()
    message(FATAL_ERROR "src/CMakeLists.txt: HIP not found!")
endif()
##########################################
