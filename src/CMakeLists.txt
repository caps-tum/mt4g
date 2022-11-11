

#TODO: IF(WIN32)....
#set(CMAKE_THREAD_LIBS_INIT "-lpthread")
#set(CMAKE_HAVE_THREADS_LIBRARY 1)
#set(CMAKE_USE_WIN32_THREADS_INIT 0)
#set(CMAKE_USE_PTHREADS_INIT 1)
#set(THREADS_PREFER_PTHREAD_FLAG ON)



include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(cuda-samples/Common)
link_directories(${CMAKE_CURRENT_BINARY_DIR})

add_executable(c15 starter_other/ConstMemory.cu)
add_executable(mt4g capture.cu)

set_property(TARGET mt4g PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(mt4g PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(mt4g PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set_property(TARGET c15 PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(c15 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(c15 PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_link_options("-lcuda")
#add_link_options("-lnvidia-ml")

message(${CUDA_TOOLKIT_ROOT_DIR})

IF(UNIX)
    #target_link_libraries( mt4g ${CUDA_LIBRARIES} ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/libnvidia-ml.so "stdc++fs" )
    target_link_libraries( mt4g ${CUDA_LIBRARIES} "stdc++fs" )
    target_link_libraries( c15 ${CUDA_LIBRARIES} "stdc++fs")
ENDIF()

IF(WIN32)
    target_link_libraries( mt4g "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib")
    #target_link_libraries( mt4g "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/nvml.lib")
    target_link_libraries( c15 "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib")
ENDIF()

IF(DEFINED COMPUTE_CAP_COMPILE_OPTIONS)
    target_compile_options(mt4g ${COMPUTE_CAP_COMPILE_OPTIONS})
    target_compile_options(c15 ${COMPUTE_CAP_COMPILE_OPTIONS})
else()
    target_compile_options(mt4g PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
    target_compile_options(c15 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
endif()

if ("${IsDebug}" STREQUAL "1")
    add_definitions(-DIsDebug)
endif()


install(TARGETS mt4g c15 DESTINATION .)
