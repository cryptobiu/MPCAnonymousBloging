cmake_minimum_required(VERSION 3.5)
project(MPCAnonymousBloging)


# enable_language(CUDA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -std=c++11 -mbmi2 -msse4.2 -march=native")
INCLUDE_DIRECTORIES($ENV{HOME} $ENV{HOME}/libscapi/install/include
        $ENV{HOME}/libscapi/install/include/OTExtensionBristol $ENV{HOME}/MPCAnonymousBloging/gemmOren)
link_directories(/usr/ssl/lib/ $ENV{HOME}/libscapi/install/lib)

set(SOURCE_FILES main.cpp ProtocolParty.cpp ProtocolParty.h)

if(CUDA_TARGET_COMPUTE_CAPABILITY)
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda/)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)


# nvcc flags
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_70 -Xcompiler -march=native)
    CUDA_ADD_EXECUTABLE(MPCAnonymousBloging ${SOURCE_FILES} gemmOren/kernel.cu)

else()
add_executable(MPCAnonymousBloging ${SOURCE_FILES})
endif(CUDA_TARGET_COMPUTE_CAPABILITY)


TARGET_LINK_LIBRARIES(MPCAnonymousBloging ntl.a
                      $ENV{HOME}/libscapi/libscapi.a gmp
                      libboost_system.a pthread crypto dl)
