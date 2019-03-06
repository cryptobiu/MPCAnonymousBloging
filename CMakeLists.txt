cmake_minimum_required(VERSION 3.5)
project(MPCAnonymousBloging)

# enable_language(CUDA)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda/)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -std=c++11 -mbmi2")

INCLUDE_DIRECTORIES($ENV{HOME} $ENV{HOME}/libscapi/install/include
        $ENV{HOME}/libscapi/install/include/OTExtensionBristol $ENV{HOME}/MPCAnonymousBloging/gemmOren)

set(SOURCE_FILES main.cpp ProtocolParty.cpp ProtocolParty.h)


link_directories(/usr/ssl/lib/ $ENV{HOME}/libscapi/install/lib)

# add_executable(MPCAnonymousBloging ${SOURCE_FILES})

CUDA_ADD_EXECUTABLE(MPCAnonymousBloging ${SOURCE_FILES} gemmOren/kernel.cu)

TARGET_LINK_LIBRARIES(MPCAnonymousBloging ntl.a
                      $ENV{HOME}/libscapi/libscapi.a gmp
                      libboost_system.a pthread crypto dl)
