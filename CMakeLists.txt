cmake_minimum_required(VERSION 3.5)
project(MPCAnonymousBloging)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -std=c++11 -mbmi2 -msse4.2 -march=native -m64 -pthread")
INCLUDE_DIRECTORIES($ENV{HOME} $ENV{HOME}/libscapi/install/include
        $ENV{HOME}/libscapi/install/include/OTExtensionBristol $ENV{HOME}/MPCAnonymousBloging/gemmOren)
link_directories(/usr/ssl/lib/ $ENV{HOME}/libscapi/install/lib)

set(SOURCE_FILES main.cpp ProtocolParty.cpp ProtocolParty.h)

add_executable(MPCAnonymousBloging ${SOURCE_FILES})
add_subdirectory(MPCAnonymousBloggingClient)

TARGET_LINK_LIBRARIES(MPCAnonymousBloging ntl.a
                      $ENV{HOME}/libscapi/libscapi.a gmp
                      boost_system pthread crypto dl)
