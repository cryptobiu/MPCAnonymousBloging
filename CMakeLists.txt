cmake_minimum_required(VERSION 2.8)
project(MPCAnonymousBloging)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -std=c++11 -maes -mpclmul -mbmi2 -mavx -mavx2 -pthread")

INCLUDE_DIRECTORIES($ENV{HOME} $ENV{HOME}/libscapi/install/include
        $ENV{HOME}/libscapi/install/include/OTExtensionBristol)

set(SOURCE_FILES main.cpp
        ProtocolParty.cpp ProtocolParty.h ProtocolTimer.cpp ProtocolTimer.h
        )


link_directories(/usr/ssl/lib/ $ENV{HOME}/libscapi/install/lib )


add_executable(MPCAnonymousBloging ${SOURCE_FILES})

TARGET_LINK_LIBRARIES(MPCAnonymousBloging ntl.a
        $ENV{HOME}/libscapi/libscapi.a gmp gmpxx
        libboost_system.a libboost_thread.a pthread crypto dl ssl z)
