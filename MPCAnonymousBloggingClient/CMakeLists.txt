cmake_minimum_required(VERSION 3.5)
project(MPCAnonymousBloggingClient)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -mbmi2")

INCLUDE_DIRECTORIES($ENV{HOME} ${BOOST_INCLUDEDIR} $ENV{HOME}/libscapi/lib/
        $ENV{HOME}/libscapi/install/include)

link_directories(/usr/ssl/lib/ $ENV{HOME}/libscapi/install/lib ${BOOST_LIBRARYDIR})

set(SOURCE_FILES main.cpp Client.cpp)
add_executable(MPCAnonymousBloggingClient ${SOURCE_FILES})

TARGET_LINK_LIBRARIES(MPCAnonymousBloggingClient $ENV{HOME}/libscapi/libscapi.a gmp crypto pthread)
