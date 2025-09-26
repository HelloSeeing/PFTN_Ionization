
cmake_minimum_required(VERSION 3.22)
project(DEMO)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

add_compile_options(-Wall -Wextra -pedantic)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pipe -std=c99")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -std=c++11")

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

file(GLOB SOURCES "src/*.cpp")
message(STATUS "sources = ${SOURCES}")

add_executable(demo ${SOURCES})
target_include_directories(demo PUBLIC ${PROJECT_SOURCE_DIR}/inc)
