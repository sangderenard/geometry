cmake_minimum_required(VERSION 3.10)
project(vecdec)

set(CMAKE_CXX_STANDARD 17)

include_directories(include third_party/eigen)

file(GLOB SOURCES "src/*.cpp")
add_library(vecdec ${SOURCES})

add_executable(example examples/minimal.cpp)
target_link_libraries(example vecdec)
