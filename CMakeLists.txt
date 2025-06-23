cmake_minimum_required(VERSION 3.10)
project(vecdec)

set(CMAKE_CXX_STANDARD 17)

include_directories(include third_party/eigen)

file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
list(FILTER CPP_SOURCES EXCLUDE REGEX "dec_toolkit_header.cpp$")
file(GLOB_RECURSE C_SOURCES "src/*.c")
add_library(vecdec ${CPP_SOURCES} ${C_SOURCES})

# Examples are optional; disable by default to avoid external deps
# add_executable(example examples/minimal.cpp)
# target_link_libraries(example vecdec)

enable_testing()
add_executable(utils_test tests/utils_test.c)
target_link_libraries(utils_test vecdec)
add_test(NAME utils_test COMMAND utils_test)
