cmake_minimum_required(VERSION 3.10)
project(geometry)

set(CMAKE_CXX_STANDARD 17)

include_directories(include third_party/eigen)

set(GEOMETRY_SOURCES
    src/geometry/utils.c
    src/geometry/guardian_platform.c
)
set_source_files_properties(${GEOMETRY_SOURCES} PROPERTIES LANGUAGE CXX)
add_library(geometry ${GEOMETRY_SOURCES})


# Examples are optional; disable by default to avoid external deps
# add_executable(example examples/minimal.cpp)
# target_link_libraries(example geometry)

enable_testing()
add_executable(utils_test tests/utils_test.c)
target_link_libraries(utils_test geometry)
add_test(NAME utils_test COMMAND utils_test)

add_executable(graph_ops_handler_test tests/graph_ops_handler_test.c)
target_link_libraries(graph_ops_handler_test geometry)
add_test(NAME graph_ops_handler_test COMMAND graph_ops_handler_test)
