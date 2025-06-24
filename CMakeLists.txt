cmake_minimum_required(VERSION 3.10)
project(vecdec)

set(CMAKE_CXX_STANDARD 17)

include_directories(include third_party/eigen)

file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
list(FILTER CPP_SOURCES EXCLUDE REGEX "dec_toolkit_header.cpp$")
file(GLOB_RECURSE C_SOURCES "src/*.c")
list(FIND C_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/geometry/guardian.c" _guardian_idx)
if(_guardian_idx EQUAL -1)
    list(APPEND C_SOURCES src/geometry/guardian.c)
endif()
set_source_files_properties(src/geometry/guardian.c PROPERTIES LANGUAGE C)
add_library(vecdec ${CPP_SOURCES} ${C_SOURCES})
target_sources(vecdec PRIVATE src/geometry/guardian.c)

add_library(geometry_stencil
    src/geometry/stencil.c
)
target_include_directories(geometry_stencil PUBLIC include)

add_library(geometry_relational_vector
    src/geometry/relational_vector.c
)
target_include_directories(geometry_relational_vector PUBLIC include)

# Examples are optional; disable by default to avoid external deps
# add_executable(example examples/minimal.cpp)
# target_link_libraries(example vecdec)

enable_testing()
add_executable(utils_test tests/utils_test.c)
target_link_libraries(utils_test vecdec)
add_test(NAME utils_test COMMAND utils_test)
