cmake_minimum_required(VERSION 3.10)
project(geometry)

set(CMAKE_CXX_STANDARD 17)

include_directories(include third_party/eigen)

set(C_SOURCES
    src/geometry/utils.c
    src/geometry/graph_ops_handler.c
    src/geometry/iterative_solver.c
    src/geometry/simd_ops.c
    src/simd/simd_dispatch.c
    src/simd/simd_impl_avx2.c
    src/simd/simd_impl_sse2.c
    src/simd/simd_impl_fallback.c
)
add_library(geometry ${C_SOURCES})

# compile SIMD backends with explicit instruction set flags
set_source_files_properties(src/simd/simd_impl_avx2.c PROPERTIES COMPILE_FLAGS "-mavx2")
set_source_files_properties(src/simd/simd_impl_sse2.c PROPERTIES COMPILE_FLAGS "-msse2")

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
# target_link_libraries(example geometry)

enable_testing()
add_executable(utils_test tests/utils_test.c)
target_link_libraries(utils_test geometry)
add_test(NAME utils_test COMMAND utils_test)

add_executable(graph_ops_handler_test tests/graph_ops_handler_test.c)
target_link_libraries(graph_ops_handler_test geometry)
add_test(NAME graph_ops_handler_test COMMAND graph_ops_handler_test)

add_executable(iterative_solver_test tests/iterative_solver_test.c)
target_link_libraries(iterative_solver_test geometry)
add_test(NAME iterative_solver_test COMMAND iterative_solver_test)

add_executable(complex_numbers_test tests/complex_numbers_test.c)
target_link_libraries(complex_numbers_test geometry)
add_test(NAME complex_numbers_test COMMAND complex_numbers_test)
