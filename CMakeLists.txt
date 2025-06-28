cmake_minimum_required(VERSION 3.10)
project(geometry)

set(CMAKE_CXX_STANDARD 17)

include_directories(include third_party/eigen)

set(C_SOURCES
    src/geometry/utils.c
    src/geometry/graph_ops_handler.c
    src/geometry/iterative_solver.c
    src/geometry/simd_ops.c
    src/assembly_backend/simd/simd_dispatch.c
    src/assembly_backend/simd/simd_impl_avx2.c
    src/assembly_backend/simd/simd_impl_sse2.c
    src/assembly_backend/simd/simd_impl_fallback.c
    src/assembly_backend/simd/memory_intrinsics.c
    src/assembly_backend/thread_ops.c
    src/assembly_backend/core.c
)
add_library(geometry ${C_SOURCES})

# Enable SIMD instruction set macros project-wide
if (CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID STREQUAL "Clang")
  # GCC/Clang: enable SSE2 and AVX2
  target_compile_options(geometry PRIVATE -msse2 -mavx2)
elseif (CMAKE_C_COMPILER_ID STREQUAL "MSVC")
  # MSVC: enable AVX2 (implies SSE2)
  target_compile_options(geometry PRIVATE /arch:AVX2)
endif()

# Enable NEON on ARM/AArch64 with GCC/Clang
if ((CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID STREQUAL "Clang") AND
    (CMAKE_SYSTEM_PROCESSOR MATCHES "arm.*" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64.*"))
  target_compile_options(geometry PRIVATE -mfpu=neon)
endif()

# compile SIMD backends with explicit instruction set flags
set_source_files_properties(src/assembly_backend/simd/simd_impl_avx2.c PROPERTIES COMPILE_FLAGS "-mavx2")
set_source_files_properties(src/assembly_backend/simd/simd_impl_sse2.c PROPERTIES COMPILE_FLAGS "-msse2")

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

add_executable(linked_list_creation_test tests/linked_list_creation_test.c)
target_link_libraries(linked_list_creation_test geometry)
add_test(NAME linked_list_creation_test COMMAND linked_list_creation_test)

# Additional configuration for GeometryTests
project(GeometryTests)

# Set the C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/tests)

# Add source files
file(GLOB_RECURSE SRC_FILES ${CMAKE_SOURCE_DIR}/src/*.c)
file(GLOB_RECURSE TEST_FILES ${CMAKE_SOURCE_DIR}/tests/*.c)

# Create the test executable
add_executable(geometry_tests ${SRC_FILES} ${TEST_FILES})

# Link libraries if needed (e.g., math library)
target_link_libraries(geometry_tests m)

# Enable testing
include(CTest)
add_test(NAME GeometryTests COMMAND geometry_tests)
