cmake_minimum_required(VERSION 3.10)

# Add clang detection on Windows
if (WIN32)
  find_program(CLANG_C_COMPILER clang PATHS "C:/Program Files/LLVM/bin" "C:/LLVM/bin" ENV PATH)
  find_program(CLANG_CXX_COMPILER clang++ PATHS "C:/Program Files/LLVM/bin" "C:/LLVM/bin" ENV PATH)
  if (CLANG_C_COMPILER AND CLANG_CXX_COMPILER)
    message(STATUS "Found Clang: using ${CLANG_C_COMPILER} and ${CLANG_CXX_COMPILER}")
    set(CMAKE_C_COMPILER "${CLANG_C_COMPILER}" CACHE FILEPATH "C compiler" FORCE)
    set(CMAKE_CXX_COMPILER "${CLANG_CXX_COMPILER}" CACHE FILEPATH "C++ compiler" FORCE)
  endif()
endif()

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
## C and C++ standards
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Include paths
include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/tests
)

## Combined tests executable
file(GLOB_RECURSE TEST_SOURCES
  ${CMAKE_SOURCE_DIR}/tests/*.c
)
add_executable(geometry_tests ${TEST_SOURCES})
target_link_libraries(geometry_tests geometry m)
include(CTest)
add_test(NAME GeometryTests COMMAND geometry_tests)

# Add resource compiler detection for Ninja/Clang on Windows
if (WIN32)
  # If no RC compiler found, try to locate rc.exe from Windows SDK
  if (NOT CMAKE_RC_COMPILER)
    find_program(RC_COMPILER rc.exe HINTS "C:/Program Files (x86)/Windows Kits/10/bin" ENV PATH)
    if (RC_COMPILER)
      message(STATUS "Found RC compiler: ${RC_COMPILER}")
      set(CMAKE_RC_COMPILER "${RC_COMPILER}" CACHE FILEPATH "Resource Compiler" FORCE)
    endif()
  endif()
endif()
