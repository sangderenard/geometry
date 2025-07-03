cmake_minimum_required(VERSION 3.12)
project(geometry LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Allow MSVC to serialize PDB writes when building in parallel
if(MSVC)
    add_compile_options(/FS)
endif()

# Add Eigen submodule.
# Make sure you have run ./init_submodules.sh first.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen/CMakeLists.txt")
    add_subdirectory(third_party/eigen)
else()
    message(FATAL_ERROR "Eigen submodule not found. Please run ./init_submodules.sh")
endif()

# --- SIMD / AVX2 Detection ---
include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)

# Check for AVX2 support and set the appropriate flag variable
if(MSVC)
    check_cxx_compiler_flag("/arch:AVX2" CXX_SUPPORTS_AVX2)
    check_c_compiler_flag("/arch:AVX2" C_SUPPORTS_AVX2)
    if(CXX_SUPPORTS_AVX2)
        set(AVX2_CXX_FLAG "/arch:AVX2")
    endif()
    if(C_SUPPORTS_AVX2)
        set(AVX2_C_FLAG "/arch:AVX2")
    endif()
else() # GCC, Clang, etc.
    check_cxx_compiler_flag("-mavx2" CXX_SUPPORTS_AVX2)
    check_c_compiler_flag("-mavx2" C_SUPPORTS_AVX2)
    if(CXX_SUPPORTS_AVX2)
        set(AVX2_CXX_FLAG "-mavx2")
    endif()
    if(C_SUPPORTS_AVX2)
        set(AVX2_C_FLAG "-mavx2")
    endif()
endif()

# --- Project Target ---

# Gather all source and header files
file(GLOB_RECURSE GEOMETRY_SOURCES 
    "src/*.c" 
    "src/*.cpp"
)

# For a real project, explicitly listing files is better than GLOB_RECURSE,
# but this is convenient for getting started.

add_library(geometry STATIC ${GEOMETRY_SOURCES})

# Add include directories
target_include_directories(geometry PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/assembly_backend"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/assembly_backend/simd"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen"
)

# Add compile flags if supported
if(AVX2_CXX_FLAG)
    target_compile_options(geometry PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${AVX2_CXX_FLAG}>)
    message(STATUS "AVX2 support for C++ enabled with flag: ${AVX2_CXX_FLAG}")
endif()
if(AVX2_C_FLAG)
    target_compile_options(geometry PRIVATE $<$<COMPILE_LANGUAGE:C>:${AVX2_C_FLAG}>)
    message(STATUS "AVX2 support for C enabled with flag: ${AVX2_C_FLAG}")
endif()

# Link to Eigen
target_link_libraries(geometry PRIVATE Eigen3::Eigen)

# --- CTest integration ---
enable_testing()
file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/tests/*_test.c")
foreach(TEST_SRC ${TEST_SOURCES})
    get_filename_component(testname ${TEST_SRC} NAME_WE)
    add_executable(${testname} ${TEST_SRC})
    target_link_libraries(${testname} PRIVATE geometry)
    add_test(NAME ${testname} COMMAND ${testname})
endforeach()