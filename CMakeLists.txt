cmake_minimum_required(VERSION 3.18)
project(desrt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Target L4 (Ada Lovelace, SM 8.9) by default. Override with -DCMAKE_CUDA_ARCHITECTURES.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

# nvcc flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Position-independent linking for sanity
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>
                       $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                       $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(desrt
    src/main.cpp
    src/shard.cpp
    src/cmd_plan.cpp
    src/cmd_sort.cpp
    src/cmd_make_target.cpp
    src/cmd_stats.cpp
    src/cmd_bench.cu
    src/cmd_build.cu
    src/cmd_crack.cu
)

set_target_properties(desrt PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

if(UNIX)
    target_link_libraries(desrt PRIVATE pthread)
endif()

install(TARGETS desrt RUNTIME DESTINATION bin)
