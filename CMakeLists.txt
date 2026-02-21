cmake_minimum_required(VERSION 3.24)
project(flashkernel LANGUAGES CXX CUDA)

# ─── CUDA Configuration ─────────────────────────────────────────────────────
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)  # SM 7.5 = Turing (T4)

# Optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0")

# ─── Find Dependencies ──────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# pybind11 — prefer installed, fallback to FetchContent
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.12.0
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# ─── CUDA Kernel Library ────────────────────────────────────────────────────
# All .cu files compiled into a static library
file(GLOB KERNEL_SOURCES "src/cuda/*.cu")
add_library(flashkernel_kernels STATIC ${KERNEL_SOURCES})
target_include_directories(flashkernel_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(flashkernel_kernels PUBLIC CUDA::cudart)
set_target_properties(flashkernel_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ─── Python Extension Module ────────────────────────────────────────────────
pybind11_add_module(_flashkernel_C src/bindings/torch_ext.cpp)
target_link_libraries(_flashkernel_C PRIVATE flashkernel_kernels CUDA::cudart)
target_include_directories(_flashkernel_C PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda
)

# ─── Install ────────────────────────────────────────────────────────────────
install(TARGETS _flashkernel_C DESTINATION flashkernel)

# ─── Info ────────────────────────────────────────────────────────────────────
message(STATUS "FlashKernel build configuration:")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUDA compiler:      ${CMAKE_CUDA_COMPILER}")
message(STATUS "  Build type:         ${CMAKE_BUILD_TYPE}")
