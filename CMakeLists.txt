cmake_minimum_required(VERSION 3.18)

project(FlashAttention)

find_package(CUDA REQUIRED)

if(NOT CUDA_VERSION VERSION_GREATER_EQUAL "11.6")
  message(FATAL_ERROR "CUDA version must be at least 11.6")
endif()

# Set CMAKE_CXX_FLAGS to make sure -DNDEBUG is not set
set(CMAKE_CXX_FLAGS_RELEASE "/MD /O2 /Ob2 /DCXX_BUILD " CACHE STRING "Release flags" FORCE)

# require c++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++17;-line-info;-O3;-U__CUDA_NO_HALF_OPERATORS__;-U__CUDA_NO_HALF_CONVERSIONS__;-U__CUDA_NO_HALF2_OPERATORS__;-U__CUDA_NO_BFLOAT16_CONVERSIONS__;--expt-relaxed-constexpr;--expt-extended-lambda;--use_fast_math;-gencode;arch=compute_80,code=sm_80;)

add_compile_definitions(FLASHATTENTION_DISABLE_LOCAL FLASHATTENTION_DISABLE_ALIBI FLASHATTENTION_DISABLE_UNEVEN_K)

if(CUDA_VERSION VERSION_GREATER "11.8")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_90,code=sm_90)
endif()

include_directories(
    fa
    cutlass/include
    ${CUDA_INCLUDE_DIRS}
)

cuda_add_library(flash_attn SHARED
    api.cpp
    # fa/flash_fwd_hdim64_fp16_sm80.cu
    # fa/flash_fwd_hdim96_fp16_sm80.cu
    fa/flash_fwd_hdim128_fp16_sm80.cu
    # fa/flash_fwd_split_hdim64_fp16_sm80.cu
    fa/flash_fwd_split_hdim128_fp16_sm80.cu
)
