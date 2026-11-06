cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

# by default, the GPUs of this machine (bf16 on the Tensor Cores needs a compute capability of 8.0 or more)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(gpt_cpp_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# the CPU implementation, on all the cores of the CPU with OpenMP
find_package(OpenMP REQUIRED)
add_executable(gpt_cpu gpt_cpu.cpp)
target_compile_options(gpt_cpu PRIVATE -march=native)
target_link_libraries(gpt_cpu PRIVATE OpenMP::OpenMP_CXX)

# the CUDA implementation and the matrix multiplication kernels, with cuBLAS
find_package(CUDAToolkit REQUIRED)
add_executable(gpt_cuda gpt_cuda.cu)
target_link_libraries(gpt_cuda PRIVATE CUDA::cublas)
add_executable(matmul matmul.cu)
target_link_libraries(matmul PRIVATE CUDA::cublas)

# the LibTorch implementation. find_package(Torch) adds the CUDA architectures of PyTorch to the flags of all the
# CUDA targets, so the flags are restored after it
set(CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
find_package(Torch REQUIRED)
set(CMAKE_CUDA_FLAGS "${CUDA_FLAGS}")
add_executable(gpt_libtorch gpt_libtorch.cpp)
target_link_libraries(gpt_libtorch PRIVATE ${TORCH_LIBRARIES})
