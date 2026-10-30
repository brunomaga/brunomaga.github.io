Support material for the post [GPT from scratch in C++ and CUDA vs. fused PyTorch](https://brunomaga.github.io/gpt-cpp-cuda).

- `gpt.h`: the dimensions of the model, and the deterministic values of its input and parameters;
- `gpt.py`: the GPT blocks in PyTorch, run in eager mode or with `torch.compile`;
- `gpt_libtorch.cpp`: the GPT blocks in LibTorch, the C++ API of PyTorch;
- `gpt_cpu.cpp`: the GPT blocks in plain C++ loops, on the CPU;
- `gpt_cuda.cu`: the GPT blocks in CUDA kernels and cuBLAS;
- `matmul.cu`: the matrix multiplication kernels of the CUDA C++ Best Practices Guide, a tiled kernel and cuBLAS;
- `ring_matmul.cu`: the ring algorithm of the distributed matrix multiplication on the GPUs of a node, with NCCL.

The CUDA programs need a GPU with bf16 Tensor Cores (compute capability 8.0 or more), the CUDA toolkit and NCCL. The
build needs CMake 3.24 or more, a C++20 compiler and PyTorch:

```shell
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`
make
```

When NCCL is not installed in a system folder (e.g. it comes with the `nvidia-nccl` Python package), its folder is
added to `CMAKE_PREFIX_PATH`. Each GPT program runs 5 forward passes of the 12 blocks to warm up, then 20 timed
ones, and prints the average runtime of a forward pass. The argument is the batch size:

```shell
python ../gpt.py --batch 8             # PyTorch
python ../gpt.py --batch 8 --compile   # PyTorch with torch.compile
./gpt_libtorch 8                       # LibTorch
./gpt_cuda 8                           # CUDA
./gpt_cpu 1                            # C++ on all the cores of the CPU (OMP_NUM_THREADS sets their number)
./matmul 4096 4096 4096                # M, N and O, multiples of 32
./ring_matmul 8192                     # M = N = O, a multiple of the number of GPUs
```

All GPT programs compute the same model on the same input, and print the first values of the output of the first
sequence, and the mean of its absolute values. The fp32 output of the CPU is
`0.7254 0.0187 1.1980 ..., mean |y| 0.6332`, and the bf16 outputs of the GPU programs are close to the bf16 output of
PyTorch on the CPU, `0.7188 0.0156 1.1953 ..., mean |y| 0.6332`.
`TORCH_LOGS=output_code python ../gpt.py --compile` prints the kernels that `torch.compile` generates.
