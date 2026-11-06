Support material for the post [GPT from scratch in LibTorch, NumPy, C++ and CUDA](https://brunomaga.github.io/gpt-cpp-cuda).

- `gpt.h`: the dimensions of the model, the type of its values, and the random values of its input and parameters;
- `gpt.py`: the GPT blocks in PyTorch, run in eager mode or with `torch.compile`;
- `gpt_libtorch.cpp`: the GPT blocks in LibTorch, the C++ API of PyTorch;
- `gpt_cpu.cpp`: the GPT blocks in plain C++ loops, on the CPU;
- `gpt_cuda.cu`: the GPT blocks in CUDA kernels and cuBLAS;
- `matmul.cu`: the matrix multiplication kernels adapted from the CUDA C++ Best Practices Guide, a tiled kernel and
  cuBLAS.

The NumPy version of the GPT blocks is only in the post.

The CUDA programs need a GPU with BF16 Tensor Cores (compute capability 8.0 or more) and the CUDA toolkit. The build
needs CMake 3.24 or more, a C++20 compiler and PyTorch:

```shell
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`
make
```

Each GPT program runs 5 forward passes of the 12 blocks to warm up, then 20 timed ones, and prints the average
runtime of a forward pass. The argument is the batch size:

```shell
python ../gpt.py --batch 8             # PyTorch
python ../gpt.py --batch 8 --compile   # PyTorch with torch.compile
./gpt_libtorch 8                       # LibTorch
./gpt_cuda 8                           # CUDA
./gpt_cpu 1                            # C++ on all the cores of the CPU (OMP_NUM_THREADS sets their number)
./matmul 4096 4096 4096                # M, N and O, multiples of 32
```

All GPT programs compute the same model on the same input, and print the first values of the output of the first
sequence, and the mean of its absolute values. The FP32 output of the CPU program is
`-0.4817 1.0697 0.2294 ..., mean |y| 0.6358`, and the BF16 outputs of the GPU programs are close to the BF16 output of
PyTorch on the CPU, `-0.4863 1.0781 0.2324 ..., mean |y| 0.6358`.
`TORCH_LOGS=output_code python ../gpt.py --compile` prints the kernels that `torch.compile` generates.
