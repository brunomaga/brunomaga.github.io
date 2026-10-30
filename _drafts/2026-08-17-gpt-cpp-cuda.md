---
layout: post
title:  "GPT from scratch in C++ and CUDA vs. fused PyTorch"
categories: ["Efficient Inference"]
tags: ["GPT", "LLM", "C++", "CUDA", "LibTorch", "cuBLAS", "Tensor Cores", "NCCL", "kernel fusion", "torch.compile", "benchmarking"]
---

The [matrix multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication) is the core operation of machine learning. The linear layers and the attention of a [Transformer](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)) spend most of their time in it, and the libraries and the hardware of ML, from the [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) libraries of CPUs to the Tensor Cores of GPUs, are built around it.

On top of these libraries, PyTorch runs a model from Python. In the [PyTorch 2.x release announcement](https://pytorch.org/get-started/pytorch-2.0/), the Torch developers moved parts of PyTorch from a Python interface over a C++ core (highly efficient) to a Python-based implementation that lowers its operations to a small set of primitive operators (developer-friendly). They also announced `torch.compile`, a method to perform [JIT compilation](https://en.wikipedia.org/wiki/Just-in-time_compilation) and optimization of the execution graph that speeds up execution.

So the main questions are: how much faster are model implementations in C++ compared to Python ones? Can a model written from scratch, in plain C++ and [CUDA](https://en.wikipedia.org/wiki/CUDA), run faster than PyTorch? How good is `torch.compile` in practice? To answer them, this post implements the matrix multiplication and the block of a [GPT](https://en.wikipedia.org/wiki/Generative_pre-trained_transformer) model from scratch, in C++ on the CPU and in CUDA on the GPU, and in LibTorch, the C++ API of PyTorch. It then benchmarks them against PyTorch, with and without `torch.compile`, and ends with the limits of kernel fusion on several GPUs, the topic of the posts that follow.

## Matrix multiplication

Take a matrix $$A$$ of size $$M \times N$$ and a matrix $$B$$ of size $$N \times O$$. Their product $$C = AB$$ is a matrix of size $$M \times O$$, whose values are given by:

$$
C_{ij} = \sum_{k=1}^{N} A_{ik} B_{kj}
$$

where $$i$$ and $$j$$ are the row and the column of a value of $$C$$, i.e. each value of $$C$$ is the dot product of a row of $$A$$ and a column of $$B$$. The product takes $$MNO$$ multiplications and as many additions, i.e. $$2MNO$$ floating-point operations ([FLOPs](https://en.wikipedia.org/wiki/FLOPS)), or $$2n^3$$ for square matrices of size $$n$$, a cubic complexity. However, it reads only $$MN + NO$$ values and writes $$MO$$, so that for square matrices each value of $$A$$ and $$B$$ is used $$n$$ times. Thus, a large matrix multiplication is *compute-bound*: its speed depends on the arithmetic units of the hardware, and on how well an implementation reuses the data that it reads (see the [roofline model]({{ site.baseurl }}{% post_url 2025-06-07-inference-prefill-decode-roofline-cache %}#roofline-model)).

In PyTorch, the product is a single operator:

```python
C = A @ B  # or torch.matmul(A, B)
```

and LibTorch, the C++ distribution of PyTorch, provides the same operator, with an API as close as possible to the Python one:

```cpp
torch::Tensor C = torch::matmul(A, B);
```

Both call the matrix multiplication of a library of the device: a BLAS library on the CPU (e.g. MKL or OpenBLAS), and [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html) on NVIDIA GPUs. The rest of this section writes the product from scratch, from loops on the CPU to the Tensor Cores of a GPU and to several GPUs.

### On the CPU

The CPU code of this post follows the style of [llama2.c](https://github.com/karpathy/llama2.c), in which Andrej Karpathy runs the inference of Llama 2 in a single file of plain C, and of [llm.c](https://github.com/karpathy/llm.c), in which he trains GPT-2 in plain C and CUDA. Both write each operation as loops over arrays of floats, and run the outer loop on all the cores of the CPU with [OpenMP](https://en.wikipedia.org/wiki/OpenMP).

The C++ standard library has no matrix product until C++26, which adds `std::linalg::matrix_product` (in the header `<linalg>`), an interface based on BLAS. So the product is written as three nested loops:

```cpp
// C (M x O) = A (M x N) B (N x O), with all matrices in row-major order
void matmul(const dt* A, const dt* B, dt* C, int M, int N, int O) {
  #pragma omp parallel for
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < O; j++)
      C[i*O + j] = 0;
    for (int k = 0; k < N; k++)       // k before j, so that the inner loop reads B and C contiguously
      for (int j = 0; j < O; j++)
        C[i*O + j] += A[i*N + k] * B[k*O + j];
  }
}
```

The matrices are stored in [row-major order](https://en.wikipedia.org/wiki/Row-_and_column-major_order), i.e. row after row, as in C and PyTorch, so that $$A_{ik}$$ is at `A[i*N + k]`, and `dt` is the type of the values (`float` on the CPU, as explained later). The loop over `k` comes before the loop over `j`, so that the inner loop reads a row of $$B$$ and a row of $$C$$ contiguously, which the compiler turns into vector instructions ([SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data)). In the textbook order, with the loop over `k` inside, each iteration would read a value of $$B$$ one row away from the previous one, and use a single value of each cache line that it loads ([locality of reference](https://en.wikipedia.org/wiki/Locality_of_reference)). The line `#pragma omp parallel for` splits the rows of $$C$$ among the threads of the CPU, which write disjoint rows and need no synchronization.

Faster CPU implementations also split the loops into tiles that fit the caches of the CPU ([loop nest optimization](https://en.wikipedia.org/wiki/Loop_nest_optimization)). This is the idea of the GPU kernels that follow.

### On the GPU, with CUDA

A GPU runs a function, called a **kernel**, on thousands of threads at once. The threads are grouped in [thread blocks](https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming)), and each block runs on one Streaming Multiprocessor (SM) of the GPU. The threads of a block share a small and fast on-chip **shared memory**, while all threads read and write the large and slower global memory, the [high-bandwidth memory](https://en.wikipedia.org/wiki/High_Bandwidth_Memory) (HBM). A block runs as groups of 32 threads, called **warps**, whose threads execute the same instruction ([SIMT](https://en.wikipedia.org/wiki/Single_instruction,_multiple_threads)), and a warp reads global memory fastest when its 32 threads read consecutive addresses, i.e. when the access is *coalesced*.

The [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-ab) illustrates the use of shared memory on the product of $$A$$ of size $$M \times w$$ and $$B$$ of size $$w \times O$$, i.e. for $$N = w$$, where $$w = 32$$ is the size of a warp (`TILE_DIM` in the code). Each thread computes one value of $$C$$, and each thread block of $$32 \times 32$$ threads computes a tile of $$32 \times 32$$ values of $$C$$. The guide presents three kernels, and measures their effective bandwidth on a Tesla V100:
1. `simpleMultiply` ("Unoptimized matrix multiplication") reads $$A$$ and $$B$$ from global memory. The 32 threads of a warp compute 32 consecutive values of a row of $$C$$, so they read consecutive values of $$B$$, but they all read the same value of $$A$$, which wastes most of each memory transaction. It reaches 119.9 GB/s;
2. `coalescedMultiply` ("Using shared memory to improve the global memory load efficiency in matrix multiplication") first reads the tile of $$A$$ into shared memory, where each warp reads one row of the tile with coalesced reads. In the loop, shared memory then *broadcasts* each value of $$A$$ to all threads of the warp. The barrier `__syncwarp()` is enough, as each warp reads only the row of the tile that it wrote. It reaches 144.4 GB/s;
3. `sharedABMultiply` ("Improvement by reading additional data into shared memory") also reads the tile of $$B$$ into shared memory, so that each value of $$B$$ is read once per thread block instead of once per warp. The barrier is now `__syncthreads()`, as a warp reads values of $$B$$ written by the other warps of the block. It reaches 195.5 GB/s.

The guide notes that the gains are not due to a better coalescing, but to fewer redundant transfers from global memory. Here are the first and the last kernels, with [bf16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) values (`dt`) and sums in [fp32](https://en.wikipedia.org/wiki/Single-precision_floating-point_format):

```cpp
constexpr int TILE_DIM = 32;   // tiles of TILE_DIM x TILE_DIM values, with one thread per value

// C (M x O) = A (M x N) B (N x O) for N = TILE_DIM, with one thread per value of C, that reads A and B from global memory
__global__ void simpleMultiply(const dt* A, const dt* B, dt* C, int O) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0;
  for (int i = 0; i < TILE_DIM; i++)
    sum += (float) A[row*TILE_DIM + i] * (float) B[i*O + col];
  C[row*O + col] = sum;
}

// the same, with the tiles of A and B read into shared memory
__global__ void sharedABMultiply(const dt* A, const dt* B, dt* C, int O) {
  __shared__ dt aTile[TILE_DIM][TILE_DIM], bTile[TILE_DIM][TILE_DIM];
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0;
  aTile[threadIdx.y][threadIdx.x] = A[row*TILE_DIM + threadIdx.x];
  bTile[threadIdx.y][threadIdx.x] = B[threadIdx.y*O + col];
  __syncthreads();
  for (int i = 0; i < TILE_DIM; i++)
    sum += (float) aTile[threadIdx.y][i] * (float) bTile[i][threadIdx.x];
  C[row*O + col] = sum;
}
```

The kernels of the guide only work for $$N = 32$$. For a larger $$N$$, a multiple of 32, each thread block loops over the $$N / 32$$ tiles of its row of tiles of $$A$$ and of its column of tiles of $$B$$, multiplies each pair of tiles as `sharedABMultiply` does, and adds the products:

```cpp
// C (M x O) = A (M x N) B (N x O) for M, N and O multiples of TILE_DIM: each thread block computes a tile of C as
// the sum of the products of the N / TILE_DIM pairs of tiles of A and B, read one pair at a time into shared memory
__global__ void tiledMultiply(const dt* A, const dt* B, dt* C, int N, int O) {
  __shared__ dt aTile[TILE_DIM][TILE_DIM], bTile[TILE_DIM][TILE_DIM];
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0;
  for (int t = 0; t < N; t += TILE_DIM) {
    aTile[threadIdx.y][threadIdx.x] = A[row*N + t + threadIdx.x];
    bTile[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y)*O + col];
    __syncthreads();
    for (int i = 0; i < TILE_DIM; i++)
      sum += (float) aTile[threadIdx.y][i] * (float) bTile[i][threadIdx.x];
    __syncthreads();   // all threads have read the tiles before the next ones overwrite them
  }
  C[row*O + col] = sum;
}
```

The first `__syncthreads()` waits for the tiles to be loaded, and the second one for all threads to finish reading them, before the next iteration overwrites them. The kernel is launched on a grid of $$O/32 \times M/32$$ thread blocks of $$32 \times 32$$ threads, one thread per value of $$C$$:

```cpp
dim3 threads(TILE_DIM, TILE_DIM), tiles(O / TILE_DIM, M / TILE_DIM);
tiledMultiply<<<tiles, threads>>>(A, B, C, N, O);
```

Each value of $$A$$ and $$B$$ is now read from global memory once per thread block, i.e. $$O/32$$ and $$M/32$$ times, with coalesced reads, while `simpleMultiply` reads each value of $$B$$ once per row of $$C$$, i.e. $$M$$ times. On **[GPU]**, for $$M = N = O = 4096$$, `tiledMultiply` computes **[x]** TFLOP/s, and cuBLAS **[y]** TFLOP/s (`matmul.cu`).

### On the Tensor Cores

Each thread of these kernels computes one multiply-add per instruction on the CUDA cores of the SM. In 2017, the [Volta](https://en.wikipedia.org/wiki/Volta_(microarchitecture)) GPUs of NVIDIA added **Tensor Cores**, units that multiply two small matrices and add the result to a third one in a single operation, i.e. a matrix [multiply-accumulate](https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation). The first ones multiplied $$4 \times 4$$ matrices of [fp16](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) values with sums in fp32, and the Ampere GPUs (2020) added bf16 values, among other types. As an example, the Tensor Cores of an A100 compute 312 TFLOP/s in bf16, 16 times the 19.5 TFLOP/s of its fp32 CUDA cores ([NVIDIA Ampere Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth)).

CUDA exposes the Tensor Cores to a kernel through its warp matrix functions (`nvcuda::wmma`), with which the 32 threads of a warp load two tiles of $$16 \times 16$$ values into *fragments*, multiply them (`mma_sync`) and store the result. A fast kernel needs much more than that, such as the tiling above, asynchronous copies from global to shared memory and layouts of shared memory without bank conflicts. So the GPT code of this post calls the matrix multiplication of cuBLAS, which uses the Tensor Cores for these types:

```cpp
// C (M x O) = A (M x N) B (N x O) in row-major order, on the Tensor Cores, with fp32 accumulation. cuBLAS reads
// matrices in column-major order, where the row-major A, B and C are A^T, B^T and C^T, so it computes C^T = B^T A^T
void matmul(const dt* A, const dt* B, dt* C, int M, int N, int O) {
  float alpha = 1, beta = 0;
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, O, M, N, &alpha, B, CUDA_R_16BF, O, A, CUDA_R_16BF, N,
               &beta, C, CUDA_R_16BF, O, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}
```

The function `cublasGemmEx` computes $$C = \alpha AB + \beta C$$, here with $$\alpha = 1$$ and $$\beta = 0$$, on inputs and an output in bf16 (`CUDA_R_16BF`), with sums in fp32 (`CUBLAS_COMPUTE_32F`). However, cuBLAS follows the column-major order of Fortran, in which a matrix stored in row-major order reads as its transpose. Instead of transposing the matrices, the call swaps $$A$$ and $$B$$: in column-major order, it computes $$C^T = B^T A^T$$, of size $$O \times M$$, which is $$C$$ in row-major order. The arguments `O`, `N` and `O` after `B`, `A` and `C` are their *leading dimensions*, i.e. the number of values between the starts of two consecutive rows.

### On several GPUs: the ring algorithm

When the matrices do not fit the memory of one GPU, or to compute faster, the product is split among $$p$$ GPUs. The post [Distributed Matrix Multiplication and Database Joins]({{ site.baseurl }}{% post_url 2014-04-02-distributed-matmul-dbs %}#shifting-one-matrix-the-ring-algorithm) describes the **ring algorithm**, in which the rank $$r$$ holds the row block $$r$$ of $$A$$, $$B$$ and $$C$$, and computes its rows of $$C$$ as:

$$
C_r = \sum_{k=0}^{p-1} A_{r,k} B_k
$$

where $$A_{r,k}$$ is the tile $$k$$ of the row block $$A_r$$ of $$A$$, of size $$M/p \times N/p$$, and $$B_k$$ is the row block $$k$$ of $$B$$, of size $$N/p \times O$$. Only $$B$$ moves: at each of the $$p$$ steps, each rank multiplies the tile of $$A$$ that matches its current block of $$B$$, adds the result to $$C_r$$, and passes the block of $$B$$ to the previous rank of the ring.

The communication of a step does not depend on its product, so the two overlap with [double buffering](https://en.wikipedia.org/wiki/Multiple_buffering). While cuBLAS multiplies the block in `B_cur`, [NCCL](https://developer.nvidia.com/nccl), the collective communication library of NVIDIA, sends this block to the previous rank, and receives the next one from the next rank into `B_next`. They run on two CUDA **streams**, i.e. queues of operations that run in order, where the operations of different streams may run at the same time:

```cpp
// C_r (M/p x O) = A_r (M/p x N) B (N x O) on the rank r of a ring of p ranks, where the rank r holds the row blocks r
// of A and C, and starts with the row block r of B in B_cur. C accumulates the p products in fp32
void ring_matmul(const dt* A, dt* B_cur, dt* B_next, float* C, int M, int N, int O, int r, int p,
                 cublasHandle_t handle, ncclComm_t comm, cudaStream_t comm_stream) {
  int m = M / p, n = N / p;           // rows of the blocks of A and C, and of the blocks of B
  for (int step = 0; step < p; step++) {
    int k = (r + step) % p;           // the rank holds the row block k of B at this step
    if (step < p - 1) {               // it sends this block to the rank r-1, and receives the next one from the rank r+1
      ncclGroupStart();
      ncclSend(B_cur, (size_t) n*O, ncclBfloat16, (r - 1 + p) % p, comm, comm_stream);
      ncclRecv(B_next, (size_t) n*O, ncclBfloat16, (r + 1) % p, comm, comm_stream);
      ncclGroupEnd();
    }
    // C_r = C_r + A_rk B_k, where A_rk is the column block k of A_r (n columns, in rows of N values)
    float alpha = 1, beta = step == 0 ? 0 : 1;
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, O, m, n, &alpha, B_cur, CUDA_R_16BF, O, A + k*n, CUDA_R_16BF, N,
                 &beta, C, CUDA_R_32F, O, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    cudaDeviceSynchronize();          // waits for the product and for the transfers of this step
    std::swap(B_cur, B_next);
  }
}
```

The index `k` follows the block of $$B$$ that the rank holds: it starts with its own block, $$B_r$$, then receives $$B_{r+1}$$ from the next rank, and so on. The handle of cuBLAS runs on a compute stream, set with `cublasSetStream`, and NCCL on `comm_stream`. Note that the sends and the receives of a step are grouped (`ncclGroupStart` and `ncclGroupEnd`), so that NCCL runs them together, as each rank sends and receives at the same time. Also, $$C_r$$ accumulates the $$p$$ products in fp32 (`CUDA_R_32F`), so that its partial sums are not rounded to bf16 at every step.

A step computes $$2 \frac{M}{p} \frac{N}{p} O$$ FLOPs and sends $$\frac{N}{p} O$$ values, i.e. $$2M/p$$ FLOPs per value sent, so a large $$M$$ hides the communication behind the computation. The program `ring_matmul.cu` runs one CPU thread per GPU of a node. On **[p]** GPUs, the ring multiplies two matrices of $$8192 \times 8192$$ values in **[x]** ms, and one GPU in **[y]** ms.

## The GPT block

The rest of this post implements the GPT model of the post [Building a GPT model in PyTorch from scratch]({{ site.baseurl }}{% post_url 2023-02-28-gptlite %}), called GPTlite. The model embeds its input tokens, runs them through $$L$$ identical blocks, and maps the output of the last block to the logits of the next token. The blocks hold most of the parameters and of the computation, and almost all of them in large models, so this post implements only the blocks, and leaves out the embeddings and the final layer normalization and linear layer. The block is illustrated below:

{: .picture}
<img width="18.0%" height="18.0%" src="{{ site.assets }}/gpt-cpp-cuda/gpt_block.svg"/>

{: .caption}
The GPT block: a masked multi-head attention and a feed-forward network, each after a layer normalization and inside a skip connection. The model stacks $$L$$ blocks. Adapted from: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

The block takes a batch of $$B$$ sequences of $$S$$ tokens, each token an embedding of $$E$$ values, and computes:

$$
x = x + \text{Attention}(\text{LayerNorm}_1(x)), \quad x = x + \text{FFN}(\text{LayerNorm}_2(x))
$$

where $$\text{Attention}$$ is the masked multi-head attention and $$\text{FFN}$$ the feed-forward network. The next sections implement these four parts in LibTorch, in C++ loops on the CPU and in CUDA on the GPU. The PyTorch version, in `gpt.py`, is the code of the GPTlite post, with the change of the attention described below, and LibTorch follows it line by line.

### Dimensions and data types

All implementations share the header `gpt.h`, with the dimensions of GPT-3 Small, the smallest model of the paper [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3, Table 2.1 in the paper):

```cpp
// the dimensions of GPT-3 Small, Table 2.1 of "Language Models are Few-Shot Learners", Brown et al., 2020
constexpr int L = 12;            // number of blocks (n_layers)
constexpr int E = 768;           // size of the embeddings (d_model)
constexpr int H = 12;            // number of attention heads (n_heads)
constexpr int D = E / H;         // size of each attention head (d_head)
constexpr int S = 2048;          // sequence length (n_ctx)
constexpr int B = 8;             // batch size of the benchmark, in sequences
constexpr float eps = 1e-5f;     // epsilon of the layer normalization, as in PyTorch
```

Only the dimensions come from GPT-3 Small. The block is the one of GPTlite, with a ReLU where GPT-3 uses a GELU, and with dense attention in all blocks, where GPT-3 alternates dense and locally banded sparse attention. Also, the batch size $$B$$ is the number of sequences of the benchmark, not the batch size of the paper (0.5M tokens), which is a setting of the training.

On the GPU, the values are stored in bf16 and summed in fp32 ([mixed precision](https://en.wikipedia.org/wiki/Mixed-precision_arithmetic)), as PyTorch does in a bf16 model. The Tensor Cores multiply bf16 values and add them in fp32, and the kernels of the softmax and of the layer normalization compute in fp32 and store bf16. On the CPU, all values are fp32, which keeps the code short. Each file sets the type `dt` of its values:

```cpp
using dt = float;            // gpt_cpu.cpp
using dt = __nv_bfloat16;    // gpt_cuda.cu
```

Moreover, to compare the outputs of the implementations, the input and the parameters are not random: their values come from an integer hash of their index, in the same order in all implementations (`init_value` in `gpt.h`). Each program prints the first three values of the output of the first sequence, and the mean of the absolute values of this output. In fp32, the C++ code on the CPU prints the same values as PyTorch (0.7254, 0.0187 and 1.1980, with a mean absolute value of 0.6332), and PyTorch in bf16 on the CPU prints 0.7188, 0.0156 and 1.1953, with the same mean.

### Feed-forward network

The simplest part of the block is the **feed-forward network** (FFN), made of matrix multiplications and element-wise operations, and emphasized below:

{: .picture}
<img width="18.0%" height="18.0%" src="{{ site.assets }}/gpt-cpp-cuda/gpt_block_ffn.svg"/>

{: .caption}
The feed-forward network of the GPT block, emphasized in red. Adapted from: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

The FFN is a two-layer [multilayer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron) applied to each token. Its output is computed as:

$$
\text{FFN}(x) = \text{ReLU}(x W_1 + b_1) \, W_2 + b_2
$$

where $$x$$ holds the $$B \cdot S$$ tokens of the batch in rows of $$E$$ values, $$W_1$$ of size $$E \times 4E$$ and $$W_2$$ of size $$4E \times E$$ are the weights of the two linear layers, $$b_1$$ and $$b_2$$ are their biases, added to each row, and the [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)), $$\max(z, 0)$$, applies to each value $$z$$.

GPTlite also applies a [dropout](https://en.wikipedia.org/wiki/Dropout_(neural_networks)) after the FFN, after the attention and to the attention weights. This post does not implement it, as dropout is the identity at inference. PyTorch implements the *inverted* dropout, which divides the values that it keeps by the probability of keeping them during training, so that the expected value of each activation is the same at inference, where nothing is dropped or scaled.

In LibTorch, the FFN is written as in Python:

```cpp
struct FeedForward : nn::Module {

  FeedForward() {
    net = register_module("net", nn::Sequential(nn::Linear(E, 4*E), nn::ReLU(), nn::Linear(4*E, E)));
  }

  Tensor forward(Tensor x) {
    return net->forward(x);
  }

  nn::Sequential net{nullptr};
};
```

To keep the code small and clean, all modules are defined as a `struct` and not as a `class`, so that all members are public by default instead of private. Note the `register_module` method, which is not needed in the Python implementation. Why is it needed? In practice, C++ has no runtime [reflection](https://en.wikipedia.org/wiki/Reflective_programming), so it cannot iterate over the variables of a class unless they are declared somewhere. However, LibTorch needs this iterator to iterate over the members of a module, e.g. to count its parameters, to copy its submodules to a device, or to update its weights in the optimizer.

There are two options to create this iterator, and this post uses both:
1. all modules can be kept inside a LibTorch container, such as `nn::Sequential` or `nn::ModuleList`. Applying a function to the container applies it recursively to every module inside;
2. `register_parameter`, `register_buffer` and `register_module` register the parameters, buffers or modules during initialization, and LibTorch keeps track of them internally.

On the CPU, a linear layer is the matrix multiplication of the previous section, followed by the addition of the bias to each row, and the ReLU applies the `std::max` of the standard library to each value:

```cpp
// y (M x O) = x (M x N) W (N x O) + b, where the bias b (O) is added to each row
void linear(const dt* x, const dt* W, const dt* b, dt* y, int M, int N, int O) {
  matmul(x, W, y, M, N, O);
  for (int i = 0; i < M; i++)
    for (int j = 0; j < O; j++)
      y[i*O + j] += b[j];
}

void relu(dt* x, int n) {
  for (int i = 0; i < n; i++)
    x[i] = std::max(x[i], 0.0f);
}
```

The weights `W` are stored as (in × out), so that a linear layer is the product $$xW$$, while `nn.Linear` stores the transposed weights, as (out × in), and computes $$xW^T$$. The function `init_params` of the PyTorch and LibTorch versions thus copies the transposed values.

On the GPU, a linear layer is the cuBLAS product of the previous section followed by a kernel that adds the bias, and the ReLU is a kernel of its own. Each kernel runs one thread per value, in thread blocks of 256 threads:

```cpp
constexpr int THREADS = 256;   // threads per thread block of the element-wise and row kernels

// number of thread blocks of THREADS threads for n elements, one thread per element
unsigned int blocks(long n) { return (n + THREADS - 1) / THREADS; }

// y = y + b, where the bias b (O) is added to each row of y (n values in rows of O)
__global__ void add_bias(dt* y, const dt* b, long n, int O) {
  long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = (float) y[i] + (float) b[i % O];
}

// y (M x O) = x (M x N) W (N x O) + b
void linear(const dt* x, const dt* W, const dt* b, dt* y, int M, int N, int O) {
  matmul(x, W, y, M, N, O);
  add_bias<<<blocks((long) M*O), THREADS>>>(y, b, (long) M*O, O);
}

__global__ void relu(dt* x, long n) {
  long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] = fmaxf((float) x[i], 0.0f);
}
```

The thread `i` handles the value `i`, so the 32 threads of a warp read and write 32 consecutive values, and all accesses to global memory are coalesced. Unlike the matrix multiplication, these kernels read each value once, so there is no data to share between threads, and shared memory would bring nothing: their speed is bound by the bandwidth of the HBM. The casts to `float` compute in fp32, and the assignment to a `dt` rounds the result to bf16. The launch `<<<blocks(n), THREADS>>>` starts enough thread blocks to cover the $$n$$ values, and the test `i < n` stops the threads of the last block that fall beyond them. Indices are `long`, as the attention weights of large batches hold more than $$2^{31}$$ values.

### Masked multi-head attention

The second part of the block is the masked multi-head attention, emphasized below:

{: .picture}
<img width="18.0%" height="18.0%" src="{{ site.assets }}/gpt-cpp-cuda/gpt_block_attention.svg"/>

{: .caption}
The masked multi-head attention of the GPT block, emphasized in red. Adapted from: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

As a reminder, the multi-head attention of the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) is given by:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_H) \, W^O
$$

$$
\text{where } \text{head}_h = \text{Attention}(Q W^Q_h, K W^K_h, V W^V_h)
$$

$$
\text{and } \text{Attention}(Q, K, V) = \text{softmax} \left( \frac{Q K^T}{\sqrt{D}} + \text{mask} \right) V
$$

where $$H$$ is the number of heads, $$D = E/H$$ the size of each head, $$W^Q_h$$, $$W^K_h$$ and $$W^V_h$$ the projections of the queries, keys and values of the head $$h$$, of size $$E \times D$$, and $$W^O$$ the output projection, of size $$E \times E$$. In a GPT block, the queries, the keys and the values all come from the input of the attention ($$Q = K = V = x$$, i.e. a *self-attention*), and each projection also adds a bias. The mask hides the next tokens of a sequence: it is $$-\infty$$ above the diagonal and 0 elsewhere, so that the softmax gives a weight of 0 to the tokens $$j > i$$, and the token $$i$$ attends only to the tokens 0 to $$i$$.

The code computes the projections of the $$H$$ heads together, as one linear layer of size $$E \times E$$ whose output holds the $$H$$ heads of $$D$$ values side by side in each row, instead of the $$H$$ modules of size $$E \times D$$ of the GPTlite post. The result is the same, with one large matrix multiplication instead of $$H$$ small ones. Similarly, the concatenation of the heads is a matrix with the outputs of the $$H$$ heads side by side in each row.

In LibTorch, the attention follows the Python code:

```cpp
struct MultiHeadAttention : nn::Module {

  MultiHeadAttention() {
    query = register_module("query", nn::Linear(E, E));
    key   = register_module("key", nn::Linear(E, E));
    value = register_module("value", nn::Linear(E, E));
    proj  = register_module("proj", nn::Linear(E, E));
    tril  = register_buffer("tril", torch::tril(torch::ones({S, S})));
  }

  Tensor forward(Tensor x) {
    int B = x.size(0), S = x.size(1);
    // the H heads, from (B, S, E) to (B, H, S, D)
    Tensor q = query(x).view({B, S, H, D}).transpose(1, 2);
    Tensor k = key(x).view({B, S, H, D}).transpose(1, 2);
    Tensor v = value(x).view({B, S, H, D}).transpose(1, 2);
    Tensor wei = torch::matmul(q, k.transpose(-2, -1)) * std::pow(D, -0.5);   // (B, H, S, S)
    wei = wei.masked_fill(tril.slice(0, 0, S).slice(1, 0, S) == 0, -INFINITY);
    wei = torch::softmax(wei, -1);
    Tensor out = torch::matmul(wei, v);                                        // (B, H, S, D)
    return proj(out.transpose(1, 2).reshape({B, S, E}));                      // (B, S, E)
  }

  nn::Linear query{nullptr}, key{nullptr}, value{nullptr}, proj{nullptr};
  Tensor tril;
};
```

The linear layer `query` computes the queries of all heads, and `view` and `transpose` turn its output of shape (B, S, E) into a tensor of shape (B, H, S, D), so that `torch::matmul` multiplies the queries and the keys of each head of each sequence. The local `B` and `S` hide the constants of `gpt.h`, so that the module also runs on shorter sequences, as in Python. The buffer `tril` holds the lower-triangular matrix of the mask, and `masked_fill` writes $$-\infty$$ where it is 0. It is registered with `register_buffer` because it is a state of the module but not a parameter, i.e. it moves to the GPU with the module, but has no gradients.

The members are declared as `nn::Linear query{nullptr}`, and this is not accidental: all LibTorch modules hold a shared pointer to the implementation of their class. E.g. the linear layer `nn::Linear` is a wrapper around `std::shared_ptr<nn::LinearImpl>`, where `nn::LinearImpl` is the implementation, so all `torch::nn` modules can be passed by value directly. Because of this, the documentation suggests initializing modules with `nullptr` as the default value of the pointer, and creating the implementation later. The alternative of not initializing the pointer would call the default constructor, `Linear()`, which is not defined, and lead to a compilation error. Finally, LibTorch has no named arguments as in Python (e.g. `bias=False`): they are wrapped inside an options object, as in `nn::Linear(nn::LinearOptions(E, E).bias(false))`.

On the CPU, the attention is computed token by token, without storing the $$S \times S$$ matrix of the attention weights. For each sequence `b`, head `h` and token `i`, the code computes the scores of the tokens `j` from 0 to `i` (the mask), their [softmax](https://en.wikipedia.org/wiki/Softmax_function), and the sum of the values weighted by the softmax. The softmax of a vector of scores $$z$$ is given by:

$$
\text{softmax}(z)_j = \frac{e^{z_j}}{\sum_l e^{z_l}} = \frac{e^{z_j - m}}{\sum_l e^{z_l - m}}
$$

where $$m = \max_l z_l$$ is the largest score. The shift by $$m$$ does not change the result, but keeps the exponentials from overflowing ([numerical stability](https://en.wikipedia.org/wiki/Numerical_stability)). The standard library has the pieces of the softmax, `std::max_element`, `std::exp` and `std::accumulate`, and the loops do the same in place:

```cpp
// masked multi-head attention softmax(q k^T / sqrt(D) + mask) v of each head of B sequences of S tokens, where
// q, k, v and the output o are (B*S x E), with the H heads of D values side by side in each row
void attention(const dt* q, const dt* k, const dt* v, dt* o, int B, int S) {
  #pragma omp parallel for collapse(2)
  for (int b = 0; b < B; b++)
    for (int h = 0; h < H; h++) {
      std::vector<dt> w(S);                    // attention weights of a token
      for (int i = 0; i < S; i++) {
        const dt* qi = q + (b*S + i)*E + h*D;
        dt max_val = -INFINITY, sum = 0;
        for (int j = 0; j <= i; j++) {         // the mask: token i attends to the tokens 0 to i only
          const dt* kj = k + (b*S + j)*E + h*D;
          w[j] = 0;
          for (int d = 0; d < D; d++)
            w[j] += qi[d] * kj[d];
          w[j] /= std::sqrt((dt) D);
          max_val = std::max(max_val, w[j]);
        }
        for (int j = 0; j <= i; j++) {         // softmax, shifted by the maximum for numerical stability
          w[j] = std::exp(w[j] - max_val);
          sum += w[j];
        }
        dt* oi = o + (b*S + i)*E + h*D;
        for (int d = 0; d < D; d++)
          oi[d] = 0;
        for (int j = 0; j <= i; j++) {
          const dt* vj = v + (b*S + j)*E + h*D;
          for (int d = 0; d < D; d++)
            oi[d] += w[j] / sum * vj[d];
        }
      }
    }
}
```

The head `h` of the token `i` of the sequence `b` starts at `(b*S + i)*E + h*D`, as each row holds the $$H$$ heads of a token side by side. The clause `collapse(2)` merges the loops over the sequences and the heads, so that OpenMP shares the $$B \cdot H$$ pairs among the threads, and each pair has its own vector `w` of weights. Note that the mask costs nothing here: the loops skip the masked tokens, and do half of the work of the full $$S \times S$$ matrix.

On the GPU, the attention of all tokens is computed at once, as in PyTorch, in five steps: the scores of each head with a batched matrix multiplication, their scale by $$1/\sqrt{D}$$, the mask, the softmax of each row, and the sum of the values weighted by the softmax, with a second batched multiplication:

```cpp
// masked multi-head attention of B sequences of S tokens, where q, k, v and the output o are (B*S x E), with the
// H heads of D values side by side in each row, and wei holds the attention weights (B*H x S x S)
void attention(const dt* q, const dt* k, const dt* v, dt* o, dt* wei, int B, int S) {
  float alpha = 1, beta = 0;
  long n = (long) B*H*S*S;
  // wei = q k^T for each head, where the heads of a sequence start D values apart, and their rows are E values apart
  for (int b = 0; b < B; b++)
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, S, S, D, &alpha,
      k + (long) b*S*E, CUDA_R_16BF, E, D, q + (long) b*S*E, CUDA_R_16BF, E, D, &beta,
      wei + (long) b*H*S*S, CUDA_R_16BF, S, (long) S*S, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  scale<<<blocks(n), THREADS>>>(wei, 1 / sqrtf(D), n);
  causal_mask<<<blocks(n), THREADS>>>(wei, S, n);
  softmax<<<B*H*S, THREADS, (S + THREADS) * sizeof(float)>>>(wei, S);
  // o = wei v for each head, written in place into the columns of the head
  for (int b = 0; b < B; b++)
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, D, S, S, &alpha,
      v + (long) b*S*E, CUDA_R_16BF, E, D, wei + (long) b*H*S*S, CUDA_R_16BF, S, (long) S*S, &beta,
      o + (long) b*S*E, CUDA_R_16BF, E, D, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}
```

The function `cublasGemmStridedBatchedEx` runs `H` matrix multiplications in one call, where the matrices of two consecutive heads start a fixed number of values apart, the *stride*. The stride is `D` values for `q`, `k` and `v`, whose heads sit side by side in rows of `E` values (their leading dimension), and `S*S` values for the weights `wei`. So cuBLAS reads the heads in place, without the copies that PyTorch makes to multiply its transposed views, and writes the output of each head into its columns of `o`, which is then the concatenation of the heads. The heads of two sequences are not one stride apart, so the code calls cuBLAS once per sequence. With the swap of the column-major order, the option `CUBLAS_OP_T` makes the first call compute the scores $$q_h k_h^T$$ of each head $$h$$.

The scale and the mask are element-wise kernels, as the bias and the ReLU:

```cpp
__global__ void scale(dt* x, float s, long n) {
  long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] = (float) x[i] * s;
}

// the mask of the attention weights x (B*H x S x S): the weight of a token j > i in the row i is -infinity
__global__ void causal_mask(dt* x, int S, long n) {
  long idx = (long) blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n && idx % S > idx / S % S)
    x[idx] = -INFINITY;
}
```

where `idx % S` is the column `j` of a weight, and `idx / S % S` its row `i`. The softmax is different: it works on rows of $$S$$ weights, and reads each value three times, to find the maximum, to sum the exponentials and to normalize them. This is where shared memory helps: one thread block of 256 threads processes one row, reads it once from global memory into shared memory, and runs the three passes on it. The maximum and the sum are *reductions* over the values of all threads, computed by a [reduction tree](https://en.wikipedia.org/wiki/Reduction_operator) in shared memory:

```cpp
// the sum, or the maximum, of the values v of all the threads of a thread block, with a tree in shared memory
__device__ float block_reduce(float v, float* buf, bool take_max) {
  buf[threadIdx.x] = v;
  __syncthreads();
  for (unsigned int s = blockDim.x / 2; s > 0; s /= 2) {
    if (threadIdx.x < s)
      buf[threadIdx.x] = take_max ? fmaxf(buf[threadIdx.x], buf[threadIdx.x + s]) : buf[threadIdx.x] + buf[threadIdx.x + s];
    __syncthreads();
  }
  float result = buf[0];
  __syncthreads();   // all threads read buf[0] before the next reduction writes buf
  return result;
}

// softmax of each row of x (rows of N values), with one thread block per row, read once into shared memory
__global__ void softmax(dt* x, int N) {
  extern __shared__ float row[];   // the N values of the row, then THREADS values for the reductions
  float* buf = row + N;
  dt* xi = x + (long) blockIdx.x * N;
  float max_val = -INFINITY, sum = 0;
  for (int j = threadIdx.x; j < N; j += blockDim.x) {
    row[j] = xi[j];
    max_val = fmaxf(max_val, row[j]);
  }
  max_val = block_reduce(max_val, buf, true);
  for (int j = threadIdx.x; j < N; j += blockDim.x) {
    row[j] = expf(row[j] - max_val);
    sum += row[j];
  }
  sum = block_reduce(sum, buf, false);
  for (int j = threadIdx.x; j < N; j += blockDim.x)
    xi[j] = row[j] / sum;
}
```

First, each thread reads the values `j`, `j + 256`, `j + 512`, etc. of the row, so that the reads of a warp are coalesced, and keeps the maximum of its values. Then, `block_reduce` combines the values of the 256 threads in $$\log_2 256 = 8$$ steps. At each step, each thread of the first half of the active threads adds (or takes the maximum of) its value and the one of a thread of the second half, until `buf[0]` holds the result of the block. Each step ends with `__syncthreads()`, as the next step reads values written by other warps. The array `row` is *dynamic* shared memory (`extern __shared__`), whose size is set at launch by the third parameter of `<<<...>>>`: $$S$$ floats for the row and 256 for the reductions, i.e. 9 KB for $$S = 2048$$.

### Skip connection

The third part of the block is the skip connection, which adds the input of the attention and of the FFN to their output, as emphasized below:

{: .picture}
<img width="18.0%" height="18.0%" src="{{ site.assets }}/gpt-cpp-cuda/gpt_block_skip.svg"/>

{: .caption}
The skip connections of the GPT block, emphasized in red. Adapted from: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

A [skip connection](https://en.wikipedia.org/wiki/Residual_neural_network) (or residual connection) computes $$x + f(x)$$, where $$f$$ is the attention or the FFN. It comes from the residual networks (ResNets) of the paper [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), where it lets the gradients of very deep networks flow through the identity during training. In LibTorch, it is the sum of two tensors in the forward pass of the block, shown later. On the CPU, the standard library would write it as `std::transform(x, x + n, y, x, std::plus<>())`, and the loop does the same:

```cpp
// the skip connection x = x + y
void add(dt* x, const dt* y, int n) {
  for (int i = 0; i < n; i++)
    x[i] += y[i];
}
```

and on the GPU, it is one more element-wise kernel:

```cpp
// the skip connection x = x + y
__global__ void add(dt* x, const dt* y, long n) {
  long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] = (float) x[i] + (float) y[i];
}
```

### Layer normalization

The last part of the block is the layer normalization, applied at the input of the attention and of the FFN, as emphasized below:

{: .picture}
<img width="18.0%" height="18.0%" src="{{ site.assets }}/gpt-cpp-cuda/gpt_block_layernorm.svg"/>

{: .caption}
The layer normalizations of the GPT block, emphasized in red. Adapted from: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

The [layer normalization](https://en.wikipedia.org/wiki/Normalization_(machine_learning)#Layer_normalization), introduced by the paper [Layer Normalization](https://arxiv.org/abs/1607.06450), normalizes each token to a mean of 0 and a variance of 1, and then scales and shifts it with two learnt vectors of size $$E$$. For a token $$x$$ of $$E$$ values, the output is computed as:

$$
y_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + \epsilon}} \, \gamma_j + \beta_j
$$

where $$\mu = \frac{1}{E} \sum_{j} x_j$$ and $$\sigma^2 = \frac{1}{E} \sum_{j} (x_j - \mu)^2$$ are the mean and the variance of the values of the token, $$\gamma$$ and $$\beta$$ are the learnt weight and bias, and the small $$\epsilon$$ avoids a division by zero. GPT-2 and GPT-3 apply it at the input of the attention and of the FFN (*pre-normalization*), while the original Transformer applies it after each skip connection.

In LibTorch, the layer normalizations are modules of the block, shown in the next section, with a subtle difference from Python. By design, `LayerNorm` takes the list of the normalized dimensions. In Python, a single `int` normalizes the last dimension of the input, of that size. However, in C++, `LayerNorm` has no constructor with a single integer argument, so the general constructor is used, with the dimension passed as a singleton list: `nn::LayerNorm(std::vector<int64_t> {E})`.

On the CPU, the standard library would compute the mean with `std::accumulate` and the sum of squares with `std::inner_product`, and the loops over the values of each token do the same:

```cpp
// layer normalization y = (x - mean) / sqrt(var + eps) * w + b of each row of x (M x N)
void layernorm(const dt* x, const dt* w, const dt* b, dt* y, int M, int N) {
  #pragma omp parallel for
  for (int i = 0; i < M; i++) {
    const dt* xi = x + i*N;
    dt mean = 0, var = 0;
    for (int j = 0; j < N; j++)
      mean += xi[j];
    mean /= N;
    for (int j = 0; j < N; j++)
      var += (xi[j] - mean) * (xi[j] - mean);
    var /= N;
    for (int j = 0; j < N; j++)
      y[i*N + j] = (xi[j] - mean) / std::sqrt(var + eps) * w[j] + b[j];
  }
}
```

where `w` and `b` are $$\gamma$$ and $$\beta$$, the `weight` and the `bias` of `nn.LayerNorm`. Note that the variance is computed in a second pass over the values, as the mean of $$(x_j - \mu)^2$$, and not in a single pass as the mean of $$x_j^2$$ minus $$\mu^2$$. The single pass subtracts two large and close numbers when the mean is large compared to the spread of the values, and loses precision ([algorithms for calculating variance](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)).

On the GPU, the layer normalization works as the softmax: one thread block per token, which reads the $$E$$ values of the token once into shared memory, and computes their mean and their variance with two reductions:

```cpp
// layer normalization of each row of x (rows of N values), with one thread block per row, read once into shared memory
__global__ void layernorm(const dt* x, const dt* w, const dt* b, dt* y, int N) {
  extern __shared__ float row[];   // the N values of the row, then THREADS values for the reductions
  float* buf = row + N;
  const dt* xi = x + (long) blockIdx.x * N;
  float mean = 0, var = 0;
  for (int j = threadIdx.x; j < N; j += blockDim.x) {
    row[j] = xi[j];
    mean += row[j];
  }
  mean = block_reduce(mean, buf, false) / N;
  for (int j = threadIdx.x; j < N; j += blockDim.x)
    var += (row[j] - mean) * (row[j] - mean);
  var = block_reduce(var, buf, false) / N;
  dt* yi = y + (long) blockIdx.x * N;
  for (int j = threadIdx.x; j < N; j += blockDim.x)
    yi[j] = (row[j] - mean) * rsqrtf(var + eps) * (float) w[j] + (float) b[j];
}
```

### The block

Finally, the block runs the four parts in turn. In LibTorch, it is a module that holds the two layer normalizations, the attention and the FFN:

```cpp
struct Block : nn::Module {

  Block() {
    ln1  = register_module("ln1", nn::LayerNorm(std::vector<int64_t> {E}));
    sa   = register_module("sa", std::make_shared<MultiHeadAttention>());
    ln2  = register_module("ln2", nn::LayerNorm(std::vector<int64_t> {E}));
    ffwd = register_module("ffwd", std::make_shared<FeedForward>());
  }

  Tensor forward(Tensor x) {
    x = x + sa->forward(ln1(x));
    x = x + ffwd->forward(ln2(x));
    return x;
  }

  nn::LayerNorm ln1{nullptr}, ln2{nullptr};
  std::shared_ptr<MultiHeadAttention> sa;
  std::shared_ptr<FeedForward> ffwd;
};
```

The attention and the FFN are created with `std::make_shared`, as `register_module` takes a shared pointer to a module, the same pointer that the holders of `torch::nn` wrap. The main program then stacks the $$L$$ blocks in an `nn::Sequential`, initializes their parameters with the values of `gpt.h`, and moves the model to the GPU in bf16:

```cpp
torch::Device device(torch::kCUDA);
nn::Sequential model;
for (int l = 0; l < L; l++)
  model->push_back(std::make_shared<Block>());
init_params(model);
model->to(device, torch::kBFloat16);
model->eval();
Tensor x = init_values(0, batch*S*E, 1).view({batch, S, E}).to(device, torch::kBFloat16);

c10::InferenceMode guard;   // the C++ equivalent of torch.inference_mode() in Python
```

As an important remark, LibTorch does not include a C++ equivalent to `torch.set_default_device`, so the input and the model are moved to the GPU explicitly. And because every parameter, buffer and module was registered, `model->to(device, torch::kBFloat16)` recursively copies all the contents of the model to the GPU, in bf16. Also, containers in C++ are declared for a fixed element type: `nn::Sequential` stores elements of type `nn::Module`, so `init_params` casts each of them dynamically to its derived type with `module->as<Block>()` before it accesses the members of the block. The guard `c10::InferenceMode` turns off the tracking of gradients in its scope, as `torch.inference_mode()` does in Python.

On the CPU and on the GPU, the parameters of a block are pointers to arrays, and the intermediate results go to buffers allocated once and reused by all blocks:

```cpp
// parameters of a block, where the weights W of the linear layers are stored as (in x out)
struct Block {
  dt *ln1_w, *ln1_b, *Wq, *bq, *Wk, *bk, *Wv, *bv, *Wo, *bo;   // layer normalization and attention
  dt *ln2_w, *ln2_b, *W1, *b1, *W2, *b2;                        // layer normalization and feed-forward network
};

// intermediate results of a block
struct Buffers {
  dt *h, *q, *k, *v, *a, *f;
};
```

The forward pass of a block on the CPU then calls the functions of the previous sections in turn, on the $$M = B \cdot S$$ tokens of the batch:

```cpp
// forward pass of a block on x (B*S x E): x = x + attention(layernorm(x)), then x = x + ffn(layernorm(x))
void block_forward(const Block& p, dt* x, const Buffers& t, int B, int S) {
  int M = B*S;
  layernorm(x, p.ln1_w, p.ln1_b, t.h, M, E);
  linear(t.h, p.Wq, p.bq, t.q, M, E, E);
  linear(t.h, p.Wk, p.bk, t.k, M, E, E);
  linear(t.h, p.Wv, p.bv, t.v, M, E, E);
  attention(t.q, t.k, t.v, t.a, B, S);
  linear(t.a, p.Wo, p.bo, t.h, M, E, E);
  add(x, t.h, M*E);
  layernorm(x, p.ln2_w, p.ln2_b, t.h, M, E);
  linear(t.h, p.W1, p.b1, t.f, M, E, 4*E);
  relu(t.f, M*4*E);
  linear(t.f, p.W2, p.b2, t.h, M, 4*E, E);
  add(x, t.h, M*E);
}
```

The CUDA version is the same, with kernel launches, and with one more buffer, `wei`, for the attention weights of the batch. The layer normalization runs one thread block per token, with the shared memory for the $$E$$ values of a token and for the reductions:

```cpp
void block_forward(const Block& p, dt* x, const Buffers& t, int B, int S) {
  int M = B*S;
  size_t smem = (E + THREADS) * sizeof(float);
  layernorm<<<M, THREADS, smem>>>(x, p.ln1_w, p.ln1_b, t.h, E);
  linear(t.h, p.Wq, p.bq, t.q, M, E, E);
  linear(t.h, p.Wk, p.bk, t.k, M, E, E);
  linear(t.h, p.Wv, p.bv, t.v, M, E, E);
  attention(t.q, t.k, t.v, t.a, t.wei, B, S);
  linear(t.a, p.Wo, p.bo, t.h, M, E, E);
  add<<<blocks((long) M*E), THREADS>>>(x, t.h, (long) M*E);
  layernorm<<<M, THREADS, smem>>>(x, p.ln2_w, p.ln2_b, t.h, E);
  linear(t.h, p.W1, p.b1, t.f, M, E, 4*E);
  relu<<<blocks((long) M*4*E), THREADS>>>(t.f, (long) M*4*E);
  linear(t.f, p.W2, p.b2, t.h, M, 4*E, E);
  add<<<blocks((long) M*E), THREADS>>>(x, t.h, (long) M*E);
}
```

And that is all: the main program initializes the parameters of the $$L$$ blocks, copies them to the GPU, and runs the blocks on the input, 5 times to warm up and then 20 times under the clock:

```cpp
std::chrono::steady_clock::time_point start;
for (int it = 0; it < warmup_iters + benchmark_iters; it++) {
  if (it == warmup_iters) {
    cudaDeviceSynchronize();
    start = std::chrono::steady_clock::now();
  }
  cudaMemcpy(x, x0, M*E * sizeof(dt), cudaMemcpyDeviceToDevice);
  for (const Block& p : blocks)
    block_forward(p, x, t, batch, S);
}
cudaDeviceSynchronize();
double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
```

Each iteration first copies the input `x0` into `x`, as the blocks update `x` in place. A kernel launch returns before the kernel ends, as the GPU runs the kernels asynchronously from the CPU. Thus, the program waits for the GPU (`cudaDeviceSynchronize`) before it reads the clock, and the PyTorch and LibTorch versions do the same with `torch.cuda.synchronize()` and `torch::cuda::synchronize()`. The first iterations are not timed, as they hold one-time costs, e.g. the allocations of memory, the choice of the algorithms of cuBLAS and, with `torch.compile`, the compilation.

## PyTorch 2, kernel fusion and torch.compile

The CUDA code above launches one kernel per operation, as PyTorch does. This section describes why PyTorch moved back to Python, and how `torch.compile` fuses these kernels.

### Change of philosophy: from Python to C++ to Python

Initial releases of PyTorch were mostly written in Python. Until the release of PyTorch 2.x, the belief was that "to keep eager execution at high-performance, we’ve had to move substantial parts of PyTorch internals into C++". In practice, Python has the overhead of the runtime itself, dynamic typing, interpreted code, etc. So moving the PyTorch API to C++ and using Python as a *thin* layer that calls the compiled C++ implementations seemed logical.

With PyTorch 2.x, the developers moved in the opposite direction, claiming that the "philosophy on PyTorch has always been to keep flexibility and hackability our top priority, and performance as a close second". They add that "moving internals into C++ makes them less hackable and increases the barrier of entry for code contributions". So, in practice, the 2000+ operations of the C++ library (counting overloads) are decomposed into a smaller set, and the "goal is to provide a primitive and stable set of ~250 operators with simplified semantics, called PrimTorch".

In brief, to favour PyTorch contributors who prefer Python over C++, the new compiler of PyTorch 2.x (TorchDynamo, AOTAutograd, PrimTorch and TorchInductor) is implemented in Python only. Also, the operators that a backend needs to implement are limited to a few hundred. Hardware vendors can then focus on their specific implementation of that subset of operators, while the Python code decomposes the higher-level operations into it.

### Kernel fusion

Every PyTorch operation launches one or more GPU kernels. Each launch costs a few microseconds of CPU and driver time, and each kernel reads its inputs from global memory and writes its outputs back. For large batches, the launches are hidden, as the CPU queues the next kernels while the GPU runs the current one. However, for small models and batches, the kernels are short, and the GPU spends much of its time idle, waiting for the next launch. The memory round-trips cost time at every batch size: an element-wise kernel computes a few operations per value, so its runtime is the time to read and write its tensors in global memory.

**Kernel fusion** merges consecutive operations into a single kernel, so that intermediate results stay in registers or on-chip memory instead of going through GPU memory. Typical candidates are the chains of element-wise operations after a matrix multiplication (bias, activation function, dropout, residual addition) and the normalization layers. [FlashAttention](https://arxiv.org/abs/2205.14135) is an extreme example of fusion, as it fuses the whole attention into a single kernel (see the post on [faster attention]({{ site.baseurl }}{% post_url 2025-12-30-inference-attention %}#flashattention)).

### torch.compile

The other big announcement of PyTorch 2.x was `torch.compile`, which "makes PyTorch code run faster by JIT-compiling PyTorch code into optimized kernels, all while requiring minimal code changes" (from the [tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)). TorchDynamo captures the graph of the model from the Python bytecode, and TorchInductor generates fused kernels, written in [Triton](https://triton-lang.org/main/index.html) for GPUs. It is a one-line change, `model = torch.compile(model)`, and a change in the shapes of the inputs can trigger a recompilation. The torch compile algorithm runs in three steps:
1. **graph acquisition** builds the execution graph of all PyTorch operations. The graph is rewritten as a graph of subgraphs: the subgraphs that can be compiled are flattened, and the others (e.g. with control flow or other unsupported Python code) fall back to eager mode;
2. **graph lowering** decomposes the previous operations into kernels that are native to the specific backend;
3. **graph compilation** converts the kernels into low-level operations that are specific to the device.

The following diagram illustrates the three steps on a function that applies a convolution, a batch normalization and a ReLU:

{: .picture}
<img width="74.5%" height="74.5%" src="{{ site.assets }}/gptlite-cpp/torch_compile.svg"/>

{: .caption}
A diagram of the three steps of `torch.compile`. Adapted from: [PyTorch 2.0 technology overview](https://pytorch.org/get-started/pytorch-2.0/#technology-overview)

Kernel fusion is complex, and there are two settings worth mentioning. The option `mode` selects the trade-off between runtime and memory in the generated graph (from the [docs](https://pytorch.org/get-started/pytorch-2.0/#user-experience) and [API](https://pytorch.org/docs/stable/generated/torch.compile.html#torch.compile)):
- `default` optimizes for large models, with low compile time and no extra memory overhead. It is "a good balance between performance and overhead";
- `reduce-overhead` reduces the Python overhead with [CUDA graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/). It is faster than the previous mode but uses some extra memory, and it helps speed up small models. CUDA graphs allow an execution graph of kernels to be defined beforehand and launched only once from the CPU, instead of having kernels launched individually;
- `max-autotune` leverages Triton-based matrix multiplications and convolutions. It enables CUDA graphs by default. It produces the fastest model, but takes a very long time to compile. `max-autotune-no-cudagraphs` is analogous, but without CUDA graphs.

The flag `fullgraph` specifies whether the compilation outputs a single graph for the whole run, or whether it can be broken into several partial graphs, i.e. whether a *graph break* is allowed. It is only relevant for users who need to squeeze out the very best performance. These two settings are specified, with their default values, as `torch.compile(model, mode='default', fullgraph=False)`.

Even after fusion, a forward pass launches many kernels from Python, one after the other. The CUDA graphs of the `reduce-overhead` and `max-autotune` modes record the whole sequence of kernel launches once, and replay it with a single launch, removing almost all the CPU overhead. The recording fixes the shapes and memory addresses of all tensors, so the inputs must have static shapes. For text generation, this requires a static KV cache (the cache of the keys and values of the previous tokens), preallocated to its maximum length.

Combining these techniques with quantization and speculative decoding, the PyTorch team's [gpt-fast](https://pytorch.org/blog/accelerating-generative-ai-2/) made the decoding of Llama-7B almost 10 times faster, in plain PyTorch. Going one step further, recent work such as [Look Ma, No Bubbles!](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) fuses a whole forward pass into a single persistent **megakernel**, removing the gaps between kernels altogether for low-latency decoding.

The kernels that TorchInductor generates for a model are printed with the environment variable `TORCH_LOGS=output_code`. On the GPT blocks of this post, it keeps the 8 matrix multiplications of each block in cuBLAS, as calls to `extern_kernels.addmm` (with the bias) and `extern_kernels.bmm`, and fuses the operations between them. Among others, it fuses the scale, the mask and the softmax of the attention into one kernel, and each skip connection with the layer normalization that follows it, also across two blocks. However, it does not turn the attention into FlashAttention: TorchInductor replaces some patterns of attention with a call to the fused attention of PyTorch, `scaled_dot_product_attention`, but not the masked attention of GPTlite.

In practice, the largest tensor of the block is the matrix of the attention weights, with $$B \cdot H \cdot S^2$$ values, i.e. 805 MB in bf16 for a batch of 8 sequences. PyTorch writes it in the first batched matrix multiplication, reads and writes it in each of its `mul`, `masked_fill` and `softmax` kernels, and reads it in the second multiplication, i.e. 8 passes over the matrix in global memory. The fused kernel of `torch.compile` reads it twice, once for the maximum and the sum of the softmax and once for its output, and writes it once, i.e. 5 passes in total. The CUDA code of this post, with one kernel per operation, makes 6.5 passes, as its mask kernel reads nothing and writes only the masked half of the matrix.

## Benchmark

The benchmark runs the forward pass of the $$L = 12$$ blocks on sequences of $$S = 2048$$ tokens, 5 times to warm up and 20 times under the clock, and reports the average runtime of a forward pass. It compares five implementations of the same model, on the same input:
1. PyTorch, with the GPTlite code (`gpt.py`);
2. PyTorch with `torch.compile`, in its default mode (`gpt.py --compile`);
3. LibTorch (`gpt_libtorch.cpp`);
4. CUDA and cuBLAS (`gpt_cuda.cu`);
5. C++ loops on the CPU (`gpt_cpu.cpp`), in fp32, on a batch of one sequence only, as it is much slower.

The build follows the [instructions in the LibTorch documentation](https://pytorch.org/cppdocs/installing.html) and uses the [CMake](https://en.wikipedia.org/wiki/CMake) build system, which finds LibTorch in the installation of PyTorch. From a build folder inside the folder of the code, the programs are built and run as:

```shell
cmake .. -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`
make
python ../gpt.py --batch 8
python ../gpt.py --batch 8 --compile
./gpt_libtorch 8
./gpt_cuda 8
./gpt_cpu 1
```

where the argument is the batch size $$B$$. The GPU implementations ran on **[GPU]**, with CUDA **[version]** and PyTorch **[version]**, and the CPU one on **[CPU]**, with **[cores]** cores. The runtimes of a forward pass are the following:

| Runtime of a forward pass (ms) | $$B = 1$$ | $$B = 8$$ |
|:-------------------------------|:---------:|:---------:|
| PyTorch                        | **[ ]**   | **[ ]**   |
| PyTorch with `torch.compile`   | **[ ]**   | **[ ]**   |
| LibTorch                       | **[ ]**   | **[ ]**   |
| CUDA                           | **[ ]**   | **[ ]**   |

The C++ code on the CPU takes **[x]** ms per forward pass, on a batch of one sequence.

**[Discussion of the results.]**

## Kernel fusion on several GPUs

Kernel fusion speeds up the computation, but not the communication. When a model runs on several GPUs, e.g. with [tensor parallelism]({{ site.baseurl }}{% post_url 2023-09-02-tensor-parallelism %}), each block exchanges partial results between the GPUs with [collective operations](https://en.wikipedia.org/wiki/Collective_operation), such as an all-reduce after the attention and another after the FFN. Fusion shrinks the time of the computation and leaves the communication as it is, so the share of the communication in the runtime grows, and the speedup on $$p$$ GPUs moves further away from $$p$$ ([Amdahl's law](https://en.wikipedia.org/wiki/Amdahl%27s_law)).

Moreover, `torch.compile` does not fuse the collectives. TorchDynamo traces a synchronous collective of `torch.distributed` into the graph (an asynchronous one, with `async_op=True`, breaks the graph), but TorchInductor runs it as a kernel of NCCL of its own, between the fused kernels of the computation. Thus, a collective is a *fusion boundary*: the kernel before it writes its full output to global memory, the collective exchanges it, and only then does the next kernel start. The optimizations of TorchInductor for several GPUs do not fuse them either: the reordering of the collectives overlaps them with independent computation, and the [asynchronous tensor parallelism](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487) splits a collective and its matrix multiplication into smaller operations that overlap.

Fusing the communication into the kernel of the computation needs communication at the level of thread blocks, from inside the kernel. Each thread block then sends its tile of the output as soon as it is computed, and the receiving thread blocks wait only for the tiles that they need. This is possible with device-initiated communication, such as [NVSHMEM](https://docs.nvidia.com/nvshmem/api/introduction.html), in which a kernel reads and writes the memory of the other GPUs directly, or the [device API of NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html) (since NCCL 2.28). The usual collectives of NCCL are different: the CPU launches them as kernels of their own, after the computation, on full buffers in global memory. As `torch.compile` does not generate fused kernels of computation and communication, they are written by hand, which is the topic of the next posts: [kernel fusion for tensor parallelism]({{ site.baseurl }}{% post_url 2026-08-18-kernel-fusion-tensor-parallelism %}) and [kernel fusion for MoE expert parallelism]({{ site.baseurl }}{% post_url 2026-10-09-kernel-fusion-expert-parallelism %}).

To find more about the topic, the paper [PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation](https://docs.pytorch.org/assets/pytorch2-2.pdf) describes TorchDynamo and TorchInductor, the two extensions of PyTorch that implement `torch.compile`. Also, the tutorial [Using the PyTorch C++ Frontend](https://docs.pytorch.org/tutorials/advanced/cpp_frontend.html) covers the LibTorch API used in this post, from the module holders to the training loop, and [llm.c](https://github.com/karpathy/llm.c) implements the forward and the backward passes of GPT-2 in plain C and in CUDA.

The code of this post is in the [gpt-cpp-cuda folder](https://github.com/{{ site.repository }}/tree/master/assets/gpt-cpp-cuda).
