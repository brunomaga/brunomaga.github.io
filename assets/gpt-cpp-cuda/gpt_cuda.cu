// The GPT blocks on a GPU, in CUDA kernels and cuBLAS, with bf16 values
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include "gpt.h"

using dt = __nv_bfloat16;
constexpr int THREADS = 256;   // threads per thread block of the element-wise and row kernels
cublasHandle_t handle;

// number of thread blocks of THREADS threads for n elements, one thread per element
unsigned int blocks(long n) { return (n + THREADS - 1) / THREADS; }

// C (M x O) = A (M x N) B (N x O) in row-major order, on the Tensor Cores, with fp32 accumulation. cuBLAS reads
// matrices in column-major order, where the row-major A, B and C are A^T, B^T and C^T, so it computes C^T = B^T A^T
void matmul(const dt* A, const dt* B, dt* C, int M, int N, int O) {
  float alpha = 1, beta = 0;
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, O, M, N, &alpha, B, CUDA_R_16BF, O, A, CUDA_R_16BF, N,
               &beta, C, CUDA_R_16BF, O, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

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

// the skip connection x = x + y
__global__ void add(dt* x, const dt* y, long n) {
  long i = (long) blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] = (float) x[i] + (float) y[i];
}

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

// parameters of a block, where the weights W of the linear layers are stored as (in x out)
struct Block {
  dt *ln1_w, *ln1_b, *Wq, *bq, *Wk, *bk, *Wv, *bv, *Wo, *bo;   // layer normalization and attention
  dt *ln2_w, *ln2_b, *W1, *b1, *W2, *b2;                        // layer normalization and feed-forward network
};

// intermediate results of a block
struct Buffers {
  dt *h, *q, *k, *v, *a, *f, *wei;
};

// forward pass of a block on x (B*S x E): x = x + attention(layernorm(x)), then x = x + ffn(layernorm(x))
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

int main(int argc, char* argv[]) {
  int batch = argc > 1 ? std::atoi(argv[1]) : B, M = batch*S;
  cublasCreate(&handle);

  // the parameters of the L blocks, with the deterministic values of gpt.h, copied to the GPU
  auto to_gpu = [](const std::vector<dt>& v) {
    dt* p;
    cudaMalloc(&p, v.size() * sizeof(dt));
    cudaMemcpy(p, v.data(), v.size() * sizeof(dt), cudaMemcpyHostToDevice);
    return p;
  };
  unsigned int i = 0;
  auto linear_param = [&](int n) { std::vector<dt> v(n); for (dt& w : v) w = init_value(i++, 0.02); return to_gpu(v); };
  auto norm_param = [&](int n, float c) { return to_gpu(std::vector<dt>(n, c)); };
  std::vector<Block> blocks(L);
  for (Block& p : blocks) {
    p.ln1_w = norm_param(E, 1); p.ln1_b = norm_param(E, 0);
    p.Wq = linear_param(E*E); p.bq = linear_param(E); p.Wk = linear_param(E*E); p.bk = linear_param(E);
    p.Wv = linear_param(E*E); p.bv = linear_param(E); p.Wo = linear_param(E*E); p.bo = linear_param(E);
    p.ln2_w = norm_param(E, 1); p.ln2_b = norm_param(E, 0);
    p.W1 = linear_param(E*4*E); p.b1 = linear_param(4*E); p.W2 = linear_param(4*E*E); p.b2 = linear_param(E);
  }
  auto alloc = [](long n) { dt* p; cudaMalloc(&p, n * sizeof(dt)); return p; };
  Buffers t = { alloc(M*E), alloc(M*E), alloc(M*E), alloc(M*E), alloc(M*E), alloc(M*4*E), alloc((long) batch*H*S*S) };
  std::vector<dt> host(M*E);
  for (int j = 0; j < M*E; j++)
    host[j] = init_value(j, 1);
  dt *x0 = to_gpu(host), *x = alloc(M*E);

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
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::printf("CUDA error: %s\n", cudaGetErrorString(err));
    return 1;
  }

  cudaMemcpy(host.data(), x, S*E * sizeof(dt), cudaMemcpyDeviceToHost);   // the output of the first sequence
  double sum = 0;
  for (int j = 0; j < S*E; j++)
    sum += std::abs((float) host[j]);
  std::printf("CUDA, batch size %d: %.1f ms per forward pass\n", batch, ms / benchmark_iters);
  std::printf("output: %.4f %.4f %.4f ..., mean |y| %.4f\n", (float) host[0], (float) host[1], (float) host[2], sum / (S*E));
}
