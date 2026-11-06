#pragma once
#include <random>

// the dimensions of GPT-3 Small, Table 2.1 of "Language Models are Few-Shot Learners", Brown et al., 2020
constexpr int L = 12;            // number of blocks (n_layers)
constexpr int E = 768;           // size of the embeddings (d_model)
constexpr int H = 12;            // number of attention heads (n_heads)
constexpr int D = E / H;         // size of each attention head (d_head)
constexpr int S = 2048;          // sequence length (n_ctx)
constexpr int B = 8;             // batch size of the benchmark, in sequences
constexpr float eps = 1e-5f;     // epsilon of the layer normalization, as in PyTorch

// the type of the values: bf16 when nvcc compiles CUDA code (it then defines __CUDACC__), and fp32 otherwise
#ifdef __CUDACC__
#include <cuda_bf16.h>
using dt = __nv_bfloat16;
#else
using dt = float;
#endif

constexpr int warmup_iters = 5;      // iterations before the timed ones
constexpr int benchmark_iters = 20;  // timed iterations

// the next random value in [-scale, scale), from the same generator and seed in every implementation, which draws
// the weights and biases of the linear layers of each block (Wq, bq, ..., W2, b2), and then the input
inline std::mt19937 generator(19);
inline float init_value(float scale) { return scale * std::uniform_real_distribution<float>(-1, 1)(generator); }
