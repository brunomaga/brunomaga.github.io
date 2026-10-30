#pragma once

// the dimensions of GPT-3 Small, Table 2.1 of "Language Models are Few-Shot Learners", Brown et al., 2020
constexpr int L = 12;            // number of blocks (n_layers)
constexpr int E = 768;           // size of the embeddings (d_model)
constexpr int H = 12;            // number of attention heads (n_heads)
constexpr int D = E / H;         // size of each attention head (d_head)
constexpr int S = 2048;          // sequence length (n_ctx)
constexpr int B = 8;             // batch size of the benchmark, in sequences
constexpr float eps = 1e-5f;     // epsilon of the layer normalization, as in PyTorch

constexpr int warmup_iters = 5;      // iterations before the timed ones
constexpr int benchmark_iters = 20;  // timed iterations

// the same deterministic values in every implementation: the input is init_value(i, 1), and the weights and biases
// of the linear layers are init_value(i, 0.02), where i counts their values block after block, in the order Wq, bq,
// Wk, bk, Wv, bv, Wo, bo, W1, b1, W2, b2, with each W stored as (in x out) in row-major order. The layer
// normalizations keep the defaults of PyTorch (weight 1, bias 0)
inline float init_value(unsigned int i, double scale) {
  i = (i ^ 61) ^ (i >> 16);   // Thomas Wang's integer hash of i, on 32 bits
  i *= 9;
  i ^= i >> 4;
  i *= 0x27d4eb2d;
  i ^= i >> 15;
  return (float) (scale * (2.0 * i / 4294967295.0 - 1.0));   // from [0, 2^32 - 1] to [-scale, scale]
}
