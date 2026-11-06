// The GPT blocks on the CPU, in plain C++ loops, with fp32 values
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "gpt.h"

using dt = float;

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

// softmax of the n values of x, shifted by their maximum for numerical stability
void softmax(dt* x, int n) {
  dt max_val = *std::max_element(x, x + n), sum = 0;
  for (int j = 0; j < n; j++) {
    x[j] = std::exp(x[j] - max_val);
    sum += x[j];
  }
  for (int j = 0; j < n; j++)
    x[j] /= sum;
}

// masked multi-head attention softmax(q k^T / sqrt(D) + mask) v of each head of B sequences of S tokens, where
// q, k, v and the output o are (B*S x E), with the H heads of D values side by side in each row, and wei holds the
// attention weights (B*H x S x S)
void attention(const dt* q, const dt* k, const dt* v, dt* o, dt* wei, int B, int S) {
  #pragma omp parallel for collapse(2)
  for (int b = 0; b < B; b++)
    for (int h = 0; h < H; h++) {
      dt* w = wei + (long) (b*H + h) * S*S;   // the S x S attention weights of the head h of the sequence b
      for (int i = 0; i < S; i++)              // w = q k^T / sqrt(D) + mask, where the mask is -infinity for j > i
        for (int j = 0; j < S; j++) {
          const dt* qi = q + (b*S + i)*E + h*D;
          const dt* kj = k + (b*S + j)*E + h*D;
          dt score = 0;
          for (int d = 0; d < D; d++)
            score += qi[d] * kj[d];
          w[i*S + j] = j <= i ? score / std::sqrt((dt) D) : -INFINITY;
        }
      for (int i = 0; i < S; i++)              // softmax of each row
        softmax(w + i*S, S);
      for (int i = 0; i < S; i++) {            // o = w v
        dt* oi = o + (b*S + i)*E + h*D;
        for (int d = 0; d < D; d++)
          oi[d] = 0;
        for (int j = 0; j < S; j++) {
          const dt* vj = v + (b*S + j)*E + h*D;
          for (int d = 0; d < D; d++)
            oi[d] += w[i*S + j] * vj[d];
        }
      }
    }
}

// the skip connection x = x + y
void add(dt* x, const dt* y, int n) {
  for (int i = 0; i < n; i++)
    x[i] += y[i];
}

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
  layernorm(x, p.ln1_w, p.ln1_b, t.h, M, E);
  linear(t.h, p.Wq, p.bq, t.q, M, E, E);
  linear(t.h, p.Wk, p.bk, t.k, M, E, E);
  linear(t.h, p.Wv, p.bv, t.v, M, E, E);
  attention(t.q, t.k, t.v, t.a, t.wei, B, S);
  linear(t.a, p.Wo, p.bo, t.h, M, E, E);
  add(x, t.h, M*E);
  layernorm(x, p.ln2_w, p.ln2_b, t.h, M, E);
  linear(t.h, p.W1, p.b1, t.f, M, E, 4*E);
  relu(t.f, M*4*E);
  linear(t.f, p.W2, p.b2, t.h, M, 4*E, E);
  add(x, t.h, M*E);
}

int main(int argc, char* argv[]) {
  int batch = argc > 1 ? std::atoi(argv[1]) : B, M = batch*S;

  // the parameters of the L blocks, with the deterministic values of gpt.h
  unsigned int i = 0;
  auto linear_param = [&](int n) { dt* p = new dt[n]; for (int j = 0; j < n; j++) p[j] = init_value(i++, 0.02); return p; };
  auto norm_param = [](int n, dt c) { dt* p = new dt[n]; std::fill(p, p + n, c); return p; };
  std::vector<Block> blocks(L);
  for (Block& p : blocks) {
    p.ln1_w = norm_param(E, 1); p.ln1_b = norm_param(E, 0);
    p.Wq = linear_param(E*E); p.bq = linear_param(E); p.Wk = linear_param(E*E); p.bk = linear_param(E);
    p.Wv = linear_param(E*E); p.bv = linear_param(E); p.Wo = linear_param(E*E); p.bo = linear_param(E);
    p.ln2_w = norm_param(E, 1); p.ln2_b = norm_param(E, 0);
    p.W1 = linear_param(E*4*E); p.b1 = linear_param(4*E); p.W2 = linear_param(4*E*E); p.b2 = linear_param(E);
  }
  Buffers t = { new dt[M*E], new dt[M*E], new dt[M*E], new dt[M*E], new dt[M*E], new dt[M*4*E],
                new dt[(long) batch*H*S*S] };
  std::vector<dt> x0(M*E), x(M*E);
  for (int j = 0; j < M*E; j++)
    x0[j] = init_value(j, 1);

  std::chrono::steady_clock::time_point start;
  for (int it = 0; it < warmup_iters + benchmark_iters; it++) {
    if (it == warmup_iters)
      start = std::chrono::steady_clock::now();
    x = x0;
    for (const Block& p : blocks)
      block_forward(p, x.data(), t, batch, S);
  }
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

  double sum = 0;   // mean absolute value of the output of the first sequence, to compare implementations
  for (int j = 0; j < S*E; j++)
    sum += std::abs(x[j]);
  std::printf("CPU, batch size %d: %.1f ms per forward pass\n", batch, ms / benchmark_iters);
  std::printf("output: %.4f %.4f %.4f ..., mean |y| %.4f\n", x[0], x[1], x[2], sum / (S*E));
}
