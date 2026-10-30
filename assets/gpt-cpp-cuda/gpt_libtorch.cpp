// The GPT blocks in LibTorch, the C++ distribution of PyTorch, with bf16 values
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <torch/torch.h>
#include "gpt.h"

namespace nn = torch::nn;
using torch::Tensor;

struct FeedForward : nn::Module {

  FeedForward() {
    net = register_module("net", nn::Sequential(nn::Linear(E, 4*E), nn::ReLU(), nn::Linear(4*E, E)));
  }

  Tensor forward(Tensor x) {
    return net->forward(x);
  }

  nn::Sequential net{nullptr};
};

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

// the values init_value(i, scale) to init_value(i+n-1, scale) of gpt.h
Tensor init_values(unsigned int i, int n, double scale) {
  std::vector<float> v(n);
  for (float& x : v)
    x = init_value(i++, scale);
  return torch::tensor(v);
}

// the parameters of gpt.h, where nn::Linear stores the transposed weights W^T (out x in)
void init_params(nn::Sequential& blocks) {
  torch::NoGradGuard no_grad;
  unsigned int i = 0;
  for (auto& module : blocks->children()) {
    Block* block = module->as<Block>();
    for (nn::LinearImpl* linear : { block->sa->query.get(), block->sa->key.get(), block->sa->value.get(),
                                    block->sa->proj.get(), block->ffwd->net[0]->as<nn::Linear>(),
                                    block->ffwd->net[2]->as<nn::Linear>() }) {
      int n_out = linear->weight.size(0), n_in = linear->weight.size(1);
      linear->weight.copy_(init_values(i, n_in*n_out, 0.02).view({n_in, n_out}).t());
      linear->bias.copy_(init_values(i + n_in*n_out, n_out, 0.02));
      i += n_in*n_out + n_out;
    }
  }
}

int main(int argc, char* argv[]) {
  int batch = argc > 1 ? std::atoi(argv[1]) : B;
  torch::Device device(torch::kCUDA);
  nn::Sequential model;
  for (int l = 0; l < L; l++)
    model->push_back(std::make_shared<Block>());
  init_params(model);
  model->to(device, torch::kBFloat16);
  model->eval();
  Tensor x = init_values(0, batch*S*E, 1).view({batch, S, E}).to(device, torch::kBFloat16);

  c10::InferenceMode guard;   // the C++ equivalent of torch.inference_mode() in Python
  Tensor y;
  std::chrono::steady_clock::time_point start;
  for (int it = 0; it < warmup_iters + benchmark_iters; it++) {
    if (it == warmup_iters) {
      torch::cuda::synchronize();
      start = std::chrono::steady_clock::now();
    }
    y = model->forward(x);
  }
  torch::cuda::synchronize();
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

  y = y[0].to(torch::kCPU, torch::kFloat);   // the output of the first sequence, to compare implementations
  std::printf("LibTorch, batch size %d: %.1f ms per forward pass\n", batch, ms / benchmark_iters);
  std::printf("output: %.4f %.4f %.4f ..., mean |y| %.4f\n", y[0][0].item<float>(), y[0][1].item<float>(),
              y[0][2].item<float>(), y.abs().mean().item<float>());
}
