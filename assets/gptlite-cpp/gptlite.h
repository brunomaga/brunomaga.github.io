#pragma once

#include <torch/torch.h>
#include <torch/script.h>

// replicate GPT-3 Small in Table 2.1 in "Language Models are Few-Shot Learners, Brown et al, 2021"

// depth of the network as number of decoder blocks.
const int n_layer = 12;

// size of the embeddings (d_model)
const int n_embd = 768;

// number of attention heads in the Multi-Attention mechanism
const int n_head = 12;

// block size ie max number of training sequence, the $n_{ctx}$ in the paper .
const int block_size = 2048;

// dropout rate (variable p) for dropout units
const float dropout_p = 0.1;

namespace nn = torch::nn;
using Tensor = torch::Tensor;

struct Head : nn::Module {
  Head(int head_size);
  Tensor forward(Tensor x);
  nn::Linear key{nullptr}, query{nullptr}, value{nullptr};
  nn::Dropout dropout;
  Tensor tril;
  int head_size;
};

struct MultiHeadAttention : nn::Module {
  MultiHeadAttention(int num_heads, int head_size);
  Tensor forward(Tensor x);
  nn::ModuleList heads;
  nn::Linear proj{nullptr};
  nn::Dropout dropout;
};

struct FeedForward : nn::Module {
  FeedForward(int n_embd);
  Tensor forward(Tensor x);
  nn::Sequential net;
};

struct Block : nn::Module {
  Block(int n_embd, int n_head);
  Tensor forward(Tensor x);
  std::shared_ptr<MultiHeadAttention> sa;
  std::shared_ptr<FeedForward> ffwd;
  nn::LayerNorm ln1{nullptr}, ln2{nullptr};
};

struct GPTlite : nn::Module {
  GPTlite(int vocab_size, torch::Device device);
  Tensor forward(Tensor idx);
  nn::Embedding token_embedding_table{nullptr}, position_embedding_table{nullptr};
  nn::Sequential blocks;
  nn::LayerNorm ln{nullptr};
  nn::Linear lm_head{nullptr};
};
