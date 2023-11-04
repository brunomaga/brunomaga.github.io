#pragma once

#include <torch/torch.h>

struct BenchmarkModel : torch::nn::Module {
  /// DNN with W input features, W neurons per layer, W output classes and L layers
  BenchmarkModel(int64_t W, int64_t L, torch::Device device);
  torch::Tensor forward(torch::Tensor input);
  torch::nn::Sequential layers;
};
    

