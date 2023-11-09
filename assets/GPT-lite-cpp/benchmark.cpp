#include <torch/torch.h>
#include "benchmark.h"

BenchmarkModel::BenchmarkModel(int64_t W, int64_t L, int64_t in_size, int64_t out_size, torch::Device device ){
  /// DNN with L layers and W neurons per layer 

  layers = torch::nn::Sequential();
	layers->push_back(torch::nn::Linear(in_size, W));
	layers->push_back(torch::nn::ReLU());
	for (int64_t l = 0; l<L-2; ++l) {
    layers->push_back(torch::nn::Linear(W, W));
    layers->push_back(torch::nn::ReLU());
    }
	layers->push_back(torch::nn::Linear(W, out_size));
	layers->push_back(torch::nn::ReLU());
  register_module("layers", layers);
  this->to(device);
}

torch::Tensor BenchmarkModel::forward(torch::Tensor input) {
  return layers->forward(input);
}


