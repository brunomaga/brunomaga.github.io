#include <torch/torch.h>
#include "benchmark.h"

BenchmarkModel::BenchmarkModel(int64_t W, int64_t L, torch::Device device ){
  layers = torch::nn::Sequential();
	for (int64_t i = 0; i < L; ++i) {
		layers->push_back(torch::nn::Linear(W, W));
		layers->push_back(torch::nn::ReLU());
    }

  register_module("layers", layers);
  this->to(device);
}

torch::Tensor BenchmarkModel::forward(torch::Tensor input) {
  return layers->forward(input);
}


