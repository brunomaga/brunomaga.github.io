#include <torch/torch.h>
#include "benchmark.h"

BenchmarkModel::BenchmarkModel(int64_t W, int64_t L, int64_t in_size, int64_t out_size, torch::Device device ){
  layers = torch::nn::Sequential();
	for (int64_t l = 0; l < L; ++l) {
		layers->push_back(torch::nn::Linear(
      l==0   ? in_size  : W,
      l==L-1 ? out_size : W));
		layers->push_back(torch::nn::ReLU());
    }
  register_module("layers", layers);
  this->to(device);
}

torch::Tensor BenchmarkModel::forward(torch::Tensor input) {
  return layers->forward(input);
}


