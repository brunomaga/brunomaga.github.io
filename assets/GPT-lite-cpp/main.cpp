#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <time.h>

#include "benchmark.h"
#include "gptlite.h"

namespace F = torch::nn::functional;
using JitModule = torch::jit::Module; 
using JitInput  = std::vector<torch::jit::IValue>;

// peform X warmup epochs before measuring performance on Y benchmark epochs 
const uint warmup_epochs = 30;
const uint benchmark_epochs = 30;
const torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;


template <typename ModelType>
void benchmark_train(ModelType & model, const torch::Tensor x, const torch::Tensor label, const std::string model_name) {

  clock_t start_time;
  torch::Tensor output, loss;
  
  model.train();
  torch::optim::Adam optimizer( model.parameters(),
    torch::optim::AdamOptions(2e-4).betas(std::make_tuple(0.5, 0.5)));

  for (int64_t epoch = 0; epoch < warmup_epochs + benchmark_epochs; ++epoch) {

    if (epoch == warmup_epochs)
      start_time = clock();

    optimizer.zero_grad();
    output = model.forward(x);
    output = F::softmax(output, F::SoftmaxFuncOptions(1));
    loss = torch::cross_entropy_loss(output, label);
    loss.backward();
    optimizer.step();
  }

  double benchmark_time = double(clock() - start_time) / CLOCKS_PER_SEC;
  std::cout << model_name << " train runtime: " << benchmark_time << " seconds" << std::endl;
  std::cout << model_name << " train throughput: " << benchmark_epochs / benchmark_time << " epochs/second" << std::endl;
}


template <typename ModelType, typename InputType = torch::Tensor>
void benchmark_inference(ModelType & model, const InputType x, const std::string model_name, const uint epochs_multiplier=10) {

  clock_t start_time;
  model.eval();
  
  { 
    torch::NoGradGuard no_grad; //no_grad scope, C++ equivalent to 'with torch.no_grad()' in Python

    for (int64_t epoch = 0; epoch < warmup_epochs*epochs_multiplier; ++epoch) 
      model.forward(x);

    start_time = clock();
    for (int64_t epoch = 0; epoch < benchmark_epochs*epochs_multiplier; ++epoch)
      model.forward(x);
  }

  double benchmark_time = double(clock() - start_time) / CLOCKS_PER_SEC;
  std::cout << model_name << " inference runtime: " << benchmark_time << " seconds" << std::endl;
  std::cout << model_name << " inference throughput: " << benchmark_epochs*epochs_multiplier / benchmark_time << " epochs/second" << std::endl;
}


JitModule load_jit_model(std::string jit_folder, std::string model_name, torch::Device device) {
  std::string model_name_curated = std::string(model_name); 
  std::replace(model_name_curated.begin(), model_name_curated.end(), ' ', '_');
  std::string model_filename = jit_folder + "/" + model_name_curated + ".pt";
  std::cout << "Loading jit model from " << model_filename << std::endl;
  JitModule model = torch::jit::load(model_filename);
  model.to(device);
  return model;
}


int main(int argc, const char* argv[]) {

  if (argc > 2)
    throw std::runtime_error("usage: ./main [jit_folder]. Run main.py to generate jit models.");

  if (!torch::cuda::is_available())
    std::cout << "WARNING: CUDA not available, using CPU." << std::endl;


  {
    // Deep DNN model (W=256, L=2048)
    const std::string model_name = "Deep DNN";
    const int W=256, L=2048, batch_size=2048;
    torch::Tensor x = torch::randn({batch_size, W}, device);
    torch::Tensor label = torch::randn({batch_size, W}, device);
    {
      BenchmarkModel model = BenchmarkModel(W, L, device);
      benchmark_train<BenchmarkModel>(model, x, label, model_name);
      benchmark_inference<BenchmarkModel>(model, x, model_name);
    }
    if (argc==2){
      JitModule model = load_jit_model(argv[1], model_name, device);
      benchmark_inference<JitModule, JitInput>(model, {x}, model_name);
    }
  }

  {
    // Wide DNN Model (W=8192, L=3)
    const std::string model_name = "Wide DNN";
    const int W=8192, L=3, batch_size=2048;
    torch::Tensor x = torch::randn({batch_size, W}, device);
    torch::Tensor label = torch::randn({batch_size, W}, device);
    {
      BenchmarkModel model = BenchmarkModel(W, L, device);
      benchmark_train<BenchmarkModel>(model, x, label, model_name);
      benchmark_inference<BenchmarkModel>(model, x, model_name);
    }if (argc==2)
    {
      JitModule model = load_jit_model(argv[1], model_name, device);
      benchmark_inference<JitModule, JitInput>(model, {x}, model_name);
    }
  }

  {
    // GPTlite Model: (B, T, C) = (batch_size_deep, block_size, n_embed)
    const std::string model_name = "GPTlite";
    const int vocab_size = 65, batch_size=1; 
    const torch::ScalarType Long = torch::ScalarType::Long;
    torch::Tensor idx = torch::randint(0, vocab_size, {batch_size, block_size}, device).to(Long);
    torch::Tensor label = torch::randint(0, vocab_size, {batch_size, block_size}, device).to(Long);
    {
      GPTlite model = GPTlite(vocab_size, device);
      benchmark_train<GPTlite>(model, idx, label, model_name);
      benchmark_inference<GPTlite>(model, idx, model_name);
    }
    if (argc==2){
      JitModule model = load_jit_model(argv[1], model_name, device);
      benchmark_inference<JitModule, JitInput>(model, {idx}, model_name);
    }
  }

}
