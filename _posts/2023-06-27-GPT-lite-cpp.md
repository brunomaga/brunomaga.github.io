---
layout: post
title:  "Building a GPT model in C++ and TorchScript from scratch"
categories: [machine learning, Transformer, GPT, LLM, C++, TorchScript]
tags: [machinelearning]
---

Previously, in [Building a GPT model from scratch]({{ site.baseurl }}{% post_url  2023-02-28-GPT-lite %}), we built a large language model based [GPT-2](https://en.wikipedia.org/wiki/GPT-2), using python and PyTorch. 

Here, we will write the same GPT2 model from scratch, in C++, and benchmark three implementations:
- the train and inference steps using the original python implementation;
- the train and inference steps using a C++ LibTorch implementation;
- the inference step, in C++ compiled code, using a TorchScript generated model in python;

We will follow the same sections as the [original post]({{ site.baseurl }}{% post_url  2023-02-28-GPT-lite %}) and provide the implementation of the header and body files.
The input data remains the same [tiny shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) text.

### Hyperparameters

Our GPT-lite will be written in the header-only format in the file `gptlite.h`. We start with the hyperparameter declarations:

```cpp
#pragma once

#include <torch/torch.h>

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

// namespace and type shortcuts
namespace nn = torch::nn;
using Tensor = torch::Tensor;
```

### Multi-Head Masked Attention

Remember the original formulation, where $$W^Q$$, $$W^K$$ and $$W^V$$ are matrices / projections / linear layers:

$$
MultiHead(Q, K, V ) = Concat(head_1, ..., head_h)W^O
$$

$$
\text{where } head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
$$

$$
\text{where } Attention(Q,K,V) = softmax \left( \frac{QK^T}{\sqrt{d_k}} \right) \, V
$$


{: style="text-align:center; font-size: small;"}
<img width="19%" height="19%" src="/assets/GPT-lite/gpt_lite_attention.png"/>

{: style="text-align:center; font-size: small;"}
The multi-head (Nx) attention module in our model, emphasized in red.

So the for the multi-attention head follows the same logic. We start by writing the code for the a single attention head:

```cpp
struct Head : nn::Module {
  Head(int head_size): head_size(head_size){
    nn::Linear key   = nn::Linear( nn::LinearOptions(n_embd, head_size).bias(false) );
    nn::Linear query = nn::Linear( nn::LinearOptions(n_embd, head_size).bias(false) );
    nn::Linear value = nn::Linear( nn::LinearOptions(n_embd, head_size).bias(false) );
    Tensor tril = torch::tril(torch::ones( {block_size, block_size} ));
    nn::Dropout dropout = nn::Dropout(dropout_p);

    register_module("key", key);
    register_module("query", query);
    register_module("value", value);
    register_buffer("tril", tril);
    register_module("dropout", this->dropout);
  }
}
```

Note the `register_module` operator that is not needed in the python implementation. Why do we need this? In practice, C++ has no reflection, so it cannot iterate over a class variables, unless they're declared somewhere. However, we need this features, so that LibTorch can iterate class members for e.g. parameter count, recursive copy of submodules to GPU when you call `moduel.to(device)`, etc. There are two options to create this iterator, and in this post we will use both:
1. We can keep all modules inside an iterator that LibTorch understands e.g. `nn::Sequential` and apply paramater count of move-to-GPU operations on the whole collection;
2. The other cleaner alternative, to avoid calling `module->to(device)` in every single module, is to run `register_parameter`, `register_buffer` and `register_module` to register them during initialization of the class.

Also, we do `register_buffer` instead of `register_parameter` on tril because it is a tensor that is not a parameter, but is a state, i.e. torch will not record it's gradients.

Finally, LibTorch does not allow named arguments like in Python e.g. `bias=False`, so these cant be passed *directly*. The possible constructors are `Linear(in_features, out_features)` or `Linear(LinearOptions(in_features, out_features).bias(False))`, so when we need to pass any named parameters, we use the second constructor and wrap all options inside `LinearOptions`. We now implement the forward pass of `Head`:

```cpp
  Tensor forward(Tensor x){
    int B=x.size(0), T=x.size(1), C=x.size(2);
    Tensor k = key(x);   //shape (B,T, head_size)
    Tensor q = query(x); //shape (B,T, head_size)
    Tensor v = value(x); //shape (B,T, head_size)

    // compute self-attention scores
    Tensor wei = torch::matmul(q, k.transpose(-2, -1)); //shape (B,T, T)
    wei = wei * std::pow(C,-0.5); //scale by sqrt(d_k) 
    wei = wei.masked_fill(tril.slice(0, 0, T).slice(1, 0, T) == 0, -inf);
    wei = F::softmax(wei, -1); // (B, T, T)
    wei = this->dropout(wei);

    // perform weighted aggregation of values
    Tensor out = torch::matmul(wei, v); // shape (B, T, head_size)
    return out;
  }
```

We'll now combine (merge) the output of all heads into a MultiHead Shared-Attention module.

```cpp
struct MultiHeadAttention : nn::Module {

  MultiHeadAttention(int num_heads, int head_size) {

    nn::ModuleList heads = torch::nn::ModuleList();
    for (int i=0; i<num_heads; i++)
      heads->push_back( Head(head_size) );
    nn::Linear proj = nn::Linear(n_embd, n_embd);
    nn::Dropout dropout = nn::Dropout(dropout_p);
    
    register_module("heads", heads);
    register_module("proj", proj);
    register_module("dropout", this->dropout);
  }


  Tensor forward(Tensor x){

    //Concatenate the outputs of the heads along the last dimension
    Tensor outputs[n_head];
    for (int i=0; i<n_head; i++){
      Head* head = heads[i]->as<Head>();
      outputs[i] = head->forward(x);
    }

    Tensor out = torch::cat(outputs, -1);
    out = proj(out);
    out = this->dropout(out);
    return out;
  }
}
```

### Feed Forward Network

The Feed-forward network simply a single-layer Deep Neural Network and is pretty straighforward to implement:

{: style="text-align:center; font-size: small;"}
<img width="19%" height="19%" src="/assets/GPT-lite/gpt_lite_feedforward.png"/>

{: style="text-align:center; font-size: small;"}
The feed forward network in our model, emphasized in red.

```cpp
struct FeedForward : nn::Module {

  FeedForward(int n_embd) {
    nn::Sequential net = nn::Sequential(
        nn::Linear(n_embd, n_embd*4),
        nn::ReLU(),
        nn::Linear(n_embd*4, n_embd),
        nn::Dropout(dropout_p)
	);
    register_module("net", net);

  Tensor forward(Tensor x) {
    return net->forward(x);
  }
}
```

### The GPT Block

We'll call GPT *block* the sequence of a multi-head attention and a feedforward module. Similarly to the python implementation, we add skip connections and normalization before the attention and feed-forward network.

{: style="text-align:center; font-size: small;"}
<img width="19%" height="19%" src="/assets/GPT-lite/gpt_lite_blocks.png"/>

{: style="text-align:center; font-size: small;"}
The GPT block(s) in our model, emphasized in red.

```cpp
struct Block : nn::Module {

  Block(int n_embd, int n_head) {
    int head_size = (int) (n_embd / n_head);
    std::shared_ptr<MultiHeadAttention> sa = 
      std::shared_ptr<MultiHeadAttention>( new MultiHeadAttention(n_head, head_size) );
    std::shared_ptr<FeedForward> ffwd =
      std::shared_ptr<FeedForward>( new FeedForward(n_embd) );
    nn::LayerNorm ln1 = nn::LayerNorm(  std::vector<int64_t> {n_embd} );
    nn::LayerNorm ln2 = nn::LayerNorm(  std::vector<int64_t> {n_embd} );

    register_module("sa", sa);
    register_module("ffwd", ffwd);
    register_module("ln1", ln1);
    register_module("ln2", ln2);
  }

  Tensor forward(Tensor x) {
    x = x + sa->forward(ln1(x));
    x = x + ffwd->forward(ln2(x));
    return x;
  }
}
```

There's a subtle difference in the `LayerNorm` initialization. By design, `LayerNorm` accepts a list of normalized dimensions as input. Alternatively, when a single `int` value is passed, only the last dimension of the input is normalized, and will resized to the integer argument value. This is the default behaviour in Python. However In C++, `LayerNorm` does not include the 'single integer' constructor initialization, so we have to pass it as a singleton list.

### Final Model

Putting it all together:

```cpp
struct GPTlite : nn::Module {

  GPTlite(int vocab_size){
    nn::Embedding token_embedding_table = nn::Embedding(vocab_size, n_embd);
    nn::Embedding position_embedding_table = nn::Embedding(block_size, n_embd);
    nn::Sequential blocks = nn::Sequential();
    for (int i=0; i<n_layer; i++)
      blocks->push_back( Block(n_embd, n_head) );
		
    nn::LayerNorm ln = nn::LayerNorm(  std::vector<int64_t> {n_embd} );
    nn::Linear lm_head = nn::Linear( nn::LinearOptions(n_embd, vocab_size).bias(false)  );

    register_module("token_embedding_table", token_embedding_table);
    register_module("position_embedding_table", position_embedding_table);
    register_module("blocks", blocks);
    register_module("ln", ln);
    register_module("lm_head", lm_head);
  }


  Tensor forward(Tensor idx){
    int T = idx.size(1);
    Tensor tok_emb = token_embedding_table(idx); //shape (B,T,C)
    Tensor pos_emb = position_embedding_table(torch::arange(T).to( idx.device() )); //shape (T,C)
    Tensor x = tok_emb + pos_emb; //shape (B,T,C)
    x = blocks->forward(x);
    x = ln(x);
    Tensor logits = lm_head(x); //shape (B,T,C)
    return logits.permute({0,2,1}); //shape (B,C,T)
  }
}
```


### Benchmark Model

We will also study a very simple benchmark model, a simple DNN with `L` layers of width `W`, with a ReLu activation between layers, and defined in `benchmark.h` as:
```cpp
#pragma once
#include <torch/torch.h>

struct BenchmarkModel : torch::nn::Module {
  /// DNN with W input features, W neurons per layer, W output classes and L layers

  BenchmarkModel(int64_t W, int64_t L){
    torch::nn::Sequential layers = torch::nn::Sequential();
    for (int64_t i = 0; i < L; ++i) {
      layers->push_back(torch::nn::Linear(W, W));
      layers->push_back(torch::nn::ReLU());
    }
    register_module("layers", layers);
  }

  torch::Tensor forward(torch::Tensor input) {
    return layers->forward(input);
  }
}
```

### Main loop

Our `main.cpp` file will contain a loop that will benchmark the train and inference operations of a model for a random input:

```cpp
torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

int main(int argc, const char* argv[]) {

    const int vocab_size = 65, batch_size=1; 
    const torch::ScalarType Long = torch::ScalarType::Long;
    torch::Tensor idx = torch::randint(0, vocab_size, {batch_size, block_size}, device).to(Long);
    torch::Tensor label = torch::randint(0, vocab_size, {batch_size, block_size}, device).to(Long);
    GPTlite model = GPTlite(vocab_size);
    model.to(device);
    benchmark_train<GPTlite>(model, idx, label);
    benchmark_inference<GPTlite>(model, idx);
}
```


As an important remark, LibTorch does not include a C++ equivalent to `torch.set_default_device`, so we have to manually move to the GPU every sample, label and model. And because we registered every parameter, buffer and module previously, doing `model.to(device)` will recursively copy all the contents in the model. The final functions `benchmark_train` and `benchmark_inference` perform the benchmark of method several train and inference epochs, respectively. The implementation is (again) very similar to PyTorch:

```cpp
const uint warmup_epochs = 30;
const uint benchmark_epochs = 30;

template <typename ModelType>
void benchmark_train(ModelType & model, torch::Tensor x, torch::Tensor label) {

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
  double throughput = benchmark_epochs / benchmark_time;
  std::cout << "train runtime: " << benchmark_time << " seconds" << std::endl;
  std::cout << "train throughput: " << throughput << " epochs/second" << std::endl;
}
``` 

The implementation of `benchmark_inference` is a much simpler loop with `model.eval()` instead and only the `model.forward(x)` in the epochs loop. However, it allows the templated types of both the model and input data, to account for the TorchScript-based inference that we will discuss later:

```cpp
template <typename ModelType, typename InputType = torch::Tensor>
void benchmark_inference(ModelType & model, InputType x) {

  clock_t start_time;
  model.eval();
  
  { 
    //no_grad scope, C++ equivalent to 'with torch.no_grad()' in Python
    torch::NoGradGuard no_grad;

    for (int64_t epoch = 0; epoch < warmup_epochs; ++epoch) 
      model.forward(x);

    start_time = clock();
    for (int64_t epoch = 0; epoch < benchmark_epochs; ++epoch)
      model.forward(x);
  }

  double benchmark_time = double(clock() - start_time) / CLOCKS_PER_SEC;
  double throughput = benchmark_epochs / benchmark_time;
  std::cout << "inference runtime: " << benchmark_time << " seconds" << std::endl;
  std::cout << "inference throughput: " << throughput << " epochs/second" << std::endl;
}
```

### Running inference on TorchScript

We now have completed both the python and C++ implementations. Ideally, we would have the flexibility and speed of development of python, with the low memory footprint and high efficiency of C++. To do that, we will train a model in python, and then load and run in on C++. To do that, we run the following python code after training our model, and this will output the model as a `pt` file:

```python
model_jit = torch.jit.script(model) 
# model_jit = torch.jit.trace(model, (x))
model_jit.save('model_jit.pt')
```

In C++, we load that module and run it with the following code:

```cpp
#include <torch/script.h>
using JitModule = torch::jit::Module;
using JitInput  = std::vector<torch::jit::IValue>;

JitModule model = torch::jit::load("model_jit.pt").to(device);
benchmark_inference<JitModule, JitInput>(model, {x});
```

Note that the type of the model and input data is not `torch::nn::Module` and `torch::Tensor`. Instead, we have `torch::jit::Module` and `std::vector<torch::jit::IValue>`, respectively. Therefore,`benchmark_interface` requires a different templated call with those types in place.

### Compilation

We follow the instructions the LibTorch documentation and use the CMake build systems to generate our binaries. The `CMakeLists.txt` is:

```cmake
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(torch_cpp_benchmark)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(SOURCE_FILES gptlite.h benchmark.h main.cpp)

add_executable(main ${SOURCE_FILES} )
target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 17)
``` 

### Benchmark

Coming soon

### Download code

See the original [source code repository](https://github.com/brunomaga/torchcpp-benchmark/) or download <a href="/assets/GPT-lite-cpp/torchcpp-benchmark-main.zip">`torchcpp-benchmark-main.zip`</a> for the complete implementation and run instructions.
