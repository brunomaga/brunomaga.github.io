---
layout: post
title:  "GPT in C++: LibTorch vs. PyTorch, TorchScript and torch.compile"
categories: ["Efficient Inference"]
tags: ["GPT", "LLM", "C++", "LibTorch", "TorchScript", "torch.compile", "benchmarking"]
---

[PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/) introduced `torch.compile`, which compiles a model into optimized kernels, and wrote its new compiler stack in Python instead of C++. The benchmark below measures how much faster a C++ implementation of a model runs than the Python one, whether a model trained in Python runs efficiently from C++, and the speedup of `torch.compile`.

## From C++ Back to Python

Since its release in 2017, PyTorch moved substantial parts of its internals into C++, to keep eager execution fast while GPUs became about 15 times faster in compute. The PyTorch 2.0 announcement notes that "moving internals into C++ makes them less hackable and increases the barrier of entry for code contributions". The compiler stack of PyTorch 2, made of TorchDynamo, AOTAutograd, PrimTorch and TorchInductor, is thus written in Python. PrimTorch decomposes the more than 2000 operators of PyTorch, counting overloads, into a closed set of about 250 primitive operators, so that a hardware vendor or a compiler backend implements only these to support all of PyTorch. The eager operators stay in C++, and LibTorch exposes them to C++ programs.

## Kernel Fusion and torch.compile

Each PyTorch operation launches one or more GPU kernels. Each launch costs a few microseconds of CPU and driver time, and each kernel reads its inputs from GPU memory and writes its outputs back. For large models and batches, these costs are small compared to the computation. For small models and batches, they dominate: a forward pass launches hundreds of small kernels, and the GPU waits for the CPU to launch the next kernel, or for the memory transfers between kernels.

*Kernel fusion* merges consecutive operations into one kernel, so that the intermediate results stay in registers or in on-chip memory instead of going through GPU memory. Typical candidates are the chains of element-wise operations after a matrix multiplication (bias, activation function, dropout, residual addition) and the normalization layers. [FlashAttention](https://arxiv.org/abs/2205.14135) (Dao et al., 2022) fuses the whole attention into one kernel, which never writes the matrix of attention scores to GPU memory.

`torch.compile` fuses kernels with a one-line change, `model = torch.compile(model)`, by [just-in-time compilation](https://en.wikipedia.org/wiki/Just-in-time_compilation). TorchDynamo captures the graph of the model from the Python bytecode, and TorchInductor generates fused kernels, in Triton for GPUs and in C++ with OpenMP for CPUs. A new input shape can trigger a recompilation. The compilation runs in three steps:
1. **Graph acquisition**: the model is rewritten as subgraphs. TorchDynamo flattens the subgraphs that it can compile, and the others, which hold control flow or other unsupported Python code, run in eager mode.
2. **Graph lowering**: the PyTorch operations are decomposed into the kernels of the chosen backend.
3. **Graph compilation**: the kernels call the low-level operations of the device.

{: .picture}
<img width="80.3%" height="80.3%" src="{{ site.assets }}/gptlite-cpp/torch_compile.svg"/>

{: .caption}
A diagram of the three steps of `torch.compile`. Adapted from: [PyTorch 2.0 technology overview](https://pytorch.org/get-started/pytorch-2.0/#technology-overview)

Two arguments of `torch.compile` control the compilation ([docs](https://pytorch.org/get-started/pytorch-2.0/#user-experience), [API](https://pytorch.org/docs/stable/generated/torch.compile.html#torch.compile)). The argument `mode` sets the trade-off between compilation time, runtime and memory:
- `default` optimizes for large models, with a short compilation and no extra memory, "a good balance between performance and overhead";
- `reduce-overhead` also runs the model as CUDA graphs (below), which uses extra memory and speeds up small models;
- `max-autotune` benchmarks several Triton and template-based kernels for the matrix multiplications and convolutions, keeps the fastest, and uses CUDA graphs. It produces the fastest model, but compiles for a long time. `max-autotune-no-cudagraphs` does the same without CUDA graphs.

With `fullgraph=True`, the compiler compiles the whole model into one graph, or raises an error that names the code it cannot compile, like the nopython mode of [Numba](https://en.wikipedia.org/wiki/Numba). With `fullgraph=False`, such code splits the model into several graphs that run with Python code in between (graph breaks). The default arguments are `torch.compile(model, mode='default', fullgraph=False)`.

Even after fusion, a forward pass launches many kernels from Python, one after the other. A [CUDA graph](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) records the whole sequence of kernel launches once, and replays it with one launch, which removes almost all the CPU overhead. The recording fixes the shapes and the memory addresses of all tensors, so the inputs must have static shapes. For text generation, the cache of the keys and values of the previous tokens (KV cache) is thus preallocated to the maximum length, and attention runs over the full buffer, with a mask over the positions not filled yet. With these techniques, int4 quantization and speculative decoding, the PyTorch team's [gpt-fast](https://pytorch.org/blog/accelerating-generative-ai-2/) decodes Llama-7B almost 10 times faster than its baseline, in plain PyTorch. A *megakernel* goes further: it runs the whole forward pass in one persistent kernel, without gaps between kernels. The megakernel of [Look Ma, No Bubbles!](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) (Spector et al., 2025) decodes Llama-1B more than 1.5 times faster than SGLang on an H100 GPU.

## Benchmark Setup

The benchmark runs the training and the inference of two kinds of models on PyTorch 1.13.1, PyTorch 2.1, LibTorch 2.1 (C++), TorchScript 2.1 (export from Python, inference in C++), and PyTorch 2.1 with `torch.compile`. The models are:
1. GPTlite, the model of [Building a GPT model in PyTorch from scratch]({{ site.baseurl }}{% post_url  2023-02-28-gptlite %}), with the dimensions of GPT-3 Small, in section [GPTlite on LibTorch C++](#gptlite-model);
2. a deep neural network (DNN) with `L` linear layers of width `W`, in section [Benchmark Model on LibTorch C++](#benchmark-model). Its depth and width are free, so it separates the cost of many small kernels from that of a few large ones.

{: .picture}
<img width="25.2%" height="25.2%" src="{{ site.assets }}/gptlite-cpp/gptlite_compact.svg"/>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
<img width="25.7%" height="25.7%" src="{{ site.assets }}/gptlite-cpp/benchmark_model.svg"/>

{: .caption}
This post details the C++ implementation of the GPT-3 Small model with N decoder blocks (left), and of a simple Deep Neural Network with L layers of dimensionality W, each followed by a ReLU (right). The C++, PyTorch, TorchScript and `torch.compile` implementations of both models are then benchmarked. Left diagram adapted from: [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

## GPTlite on LibTorch C++ {#gptlite-model}

The C++ modules follow the Python modules of the GPTlite post.

### Hyperparameters

The header-only file `gptlite.h` holds the whole model. It starts with the hyperparameters of GPT-3 Small:

```cpp
#pragma once
#include <torch/torch.h>

// replicate GPT-3 Small, Table 2.1, "Language Models are Few-Shot Learners", Brown et al., 2020

// depth of the network as number of decoder blocks.
const int n_layer = 12;

// size of the embeddings (d_model)
const int n_embd = 768;

// number of attention heads in the Multi-Attention mechanism
const int n_head = 12;

// block size, i.e. the maximum sequence length, the n_ctx in the paper
const int block_size = 2048;

// dropout rate (variable p) for dropout units, renamed to avoid ambiguity
const float dropout_p = 0.1;

// namespace and data type aliases
namespace nn = torch::nn;
namespace F = torch::nn::functional;
using Tensor = torch::Tensor;

// infinity, used to mask attention scores
const float inf = std::numeric_limits<float>::infinity();
```

### Multi-Head Masked Attention

The multi-head attention, with the projections $$W^Q_i$$, $$W^K_i$$ and $$W^V_i$$ of head $$i$$ and the output projection $$W^O$$, is:

$$
\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h) \, W^O
$$

$$
\text{where } \mathrm{head}_i = \mathrm{Attention}(Q W^Q_i, K W^K_i, V W^V_i)
$$

$$
\text{and } \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V
$$

where $$h$$ is the number of heads and $$d_k$$ is the size of a head. Each of the $$N$$ blocks of the model applies it with the lower-triangular mask `tril`:

{: .picture}
<img width="25.2%" height="25.2%" src="{{ site.assets }}/gptlite/gptlite_attention.svg"/>

{: .caption}
The multi-head attention module in the GPTlite model, emphasized in red. Adapted from: [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

The first module is one attention head:

```cpp
struct Head : nn::Module {
  Head(int head_size) : head_size(head_size) {
    key   = nn::Linear( nn::LinearOptions(n_embd, head_size).bias(false) );
    query = nn::Linear( nn::LinearOptions(n_embd, head_size).bias(false) );
    value = nn::Linear( nn::LinearOptions(n_embd, head_size).bias(false) );
    tril = torch::tril(torch::ones( {block_size, block_size} ));
    dropout = nn::Dropout(dropout_p);

    register_module("key", key);
    register_module("query", query);
    register_module("value", value);
    register_buffer("tril", tril);
    register_module("dropout", this->dropout);
  }


  Tensor forward(Tensor x){
    int B=x.size(0), S=x.size(1), E=x.size(2);
    Tensor k = key(x);   //shape (B,S, head_size)
    Tensor q = query(x); //shape (B,S, head_size)
    Tensor v = value(x); //shape (B,S, head_size)

    // compute self-attention scores
    Tensor wei = torch::matmul(q, k.transpose(-2, -1)); //shape (B,S, S)
    wei = wei * std::pow(E,-0.5); //scale by sqrt(d_k)
    wei = wei.masked_fill(tril.slice(0, 0, S).slice(1, 0, S) == 0, -inf);
    wei = F::softmax(wei, -1); // (B, S, S)
    wei = this->dropout(wei);

    // perform weighted aggregation of values
    Tensor out = torch::matmul(wei, v); // shape (B, S, head_size)
    return out;
  }

  nn::Linear key{nullptr}, query{nullptr}, value{nullptr};
  nn::Dropout dropout;
  Tensor tril;
  int head_size;
};
```

`Head` and the following modules are declared as `struct` instead of `class`, so that their members are public by default. C++ has no reflection: a module cannot list its members unless they are declared to it. LibTorch needs this list, e.g. to give the parameters to the optimizer, to move all submodules to a device, or to switch them between training and evaluation mode. `register_parameter`, `register_buffer` and `register_module` declare the parameters, buffers and submodules in the constructor. A container such as `nn::Sequential` or `nn::ModuleList` registers the modules it holds, and is registered itself. `tril` is registered as a buffer: it is part of the state of the module and moves with it to the device, but it is not a parameter and has no gradient.

C++ has no named arguments, such as `bias=False` in Python. LibTorch passes options through an options object with one setter per option, e.g. `Linear(LinearOptions(in_features, out_features).bias(false))`.

The multi-head attention concatenates the outputs of its heads:

```cpp
struct MultiHeadAttention : nn::Module {

  MultiHeadAttention(int num_heads, int head_size) {

    heads = torch::nn::ModuleList();
    for (int i=0; i<num_heads; i++)
      heads->push_back( Head(head_size) );
    proj = nn::Linear(num_heads*head_size, n_embd);
    dropout = nn::Dropout(dropout_p);
    
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

  nn::ModuleList heads;
  nn::Linear proj{nullptr};
  nn::Dropout dropout;
};
```

`nn::ModuleList` holds elements of the base type `nn::Module`, so `heads[i]->as<Head>()` casts each element to `Head` before the call to its `forward`.

### Feed-Forward Network

The feed-forward network has two linear layers with a ReLU between them:

{: .picture}
<img width="25.2%" height="25.2%" src="{{ site.assets }}/gptlite/gptlite_feedforward.svg"/>

{: .caption}
The feed forward network in the GPTlite model, emphasized in red. Adapted from: [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

```cpp
struct FeedForward : nn::Module {

  FeedForward(int n_embd) {
    net = nn::Sequential(
      nn::Linear(n_embd, n_embd*4),
      nn::ReLU(),
      nn::Linear(n_embd*4, n_embd),
      nn::Dropout(dropout_p)
      );
    register_module("net", net);
  }

  Tensor forward(Tensor x) {
    return net->forward(x);
  }

  nn::Sequential net;
};
```

### The GPT Block

As in Python, a block applies the multi-head attention and the feed-forward network, each with a layer normalization of its input and a residual connection:

{: .picture}
<img width="25.4%" height="25.4%" src="{{ site.assets }}/gptlite/gptlite_blocks.svg"/>

{: .caption}
The GPT block(s) in the GPTlite model, emphasized in red. Adapted from: [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

```cpp
struct Block : nn::Module {

  Block(int n_embd, int n_head) {
    int head_size = (int) (n_embd / n_head);
    sa = std::shared_ptr<MultiHeadAttention>( new MultiHeadAttention(n_head, head_size) );
    ffwd = std::shared_ptr<FeedForward>( new FeedForward(n_embd) );
    ln1 = nn::LayerNorm(  std::vector<int64_t> {n_embd} );
    ln2 = nn::LayerNorm(  std::vector<int64_t> {n_embd} );

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

  std::shared_ptr<MultiHeadAttention> sa;
  std::shared_ptr<FeedForward> ffwd;
  nn::LayerNorm ln1{nullptr}, ln2{nullptr};
};
```

Every LibTorch module type is a *module holder*, a wrapper around a `std::shared_ptr` to its implementation: `nn::Linear` holds a `std::shared_ptr<nn::LinearImpl>`. Copying a module holder copies the pointer, not the module, so modules are passed by value. The code wraps its own modules, `MultiHeadAttention` and `FeedForward`, in a `std::shared_ptr` in the same way. A member declared without initializer default-constructs its implementation, e.g. `LinearImpl()`, which does not exist and fails to compile. The members are thus initialized to `nullptr`, an empty holder, and assigned in the constructor.

In Python, `LayerNorm(n_embd)` normalizes the last dimension of the input, of size `n_embd`. The C++ `LayerNorm` takes the list of normalized dimensions, so the code passes the singleton list `{n_embd}`.

### Final GPTlite Model

The model sums the token and the position embeddings, applies the blocks, the final layer normalization and the linear layer, and returns the logits in the shape (B, C, S) that the cross-entropy loss expects:

```cpp
struct GPTlite : nn::Module {

  GPTlite(int vocab_size){
    token_embedding_table = nn::Embedding(vocab_size, n_embd);
    position_embedding_table = nn::Embedding(block_size, n_embd);
    blocks = nn::Sequential();
    for (int i=0; i<n_layer; i++)
      blocks->push_back( Block(n_embd, n_head) );
		
    ln = nn::LayerNorm(  std::vector<int64_t> {n_embd} );
    lm_head = nn::Linear( nn::LinearOptions(n_embd, vocab_size).bias(false)  );

    register_module("token_embedding_table", token_embedding_table);
    register_module("position_embedding_table", position_embedding_table);
    register_module("blocks", blocks);
    register_module("ln", ln);
    register_module("lm_head", lm_head);
  }


  Tensor forward(Tensor idx){
    int S = idx.size(1);
    Tensor tok_emb = token_embedding_table(idx); //shape (B,S,E)
    Tensor pos_emb = position_embedding_table(torch::arange(S).to( idx.device() )); // (S,E)
    Tensor x = tok_emb + pos_emb; //shape (B,S,E)
    x = blocks->forward(x);
    x = ln(x);
    Tensor logits = lm_head(x); //shape (B,S,C)
    return logits.permute({0,2,1}); //shape (B,C,S)
  }

  nn::Embedding token_embedding_table{nullptr}, position_embedding_table{nullptr};
  nn::Sequential blocks;
  nn::LayerNorm ln{nullptr};
  nn::Linear lm_head{nullptr};
};
```

## Benchmark Model on LibTorch C++ {#benchmark-model}

The benchmark model is a DNN with `L` linear layers of width `W`, each followed by a ReLU. `benchmark.h` defines it as:

```cpp
#pragma once
#include <torch/torch.h>

struct BenchmarkModel : torch::nn::Module {
  /// DNN with L layers and W neurons per layer

  BenchmarkModel(int64_t W, int64_t L, int64_t in_size, int64_t out_size, torch::Device device){
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

  torch::Tensor forward(torch::Tensor input) {
    return layers->forward(input);
  }

  torch::nn::Sequential layers;
};
```

## Main Loop

`main.cpp` benchmarks the training and the inference of a model on random inputs:

```cpp
#include "gptlite.h"
#include "benchmark.h"

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

LibTorch has no equivalent of `torch.set_default_device`, so the code creates the inputs on the device and moves the model there. Since all the parameters, buffers and submodules are registered, `model.to(device)` moves the whole model. `benchmark_train` and `benchmark_inference` run warm-up iterations, then time a fixed number of iterations, each on one batch; the code calls them epochs. The template parameter `ModelType` lets the same function benchmark any model:

```cpp
const uint warmup_epochs = 30;  // number of epochs to run before benchmarking
const uint benchmark_epochs = 30;  // number of epochs to benchmark

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

`benchmark_inference` calls `model.eval()`, runs only forward passes, and disables the gradients with `torch::NoGradGuard`, the equivalent of `with torch.no_grad()` in Python. A second template parameter, the type of the input, supports the TorchScript models of the next section:

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

## TorchScript: Python for Training, C++ for Inference

[TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) exports a model trained in Python to a file that a C++ program loads and runs without Python. The model is exported as `model_jit.pt` with:

```python
model_jit = torch.jit.script(model) 
# model_jit = torch.jit.trace(model, (x))
model_jit.save('model_jit.pt')
```

Scripting compiles the Python code of the module, including its control flow, so the exported model handles any input. Tracing records the operations of one forward pass on an example input, so a branch that depends on the data stays fixed to the path of that input. `torch.jit.script` fails on a model compiled with `torch.compile`, with the error `Compiled functions can't take variable number of arguments or use keyword-only arguments with defaults`, so the benchmark exports the models without `torch.compile`.

The C++ program loads and runs the model as in the [LibTorch documentation](https://pytorch.org/tutorials/advanced/cpp_export.html):

```cpp
#include <torch/script.h>
using JitModule = torch::jit::Module;
using JitInput  = std::vector<torch::jit::IValue>;

JitModule model = torch::jit::load("model_jit.pt").to(device);
benchmark_inference<JitModule, JitInput>(model, {x});
```

The model has the type `torch::jit::Module`, and its input the type `std::vector<torch::jit::IValue>`, instead of `torch::nn::Module` and `torch::Tensor`; hence the template parameters of `benchmark_inference`.

## Compilation

CMake builds the binaries, following the [instructions in the LibTorch documentation](https://pytorch.org/cppdocs/installing.html), with this `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torch_cpp_benchmark)

# Fix error "CMAKE_CUDA_ARCHITECTURES must be non-empty if set."
set(CMAKE_CUDA_ARCHITECTURES "native")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(SOURCE_FILES gptlite.h benchmark.h main.cpp)

add_executable(main ${SOURCE_FILES} )
target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 17)
``` 

CMake runs in Release mode, for the compiler optimizations, with the path of LibTorch and two optional flags that enable the cuDNN and cuSPARSELt libraries:

```shell
cmake .. \
 -DCMAKE_BUILD_TYPE=Release \
 -DCAFFE2_USE_CUDNN=1 -DCAFFE2_USE_CUSPARSELT=1 \
 -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`
```

## Benchmark

The benchmark measures the throughput, in training or inference iterations per second, and the GPU memory of three models:
1. GPTlite with the dimensions of GPT-3 Small, about 87M parameters with its vocabulary of 65 characters, on a batch of one sequence of 2048 tokens;
2. a deep DNN of 2048 layers of width 256, on batches of 2048 samples of size 256;
3. a wide DNN of 3 layers of width 8192, on batches of 2048 samples of size 8192.

Each model runs on PyTorch 1.13.1 and 2.1, on LibTorch 2.1, and on PyTorch 2.1 with `torch.compile`. TorchScript runs only the inference, in C++, of a model exported from PyTorch 2.1. The caching allocators of PyTorch and LibTorch keep freed GPU memory reserved for later allocations, so the memory measured for a model depends on the models allocated before it in the same process. Each value was thus measured in a separate run. The results for GPTlite are:

{: .picture}
<img width="88.1%" height="88.1%" src="{{ site.assets }}/gptlite-cpp/benchmark_gptlite.svg"/>

{: .caption}
Memory usage and throughput of GPTlite Small in PyTorch 1.13.1, PyTorch 2.1, C++ LibTorch 2.1 and TorchScript 2.1 (inference only).

The results for the DNNs are in <a href="{{ site.assets }}/gptlite-cpp/benchmark_wide.svg">benchmark_wide.svg</a> and <a href="{{ site.assets }}/gptlite-cpp/benchmark_deep.svg">benchmark_deep.svg</a>. The runs that train and then infer use 3.6 to 10.4 times the GPU memory of the inference-only runs of TorchScript (purple bars): training also stores the gradients, the states of the optimizer, and the activations that the backward pass needs.

PyTorch 2.1 is 0% to 2% faster than PyTorch 1.13.1 (orange and blue bars), except for the training of the deep DNN, which is 19% faster. LibTorch is up to 11% faster than PyTorch 2.1 (green and orange bars), for the inference of the wide DNN, and as fast for the inference of the deep DNN. The inference with TorchScript is within 1% of LibTorch for the DNNs, and 5% slower for GPTlite.

On GPTlite, PyTorch 2.1 without `torch.compile` trains at 1.41 and infers at 4.28 sequences per second. With `torch.compile`, the throughput is:

<table>
  <thead>
    <tr><th rowspan="2">throughput (samples/sec)</th><th colspan="2"><code>fullgraph=False</code></th><th colspan="2"><code>fullgraph=True</code></th></tr>
    <tr><th>train</th><th>inference</th><th>train</th><th>inference</th></tr>
  </thead>
  <tbody>
    <tr><td><code>mode='default'</code></td><td>1.591</td><td>5.124</td><td>1.581</td><td>5.11</td></tr>
    <tr><td><code>mode='reduce-overhead'</code></td><td>1.625</td><td>5.256</td><td>1.624</td><td>5.258</td></tr>
    <tr><td><code>mode='max-autotune'</code></td><td>1.636</td><td>5.258</td><td>1.635</td><td>5.26</td></tr>
  </tbody>
</table>

`torch.compile` increases the throughput over PyTorch 2.1 by 12% to 16% in training and by 19% to 23% in inference, and is 8% to 16% faster than LibTorch. The three modes differ by less than 3%: `max-autotune` gains less than 1% over `reduce-overhead`, for a much longer compilation. `fullgraph=True` changes the throughput by less than 1%.

A model trained in PyTorch thus runs from C++, through TorchScript, within 5% of the speed of the same model written with LibTorch. TorchScript is now deprecated in favour of [torch.export](https://docs.pytorch.org/docs/stable/export.html). [AOTInductor](https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html) compiles an exported model ahead of time, with the kernels of TorchInductor, into a package that a C++ program loads with `torch::inductor::AOTIModelPackageLoader` and runs without Python.

The code is in the [GPTlite-cpp repo](https://github.com/{{ site.repository }}/tree/master/assets/gptlite-cpp).
