---
layout: post
title:  "Building a GPT model in C++, and benchmarking LibTorch, PyTorch, TorchScript and torch.compile"
categories: ["Efficient Inference"]
tags: ["GPT", "LLM", "C++", "LibTorch", "TorchScript", "torch.compile", "benchmarking"]
---

In the [PyTorch 2.x release announcement](https://pytorch.org/get-started/pytorch-2.0/), the Torch developers announced a change in the structure of PyTorch: from a Python interface over a highly efficient C++ core, to a developer-friendly Python-based implementation that calls a small set of C++ kernels. They also announced `torch.compile`, a method that performs JIT compilation and optimization of the execution graph and greatly speeds up execution.

This raises three questions. How much faster are C++ model implementations compared to Python ones? If Python is the *de facto* core language for training, can inference be performed efficiently with compiled C++ code? And how good is `torch.compile` in practice? 

### Change of philosophy: from Python to C++ to Python

Initial releases of PyTorch were mostly written in Python. Until the release of PyTorch 2.x, the belief was that "to keep eager execution at high-performance, we’ve had to move substantial parts of PyTorch internals into C++". In practice, Python has overheads from the runtime itself, dynamic typing, JIT, interpreted code, etc. Moving the PyTorch API to C++, with Python as a *thin* layer that calls the compiled C++ implementations, was therefore a logical choice.

With PyTorch 2.x, the developers move in the opposite direction. They claim that the "philosophy on PyTorch has always been to keep flexibility and hackability our top priority, and performance as a close second" and that "moving internals into C++ makes them less hackable and increases the barrier of entry for code contributions". In practice, the library of 2000+ operations written in C++ is being reduced, and the "goal is to provide a primitive and stable set of ~250 [C++] operators with simplified semantics, called PrimTorch".

In brief, to favour contributors who prefer Python over C++, the C++ code is limited to a few hundred kernels, and all remaining code is implemented in Python only. Hardware vendors can then focus on their implementation of that subset of C++ methods, while the Python runtime executes the higher-level operations. However, it remains uncertain whether future releases of PyTorch will still allow training a model using only C++.

### Kernel optimization via torch.compile

Every PyTorch operation launches one or more GPU kernels. Each launch costs a few microseconds of CPU and driver time, and each kernel reads its inputs from GPU memory and writes its outputs back. For large models and batches, these costs hide behind the math. For small models and batches, they dominate: a forward pass launches hundreds of small kernels, and the GPU spends much of its time idle, waiting for the CPU to launch the next kernel, or for the memory round-trips between kernels.

*Kernel fusion* merges consecutive operations into a single kernel, so that intermediate results stay in registers or on-chip memory instead of going through GPU memory. Typical candidates are the chains of element-wise operations after a matrix multiplication (bias, activation function, dropout, residual addition) and the normalization layers. FlashAttention is an extreme example of fusion.

The other major announcement of PyTorch 2.x was `torch.compile`, which "makes PyTorch code run faster by JIT-compiling PyTorch code into optimized kernels, all while requiring minimal code changes". TorchDynamo captures the graph of the model from the Python bytecode, and TorchInductor generates fused kernels, written in Triton for GPUs. It is a one-line change: `model = torch.compile(model)`. A change in the shapes of the inputs triggers a recompilation. The compilation runs in three steps:
1. **graph acquisition** builds the execution graph of all PyTorch operations. Nodes that can be combined and optimized together are merged, and the graph is rewritten as a graph of subgraphs. This exposes parallel (graph leaves) and sequential modules.
2. **graph lowering** decomposes the previous operations into kernels that are native to the specific backend.
3. **graph compilation** converts the kernels into low-level operations that are specific to the device.

{: style="text-align:center; font-size: small;"}
<img width="85%" height="85%" src="{{ site.assets }}/gptlite-cpp/torch_compile.svg"/>

{: style="text-align:center; font-size: small;"}
A diagram of the three steps of `torch.compile`. Adapted from: [PyTorch 2.0 technology overview](https://pytorch.org/get-started/pytorch-2.0/#technology-overview)
 
Kernel fusion is complex, and two settings are worth mentioning. The option `mode` selects the trade-off between runtime and memory in the generated graph (from the [docs](https://pytorch.org/get-started/pytorch-2.0/#user-experience) and [API](https://pytorch.org/docs/stable/generated/torch.compile.html#torch.compile)):
- `default` optimizes for large models, with low compile time and no extra memory overhead. It is "a good balance between performance and overhead";
- `reduce-overhead` reduces the Python overhead with [CUDA graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/). It is faster than the previous mode but uses extra memory, and it helps speed up small models. CUDA graphs define an execution graph of kernels beforehand and launch it only once from the CPU, instead of launching each kernel individually;
- `max-autotune` uses Triton-based matrix multiplications and convolutions and enables CUDA graphs by default. It produces the fastest model, but compilation takes a very long time. `max-autotune-no-cudagraphs` is analogous, but without CUDA graphs.

The flag `fullgraph` specifies whether the compilation outputs a single graph for the whole run, or whether the graph can be broken into several partial graphs (that can be reused across the main graph). It is only relevant for users who need the very best performance. In practice, this is *similar* to the `inline` logic in the C programming language. With their default values, these two settings are specified as `torch.compile(model, mode='default', fullgraph=False)`.

Even after fusion, a forward pass launches many kernels from Python, one after the other. A *CUDA graph* records the whole sequence of kernel launches once, and replays it with a single launch, removing almost all the CPU overhead. The recording fixes the shapes and memory addresses of all tensors, so the inputs must have static shapes. For text generation, this requires a preallocated KV cache of maximum length, with attention over the full buffer and a mask for the positions not filled yet. Combining these techniques with quantization and speculative decoding, the PyTorch team's [gpt-fast](https://pytorch.org/blog/accelerating-generative-ai-2/) made the decoding of Llama-7B almost 10 times faster, in plain PyTorch. Going one step further, recent work fuses a whole forward pass into a single persistent *megakernel*, removing the gaps between kernels altogether for low-latency decoding.

## About this post

This post benchmarks and analyses several implementations of ML training and inference on different backends: PyTorch 1.13.1 (Python), PyTorch 2.1 (Python), LibTorch 2.1 (C++), TorchScript 2.1 (Python and C++), and PyTorch 2.1 with `torch.compile` (Python).  
Two testbenches are covered, and readers may pick one based on their expertise and interest:
1. the GPT-3 Small model, detailed in section [GPTlite on LibTorch C++](#gptlite-model). This is a complex example, specific to large language models; 
2. a Deep Neural Network of arbitrary width and depth, in section [Benchmark Model on LibTorch C++](#benchmark-model). This is a very simple example, used to benchmark models of varying depths and widths. It is mostly suitable for readers interested in performance modeling, ML engineering, and the C++/Python/TorchScript comparison.  

{: style="text-align:center; font-size: small;"}
<img width="24%" height="24%" src="{{ site.assets }}/gptlite-cpp/gptlite_compact.svg"/>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
<img width="22%" height="22%" src="{{ site.assets }}/gptlite-cpp/benchmark_model.svg"/>

{: style="text-align:center; font-size: small;"}
This post details the C++ implementation of the GPT-3 Small model with N decoder blocks (left), and of a simple Deep Neural Network with L layers of dimensionality W (right). The left diagram is adapted from [Attention is all you need](https://arxiv.org/abs/1706.03762). The C++, PyTorch, TorchScript and `torch.compile` implementations of both models are then benchmarked.

## GPTlite on LibTorch C++ {#gptlite-model}

The GPT implementation in C++ comes first.

### Hyperparameters

GPTlite is written in the header-only format in the file `gptlite.h`. The file starts with the hyperparameter declarations:

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

Recall the original formulation of the multi-head shared attention, where $$W^Q$$, $$W^K$$ and $$W^V$$ are matrices / projections / linear layers:

$$
MultiHead(Q, K, V ) = Concat(head_1, ..., head_h)W^O
$$

$$
\text{where } head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
$$

$$
\text{where } Attention(Q,K,V) = softmax \left( \frac{QK^T}{\sqrt{d_k}} \right) \, V
$$


Together with the lower-triangular mask (`tril`), this is the underlying structure of each of the N attention blocks in the model:

{: style="text-align:center; font-size: small;"}
<img width="24%" height="24%" src="{{ site.assets }}/gptlite/gptlite_attention.svg"/>

{: style="text-align:center; font-size: small;"}
The multi-head attention module in the GPTlite model, emphasized in red. Adapted from [Attention is all you need](https://arxiv.org/abs/1706.03762).

The C++ code is analogous to the Python implementation. The first module is a single attention head:

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

To keep the code small and clean, `Head` and all following modules are defined as a `struct` and not as a `class`, so that all members are public by default instead of private.
Note the `register_module` operator, which is not needed in the Python implementation. C++ has no reflection, so it cannot iterate over the variables of a class unless they are declared somewhere. However, LibTorch needs to iterate over class members, e.g. for parameter count, recursive copy of submodules to a device, back propagation of weights among several heads, etc. There are two options to make this possible, and this post uses both:
1. Keep all modules inside a LibTorch container such as `nn::Sequential` or `nn::ModuleList`. Applying a function to the container recursively applies it to every module inside;
2. Call `register_parameter`, `register_buffer` and `register_module` to register parameters, buffers or modules during initialization. LibTorch then keeps track of them internally.

In the code above, `register_buffer` is called on tril because tril is a state and not a parameter, i.e. torch does not record its gradients.

Finally, LibTorch does not allow named arguments as in Python (e.g. `bias=False`), so these cannot be passed *directly*. The possible constructors are `Linear(in_features, out_features)` or `Linear(LinearOptions(in_features, out_features).bias(false))`. To pass any named parameter, the second constructor is used, with all options wrapped inside `LinearOptions`.

The multi-head shared-attention module combines (concatenates) the outputs of all heads:

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

Again, `nn::ModuleList` is used as a container, instead of a standard-library container. Containers in C++ are declared for a fixed element type. The tricky detail is that `nn::ModuleList` stores elements of type `nn::Module`. These must be cast dynamically to their derived type `Head` with `module->as<Head>()` before calling the internal members of the instantiated `Head`.

### Feed Forward Network

The feed-forward network is a two-layer Deep Neural Network:

{: style="text-align:center; font-size: small;"}
<img width="24%" height="24%" src="{{ site.assets }}/gptlite/gptlite_feedforward.svg"/>

{: style="text-align:center; font-size: small;"}
The feed forward network in the GPTlite model, emphasized in red. Adapted from [Attention is all you need](https://arxiv.org/abs/1706.03762).

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

A GPT *block* is the sequence of a multi-head attention and a feed-forward module. As in the Python implementation, it includes skip connections, and normalization before the attention and the feed-forward network.

{: style="text-align:center; font-size: small;"}
<img width="24%" height="24%" src="{{ site.assets }}/gptlite/gptlite_blocks.svg"/>

{: style="text-align:center; font-size: small;"}
The GPT block(s) in the GPTlite model, emphasized in red. Adapted from [Attention is all you need](https://arxiv.org/abs/1706.03762).

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

Note the use of `shared_ptr` to wrap the classes. This is intentional: all LibTorch modules are shared pointers to the implementation of a given class. Thus, all `torch::nn` modules can be passed by value directly. E.g. the linear layer `nn::Linear` is an alias for `std::shared_ptr<nn::LinearImpl>`, where `nn::LinearImpl` is the implementation. For this reason, the documentation suggests initializing modules with `nullptr` as the default value of the pointer, and initializing the implementation dynamically later when needed. Leaving the pointer uninitialized would instead call the default constructor, e.g. `Linear()`, which is not defined, and lead to a compilation error.

There is also a subtle difference in the `LayerNorm` initialization. By design, `LayerNorm` accepts a list of normalized dimensions as input. In the Python implementation, when a single `int` value is passed, only the last dimension of the input is normalized, and it is expected to be of that size. However, in C++, `LayerNorm` has no constructor with a single integer argument, so the general constructor is used, with the dimension passed as a singleton list.

### Final GPTlite Model

Putting it all together:

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

The benchmark model is a simple DNN with `L` layers of width `W` and a ReLU activation between layers. It is defined in `benchmark.h` as:

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

## Main loop

The `main.cpp` file contains a loop that benchmarks the train and inference operations of a model on a random input:

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

Note that LibTorch has no C++ equivalent to `torch.set_default_device`, so every datapoint and module must be moved to the GPU manually. Because every parameter, buffer and module was registered previously, `model.to(device)` recursively copies all the contents of the model to the device. The functions `benchmark_train` and `benchmark_inference` benchmark several train and inference epochs, respectively. The C++ implementation is analogous to its Python counterpart, but uses a templated `typename ModelType` to cover all possible model implementations:

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

The implementation of `benchmark_inference` is a simpler loop, with `model.eval()` instead, the `torch::NoGradGuard` variable (equivalent to `with torch.no_grad()` in Python), and only a forward pass in the epoch loop. However, it has an extra template type for the input data, to support the TorchScript-based inference discussed in the next section:

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

## TorchScript: python for training, C++ for inference

Ideally, one would combine the flexibility and development speed of Python with the low memory footprint and high efficiency of C++. [TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) makes this possible. The model `model` is trained in Python and exported as the binary file `model_jit.pt` via:

```python
model_jit = torch.jit.script(model) 
# model_jit = torch.jit.trace(model, (x))
model_jit.save('model_jit.pt')
```

The main difference between `script` and `trace` is that scripting computes the execution graph from the model definition (`nn.Module`), while tracing runs a forward pass and freezes the resulting execution graph. Thus, tracing enforces a deterministic execution (no control flow allowed) and is limited to PyTorch tensors and PyTorch operations in the `forward` pass. In case of doubt, favor scripting for portability, and tracing for performance.

**Important**: `torch.jit.script` cannot be used on a model optimized with `torch.compile`, as it fails with the error `Compiled functions can't take variable number of arguments or use keyword-only arguments with defaults`. Use `torch.jit.trace(model, (x))` instead.

On the inference side, in C++, the model is loaded and run with the following code, as described in the [LibTorch documentation](https://pytorch.org/tutorials/advanced/cpp_export.html):

```cpp
#include <torch/script.h>
using JitModule = torch::jit::Module;
using JitInput  = std::vector<torch::jit::IValue>;

JitModule model = torch::jit::load("model_jit.pt").to(device);
benchmark_inference<JitModule, JitInput>(model, {x});
```

Note that the types of the model and input data are not `torch::nn::Module` and `torch::Tensor` as before, but `torch::jit::Module` and `std::vector<torch::jit::IValue>`, respectively. This justifies the use of templates in the definition of `benchmark_inference`.

## Compilation

The binaries are built with the CMake build system, following the [instructions in the LibTorch documentation](https://pytorch.org/cppdocs/installing.html). The `CMakeLists.txt` is:

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

CMake is run in Release mode (for compile optimizations), with the path of LibTorch and two optional flags to compile the code with the cuDNN and cuSPARSELt libraries:
```shell
cmake .. \
 -DCMAKE_BUILD_TYPE=Release \
 -DCAFFE2_USE_CUDNN=1 -DCAFFE2_USE_CUSPARSELT=1 \
 -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`
```

## Benchmark

The benchmark compares the throughput (samples/sec) and GPU memory usage (GBs) of three distinct models:
1. the small variant of GPTlite;
2. a deep benchmark model made of 2048 layers with 256 activations per layer, with input and output of size 2048; and
3. a wide benchmark model of 3 layers with 8192 activations each, with input and output of size 2048.

In practice, this covers a very deep DNN, a very shallow DNN and a large language model. Each model was tested with:
- the Python PyTorch implementation of training and inference, on versions 1.13.1 and 2.1;
- the C++ LibTorch 2.1 implementation of train and inference;
- the TorchScript combo, using PyTorch 2.1 to train and output the model, and C++ LibTorch 2.1 to load it and perform inference; and
- the train and inference using `torch.compile` on PyTorch 2.1. 

Both the Python and C++ implementations of Torch leak GPU memory when several models are allocated in the same run, as deallocation does not clear the memory completely. For that reason, each benchmark value was measured in a new run.
The results for the GPTlite model are: 

{: style="text-align:center; font-size: small;"}
<img width="90%" height="90%" src="{{ site.assets }}/gptlite-cpp/benchmark_gptlite.svg"/>

The results for the benchmark models are in <a href="{{ site.assets }}/gptlite-cpp/benchmark_wide.svg">benchmark_wide.svg</a> and <a href="{{ site.assets }}/gptlite-cpp/benchmark_deep.svg">benchmark_deep.svg</a>. Inference-only runs (purple bars) require much less GPU memory than runs that perform training and inference (blue, orange, and green bars). This is expected, due to the extra parameters and optimizer values required for training. Training increases memory usage by a factor of 4x to 10x.  

In terms of overall performance, moving from PyTorch 1.13.1 to 2.1 (blue to orange bars) increases throughput by up to 19%. Moving from PyTorch 2.1 to its C++ LibTorch equivalent (orange to green bars) gives a small throughput increase of up to 10%, explained by the Python overhead (runtime, dynamic typing, etc.). Finally, the difference in inference performance between the pure C++ implementation and the TorchScript implementation (train in Python, inference in C++) is negligible. This shows that TorchScript (de)serializes the model efficiently.

The speed-up of `torch.compile` was also measured on the GPTlite model. Without `torch.compile`, PyTorch-based training runs at a throughput of 1.41 samples/sec and inference at 4.28 samples/sec. The following table shows the throughput of the same model compiled in different modes, with and without a full graph:

| throughput (samples/sec)  | `fullgraph=False`, train | `fullgraph=False`, inference | `fullgraph=True`, train | `fullgraph=True`, inference |  
|---|---|---|---|---|
| `mode='default'`  | 1.591 | 5.124	| 1.581 |	5.11 |
| `mode='reduce-overhead'`  | 1.625	| 5.256	| 1.624	| 5.258 |
| `mode='max-autotune'`  | 1.636 |	5.258	| 1.635 |	5.26 |

In summary, **running PyTorch 2.1 does not incur a major loss of performance compared to LibTorch C++**, which justifies the move of the Torch team back to Python to ease development. Also, **it is viable to train on PyTorch and run inference on compiled LibTorch to fit light hardware setups** that do not include the Python runtime. However, **`torch.compile` increases the training throughput by 12% to 16%, and the inference throughput by 19% to 23%**, compared to the Torch/LibTorch implementation, and is the most efficient setup for most users. Regarding compilation settings, a full graph brought no major visible gain, and `reduce-overhead` gave the best balance between compilation time, memory and runtime.

Two questions remain open. Will C++ LibTorch become deprecated and eventually disappear? If not, will `torch.compile` allow exporting compiled models with `torch.jit.trace` / `.script`, so that they can be loaded and run on LibTorch C++? 

To replicate these results, see the [GPTlite-cpp repo](https://github.com/{{ site.repository }}/tree/master/assets/gptlite-cpp).
