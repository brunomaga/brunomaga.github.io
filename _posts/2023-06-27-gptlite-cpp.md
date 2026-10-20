---
layout: post
title:  "GPT in C++: LibTorch vs. PyTorch, TorchScript and torch.compile"
categories: ["Efficient Inference"]
tags: ["GPT", "LLM", "C++", "LibTorch", "TorchScript", "torch.compile", "benchmarking"]
---

In the recent [PyTorch 2.x release announcement](https://pytorch.org/get-started/pytorch-2.0/), the Torch developers moved parts of PyTorch from a Python interface over a C++ core (highly efficient) to a Python-based implementation that lowers its operations to a small set of primitive operators (developer-friendly). They also announced `torch.compile`, a method to perform [JIT compilation](https://en.wikipedia.org/wiki/Just-in-time_compilation) and optimization of the execution graph that speeds up execution.

So the main questions are: how much faster are C++ model implementations compared to Python ones? If Python is the *de facto* core language for training, can inference be performed efficiently with compiled C++ code? How good is `torch.compile` in practice? To answer them, this post first describes both changes, then implements a GPT model and a simple Deep Neural Network (DNN) in C++ with LibTorch, and benchmarks them against PyTorch, TorchScript and `torch.compile`.

### Change of philosophy: from Python to C++ to Python

Initial releases of PyTorch were mostly written in Python. Until the release of PyTorch 2.x, the belief was that "to keep eager execution at high-performance, we’ve had to move substantial parts of PyTorch internals into C++". In practice, Python has the overhead of the runtime itself, dynamic typing, interpreted code, etc. So moving the PyTorch API to C++ and using Python as a *thin* layer that calls the compiled C++ implementations seemed logical.

Now, with PyTorch 2.x, the developers are moving in the complete opposite direction, claiming that the "philosophy on PyTorch has always been to keep flexibility and hackability our top priority, and performance as a close second". They add that "moving internals into C++ makes them less hackable and increases the barrier of entry for code contributions". So, in practice, the 2000+ operations of the C++ library (counting overloads) are being decomposed into a smaller set, and the "goal is to provide a primitive and stable set of ~250 operators with simplified semantics, called PrimTorch".

In brief, to favour PyTorch contributors who prefer Python over C++, the new compiler of PyTorch 2.x (TorchDynamo, AOTAutograd, PrimTorch and TorchInductor) is implemented in Python only. Also, the operators that a backend needs to implement are limited to a few hundred. Hardware vendors can then focus on their specific implementation of that subset of operators, while the Python code decomposes the higher-level operations into it. Sounds good, but the possibility of training a model using only C++ in the next releases of PyTorch remains uncertain.

### Kernel optimization via torch.compile

Every PyTorch operation launches one or more GPU kernels. Each launch costs a few microseconds of CPU and driver time, and each kernel reads its inputs from GPU memory and writes its outputs back. For large models and batches, these costs are hidden by the computation. However, for small models and batches, they dominate: a forward pass launches hundreds of small kernels, and the GPU spends much of its time idle, waiting for the next kernel launch or for the memory round-trips between kernels.

**Kernel fusion** merges consecutive operations into a single kernel, so that intermediate results stay in registers or on-chip memory instead of going through GPU memory. Typical candidates are the chains of element-wise operations after a matrix multiplication (bias, activation function, dropout, residual addition) and the normalization layers. [FlashAttention](https://arxiv.org/abs/2205.14135) is an extreme example of fusion, as it fuses the whole attention into a single kernel.

The other big announcement of PyTorch 2.x was `torch.compile`, which "makes PyTorch code run faster by JIT-compiling PyTorch code into optimized kernels, all while requiring minimal code changes" (from the [tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)). TorchDynamo captures the graph of the model from the Python bytecode, and TorchInductor generates fused kernels, written in Triton for GPUs. It is a one-line change, `model = torch.compile(model)`, and a change in the shapes of the inputs can trigger a recompilation. The torch compile algorithm runs in three steps:
1. **graph acquisition** will build the execution graph of all PyTorch operations. The graph will be rewritten as a graph of subgraphs: the subgraphs that can be compiled will be flattened, and the others (e.g. with control flow or other unsupported Python code) will fall back to eager mode. Parallel (graph leaves) and sequential modules are now exposed.
2. **graph lowering** will decompose the previous operations into kernels that are native to the specific backend.
3. **graph compilation** will convert the kernels into low-level operations that are specific to the device.

The following diagram illustrates the three steps on a function that applies a convolution, a batch normalization and a ReLU:

{: .picture}
<img width="80.4%" height="80.4%" src="{{ site.assets }}/gptlite-cpp/torch_compile.svg"/>

{: .caption}
A diagram of the three steps of `torch.compile`. Adapted from: [PyTorch 2.0 technology overview](https://pytorch.org/get-started/pytorch-2.0/#technology-overview)
 
Kernel fusion is complex, and there are two settings worth mentioning. The option `mode` selects the trade-off between runtime and memory in the generated graph (from the [docs](https://pytorch.org/get-started/pytorch-2.0/#user-experience) and [API](https://pytorch.org/docs/stable/generated/torch.compile.html#torch.compile)):
- `default` optimizes for large models, with low compile time and no extra memory overhead. It is "a good balance between performance and overhead";
- `reduce-overhead` reduces the Python overhead with [CUDA graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/). It is faster than the previous mode but uses some extra memory, and it helps speed up small models. CUDA graphs allow an execution graph of kernels to be defined beforehand and launched only once from the CPU, instead of having kernels launched individually;
- `max-autotune` leverages Triton-based matrix multiplications and convolutions. It enables CUDA graphs by default. It produces the fastest model, but takes a very long time to compile. `max-autotune-no-cudagraphs` is analogous, but without CUDA graphs.

The flag `fullgraph` specifies whether the compilation outputs a single graph for the whole run, or whether it can be broken into several partial graphs (that can be reused across the main graph). It is only relevant for users who need to squeeze out the very best performance. In practice, this is *similar* to the `inline` logic in the C programming language. These two settings are specified, with their default values, as `torch.compile(model, mode='default', fullgraph=False)`.

Even after fusion, a forward pass launches many kernels from Python, one after the other. The CUDA graphs of the `reduce-overhead` and `max-autotune` modes record the whole sequence of kernel launches once, and replay it with a single launch, removing almost all the CPU overhead. The recording fixes the shapes and memory addresses of all tensors, so the inputs must have static shapes. For text generation, this requires a static KV cache (the cache of the keys and values of the previous tokens), preallocated to its maximum length.

Combining these techniques with quantization and speculative decoding, the PyTorch team's [gpt-fast](https://pytorch.org/blog/accelerating-generative-ai-2/) made the decoding of Llama-7B almost 10 times faster, in plain PyTorch. Going one step further, recent work such as [Look Ma, No Bubbles!](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) fuses a whole forward pass into a single persistent **megakernel**, removing the gaps between kernels altogether for low-latency decoding.

## About this post

This post benchmarks and analyses several implementations of ML training and inference performed on different backends: PyTorch 1.13.1 (Python), PyTorch 2.1 (Python), LibTorch 2.1 (C++), TorchScript 2.1 (Python and C++), and PyTorch 2.1 with `torch.compile` (Python).
It looks at two different testbenches, so readers may pick one based on their level of expertise and interest:
1. the small variant of the GPT-3 model, built as the GPTlite model introduced in [Building a GPT model in PyTorch from scratch]({{ site.baseurl }}{% post_url  2023-02-28-gptlite %}), will be detailed in section [GPTlite on LibTorch C++](#gptlite-model). This is a complex example that is specific to the use case of large language models;
2. a Deep Neural Network of arbitrary width and depth, in section [Benchmark Model on LibTorch C++](#benchmark-model). This is a very simple example that will be used to benchmark the metrics on models of varying depths and widths. It is mostly suitable for those who are interested in performance modeling, ML engineering, and the C++/Python/TorchScript comparison.

The two models are illustrated below:

{: .picture}
<img width="25.3%" height="25.3%" src="{{ site.assets }}/gptlite-cpp/gptlite_compact.svg"/>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
<img width="25.8%" height="25.8%" src="{{ site.assets }}/gptlite-cpp/benchmark_model.svg"/>

{: .caption}
This post details the C++ implementation of the GPT-3 Small model with N decoder blocks (left), and of a simple Deep Neural Network with L layers of dimensionality W, each followed by a ReLU (right). The C++, PyTorch, TorchScript and `torch.compile` implementations of both models are then benchmarked. Left diagram adapted from: [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

## GPTlite on LibTorch C++ {#gptlite-model}

The post starts with the GPT implementation in C++. It uses LibTorch, the C++ distribution of PyTorch, whose C++ frontend provides an API as close as possible to the Python one. The subsections that follow match exactly the structure of the post with the GPTlite implementation in Python.

### Hyperparameters

GPTlite will be written in the header-only format in the file `gptlite.h`. It starts with the hyperparameter declarations:

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

As a reminder, the original formulation of the multi-head shared attention is the following, where $$W^Q$$, $$W^K$$ and $$W^V$$ are matrices / projections / linear layers:

$$
MultiHead(Q, K, V ) = Concat(head_1, ..., head_h)W^O
$$

$$
\text{where } head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
$$

$$
\text{where } Attention(Q,K,V) = softmax \left( \frac{QK^T}{\sqrt{d_k}} \right) \, V
$$


where $$h$$ is the number of heads, $$d_k$$ is the dimension of the keys of each head and $$W^O$$ is the output projection. Together with the lower-triangular mask (`tril`), this is the underlying structure of each of the N attention blocks in the model:

{: .picture}
<img width="25.3%" height="25.3%" src="{{ site.assets }}/gptlite/gptlite_attention.svg"/>

{: .caption}
The multi-head attention module in the GPTlite model, emphasized in red. Adapted from: [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

The C++ code is analogous to the Python implementation. It starts by defining a single attention head:

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

To keep the code small and clean, `Head` and all the modules that follow will be defined as a `struct` and not as a `class`, so that all members are public by default instead of private.

Note the `register_module` method, which is not needed in the Python implementation. Why is it needed? In practice, C++ has no runtime [reflection](https://en.wikipedia.org/wiki/Reflective_programming), so it cannot iterate over a class's variables unless they are declared somewhere. However, LibTorch needs this iterator feature to iterate over class members for e.g. parameter count, recursive copy of submodules to a device, update of the weights of all heads by the optimizer, etc.

There are two options to create this iterator, and this post uses both:
1. All modules can be kept inside a LibTorch container such as `nn::Sequential` or `nn::ModuleList`. Applying a function to the container will recursively apply it to every module inside;
2. `register_parameter`, `register_buffer` and `register_module` can be called to register parameters, buffers or modules during initialization, and LibTorch will keep track of these internally.

Also, in the code above, `register_buffer` is called on tril because it is a tensor that is not a parameter but a state, i.e. torch will not record its gradients.

Finally, LibTorch does not allow named arguments as in Python (e.g. `bias=False`), so these cannot be passed *directly*. The possible constructors are `Linear(in_features, out_features)` or `Linear(LinearOptions(in_features, out_features).bias(false))`, so when any named parameters need to be passed, the second constructor is used, with all options wrapped inside `LinearOptions`.

The output of all heads is now combined (concatenated) into the multi-head shared-attention module:

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

Again, `nn::ModuleList` is used as a container, instead of a standard-library container, so that the heads are registered. Containers in C++ are declared for a fixed element type. So, the tricky bit here is that `nn::ModuleList` stores elements of type `nn::Module`, which need to be cast dynamically to their derived type `Head` with `module->as<Head>()` before calling the internal members of the instantiated `Head`.

### Feed Forward Network

The feed-forward network is a two-layer Deep Neural Network that follows the attention in each block, as illustrated below:

{: .picture}
<img width="25.3%" height="25.3%" src="{{ site.assets }}/gptlite/gptlite_feedforward.svg"/>

{: .caption}
The feed forward network in the GPTlite model, emphasized in red. Adapted from: [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

It is pretty straightforward to implement:

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

The sequence of a multi-head attention and a feedforward module is called a GPT **block**. Similarly to the Python implementation, the block adds [skip connections](https://en.wikipedia.org/wiki/Residual_neural_network), and normalization before the attention and the feed-forward network, as illustrated below:

{: .picture}
<img width="25.5%" height="25.5%" src="{{ site.assets }}/gptlite/gptlite_blocks.svg"/>

{: .caption}
The GPT block(s) in the GPTlite model, emphasized in red. Adapted from: [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

The block is then implemented as:

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

The code uses `shared_ptr` to wrap its classes. This is not accidental: all LibTorch modules hold a shared pointer to the implementation of a given class. Thus, all `torch::nn` modules can be passed by value directly. E.g. the linear layer `nn::Linear` is a wrapper around `std::shared_ptr<nn::LinearImpl>`, where `nn::LinearImpl` is the implementation.

Because of this, the documentation suggests initializing modules with `nullptr` as the default value of the pointer, and initializing the implementation dynamically later when needed. This is because the alternative of not initializing the pointer would call the default constructor, e.g. `Linear()`, which is not defined, and lead to a compilation error.

There is also a subtle difference in the `LayerNorm` initialization. By design, `LayerNorm` accepts a list of normalized dimensions as input. Alternatively, in the Python implementation, when a single `int` value is passed, only the last dimension of the input is normalized, and it is expected to be of that size. However, in C++, `LayerNorm` does not include a constructor with a single integer argument, so the general constructor has to be used, with the dimension passed as a singleton list.

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

A simple benchmark model is used, which is a DNN with `L` layers of width `W` and a [ReLU](https://en.wikipedia.org/wiki/Rectified_linear_unit) activation after each layer. It is defined in `benchmark.h` as:

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

The `main.cpp` file will contain a loop that benchmarks the train and inference operations of a model on a random input:

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

As an important remark, LibTorch does not include a C++ equivalent to `torch.set_default_device`, so every datapoint and module has to be moved to the GPU manually. And because every parameter, buffer and module was registered previously, doing `model.to(device)` will recursively copy all the contents of the model to the device. The final functions `benchmark_train` and `benchmark_inference` benchmark several train and inference epochs, respectively, after a few warm-up epochs that keep one-time costs (e.g. memory allocations) out of the measurement. The C++ implementation is analogous to its Python counterpart, but uses a templated `typename ModelType` to cover all possible model implementations:

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

The implementation of `benchmark_inference` is a much simpler loop, with `model.eval()` instead, the `torch::NoGradGuard` variable (equivalent to `with torch.no_grad()` in Python), and only a forward pass in the epoch loop. However, an extra template type is added for the input data, to support the TorchScript-based inference discussed in the next section:

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

In ideal scenarios, one would want the flexibility and development speed of Python, with the low memory footprint and high efficiency of C++. This is possible with [TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html), which turns PyTorch code into a serializable and optimizable model that can be loaded and run without any Python dependency. To do that, the model `model` is trained in Python and output as the binary file `model_jit.pt`, via:

```python
model_jit = torch.jit.script(model) 
# model_jit = torch.jit.trace(model, (x))
model_jit.save('model_jit.pt')
```

The main difference between `script` and `trace` is that scripting computes the execution graph by looking at the model definition (`nn.Module`), while tracing runs a forward pass and freezes the resulting execution graph. Thus, tracing enforces a deterministic execution (the control flow is not recorded) and is limited to PyTorch tensors and PyTorch operations in the `forward` pass. In case of doubt, scripting is the better choice for portability, and tracing for performance.

**Important**: `torch.jit.script` cannot be used on a model optimized with `torch.compile`, or it fails with the error `Compiled functions can't take variable number of arguments or use keyword-only arguments with defaults`. `torch.jit.trace(model, (x))` has to be used instead (in PyTorch 2.1; later versions raise an error, so the model is traced before it is compiled, and the TorchScript file does not hold the compiled kernels).

On the inference side, in C++, the following code [follows the LibTorch documentation](https://pytorch.org/tutorials/advanced/cpp_export.html) to load and run inference on that model:

```cpp
#include <torch/script.h>
using JitModule = torch::jit::Module;
using JitInput  = std::vector<torch::jit::IValue>;

JitModule model = torch::jit::load("model_jit.pt").to(device);
benchmark_inference<JitModule, JitInput>(model, {x});
```

Note that the types of the model and input data are not `torch::nn::Module` and `torch::Tensor` as before. Instead, they are `torch::jit::Module` and `std::vector<torch::jit::IValue>`, respectively. This justifies the use of templates in the definition of `benchmark_inference`.

## Compilation

The build follows the [instructions in the LibTorch documentation](https://pytorch.org/cppdocs/installing.html) and uses the [CMake](https://en.wikipedia.org/wiki/CMake) build system to generate the binaries. The `CMakeLists.txt` is:

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

and cmake is run in Release mode (for compile optimizations), specifying the path of LibTorch and (optionally) two extra flags to compile the code with the cuDNN and cuSPARSELt libraries:
```shell
cmake .. \
 -DCMAKE_BUILD_TYPE=Release \
 -DCAFFE2_USE_CUDNN=1 -DCAFFE2_USE_CUSPARSELT=1 \
 -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'`
```

## Benchmark

The benchmark compared the throughput (iterations/sec) and GPU memory usage (GBs) of three distinct models:
1. the small variant of GPTlite, on a batch of one sequence of 2048 tokens;
2. a deep benchmark model made of 2048 layers with 256 activations per layer, with input and output of size 256, on batches of 2048 samples; and
3. a wide benchmark model of 3 layers with 8192 activations each, with input and output of size 8192, on batches of 2048 samples.

So, in practice, the benchmark tests a very deep DNN, a very shallow DNN and a large language model. For each model, it tested:
- the Python PyTorch implementation of training and inference, on versions 1.13.1 and 2.1;
- the C++ LibTorch 2.1 implementation of train and inference;
- the TorchScript combo, using PyTorch 2.1 to train and output the model, and C++ LibTorch 2.1 to load it and perform inference; and
- the train and inference using `torch.compile` on PyTorch 2.1.

As an important remark, both the Python and C++ implementations of Torch keep memory reserved on the GPU when several models are allocated in the same run, as the deallocation returns the memory to their [caching memory allocators](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management). For that reason, a new run was executed per benchmark value.
The results for the GPTlite model are the following: 

{: .picture}
<img width="88.1%" height="88.1%" src="{{ site.assets }}/gptlite-cpp/benchmark_gptlite.svg"/>

{: .caption}
Memory usage and throughput of GPTlite Small in PyTorch 1.13.1, PyTorch 2.1, C++ LibTorch 2.1 and TorchScript 2.1 (inference only).

The results for the benchmark models are in <a href="{{ site.assets }}/gptlite-cpp/benchmark_wide.svg">benchmark_wide.svg</a> and <a href="{{ site.assets }}/gptlite-cpp/benchmark_deep.svg">benchmark_deep.svg</a>. Looking at the GPU memory usage, there is a much smaller memory requirement for inference-only runs (purple bars) compared to runs that performed train and inference (blue, orange, and green bars). This is expected, due to the gradients, optimizer values and activations stored for training. Training leads to an increase in memory in the order of 4x to 10x.

Looking at the overall performance, there is a gain of up to 19% in throughput when moving from PyTorch 1.13.1 to 2.1 (blue to orange bars). There is also a small throughput increase of up to 11% when moving from PyTorch 2.1 to its C++ LibTorch equivalent (from orange to green bars), explained by the Python overhead (runtime, dynamic typing, etc.). Finally, the difference in inference performance between the pure C++ implementation and the TorchScript implementation (train in Python, inference in C++) is negligible for the benchmark models, and about 5% for GPTlite. Thus, TorchScript does a pretty good job at (de)serializing the model.

The speed-up gains from using `torch.compile` were also tested on the GPTlite model. The base case without `torch.compile` performs the PyTorch-based training at a throughput of 1.41 samples/sec and inference at 4.28 samples/sec. The following table compares it with the same model compiled in different modes and with(out) a full graph:

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

The final message is: **running PyTorch 2.1 does not incur a massive loss of performance compared to LibTorch C++**, and this justifies the move of the Torch team back to Python, to ease development. Also, **it is viable to perform training on PyTorch and inference on compiled LibTorch to fit light hardware setups** that do not include the Python runtime. However, **`torch.compile` increases the training throughput by 12% to 16%, and the inference throughput by 19% to 23%**, compared to PyTorch (8% to 16% compared to LibTorch), and this is the most efficient setup for most users. In terms of compilation settings, there was no major visible gain in using a full torch graph in the compilation, and `reduce-overhead` yielded the best balance between compilation time, memory and runtime.

Now the main questions are: is C++ LibTorch going to become deprecated and eventually disappear? If not, will `torch.compile` allow the export of compiled models with `torch.jit.trace` / `.script` so that they can be loaded and run on LibTorch C++?

As a side note, later releases answered the second question in another way. TorchScript is now deprecated in favour of [torch.export](https://docs.pytorch.org/docs/stable/export.html), and [AOTInductor](https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html) compiles an exported model ahead of time, with the kernels of TorchInductor, into a package that a C++ program loads and runs without Python.

To find more about the topic, the paper [PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation](https://docs.pytorch.org/assets/pytorch2-2.pdf) describes TorchDynamo and TorchInductor, the two extensions of PyTorch that implement `torch.compile`. Also, the tutorial [Using the PyTorch C++ Frontend](https://docs.pytorch.org/tutorials/advanced/cpp_frontend.html) covers the LibTorch API used in this post, from the module holders to the training loop.

The code to replicate these results is in the [GPTlite-cpp repo](https://github.com/{{ site.repository }}/tree/master/assets/gptlite-cpp).
