---
layout: post
title:  "Building a GPT model in C++ LibTorch, and benchmarking against PyTorch and TorchScript"
categories: [machine learning, Transformer, GPT, LLM, C++, TorchScript]
tags: [machinelearning]
---

In the recent [Pytorch 2.x release announcement](https://pytorch.org/get-started/pytorch-2.0/), the developers stated that "to keep eager execution at high-performance, we’ve had to move substantial parts of PyTorch internals into C++." I was keen to measure the new efficiency improvements that came from porting much of the python code into C++ primitives. In general, we know that python's execution of interpreted code with dynamic typing on a runtime is not efficient. And in many use cases, using C++ compiled code is necessary for e.g. embedded systems, low memory usage and systems without the python runtime installed. However, C++ code is harder and slow to write compared to python. The question is: is it worh it? How much faster are the C++ model implementations compared to Python? 

In this post, we will look on how to implement in C++ the GPT2 model introduced in [Building a GPT model from scratch]({{ site.baseurl }}{% post_url  2023-02-28-GPT-lite %}) and a Deep Neural Network of arbitrary width and depth. We will then benchmark these two models on three distinct implementations:
- the train and inference steps using the original python implementation in python 1.3.1 and 2.1.0;
- the train and inference steps using the C++ LibTorch 2.1.0 implementation;
- the inference step, using [TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) 2.1.0 to output a model trained in python, and then load it with C++ Libtorch to perform inference;

{: style="text-align:center; font-size: small;"}
<img width="20%" height="20%" src="/assets/GPT-lite/gpt_lite_compact.png"/>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
<img width="22%" height="22%" src="/assets/GPT-lite-cpp/benchmark_model.png"/>

{: style="text-align:center; font-size: small;"}
In this post, we will detail and benchmark the C++ implementation of a [small variant of the GPT2 model]({{ site.baseurl }}{% post_url  2023-02-28-GPT-lite %}) with N decoder blocks (left), and of a Deep Neural Network with L layers of dimensionality W (right). Then we will benchmark the C++, PyTorch and TorchScript implementations.

### GPTlite on LibTorch C++

We will start with the GPT implementation in C++. The sections that follow match exactly the structure of the [post with the GPTlite implementation in Python]({{ site.baseurl }}{% post_url  2023-02-28-GPT-lite %})

#### Hyperparameters

Our GPTlite will be written in the header-only format in the file `gptlite.h`. We start with the hyperparameter declarations:

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

// dropout rate (variable p) for dropout units, renamed to avoid ambiguity
const float dropout_p = 0.1;

// namespace and data type aliases
namespace nn = torch::nn;
using Tensor = torch::Tensor;
```

#### Multi-Head Masked Attention

Remember the original formulation of the multi-head shared attention heads, where $$W^Q$$, $$W^K$$ and $$W^V$$ are matrices / projections / linear layers:

$$
MultiHead(Q, K, V ) = Concat(head_1, ..., head_h)W^O
$$

$$
\text{where } head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
$$

$$
\text{where } Attention(Q,K,V) = softmax \left( \frac{QK^T}{\sqrt{d_k}} \right) \, V
$$


Together with the upper-diagonal mask, this is the underlying structure of each of the N attention blocks in the model:

{: style="text-align:center; font-size: small;"}
<img width="19%" height="19%" src="/assets/GPT-lite/gpt_lite_attention.png"/>

{: style="text-align:center; font-size: small;"}
The multi-head (Nx) attention module in the GPTlite model, emphasized in red.

The C++ code is analogous to the python implementation. We start by defining a single attention head:

```cpp
struct Head : nn::Module {
  Head(int head_size) {
    int head_size = head_size;
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

}
```

In order to keep the code small and clean, `Head` and all our modules that follow will be defined as a `struct` and not as a `class`, so that all members are public and not private by default.
Note the `register_module` operator that is not needed in the python implementation. Why do we need this? In practice, C++ has no reflection, so it cannot iterate over a class variables, unless they're declared somewhere. However, we need this iterator feature, so that LibTorch can iterate class members for e.g. parameter count, recursive copy of submodules to a device, back propagation of weights among several heads, etc. There are two options to create this iterator, and in this post we will use both:
1. We can keep all modules inside an LibTorch container such `nn::Sequential` or `nn::ModuleList`. Applying a function to the container will recursively apply it to every module inside;
2. We can call `register_parameter`, `register_buffer` and `register_module` to register parameters, buffers or modules during initialization, and LibTorch will keep track of these internally.

Also, in the code above, we do `register_buffer` on tril because it is a tensor that is not a parameter, but a state, i.e. torch will not record its gradients.

Finally, LibTorch does not allow named arguments like in Python e.g. `bias=False`, so these cannot be passed *directly*. The possible constructors are `Linear(in_features, out_features)` or `Linear(LinearOptions(in_features, out_features).bias(False))`, so when we need to pass any named parameters, we use the second constructor and wrap all options inside `LinearOptions`.

We will now combine (concatenate) the output of all heads into our multi-head shared-attention module:

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

Again, we used `nn::ModuleList` as a container, instead of any std-library container. Containers in C++ are declared for a given fixed element type. So, the tricky bit here is that `nn::ModuleList` stores elements of type `nn::Module` that needs to be casted dynamically to its base type `Head` with `module->as<Head>()` before calling the internal members of the instantiated `Head`.

#### Feed Forward Network

The Feed-forward network is a two-layer Deep Neural Network and is pretty straighforward to implement:

{: style="text-align:center; font-size: small;"}
<img width="19%" height="19%" src="/assets/GPT-lite/gpt_lite_feedforward.png"/>

{: style="text-align:center; font-size: small;"}
The feed forward network in the GPTlite model, emphasized in red.

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

#### The GPT Block

We'll call GPT *block* the sequence of a multi-head attention and a feedforward module. Similarly to the python implementation, we add skip connections and normalization before the attention and feed-forward network.

{: style="text-align:center; font-size: small;"}
<img width="19%" height="19%" src="/assets/GPT-lite/gpt_lite_blocks.png"/>

{: style="text-align:center; font-size: small;"}
The GPT block(s) in the GPTlite model, emphasized in red.

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

You will notice we will be using `shared_ptr` to wrap our classes. It is not accidental. In fact, all LibTorch modules are a shared pointer to the implementation of a given class. Thus, all `torch::nn` modules can be passed by value directly. E.g. the linear layer `nn::Linear` is just an alias for `std::shared_ptr<nn::LinearImpl>`, where `nn::LinearImpl` is the implementation. Because of this, the documentation suggests initializing modules with `nullptr` as the default value of the pointer, and initializing the implementation dynamically later when needed. This is because the alternative of not initializing the pointer would call the default constructor e.g. `Linear()` which is not defined, and lead to a compilation error.

There's also a subtle difference in the `LayerNorm` initialization. By design, `LayerNorm` accepts a list of normalized dimensions as input. Alternatively, in the python implementation, when a single `int` value is passed, only the last dimension of the input is normalized, and will be resized to the integer value. However, in C++, `LayerNorm` does not include the constructor initialization with a single integer argument, so we have to use the general constructor and pass it as a singleton list.

#### Final GPTlite Model

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


### Benchmark Model on LibTorch C++

We will define a simple benchmark model, which is simply a DNN with `L` layers of width `W`, input of size `W`, and a categorical output of `W` possible classes, with a ReLu activation between layers. This is defined in `benchmark.h` as:

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

### Main Benchmark loop

Our `main.cpp` file will contain a loop that will benchmark the train and inference operations of a model for a random input:

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


As an important remark, LibTorch does not include a C++ equivalent to `torch.set_default_device`, so we have to manually move to the GPU every datapoint and module. And because we registered every parameter, buffer and module previously, doing `model.to(device)` will recursively copy all the contents in the model to the device. The final functions `benchmark_train` and `benchmark_inference` perform the benchmark of method several train and inference epochs, respectively. The C++ implementation is analogous to its python counterpart, however we'll use a templated `typename ModelType` to cover all possible model implementations:

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

The implementation of `benchmark_inference` is a much simpler loop with `model.eval()` instead, the `torch::NoGradGuard` variable (equivalent to `with torch.no_grad()` in python), and only a forward pass in the epochs loop. However, we add an extra templated type for the input data, to support the TorchScript-based inference that we will discuss in the next chapter:

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

### TorchScript: python for training, C++ for inference

In ideal scenarions, we would want the flexibility and speed of development of python, with the low memory footprint and high efficiency of C++. This is possible with [TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html). To do that, we will train the model `model` in python and output it as the binary `model_jit.pt` file, via:

```python
model_jit = torch.jit.script(model) 
# model_jit = torch.jit.trace(model, (x))
model_jit.save('model_jit.pt')
```

[//]: # As a side note, `torch.jit.script` requires optional arguments in python to be explicitly declared with their `typing` type e.g.:
[//]: # ```python
[//]: #   def forward(self, idx, targets: Optional[torch.Tensor]=None):
[//]: # ```
[//]: # instead of:
[//]: # ```python
[//]: #   def forward(self, idx, targets=None):
[//]: # ```

On the inference side, in C++, we [follow the LibTorch documentation](https://pytorch.org/tutorials/advanced/cpp_export.html) and will load and run inference on that model with the following code:

```cpp
#include <torch/script.h>
using JitModule = torch::jit::Module;
using JitInput  = std::vector<torch::jit::IValue>;

JitModule model = torch::jit::load("model_jit.pt").to(device);
benchmark_inference<JitModule, JitInput>(model, {x});
```

Note that the type of the model and input data is not `torch::nn::Module` and `torch::Tensor` as before. Instead, we have `torch::jit::Module` and `std::vector<torch::jit::IValue>`, respectively. This justifies the use of the templates on the definition of `benchmark_interface`.

### Compilation

We follow the [instructions on the LibTorch documentation](https://pytorch.org/cppdocs/installing.html#installing-c-distributions-of-pytorch) and use the CMake build systems to generate our binaries. The `CMakeLists.txt` is:

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

and we will run cmake with 2 extra (optional) flags to compile our C++ code with the cuDNN and cuSPARSELt libraries:
```shell
cmake .. -DCMAKE_BUILD_TYPE=Release -DCAFFE2_USE_CUDNN=1 -DCAFFE2_USE_CUSPARSELT=1
```

### Benchmark

We compared the throughput (samples/sec) and GPU memory usage (GBs) of three distinct implementations: the small variant of GPTlite, a deep benchmark model made of 2048 layers with 256 activations per layer, and a wide benchmark model of 3 layers with 8192 activations each. So we are testing a very deep, a very shallow and a large text generation model. For each model, we tested:
- the python PyTorch implementation of training and inference on python 1.3.1 and python 2.1.0;
- the C++ LibTorch 2.1.0 implementation of train and inference; and
- the TorchScript combo, using PyTorch 2.1.0 to train and output the model, and C++ LibTorch 2.1.0 to load and perform inference. 

As an important remark, I noticed that both the python and C++ implementations of Torch leak memory on the GPU when several models are allocated in the same run, as the deallocation does not clear the memory completely. For that reason, I executed a new run per benchmark value.
The results are the following: 

{: style="text-align:center; font-size: small;"}
<img width="90%" height="90%" src="/assets/GPT-lite-cpp/benchmark_wide.png"/>

{: style="text-align:center; font-size: small;"}
<img width="90%" height="90%" src="/assets/GPT-lite-cpp/benchmark_deep.png"/>

{: style="text-align:center; font-size: small;"}
<img width="90%" height="90%" src="/assets/GPT-lite-cpp/benchmark_gptlite.png"/>

Looking at the memory usage, we see that there is a much smaller memory requirement for inference-only runs (light blue bars), compared to runs that performed train and inference (navy blue, orange, and green bars). This is expected, due to the extra parameters and optimizer values required for training. Training leads to an increase in memory in the order of 4x to 10x. 

Looking at the performance, there is a gain of up to 15% in throughput when moving from PyTorch 1.3.1 to 2.1.0 (navy blue to orange bars), so indeed, porting several PyTorch instructions to C++ really helped in performance, due to less python instruction executed on the python runtime. There is also a small throughput increase of up to 10% when moving from python and PyTorch 2.1.0 to its C++ LibTorch equivalent (from orange to green bars), and this is explained again by the python runtime overhead. Finally, the inference when comparing the pure C++ implementation and the TorchScript implementation (train in python, inference in C++) is neglegible, which means that TorchScript does a pretty good job in (de)serializing the model. All these gains were not visible in the Deep DNN model, and that is something that is counter intuitive to me.

The final message is: **for the best training flexibility and inference efficiency, use PyTorch 2.x to train, LibTorch 2.x to do inference, and TorchScript to glue both**.

And we reached the end of this post! If you want to replicate these results, see the [GPT-lite-cpp repo](https://github.com/brunomaga/brunomaga.github.io/tree/master/assets/GPT-lite-cpp).

