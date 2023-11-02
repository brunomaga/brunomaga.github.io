---
layout: post
title:  "Faster inference on a GPT model via model compression and distillation"
categories: [machine learning, Transformer, GPT, DeepSpeed]
tags: [machinelearning]
---

Previously, in [Distributed training of a large GPT model with DeepSpeed]({{ site.baseurl }}{% post_url 2023-08-18-GPT-lite-DeepSpeed %}), we foccused on training a very large model on a distributed network of GPUs, to increase training speedup or model accuracy. In this post, we look at the other side of the spectrum: we will look at techniques for model speedup and compression that allow for a lower runtime and memory footprint during inference. This is particularly relevant for systems that require low latency or low cost to operate.

### Model and dataset

Just like in our [previous post]({{ site.baseurl }}{% post_url 2023-08-18-GPT-lite-DeepSpeed %}), we will define the methods `get_dataset()` and `get_model()` that return a `torch.utils.data.Dataset` and `torch.nn.Module` for the two implementations that we will study in this post: 

- A GPTlite model in <a href="/assets/GPT-lite-DeepSpeed/gptlite.py">`gptlite.py`</a> is the small variant of the GPT2 model ([Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165). This model is trained on the [tiny shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) dataset, and its objective is to predict the next character in a sequence.

```python
def get_dataset():
  from gptlite import load_tiny_shakespeare_data, GPTliteDataset
  train_data, _, vocab_size = load_tiny_shakespeare_data() #load encoded data from text file
  dataset = GPTliteDataset(train_data, gptlite.block_size)
  return dataset, vocab_size

def get_model(vocab_size):
  from gptlite import GPTlite
  return GPTlite(vocab_size)
```

- the Benchark model in <a href="/assets/GPT-lite-DeepSpeed/benchmark.py">`benchmark.py`</a> is a Deep Neural Network with user defined width `W` and number of layers `L`, with input of size `W` and a categorical output of `W` classes:

```python
from benchmark import BenchmarkDataset, BenchmarkModel
get_dataset = lambda W: BenchmarkDataset(W)
get_model = lambda W, L: BenchmarkModel(W, L)
```

{: style="text-align:center; font-size: small;"}
<img width="20%" height="20%" src="/assets/GPT-lite/gpt_lite_compact.png"/>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
<img width="22%" height="22%" src="/assets/GPT-lite-cpp/benchmark_model.png"/>

{: style="text-align:center; font-size: small;"}
The diagram our [GPT2 model]({{ site.baseurl }}{% post_url  2023-02-28-GPT-lite %}) model with N decoder blocks (left), and the benchmark model, a Deep Neural Network with L layers of dimensionality W (right).

### Knowledge Distillation

When training the smaller model, we will use KL-divergenge as the ~~metric~~ distance to minimize. There is also the claim that Mean Square Error is a better metric for Knowledge Distillation (in [Comparing Kullback-Leibler Divergence and Mean Squared Error Loss
in Knowledge Distillation](https://arxiv.org/pdf/2105.08919.pdf) ) but we ignore that for now. 


