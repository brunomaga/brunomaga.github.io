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

The main argument for using KE is that training a smaller model from a larger one yields better results than training the smaller model alone. There are many claims, but the rationale is that - e.g. in the multi-class classification use case - the soft labels that are output from a larger model are much better indicators of the input than the hard labels usually passed by humans. This means that the small model will learn better as its objective has better/cleaner labels. Quick example: on a 2-class (dog, cat) classifier, an image of a dog-looking cat has `[0,1]` labels when user provided but `[0.4, 0.6]` when passed by a trained model, which represents better the labels distribution. This in practice avoids confusion on the smaller model and saves capacity for the real task as hand (instead of the task of *cleaning* noise). 

When training the smaller model, we will use KL-divergenge as the ~~metric~~ distance to minimize. There is also the claim that Mean Square Error is a better metric for Knowledge Distillation (in [Comparing Kullback-Leibler Divergence and Mean Squared Error Loss
in Knowledge Distillation](https://arxiv.org/pdf/2105.08919.pdf) ) but we ignore that for now. 

There's also a **Temperature parameter** that scales the teacher and student logits to control the convergeance of the learning. This is better detailed in [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531).

In the [NLLLoss documentation](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss): 
"The input given through a forward call is expected to contain log-probabilities of each class. [...] Obtaining log-probabilities in a neural network is easily achieved by adding a LogSoftmax layer in the last layer of your network. You may use CrossEntropyLoss instead, if you prefer not to add an extra layer.."

When using [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.h), the input is expected to contain the unnormalized logits for each class"


Finally, [KL-divergence](https://pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html) expects an `input` and a `target` argument to be passed as log-probability and o

In our example, for simplccity, we simply train a studetn as a smaller version of the teacher. But in practice, any architecture could be used for student and teacher, as long as the objective would be the same.
More importantly, this post covers offline distilation where teacher and student are trained separately. You can also load the teacher in memory, train it, then in `eval` mode use it to train a new student model in `train` mode. But this requires both to be loaded in memory, limiting the maximum size of the models. For that reason, I prefer to train a large teacher as base line, dump all lables, then subsquentely load and train a smaller student alone.

However, for faster runtime, you may you prefer instead to load and train both the teacher and teacher in one run, or you may also have a KD loss function that includes a Mean Square Error of an intermediatte layer of the teacher and the studet. If that's of interest, see the [Knowledge distillation tutorial on pytorch](https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html#knowledge-distillation-run).
