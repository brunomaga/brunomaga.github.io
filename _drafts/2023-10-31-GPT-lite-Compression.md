---
layout: post
title:  "Faster inference on a GPT model via model compression and distillation"
categories: [machine learning, Transformer, GPT, DeepSpeed]
tags: [machinelearning]
---

Previously, in [Distributed training of a large GPT model with DeepSpeed]({{ site.baseurl }}{% post_url 2023-08-18-GPT-lite-DeepSpeed %}), we foccused on training a very large model on a distributed network of GPUs, to increase training speedup or model accuracy. In this post, we look at the other side of the spectrum: we will look at techniques for model speedup and compression that allow for a lower runtime and memory footprint during inference. This is particularly relevant for systems that require low latency or low cost to operate.

### Model and dataset

Just like in our [previous post]({{ site.baseurl }}{% post_url 2023-08-18-GPT-lite-DeepSpeed %}), we will define the methods `get_dataset()` and `get_model()` that return a `torch.utils.data.Dataset` and `torch.nn.Module` for the two implementations that we will study in this post:
1. the small variant of the GPT2 model ([Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165), that we call the GPTlite model, <a href="/assets/GPT-lite-DeepSpeed/gptlite.py">`gptlite.py`</a>, trained on the [tiny shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) dataset, whose objective is to generate text by predicting the next character in a sequence;
2. the Benchark model in <a href="/assets/GPT-lite-DeepSpeed/benchmark.py">`benchmark.py`</a>, a Deep Neural Network with user defined width `W` and number of layers `L`, with input of size `W` and a categorical output of `W` classes.

{: style="text-align:center; font-size: small;"}
<img width="20%" height="20%" src="/assets/GPT-lite/gpt_lite_compact.png"/>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
<img width="22%" height="22%" src="/assets/GPT-lite-cpp/benchmark_model.png"/>

{: style="text-align:center; font-size: small;"}
The diagram our [GPT2 model]({{ site.baseurl }}{% post_url  2023-02-28-GPT-lite %}) model with N decoder blocks (left), and the benchmark model, a Deep Neural Network with L layers of dimensionality W (right).

### Knowledge Distillation

Knowledge Distillation (KD) is a technique used to train a smaller model (student) from a larger one (teacher). There are many claims of why we should perform KD instead of training a small model alone, but the main rationale is that the soft labels (distribution of assignments) of the large network is a much better label for the input than the user-provided hard labels, thus the smaller network can now be trained on the same input against a "cleaner" output, and requires less capacity to perform equally *or better* than the larger model.

As a quick example, take the two-label (dog vs cat) classification task. An image of a dog-looking cat will have the groundtruth label distribution `[0,1]` . After training the model, querying the model that that input would yield an output simillat to `[0.4, 0.6]` (i.e. the model believes it's a cat, but could less likely be a dog). In practice, the soft label `[0.4, 0.6]` is a better classification than the hard label `[0,1]` and using this better labels to train a smaller models will allow that student model to be smaller and maybe more accurate (this soft labels avoid confusion on the smaller model and saves model capacity for the real task as hand instead of the task of *cleaning* noise). 

There are several categories of KD methods. We can try to minimized the soft labels between student and teacher models (as the example above), or intermeddiate layers such as logits or feature maps. We can also perform **offline distillation** where we train the teacher first, and then train the student based on the soft labels provided by the pre-trained teacher, or we can perform **online distillation**, where we train both the train and teacher simultaneously. We can use a single teacher of an ensemble of teachers. And we can use a student that is a scaled down version of the teacher, or a completely different architectures. If you are looking for details related to different KD methods, see [Distilling the Knowledge in a Neural Network, Google](https://arxiv.org/abs/1503.02531), [Knowledge distillation in deep learning and its applications](https://peerj.com/articles/cs-474/), and [Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525). 

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="/assets/GPT-lite-Compression/kd_methods.jpg"/>

{: style="text-align:center; font-size: small;"}
 An illustration of the different knowledge distillation categories of methods and the different branches within each category. Source: [Knowledge distillation in deep learning and its applications](https://peerj.com/articles/cs-474/).

Here, we will perform offline distilallion of a student model using only the soft labels of a pre-trained teacher model. This is the simplest and most common use case, as large pre-trained models are readily available online these days, so we can train a smaller version of a very-large model that would be infeasible to train alone. 

As a small nuance, when using a loss function that approximates student-teacher outputs, we will use KL-divergenge as the ~~metric~~ distance to minimize, instead of Cross Entropy (CE). In practice, Cross entropy loss is the same as the KL divergence off by a constant. Therefore, minimizing CE or KL is equivalent, however the loss value itself is not. In practice, KL will give zero for equal distributions, while CE will still give a constant.  There is also the claim that Mean Square Error is a better metric for Knowledge Distillation (in [Comparing Kullback-Leibler Divergence and Mean Squared Error Loss
in Knowledge Distillation](https://arxiv.org/pdf/2105.08919.pdf) ) but we ignore that for now. 

In the [NLLLoss documentation](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss): 
"The input given through a forward call is expected to contain log-probabilities of each class. [...] Obtaining log-probabilities in a neural network is easily achieved by adding a LogSoftmax layer in the last layer of your network. You may use CrossEntropyLoss instead, if you prefer not to add an extra layer". When using [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.h), the input is expected to contain the unnormalized logits for each class". [KL-divergence](https://pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html) expects an `input` and a `target` argument to be passed as log-probability and o

Finally, here we ignored the KD hypermarameter **Temperature parameter** that scales the teacher and student logits to control the convergeance of the learning. And we use a single teacher model, where sometimes the best results come from using the avergage of an ensemble of models as teacher. This is better detailed in [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531).

More importantly, in here, teacher and student are trained in different runs, but using the soft labels loaded from disk. I prefer this approach as you can train a very large teacher alone, and subsequentely, train smaller students, where the student in one iteration becomes the teacher of the next one.


As an alternative, for a faster distillation run, you could have loaded both the teacher and student in memory, train the teacher, then perform inference while training a student. But this requires all models to be loaded in memory, limiting the maximum size of the models, and only let's you to one distillation session. You may also have a KD loss function that includes in the loss function not only the KL-div of the final output of both models but the MSE of intermediatte layers of the teacher and the student. This is detailed in the [Knowledge distillation tutorial on pytorch](https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html#knowledge-distillation-run).


