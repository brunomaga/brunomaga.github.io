---
layout: post
title:  "GPT model inference optimization via kernel fusion, distillation, pruning, quantization, flash attention and KV cache"
categories: [machine learning, Transformer, GPT, pruning, distillation, quantization, pruning]
tags: [machinelearning]
---

Previously, in [Distributed training of a large GPT model with DeepSpeed]({{ site.baseurl }}{% post_url 2023-08-18-GPTlite-data-parallelism %}), we foccused on training a very large model on a distributed network of GPUs. The aim was to reduce training runtime via increased parallelism, and to increase model accuracy by increasing model size. In this post, we will look at the opposite problem in the ML spectrum: model compression for lower runtime and lower memory footprint during inference. This is particularly relevant for embeddeded and real time systems where time and cost are an issue.

Note: Easiest way to speed up the model is to save memory, allowing for faster batches, therefore higher throughput (as shown in ZeRO and ZeRO-infinity). There are several efforts

- [TensorRT-LLM](https://developer.nvidia.com/blog/achieving-top-inference-performance-with-the-nvidia-h100-tensor-core-gpu-and-nvidia-tensorrt-llm/)
- [ZeRO-Inference](https://www.deepspeed.ai/2022/09/09/zero-inference.html) and [inference tutorial](https://www.deepspeed.ai/tutorials/inference-tutorial/)
- [DeepSpeed FastGen](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen/2024-01-19)
- from before: torch.compile and ZeRO++
- LoRA/QLoRA for fine-tuning
- Flash Attention for diagonal attention matrices (?)
- [Continuous batching](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- KV caching and multi-query attention for decoding only
- DeepSpeed zero presentation, Ulysses, etc
- [DeepSpeed model compression](https://www.deepspeed.ai/tutorials/model-compression/)
- [Deep Model Fusion: A Survey](https://arxiv.org/abs/2309.15698)
- [Lilian Weng blog: inference optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)
 
Just like in our [previous post]({{ site.baseurl }}{% post_url 2023-08-18-GPTlite-data-parallelism %}), we will focus on the small variant of the ([GPT2 model](https://arxiv.org/abs/2005.14165), that we call the **GPTlite model**, whose objective is to generate text by predicting the next character in a sequence.
We will discuss and apply Knowledge distilation for improved accuracy and prunning/compression, TensorRT for quantization and acceleration via kernel fusion, `torch.compile` for model acceleration, and flash attention and KV-cache for lower memory and higher acceleration.

{: style="text-align:center; font-size: small;"}
<img width="20%" height="20%" src="/assets/GPTlite/gpt_lite_compact.png"/>


## Knowledge Distillation

### Detour: background

Knowledge Distillation (KD) is a technique used to train a (student) model from another (teacher) model. The goal is to co-train two models and pass label/embedding information from a larger or pre-trained *teacher* model to a smaller or untrained *student* model, to make the student smaller and/or batter than that teacher. There are many claims of why KD works. But the main rationale is that, if you take as an example the use case of multi-label classification, the soft labels (distribution of assignments) yielded by the trained network is a more accurate representation of the distribution of the classes assigned to the input, when compared with the user-provided hard labels (binary). Thus, training a second network against a better distribution of labels allows the model capacity to be better utilized (yielding better performance) or to achieve similar performance with a smaller model.

As a quick example, take the two-label (dog, cat) classification task. An image of a cat that looks like a dog cat will have the groundtruth label distribution `[0,1]` . After training the model, querying the model for that same image would yield an output simillar to `[0.4, 0.6]`, i.e. the model believes it's a cat, but could also be a dog. In practice, the soft label `[0.4, 0.6]` is a better classification than the hard label `[0,1]` and using this better labels to train another models will allow the capacity of that second model to be used better (i.e. less capacity spent on learning noise).

There are several categories of KD methods. As loss function, we can try to minimize the soft labels between student and teacher models (as the example above), intermediate layers such as logits, feature maps, etc. We can distil information between similar or different student and teacher model architectures, in order to **improve accuracy**. We can perform **offline distillation** where we train the teacher first, and then train the student against the output of the pre-trained teacher, or we can perform **online distillation**, where we train both the train and teacher simultaneously. We can use a single teacher or an ensemble of teachers.

{: style="text-align:center; font-size: small;"}
<img width="70%" height="70%" src="/assets/GPTlite-Compression/model_distillation_offline_online.png"/>

We can use a student that is a scaled down version of the teacher's architecture, effectively performing **prunning via distillation**. If you are looking for details related to different KD methods, see [Distilling the Knowledge in a Neural Network, Google](https://arxiv.org/abs/1503.02531), [Knowledge distillation in deep learning and its applications](https://peerj.com/articles/cs-474/), and [Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525).  

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="/assets/GPTlite-Compression/kd_methods.jpg"/>

{: style="text-align:center; font-size: small;"}
 An illustration of the different knowledge distillation categories of methods and the different branches within each category. In this section, we will implement offline distillation using soft labels, underlined in red in the picture. Adapted from [Knowledge distillation in deep learning and its applications](https://peerj.com/articles/cs-474/).

### Implementing offline distillation using soft labels

Here, we will implement offline distillation on a student model using only the soft labels of a pre-trained teacher model. This is the simplest and most common implementation of distillation. This is particularly relevant nowadays where large pre-trained models are readily available online, and we only have access to the model output during inference (no intermediatte layer outputs), so we can train a smaller version of a large teacher model that would be infeasible to train alone. 

We will start by training the teacher model, dump its soft labels to disk, and then load a student model and training against those labels. This allows for KD to be implemented with a minimal code change, and perform several iterations of KD, where the student of one iteration can become the teacher of the next one. As an alternative, one could perform KD by loading both the teacher and student in memory. This alternative is faster to run, but requires both models to be loaded in memory, limiting the maximum size of the models, and only let's you do one distillation session. Also the code is not so *clean*, as you can see in the example in the [Knowledge distillation tutorial on pytorch](https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html#knowledge-distillation-run).

In our implementation, the KD training loops require almost no changes compared to a regular training loop in PyTorch. The only change is the flat `teacher_model` that defines wether we perform the training of the teacher (training against hard labels) or the student (training against soft labels):  

```python
output_folder = "output"
label_filename = lambda batch: os.path.join(output_folder,f"logits_{batch}.pt")
```

```python
def training(model, dataloader, epochs, teacher_model=False, temperature=2):
  model.train()
  optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
  start_time = time.time()
  for epoch in range(epochs):
    running_loss = 0.0
    for b, (x, label) in enumerate(dataloader):
      x, label = x.to(device), label.to(device)
      optimizer.zero_grad() 
      logits = model(x)
      if teacher_model:
        loss = F.cross_entropy(logits, label)
      else:
        student_log_probs = F.log_softmax(logits / temperature, dim=-1)
        teacher_logits = torch.load(label_filename(b)).to(device)
        teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
        loss = F.kl_div(student_log_probs, teacher_log_probs, log_target=True)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
    # print(f"{epoch}:: loss {running_loss / (epoch+1)}")
  print(f"Train loss: {running_loss / (epoch+1)}.")
  print(f"Train runtime: {float(time.time() - start_time)} seconds")
```

Let's dissect our loss functions. In the teacher model, we try to maximize a log-likelihood of a distribution, so [NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss) seems like the right options. It expects the input to be the log-probabilities of each class, ie the [LogSoftmax](https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html) of the output of our network. However, we use [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.h), where input is expected to contain the unnormalized logits for each class, so we area avoiding one extra layer in our model. In practice:

$$
CrossEntropyLoss(x) = NLLLoss( LogSoftmax(x) )
$$

Nevertheless, we choose CrossEntropy as our loss function because - despite mathematically equivalent - it is numerically more stable (avoids some $$\log$$ and $$\exp$$ operations). 

Now we look at the loss of the student model. Note that [KL-divergence](https://pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html) is the ~~metric~~ value that we are minimizing when doing student training, instead of Cross Entropy (CE). In practice, Cross entropy loss is the same as the KL divergence off by a constant (the target distribution entropy). Mathematically speaking:

$$
\begin{equation}
\begin{split}
D_{kl}(p \mid q) & = H(p,q) - \, H(p) \\
 & = - \sum_i p_i \log (q_i) - \left( -\sum_i p_i \log (p_i) \right) \\
 & = - \sum_i p_i \log (q_i) + \sum_i p_i \log (p_i) \\
 & = \sum_i p_i \log \frac{p_i}{q_i} 
\end{split}
\end{equation}
$$

Therefore, minimizing CE is equivalent to minimizing KL. However the loss value itself is not, as the KL value of equivalent distributions will be zero and the CE will be the value of entropy of the target distribution, at every mini-batch. Thus, Cross entropy is typically used on fixed-target distributions (hard labels) as entropy is zero anyways, while KL divergence is more suitable for applications involving the aproximation of two probability distributions. In PyTorch, KL divergence loss expects an input to be a log-probability and a target that is by default passed as a probability. We also passed the target as probability as, but according to the documentation, "it is recommended to pass certain distributions (like softmax) in the log space to avoid numerical issues caused by explicit log".

Note that we did not tune the **Temperature hyper-parameter** that controls the softness of the softmax distributions in order to utilize rations of smaller probabilities, and can make the student learn better. In practice, for a temperature $$t$$, the ouput of a layer is a softmax in the form:

$$
y_i (x \mid t) = \frac{ \exp\frac{z_i(x)}{t} }{ \sum_j \, \exp\frac{z_j(x)}{t} }
$$


Finally, There is also the claim that Mean Square Error is a better metric for Knowledge Distillation, in [Comparing Kullback-Leibler Divergence and Mean Squared Error Loss in Knowledge Distillation](https://arxiv.org/pdf/2105.08919.pdf), but we ignore that for now.


Now, the inference loop is pretty straightforward. The only subtle change is to make the teacher model output soft labels when needed, with `output_labels=Trrue`:

```python
def inference(model, dataloader, output_labels=False):
  model.eval()
  start_time = time.time()
  running_acc=0
  with torch.no_grad():
    for b, (x, label) in enumerate(dataloader):
      x, label = x.to(device), label.to(device)
      output = model(x)
      running_acc += (x.argmax(-1)==label).sum()/len(x) 
      if output_labels:
        torch.save(output, label_filename(b))
  print(f"Inference accuracy: {running_acc/(b+1)*100}%.")
  print(f"Inference runtime: {float(time.time() - start_time)} seconds")
```

Our main loop starts by checking if the output folder exists. If it exists, it contains the soft labels used for training (and we are training a teacher). Otherwise, we are training a teacher against the user-provided hard labels:
The main distillation loop needs to be executed twice: once to train the teacher, once to train the student. When the teacher runs, the output folder will be created with the soft labels. On the second run, the student will load the soft labels output by the teacher and train the model against them. 


```python
def main(train_epochs=30):
  #if folder does not exist: we are training our first teacher
  teacher_model = not os.path.exists(output_folder)
  os.makedirs(output_folder, exist_ok=True)
```

We then load our model and dataset. Any model or dataset can be used. Here we'll simply use the `get_dataset()` and `get_model()` methods detailed in our [previous post]({{ site.baseurl }}{% post_url 2023-08-18-GPTlite-data-parallelism %}), that return a `torch.utils.data.Dataset` and `torch.nn.Module` for our testbenches:

```python
  import gptlite
  branch_size=1
  train_dataset, valid_dataset, vocab_size = gptlite.get_dataset()
  model = gptlite.get_model(vocab_size).to(device)
```

We then create the `DataLoader`, and here there is a small *nuance* to keep in mind. Because it has to go over the exact same batches of data (in the same order) between a teacher and a student runs, we have to make it deterministic by defining a `seed_init_fn` that resets all seeds every time the data loader is started (in the `enumerate` loop).

```python
  def seed_init_fn(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
```

```python
  dataloader_kwargs = {'batch_size': batch_size, 'shuffle': True, 'worker_init_fn': seed_init_fn }
  train_dataloader = DataLoader(train_dataset, **dataloader_kwargs)
  valid_dataloader = DataLoader(valid_dataset, **dataloader_kwargs)
```

The rest of the main loop is simple. We train the teacher on the train dataset, then we test its accuracy on the validation dataset, then iterate again the train dataset on inference mode to output its soft labels in order to train the student: 

```python
  training (model, train_dataloader, epochs=train_epochs, teacher_model=teacher_model) #train teacher
  inference(model, valid_dataloader, output_labels=False) # test accuracy of teacher
  inference(model, train_dataloader, output_labels=True)  # output soft labels for next student
```

All done. To perform distillation from a student to a teacher, we run:

```python
import gptlite
if __name__ == "__main__":
  main() # train teacher and output labels
  main() # train student against labels output by teacher
```

### Layer and parameter prunning via knowledge distillation

We will now perfor **prunning** of layers, activations or other architectural parameters we may need. We will introduce a varialbe `scale_factor` that reduces the architecture size by a given factor. We simply change the distillation loop to scale the model upon initialization:


```python
def main(scale_factor=1.0, train_epochs=30):
  # ...

  import gptlite
  batch_size=1
  gptlite.n_layer    = int(gptlite.n_layer*scale_factor)
  gptlite.n_embd     = int(gptlite.n_embd*scale_factor)
  gptlite.n_head     = int(gptlite.n_head*scale_factor)
  gptlite.block_size = int(gptlite.block_size*scale_factor)
  train_dataset, valid_dataset, vocab_size = gptlite.get_dataset(filename=tinyshakespeare_path)
  model = gptlite.get_model(vocab_size).to(device)
```

and try to prune the model to half of its original size by iteratively reduce its size by 10% and distil knowledge to a new student recursively:

```python
if __name__ == "__main__":
  for scale_factor in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
    main(scale_factor=scale_factor)
```

We can then compare the performance of the real size model, the model that was distilled iteratively to half of it original size, and the model that was trained directly with half the size.

**COMING SOON**

## ONNX: Open Neural Network eXchange

[ONNX (Open Neural Network eXchange)](https://onnx.ai/) is an open format built to represent machine learning models, that is [independent of the framework](https://onnx.ai/onnx/intro/converters.html) used to create the model (PyTorch, Tensorflow, Scikit, etc). The `onnx` data format is a model description that can be run through the [onnx runtime](https://onnxruntime.ai/), or processed by other tools for further optimisation.

To start with ONNX, install the pip packages `onnx`, the `onnx runtime`, and `onnxscript`.
To visualise the intermediate graphs, we can will run the code in [onnx_to_png.py](onnx_to_png.py) to generate a graphviz [`.dot` format](https://graphviz.org/doc/info/lang.html) and then convert it a `png` image, or simply use the [Netron app](https://netron.app/) to load a `onnx` file and then to vizualise/export the model. 
For the sake of simplicity, we will only load a two-head attention block of the `GPTlite` model (a GPT2 variant) that we focused on in the previous posts.

```python
import os
import sys
import torch

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', 'GPTlite'))
from gptlite import GPTlite, Block, block_size, n_embd, n_head
n_head=2

B, T, C = 2, block_size, n_embd
embd = torch.ones((B, T, C)).to(device) # embedding as block input
model = Block(n_embd, n_head).to(device)
```

We then export our `torch.nn.Module` to a `onnx` file that contains the graph we want to look at, using two distinct methods, following the onnx [tutorial](https://pytorch.org/docs/stable/onnx.html#overview). An **exporter based on Torch.script**, that creates the ONNX graph using TorchScript's [`trace` or `script` modes](https://glaringlee.github.io/onnx.html#tracing-vs-scripting). The graphic captured with `trace` is static (relates only to the input passed), therefore it does not capture control-flow statements like `ifs` or loops or dynamic type inputs. In `script` mode, because "TorchScript itself is a subset of the Python language, not all features in Python are supported, such as in-place operations."
   ```python
  def export_torch_jit_onnx(model, mode='train'
      torch.onnx.export(
        model.eval() if mode=='eval' else model.train(), # model being run
        embd, "model.onnx",
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})

  export_torch_jit_onnx(model, "model")
  export_torch_jit_onnx(torch.jit.trace(model, embd), "model_jit_trace") #same as above
  export_torch_jit_onnx(torch.jit.script(model), "model_jit_script")
   ```

The graph for the train model using jit trace can be seen below (for brevity, only layer names are displayed):

  <img width="100%" height="100%" src="/assets/GPTlite-Compression/model_jit_trace_train.onnx.png"/>

Secondly, an **exporter based on [TorchDynamo](https://pytorch.org/docs/stable/torch.compiler_deepdive.html)** the newest/beta exporter for `torch>=2.0`, that uses a feature called [Python frame evaluation](https://peps.python.org/pep-0523/) to convert a bytecode analysis into the [FX graph](https://pytorch.org/docs/stable/fx.html), that is then polished and translated into the ONNX graph, and contrarily to the previous, captures dynamic graphs.
Torch.FX includes a symbolic tracer that works the same as the `trace` mode above, but by feeding fake values to the trace operation (or optionally allowing user defined values, making it equivalent to the `torch.jit.trace`).
In this [discussion](https://dev-discuss.pytorch.org/t/the-nuances-of-pytorch-graph-capture/501/9) detailing the differences between different export methods, it seems like the *de facto* method for graph capturing in the future will be this. To generate the onnx file we run:
   ```python
  def export_dynamo_onnx(model, name, mode='train'):
    export_output = torch.onnx.dynamo_export(
       model.eval() if mode=='eval' else model.train(), embd)
    export_output.save(f"{name}.onnx")   

  export_dynamo_onnx(model, "model_dynamo")
  export_dynamo_onnx(torch.compile(model), "model_compile_dynamo") #same as above
   ```

   And the output for the train graph is:

   <img width="100%" height="100%" src="/assets/GPTlite-Compression/model_dynamo_train.onnx.png"/>


Few points worth mentioning when we look at the graphs. The graphs refer to the graph in train mode. If we set the model to evaluation mode (`model.eval()`) the graphs will differ, because evaluation disables dropout, batchnorm layers will use their running stats to normalize the data, etc. So the train graphs are usually bigger. Also, trace graphs are smaller than script graphs because the graph is generated for a single input size and type, therefore operations such as castings are removed. As a side note, the graphs generated with dynamo look smaller, but in practice, it's displaying only the top-level modules, and the sub-graph inside each module can be visualized int the Neutron app, [clicking the "f" symbol on the top-right corner](https://pytorch.org/docs/stable/onnx_dynamo.html#inspecting-the-onnx-model-using-gui) of each module to navigate the corresponding sub-graph.
Finally, if you are curious to see all train/eval graphs generated, check the [repo for this post](https://github.com/brunomaga/brunomaga.github.io/tree/master/assets/GPTlite-Compression).

## Flash Attention

Speeding up transformer training/inference can be achieved by two memory optimization technique which does not require modification of the mode: flash attention and continuous batching. about continuous batching, see [here](https://www.anyscale.com/blog/continuous-batching-llm-inference)


## Final remarks: model compression for reduced memory footprint and runtime

Many use cases will require model size to be small for deployment (particularly onto embedded systems), or require inter-layer communication to be small due to storage or network bandwidth limitations, or even benefit from a smaller numerical representation to increase vectorization. To handle that, some commonly used techniques are:  
- [Pruning methods](https://arxiv.org/abs/2101.09671), where weights or neurons are *dropped* after training or during training (via a train-prune-train-prune-etc workflow). Note that prunning of weights alone will reduce memory footprint but not compute time in GPUs due to the registers being filled with the same neurons as pre-prunning;
- [Quantization methods](https://arxiv.org/abs/2103.13630) to reduce the numerical representation, value ranges and bit counts of values. The common use case is to use reduce of mixed floating point representation of values, reducing memory footprint and runtime (by increasing vectorization). Few relevant topics:
  - the paper [Mixed Precision Training](https://arxiv.org/abs/1710.03740) discusses which data types (parameters, gradients, accumulators) required which precision and provides good guidances on mixed precision training.
  - a recent floating point representation called [bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) (*brain floating point*), a different 16-bit representation, is important for ML as it represents a wider dynamic range of numeric values than the regular 16-bit representation. This is due to having more bits reserved to the exponent (8 bits, just like 32-bit f.p.) compared to the traditional 16-bit representation (5 bit). In practice, it deliveres the range of a 32-bit representation with the memory consumption and runtime of a 16-bit representation, due to a tradeoff of range vs precision. 
  - <img class="mt-3" width="80%" height="80%" src="/assets/AI-Supercomputing/floating_point_representation.png"/>
  - reduced floating point representations (e.g. 16-bit) are commonly combined with [(dynamic) loss scaling](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html), a technique that scales up the values of the gradients so that very small gradient values are not represented as zero in the fraction bits of the f.p. representation.  
- [Knowledge distillation](https://research.google/pubs/pub44873/), a special case of model compression, that transfer learning from a larger to a smaller model. This allows the smaller model to be smaller and lighter, and *some times* of increased performance;
- [Activation/gradietn checkpointing](https://arxiv.org/abs/2012.00825) to avoid storing all activations of intermediate layers --- required for back-propagation --- *in lieu* of on-the-fly computation of activations from a previous checkpoint (layer). The ammount of checkpointed layers guides the tradeoff between runtime increase and memory decrease; 
- [Neural architecture search (NAS)](https://en.wikipedia.org/wiki/Neural_architecture_search), a method to search for the parameters that define the architecture of the models (e.g. number of layers, layer sizes, etc). I have no applied exposure to this method, but found [this paper](https://arxiv.org/abs/2301.08727) to be very insightful in surveying existing methods;
