---
layout: post
title:  "Faster inference on a GPT model via model compression and distillation"
categories: [machine learning, Transformer, GPT, DeepSpeed]
tags: [machinelearning]
---

Previously, in [Distributed training of a large GPT model with DeepSpeed]({{ site.baseurl }}{% post_url 2023-08-18-GPT-lite-DeepSpeed %}), we foccused on training a very large model on a distributed network of GPUs. The aim is to increase training speedup via parallelism, and increase model accuracy by adding model complexity. In this post, we will look at the opposite problem in the ML spectrum: model compression for lower runtime and lower memory footprint during inference. This is particularly relevant for embeddeded and real time systems where time and cost are an issue.

### Model and dataset

Just like in our [previous post]({{ site.baseurl }}{% post_url 2023-08-18-GPT-lite-DeepSpeed %}), we create the methods `get_dataset()` and `get_model()` that return a `torch.utils.data.Dataset` and `torch.nn.Module` for the two testbenches we will play with:
1. the small variant of the ([GPT2 model](https://arxiv.org/abs/2005.14165), that we call the **GPTlite model**, implemented in <a href="/assets/GPT-lite-DeepSpeed/gptlite.py">`gptlite.py`</a>, trained on the [tiny shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) dataset, whose objective is to generate text by predicting the next character in a sequence;
2. the **Benchark model**, implemented in <a href="/assets/GPT-lite-DeepSpeed/benchmark.py">`benchmark.py`</a>, a Deep Neural Network with user-defined width `W` and number of layers `L`, with input of size `W`, and a categorical output of `W` classes, whose objective is to compute the modulo of the sum of squares of a random input vector.

{: style="text-align:center; font-size: small;"}
<img width="20%" height="20%" src="/assets/GPT-lite/gpt_lite_compact.png"/>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
<img width="22%" height="22%" src="/assets/GPT-lite-cpp/benchmark_model.png"/>

{: style="text-align:center; font-size: small;"}
The diagram our [GPT2 model]({{ site.baseurl }}{% post_url  2023-02-28-GPT-lite %}) model with N decoder blocks (left), and the benchmark model, a Deep Neural Network with L layers of dimensionality W (right).

### Knowledge Distillation

#### Detour: background

Knowledge Distillation (KD) is a technique used to train a smaller model (student) from a larger one (teacher). There are many claims of why we should perform KD instead of training a small model alone. The main rationale is that, in the case of multi-label classification, the soft labels (distribution of assignments) yielded by the trained large network is a more accurate representation of the distribution of the classes assigned to the input, when compared with the user-provided hard labels (binary). Thus, training a second network against a better distribution of labels allows the model capacity to be better utilized (yielding better performance) or to achieve similar performance with a smaller model.

As a quick example, take the two-label (dog, cat) classification task. An image of a cat that looks like a dog cat will have the groundtruth label distribution `[0,1]` . After training the model, querying the model for that same image would yield an output simillar to `[0.4, 0.6]`, i.e. the model believes it's a cat, but could also be a dog. In practice, the soft label `[0.4, 0.6]` is a better classification than the hard label `[0,1]` and using this better labels to train a smaller models will allow the capacity of the smaller model to be used better (i.e. less capacity spent on learning noise).

There are several categories of KD methods. We can try to minimize the soft labels between student and teacher models (as the example above), intermediate layers such as logits, feature maps, etc. We can perform **offline distillation** where we train the teacher first, and then train the student against the output of the pre-trained teacher, or we can perform **online distillation**, where we train both the train and teacher simultaneously. We can use a single teacher or an ensemble of teachers. And we can use a student that is a scaled down version of the teacher's architecture, or a completely different architectures. If you are looking for details related to different KD methods, see [Distilling the Knowledge in a Neural Network, Google](https://arxiv.org/abs/1503.02531), [Knowledge distillation in deep learning and its applications](https://peerj.com/articles/cs-474/), and [Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525). 

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="/assets/GPT-lite-Compression/kd_methods.jpg"/>

{: style="text-align:center; font-size: small;"}
 An illustration of the different knowledge distillation categories of methods and the different branches within each category. In this section, we will implement offline distillation using soft labels, underlined in red in the picture. Adapted from [Knowledge distillation in deep learning and its applications](https://peerj.com/articles/cs-474/).

#### Implementing offline distillation using soft labels

Here, we will implement offline distillation on a student model using only the soft labels of a pre-trained teacher model. This is the simplest and most common implementation of distillation. This is particularly relevant nowadays where large pre-trained models are readily available online, and we only have access to the model output during inference (no intermediatte layer outputs), so we can train a smaller version of a large teacher model that would be infeasible to train alone. 

We will start by training the teacher model, dump its soft labels to disk, and then load a student model and training against those labels. This allows for KD to be implemented with a minimal code change, and perform several iterations of KD, where the student of one iteration can become the teacher of the next one. As an alternative, one could perform KD by loading both the teacher and student in memory. This alternative is faster to run, but requires both models to be loaded in memory, limiting the maximum size of the models, and only let's you do one distillation session. Also the code is not so *clean*, as you can see in the example in the [Knowledge distillation tutorial on pytorch](https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html#knowledge-distillation-run).

In our implementation, the KD training loops require almost no changes compared to a regular training loop in PyTorch. The only change is the flat `teacher_model` that defines wether we perform the training of the teacher (training against hard labels) or the student (training against soft labels):  

```python
def training(model, dataloader, epochs, teacher_model=False):
  # reminder: CrossEntropyLoss(x) = NLLLoss(LogSoftmax(x))
  # CrossEntropyLog expects unnormalized logits; NLLLoss expects log probabilities
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
        student_log_probs = F.log_softmax(logits, dim=-1)
        teacher_logits = torch.load(label_filename(b)).to(device)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        loss = F.kl_div(student_log_probs, teacher_log_probs, log_target=True)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
    # print(f"{epoch}:: loss {running_loss / (epoch+1)}")
  print(f"Train loss: {running_loss / (epoch+1)}. Runtime: {float(time.time() - start_time)} seconds")
```

Note that KL-divergenge as the ~~metric~~ value to minimize when doing student training, instead of Cross Entropy (CE). In practice, Cross entropy loss is the same as the KL divergence off by the target distribution entropy (a constant). Mathematically speaking:

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

Therefore, minimizing CE is equivalent to minimizing KL, however the loss value itself is not, as it will include the value of entropy. In practice the value of CE of equivalent distributions will change a constant (the target distribution entropy) at every mini-batch, while the value of KL will be zero when distributions match. Thus, Cross entropy is typically used on fixed-target distributions (binary labels), while KL divergence is more suitable for applications involving the aproximation of two probability distributions. There is also the claim that Mean Square Error is a better metric for Knowledge Distillation (in [Comparing Kullback-Leibler Divergence and Mean Squared Error Loss in Knowledge Distillation](https://arxiv.org/pdf/2105.08919.pdf) ) but we ignore that for now. 

In the [NLLLoss documentation](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss): 
"The input given through a forward call is expected to contain log-probabilities of each class. [...] Obtaining log-probabilities in a neural network is easily achieved by adding a LogSoftmax layer in the last layer of your network. You may use CrossEntropyLoss instead, if you prefer not to add an extra layer". When using [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.h), the input is expected to contain the unnormalized logits for each class". [KL-divergence](https://pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html) expects an `input` and a `target` argument to be passed as log-probability and o

Finally, here we ignored the KD hypermarameter **Temperature parameter** that scales the teacher and student logits to control the convergeance of the learning. And we use a single teacher model, where sometimes the best results come from using the avergage of an ensemble of models as teacher. This is better detailed in [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531).

docs:  It is recommended to pass certain distributions (like softmax) in the log space to avoid numerical issues caused by explicit log.

The inference loop is pretty straightforward, with the subtle change that requires teacher model to output soft labels:

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
  print(f"Inference accuracy: {running_acc/(b+1)*100}%. Runtime: {float(time.time() - start_time)} seconds")
```

The main distillation loop needs to be executed twice: once to train the teacher, one to train the student. When the teacher runs, the `output` folder will be created with the soft labels, and that is the indicator for the second run to know that it must now train the student model. Also, just like in our [previous post]({{ site.baseurl }}{% post_url 2023-08-18-GPT-lite-DeepSpeed %}), we will define the methods `get_dataset()` and `get_model()` that return a `torch.utils.data.Dataset` and `torch.nn.Module` for the two models we will use as testbench:

```python
import gptlite

def main(scale_factor=1.0, train_epochs=30):

  #if folder does not exist: we are training our first teacher
  teacher_model = not os.path.exists(output_folder)
  os.makedirs(output_folder, exist_ok=True)

  batch_size=1
  gptlite.n_layer    = int(gptlite.n_layer*scale_factor)
  gptlite.n_embd     = int(gptlite.n_embd*scale_factor)
  gptlite.n_head     = int(gptlite.n_head*scale_factor)
  gptlite.block_size = int(gptlite.block_size*scale_factor)
  train_dataset, valid_dataset, vocab_size = gptlite.get_dataset(filename=tinyshakespeare_path)
  model = gptlite.get_model(vocab_size).to(device)
```

We then craete the `DataLoader`, where there is a small *nuance* to keep in mind. Because it has to go over the exact same batches of data (in the same order) between a teacher and a student runs, we have to make it deterministic by defining a `seed_init_fn` that resets all seeds every time the data loader is started (in the `enumerate` loop).

```python
  dataloader_kwargs = {'batch_size': batch_size, 'shuffle': True, 'worker_init_fn': seed_init_fn }
  train_dataloader = DataLoader(train_dataset, **dataloader_kwargs)
  valid_dataloader = DataLoader(valid_dataset, **dataloader_kwargs)
```

where
```python
def seed_init_fn(seed=42):
  np.random.seed(seed)
  random.seed(seed)
  torch.manual_seed(seed)
```

The rest of the main loop is simple. We train the teacher on the train dataset, then we test its accuracy on the validation dataset, then iterate again the train dataset on inference mode to output its soft labels in order to train the student: 
```python
  training (model, train_dataloader, epochs=train_epochs, teacher_model=teacher_model) #train teacher model
  inference(model, valid_dataloader, output_labels=False) # test accuracy of teacher model
  inference(model, train_dataloader, output_labels=True) # output soft labels for next student
```

We can then compare the performance of the real size model, to a model that is distilled iteratively from half of the original size, to a model that is trained directly with hald the size:

```python
if __name__ == "__main__":
  import shutil

  # iteratively distil a model to smaller sizes until we reach 1/2 the original size
  if os.path.exists(output_folder): shutil.rmtree(output_folder)
  for scale_factor in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
    main(scale_factor=scale_factor)

  # test a model that is half the size 
  if os.path.exists(output_folder): shutil.rmtree(output_folder)
  main(scale_factor=0.5)
```
