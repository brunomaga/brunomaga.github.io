---
layout: post
title:  "Cheaper and faster GPT inference via knowledge distillation, pruning, quantization and Mixture of Experts"
categories: [machine learning, Transformer, GPT, DeepSpeed, pruning, distillation, quantization, mixture-of-experts]
tags: [machinelearning]
---

Previously, in [Distributed training of a large GPT model with DeepSpeed]({{ site.baseurl }}{% post_url 2023-08-18-GPT-lite-DeepSpeed %}), we foccused on training a very large model on a distributed network of GPUs. The aim was to reduce training runtime via increased parallelism, and to increase model accuracy by increasing model size. In this post, we will look at the opposite problem in the ML spectrum: model compression for lower runtime and lower memory footprint during inference. This is particularly relevant for embeddeded and real time systems where time and cost are an issue.

Just like in our [previous post]({{ site.baseurl }}{% post_url 2023-08-18-GPT-lite-DeepSpeed %}), we will analyse two testbenches:
1. the small variant of the ([GPT2 model](https://arxiv.org/abs/2005.14165), that we call the **GPTlite model**, trained on the [tiny shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) dataset, whose objective is to generate text by predicting the next character in a sequence;
2. the **Benchark model**, a Deep Neural Network with user-defined width `W` and number of layers `L`, for multi-label classification, whose objective is to compute the modulo of the sum of squares of a random input vector.

{: style="text-align:center; font-size: small;"}
<img width="20%" height="20%" src="/assets/GPT-lite/gpt_lite_compact.png"/>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
<img width="22%" height="22%" src="/assets/GPT-lite/benchmark_model.png"/>

{: style="text-align:center; font-size: small;"}
The diagram our [GPT2 model]({{ site.baseurl }}{% post_url  2023-02-28-GPT-lite %}) model with N decoder blocks (left), and the benchmark model, a Deep Neural Network with L layers of dimensionality W (right).

## Knowledge Distillation

### Detour: background

Knowledge Distillation (KD) is a technique used to train a (student) model from another (teacher) model. There are many claims of why KD works. But the main rationale is that, if you take as an example the use case of multi-label classification, the soft labels (distribution of assignments) yielded by the trained network is a more accurate representation of the distribution of the classes assigned to the input, when compared with the user-provided hard labels (binary). Thus, training a second network against a better distribution of labels allows the model capacity to be better utilized (yielding better performance) or to achieve similar performance with a smaller model.

As a quick example, take the two-label (dog, cat) classification task. An image of a cat that looks like a dog cat will have the groundtruth label distribution `[0,1]` . After training the model, querying the model for that same image would yield an output simillar to `[0.4, 0.6]`, i.e. the model believes it's a cat, but could also be a dog. In practice, the soft label `[0.4, 0.6]` is a better classification than the hard label `[0,1]` and using this better labels to train another models will allow the capacity of that second model to be used better (i.e. less capacity spent on learning noise).

There are several categories of KD methods. As loss function, we can try to minimize the soft labels between student and teacher models (as the example above), intermediate layers such as logits, feature maps, etc. We can distil information between similar or different student and teacher model architectures, in order to **improve accuracy**. We can perform **offline distillation** where we train the teacher first, and then train the student against the output of the pre-trained teacher, or we can perform **online distillation**, where we train both the train and teacher simultaneously. We can use a single teacher or an ensemble of teachers. And we can use a student that is a scaled down version of the teacher's architecture, effectively performing **prunning via distillation**. If you are looking for details related to different KD methods, see [Distilling the Knowledge in a Neural Network, Google](https://arxiv.org/abs/1503.02531), [Knowledge distillation in deep learning and its applications](https://peerj.com/articles/cs-474/), and [Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525). 

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="/assets/GPT-lite-Compression/kd_methods.jpg"/>

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

We then load our model and dataset. Any model or dataset can be used. Here we'll simply use the `get_dataset()` and `get_model()` methods detailed in our [previous post]({{ site.baseurl }}{% post_url 2023-08-18-GPT-lite-DeepSpeed %}), that return a `torch.utils.data.Dataset` and `torch.nn.Module` for our testbenches:

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

# Prunning via Knowledge Distillation

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
