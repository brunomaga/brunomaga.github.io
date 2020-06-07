---
layout: post
title:  "AI Supercomputing: from Linear Regression to BERT"
categories: [machine learning, supercomputing]
tags: [machinelearning]
---


[coordinate-ascent]: {{ site.baseurl }}{% post_url 2018-02-17-Supervised-Learning %}

tion can be performed using the [coordinate ascent][coordinate-ascent] method. 


Machine Learning is driven by mathematical models that try to *learn* from data. The complexity of the data is continuously increasing, due to higher-resolution photos, larger textual databases, higher number of observable features on input datapoints, etc. The computing power available *tends* to follow, or somehow adapt, as observed by [Moore's law](https://en.wikipedia.org/wiki/Moore%27s_law). However, in the occurence of very large datasets and very computationally-expensive learning models, the learning process is limited due to insufficient memory or an infeasibly-high training time. This is where **AI Supercomputing --- or technically speaking Parallel/Distributed computing of Machine Learning algorithms** ---  comes into place. 

AI Supercomputing focuses on how to distribute data (models, inputs) and computation across several compute units (vector units, CPU cores, GPU cores, and machines). The goal is to distribute the data in such way that the learning algorithm can be execution simultaneously (*parallelized*) with different datasets across all processors, speeding up the solution and reducing memory usage per machine. On the other hand, distributing data that are computationally dependent --- or equally, that need to be in the same memory region (machine) to be computed together at some point --- introduces another layer of analysis of efficiency due to the overhead on the communication required to move variables across machines or compute units. So the ideal algorithm can be characterized as the one that allows for:

1. homogeneous distribution of data across memory units, i.e. balanced memory usage;
2. homogeneous amount of computation assigned to each compute unit, i.e. balanced computation; and
3. a minimal amount of communication across memory/compute units, or ideally a zero communication overhead if overlaping of communication and computation is possible.

When these three properties are achieved, then we guarantee the **linear scaling** of the algorithm. This means that, by increasing (e.g. doubling) the compute resources, we decrease (halve) the computation. In practice, perfectly-linear scaling is very hard, but quasi-linear scaling is commonly achieved on cleverly designed algorithms. Let's start with the basics.


# Linear Regression and Mutual Exclusion

That a basic linear example of input variables $x$, labels $y$, learnt weights $w$ and a loss functions set at the Mean Absolute Error.
We want to minimize:

$$
\begin{align*}
MAE(w)  = \frac{1}{N} \sum_{n=1}^N | y_n - f(x_n) | \text{, where } f(x_n) = \sum_{j=1}^{M} w_j x_j
\end{align*}
$$


To speed-up the solution, we can *parallelize* both sums in $MAE$ and $f(x_n)$ with several compute units (let's say $T$ compute threads) and decompose the computation as:

$$
\begin{align*}
MAE(w) & = MAE (w_{thread_1}) + MAE (w_{thread_2}) + ... + MAE (w_{thread_T})\\
& = \sum_{n=1}^{\lfloor N/T \rfloor} |y_n - f(x_n)| +  \sum_{n=\lfloor N/T \rfloor +1}^{2\lfloor N/T \rfloor} |y_n - f(x_n)| + ... + \sum_{n=(T-1)\lfloor N/T \rfloor +1}^{N} |y_n - f(x_n)|
\end{align*}
$$

This operation is \textit{memory-safe} for $f(x_n)$, but unsafe for $MAE(w)$, or equivalently, all write operations (variable assignments) in $f(x_n)$ write on different positions while in $MAE(w)$ they don't. In practice, computing $f(x_n)$ requires a sum of $M$ independent products written on each index of $x$, while the final value holding $MAE(w)$ is a constant that needs to be updated with the valuer on every term $\|y_n - f(x_n)\|$. How will that value be updated when several threads try to write simultaneously to the memory space holding it? Let's study four options:

1. base case: no parallelism, i.e. use only a single thread. The output is correct but its computation is **slow**. This implementation is available in <a href="/assets/AI-Supercomputing/AI_SC_1.cpp">AI\_SC\_1.cpp</a>;
2. each thread updates the MAE sum continuously. This implementation provides an almost linear scaling of computation, however the output is **wrong** as several memory corruptions occured when several threads try to update the value continuously (<a href="/assets/AI-Supercomputing/AI_SC_2.cpp">AI\_SC\_2.cpp</a>);
3. same as before, however we add a *mutex* (mutual exclusion) control object. A mutex allows a region of the code to be *locked* by a thread, in such way that no other thread  can enter it until the first thread has unlocked it. We can use it to protect the operation responsiblefor the memory update of the variable being accessed *concurrently* for every update operation. This operation gives an accurate result, however it is **Very slow!**, due to the computational overhead introduced by the mutex itself (<a href="/assets/AI-Supercomputing/AI_SC_3.cpp">AI\_SC\_3.cpp</a>);
4. finally, the same as before, but we compute the sums of each thread independently and perform a single concurrent memory update of the final variable once, and only at the ned of the execution. This approach gives the correct result with almost **linear scaling** of the algorithm (<a href="/assets/AI-Supercomputing/AI_SC_4.cpp">AI\_SC\_4.cpp</a>);

These examples are relevant as they show several points: concurrency control is computational expensive; parallelization is worth it; and matrix-vector multiplications and updates (that underlie most operations in Machine Learning) can be performed efficiently when individual contributions from compute units are decomposed, computed independently and *reduced* at the end.

However, not all operations can be easily parallelizable and reproducible. An example is any method based on random number generation, such as [Markov chain Monte Carlo (MCMC)](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) methods. In such scenarios, where comput units *draw* continuosly random number(s) for every datapoint, we need to think of the efficiency problem as a trade-off between reproduciblity and efficency. We may allow a fully-parallel algorithm (by having a random number generator per compute unit, therefore yielding efficient computation with results that can only be reproduced on executions with the same number of processors). Or we can have a single random generator with mutually-exclusive access and the sequence of randomly-generated numbers consistent across executions, such as drawing numbers following the input offset. Such details fall out of the scope of this post, so we'll move on with the main topic.

# From CPU to GPU and TPU/IPU

We talk about compute units over and over. But what does it really mean? On the subject of Machine Learning, three main architectures are relevant:
1. CPU (Central Processing Unit): the main processor of any desktop, laptop or mobile computer. Intended to be a high speed processor to perform efficiently most *iterative* tasks that we do on a daily basis, such as text editors, mouse/keyboard interfaces, browser, and most applications that run on any operative systems. As we interact of few of these apps at a time, while others run a minimal service on the background, the CPU is defined by a few cores of high-frequency clockspeed. 
2. GPU (Graphics Processing Unit): initially designed for graphics processing, which are by nature highly parallel (e.g. the computaion of each pixel value on a screen) and based on Matrix/Vector operations (such as rotation, translation and scaling of 3D shapes). As these operations are usually simple algebraic computation performed simultaneously for millions of inputs, the GPU was designed as a high number of compute cores, running at a low-clock speed;
3. IPU (Inteligent Processing Unit) or TPU (Tensor Processing Unit): a natural evolution of the GPU towards Machine Learning applications. Allows for a higher throughput (or total compute power) due to reducing the operation of each processor to Machine Learning purposes, thus decreaing processor size and allowing an increased number of cores. Because the emphasis of ML application is typically the processing of large batch sizes, IPUs are designed to compute at a rate lower than the GPU. 

The main question is: what drives processor designed to lower the clockspeed of their processors, instead of just designing GPU/TPU/IPUs at the same clock speed as CPUs? The answer is **power consumption**, **[heat dissipation](https://en.wikipedia.org/wiki/List_of_CPU_power_dissipation_figures)** and **temperature**. In practice, to increase the clockspeed we usually reduce the transistor size thus inserting [more transistors in the same chip](https://en.wikipedia.org/wiki/Transistor_count) or placing them more tightly in the same area. This leads to an increase of power comsunption and temperature that has been observed to grow exponentially with the increase of the clockspeed: 

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/a53-power-curve.png"/><br/>
<br/><small>Exponential increase of power comsumption (y axis) for a linear increase of processor frequency (x axis),<br/> for one to four cores (colour coded) of the Samsung Exynos 7420 processor. (source: <a href="https://www.anandtech.com/show/9330/exynos-7420-deep-dive/5">AnandTech</a>)</small>
</p>


The take-home message is: in regression problems, since computational reductions happens rarely and are very efficient (as we saw on the Linear Regression example), then the hardware feature that dictates performance is total GHz across all compute cores. Or more importantly, number of **FLOPs** )(Floating Point Operations per second), since a processor instruction can execute simultaneously several operations, using a techique called [SIMD (Single Instruction Multiple Data](https://en.wikipedia.org/wiki/SIMD#:~:text=Single%20instruction%2C%20multiple%20data%20(SIMD,on%20multiple%20data%20points%20simultaneously.) or [MIMD (Multiple Instructions Multiple Data)[https://en.wikipedia.org/wiki/MIMD]. We'll skip the details about SIMD and MIMD functioning as they're not relevant in the context of this post. 

Looking at the previous plot, we see that, to efficiently maximize GHz/FLOPs throughput, one is much more efficient by having several processors of low clock frequency, instead of fewer of a higher frequency. This is, at a very high level, the main different between a CPU and a GPU architecture, and this explains why GPUs tend to be the preferred choice to compute Machine Learning trainign problems. This phylosophy led to the creation of [TPUs (Tenso Processing Units)](https://en.wikipedia.org/wiki/Tensor_processing_unit) and [IPUs (Inteligent Processing Unit)](https://www.graphcore.ai/products/ipu), that explore this trade-off of number of cores vs clock-frequency, with lower-precision floating point representations (to maximize SIMD), and ML-specialized logical units on the processors, to augment further the throughput. Comparing a common CPU, GPU, and IPU used in compute clusters dedicated to ML tasks:


|                    | **cores x clock-frequency**  $\hspace{1cm}$ | **FLOPs (32 bits representation)**  $\hspace{1cm}$ | **Max RAM** |
|---------------------	|-----------------------------	|------------------------------------	|-------------	|
| **Intel Xeon 8180** $\hspace{1cm}$ | 28x 2.5 Ghz 	| 1.36 TFLOPS 		| 768 GB       	|
| **Tesla K80**       	| 4992x 0.56 Ghz             	| 8.73 TFLOPS                         	| 2x 12GB     	|
| **Graphcore IPU**   	| 1216 x 1.6Ghz [1]           	| 31.1 TFLOPS                     	| 304 MiB [2] 	|
|---------------------	|-----------------------------	|------------------------------------	|-------------	|

<br/>
Some important remarks on the IPU architecture: [1] TPUs use Accumulating Matrix Product (AMP) units, allowing 16 single-precision floating point operations per clock cycle, therefore the processor is not directly comparable by looking simply at core count and clock-frequency. Also, [2] small memory is compensated by a very low latency between processor and memory, allowing onloading of offloading of large datasets more efficiently. To learn more about Graphcore's IPU, see the technical report [Dissecting the Graphcore IPU Architecture via Microbenchmarking, Citadel Technical Report, 7 December 2019](https://www.graphcore.ai/products/ipu).

One main observation derives from the previous table. Memory bandwidth increases from CPU to GPU to IPU, however its total capacity is reduced. We will discuss next how this limitation by continuously on-/offloading only the required data to the memory.

# CPU offloading of Deep Neural Nets




GPUs are faster, but... \alert{how to overcome the memory limitations?}

We've seen before on a previous post about [Deep Neural Networks]({{ site.baseurl }}{% post_url 2018-02-27-Deep-Neural-Networks %}), that for a given layer $l$ of the network, the input is represent as:

$$
x^{(l)} = f^{(l)} (x^{(l-1)}) = \phi ((W^{(l)})^T x^{(l-1)})
$$

where $\phi$ is the activation function. The loss is then computed by taking into account the groundtrugh $y$ and the composition of the ouputs of all layers in the neural network, ie:

$$
L = \frac{1}{N} \sum_{n=1}^N | y_n - f^{(L+1)} \circ ... \circ f^{(2)} \circ f^{(1)} (x_n^{(0)}) |
$$

The important concept here is on the **composition**. In practice one only needs the current layer's state and previous layer output to perform the computation at every layer. This concept has been explored by the (vDNN (Rhu et al.))[https://arxiv.org/pdf/1602.08124.pdf] and (vDNN+ (Shiram et al))[https://www.cse.iitb.ac.in/~shriramsb/submissions/GPU_mem_ML.pdf] implementations: 

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/vDNN.png"/><br/>
<br/><small>An overview of the vDNN(+) implementation on convolutional neural network. Red arrays represent the data flow of variables $x$ and $y$ (layers input and output) during forward propagation. Blue arrows represent data flow during bacward progagation. Yellow arrows represent weight variables. Yellow arrow represent is the variables *workspace in cuDNN*, needed in certain convolutional algorithms. Source: <a href="[https://arxiv.org/pdf/1602.08124.pdf">vDNN (Rhu et al.)</a></small>
</p>

The concept is simple: we store the complete model in CPU memory (or hard-drive if required), and move the active layer into GPU memory when it needs to be computed. To reduce the waiting time of pushing and pulling a layer into the GPU, a viable optimization is to copy asynchronously (ie on the background) the next layer to be computed, while computing the current layer's update.

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/vDNN2.png"/><br/>
<br/><small>The forward pass on the vDNN(+) implementation on convolutional neural network. Data not associated with the curent layer (N, yellow arrow) being processed are marked with black cross and can safely be removed from the GPU's memory. Source: <a href="[https://arxiv.org/pdf/1602.08124.pdf">vDNN (Rhu et al.)</a></small>
</p>


$$
\delta_j^{(l)} =  \frac{\partial L_n}{\partial z_j^{(l)}} = \sum \frac{\partial L_n}{\partial z_k^{(l+1)}} \frac{\partial z_k^{(l+1)}}{\partial z_j^{(l)}} = \sum_k \delta_k^{(l+1)} W_{j,k}^{(l+1)} \phi '(z_j^{(l)})
\text{ \hspace{0.3cm}\, where \hspace{0.3cm} } 
z_j^{(l)} =  (W^{(l)})^T x^{(l-1)}
$$

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/vDNN3.png"/><br/>
<br/><small>The back propagation phase on the vDNN(+) implementation on convolutional neural network. Data not associated with the curent layer (2, yellow arrow) being processed are marked with black cross and can safely be removed from the GPU's memory. Source: <a href="https://arxiv.org/pdf/1602.08124.pdf">vDNN (Rhu et al.)</a></small>
</p>


# Pipeline Parallelims Pipeline Parallelism (G-Pipe, PipeDream)

<p align="center">
<br/>
<img width="35%" height="35%" src="/assets/AI-Supercomputing/Pipedream_DNN_pipeline.PNG"/><br/>
<br/><small>A regular execution of a deep/convolutional neural net using serial processing. Source: <a href="https://www.microsoft.com/en-us/research/publication/pipedream-generalized-pipeline-parallelism-for-dnn-training/">PipeDream: Generalized Pipeline Parallelism for DNN Training (Mirosoft, arXiv)</a>
</small>
</p>


[Google, GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism, ArXiv](https://arxiv.org/abs/1811.06965)

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/Pipedream_DNN_pipeline_parallel.PNG"/><br/>
<br/><small>A regular execution of a deep/convolutional neural net using serial processing. Source: <a href="https://www.microsoft.com/en-us/research/publication/pipedream-generalized-pipeline-parallelism-for-dnn-training/">PipeDream: Generalized Pipeline Parallelism for DNN Training (Mirosoft, arXiv)</a>
</small>
</p>


Backward prop. starts after all forward pass. finishes. Can we do better?

[Microsoft, PipeDream: Generalized Pipeline Parallelism for DNN Training, arXiv](https://www.microsoft.com/en-us/research/publication/pipedream-generalized-pipeline-parallelism-for-dnn-training/)

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/Pipedream_DNN_pipeline_parallel_Microsoft.PNG"/><br/>
<br/><small>A regular execution of a deep/convolutional neural net using serial processing. Source: <a href="https://www.microsoft.com/en-us/research/publication/pipedream-generalized-pipeline-parallelism-for-dnn-training/">PipeDream: Generalized Pipeline Parallelism for DNN Training (Mirosoft, arXiv)</a>
</small>
</p>



# Data Parallelism 

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/DNN_data_parallelism.pdf"/><br/>
<br/><small>Data Parallelism.</a>
</small>
</p>

Not memory efficient. Model is duplicated!

[Google Labs, Measuring the Effects of Data Parallelism on Neural Network Training, arXiv](https://arxiv.org/abs/1811.03600)

#Model parallelism

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/DNN_model_parallelism.pdf"/><br/>
<br/><small>Model Parallelism.</a>
</small>
</p>

Not communication efficient!


## Partial Model Parallelism (CNN)


<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/argonne_parallel_2.PNG"/><br/>
<br/><small>Model Parallelism.</a>
</small>
</p>

Parallelism of image across *four processors*;  Red box: center of 3x3 convolution filter; {\color{red}red arrow}: data movement; {\color{violet}violet region:} elements to be communicated at every step so perform filter of elements at the border. \textbf{RIGHT:} communication across \alert{two processors}. {\color{red}Red arrow:} forward phase dependencies; {\color{blue}Blue arrow:} back-propagation dependencies;

\textbf{Equations:}

\small
\vspace{0.5cm}
{\small Equation 1:} \hspace{0.1cm} $ y_{k,f,i,j} = \sum_{c=0}^{C-1} \sum_{a=-O}^{O} \sum_{b=-O}^{O} x_{k,c,i+a,j+b} w_{f,c,a+O,b+O} $

{\small Equation 2:} \hspace{0.1cm}  $ \frac{dL}{dw_{f,c,a,b}} = \sum_{k=0}^{N-1} \sum_{i=0}^{H-1} \sum_{j=0}^{W-1} \frac{dL}{dy_{k, f, i, j}} x_{k, c, i+a-O, j+b-O} $

{\small Equation 3:} \hspace{0.1cm}  $ \frac{dL}{dx_{k,c,i,j}} = \sum_{j=0}^{F-1} \sum_{a=-O}^{O} \sum_{b=-O}^{O} \frac{dL}{dy_{k, f, i-a, j-b}} w_{f, c, a+O, b+O} $

[Dryden et al., Improving Strong-Scaling of CNN Training by Exploiting Finer-Grained Parallelism, Proc. IPDPS 2019](https://arxiv.org/pdf/1903.06681.pdf)

---

So far, we know that:
- ML is very parallel due to efficient Matrix-vector multiplication;
- Memory is limited but we overcome it with \textbf{CPU-offloading};
- We can \textbf{pipeline} parallelism;
- We can parallelize data and use mean batch gradients;
- We can parallelize the model (locally e.g. CNN);

*Any ML model that is not covered?*


# Lmitations of parallelism (Encoder-Decoder, Seq-to-Seq)


<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/Encoder_Decoder.pdf"/><br/>
<br/><small>Model Parallelism.</a>
</small>
</p>

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/encoder_decoder_3.png"/><br/>
<br/><small>Model Parallelism.</a>
</small>
</p>

- encoding/decoding is a recursive algorithm $\rightarrow$ iterations can't the parallelized;
- Single hidden layer with *small* embedding  $\rightarrow$ no performance gain on parallelizing layers;
- Inputs/outputs of different lengths $\rightarrow$  only matching batch sizes can be parallelized;

...also important: *Attention Mechanism*

also, long sentences lead to vanishing/exploding gradients.

# Transformer 

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/transformer.PNG"/><br/>
<br/><small>The transformer architecture.</a>
</small>
</p>


[Vaswani et al. (Google), Attention is all you need, Arxiv](https://arxiv.org/abs/1706.03762)


## Encoder

Model has no recurrence or convolution, so we need a \textbf{positional encoder} to give context of order of words in sentence. Example: \textit{My \textbf{dog} is loud} vs \textit{I look like a \textbf{dog}}. %Dimensionality of PE is the same as embeddings $d$ so that they can be summed.

$$
 PE_{(pos,2i)} = sin\left(\frac{pos}{10000^{2i/d}}\right) \text {\hspace{0.25cm} and \hspace{0.25cm}} PE_{(pos,2i+1)} = cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

**Multi-head Attention** solves for $n$ heads, *What part of the input should I focus on?*

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/transformer_attention.PNG"/><br/>
<br/><small>The transformer architecture.</a>
</small>
</p>

$$
Attention(K, V, Q) = softmax\left(QK^T / \sqrt{d_k}\right) V
$$

$$
MHA(K, V, Q) = [head_0,.., head_n]W^M \text{, \hspace{0.2cm}} head_i = Attention(KW^K_i, VW^V_i, QW^Q_i)
$$

**Feed Forward** is a regressor (single hidden-layer DNN) that transforms the attention vectors into a form that is valid as input to the decoder.

\end{tiny}

## Decoder 

**Masked Multi-head Attention** similar to regular MHA, but replaces upper diagonal of attention vector by zeros, to hide next word from model model.

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/transformer_attention_masked.png"/><br/>
<br/><small>The transformer  attention mechanism masked.</a>
</small>
</p>

Multi-head attention determines how the words in input \& output sentences interact.

*Linear* expands the space into an array of size equals to French vocabulary. 

*Softmax* tranforms into a prob. distribution. Word with highest probability is picked.

**Computational Complexity**:


<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/transformer_table.png"/><br/>
<br/><small>The transformer  attention mechanism masked.</a>
</small>
</p>

$n$: sequence length, $d$: representation dim., $k$: kernel size; $r$: size of neighbourhood.

RNN: $d^2$ multiplications (multiplication of weights in fully connected layer of DNN) for each of the $n$ words in the sentence

Self-Attn (Encoder):  Attention matrix $n^2$, where each element of attention matrix has embedding $d$

amount of computation that can be parallelized, as measured by the minimum number of sequential operations

Why is it that $n^2 d$ is better than $n d^2$?

Sentences length $n \approx 70$, and word embeddings $d \approx 2000$.

**Parallelism:** till limited Dec. batch size, but no more Enc. recursion!

**Rationale:**
- Encoder learns English language (context);
- Decoder learnt the English-to-French translation;

Can we get rid of the Decoder and use only the Encoder to learn complex tasks?

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/transformer_3d_original.png"/><br/>
<br/><small>The transformer  attention mechanism masked.</a>
</small>
</p>

# BERT: Bidirectional Encoder Representation from Transformers

BERT is a stack of Transformer encoders. Learns language \textit{context}.


<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/BERT.PNG"/><br/>
<br/><small>The transformer  attention mechanism masked.</a>
</small>
</p>


Pre-Training: 2 self-supervised prediction tasks at same time:
- tasks: Masked Language Model; and Next Sentence Prediction;
- trained on Wikipedia, 24 BERT layers, batches of 256 sentences * 512 tokens;

Input = [CLS] the man went to [MASK] store [SEP]
\hspace{0.7cm}he bought a gallon [MASK] milk [SEP]
Label = IsNext

Input = [CLS] the man [MASK] to the store [SEP]
\hspace{0.7cm}penguin [MASK] are flight \#\#less birds [SEP]
Label = NotNext

### Input Layout

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/BERT_input.PNG"/><br/>
<br/><small>The transformer  attention mechanism masked.</a>
</small>
</p>

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, Google AI](https://arxiv.org/abs/1810.04805)

- Fine-Tuning:  Adding one layer to a pre-trained BERT to learn to solve most tasks.

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/BERT_tasks.png"/><br/>
<br/><small>The transformer  attention mechanism masked.</a>
</small>
</p>

Information encoded by BERT is useful but, on its own, insufficient to perform a translation task [due to no left-to-right prediction]. However, "BERT pre-training allows for a better initialization point for [an] NMT model", \href{https://arxiv.org/abs/1909.12744}{Clichant et al.,On the use of BERT for Neural Machine Translation, arXiv}

# Microsoft ZeRO \& DeepSpeed

Remember BERT is a stack of Transformer Encoders, i.e. a sequence of matrix-vector multiplication?

Remember Data Parallelism (DP) and Model Parallelism (MP)?

Remember the inputs and outputs on each layer of forward and backward propagation?

**ZeRO (Zero Redundancy Optimizer) combines all**: \textit{"[...] achieves the computation/communication efficiency of DP while achieving memory efficiency of MP, [...] retaining the computational granularity and communication volume of DP using a dynamic
communication schedule during training"

[Video](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)

%ZeRO removes the memory redundancies across data-parallel processes by partitioning the model states—parameters, gradients, and optimizer (Adam) state—across data parallel processes instead of replicating them. 

%partitions optimizer states, gradients and parameters

%We show that ZeRO can be combined with any model parallelism
 
%We call this ZeRO-powered data parallelism, which allows per-device memory usage to scale linearly with the degree of data parallelism and incurs similar communication volume as data parallelism. 

Sources:
- [ZeRO \& DeepSpeed announcement page](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/);
- [Turing-NLG blog post](ttps://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft/);
- [www.deepspeed.ai/](https://www.deepspeed.ai/);
- [DeepSpeed github docs](https://github.com/microsoft/DeepSpeed\#further-reading);
- [ZeRO paper](https://www.microsoft.com/en-us/research/publication/zero-memory-optimizations-toward-training-trillion-parameter-models/);

**State of the Art**: \pause Bert-large (0.3B), GPT-2 (1.5B), Megatron-LM (8.3B), T5 (11B). ZeRO can run 100B parameters but they didn't, takes longer than a year for training! So they ran 17B.

### Superlinear speed-up

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/ZeRO_superlinear_speedup_60B_parameter.PNG"/><br/>
<br/><small>The transformer  attention mechanism masked.</a>
</small>
</p>

super-linear speedup in the regime of 64-400 GPUs

This is a property of ZeRO-DP
which reduces the memory footprint of the model states as we increase the DP degree, allowing
us to fit larger batch sizes per GPU"

