---
layout: post
title:  "AI Supercomputing: Levels of Parallelism, Linear Regression, DNNs and CNNs"
categories: [machine learning, supercomputing]
tags: [machinelearning]
---


Machine Learning is driven by mathematical models that try to *learn* from data. With time, the complexity of the data is continuously increasing, due e.g. to higher-resolution photos, larger text databases, higher number of observable features on input datapoints, etc. The computing power available *tends* to follow, or somehow adapt, as observed by [Moore's law](https://en.wikipedia.org/wiki/Moore%27s_law). However, in the occurence of very large datasets and very computationally-expensive learning models, the learning process can't be performed in regular machine, due to insufficient memory or an infeasibly-high training time. This is where **AI Supercomputing** --- or technically speaking **Parallel-Distributed computing of Machine Learning algorithms** ---  comes into place. 

AI Supercomputing focuses on how to distribute data (models, inputs) and computation across several compute units (vector units, CPU cores, GPU cores, and machines). The goal is to distribute the data in such way that the learning algorithm can be executed simultaneously (*parallelized*) with different datasets and computation across all processors. This leads to a speedup of the solution and a reduction of memory usage per machine. On the other hand, distributing data that are computationally dependent --- i.e. that need to be used together on a computation at some point --- introduces another layer of complexity due to the overhead on the communication required to move variables across machines or compute units. So the ideal algorithm can be characterized as the one that exhibits:

1. homogeneous distribution of data across memory units, i.e. balanced memory usage, ideally without data repetitions; and
2. homogeneous amount of computation assigned to each compute unit, i.e. balanced computation; and
3. a minimal amount of communication across memory/compute units, or ideally a zero-communication overhead if overlaping of communication and computation is possible.
4. the **linear scaling** of the execution time of the algorithm based on the number of precessors. This means that, by increasing (e.g. doubling) the compute resources, we decrease (halve) the computation.

In practice, guaranteeting these properties is very hard (particularly the perfectly-linear scaling), but an almost-ideal homogeneity of computation and data with quasi-linear scaling can be achieve on cleverly designed algorithms. Let's start with the basics.


# Linear Regression and Mutual Exclusion

Take a simple example of linear regression with input variables $x$, labels $y$, learnt weights $w$ and a loss function set as the Mean Absolute Error.
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

This operation is *memory-safe* for $f(x_n)$, but unsafe for $MAE(w)$. This means that all write operations (variable assignments) in $f(x_n)$ are performed on different memory positions (vector indices) while in $MAE(w)$ this is not the case. In practice, computing $f(x_n)$ requires a sum of $M$ independent products written on each index of $x$, while the final value holding $MAE(w)$ is a constant that needs to be updated with the value of every term $\|y_n - f(x_n)\|$. What happens to the execution time and final value when several threads try to write simultaneously to the memory space holding it? Let's study four options:

1. base case: no parallelism, i.e. use only a single thread. The output is correct but its computation is **slow**. This implementation is available in <a href="/assets/AI-Supercomputing/AI_SC_1.cpp">AI\_SC\_1.cpp</a>;
2. each thread updates the MAE sum, at every update of the term $\|y_n - f(x_n)\|$. This implementation provides an almost-linear scaling of computation (there's some overhead on initiating threads). However the output is **wrong** as several memory corruptions occur when multiple threads try to update the final value (or its memory position) at the same time (<a href="/assets/AI-Supercomputing/AI_SC_2.cpp">AI\_SC\_2.cpp</a>);
3. same as before, however we add a *mutex* (mutual exclusion) control object. A mutex allows for a section of the code to be *locked* by a thread, in such way that no other thread  can enter it until the first thread has unlocked it. We can use it to protect the operation responsible for the memory update of the variable being accessed *concurrently* at every update operation. This method gives an accurate result, however it is **very slow**, due to the computational overhead introduced by the mutex itself (<a href="/assets/AI-Supercomputing/AI_SC_3.cpp">AI\_SC\_3.cpp</a>);
4. finally, we try the same as before, but we compute the sums of each thread independently and perform a single concurrent memory update of the final variable once, and only at the end of the execution. This approach gives the correct result with an almost-linear scaling of the algorithm (<a href="/assets/AI-Supercomputing/AI_SC_4.cpp">AI\_SC\_4.cpp</a>);

These examples give us an important message: concurrency control is computational expensive; parallelization is worth it; and matrix-vector multiplications and updates (that underlie most operations in Machine Learning) can be performed efficiently when individual contributions from compute units are decomposed, computed independently and *reduced* at the end.

However, not all operations can be easily parallelizable and reproducible. A common example is any method based on random number generation, such as [Markov chain Monte Carlo (MCMC)](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) methods applied in Bayesian Optimization. In such scenarios, where compute units *draw* continuosly random number(s) for every datapoint, we need to think of the efficiency problem as a trade-off between reproduciblity and efficency. We may allow a fully-parallel algorithm (by having a random number generator per compute unit, therefore yielding efficient computation with results that can only be reproduced on executions with the same number of processors and seeds). Or we can have a single random generator with mutually-exclusive access and then generate (at the beginning of the execution) a sequence of randomly-generated numbers which consistent across executions. Such details fall out of the scope of this post, so we'll move on with the main topic.

# From CPU to GPU and TPU/IPU

We've mentioned several time the words "compute unit", "compute architecture" and "processor". Those are random jargon used to describe any piece of hardware that can perform algorithmic computations. On the subject of Machine Learning, three main architectures are relevant:
1. CPU (Central Processing Unit): the main processor of any desktop, laptop or mobile computer. Intended to be a high speed processor to perform efficiently most *iterative* tasks that we do on a daily basis, such as text editors, mouse/keyboard interfaces, browser, and most applications that run on the operative systems. As on a normal session we interact with few of these apps at a time (and keep others on the background), the CPU was suited to this type of behavior and built as a small set of cores with a high-frequency clockspeed. 
2. GPU (Graphics Processing Unit): initially designed for graphics processing, which are by nature highly parallel (e.g. the computaion of each pixel value on a screen) and based on Matrix/Vector operations (such as rotation, translation and scaling of 3D meshes). As these operations are usually simple and simillar algebraic operations performed simultaneously on millions of inputs (or large vectors/matrices), the GPU was designed as a high number of compute cores, running at a low-clock speed (we'll see later why);
3. IPU (Inteligent Processing Unit) or TPU (Tensor Processing Unit): a natural evolution of the GPU towards Machine Learning applications. Allows for a higher throughput (or total compute power) by reducing the processir's operation set to Machine Learning purposes, thus decreaing processor size and allowing an increased number of cores. 

The main question is: what drives GPU/TPU/IPU processor designers to lower the clockspeed of their processors and increase the number of cores, instead of just designing their chips at the same clock speed and core count as CPUs? The answer is **power consumption**, **[heat dissipation](https://en.wikipedia.org/wiki/List_of_CPU_power_dissipation_figures)** and **temperature**. In practice, to increase the clockspeed we usually reduce the transistor size thus inserting [more transistors in the same chip](https://en.wikipedia.org/wiki/Transistor_count) or placing them more tightly in the same area. This leads to an increase of power comsumption and temperature that has been observed to grow exponentially with the increase of the clockspeed: 

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/a53-power-curve.png"/><br/>
<br/><small>Exponential increase of power comsumption (y axis) for a linear increase of processor frequency (x axis),<br/> for processor with one to four cores (colour coded) of the Samsung Exynos 7420 processor. (source: <a href="https://www.anandtech.com/show/9330/exynos-7420-deep-dive/5">AnandTech</a>)</small>
</p>

Therefore, for a simillar throughput, many cores of low clock frequency yield the same results of few cores of high frequency, yet at a much lower power comsunption. Equivalently, For a fixed power consumption, one can extract more compute power from many low frequency cores than from a few high frequency cores.

The take-home message is: in regression problems, since computational reductions happen rarely and are very efficient (as we saw on the Linear Regression example), then the *only* hardware feature that dictates performance is total GHz across all compute cores (i.e. efficiency is independent of the number of cores). Or more importantly, one unit of throughput commonly used is the number of **FLOPs** )(Floating Point Operations per second), since an instruction in a processor can execute simultaneously several operations, using a techique called [SIMD (Single Instruction Multiple Data](https://en.wikipedia.org/wiki/SIMD) or [MIMD (Multiple Instructions Multiple Data)](https://en.wikipedia.org/wiki/MIMD). We'll skip the details about SIMD and MIMD functioning as they're not relevant in the context of this post. 

Looking at the previous plot, we see that, to efficiently maximize GHz/FLOPs throughput, one is much more efficient by having several processors of low clock frequency, instead of fewer of a higher frequency. This is, at a very high level, the main different between a CPU and a GPU architecture, and this explains why GPUs tend to be the preferred choice to compute Machine Learning trainign problems. This phylosophy led to the creation of [TPUs (Tenso Processing Units)](https://en.wikipedia.org/wiki/Tensor_processing_unit) and [IPUs (Inteligent Processing Unit)](https://www.graphcore.ai/products/ipu), that explore this trade-off of number of cores vs clock-frequency, with lower-precision floating point representations (to maximize SIMD), and ML-specialized logical units on the processors, to augment further the throughput. Let's check the  common CPU, GPU, and IPU specifications for processors used in compute clusters dedicated to ML tasks:


|                    | **cores x clock-frequency**  $\hspace{1cm}$ | **FLOPs (32 bits representation)**  $\hspace{1cm}$ | **Max RAM** |
|---------------------	|-----------------------------	|------------------------------------	|-------------	|
| **Intel Xeon 8180** $\hspace{1cm}$ | 28x 2.5 Ghz 	| 1.36 TFLOPS 		| 768 GB       	|
| **Tesla K80**       	| 4992x 0.56 Ghz             	| 8.73 TFLOPS                         	| 2x 12GB     	|
| **Graphcore IPU**   	| 1216 x 1.6Ghz [1]           	| 31.1 TFLOPS                     	| 304 MiB [2] 	|
|---------------------	|-----------------------------	|------------------------------------	|-------------	|

<br/>
Some important remarks on the IPU architecture: [1] TPUs use Accumulating Matrix Product (AMP) units, allowing 16 single-precision floating point operations per clock cycle, therefore the processor is not directly comparable by looking simply at core count and clock-frequency. To learn more about Graphcore's IPU, see the technical report [Dissecting the Graphcore IPU Architecture via Microbenchmarking, Citadel Technical Report, 7 December 2019](https://www.graphcore.ai/products/ipu).

One main observation derives from the previous table. Memory bandwidth increases from CPU to GPU to IPU, however its total capacity is reduced. In practice, small memory is compensated by a very low latency between processor and memory, allowing onloading of offloading of large datasets more efficiently. So how do we train large models on small memory regions?

# CPU offloading (vDNN)

A common technique to handle memory limitations is offloading. In this particular example, we'll focus on GPU to CPU offloading. The main goal of this method is to identify and move to the GPU only the portions of data that are required for each computation step, and keep the remaining on the CPU.

Take this example of training of a multi-layer Deep Neural Network.  We've seen on a [previous post about DNNs]({{ site.baseurl }}{% post_url 2018-02-27-Deep-Neural-Networks %}) that the output $x$ for a given layer $l$ of the network, is represent as:

$$
x^{(l)} = f^{(l)} (x^{(l-1)}) = \phi ((W^{(l)})^T x^{(l-1)})
$$

where $\phi$ is the activation function. The loss is then computed by taking into account the groundtrugh $y$ and the composition of the ouputs of all layers in the neural network, ie:

$$
L = \frac{1}{N} \sum_{n=1}^N | y_n - f^{(L+1)} \circ ... \circ f^{(2)} \circ f^{(1)} (x_n^{(0)}) |
$$

The important concept here is the **composition** of the $f$ function throughout layers. In practice one only needs the current layer's state and previous layer output to perform the computation at every layer. This concept has been explored by the [vDNN (Rhu et al.)](https://arxiv.org/pdf/1602.08124.pdf) and [vDNN+ (Shiram et al)](https://www.cse.iitb.ac.in/~shriramsb/submissions/GPU_mem_ML.pdf) implementations: 

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/vDNN.png"/><br/>
<br/><small>An overview of the vDNN(+) implementation on a convolutional neural network. Red arrays represent the data flow of variables $x$ and $y$ (layers input and output) during forward propagation. Blue arrows represent data flow during backward progagation. Green arrows represent weight variables. Yellow arrows represent the *variables workspace in cuDNN*, needed in certain convolutional algorithms. Source: <a href="https://arxiv.org/pdf/1602.08124.pdf">vDNN (Rhu et al.)</a></small>
</p>

The concept is simple: we store the complete model insmall memory is compensated by a very low latency between processor and memory, allowing onloading of offloading of large datasets more efficiently. T CPU memory (or hard-drive if required), and move the active layer into GPU memory when it needs to be computed. To reduce the waiting time of pushing and pulling a layer into the GPU, a viable optimization is to copy asynchronously (ie on the background) the next layer to be computed, while computing the current layer's update. This way, when the algorithm finished to compute a given layer, it can proceed immediately to the next one as it is already available in memory, thus removing onloading waiting time.

We'll start with the forward pass. Looking at the initial formulation of $x^{(l)}$, we can isolate which variables are used during the forward pass of a given layer. For the computation of the output of a given layer, we need the weights of the neurons in the current layer ($W^{(l)}$) and the outputs of neurons on the previous layer $x^{(l-1)}$.
Therefore, for a given layer, the forward pass is represented as:

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/vDNN2.png"/><br/>
<br/><small>The forward pass on the vDNN(+) implementation on convolutional neural networks. Data not associated with the current layer being processed (layer N) are marked with a black cross and can safely be removed from the GPU's memory. Input variables are $x^{(l-1)}$ (represented as X) and $W^{(l)}$ (as WS). Source: <a href="https://arxiv.org/pdf/1602.08124.pdf">vDNN (Rhu et al.)</a></small>
</p>


The backward propagation phase is trickier. Referring to the same DNN post, we have represented the derivative of the loss of a given neuron $j$ in a given layer $l$, on the input $z^{(l)} = (W^{(l)})^T x^{(l-1)}$ as $\delta_j^{(l)}$, where:

$$
\delta_j^{(l)} =  \frac{\partial L_n}{\partial z_j^{(l)}} = \sum \frac{\partial L_n}{\partial z_k^{(l+1)}} \frac{\partial z_k^{(l+1)}}{\partial z_j^{(l)}} = \sum_k \delta_k^{(l+1)} W_{j,k}^{(l+1)} \phi '(z_j^{(l)})
$$

[//]: # and the final loss function over the weights as:
[//]: #
[//]: # $$
[//]: # \frac{\partial L_n}{\partial w_{i,j}^{(l)}} = \sum \frac{\partial L_n}{\partial z_k^{(l)}} \frac{\partial z_k^{(l)}}{\partial w_{i,j}^{(l)}} = \delta_j^{(l)} x_j^{(l-1)}
[//]: # $$


i.e., for the backward propagation, we require both the input variable $x^{(l-1)}$ (inside $z_j^{(l)}$), the weights $W^{(l+1)}$ and the derivatives $\delta_j^{(l+1)}$. This can now be represented as: 

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/vDNN3.png"/><br/>
<br/><small>The back propagation phase on the vDNN(+) implementation on convolutional neural networks. Data not associated with the current layer being processed (layer 2) are marked with a black cross and can safely be removed from the GPU's memory. Input variables are $x^{(l-1)}$ (represented as X),  $W^{(l+1)}$ (as WS) and $\delta_j^{(l+1)}$ (as dY). Source: <a href="https://arxiv.org/pdf/1602.08124.pdf">vDNN (Rhu et al.)</a></small>
</p>


# Pipeline Parallelism (G-Pipe, PipeDream)

Take the previous neural network with 4 layers stored across a network of processors. For simplicity, we'll call the designated compute unit as a *Worker*. If we allocate each Worker to a layer of the network, we can perform a distributed execution of the training where input and output of connecting layers are communited among the respective Workers. I.e. instead of offloading a layer at a time from GPU to CPU and do the inverse when required, we simple have a network GPUs where layears are distributed. A timeline of the execution could then be represented as:

<p align="center">
<br/>
<img width="35%" height="35%" src="/assets/AI-Supercomputing/Pipedream_DNN_pipeline.PNG"/><br/>
<br/><small>Left-to-right timeline of a serial execution of the training of a deep/convolutional neural net divided across 4 compute units (Workers). Blue squares represent forward passes. Green squares represent backward passes and are defined by two computation steps. The number on each square is the input batch index. Black squares represent moments of idleness, i.e. worker is not performing  any computation. <br/>Source: <a href="https://www.microsoft.com/en-us/research/publication/pipedream-generalized-pipeline-parallelism-for-dnn-training/">PipeDream: Generalized Pipeline Parallelism for DNN Training (Microsoft, arXiv)</a>
</small>
</p>

We notice that most of the available compute time is spent doing nothing. This is due to the data dependency across layers: one worker can only proceed with the forward (backward) pass when the worker with the previous (next) index has finished its computation. A possible improvement is to process a group of input batches simultaneously by using a pipelining technique. In practice, we *feed* to the neural network one group of batches (with a batch count equal to the number of workers), that are past iteratively to the model, i.e. one batch per timestep. At every iteration, a worker performs its forward (backward) pass and passes the relevant data to the worker holding the following (previous) layer of the network. Therefore, after a number of phases equal to the workers count, all workers have been allocated some computation. When all batches inside the group have their backward propagation finished, the model update is performed based on the weights (states) of all batches in the groups. This approach is detailled on the paper [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism (Google, 2018, ArXiv)](https://arxiv.org/abs/1811.06965) and can be illustrated as:

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/Pipedream_DNN_pipeline_parallel.PNG"/><br/>
<br/><small>A pipeline execution of groups of batches, computed as a forward phase of all batches in a group, followed by a backward phase of all batches in the same group. Implementation details in <a href="https://arxiv.org/abs/1811.06965">GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism (Google, 2018, ArXiv)</a>. Image source: <a href="https://www.microsoft.com/en-us/research/publication/pipedream-generalized-pipeline-parallelism-for-dnn-training/">PipeDream: Generalized Pipeline Parallelism for DNN Training (Microsoft, arXiv)</a>
</small>
</p>

Can we do better? Yes! In fact, there's still a big limitation on the previous method: the computation is divided in two chunks referring to a set of forward and a set of backward computation steps, leading to high moments of idleness at the start and end of each computation chunk. Moreover, this is a very restrictive dependency: in fact, to start the backward pass of a single batch we need only the forward pass of that particular batch, and not of all backward passes. This property has been explored by Microsoft and detailed in [PipeDream: Generalized Pipeline Parallelism for DNN Training (Microsoft, arXiv)](https://www.microsoft.com/en-us/research/publication/pipedream-generalized-pipeline-parallelism-for-dnn-training/), and the main ideas are:
- the backward pass of a given batch starts immediately after the forward pass has finished;
- if a worker is allocated a forward pass and a backward pass on the same time iteration, it prioritizes the backward pass and computes the forward pass when it's idle;

The following workflow illustration provides a better overview of the algorithm and its usage of compute resources:

<p align="center">
<br/>
<img width="45%" height="45%" src="/assets/AI-Supercomputing/Pipedream_DNN_pipeline_parallel_Microsoft.PNG"/><br/>
<br/><small>A pipeline execution of a sequence of batches using the PipeDream strategy. A backward propagation of a batch is initiated as soon as its related forward pass has finished. Bacward passes are prioritized over forward passes on each worker. Implementation details and image aource: <a href="https://www.microsoft.com/en-us/research/publication/pipedream-generalized-pipeline-parallelism-for-dnn-training/">PipeDream: Generalized Pipeline Parallelism for DNN Training (Microsoft, arXiv)</a>
</small>
</p>


# Data Parallelism 

Data Parallelism refers to the family of methods that perform parallelism at the data level, i.e. by allocating distinct batches of data to the processors. The previous examples of pipelining are also part of the data parallelism family, since multiple batches of data are executed simultaneously, even though it's not a *purely-parallel* implementation as the batches are processed iteratively and not simultaneously.

Another common approach relies simple on the duplication of the model and batch-parallelism by allocating different batches to different models. The rationale is simple: (1) a copy of the model is instantiated on every processor; (2) the input dataset is distributed equally across all processors; (3) rhe final weight update is computed as the average of the weights of the model on every processor.

<p align="center">
<br/>
<img width="50%" height="50%" src="/assets/AI-Supercomputing/DNN_data_parallelism.png"/><br/>
<br/><small>An illustration of DNN data parallelism on two processors $p0$ and $p1$ computing a dataset divided on two equally-sized batches of datapoints. Execution of both batches occurs in parallel on both processors, containing each a full copy of the DNN model. The final weight update is provided by the average weights of the models.
</small>
</p>

The main advantadge of this method is the linear increase in efficiency, i.e. by doubling the amount of processors, we reduce the training time by half. However, it's not memory efficient, since it requires a duplication of the entire model on all compute units, i.e. increasing number of processors allows only for a speedup in solution, not on the training of larger models.

As a final note, this method does not always require the same network to be copied over to each compute unit. An example of this property is the [dropout]({{ site.baseurl }}{% post_url 2018-02-27-Deep-Neural-Networks %}) technique utilized in Deep Neural Nets, where training on several distinct networks are executed simultaneously (even though usually the same data is executed on all models).

For a thorough analysis of the topic, take a look at the paper [Measuring the Effects of Data Parallelism on Neural Network Training (Google Labs, arXiv)](https://arxiv.org/abs/1811.03600)

# Model parallelism

Model parallelism is another general term for the family of methods that perform parallelism at the model level, i.e. the data being distributed across different processors is not the input dataset, but the model states instead. In fact, we can think of the previous pipelining parallelism as model parallelism as well, as the model is divided layer-wise across several compute nodes. There are other methods for model parallelims, with the most common being the division and allocation of the dimensionality space of input data and model across processors: 

<p align="center">
<br/>
<img width="40%" height="40%" src="/assets/AI-Supercomputing/DNN_model_parallelism.png"/><br/>
<br/><small>A representation of a DNN model parallelism on two processors $p0$ and $p1$. Input dataset and model parameters are divided across processors based on the dimensionality of the input features. Red lines represent weights that have to be communicated to a processor different than the one holding the state of the input data for the same dimension.
</small>
</p>

Looking at the previous picture, we notice a major drawback in this method. During trainig, the constant usage of sums of products using all dimensions on the input space will force processors to continuously communicate those variables among themselves (red lines in the picture). This creates a major drawback on the execution as it requires a tremendous ammount of communication at every layer of the network and for every input batch. Moreover, since the number of weights between two layers grows quadratically with the increase of neurons (e.g. for layers with neuron count $N_1$ and $N_2$, the number of weights are $N_1*N_2$), this method is not usable on large input spaces, as the communication becomes a bottleneck.

## Partial Model Parallelism (CNN)

Before throwing the towel on model parallelism, it is relevant to mention that this type of parallelism has some use cases where it is applicable and highly efficient. A common example is on the parallelism of very high resolution pictures on [Convolutional Neural Networks]({{ site.baseurl }}{% post_url 2018-02-27-Deep-Neural-Networks %}). In practice, due to the filter operator in CNNs, the dependencies (weights) between two neurons on sequential layers is not quadratic on the input (as before), but constant with size $F*F$ for a filter of size $F$.

This method has been detailed by [Dryden et al. (Improving Strong-Scaling of CNN Training by Exploiting Finer-Grained Parallelism, Proc. IPDPS 2019)](https://arxiv.org/pdf/1903.06681.pdf). The functioning is illustrated in the picture below and is as follows:
1. Input dataset (image pixels) are divided on the height and width dimensions across processors;
2. Dependencies among neurons on different dimenstions are limited to the $F \times F$ filter around each pixel. The weight updates can be computed directly if the neurons in the filter fall in the same processor's region, or need to be communicated (as before) otherwise. Neurons that need to be communicated are denominated part of the *halo region* (marked as a violet region in the picture below);
3. Similarly to the "CPU offloading (vDNN)" example above, values that need to be communicated are:
	- input and weights during forward pass;
	- input weights and derivatives during backward pass;

<p align="center">
<br/>
<img width="70%" height="70%" src="/assets/AI-Supercomputing/argonne_parallel_2.PNG"/><br/>
<br/><small>
<b>Illustration of model parallelism applied to Convolutional Neural network. LEFT:</b> Parallelism of the pixels of an image across four processors $p0-p3$. <b><span style="color: red;">red box</span></b>: center of the 3x3 convolution filter; <b><span style="color: red;">red arrow</span></b>: data movement required for updating neuron in center of filter; <b><span style="color: violet;">violet region:</span></b> <i>halo region</i> formed of the elements that need to be communicated at every step. <b>RIGHT:</b> communication between processors $p0$ and $p1$. <b><span style="color: red;">Red arrow</span></b>: forward pass dependencies; <b><span style="color: blue;">blue arrow</span></b>: backward pass dependencies;
</small>
</p>


For completion, the equations of the previous picture are the following:
1. $ y_{k,f,i,j} = \sum_{c=0}^{C-1} \sum_{a=-O}^{O} \sum_{b=-O}^{O} x_{k,c,i+a,j+b} w_{f,c,a+O,b+O} $
2. $ \frac{dL}{dw_{f,c,a,b}} = \sum_{k=0}^{N-1} \sum_{i=0}^{H-1} \sum_{j=0}^{W-1} \frac{dL}{dy_{k, f, i, j}} x_{k, c, i+a-O, j+b-O} $
3. $ \frac{dL}{dx_{k,c,i,j}} = \sum_{j=0}^{F-1} \sum_{a=-O}^{O} \sum_{b=-O}^{O} \frac{dL}{dy_{k, f, i-a, j-b}} w_{f, c, a+O, b+O} $

Can you infer the data dependencies displayed in the picture (red and blue arrows) from these equations? We won't go on details here, but read the  [original paper](https://arxiv.org/pdf/1903.06681.pdf) if you are interested.


### Closing Remarks

In this post, we have shown that:
- Machine Learning problems are highly-parallelizable due to efficient Matrix-vector multiplication, and computational reductions that happen rarely;
- Memory in fast compute architectures is limited in size, but this limitation can be usually overcome by utilizing memory dynamic offloading and onloading between GPU, CPU and Hard-drive;
- Multi-layer architectures can be efficiently parallized by utilizing pipeline techniques;
- Other data parallelism techniques allow for a linear efficiency increase be replicating the model across compute resources and performing a final weight update by averaging across all models;
- Other model parallelism techniques that parallelize on the dimensions of features and latent space are highly ineficient as the communication increases quadratically with the input and hidden layers size;
	- However, models of local model partitioning such as Convolutional Neural Networks can utilise this technique efficiently, due to the local filtering that limits the communication space to the neighborhood of neurons set by the image filter;

There's a class of models that have not been covered: sequence data such as textual sentences. In such cases, the previous techniques can hardly be applied due to the recursive nature of the training algorithm. These topics will be covered on the [next post]({{ site.baseurl }}{% post_url 2020-05-12-AI-Supercomputing %}).

