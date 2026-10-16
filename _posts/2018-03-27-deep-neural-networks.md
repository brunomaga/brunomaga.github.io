---
layout: post
title:  "Deep Neural Networks, backpropagation, autodiff, dropout, CNNs and embeddings"
categories: ["Machine Learning"]
tags: ["deep learning", "backpropagation", "autodiff", "CNN", "embeddings"]
---

Deep Neural Networks -- also known as Multi-Layer Perceptrons -- are **[universal approximators](https://en.wikipedia.org/wiki/Universal_approximation_theorem)**, i.e. they can approximate any continuous function on a bounded domain. They are the most relevant ML tool in supervised learning, as they are part of the models used for most supervised tasks.

A basic NN has one input layer of size $$D$$, $$L$$ hidden layers of size $$K$$, and one output layer. It is a **feedforward network**: the computation starts with the input on the left and flows to the right, with no feedback loop. It can be used for regression (when input and output are provided) and for classification (when the input is provided and the output is the class).

{: style="text-align:center; font-size: small;"}
<img width="33%" height="33%" src="{{ site.assets }}/deep-neural-networks/dnn.svg">

{: style="text-align:center; font-size: small;"}
Source: Machine Learning lecture notes, M. Jaggi, EPFL

The input and output layers work as in the regression methods described previously. The hidden layers are different. The output at node $$j$$ of hidden layer $$l$$ is:

$$
x_j^{(l)} = \phi ( \sum_i w_{i,j}^{(l)} x_i^{(l-1)}+b_j^{(l)}). 
$$

where $$b$$ is the bias term, trained like any other weight. The output is computed in two steps: first the weighted sum of the inputs, then an **activation function** $$\phi$$ applied to this sum. Equivalently, the function implemented by each layer is:

$$
x^{(l)} = f^{(l)} (x^{(l-1)}) = \phi ((W^{(l)})^T x^{(l-1)}+b^{(l)})
$$

where $$f$$ is the regression function of the neurons, as before. Popular choices of activation functions are:
- [sigmoid function](https://www.wolframalpha.com/input/?i=1%2F(1%2Be%5E%7B-x%7D)): $$ \phi(x) = \frac{1}{1+e^{-x}} $$, positive and bounded;
- [*tanh*](https://www.wolframalpha.com/input/?i=%5Cfrac%7Be%5Ex+-+e%5E%7B-x%7D%7D%7Be%5Ex%2Be%5E%7B-x%7D%7D): $$ \phi(x) = \frac{e^x - e^{-x}}{e^x+e^{-x}} $$, balanced (positive and negative) and bounded;
- [Rectified Linear Unit – ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)): $$\phi(x) = [x]_+ = max\{0,x\}$$, always non-negative and unbounded. It is often used because it is computationally *cheap* and mitigates the *vanishing gradients* problem in DNNs, since the positive output region does not have a small derivative; 
- Leaky ReLU: $$\phi(x) = max \{\alpha x,x\}$$. It fixes the *dying ReLU* problem of non-leaky rectified linear units;
  - If the dot product of the input to a ReLU with its weights is negative, the output is 0. The gradient of $$ max\{ 0,x \} $$ is $$0$$ when the output is $$0$$. If the output is consistently $$0$$ (e.g. if the ReLU has a large negative bias), then the gradient is consistently 0. The error signal backpropagated from later layers is multiplied by this 0, so no error signal reaches earlier layers: the ReLU has *died*. With leaky ReLUs and $$\alpha > 0$$, the gradient is never 0, which avoids this problem.  

## Back-propagation

As in previous regression use cases, training minimizes the loss of the system, e.g. with Stochastic Gradient Descent. Gradient descent computes the gradient of the cost function for a given input sample (with respect to all weights and bias terms of the network), and then takes a small step in the opposite direction. The derivative with respect to a given parameter is computed with the chain rule of calculus. The cost function can be written as (details omitted):

$$
\mathcal{L} = \frac{1}{N} \sum_{n=1}^N ( y_n - f^{(L+1)} \circ ... \circ f^{(2)} \circ f^{(1)} (x_n^{(0)}) ) ^2
$$

Since there are usually many parameters, computing these derivatives for each parameter individually is not efficient. The **back propagation** algorithm computes them more efficiently. The aim is to compute:

$$
\frac{\partial \mathcal{L}_n}{\partial w_{i,j}^{(l)}}, l=1, ..., L+1 \text {, and } \frac{\partial \mathcal{L}_n}{\partial b_{j}^{(l)}}, l=1, ..., L+1 
\label{eq_gradient}
$$

For simplicity, the total input computed at the $$l$$-th layer is denoted as

$$
z^{(l)} = (W^{(l)})^T x^{(l-1)}+b^{(l)} .
\label{eq_z}
$$

i.e. $$z^{(l)}$$ is the total input of layer $$l$$ before the activation function is applied.
These quantities are computed by a forward pass in the network, starting at $$ x^{(0)} = x_n $$ and applying this recursion for $$ l=1,...,L+1 $$.

Now, define

$$
\delta_j^{(l)} = \frac{\partial \mathcal{L}_n}{\partial z_j^{(l)}}.
\label{eq_delta}
$$

These quantities are easier to compute with a backward pass:

$$
\begin{align*}
\delta_j^{(l)} = & \frac{\partial \mathcal{L}_n}{\partial z_j^{(l)}} \\
= & \sum_k \frac{\partial \mathcal{L}_n}{\partial z_k^{(l+1)}} \frac{\partial z_k^{(l+1)}}{\partial z_j^{(l)}} \\ 
= & \sum_k \delta_k^{(l+1)} \frac{\partial \left[ (W^{(l+1)})^T x^{(l)}+b^{(l+1)} \right]_k}{\partial z_j^{(l)}} \\
= & \sum_k \delta_k^{(l+1)}  W_{j,k}^{(l+1)} \frac{\partial x_j^{(l)}}{\partial z_j^{(l)}} \\
= & \sum_k \delta_k^{(l+1)} W_{j,k}^{(l+1)} \phi '(z_j^{(l)}) 
\end{align*}
$$

The second equality follows from the chain rule. Going back to equation \ref{eq_gradient}:

$$
\frac{\partial \mathcal{L}_n}{\partial w_{i,j}^{(l)}} = \sum_k \frac{\partial \mathcal{L}_n}{\partial z_k^{(l)}} \frac{\partial z_k^{(l)}}{\partial w_{i,j}^{(l)}} = \delta_j^{(l)} x_i^{(l-1)}
\label{eq_w}
$$

This follows from the chain rule, the definition of $$\delta$$ in equation \ref{eq_delta}, and the partial derivative of equation \ref{eq_z} with respect to $$w$$. Similarly:

$$
\frac{\partial \mathcal{L}_n}{\partial b_{j}^{(l)}} = \sum_k \frac{\partial \mathcal{L}_n}{\partial z_k^{(l)}} \frac{\partial z_k^{(l)}}{\partial b_{j}^{(l)}} = \delta_j^{(l)} \cdot 1 = \delta_j^{(l)}
\label{eq_b}
$$

The complete back propagation algorithm is summarized as:
1. **Forward pass**: set $$ x^{(0)} = x_n $$. Compute $$ z^{(l)} $$ for $$ l=1,...,L+1 $$;
2. **Backward pass**: set $$ \delta ^{(L+1)} $$ using the appropriate loss and activation function. Compute $$ \delta ^{(l)} $$ for $$ l = L, ..., 1 $$;
3. **Final computation**: for all parameters, compute $$ \frac{\partial \mathcal{L}_n}{\partial w_{i,j}^{(l)}} $$ (eq. \ref{eq_w}) and $$ \frac{\partial \mathcal{L}_n}{\partial b_{j}^{(l)}} $$ (eq. \ref{eq_b}).

## Automatic Differentiation

The chain rule provides the theoretical framework, but not an efficient way to compute it. In practice, unlike the back-propagation algorithm above, <a href="https://en.wikipedia.org/wiki/Automatic_differentiation">automatic differentiation ("autodiff")</a> algorithms such as <a href="https://pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html">PyTorch's autograd</a> do not require the derivatives of the loss and activation functions to be specified explicitly.

Instead, a computational graph (a DAG) is created dynamically: nodes are added to the graph as the modules' `forward` methods are called. The backward propagation then applies the chain rule to the derivatives (`grad_fn`) of the nodes in the graph. Recursive calls add new nodes to the graph at every call, and conditional calls may add different nodes at each `forward` pass. Therefore, the computational graph cannot be extracted directly from the model definition (see an example [here](https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html)). When `tensor.backward()` is called, it recursively computes the gradients up to the leaf nodes of the graph. The graph is then discarded, and a new one may be created by a new `forward` call (unless `retain_graph=True` is set in the `backward` function). <a href="https://towardsdatascience.com/automatic-differentiation-explained-b4ba8e60c2ad">Chi-Feng Wang's post</a> gives an illustrative example of the automatic differentiation graph, and the <a href="{{ site.assets }}/resources/princeton_course_autodiff.pdf">Princeton course COS 324 on automatic differentiation</a> gives a thorough explanation of the whole process.

Storing a computational graph is also useful for activation checkpointing, where a module's output is recomputed during the backward pass. The post [How Activation Checkpointing enables scaling up training deep learning models](https://medium.com/pytorch/how-activation-checkpointing-enables-scaling-up-training-deep-learning-models-7a93ae01ff2d) gives a good example of activation checkpointing with the reentrant and non-reentrant PyTorch implementations.

It also follows that inference requires less memory than training, since training requires the computational graph to store all activations. Step-by-step overviews of back propagation applied to a real example are available in [Hooks for autograd saved tensors](https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html) and [PyTorch Autograd: Backpropagation](https://colab.research.google.com/drive/1aWNdmYt7RcHMbUk-Xz2Cv5-cGFSWPXe0#scrollTo=LfW6wYg42SEW).

## Dropout

[Dropout](https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5) is a method that drops neurons (in different layers) with a given probability $$p$$ at training time. A new network is sampled for each training minibatch. 
The outputs of the kept neurons are scaled by $$1/(1-p)$$. At inference time, all neurons are used and no dropout is applied. A simple implementation is:

```python
class MyDropout(nn.Module):
    def forward(self, X):
        if self.training:
            binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
            return X * binomial.sample(X.size()) * (1.0/(1-self.p))
        return X
```
Dropout reduces overfitting: the network learns not to rely on any single activation, so it learns *redundant* ways of solving the task with multiple neurons. It also leads to sparse activations, similar to a regularization (L1).
Dropout can be improved by adding max-norm regularization, a decaying learning rate and high momentum.
The original publication found that dropping 20% of input units and 50% of hidden units was often optimal. Dropout is computationally cheaper than regular model averaging of multiple trained DNNs. However, it takes 2-3 times longer to train than a single fully-connected DNN, because it requires more epochs, as parameter updates are very noisy.
Since a fully connected layer holds most of the parameters, it is prone to overfitting. Therefore, dropout improves model generalization.

{: style="text-align:center; font-size: small;"}
<img width="35%" height="35%" src="{{ site.assets }}/deep-neural-networks/dropout.svg">

## Convolutional Neural Networks
 
Large neural networks have a very high number of parameters, so they may require a large amount of data to be trained. In some scenarios, local connectivity should suffice. E.g. in an audio stream, it is natural to process an input stream $$ x^{(0)}[n] $$ with a linear time-invariant filter, whose output $$ x^{(1)}[n] $$ is the convolution of the input $$ x^{(0)}[n] $$ and the filter $$ f [n] $$:

$$
x^{(1)}[n] = \sum_k f[k]x^{(0)} [n-k]
$$


The filter $$f$$ is local, i.e. $$ f[k]=0 $$ for $$ \vert k \vert \ge K $$. Different types of filters bring out different aspects of the underlying signal, e.g. averaging smooths features, and a *high-pass* filter enhances differences between neighboring elements. The same applies to an image:

$$
x^{(1)}[n,m] = \sum_{k,l} f[k,l]x^{(0)} [n-k,m-l]
$$

The output $$ x^{(1)} $$ at position $$ [n, m] $$ depends only on the values of the input $$ x^{(0)} $$ at positions close to $$ [n, m] $$. Therefore, a fully connected network is not needed: a network that represents the local structure suffices, with fewer parameters. This structure also implies that the same filter (i.e., the same connection pattern and the same weights) should be used at every position. This is called **weight sharing**, and it reduces the number of parameters even further. The difference between a locally/sparsely connected and a fully connected network is illustrated below:

{: style="text-align:center; font-size: small;"}
<img width="35%" height="35%" src="{{ site.assets }}/deep-neural-networks/fully_sparse_connectivity.svg">

It is common to compute the outputs of multiple filters instead of a single one. These outputs are called channels, and they add parameters to the model.
With several channels, the output of the next level is not 2D but 3D. Deeper layers have more channels but a smaller “footprint.” Applying local connectivity to a 2D input in a *deep* neural network leads to the following structure:

{: style="text-align:center; font-size: small;"}
<img width="65%" height="65%" src="{{ site.assets }}/deep-neural-networks/cnn.png">

There are three types of operations:
- convolutional layer: applies a convolution operation to the input and passes the result to the next layer. The convolution emulates the response of an individual neuron to visual stimuli;
  - convolution is a mathematical operation on two functions that produces a third function, which expresses how the shape of one is modified by the other. Convolution is similar to cross-correlation;
- pooling: combines the outputs of neuron clusters at one layer into a single neuron in the next layer.
  - A very common operation is max pooling, which uses the maximum value of each cluster of neurons at the prior layer;
- fully connected: similar to a multi-layer perceptron, provides the *universal approximator* function;

The size of each image typically decreases through the layers, due to the handling of the boundaries or to subsampling.

Training is done with backpropagation, ignoring that some weights are shared, i.e. treating the weight of each edge as an independent variable. Once the gradient is computed for this network with independent weights, the gradients of all edges that share the same weight are summed. This gives the gradient of the network with weight sharing. 

## Skip connections

In very deep networks, the performance of the model drops as the depth increases. This is known as the *degradation problem*. It is not caused by overfitting, since the training error also increases. Possible causes are vanishing and/or exploding gradients. One remedy is to use [normalization](https://arxiv.org/abs/1803.08494) techniques, which keep the gradient norms in a healthy range.

A better remedy is to feed layers with the outputs of layers that are not directly connected to them. These connections *skip* some layers and feed the output of one layer as input to later layers, hence the name *skip connections*. In practice, the input of a given layer is a combination (typically an addition or concatenation) of the outputs of several previous layers. 

{: style="text-align:center; font-size: small;"}
<img width="65%" height="65%" src="{{ site.assets }}/deep-neural-networks/densenet.svg">

{: style="text-align:center; font-size: small;"}
An example of a CNN with skip connections (DenseNet). Adapted from: <a href="https://arxiv.org/abs/1608.06993">Densely Connected Convolutional Networks</a> 

Many CNN implementations exist with different layouts of skip connections, e.g. [ResNet](https://arxiv.org/abs/1512.03385), [DenseNet](https://arxiv.org/abs/1608.06993) and [U-net](https://arxiv.org/abs/1505.04597).


## Embeddings

Embeddings in machine learning are dense, low-dimensional vector representations of data (e.g., words, images, or users) that capture their semantic or contextual meaning in a continuous vector space. They are collected by passing input data (e.g., text, images) through a *trained* machine learning model (e.g., a neural network) and extracting the values of a specific intermediate layer, typically one before the final output layer.

The most common data types and their embeddings are:
- words: one-hot vectors or [Word2vec](https://en.wikipedia.org/wiki/Word2vec) (*skipgram* or *continuous bag of words*)
  - skipgram: slower to train ("3 days"), better at capturing semantic relationships, e.g. for 'cat' it returns 'dog' as a word with a close embedding;
  - CBOW: faster to train ("few hours"), better at capturing syntactic relationships between words, e.g. for 'cat' it returns 'cats';
- text (word sequences): [BERT]({{ site.baseurl }}{% post_url 2020-02-28-learning-from-sequences %})
- non-textual sequences: [Encoder-Decoders, e.g. LSTMs, RNNs]({{ site.baseurl }}{% post_url 2020-02-28-learning-from-sequences %})
- point cluster or array: [Principal Component Analysis]({{ site.baseurl }}{% post_url 2017-11-01-unsupervised-learning %})
- images:
  - in a classification task: the activation of the last layer *before* the logit/softmax layer, i.e. the input to the final layer, which is the output of the second-to-last layer;
  - in an image-to-image task, e.g. segmentation with a [U-net](https://arxiv.org/abs/1505.04597): the activation of the last downsampling layer, which is the input of the first upsampling layer, i.e. the [*information bottleneck*](https://en.wikipedia.org/wiki/Information_bottleneck_method);
- audio: a waveform is a long 1D sequence of samples (typically 16-bit integers at 16 to 48 kHz). It is too long to model directly, so it is first converted into a shorter sequence of frames:
  - spectrograms: the waveform is converted into a log-Mel spectrogram, a 2D time-frequency image, and fed to a CNN or Transformer encoder whose activations are the embeddings. E.g. [Whisper](https://arxiv.org/abs/2212.04356) resamples audio to 16 kHz and computes 80 Mel channels on 25 ms windows with a 10 ms stride. A vocoder turns a spectrogram back into a waveform, as in the text-to-speech model [TorToise](https://arxiv.org/abs/2305.07243);
  - neural audio codecs: a convolutional encoder downsamples the waveform into embeddings, and a decoder with transposed convolutions reconstructs it. In between, residual vector quantization (RVQ) turns each embedding into discrete tokens: the embedding is replaced by its nearest entry in a codebook, the residual is quantized with a second codebook, and so on. E.g. [EnCodec](https://arxiv.org/abs/2210.13438) turns 24 kHz audio into 75 embeddings per second, a 320-fold reduction, and the number of codebooks sets the bitrate (8 codebooks of 1024 entries give 6 kbps), while [DAC](https://arxiv.org/abs/2306.06546) compresses 44.1 kHz audio into 8 kbps of tokens. These tokens can be modelled like text, e.g. [VALL-E](https://arxiv.org/abs/2301.02111) does text-to-speech by predicting EnCodec tokens with a language model;
- videos: the input is 5D (batch size, time, channels, height, width), i.e. a sequence of video frames stacked along the time dimension. Embeddings are collected as in a regular CNN (adapted to use 3D instead of 2D convolutions, etc.);
- graphs: the embedding of a node is its state after several steps of the [message passing algorithm](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html), and the embedding of the graph is a readout (e.g. sum, mean or max) over all node embeddings. A message passing step is:

  $$x_i^{(k)} = \gamma^{(k)} \left( x_i^{(k-1)}, \Box_{j \in \mathcal{N}(i)} \, \phi^{(k)} \left( x_i^{(k-1)}, x_j^{(k-1)}, e_{j,i} \right) \right)$$

  where:
  - $$x_i^{(k)}$$ is the embedding of node $$i$$ at message passing step $$k$$;
  - $$\mathcal{N}(i)$$ is the set of neighbors of node $$i$$;
  - $$e_{j,i}$$ is the embedding of the edge from node $$j$$ to node $$i$$;
  - $$\Box$$ is a differentiable, permutation-invariant function, e.g. sum, mean or max;
  - and $$\gamma$$ and $$\phi$$ are differentiable functions such as DNNs or RNNs (LSTMs, GRUs).
 

