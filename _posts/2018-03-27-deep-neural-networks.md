---
layout: post
title:  "Deep Neural Networks, backpropagation, autodiff, dropout, CNNs and embeddings"
categories: ["Machine Learning"]
tags: ["deep learning", "backpropagation", "autodiff", "CNN", "embeddings"]
---

Deep Neural Networks -- also known as Multi-Layer Perceptrons -- are **[universal approximators](https://en.wikipedia.org/wiki/Universal_approximation_theorem)**, i.e. they can approximate any continuous function on a bounded domain. They're the most important and relevant ML tool in the field of supervised learning, as they're part of the models used for most supervised tasks.

The structure of a basic NN is the following: one input layer of size $$D$$, $$L$$ hidden layers of size $$K$$, and one output layer. It is a **feedforward network**: the computation performed by the network starts with the input from the left and flows to the right. There is no feedback loop. It can be used for regression (when input and output are provided) and for classification (when the input is provided and the output is the class).

{: style="text-align:center; font-size: small;"}
<img width="30%" height="30%" src="{{ site.assets }}/deep-neural-networks/dnn.svg">

{: style="text-align:center; font-size: small;"}
source: Machine Learning lecture notes, M. Jaggi, EPFL

Input and output layers work similarly to any previous regression method. Hidden layers are different. The output at node $$j$$ in hidden layer $$l$$ is given by:

$$
x_j^{(l)} = \phi ( \sum_i w_{i,j}^{(l)} x_i^{(l-1)}+b_j^{(l)}). 
$$

where $$b$$ is the bias term, trained like any other weight. Note that, to compute the output, we first compute the weighted sum of the inputs and then apply an **activation function** $$\phi$$ to this sum. We can equally describe the function implemented by each layer in the form:

$$
x^{(l)} = f^{(l)} (x^{(l-1)}) = \phi ((W^{(l)})^T x^{(l-1)}+b^{(l)})
$$

where $$f$$ is the regression function of neurons, as previously. Some popular choices of activation functions are:
- [sigmoid function](https://www.wolframalpha.com/input/?i=1%2F(1%2Be%5E%7B-x%7D)): $$ \phi(x) = \frac{1}{1+e^{-x}} $$, positive and bounded;
- [*tanh*](https://www.wolframalpha.com/input/?i=%5Cfrac%7Be%5Ex+-+e%5E%7B-x%7D%7D%7Be%5Ex%2Be%5E%7B-x%7D%7D): $$ \phi(x) = \frac{e^x - e^{-x}}{e^x+e^{-x}} $$, balanced (positive and negative) and bounded;
- [Rectified Linear Unit – ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)): $$\phi(x) = [x]_+ = max\{0,x\}$$, always non-negative and unbounded; often used as it's computationally *cheap* and helps fight the issue of *vanishing gradients* in DNNs, as the positive output region does not have a small derivative; 
- Leaky ReLU: $$\phi(x) = max \{\alpha x,x\}$$. Fixes the *dying ReLU* problem of non-leaky rectified linear units;
  - In brief, if the dot product of the input to a ReLU with its weights is negative, the output is 0. The gradient of $$ max\{ 0,x \} $$ is $$0$$ when the output is $$0$$. If for any reason the output is consistently $$0$$ (for example, if the ReLU has a large negative bias), then the gradient will be consistently 0. The error signal backpropagated from later layers gets multiplied by this 0, so no error signal ever passes to earlier layers. The ReLU has died. With leaky ReLUs, the gradient is never 0, and this problem is avoided.  

## Back-propagation

Similarly to previous regression use cases, we need to minimize the loss of the system, using e.g. Stochastic Gradient Descent. As always when dealing with gradient descent, we compute the gradient of the cost function for a particular input sample (with respect to all weights of the network and all bias terms) and then take a small step in the direction opposite to this gradient. Computing the derivative with respect to a particular parameter is really just applying the chain rule of calculus. The cost function can be written as (details omitted):

$$
\mathcal{L} = \frac{1}{N} \sum_{n=1}^N ( y_n - f^{(L+1)} \circ ... \circ f^{(2)} \circ f^{(1)} (x_n^{(0)}) ) ^2
$$

Since in general there are many parameters, it would not be efficient to do this for each parameter individually. However, the **back propagation** algorithm allows us to do it more efficiently. Our aim is to compute:

$$
\frac{\partial \mathcal{L}_n}{\partial w_{i,j}^{(l)}}, l=1, ..., L+1 \text {, and } \frac{\partial \mathcal{L}_n}{\partial b_{j}^{(l)}}, l=1, ..., L+1 
\label{eq_gradient}
$$

For simplicity, we will represent the total input computed at the $$l$$-th layer as

$$
z^{(l)} = (W^{(l)})^T x^{(l-1)}+b^{(l)} .
\label{eq_z}
$$

i.e. $$z^{(l)}$$ is the total input computed at layer $$l$$ before applying the activation function.
These quantities are easy to compute by a forward pass in the network, starting at $$ x^{(0)} = x_n $$ and then applying this recursion for $$ l=1,...,L+1 $$.

Now, let

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

We used the chain rule in the second equality. Going back to equation \ref{eq_gradient}:

$$
\frac{\partial \mathcal{L}_n}{\partial w_{i,j}^{(l)}} = \sum_k \frac{\partial \mathcal{L}_n}{\partial z_k^{(l)}} \frac{\partial z_k^{(l)}}{\partial w_{i,j}^{(l)}} = \delta_j^{(l)} x_i^{(l-1)}
\label{eq_w}
$$

Here we used the chain rule, the definition of $$\delta$$ in equation \ref{eq_delta}, and the partial derivative of equation \ref{eq_z} with respect to $$w$$. In a similar manner:

$$
\frac{\partial \mathcal{L}_n}{\partial b_{j}^{(l)}} = \sum_k \frac{\partial \mathcal{L}_n}{\partial z_k^{(l)}} \frac{\partial z_k^{(l)}}{\partial b_{j}^{(l)}} = \delta_j^{(l)} \cdot 1 = \delta_j^{(l)}
\label{eq_b}
$$

The complete back propagation algorithm is then summarized as:
1. **Forward pass**: set $$ x^{(0)} = x_n $$. Compute $$ z^{(l)} $$ for $$ l=1,...,L+1 $$;
2. **Backward pass**: set $$ \delta ^{(L+1)} $$ using the appropriate loss and activation function. Compute $$ \delta ^{(l)} $$ for $$ l = L, ..., 1 $$;
3. **Final computation**: for all parameters, compute $$ \frac{\partial \mathcal{L}_n}{\partial w_{i,j}^{(l)}} $$ (eq. \ref{eq_w}) and $$ \frac{\partial \mathcal{L}_n}{\partial b_{j}^{(l)}} $$ (eq. \ref{eq_b}).

## Automatic Differentiation

The chain rule shows us the theoretical framework, but not how to compute it efficiently. In practice, contrary to the back-propagation algorithm above, <a href="https://en.wikipedia.org/wiki/Automatic_differentiation">automatic differentiation ("autodiff")</a> algorithms such as <a href="https://pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html">PyTorch's autograd</a> do not require the derivatives of the loss and activation functions to be specified explicitly.

Instead, a computational graph (a DAG) is created dynamically, where nodes are added to the graph as the modules' `forward` methods are called. The backward propagation is then computed by applying the chain rule to the derivatives (`grad_fn`) of the nodes in the computational graph. Recursive calls will add new nodes to the graph for every call, and conditional calls may add different nodes at each `forward` pass; therefore, the computational graph cannot be extracted from the model definition directly (see an example [here](https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html)). Finally, when `tensor.backward()` is called, it recursively computes the gradients up to the leaf nodes of the graph. When this process ends, the computational graph is discarded, and a new one may be created via a new `forward` call (unless `retain_graph=True` is set in the `backward` function). For more details, refer to <a href="https://towardsdatascience.com/automatic-differentiation-explained-b4ba8e60c2ad">Chi-Feng Wang's post</a> for an illustrative example of the automatic differentiation graph, or the <a href="{{ site.assets }}/resources/princeton_course_autodiff.pdf">Princeton course COS 324 on automatic differentiation</a> for a thorough explanation of the whole process.

Storing a computational graph is also useful for activation checkpointing, where one needs to recompute a module's output during the backward pass. A good example of activation checkpointing based on the reentrant and non-reentrant PyTorch implementations can be found in the post [How Activation Checkpointing enables scaling up training deep learning models](https://medium.com/pytorch/how-activation-checkpointing-enables-scaling-up-training-deep-learning-models-7a93ae01ff2d).

It also follows that inference requires less memory than training, as training requires the computational graph to save all activations. For completeness, very good step-by-step overviews of the back propagation theory applied to a real example can be found in [Hooks for autograd saved tensors](https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html) and [PyTorch Autograd: Backpropagation](https://colab.research.google.com/drive/1aWNdmYt7RcHMbUk-Xz2Cv5-cGFSWPXe0#scrollTo=LfW6wYg42SEW).

## Dropout

[Dropout](https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5) is a method that drops neurons (in different layers) with a given probability $$p$$ at training time. For each training minibatch, a new network is sampled. 
The outputs of the kept neurons are scaled by $$1/(1-p)$$. At inference time, all neurons are used, and no dropout is applied. A simple code snippet to explain this is:

```python
class MyDropout(nn.Module):
    def forward(self, X):
        if self.training:
            binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
            return X * binomial.sample(X.size()) * (1.0/(1-self.p))
        return X
```
Dropout helps reduce overfitting, as the network learns to never rely on any given activation, so it learns *redundant* ways of solving the task with multiple neurons. It also leads to sparse activations, similar to a regularization (L2).
Dropout can be improved by adding max-norm regularization, a decaying learning rate and high momentum.
Dropping 20% of input units and 50% of hidden units was often found to be optimal in the original publication. It's computationally less expensive than regular model averaging of multiple trained DNNs. However, it takes 2-3 times longer to train than a single fully-connected DNN, because it requires way more epochs, as parameter updates are very noisy.
Because a fully connected layer holds most of the parameters, it is prone to overfitting. Therefore, dropout increases model generalization.

{: style="text-align:center; font-size: small;"}
<img width="35%" height="35%" src="{{ site.assets }}/deep-neural-networks/dropout.png">

## Convolutional Neural Networks
 
The disadvantage of large neural networks is that they have a very high number of parameters, so they may require lots of data to be trained. In some scenarios, local training should suffice: e.g. in an audio stream, it is natural to process an input stream $$ x^{(0)}[n] $$ by running it through a linear time-invariant filter, whose output $$ x^{(1)}[n] $$ is given by the convolution of the input $$ x^{(0)}[n] $$ and the filter $$ f [n] $$, as:

$$
x^{(1)}[n] = \sum_k f[k]x^{(0)} [n-k]
$$


The filter $$f$$ is local, i.e. $$ f[k]=0 $$ for $$ \vert k \vert \ge K $$. By choosing an appropriate type of filter, we can bring out various aspects of the underlying signal, e.g. we can smooth features by averaging, or we can enhance differences between neighboring elements by using a *high-pass* filter. An analogous scenario applies to a picture:

$$
x^{(1)}[n,m] = \sum_{k,l} f[k,l]x^{(0)} [n-k,m-l]
$$

We see that the output $$ x^{(1)} $$ at position $$ [n, m] $$ only depends on the values of the input $$ x^{(0)} $$ at positions close to $$ [n, m] $$. So we no longer need a fully connected network, but only one that represents the local structure, leading to fewer parameters to deal with. This structure implies that we should use the same filter (i.e., not only the same connection pattern but also the same weights) at every position! This is called **weight sharing**, and it reduces the number of parameters even further. The difference between a locally/sparsely connected and a fully connected network is obvious:

{: style="text-align:center; font-size: small;"}
<img width="35%" height="35%" src="{{ site.assets }}/deep-neural-networks/fully_sparse_connectivity.png">

It is common to not only compute the output of a single filter but to use multiple filters. The various outputs are called channels. This introduces some additional parameters into the model.
If we add several channels, we do not end up with a 2D output in the next level, but in fact with a 3D output. Layer after layer, we have increasingly more channels but a smaller “footprint.” In brief, applying local connectivity to a 2D input dataset on a *deep* neural network, we get the following final structure:

{: style="text-align:center; font-size: small;"}
<img width="65%" height="65%" src="{{ site.assets }}/deep-neural-networks/cnn.png">

Notice the three types of operations involved:
- convolutional layer: applies a convolution operation to the input, passing the result to the next layer. The convolution emulates the response of an individual neuron to visual stimuli;
  - convolution is a mathematical operation on two functions that produces a third function, which expresses how the shape of one is modified by the other. Convolution is similar to cross-correlation;
- pooling: combines the outputs of neuron clusters at one layer into a single neuron in the next layer.
  - A very common operation is max pooling, which uses the maximum value of each cluster of neurons at the prior layer;
- fully connected: similar to a multi-layer perceptron, providing the *universal approximator* function;

The size of each picture typically gets smaller and smaller as we proceed through the layers, either due to the handling of the boundary or because we might perform subsampling.

Training is done with backpropagation, ignoring that some weights are shared and considering each weight on each edge to be an independent variable. Once the gradient has been computed for this network with independent weights, just sum up the gradients of all edges that share the same weight. This gives us the gradient for the network with weight sharing. 

## Skip connections

An issue with very deep networks is that the performance of the model drops as the depth of the architecture increases. This is known as the *degradation problem*. It is not caused by overfitting, since the training error increases as well. Possible reasons are vanishing and/or exploding gradients. One way to improve this is to use [normalization](https://arxiv.org/abs/1803.08494) techniques to ensure that gradients have healthy norms.

A better way to improve this is to provide layers of the network with the outputs of layers that are not directly connected to them. These connections *skip* some of the layers in the neural network and feed the output of one layer as the input of later layers --- hence the name *skip connections*. In practice, the input to a given layer will be the combination (typically via addition or concatenation) of the outputs of several previous layers. 

{: style="text-align:center; font-size: small;"}
<img width="65%" height="65%" src="{{ site.assets }}/deep-neural-networks/densenet.png">

{: style="text-align:center; font-size: small;"}
An example of a CNN with skip connections (DenseNet). source: <a href="https://arxiv.org/abs/1608.06993">Densely Connected Convolutional Networks</a> 

There is currently a plethora of different implementations of CNNs with several layouts of skip connections. If you are curious to know more, have a look at [ResNet](https://arxiv.org/abs/1512.03385), [DenseNet](https://arxiv.org/abs/1608.06993) and [U-net](https://arxiv.org/abs/1505.04597).


## Embeddings

Embeddings in machine learning are dense, low-dimensional vector representations of data (e.g., words, images, or users) that capture their semantic or contextual meaning in a continuous vector space. Embeddings can be collected by passing input data (e.g., text, images) through a *trained* machine learning model (e.g., a neural network) and extracting the values from a specific intermediate layer, typically before the final output layer.

A list of the most common data types and embedding types:
- words: one-hot vectors or [Word2vec](https://en.wikipedia.org/wiki/Word2vec) (*skipgram* or *continuous bag of words*)
  - skipgram: slower to train ("3 days"), better at capturing semantic relationships, e.g. for 'cat' return 'dog' as a word with close embeddings;
  - CBOW: faster to train ("few hours"), better at capturing syntactic relationships between words, e.g. for 'cat' return 'cats';
- text (word sequences): [BERT]({{ site.baseurl }}{% post_url 2020-02-28-learning-from-sequences %})
- non-textual sequences: [Encoder-Decoders, e.g. LSTMs, RNNs]({{ site.baseurl }}{% post_url 2020-02-28-learning-from-sequences %})
- point cluster or array: [Principal Component Analysis]({{ site.baseurl }}{% post_url 2017-11-01-unsupervised-learning %})
- images:
  - in a classification task: use the activation of the last layer *before* the layer that does logit/softmax, i.e. the input to the final layer, which is the output of the one before last;
  - in an image-to-image task, e.g. segmentation using a [U-net](https://arxiv.org/abs/1505.04597): use the activation of the last downsampling layer, which is the input of the first upsampling layer, i.e. the [*information bottleneck*](https://en.wikipedia.org/wiki/Information_bottleneck_method);
- videos: the input is now 5D (batch size, time, channels, height, width), i.e. a sequence of video frames stacked on the time dimension. Embeddings are collected similarly to a regular CNN (adapted to use 3D instead of 2D convolutions, etc.);
- graphs: the embedding of a graph is given by the embedding of a node after several steps of the [message passing algorithm](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html) as $$x_i^{(k)} = \gamma^{(k)} \left( x_i^{(k-1)}, \Box_{j \in \mathcal{N}(i)} \, \phi^{(k)} \left( x_i^{(k-1)}, x_j^{(k-1)}, e_{j,i} \right) \right)$$, where:
  - $$x_i^{(k)}$$ is the embedding of node $$i$$ at message passing step $$k$$;
  - $$\mathcal{N}(i)$$ is the set of neighbors of node $$i$$;
  - $$e_{j,i}$$ is the embedding of the edge from node $$j$$ to node $$i$$;
  - $$\Box$$ is a differentiable, permutation-invariant function, e.g. sum, mean or max;
  - and $$\gamma$$ and $$\phi$$ are differentiable functions such as DNNs or RNNs (LSTMs, GRUs).
 

