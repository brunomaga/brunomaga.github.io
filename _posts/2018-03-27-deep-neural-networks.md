---
layout: post
title:  "Deep Neural Networks, Backpropagation, Dropout, CNNs and Embeddings"
categories: ["Machine Learning"]
tags: ["deep learning", "backpropagation", "autodiff", "activation functions", "dropout", "CNN", "skip connections", "embeddings"]
description: "How deep neural networks compute and learn: activation functions and XOR, backpropagation derived and run on numbers, automatic differentiation, dropout, convolutional neural networks (CNNs), skip connections and embeddings."
---

In 1958, Frank Rosenblatt described the [perceptron](https://en.wikipedia.org/wiki/Perceptron), an artificial neuron that learns to separate two classes with a linear boundary. In 1969, the book [Perceptrons](https://en.wikipedia.org/wiki/Perceptrons_(book)) by Marvin Minsky and Seymour Papert showed that such a neuron cannot learn a function as simple as the [exclusive or](https://en.wikipedia.org/wiki/Exclusive_or) (XOR) of two binary inputs. No line separates the inputs $$(0, 1)$$ and $$(1, 0)$$, of output 1, from $$(0, 0)$$ and $$(1, 1)$$, of output 0. Networks with several layers of such neurons can learn XOR and much more complex functions. In 1986, the paper [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0), by David Rumelhart, Geoffrey Hinton and Ronald Williams, showed how to train them with gradient descent, by propagating the errors backwards from the output layer to the first one.

Deep Neural Networks (also known as [Multi-Layer Perceptrons](https://en.wikipedia.org/wiki/Multilayer_perceptron)) are **[universal approximators](https://en.wikipedia.org/wiki/Universal_approximation_theorem)**, i.e. they can approximate any continuous function on a closed and bounded domain, even with a single hidden layer, given enough neurons. They are the most important and relevant ML tool in the field of supervised learning, as they are part of the models used for most supervised tasks. This post covers their structure, the backpropagation algorithm that trains them and its implementation with automatic differentiation, and then dropout, convolutional networks, skip connections and embeddings.

## Feedforward Networks and Activation Functions

The structure of a basic NN is the following: one input layer of size $$D$$, $$L$$ hidden layers of size $$K$$, and one output layer. It is a **[feedforward network](https://en.wikipedia.org/wiki/Feedforward_neural_network)**: the computation performed by the network starts with the input from the left and flows to the right, with no feedback loop. It can be used for regression (when the output is a real value) and for classification (when the output is the class). The following picture illustrates a network with one hidden layer, where each node is connected to every node of the next layer, i.e. a **fully connected** network:

{: .picture}
<img width="41.1%" height="41.1%" src="{{ site.assets }}/deep-neural-networks/dnn.svg">

{: .caption}
A fully connected feedforward network with an input layer of size $$D = 4$$, one hidden layer of size $$K = 5$$ and one output $$\hat{y}$$. Adapted from: [Machine Learning course CS-433 lecture notes, M. Jaggi, EPFL](https://github.com/epfml/ML_course)

Input and output layers work similarly to any previous regression method. Hidden layers are different. The output at node $$j$$ in hidden layer $$l$$ is given by:

$$
x_j^{(l)} = \phi \left( \sum_i w_{i,j}^{(l)} x_i^{(l-1)} + b_j^{(l)} \right)
$$

where $$w_{i,j}^{(l)}$$ is the weight of the connection from node $$i$$ of layer $$l-1$$ to node $$j$$ of layer $$l$$, and $$b_j^{(l)}$$ is the bias term, trained like any other weight. Note that, to compute the output, each neuron first computes the weighted sum of its inputs and then applies an **[activation function](https://en.wikipedia.org/wiki/Activation_function)** $$\phi$$ to this sum. The activation function must be non-linear, otherwise the whole network would be equivalent to a single linear layer. The function implemented by each layer can equally be described in matrix form:

$$
x^{(l)} = f^{(l)} (x^{(l-1)}) = \phi ((W^{(l)})^T x^{(l-1)}+b^{(l)})
$$

where $$W^{(l)}$$ is the matrix of the weights $$w_{i,j}^{(l)}$$ of layer $$l$$, $$b^{(l)}$$ is the vector of its biases, $$\phi$$ is applied to each entry, and $$f^{(l)}$$ is the function implemented by the layer. Some popular choices of activation functions are:
- [sigmoid function](https://www.wolframalpha.com/input/?i=1%2F(1%2Be%5E%7B-x%7D)): $$ \phi(x) = \frac{1}{1+e^{-x}} $$, positive and bounded;
- [*tanh*](https://www.wolframalpha.com/input/?i=%5Cfrac%7Be%5Ex+-+e%5E%7B-x%7D%7D%7Be%5Ex%2Be%5E%7B-x%7D%7D): $$ \phi(x) = \frac{e^x - e^{-x}}{e^x+e^{-x}} $$, balanced (positive and negative) and bounded;
- [Rectified Linear Unit – ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)): $$\phi(x) = [x]_+ = \max\{0,x\}$$, always non-negative and unbounded; often used as it is computationally *cheap* and helps fight the issue of [*vanishing gradients*](https://en.wikipedia.org/wiki/Vanishing_gradient_problem) in DNNs, as the positive output region does not have a small derivative;
- Leaky ReLU: $$\phi(x) = \max \{\alpha x,x\}$$, with $$0 < \alpha < 1$$. Fixes the *dying ReLU* problem of non-leaky rectified linear units;
  - In brief, if the dot product of the input to a ReLU with its weights is negative, the output is 0. The gradient of $$ \max\{ 0,x \} $$ is $$0$$ when the output is $$0$$. If for any reason the output is consistently $$0$$ (for example, if the ReLU has a large negative bias), then the gradient will be consistently 0. The error signal backpropagated from later layers gets multiplied by this 0, so no error signal ever passes to earlier layers. The ReLU has died. With leaky ReLUs, the gradient is never 0, and this problem is avoided.

In practice, ReLU and its variants are the default choice for hidden layers, while the sigmoid and *tanh* are used where a bounded output is needed, e.g. a probability at the output of a binary classifier.

As an example, a network with two inputs, one hidden layer of two ReLU neurons and one output with no activation function computes the XOR that a single perceptron cannot learn. With the weights given in the [Deep Learning](https://www.deeplearningbook.org/) book (section 6.1), the hidden layer computes $$x_1^{(1)} = \max\{0, x_1^{(0)} + x_2^{(0)}\}$$ and $$x_2^{(1)} = \max\{0, x_1^{(0)} + x_2^{(0)} - 1\}$$, and the output is $$\hat{y} = x_1^{(1)} - 2 x_2^{(1)}$$. The four inputs $$(0, 0)$$, $$(0, 1)$$, $$(1, 0)$$ and $$(1, 1)$$ give the hidden outputs $$(0, 0)$$, $$(1, 0)$$, $$(1, 0)$$ and $$(2, 1)$$, and the outputs 0, 1, 1 and 0. The hidden layer maps the two inputs of output 1 to the same point, $$(1, 0)$$, where a linear output layer can separate them from the other two.

Without the ReLU, the same weights compute the linear function $$\hat{y} = 2 - x_1^{(0)} - x_2^{(0)}$$, whose outputs 2, 1, 1 and 0 are not XOR: the non-linearity is what lets the hidden layer reshape its inputs. Here, the 6 weights and 3 biases of the network were set by hand. A network of millions of parameters learns them from data with gradient descent, which needs the derivative of the loss with respect to every parameter: this is the job of backpropagation.

## Backpropagation

Similarly to previous regression use cases, the loss of the system needs to be minimized, using e.g. [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent). As always with gradient descent, the algorithm computes the gradient of the cost function for a particular input sample (with respect to all weights and bias terms), and takes a small step in the direction opposite to this gradient. The size of the step is set by the *learning rate* $$\eta$$. Computing the derivative with respect to a particular parameter is an application of the [chain rule](https://en.wikipedia.org/wiki/Chain_rule) of calculus. The cost function for the squared loss over $$N$$ samples is written as:

$$
\mathcal{L} = \frac{1}{N} \sum_{n=1}^N ( y_n - f^{(L+1)} \circ ... \circ f^{(2)} \circ f^{(1)} (x_n^{(0)}) ) ^2
$$

Since in general there are many parameters, it would not be efficient to do this for each parameter individually. As an example, estimating the gradient with [finite differences](https://en.wikipedia.org/wiki/Finite_difference) takes one extra forward pass per parameter, i.e. a million forward passes for a network of a million parameters. However, the **[backpropagation](https://en.wikipedia.org/wiki/Backpropagation)** algorithm does it more efficiently, as it computes the derivatives of all parameters with one forward and one backward pass through the network. The aim is to compute, for the loss $$\mathcal{L}_n$$ of a sample $$n$$:

$$
\frac{\partial \mathcal{L}_n}{\partial w_{i,j}^{(l)}}, l=1, ..., L+1 \text {, and } \frac{\partial \mathcal{L}_n}{\partial b_{j}^{(l)}}, l=1, ..., L+1 
\label{eq_gradient}
$$

For simplicity, the total input computed at the $$l$$-th layer, before the activation function, is represented as:

$$
z^{(l)} = (W^{(l)})^T x^{(l-1)}+b^{(l)} .
\label{eq_z}
$$

These quantities are easy to compute by a forward pass in the network, starting at $$ x^{(0)} = x_n $$ and then applying this recursion for $$ l=1,...,L+1 $$.

Now, consider the derivative of the loss with respect to the total input of node $$j$$ of layer $$l$$, often called the *error* of the node:

$$
\delta_j^{(l)} = \frac{\partial \mathcal{L}_n}{\partial z_j^{(l)}}.
\label{eq_delta}
$$

These quantities are easier to compute with a backward pass, from the output layer to the first one, as the errors of each layer follow from those of the next layer:

$$
\begin{align*}
\delta_j^{(l)} = & \frac{\partial \mathcal{L}_n}{\partial z_j^{(l)}} \\
= & \sum_k \frac{\partial \mathcal{L}_n}{\partial z_k^{(l+1)}} \frac{\partial z_k^{(l+1)}}{\partial z_j^{(l)}} \\ 
= & \sum_k \delta_k^{(l+1)} \frac{\partial \left[ (W^{(l+1)})^T x^{(l)}+b^{(l+1)} \right]_k}{\partial z_j^{(l)}} \\
= & \sum_k \delta_k^{(l+1)}  W_{j,k}^{(l+1)} \frac{\partial x_j^{(l)}}{\partial z_j^{(l)}} \\
= & \sum_k \delta_k^{(l+1)} W_{j,k}^{(l+1)} \phi '(z_j^{(l)}) 
\end{align*}
$$

The chain rule is used in the second equality, and, in the fourth, the fact that $$x_j^{(l)}$$ is the only entry of $$x^{(l)}$$ that depends on $$z_j^{(l)}$$. The recursion starts at the output layer, where $$\delta^{(L+1)}$$ follows from the derivative of the loss. For the squared loss of a sample, $$\mathcal{L}_n = (y_n - x^{(L+1)})^2$$, and an output layer with no activation function, i.e. $$x^{(L+1)} = z^{(L+1)}$$, it is given by $$\delta^{(L+1)} = -2 (y_n - x^{(L+1)})$$.

Going back to equation \ref{eq_gradient}:

$$
\frac{\partial \mathcal{L}_n}{\partial w_{i,j}^{(l)}} = \sum_k \frac{\partial \mathcal{L}_n}{\partial z_k^{(l)}} \frac{\partial z_k^{(l)}}{\partial w_{i,j}^{(l)}} = \delta_j^{(l)} x_i^{(l-1)}
\label{eq_w}
$$

Here, the derivation uses the chain rule, the definition of $$\delta$$ in equation \ref{eq_delta}, and the partial derivative of equation \ref{eq_z} with respect to $$w$$, which is zero for $$k \neq j$$. In a similar manner:

$$
\frac{\partial \mathcal{L}_n}{\partial b_{j}^{(l)}} = \sum_k \frac{\partial \mathcal{L}_n}{\partial z_k^{(l)}} \frac{\partial z_k^{(l)}}{\partial b_{j}^{(l)}} = \delta_j^{(l)} \cdot 1 = \delta_j^{(l)}
\label{eq_b}
$$

The complete backpropagation algorithm is then summarized as:
1. **Forward pass**: set $$ x^{(0)} = x_n $$. Compute $$ z^{(l)} $$ for $$ l=1,...,L+1 $$;
2. **Backward pass**: set $$ \delta ^{(L+1)} $$ using the appropriate loss and activation function. Compute $$ \delta ^{(l)} $$ for $$ l = L, ..., 1 $$;
3. **Final computation**: for all parameters, compute $$ \frac{\partial \mathcal{L}_n}{\partial w_{i,j}^{(l)}} $$ (eq. \ref{eq_w}) and $$ \frac{\partial \mathcal{L}_n}{\partial b_{j}^{(l)}} $$ (eq. \ref{eq_b}).

As an example, consider a network with one neuron per layer, so that the indices $$i$$ and $$j$$ are dropped. Its input is $$x^{(0)} = x_n = 2$$, its hidden neuron has the weight $$w^{(1)} = 1$$ and a ReLU activation, and its output has the weight $$w^{(2)} = 0.5$$ and no activation function. The biases are zero, and the target is $$y_n = 3$$. The three steps of the algorithm give:
1. **Forward pass**: $$z^{(1)} = 1 \times 2 = 2$$, $$x^{(1)} = \max\{0, 2\} = 2$$ and $$x^{(2)} = z^{(2)} = 0.5 \times 2 = 1$$, so the loss is $$(3 - 1)^2 = 4$$;
2. **Backward pass**: $$\delta^{(2)} = -2(3 - 1) = -4$$, and $$\delta^{(1)} = \delta^{(2)} w^{(2)} \phi'(z^{(1)}) = -4 \times 0.5 \times 1 = -2$$, as the derivative of the ReLU is 1 for a positive input;
3. **Final computation**: $$\partial \mathcal{L}_n / \partial w^{(2)} = \delta^{(2)} x^{(1)} = -8$$ and $$\partial \mathcal{L}_n / \partial w^{(1)} = \delta^{(1)} x^{(0)} = -4$$, and the derivatives of the biases are the errors, $$-4$$ and $$-2$$.

Both derivatives are negative, so gradient descent increases both weights, which moves the output toward the target. One step $$w \leftarrow w - \eta \, \partial \mathcal{L}_n / \partial w$$ on the two weights, with a learning rate $$\eta = 0.05$$ and the biases kept at zero, gives $$w^{(1)} = 1.2$$ and $$w^{(2)} = 0.9$$, the output 2.16 and the loss 0.71. Note that the error $$\delta^{(1)}$$ reuses $$\delta^{(2)}$$: in a deep network, the errors of each layer are computed once, and shared by the derivatives of all the layers before it.

In practice, the backward pass computes the errors of all the nodes of a layer at once, in vector form:

$$
\begin{align*}
\delta^{(l)} & = \left( W^{(l+1)} \delta^{(l+1)} \right) \odot \phi'(z^{(l)}) \\
\frac{\partial \mathcal{L}_n}{\partial W^{(l)}} & = x^{(l-1)} \left( \delta^{(l)} \right)^T \\
\frac{\partial \mathcal{L}_n}{\partial b^{(l)}} & = \delta^{(l)}
\end{align*}
$$

where $$\odot$$ is the element-wise ([Hadamard](https://en.wikipedia.org/wiki/Hadamard_product_(matrices))) product. For a mini-batch, the vectors $$x$$ and $$\delta$$ become matrices with one column per sample, and the product $$x^{(l-1)} (\delta^{(l)})^T$$ then adds up the derivatives of all samples. Each layer then runs one matrix product in the forward pass, $$(W^{(l)})^T x^{(l-1)}$$, and two of the same size in the backward pass, one for the errors and one for the derivatives of the weights. Therefore, the backward pass takes about twice the floating-point operations of the forward pass, and a training step about three times.

### Automatic Differentiation

The chain rule shows the theoretical framework, but not how to compute it in software. Backpropagation is a special case of <a href="https://en.wikipedia.org/wiki/Automatic_differentiation">automatic differentiation ("autodiff")</a>, in its *reverse mode*. The reverse mode computes the derivatives of one output (the loss) with respect to all inputs (the parameters) in one backward pass, while the *forward mode* needs one pass per input. In practice, autodiff libraries such as <a href="https://pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html">PyTorch's autograd</a> apply the chain rule to the derivatives of the elementary operations of the forward pass. Thus, contrary to the derivation above, the derivatives of the loss and activation functions do not have to be specified explicitly.

Instead, a computational graph (a [DAG](https://en.wikipedia.org/wiki/Directed_acyclic_graph)) is created dynamically, where nodes are added to the graph as the modules' `forward` methods are called. The backward propagation is then computed by applying the chain rule to the derivatives (`grad_fn`) of the nodes in the computational graph. Recursive calls add new nodes to the graph for every call, and conditional calls may add different nodes at each `forward` pass; therefore, the computational graph cannot be extracted from the model definition directly (see an example [here](https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html)).

Finally, when `tensor.backward()` is called, it recursively computes the gradients up to the leaf nodes of the graph. When this process ends, the computational graph is discarded, and a new one may be created via a new `forward` call (unless `retain_graph=True` is set in the `backward` function). For more details, <a href="https://towardsdatascience.com/automatic-differentiation-explained-b4ba8e60c2ad">Chi-Feng Wang's post</a> gives an illustrative example of the automatic differentiation graph, and the <a href="{{ site.assets }}/resources/princeton_course_autodiff.pdf">Princeton course COS 324 on automatic differentiation</a> a thorough explanation of the whole process.

Storing a computational graph is also useful for [activation checkpointing](https://arxiv.org/abs/1604.06174), where one needs to recompute a module's output during the backward pass, as it is discarded during the forward pass to save memory. A good example of activation checkpointing based on the reentrant and non-reentrant PyTorch implementations can be found in the post [How Activation Checkpointing enables scaling up training deep learning models](https://medium.com/pytorch/how-activation-checkpointing-enables-scaling-up-training-deep-learning-models-7a93ae01ff2d).

It also follows that inference requires less memory than training. The derivative of the weights of a layer, $$x^{(l-1)} (\delta^{(l)})^T$$, needs the input of the layer. Therefore, training saves the activations of all layers in the computational graph until the backward pass, while inference can discard them as soon as the next layer is computed.

For completeness, very good step-by-step overviews of the backpropagation theory applied to a real example can be found in [Hooks for autograd saved tensors](https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html) and [PyTorch Autograd: Backpropagation](https://colab.research.google.com/drive/1aWNdmYt7RcHMbUk-Xz2Cv5-cGFSWPXe0#scrollTo=LfW6wYg42SEW).

## Dropout

A network with many parameters is prone to [overfitting](https://en.wikipedia.org/wiki/Overfitting): it learns the noise of the training data, and generalizes poorly to new data. **Dropout**, described in the paper [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html), is a method that drops neurons (in different layers) with a given probability $$p$$ at training time. For each training sample in a mini-batch, a new *thinned* network is sampled, made of the neurons that are kept.

The outputs of the kept neurons are scaled by $$1/(1-p)$$, so that their expected value is the same at training and inference time. An output $$x_j^{(l)}$$ is kept with probability $$1-p$$, so its expected value is $$(1-p) \times x_j^{(l)}/(1-p) = x_j^{(l)}$$. At inference time, all neurons are used, and no dropout is applied. This scaling at training time is known as *inverted dropout*; note that the paper scales the weights at inference time instead, and writes $$p$$ for the probability of keeping a unit. A simple code snippet to explain this is:

```python
class MyDropout(nn.Module):
    def forward(self, X):
        if self.training:
            binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
            return X * binomial.sample(X.size()) * (1.0/(1-self.p))
        return X
```

Dropout helps reduce overfitting, as the network learns to never rely on any given activation, so it learns *redundant* ways of solving the task with multiple neurons. As a side effect, it leads to sparse activations, similar to an L1 regularization of the activations (section 7.2 of the paper). For linear regression, it is equivalent in expectation to an L2 regularization of the weights, i.e. [ridge regression](https://en.wikipedia.org/wiki/Ridge_regression) (section 9.1).

In practice, dropout works best with max-norm regularization (an upper bound on the norm of the incoming weights of each neuron), a large decaying learning rate and a high momentum (0.95 to 0.99 instead of the usual 0.9). Dropping 20% of the input units and 50% of the hidden units was often found to be close to optimal in the [original publication](https://jmlr.org/papers/v15/srivastava14a.html). However, a dropout network takes 2 to 3 times longer to train than a standard network of the same architecture, because it requires many more epochs, as parameter updates are very noisy.

Because the fully connected layers hold most of the parameters of a network such as the CNNs below, they are the most prone to overfitting, and dropout is often applied to them. The following picture illustrates a standard network (left) and one of the thinned networks that dropout samples from it (right):

{: .picture}
<img width="68.5%" height="68.5%" src="{{ site.assets }}/deep-neural-networks/dropout.svg">

{: .caption}
Dropout. Left: a standard network with two hidden layers ($$p = 0$$). Right: a thinned network produced by dropout with $$p = 0.5$$, where $$p$$ is the probability of dropping a unit: the crossed units are dropped, with their connections. Adapted from: [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html)

The network on the left has 15 units that dropout can drop, its 5 inputs and 10 hidden units. Each training sample thus uses one of $$2^{15} = 32{,}768$$ thinned networks, which all share the same weights. At inference time, the full network with the scaled outputs approximates the average of the predictions of all these thinned networks, which is computationally much less expensive than the regular model averaging of multiple trained DNNs.

## Convolutional Neural Networks
 
The disadvantage of large neural networks is that they have a very high number of parameters, so they may require lots of data to be trained. In some scenarios, local connectivity should suffice. For example, in an audio stream, it is natural to process an input stream $$ x^{(0)}[n] $$ by running it through a [linear time-invariant](https://en.wikipedia.org/wiki/Linear_time-invariant_system) filter, whose output $$ x^{(1)}[n] $$ is given by the [convolution](https://en.wikipedia.org/wiki/Convolution) of the input $$ x^{(0)}[n] $$ and the filter $$ h[n] $$, as:

$$
x^{(1)}[n] = \sum_u h[u]x^{(0)} [n-u]
$$

The filter $$h$$ is local, i.e. $$ h[u]=0 $$ for $$ \vert u \vert \ge r $$. By choosing an appropriate type of filter, one can bring out various aspects of the underlying signal, e.g. smooth features by averaging, or enhance differences between neighboring elements by using a *high-pass* filter. As an example, the filter $$h[-1] = h[0] = h[1] = 1/3$$ averages three neighbors, and the filter $$h[0] = 1$$, $$h[1] = -1$$ computes the difference $$x^{(0)}[n] - x^{(0)}[n-1]$$. An analogous scenario applies to a picture:

$$
x^{(1)}[n,m] = \sum_{u,v} h[u,v]x^{(0)} [n-u,m-v]
$$

The output $$ x^{(1)} $$ at position $$ [n, m] $$ only depends on the values of the input $$ x^{(0)} $$ at positions close to $$ [n, m] $$. So a fully connected network is no longer needed, but only one that represents the local structure, leading to fewer parameters to deal with.

This structure implies that the same filter (i.e., not only the same connection pattern but also the same weights) should be used at every position! This is called **weight sharing**, and it reduces the number of parameters even further, as a pattern such as an edge is detected by the same filter wherever it appears in the input. The following picture shows the difference between a locally (sparsely) connected and a fully connected layer:

{: .picture}
<img width="30%" height="30%" src="{{ site.assets }}/deep-neural-networks/fully_sparse_connectivity.svg">

{: .caption}
Sparse and full connectivity, viewed from below. Top: a locally connected layer with a kernel of width 3, where the input $$x_3$$ affects only the outputs $$s_2$$, $$s_3$$ and $$s_4$$. Bottom: a fully connected layer, where $$x_3$$ affects every output. Adapted from: [Deep Learning, I. Goodfellow, Y. Bengio and A. Courville, Figure 9.2](https://www.deeplearningbook.org/contents/convnets.html)

In the picture, the inputs $$x_i$$ and the outputs $$s_i$$ stand for the $$x^{(0)}$$ and $$x^{(1)}$$ of the text. The fully connected layer has $$5 \times 5 = 25$$ weights, one per connection. The locally connected layer has 13 connections, as each output reads at most 3 inputs, and, with weight sharing, only the 3 weights of its filter.

It is common to not only compute the output of a single filter but to use multiple filters. The various outputs are called **channels** (or *feature maps*). This introduces some additional parameters into the model, as each filter has its own weights, and spans all the channels of its input.

With several channels, the result is not a 2D output in the next level, but in fact a 3D output. Layer after layer, there are increasingly more channels but a smaller “footprint”. In brief, applying local connectivity to a 2D input dataset on a *deep* neural network leads to the following final structure, a **[convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network)** (CNN):

{: .picture}
<img width="92%" height="92%" src="{{ site.assets }}/deep-neural-networks/cnn.svg">

{: .caption}
A CNN that classifies $$16 \times 16$$ RGB patches of a satellite image as road or non-road: two stages of convolution and pooling, two fully connected layers, and the output probabilities of the two classes. Each label gives the shape of the data after a layer for $$N$$ input images, with the number of channels or units in brackets.

Notice the three types of operations involved:
- convolutional layer: applies a convolution operation to the input, passing the result to the next layer. The convolution emulates the response of an individual neuron to visual stimuli;
  - convolution is a mathematical operation on two functions that produces a third function, which expresses how the shape of one is modified by the other. In practice, the convolutional layers of deep learning libraries compute a [cross-correlation](https://en.wikipedia.org/wiki/Cross-correlation), $$x^{(1)}[n,m] = \sum_{u,v} h[u,v] \, x^{(0)}[n+u,m+v]$$, i.e. a convolution with a flipped filter (Deep Learning book, section 9.1); as the filter is learned, the two are equivalent;
- [pooling](https://en.wikipedia.org/wiki/Pooling_layer): combines the outputs of neuron clusters at one layer into a single neuron in the next layer;
  - a very common operation is max pooling, which uses the maximum value of each cluster of neurons at the prior layer, e.g. of each $$2 \times 2$$ window in the picture;
- fully connected: similar to a multi-layer perceptron, providing the *universal approximator* function.

The size of each feature map typically gets smaller and smaller as the data proceeds through the layers, either due to the handling of the boundary or because the network may perform subsampling. Along each dimension, the size of the output of a convolution or a pooling is given by:

$$
d_{\text{out}} = \left\lfloor \frac{d_{\text{in}} + 2 \times \text{padding} - \text{kernel size}}{\text{stride}} \right\rfloor + 1
$$

where $$d_{\text{in}}$$ is the size of the input and the *kernel size* is the size of the filter or of the pooling window. The *stride* is the step by which the window moves, and the *padding* is the number of zeros added on each side of the input. In the picture, the convolutions pad their input so that the output keeps the size of $$16 \times 16$$, and each max pooling of $$2 \times 2$$ with a stride of 2 halves it, to $$\lfloor (16 - 2)/2 \rfloor + 1 = 8$$ and then 4.

Weight sharing makes a convolutional layer much smaller than a fully connected one. Consider the first layer of the network above, which turns a $$16 \times 16$$ RGB patch (768 inputs) into 32 channels of $$16 \times 16$$ (8,192 outputs). A fully connected layer between them has $$768 \times 8{,}192$$ weights and 8,192 biases, about 6.3 million parameters. A convolutional layer with 32 filters of $$3 \times 3$$ pixels over the 3 color channels has $$32 \times (3 \times 3 \times 3 + 1) = 896$$, 7,000 times fewer, and the same 896 for an input of any size. The fully connected layers, instead, hold most of the parameters of a CNN. The first one connects the $$4 \times 4 \times 64 = 1{,}024$$ outputs of the last pooling layer to 512 units, with $$1{,}024 \times 512 + 512 = 524{,}800$$ parameters, more than all the convolutional layers together.

Training is done with backpropagation, ignoring that some weights are shared and considering each weight on each edge to be an independent variable. Once the gradient has been computed for this network with independent weights, the gradients of all edges that share the same weight are summed up. This gives the gradient for the network with weight sharing, as the loss depends on the shared weight through each of these edges. 

## Skip Connections

An issue with very deep networks is that the performance of the model drops as the depth of the architecture increases. This is known as the *degradation problem*. It is not caused by overfitting, since the training error increases as well.

Possible reasons are vanishing and/or exploding gradients. One way to improve this is to use [normalization](https://arxiv.org/abs/1803.08494) techniques to ensure that gradients have healthy norms. However, the authors of [ResNet](https://arxiv.org/abs/1512.03385) observed the degradation problem even with normalization layers and healthy gradient norms. On the [CIFAR-10](https://en.wikipedia.org/wiki/CIFAR-10) dataset, a plain network of 56 layers has a higher training error than one of 20 layers.

A better way to improve this is to provide layers of the network with the outputs of layers that are not directly connected to them. These connections *skip* some of the layers in the neural network and feed the output of one layer as the input of later layers, hence the name **skip connections**. In practice, the input to a given layer is the combination (typically via addition or concatenation) of the outputs of several previous layers.

With an addition, as in ResNet, a block of layers with the input $$x$$ computes $$y = x + F(x)$$, where $$F$$ is the function of the skipped layers. These layers only need to learn a correction (a *residual*) to their input, which is easier to optimize than the whole mapping.

The addition also helps the backward pass. The derivative of the output of the block with respect to its input (its [Jacobian matrix](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant)) is $$I + \partial F / \partial x$$, where $$I$$ is the identity matrix. Thus, the gradient of the loss with respect to the input of the block is given by:

$$
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} + \left( \frac{\partial F}{\partial x} \right)^T \frac{\partial \mathcal{L}}{\partial y}
$$

where the first term passes the gradient of the layers above to the layers below unchanged, even when the derivatives of the skipped layers are small. Over a stack of blocks, the paper [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) showed that the gradient at any block contains the gradient at the last block as an additive term (Eq. 5 in the paper). Therefore, it does not vanish, even when the weights are arbitrarily small.

With a concatenation, as in DenseNet, each layer receives the feature maps of all the layers before it, so the number of channels grows with every layer. If each layer produces $$k$$ feature maps (the *growth rate*), the layer $$H_l$$ receives $$k_0 + k(l-1)$$ of them, where $$k_0$$ is the number of channels of the input of the block (DenseNet paper, section 3). The following picture illustrates a dense block of DenseNet:

{: .picture}
<img width="82.1%" height="82.1%" src="{{ site.assets }}/deep-neural-networks/densenet.svg">

{: .caption}
A 5-layer dense block of DenseNet with a growth rate of $$k = 4$$: each layer $$H_i$$ (BN-ReLU-Conv) takes the feature maps $$x_0, \ldots, x_{i-1}$$ of all preceding layers as input and produces $$k$$ new feature maps $$x_i$$. The transition layer takes all of them. Adapted from: [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

In the picture, the input $$x_0$$ has $$k_0 = 6$$ feature maps and $$k = 4$$, so the layer $$H_4$$ receives $$6 + 4 \times 3 = 18$$ feature maps, and the transition layer, which takes all of them, $$6 + 4 \times 4 = 22$$. To sum up, an addition (ResNet) keeps the number of channels constant, while a concatenation (DenseNet) lets every layer read the feature maps of all the layers before it. The price of DenseNet is an input that grows with the depth of the block, which is why it uses small growth rates ($$k$$ from 12 to 40 in its experiments).

There is currently a plethora of different implementations of CNNs with several layouts of skip connections. To know more, the curious reader can have a look at [ResNet](https://arxiv.org/abs/1512.03385), [DenseNet](https://arxiv.org/abs/1608.06993) and [U-net](https://arxiv.org/abs/1505.04597).


## Embeddings

**[Embeddings](https://en.wikipedia.org/wiki/Embedding_(machine_learning))** in machine learning are dense, low-dimensional vector representations of data (e.g., words, images, or users) that capture their semantic or contextual meaning in a continuous vector space. Similar data have close embeddings, and the difference between two embeddings can carry a meaning: in the paper that introduced Word2vec, [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781), $$\text{vector}(\text{King}) - \text{vector}(\text{Man}) + \text{vector}(\text{Woman})$$ is closest to $$\text{vector}(\text{Queen})$$. Embeddings can be collected by passing input data (e.g., text, images) through a *trained* machine learning model (e.g., a neural network) and extracting the values from a specific intermediate layer, typically before the final output layer.

The most common data types and their embeddings are:
- words: [Word2vec](https://en.wikipedia.org/wiki/Word2vec) learns an embedding for each word of a vocabulary, from inputs that are one-hot vectors of the words, with one of two models:
  - *skipgram*, which predicts the surrounding words from the current word: slower to train ("3 days" in the [original paper](https://arxiv.org/abs/1301.3781)), better at capturing semantic relationships, e.g. for 'cat' it returns 'dog' as a word with close embeddings;
  - *continuous bag of words* (CBOW), which predicts the current word from the surrounding words: faster to train ("1 day" on the same data), better at capturing syntactic relationships between words, e.g. for 'cat' it returns 'cats';
- text (word sequences): [BERT](https://en.wikipedia.org/wiki/BERT_(language_model));
- non-textual sequences: Encoder-Decoders, e.g. [LSTMs](https://en.wikipedia.org/wiki/Long_short-term_memory), [RNNs](https://en.wikipedia.org/wiki/Recurrent_neural_network);
- point cluster or array: [Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis);
- images:
  - in a classification task: the activation of the last layer *before* the layer that does logit/softmax, i.e. the input to the final layer, e.g. the 512 values of the first fully connected layer of the CNN above;
  - in an image-to-image task, e.g. segmentation using a [U-net](https://arxiv.org/abs/1505.04597): the activation of the last downsampling layer, which is the input of the first upsampling layer, i.e. the *bottleneck* of the network;
- audio: the frames of a spectrogram, or the tokens of a neural audio codec, detailed below;
- videos: a sequence of video frames stacked on a time dimension, i.e. a 5D input, which a 3D convolution takes with the shape (batch size, channels, time, height, width). Embeddings are collected similarly to a regular CNN, with 3D instead of 2D convolutions;
- graphs: the states of the nodes after several steps of message passing, detailed below.

An audio waveform is a long 1D sequence of samples (typically 16-bit integers at 16 to 48 kHz), too long to model directly. It is thus first turned into a shorter sequence of frames, in one of two ways. The first is a log-Mel [spectrogram](https://en.wikipedia.org/wiki/Spectrogram), a 2D time-frequency image of the waveform, fed to a CNN or Transformer encoder whose activations are the embeddings. E.g. [Whisper](https://arxiv.org/abs/2212.04356) resamples audio to 16 kHz and computes 80 Mel channels on 25 ms windows with a 10 ms stride. A vocoder turns a spectrogram back into a waveform, as in the text-to-speech model [TorToise](https://arxiv.org/abs/2305.07243).

The second is a neural audio codec: a convolutional encoder downsamples the waveform into embeddings, and a decoder with transposed convolutions reconstructs it. In between, residual [vector quantization](https://en.wikipedia.org/wiki/Vector_quantization) (RVQ) turns each embedding into discrete tokens: the embedding is replaced by its nearest entry in a codebook, the residual is quantized with a second codebook, and so on. E.g. [EnCodec](https://arxiv.org/abs/2210.13438) turns 24 kHz audio into 75 embeddings per second, a 320-fold reduction, and the number of codebooks sets the bitrate (8 codebooks of 1,024 entries give 6 kbps). [DAC](https://arxiv.org/abs/2306.06546) compresses 44.1 kHz audio into 8 kbps of tokens. These tokens can be modeled like text, e.g. [VALL-E](https://arxiv.org/abs/2301.02111) does text-to-speech by predicting EnCodec tokens with a language model.

For a graph, the embedding of a node is given by its state after several steps of the [message passing algorithm](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html), and the embedding of a graph by a readout (e.g. sum, mean or max) over all node embeddings. A message passing step is given by:

$$x_i^{(k)} = \gamma^{(k)} \left( x_i^{(k-1)}, \Box_{j \in \mathcal{N}(i)} \, \psi^{(k)} \left( x_i^{(k-1)}, x_j^{(k-1)}, e_{j,i} \right) \right)$$

where:
- $$x_i^{(k)}$$ is the embedding of node $$i$$ at message passing step $$k$$;
- $$\mathcal{N}(i)$$ is the set of neighbors of node $$i$$;
- $$e_{j,i}$$ is the embedding of the edge from node $$j$$ to node $$i$$;
- $$\Box$$ is a differentiable, permutation-invariant function, e.g. sum, mean or max;
- and $$\gamma$$ and $$\psi$$ are differentiable functions such as DNNs or RNNs (LSTMs, GRUs).

To find more about the topic, the [Deep Learning](https://www.deeplearningbook.org/) book by Goodfellow, Bengio and Courville covers feedforward networks and backpropagation (chapter 6), regularization and dropout (chapter 7) and convolutional networks (chapter 9).
