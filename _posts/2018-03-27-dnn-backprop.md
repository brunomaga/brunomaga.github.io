---
layout: post
title:  "Deep Neural Networks, Backpropagation, Dropout, CNNs and Embeddings"
categories: ["Machine Learning"]
tags: ["deep learning", "backpropagation", "autodiff", "NumPy", "activation functions", "dropout", "CNN", "skip connections", "optimizers", "SGD", "Adam", "embeddings"]
description: "How deep neural networks compute and learn: activation functions and XOR, backpropagation derived and implemented in NumPy, automatic differentiation, dropout, convolutional neural networks (CNNs), skip connections, optimizers (SGD, Adam) and embeddings."
---

In 1958, Frank Rosenblatt described the [perceptron](https://en.wikipedia.org/wiki/Perceptron), an artificial neuron that learns to separate two classes with a linear boundary. In 1969, the book [Perceptrons](https://en.wikipedia.org/wiki/Perceptrons_(book)) by Marvin Minsky and Seymour Papert showed that such a neuron cannot learn a function as simple as the [exclusive or](https://en.wikipedia.org/wiki/Exclusive_or) (XOR) of two binary inputs. No line separates the inputs $$(0, 1)$$ and $$(1, 0)$$, of output 1, from $$(0, 0)$$ and $$(1, 1)$$, of output 0. Networks with several layers of such neurons can learn XOR and much more complex functions. In 1986, the paper [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0), by David Rumelhart, Geoffrey Hinton and Ronald Williams, showed how to train them with gradient descent, by propagating the errors backwards from the output layer to the first one.

Deep Neural Networks (also known as [Multi-Layer Perceptrons](https://en.wikipedia.org/wiki/Multilayer_perceptron)) are **[universal approximators](https://en.wikipedia.org/wiki/Universal_approximation_theorem)**, i.e. they can approximate any continuous function on a closed and bounded domain, even with a single hidden layer, given enough neurons. They are the most important and relevant ML tool in the field of supervised learning, as they are part of the models used for most supervised tasks. This post covers their structure, the backpropagation algorithm that trains them and its implementation in NumPy and with automatic differentiation, and then dropout, convolutional networks, skip connections, optimizers with a training loop, and embeddings.

## Feedforward Networks and Activation Functions

The structure of a basic NN is the following: one input layer of size $$D$$, $$L$$ hidden layers of size $$K$$, and one output layer. It is a **[feedforward network](https://en.wikipedia.org/wiki/Feedforward_neural_network)**: the computation performed by the network starts with the input from the left and flows to the right, with no feedback loop. It can be used for regression (when the output is a real value) and for classification (when the output is the class). The following picture illustrates a network with one hidden layer, where each node is connected to every node of the next layer, i.e. a **fully connected** network:

{: .picture}
<img width="40.9%" height="40.9%" src="{{ site.assets }}/dnn-backprop/dnn.svg">

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

where $$\odot$$ is the element-wise ([Hadamard](https://en.wikipedia.org/wiki/Hadamard_product_(matrices))) product. For a mini-batch of $$B$$ samples, the vectors of all samples are stacked into matrices. Deep learning libraries such as PyTorch store one sample per row (*batch first*), so that $$x^{(l)}$$, $$z^{(l)}$$ and $$\delta^{(l)}$$ become matrices with the vector of one sample in each row, and $$b^{(l)}$$ a row vector added to every row. The equations then become:

$$
\begin{align*}
z^{(l)} & = x^{(l-1)} W^{(l)} + b^{(l)} \\
\delta^{(l)} & = \left( \delta^{(l+1)} (W^{(l+1)})^T \right) \odot \phi'(z^{(l)}) \\
\frac{\partial \mathcal{L}}{\partial W^{(l)}} & = (x^{(l-1)})^T \delta^{(l)} \\
\frac{\partial \mathcal{L}}{\partial b^{(l)}} & = \sum_{n=1}^{B} \delta_n^{(l)}
\end{align*}
$$

where $$\mathcal{L}$$ is the loss of the mini-batch, and $$\delta_n^{(l)}$$ is the row of the sample $$n$$. The product $$(x^{(l-1)})^T \delta^{(l)}$$ adds up the derivatives of all samples, as the sum does for the biases. Each layer then runs one matrix product in the forward pass, $$x^{(l-1)} W^{(l)}$$, and two of the same size in the backward pass, one for the errors and one for the derivatives of the weights. Therefore, the backward pass takes about twice the floating-point operations of the forward pass, and a training step about three times.

These equations map directly to [NumPy](https://en.wikipedia.org/wiki/NumPy) code, with one sample per row. First, each parameter holds its values (`data`) and the derivative of the loss with respect to them (`grad`), as a PyTorch tensor does. A fully connected (*linear*) layer holds the parameters $$W^{(l)}$$ and $$b^{(l)}$$, and its `forward` method computes $$z^{(l)}$$ of the first equation above. Its `backward` method receives $$\delta^{(l)}$$, adds $$\partial \mathcal{L} / \partial W^{(l)}$$ and $$\partial \mathcal{L} / \partial b^{(l)}$$ to the `grad` of the parameters, and returns $$\delta^{(l)} (W^{(l)})^T$$, from which the previous layer computes its error.

The weights start with the *normalized initialization* of [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html), known as **Xavier initialization**: a uniform distribution of variance $$2 / (n_{\text{in}} + n_{\text{out}})$$, for a layer of $$n_{\text{in}}$$ inputs and $$n_{\text{out}}$$ outputs (Eq. 16 in the paper). This variance keeps the activations and the gradients at about the same scale across layers, while too small weights make them vanish and too large ones make them explode. [He initialization](https://arxiv.org/abs/1502.01852), a normal distribution of variance $$2 / n_{\text{in}}$$, is its variant for ReLU activations, which set about half of their inputs to zero. The biases start at zero:

```python
import numpy as np

class Parameter:
    def __init__(self, data):
        self.data, self.grad = data, np.zeros_like(data)

class Linear:
    def __init__(self, n_in, n_out):
        limit = np.sqrt(6 / (n_in + n_out))  # Xavier initialization
        self.W = Parameter(np.random.uniform(-limit, limit, (n_in, n_out)))
        self.b = Parameter(np.zeros((1, n_out)))

    def forward(self, x):
        self.x = x  # saved for the backward pass
        return x @ self.W.data + self.b.data

    def backward(self, delta):
        self.W.grad += self.x.T @ delta
        self.b.grad += delta.sum(axis=0, keepdims=True)
        return delta @ self.W.data.T

    def parameters(self):
        return [self.W, self.b]
```

Its three matrix products are the ones counted above. Note that `forward` saves its input $$x^{(l-1)}$$, which the derivative of the weights needs, and that `backward` adds to `grad` instead of overwriting it, which also sums the derivatives of a parameter used more than once, such as the shared weights of the CNNs below.

Then, the ReLU, $$\phi(z) = \max\{0, z\}$$, whose derivative $$\phi'(z)$$ is 1 for $$z > 0$$ and 0 otherwise. Its `backward` method multiplies the derivative of the loss with respect to its output, $$\delta^{(l+1)} (W^{(l+1)})^T$$ (`dx`), by $$\phi'(z^{(l)})$$ (`self.z > 0`), which gives $$\delta^{(l)}$$ as in the second equation above. It has no parameters:

```python
class ReLU:
    def forward(self, z):
        self.z = z  # saved for the backward pass
        return np.maximum(0, z)

    def backward(self, dx):
        return dx * (self.z > 0)

    def parameters(self):
        return []
```

### Automatic Differentiation

The chain rule shows the theoretical framework, but not how to compute it in software. Backpropagation is a special case of <a href="https://en.wikipedia.org/wiki/Automatic_differentiation">automatic differentiation ("autodiff")</a>, in its *reverse mode*. The reverse mode computes the derivatives of one output (the loss) with respect to all inputs (the parameters) in one backward pass, while the *forward mode* needs one pass per input. In practice, autodiff libraries such as <a href="https://pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html">PyTorch's autograd</a> apply the chain rule to the derivatives of the elementary operations of the forward pass. Thus, contrary to the derivation above, the derivatives of the loss and activation functions do not have to be specified explicitly.

Instead, a computational graph (a [DAG](https://en.wikipedia.org/wiki/Directed_acyclic_graph)) is created dynamically, where nodes are added to the graph as the modules' `forward` methods are called. The backward propagation is then computed by applying the chain rule to the derivatives (`grad_fn`) of the nodes in the computational graph. Recursive calls add new nodes to the graph for every call, and conditional calls may add different nodes at each `forward` pass; therefore, the computational graph cannot be extracted from the model definition directly (see an example [here](https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html)).

Finally, when `tensor.backward()` is called, it recursively computes the gradients up to the leaf nodes of the graph. When this process ends, the computational graph is discarded, and a new one may be created via a new `forward` call (unless `retain_graph=True` is set in the `backward` function). For more details, <a href="https://towardsdatascience.com/automatic-differentiation-explained-b4ba8e60c2ad">Chi-Feng Wang's post</a> gives an illustrative example of the automatic differentiation graph, and the <a href="{{ site.assets }}/resources/princeton_course_autodiff.pdf">Princeton course COS 324 on automatic differentiation</a> a thorough explanation of the whole process.

Storing a computational graph is also useful for [activation checkpointing](https://arxiv.org/abs/1604.06174), where one needs to recompute a module's output during the backward pass, as it is discarded during the forward pass to save memory. A good example of activation checkpointing based on the reentrant and non-reentrant PyTorch implementations can be found in the post [How Activation Checkpointing enables scaling up training deep learning models](https://medium.com/pytorch/how-activation-checkpointing-enables-scaling-up-training-deep-learning-models-7a93ae01ff2d).

It also follows that inference requires less memory than training. The derivative of the weights of a layer, $$x^{(l-1)} (\delta^{(l)})^T$$, needs the input of the layer (`self.x` of the linear layer above). Therefore, training saves the activations of all layers in the computational graph until the backward pass, while inference can discard them as soon as the next layer is computed.

The two layers above are enough for a minimal form of reverse-mode autodiff, whose elementary operations are the layers. A multi-layer perceptron (MLP) stores its layers in a list, in the order of the forward pass, which computes $$f^{(L+1)} \circ ... \circ f^{(1)}$$. This list plays the role of the computational graph, and each layer saves what its backward pass needs (`self.x` and `self.z`), as the nodes of PyTorch's graph do. The `backward` method then runs the layers in reverse order, from the error $$\delta^{(L+1)}$$ of the output layer to the first layer:

```python
class MLP:
    def __init__(self, sizes):
        self.layers = []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            self.layers += [Linear(n_in, n_out), ReLU()]
        self.layers.pop()  # no activation after the output layer

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, delta):
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

    def parameters(self):
        return [theta for layer in self.layers for theta in layer.parameters()]
```

Contrary to PyTorch, the graph is fixed by the model definition, and the derivatives of each layer are written by hand. As an example, `MLP([2, 16, 16, 1])` builds a network with $$D = 2$$ inputs, $$L = 2$$ hidden layers of $$K = 16$$ ReLU neurons and one output. The finite differences above can check its derivatives: on a small network, the derivatives computed both ways agree to about $$10^{-10}$$.

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
<img width="68.5%" height="68.5%" src="{{ site.assets }}/dnn-backprop/dropout.svg">

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
<img width="30%" height="30%" src="{{ site.assets }}/dnn-backprop/fully_sparse_connectivity.svg">

{: .caption}
Sparse and full connectivity, viewed from below. Top: a locally connected layer with a kernel of width 3, where the input $$x_3$$ affects only the outputs $$s_2$$, $$s_3$$ and $$s_4$$. Bottom: a fully connected layer, where $$x_3$$ affects every output. Adapted from: [Deep Learning, I. Goodfellow, Y. Bengio and A. Courville, Figure 9.2](https://www.deeplearningbook.org/contents/convnets.html)

In the picture, the inputs $$x_i$$ and the outputs $$s_i$$ stand for the $$x^{(0)}$$ and $$x^{(1)}$$ of the text. The fully connected layer has $$5 \times 5 = 25$$ weights, one per connection. The locally connected layer has 13 connections, as each output reads at most 3 inputs, and, with weight sharing, only the 3 weights of its filter.

It is common to not only compute the output of a single filter but to use multiple filters. The various outputs are called **channels** (or *feature maps*). This introduces some additional parameters into the model, as each filter has its own weights, and spans all the channels of its input.

With several channels, the result is not a 2D output in the next level, but in fact a 3D output. Layer after layer, there are increasingly more channels but a smaller “footprint”. In brief, applying local connectivity to a 2D input dataset on a *deep* neural network leads to the following final structure, a **[convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network)** (CNN):

{: .picture}
<img width="92%" height="92%" src="{{ site.assets }}/dnn-backprop/cnn.svg">

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
<img width="82.1%" height="82.1%" src="{{ site.assets }}/dnn-backprop/densenet.svg">

{: .caption}
A 5-layer dense block of DenseNet with a growth rate of $$k = 4$$: each layer $$H_i$$ (BN-ReLU-Conv) takes the feature maps $$x_0, \ldots, x_{i-1}$$ of all preceding layers as input and produces $$k$$ new feature maps $$x_i$$. The transition layer takes all of them. Adapted from: [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

In the picture, the input $$x_0$$ has $$k_0 = 6$$ feature maps and $$k = 4$$, so the layer $$H_4$$ receives $$6 + 4 \times 3 = 18$$ feature maps, and the transition layer, which takes all of them, $$6 + 4 \times 4 = 22$$. To sum up, an addition (ResNet) keeps the number of channels constant, while a concatenation (DenseNet) lets every layer read the feature maps of all the layers before it. The price of DenseNet is an input that grows with the depth of the block, which is why it uses small growth rates ($$k$$ from 12 to 40 in its experiments).

There is currently a plethora of different implementations of CNNs with several layouts of skip connections. To know more, the curious reader can have a look at [ResNet](https://arxiv.org/abs/1512.03385), [DenseNet](https://arxiv.org/abs/1608.06993) and [U-net](https://arxiv.org/abs/1505.04597).

## Optimizers

Backpropagation computes the gradient $$g_t$$ of the loss with respect to the parameters $$\theta$$ of the network (all its weights and biases), at the step $$t$$ of the training. An **optimizer** uses this gradient to update the parameters. The simplest one, **gradient descent** (GD), takes a step against the gradient, of a size set by the learning rate $$\eta$$:

$$
\theta_t = \theta_{t-1} - \eta \, g_t
$$

In GD, $$g_t$$ is the gradient of the loss over all $$N$$ training samples, so each step follows the exact gradient, but costs a pass over the whole dataset. **Stochastic gradient descent** (SGD) computes it on one sample or, in practice, on a mini-batch of $$B$$ samples drawn at random (mini-batch SGD), which gives a noisy but unbiased estimate of the gradient, $$N/B$$ times cheaper to compute. **Momentum** smooths the steps of SGD: it keeps a decaying sum $$m_t$$ of the past gradients, and steps along it:

$$
\begin{align*}
m_t & = \mu \, m_{t-1} + g_t \\
\theta_t & = \theta_{t-1} - \eta \, m_t
\end{align*}
$$

where the momentum $$\mu$$ is usually 0.9 (0.95 to 0.99 with dropout, as seen above). Gradients that agree from step to step add up, while those that change sign cancel out, which also speeds up training along the long, narrow valleys of the loss.

In code, an optimizer holds the parameters of a model. Its `zero_grad` method resets their derivatives, which the backward pass accumulates, and its `step` method updates them. The following SGD implements the equations above, and plain SGD for $$\mu = 0$$, its default:

```python
class Optimizer:
    def __init__(self, parameters, eta):
        self.parameters, self.eta = parameters, eta

    def zero_grad(self):
        for theta in self.parameters:
            theta.grad.fill(0)

class SGD(Optimizer):
    def __init__(self, parameters, eta, mu=0):
        super().__init__(parameters, eta)
        self.mu = mu
        self.m = [np.zeros_like(theta.data) for theta in parameters]

    def step(self):
        for theta, m in zip(self.parameters, self.m):
            m[:] = self.mu * m + theta.grad
            theta.data -= self.eta * m
```

**Adam** (adaptive moment estimation), introduced in [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980), keeps the momentum $$m_t$$ as a running average of the gradients instead of a sum, and a running average $$v_t$$ of their squares. It divides the step of each parameter by $$\sqrt{v_t}$$, so that each parameter moves by at most about $$\eta$$ per step, whatever the scale of its gradients (section 2.1 of the paper). Its update is given by (Algorithm 1 in the paper, which writes $$\alpha$$ for $$\eta$$):

$$
\begin{align*}
m_t & = \beta_1 m_{t-1} + (1 - \beta_1) \, g_t \\
v_t & = \beta_2 v_{t-1} + (1 - \beta_2) \, g_t^2 \\
\hat{m}_t & = m_t / (1 - \beta_1^t) \\
\hat{v}_t & = v_t / (1 - \beta_2^t) \\
\theta_t & = \theta_{t-1} - \eta \, \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)
\end{align*}
$$

where the square, the root and the division apply to each entry. As $$m_0 = v_0 = 0$$, both averages are biased toward zero in the first steps, and $$\hat{m}_t$$ and $$\hat{v}_t$$ correct this bias. The paper suggests $$\eta = 0.001$$, $$\beta_1 = 0.9$$, $$\beta_2 = 0.999$$ and $$\epsilon = 10^{-8}$$, the default values of the code:

```python
class Adam(Optimizer):
    def __init__(self, parameters, eta=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(parameters, eta)
        self.beta1, self.beta2, self.eps, self.t = beta1, beta2, eps, 0
        self.m = [np.zeros_like(theta.data) for theta in parameters]
        self.v = [np.zeros_like(theta.data) for theta in parameters]

    def step(self):
        self.t += 1
        for theta, m, v in zip(self.parameters, self.m, self.v):
            m[:] = self.beta1 * m + (1 - self.beta1) * theta.grad
            v[:] = self.beta2 * v + (1 - self.beta2) * theta.grad**2
            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)
            theta.data -= self.eta * m_hat / (np.sqrt(v_hat) + self.eps)
```

In practice, Adam needs less tuning of the learning rate than SGD, and Adam or its variants train most Transformers, while SGD with momentum remains common for CNNs. Adam's variant AdamW, from [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101), shrinks the weights at every step (*weight decay*) directly, instead of adding an L2 regularization term to the gradient, which Adam would rescale.

### Training Loop

The training loop needs one more piece, the loss. The squared loss of the backpropagation section, averaged over the $$B$$ samples of a mini-batch, is the mean squared error (MSE):

$$
\mathcal{L} = \frac{1}{B} \sum_{n=1}^{B} (y_n - \hat{y}_n)^2
$$

where $$\hat{y}_n$$ is the output of the network for the sample $$n$$. The `backward` method of the loss returns the error of the output layer, $$\delta_n^{(L+1)} = -2 (y_n - \hat{y}_n)$$ as in the backpropagation section, divided by $$B$$ as the loss is a mean:

```python
class MSELoss:
    def forward(self, y_hat, y):
        self.y_hat, self.y = y_hat, y  # saved for the backward pass
        return np.mean((y - y_hat)**2)

    def backward(self):
        B = self.y.size  # samples in the mini-batch, with one output each
        return -2 * (self.y - self.y_hat) / B
```

For classification, a [softmax](https://en.wikipedia.org/wiki/Softmax_function) output with the [cross-entropy](https://en.wikipedia.org/wiki/Cross-entropy) loss gives an error of the same form, $$\hat{y}_n - y_n$$, where $$\hat{y}_n$$ holds the predicted probabilities of the classes and $$y_n$$ is the [one-hot](https://en.wikipedia.org/wiki/One-hot) vector of the true class.

Then, the loop. The following example trains an MLP on a continuous version of XOR: points drawn uniformly in $$[-1, 1]^2$$, with the target 1 when their two coordinates have different signs and 0 otherwise. As for XOR, no line separates the two classes. At each step, the loop draws a random mini-batch of $$B = 32$$ samples, resets the derivatives, and runs the forward pass, the loss, the backward pass and the optimizer step, in the order of a PyTorch training loop:

```python
def xor_data(N):
    x = np.random.uniform(-1, 1, (N, 2))
    y = (x[:, 0] * x[:, 1] < 0).astype(float).reshape(N, 1)  # 1 if the signs differ
    return x, y

np.random.seed(0)
N, B = 1024, 32  # training samples, mini-batch size
x, y = xor_data(N)
x_test, y_test = xor_data(256)
model = MLP([2, 16, 16, 1])
loss_fn = MSELoss()
optimizer = Adam(model.parameters(), eta=0.01)

for step in range(2000):
    n = np.random.choice(N, B, replace=False)  # samples of a random mini-batch
    optimizer.zero_grad()
    loss = loss_fn.forward(model.forward(x[n]), y[n])
    model.backward(loss_fn.backward())
    optimizer.step()
    if step % 500 == 0:
        print(f"step {step}: loss {loss:.4f}")

print("test accuracy:", np.mean((model.forward(x_test) > 0.5) == y_test))
```

With $$B = N$$, the same loop runs GD. After 2,000 steps, the network classifies 99.2% of the 256 test points correctly, with an output above 0.5 for the target 1, and the two misclassified points lie within 0.006 of an axis, where the class changes. SGD with momentum, `SGD(model.parameters(), eta=0.1, mu=0.9)`, classifies 98.8% of them correctly.

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

To find more about the topic, the [Deep Learning](https://www.deeplearningbook.org/) book by Goodfellow, Bengio and Courville covers feedforward networks and backpropagation (chapter 6), regularization and dropout (chapter 7), optimization (chapter 8) and convolutional networks (chapter 9).
