---
layout: post
title:  "Deep Neural Networks, Backpropagation, Dropout, CNNs and Embeddings"
categories: ["Machine Learning"]
tags: ["deep learning", "backpropagation", "autodiff", "CNN", "embeddings"]
---

A deep neural network (DNN) is a stack of layers of artificial neurons. The simplest DNN is the [multilayer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron) (MLP), a fully connected feedforward network. MLPs are **[universal approximators](https://en.wikipedia.org/wiki/Universal_approximation_theorem)**: with a non-polynomial activation function and enough hidden neurons, a network with a single hidden layer approximates any continuous function on a compact domain to any accuracy.

An MLP has an input layer of size $$D$$, $$L$$ hidden layers of size $$K$$, and an output layer. It is a **feedforward network**: the computation flows from the input on the left to the output on the right, with no feedback loop. It serves regression, where the output is a real value, and classification, where the output scores each class.

{: .picture}
<img width="33%" height="33%" src="{{ site.assets }}/deep-neural-networks/dnn.svg">

{: .caption}
Source: Machine Learning lecture notes, M. Jaggi, EPFL

The output of neuron $$j$$ of layer $$l$$ is:

$$
x_j^{(l)} = \phi ( \sum_i w_{i,j}^{(l)} x_i^{(l-1)}+b_j^{(l)})
$$

where $$w_{i,j}^{(l)}$$ is the weight from neuron $$i$$ of layer $$l-1$$, $$b_j^{(l)}$$ is the bias, trained like the weights, and $$\phi$$ is the **activation function**, applied to the weighted sum of the inputs. In matrix form, layer $$l$$ computes:

$$
x^{(l)} = f^{(l)} (x^{(l-1)}) = \phi ((W^{(l)})^T x^{(l-1)}+b^{(l)})
$$

where the matrix $$W^{(l)}$$ holds the weights $$w_{i,j}^{(l)}$$, and $$\phi$$ applies to each entry. The output layer is a linear regression on the output of the last hidden layer, or a logistic or softmax regression for classification. Common activation functions are:
- [sigmoid](https://www.wolframalpha.com/input/?i=1%2F(1%2Be%5E%7B-x%7D)): $$\phi(x) = \frac{1}{1+e^{-x}}$$, with values in $$(0, 1)$$;
- [*tanh*](https://www.wolframalpha.com/input/?i=%5Cfrac%7Be%5Ex+-+e%5E%7B-x%7D%7D%7Be%5Ex%2Be%5E%7B-x%7D%7D): $$\phi(x) = \frac{e^x - e^{-x}}{e^x+e^{-x}}$$, with values in $$(-1, 1)$$, centered at $$0$$;
- [Rectified Linear Unit – ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)): $$\phi(x) = \max\{0,x\}$$, non-negative and unbounded, and computed with one comparison;
- leaky ReLU: $$\phi(x) = \max \{\alpha x,x\}$$, with $$0 < \alpha < 1$$, e.g. $$\alpha = 0.01$$.

The derivative of the sigmoid is at most $$1/4$$, and the derivatives of the sigmoid and *tanh* tend to $$0$$ for large $$\vert x \vert$$. Back-propagation (below) multiplies these derivatives across the layers, so the gradients of the first layers can shrink exponentially with the depth: this is the [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem). The derivative of ReLU is $$1$$ for positive inputs, so it does not shrink the gradients. For negative inputs, the output and the derivative of ReLU are $$0$$. A ReLU neuron whose input is negative for all training samples, e.g. after a large negative update of its bias, receives no gradient: its weights stop changing, and the neuron has *died*. The derivative of the leaky ReLU is $$\alpha > 0$$ for negative inputs, so its neurons cannot die.

## Back-propagation

Training minimizes a loss over the $$N$$ training samples, e.g. with [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) (SGD), which takes a small step against the gradient of the loss of one sample or of a mini-batch. For regression with the squared error, the loss is:

$$
\mathcal{L} = \frac{1}{N} \sum_{n=1}^N \mathcal{L}_n = \frac{1}{N} \sum_{n=1}^N ( y_n - f^{(L+1)} \circ ... \circ f^{(2)} \circ f^{(1)} (x_n) ) ^2
$$

SGD needs the derivatives of the loss $$\mathcal{L}_n$$ of sample $$n$$ with respect to all parameters:

$$
\frac{\partial \mathcal{L}_n}{\partial w_{i,j}^{(l)}} \text{ and } \frac{\partial \mathcal{L}_n}{\partial b_{j}^{(l)}}, \text{ for } l=1, ..., L+1
\label{eq_gradient}
$$

Computing each derivative separately repeats most of the work: a finite-difference estimate, for example, needs one forward pass per parameter, so $$P$$ forward passes for $$P$$ parameters. The **back-propagation** algorithm computes all derivatives with one forward and one backward pass. It uses the total input of layer $$l$$, before the activation function:

$$
z^{(l)} = (W^{(l)})^T x^{(l-1)}+b^{(l)}
\label{eq_z}
$$

The forward pass computes $$z^{(l)}$$ and $$x^{(l)} = \phi(z^{(l)})$$ for $$l=1, ..., L+1$$, starting at $$x^{(0)} = x_n$$. The backward pass computes, for $$l = L+1, ..., 1$$:

$$
\delta_j^{(l)} = \frac{\partial \mathcal{L}_n}{\partial z_j^{(l)}}
\label{eq_delta}
$$

By the chain rule, and since $$z^{(l+1)}$$ depends on $$z_j^{(l)}$$ only through $$x_j^{(l)} = \phi(z_j^{(l)})$$:

$$
\begin{align*}
\delta_j^{(l)} = & \sum_k \frac{\partial \mathcal{L}_n}{\partial z_k^{(l+1)}} \frac{\partial z_k^{(l+1)}}{\partial z_j^{(l)}} \\
= & \sum_k \delta_k^{(l+1)} \frac{\partial \left[ (W^{(l+1)})^T x^{(l)}+b^{(l+1)} \right]_k}{\partial z_j^{(l)}} \\
= & \sum_k \delta_k^{(l+1)}  w_{j,k}^{(l+1)} \frac{\partial x_j^{(l)}}{\partial z_j^{(l)}} \\
= & \sum_k \delta_k^{(l+1)} w_{j,k}^{(l+1)} \phi '(z_j^{(l)})
\end{align*}
$$

In matrix form, $$\delta^{(l)} = \left( W^{(l+1)} \delta^{(l+1)} \right) \odot \phi'(z^{(l)})$$, where $$\odot$$ is the element-wise product. The derivatives of equation \ref{eq_gradient} follow from equations \ref{eq_z} and \ref{eq_delta}:

$$
\frac{\partial \mathcal{L}_n}{\partial w_{i,j}^{(l)}} = \sum_k \frac{\partial \mathcal{L}_n}{\partial z_k^{(l)}} \frac{\partial z_k^{(l)}}{\partial w_{i,j}^{(l)}} = \delta_j^{(l)} x_i^{(l-1)}
\label{eq_w}
$$

$$
\frac{\partial \mathcal{L}_n}{\partial b_{j}^{(l)}} = \sum_k \frac{\partial \mathcal{L}_n}{\partial z_k^{(l)}} \frac{\partial z_k^{(l)}}{\partial b_{j}^{(l)}} = \delta_j^{(l)}
\label{eq_b}
$$

In matrix form, the gradient of $$W^{(l)}$$ is the outer product $$x^{(l-1)} (\delta^{(l)})^T$$. The algorithm is:
1. **Forward pass**: set $$x^{(0)} = x_n$$, and compute $$z^{(l)}$$ and $$x^{(l)}$$ for $$l=1, ..., L+1$$;
2. **Backward pass**: compute $$\delta^{(L+1)}$$ from the loss and the activation of the output layer, e.g. $$\delta^{(L+1)} = -2 (y_n - z^{(L+1)})$$ for the squared error and a linear output. Then compute $$\delta^{(l)}$$ for $$l = L, ..., 1$$;
3. **Gradients**: compute the derivatives of equations \ref{eq_w} and \ref{eq_b} for all parameters.

For a mini-batch, the forward pass of a layer is one matrix product, and its backward pass two: $$W^{(l)} \delta^{(l)}$$ for the previous layer, and the weight gradients. The backward pass thus costs about twice the forward pass, and a training step about three forward passes. The backward pass needs the $$z^{(l)}$$ and $$x^{(l)}$$ of all layers, so training stores the activations of all layers for all samples of the mini-batch, while inference keeps only those of the current layer.

## Automatic Differentiation

Back-propagation is a special case of reverse-mode <a href="https://en.wikipedia.org/wiki/Automatic_differentiation">automatic differentiation ("autodiff")</a>. Autodiff frameworks, such as <a href="https://pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html">PyTorch's autograd</a>, define the derivative of each primitive operation once, and apply the chain rule to any composition of these operations. The derivatives of the loss and of the activation functions are thus not derived by hand.

Forward-mode autodiff propagates derivatives from the inputs to the outputs, together with the forward computation. One pass gives the derivatives of all outputs with respect to one input, so the gradient with respect to $$P$$ parameters needs $$P$$ passes. Reverse mode propagates derivatives from the outputs to the inputs: one backward pass gives the derivatives of one output, such as the loss, with respect to all inputs. Reverse mode stores the intermediate values of the forward pass; forward mode does not. Training uses reverse mode, since the loss is a scalar and the parameters are many.

PyTorch builds the computational graph, a directed acyclic graph (DAG) of the operations, while the `forward` methods run: each operation adds a node, whose `grad_fn` computes its derivative. Loops and conditionals in the code may add different nodes at each `forward` call, so the graph cannot be extracted from the model definition alone (see an example [here](https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html)). `tensor.backward()` traverses the graph from the loss to the leaf nodes, and accumulates the gradients of the leaf tensors in their `.grad` fields. It then frees the graph, unless `retain_graph=True` is passed to `backward`, and the next `forward` call builds a new graph. <a href="https://towardsdatascience.com/automatic-differentiation-explained-b4ba8e60c2ad">Chi-Feng Wang's post</a> gives an example of an autodiff graph, and the <a href="{{ site.assets }}/resources/princeton_course_autodiff.pdf">Princeton course COS 324 on automatic differentiation</a> explains the whole process.

Activation checkpointing reduces the memory of training: it stores only the inputs of selected modules, and recomputes their activations during the backward pass. Checkpointing every $$\sqrt{L}$$ layers of an $$L$$-layer network reduces the memory of the activations from $$O(L)$$ to $$O(\sqrt{L})$$, for the cost of one extra forward pass ([Chen et al., 2016](https://arxiv.org/abs/1604.06174)). The post [How Activation Checkpointing enables scaling up training deep learning models](https://medium.com/pytorch/how-activation-checkpointing-enables-scaling-up-training-deep-learning-models-7a93ae01ff2d) shows the reentrant and non-reentrant implementations of PyTorch. Step-by-step examples of back-propagation in PyTorch are in [Hooks for autograd saved tensors](https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html) and [PyTorch Autograd: Backpropagation](https://colab.research.google.com/drive/1aWNdmYt7RcHMbUk-Xz2Cv5-cGFSWPXe0#scrollTo=LfW6wYg42SEW).

## Dropout

[Dropout](https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5) drops each neuron of the selected layers with probability $$p$$ during training. Each training sample uses a new random mask, i.e. a new *thinned* network. The kept outputs are scaled by $$1/(1-p)$$, which keeps their expected value unchanged, so inference uses all neurons, without dropout and without rescaling. The original paper ([Srivastava et al., 2014](https://jmlr.org/papers/v15/srivastava14a.html)) instead multiplies the outgoing weights by the retention probability $$1-p$$ at test time, which gives the same expected activations. A simple implementation is:

```python
class MyDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, X):
        if self.training:
            binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
            return X * binomial.sample(X.size()) * (1.0/(1-self.p))
        return X
```
Dropout reduces overfitting: the network cannot rely on any single neuron, so it learns redundant representations, spread over several neurons. It also makes the activations sparse, without a sparsity penalty such as an L1 norm. Max-norm regularization, a large decaying learning rate and a high momentum improve its results further. Typical drop probabilities are $$0.2$$ for real-valued inputs, and $$0.2$$ to $$0.5$$ for hidden layers. Dropout approximates the average of the predictions of exponentially many thinned networks, at the cost of a single network. Training takes 2 to 3 times longer than without dropout, because the parameter updates are noisy. In CNNs such as AlexNet and VGG, the fully connected layers hold most of the parameters, and dropout is applied to them.

{: .picture}
<img width="35%" height="35%" src="{{ site.assets }}/deep-neural-networks/dropout.svg">

## Convolutional Neural Networks

A fully connected layer connects every input to every output. Audio and images have a local structure: nearby samples or pixels are strongly correlated. An audio stream $$x^{(0)}[n]$$ is commonly processed by a linear time-invariant filter $$h$$, whose output is the [convolution](https://en.wikipedia.org/wiki/Convolution) of the input and the filter:

$$
x^{(1)}[n] = \sum_u h[u] \, x^{(0)} [n-u]
$$

The filter is local: $$h[u]=0$$ for $$\vert u \vert > r$$, for a small radius $$r$$. Different filters extract different features: an averaging filter smooths the signal, and a high-pass filter enhances the differences between neighbouring samples. The same holds for an image:

$$
x^{(1)}[n,m] = \sum_{u,v} h[u,v] \, x^{(0)} [n-u,m-v]
$$

The output at position $$[n, m]$$ depends only on the input at the positions close to $$[n, m]$$, so a locally connected layer suffices. The same filter, i.e. the same weights, applies at every position. This **weight sharing** reduces the number of parameters further. For a $$256 \times 256$$ image and an output of the same size, a fully connected layer has $$256^4 \approx 4.3 \times 10^9$$ weights, a locally connected layer with $$3 \times 3$$ windows has $$256^2 \times 9 \approx 5.9 \times 10^5$$, and a convolution with a $$3 \times 3$$ filter has $$9$$. The picture compares local (top) and full (bottom) connectivity:

{: .picture}
<img width="35%" height="35%" src="{{ site.assets }}/deep-neural-networks/fully_sparse_connectivity.svg">

A convolutional layer applies several filters to the same input, and each filter produces one output *channel*. For images, the input and the output of a layer are thus 3D: height, width and channels. A layer with $$C_{in}$$ input channels, $$C_{out}$$ output channels and filters of $$(2r+1) \times (2r+1)$$ has $$C_{in} C_{out} (2r+1)^2$$ weights, and computes as many multiply-accumulate operations per output position. Each weight is reused at every output position, which gives convolutions with many channels a high [arithmetic intensity](https://en.wikipedia.org/wiki/Roofline_model#Arithmetic_intensity). Deeper layers typically have more channels, and a smaller height and width. A convolutional neural network (CNN) for images has the following structure:

{: .picture}
<img width="65%" height="65%" src="{{ site.assets }}/deep-neural-networks/cnn.svg">

It has three types of layers:
- convolutional layers apply the filters. Deep learning libraries compute a [cross-correlation](https://en.wikipedia.org/wiki/Cross-correlation), $$\sum_{u} h[u] \, x[n+u]$$, which is a convolution with the flipped filter. The filters are learned, so the two are equivalent;
- pooling layers reduce each window of neighbouring outputs, e.g. of $$2 \times 2$$, to one value, e.g. the maximum (*max pooling*) or the average;
- fully connected layers, an MLP at the end, map the features to the output, e.g. to the class scores.

The height and width decrease through the layers, by pooling, by strides larger than 1, or by convolutions without padding at the boundaries. Back-propagation treats each connection as if it had its own weight. The gradient of a shared weight is then the sum of the gradients of all connections that share it.

## Skip connections

Deep *plain* networks, without skip connections, train worse as their depth grows: beyond some depth, both the training and the test errors increase. This *degradation problem* is not overfitting, since the training error increases too. Normalized initialization and [normalization](https://arxiv.org/abs/1803.08494) layers largely address the vanishing and exploding gradients, but not the degradation: the authors of [ResNet](https://arxiv.org/abs/1512.03385) measured healthy gradient norms in degraded networks with batch normalization, and attribute the degradation to the difficulty of optimizing deep plain networks.

*Skip connections* feed the output of a layer to later layers that are not directly connected to it. A residual block of ResNet computes $$y = F(x) + x$$, where $$F$$ is a stack of two or three layers. The block learns only the residual $$F(x) = y - x$$, and the gradient flows to the earlier layers through the identity path. [DenseNet](https://arxiv.org/abs/1608.06993) instead concatenates the outputs of all previous layers of a block as the input of each layer. Addition keeps the number of channels constant. Concatenation increases it with every layer, which costs memory, but gives each layer direct access to all earlier features.

{: .picture}
<img width="65%" height="65%" src="{{ site.assets }}/deep-neural-networks/densenet.svg">

{: .caption}
An example of a CNN with skip connections (DenseNet). Adapted from: <a href="https://arxiv.org/abs/1608.06993">Densely Connected Convolutional Networks</a> 

[U-net](https://arxiv.org/abs/1505.04597) concatenates the features of each downsampling stage to the upsampling stage of the same resolution.

## Embeddings

An *embedding* is a dense, low-dimensional vector that represents a data item, e.g. a word, an image or a user, such that similar items have close vectors. A common way to obtain one is to pass the input through a trained network, and to take the output of an intermediate layer, typically one before the output layer. Common embeddings per data type are:
- words: [Word2vec](https://en.wikipedia.org/wiki/Word2vec) learns one vector per word of the vocabulary, with one of two models ([Mikolov et al., 2013](https://arxiv.org/abs/1301.3781)). *Continuous bag-of-words* (CBOW) predicts a word from its neighbours, and *skip-gram* predicts the neighbours from the word. In the original paper, CBOW trained in about a day and skip-gram in about three days. Skip-gram is more accurate on semantic analogies, e.g. Athens : Greece = Oslo : Norway, and CBOW slightly more accurate on syntactic analogies, e.g. great : greater = tough : tougher. One-hot vectors, the usual input of these models, are not embeddings: their size is the size of the vocabulary, and all pairs of words are at the same distance;
- sentences and documents: the outputs of a Transformer encoder such as [BERT](https://en.wikipedia.org/wiki/BERT_(language_model)), e.g. at its `[CLS]` token, or averaged over all tokens;
- other sequences: the last hidden state of a recurrent encoder, e.g. an LSTM;
- vectors: their coordinates along the first principal components, from Principal Component Analysis (PCA);
- images:
  - in a classification network: the input of the last layer, the one that computes the class scores (logits);
  - in an image-to-image network, e.g. a [U-net](https://arxiv.org/abs/1505.04597) for segmentation: the output of the last downsampling stage, the *bottleneck* between the downsampling and the upsampling paths;
- audio: a waveform is a long 1D sequence of samples, typically 16-bit integers at 16 to 48 kHz. It is too long to model directly, so it is first converted into a shorter sequence of frames:
  - spectrograms: the waveform is converted into a log-Mel spectrogram, a 2D time-frequency image, and fed to a CNN or Transformer encoder whose activations are the embeddings. E.g. [Whisper](https://arxiv.org/abs/2212.04356) resamples audio to 16 kHz and computes 80 Mel channels on 25 ms windows with a 10 ms stride. A vocoder turns a spectrogram back into a waveform, as in the text-to-speech model [TorToise](https://arxiv.org/abs/2305.07243);
  - neural audio codecs: a convolutional encoder downsamples the waveform into embeddings, and a decoder with transposed convolutions reconstructs it. In between, residual vector quantization (RVQ) turns each embedding into discrete tokens: the embedding is replaced by its nearest entry in a codebook, the residual is quantized with a second codebook, and so on. E.g. [EnCodec](https://arxiv.org/abs/2210.13438) turns 24 kHz audio into 75 embeddings per second, a 320-fold reduction, and the number of codebooks sets the bitrate: 8 codebooks of 1024 entries, i.e. 10 bits each, give $$75 \times 8 \times 10 = 6000$$ bit/s. [DAC](https://arxiv.org/abs/2306.06546) compresses 44.1 kHz audio into 8 kbps of tokens. These tokens can be modelled like text, e.g. [VALL-E](https://arxiv.org/abs/2301.02111) does text-to-speech by predicting EnCodec tokens with a language model;
- videos: the input is a 5D tensor, of batch size, time, channels, height and width, i.e. a sequence of frames. The embeddings are taken as in an image CNN, with 3D convolutions over time, height and width instead of 2D convolutions;
- graphs: the embedding of a node is its state after several steps of the [message passing algorithm](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html), and the embedding of the graph is a readout, e.g. the sum, mean or maximum, over all node embeddings. A message passing step is:

  $$x_i^{(k)} = \gamma^{(k)} \left( x_i^{(k-1)}, \Box_{j \in \mathcal{N}(i)} \, \psi^{(k)} \left( x_i^{(k-1)}, x_j^{(k-1)}, e_{j,i} \right) \right)$$

  where:
  - $$x_i^{(k)}$$ is the embedding of node $$i$$ after step $$k$$;
  - $$\mathcal{N}(i)$$ is the set of neighbours of node $$i$$;
  - $$e_{j,i}$$ is the embedding of the edge from node $$j$$ to node $$i$$;
  - $$\Box$$ is a differentiable, permutation-invariant function, e.g. sum, mean or max;
  - $$\gamma$$ and $$\psi$$ are differentiable functions, e.g. MLPs, or recurrent units such as GRUs.
