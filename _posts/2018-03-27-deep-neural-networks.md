---
layout: post
title:  "Deep Neural Networks, Backpropagation, Dropout, CNNs and Embeddings"
categories: ["Machine Learning"]
tags: ["deep learning", "backpropagation", "autodiff", "CNN", "embeddings"]
---

A deep neural network (DNN) is a stack of layers of artificial neurons, and the simplest one is the [multilayer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron) (MLP), a fully connected feedforward network. MLPs are **[universal approximators](https://en.wikipedia.org/wiki/Universal_approximation_theorem)**: with a non-polynomial activation function and enough hidden neurons, a network with a single hidden layer approximates any continuous function on a compact domain to any accuracy. The theorem guarantees that such a network exists, but not that training finds it, nor how many neurons it needs. DNNs are part of the models used for most supervised learning tasks.

An MLP has an input layer of size $$D$$, $$L$$ hidden layers of size $$K$$, and an output layer. It is a **feedforward network**: the computation starts with the input on the left and flows to the output on the right, with no feedback loop. It can be used for regression, where the output is a real value, and for classification, where the output gives a score to each class.

{: .picture}
<img width="36.5%" height="36.5%" src="{{ site.assets }}/deep-neural-networks/dnn.svg">

{: .caption}
A fully connected feedforward network with an input layer of size $$D = 4$$, one hidden layer of size $$K = 5$$ and one output $$\hat{y}$$. Adapted from: [Machine Learning course CS-433 lecture notes, M. Jaggi, EPFL](https://github.com/epfml/ML_course)

Each neuron computes its output in two steps: first a weighted sum of its inputs, then a non-linear function of this sum. The output of neuron $$j$$ of layer $$l$$ is:

$$
x_j^{(l)} = \phi ( \sum_i w_{i,j}^{(l)} x_i^{(l-1)}+b_j^{(l)})
$$

where $$w_{i,j}^{(l)}$$ is the weight from neuron $$i$$ of layer $$l-1$$, $$b_j^{(l)}$$ is the bias, trained like the weights, and $$\phi$$ is the **activation function**. In matrix form, layer $$l$$ computes:

$$
x^{(l)} = f^{(l)} (x^{(l-1)}) = \phi ((W^{(l)})^T x^{(l-1)}+b^{(l)})
$$

where the matrix $$W^{(l)}$$ holds the weights $$w_{i,j}^{(l)}$$, and $$\phi$$ applies to each entry. Without the activation function, the layers would compose into a single linear map, so the non-linearity is what lets the network represent non-linear functions. The output layer works as the [regression methods described previously]({{ site.baseurl }}{% post_url 2017-08-17-linear-regression-matrix-factorization %}): it is a linear regression on the output of the last hidden layer, or a logistic or softmax regression for classification. Common activation functions are:
- [sigmoid](https://www.wolframalpha.com/input/?i=1%2F(1%2Be%5E%7B-x%7D)): $$\phi(x) = \frac{1}{1+e^{-x}}$$, with values in $$(0, 1)$$;
- [*tanh*](https://www.wolframalpha.com/input/?i=%5Cfrac%7Be%5Ex+-+e%5E%7B-x%7D%7D%7Be%5Ex%2Be%5E%7B-x%7D%7D): $$\phi(x) = \frac{e^x - e^{-x}}{e^x+e^{-x}}$$, with values in $$(-1, 1)$$, centered at $$0$$;
- [Rectified Linear Unit – ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)): $$\phi(x) = \max\{0,x\}$$, non-negative and unbounded, and cheap to compute, with one comparison;
- leaky ReLU: $$\phi(x) = \max \{\alpha x,x\}$$, with $$0 < \alpha < 1$$, e.g. $$\alpha = 0.01$$.

The choice matters for training. The derivative of the sigmoid is at most $$1/4$$, and the derivatives of the sigmoid and *tanh* tend to $$0$$ for large $$\vert x \vert$$, where the functions saturate. Backpropagation (below) multiplies these derivatives across the layers, so the gradients of the first layers can shrink exponentially with the depth: this is the [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem). ReLU mitigates it, since its derivative is $$1$$ for positive inputs. For negative inputs, however, the output and the derivative of ReLU are $$0$$. A ReLU neuron whose input is negative for all training samples, e.g. after a large negative update of its bias, receives no gradient: the error signal sent back to it is multiplied by $$0$$, its weights stop changing, and the neuron has *died*. The derivative of the leaky ReLU is $$\alpha > 0$$ for negative inputs, so its neurons cannot die.

## Backpropagation

As in regression, training minimizes a loss over the $$N$$ training samples, e.g. with [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) (SGD). SGD computes the gradient of the loss of one sample, or of a mini-batch, with respect to all weights and biases of the network, and then takes a small step in the opposite direction. The network is a composition of layers, so these derivatives follow from the chain rule of calculus. For regression with the squared error, the loss is:

$$
\mathcal{L} = \frac{1}{N} \sum_{n=1}^N \mathcal{L}_n = \frac{1}{N} \sum_{n=1}^N ( y_n - f^{(L+1)} \circ ... \circ f^{(2)} \circ f^{(1)} (x_n) ) ^2
$$

SGD needs the derivatives of the loss $$\mathcal{L}_n$$ of sample $$n$$ with respect to all parameters:

$$
\frac{\partial \mathcal{L}_n}{\partial w_{i,j}^{(l)}} \text{ and } \frac{\partial \mathcal{L}_n}{\partial b_{j}^{(l)}}, \text{ for } l=1, ..., L+1
\label{eq_gradient}
$$

A network has many parameters, often millions, and computing each derivative separately would repeat most of the work: a finite-difference estimate, for example, needs one forward pass per parameter, so $$P$$ forward passes for $$P$$ parameters. The **backpropagation** algorithm computes all derivatives with one forward and one backward pass, by reusing the intermediate results. It uses the total input of layer $$l$$, before the activation function:

$$
z^{(l)} = (W^{(l)})^T x^{(l-1)}+b^{(l)}
\label{eq_z}
$$

The forward pass computes $$z^{(l)}$$ and $$x^{(l)} = \phi(z^{(l)})$$ for $$l=1, ..., L+1$$, starting at $$x^{(0)} = x_n$$. The backward pass then computes the derivatives of the loss with respect to these total inputs, from the last layer to the first, for $$l = L+1, ..., 1$$:

$$
\delta_j^{(l)} = \frac{\partial \mathcal{L}_n}{\partial z_j^{(l)}}
\label{eq_delta}
$$

Each $$\delta^{(l)}$$ follows from $$\delta^{(l+1)}$$. By the chain rule, and since $$z^{(l+1)}$$ depends on $$z_j^{(l)}$$ only through $$x_j^{(l)} = \phi(z_j^{(l)})$$:

$$
\begin{align*}
\delta_j^{(l)} = & \sum_k \frac{\partial \mathcal{L}_n}{\partial z_k^{(l+1)}} \frac{\partial z_k^{(l+1)}}{\partial z_j^{(l)}} \\
= & \sum_k \delta_k^{(l+1)} \frac{\partial \left[ (W^{(l+1)})^T x^{(l)}+b^{(l+1)} \right]_k}{\partial z_j^{(l)}} \\
= & \sum_k \delta_k^{(l+1)}  w_{j,k}^{(l+1)} \frac{\partial x_j^{(l)}}{\partial z_j^{(l)}} \\
= & \sum_k \delta_k^{(l+1)} w_{j,k}^{(l+1)} \phi '(z_j^{(l)})
\end{align*}
$$

The first line sums over the neurons $$k$$ of layer $$l+1$$, all of which receive the output of neuron $$j$$. The second line writes $$z^{(l+1)}$$ with its definition, and the third keeps the only term of $$z_k^{(l+1)}$$ that depends on $$z_j^{(l)}$$, the weight $$w_{j,k}^{(l+1)}$$ times $$x_j^{(l)}$$. In matrix form, $$\delta^{(l)} = \left( W^{(l+1)} \delta^{(l+1)} \right) \odot \phi'(z^{(l)})$$, where $$\odot$$ is the element-wise product: the errors of layer $$l+1$$ are sent back through the same weights, and scaled by the slope of the activation function. The derivatives of equation \ref{eq_gradient} then follow from equations \ref{eq_z} and \ref{eq_delta}, since $$z_k^{(l)}$$ depends on $$w_{i,j}^{(l)}$$ and $$b_j^{(l)}$$ only for $$k = j$$, with the derivatives $$x_i^{(l-1)}$$ and $$1$$:

$$
\frac{\partial \mathcal{L}_n}{\partial w_{i,j}^{(l)}} = \sum_k \frac{\partial \mathcal{L}_n}{\partial z_k^{(l)}} \frac{\partial z_k^{(l)}}{\partial w_{i,j}^{(l)}} = \delta_j^{(l)} x_i^{(l-1)}
\label{eq_w}
$$

$$
\frac{\partial \mathcal{L}_n}{\partial b_{j}^{(l)}} = \sum_k \frac{\partial \mathcal{L}_n}{\partial z_k^{(l)}} \frac{\partial z_k^{(l)}}{\partial b_{j}^{(l)}} = \delta_j^{(l)}
\label{eq_b}
$$

In matrix form, the gradient of $$W^{(l)}$$ is the outer product $$x^{(l-1)} (\delta^{(l)})^T$$. The complete algorithm is:
1. **Forward pass**: set $$x^{(0)} = x_n$$, and compute $$z^{(l)}$$ and $$x^{(l)}$$ for $$l=1, ..., L+1$$;
2. **Backward pass**: compute $$\delta^{(L+1)}$$ from the loss and the activation of the output layer, e.g. $$\delta^{(L+1)} = -2 (y_n - z^{(L+1)})$$ for the squared error and a linear output. Then compute $$\delta^{(l)}$$ for $$l = L, ..., 1$$;
3. **Gradients**: compute the derivatives of equations \ref{eq_w} and \ref{eq_b} for all parameters.

What does a training step cost? For a mini-batch, the forward pass of a layer is one matrix product, and its backward pass two: $$W^{(l)} \delta^{(l)}$$ for the previous layer, and the weight gradients. The backward pass thus costs about twice the forward pass, and a training step about three forward passes. The backward pass also needs the $$z^{(l)}$$ and $$x^{(l)}$$ of all layers, so training stores the activations of all layers for all samples of the mini-batch, while inference keeps only those of the current layer. This is why inference needs much less memory than training.

## Automatic Differentiation

The backpropagation algorithm above needs the derivatives of the loss and of the activation functions, written by hand for each network. In practice, <a href="https://en.wikipedia.org/wiki/Automatic_differentiation">automatic differentiation ("autodiff")</a> removes this work, and backpropagation is a special case of its reverse mode. Autodiff frameworks, such as <a href="https://pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html">PyTorch's autograd</a>, define the derivative of each primitive operation once, e.g. of a matrix product or an exponential, and apply the chain rule to any composition of these operations.

Autodiff has two modes, which apply the chain rule in opposite directions. Forward-mode autodiff propagates derivatives from the inputs to the outputs, together with the forward computation. One pass gives the derivatives of all outputs with respect to one input, so the gradient with respect to $$P$$ parameters needs $$P$$ passes. Reverse mode propagates derivatives from the outputs to the inputs: one backward pass gives the derivatives of one output, such as the loss, with respect to all inputs. In exchange, reverse mode stores the intermediate values of the forward pass, which forward mode does not need. Training uses reverse mode, since the loss is a scalar and the parameters are many.

PyTorch builds the computational graph, a directed acyclic graph (DAG) of the operations, while the `forward` methods run: each operation adds a node, whose `grad_fn` computes its derivative. Loops and conditionals in the code may add different nodes at each `forward` call, so the graph cannot be extracted from the model definition alone (see an example [here](https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html)). `tensor.backward()` traverses the graph from the loss to the leaf nodes, and accumulates the gradients of the leaf tensors in their `.grad` fields. It then frees the graph, unless `retain_graph=True` is passed to `backward`, and the next `forward` call builds a new graph. <a href="https://towardsdatascience.com/automatic-differentiation-explained-b4ba8e60c2ad">Chi-Feng Wang's post</a> gives an example of an autodiff graph, and the <a href="{{ site.assets }}/resources/princeton_course_autodiff.pdf">Princeton course COS 324 on automatic differentiation</a> explains the whole process.

The stored activations can take most of the memory of training. Activation checkpointing reduces it: it stores only the inputs of selected modules, and recomputes their activations during the backward pass. Checkpointing every $$\sqrt{L}$$ layers of an $$L$$-layer network reduces the memory of the activations from $$O(L)$$ to $$O(\sqrt{L})$$, for the cost of one extra forward pass ([Chen et al., 2016](https://arxiv.org/abs/1604.06174)): the backward pass then holds the $$\sqrt{L}$$ checkpoints and the recomputed activations of one segment of $$\sqrt{L}$$ layers at a time. The post [How Activation Checkpointing enables scaling up training deep learning models](https://medium.com/pytorch/how-activation-checkpointing-enables-scaling-up-training-deep-learning-models-7a93ae01ff2d) shows the reentrant and non-reentrant implementations of PyTorch. Step-by-step examples of backpropagation in PyTorch are in [Hooks for autograd saved tensors](https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html) and [PyTorch Autograd: Backpropagation](https://colab.research.google.com/drive/1aWNdmYt7RcHMbUk-Xz2Cv5-cGFSWPXe0#scrollTo=LfW6wYg42SEW).

## Dropout

Large networks have many parameters, and tend to overfit their training data. [Dropout](https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5) reduces overfitting with a simple idea: it drops each neuron of the selected layers with probability $$p$$ during training. Each training sample uses a new random mask, i.e. a new *thinned* network. The kept outputs are scaled by $$1/(1-p)$$, which keeps their expected value unchanged: an output $$x$$ is kept with probability $$1-p$$ and then becomes $$x/(1-p)$$, so its expected value is $$x$$. Inference therefore uses all neurons, without dropout and without rescaling. The original paper ([Srivastava et al., 2014](https://jmlr.org/papers/v15/srivastava14a.html)) instead multiplies the outgoing weights by the retention probability $$1-p$$ at test time, which gives the same expected activations. A simple implementation, with the drop probability $$p$$ in `self.p`, is:

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

Why does this reduce overfitting? The network cannot rely on any single neuron, since any neuron may be dropped, so it learns redundant representations, spread over several neurons. It also makes the activations sparse, without a sparsity penalty such as an L1 norm. Dropout also trains many networks at once: a network with $$n$$ droppable neurons has $$2^n$$ thinned networks, which share their weights, and the full network at inference approximates the average of their predictions, at the cost of training and running a single network instead of many. The original paper adds that max-norm regularization, a large decaying learning rate and a high momentum improve the results further, and drops typically $$20\%$$ of the real-valued inputs and $$20\%$$ to $$50\%$$ of the hidden neurons. Dropout has a cost too: training takes 2 to 3 times longer than without it, because the parameter updates are noisy. In CNNs such as AlexNet and VGG, the fully connected layers hold most of the parameters, and are the most prone to overfitting, so dropout is applied to them.

{: .picture}
<img width="54.8%" height="54.8%" src="{{ site.assets }}/deep-neural-networks/dropout.svg">

{: .caption}
Dropout. Left: a standard network with two hidden layers ($$p = 0$$). Right: a thinned network produced by dropout with $$p = 0.5$$, where $$p$$ is the probability of dropping a unit: the crossed units are dropped, with their connections. Adapted from: [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html)

## Convolutional Neural Networks

A fully connected layer connects every input to every output, so large networks have many parameters, and need a large amount of data to be trained. Audio and images, however, have a local structure, in which nearby samples or pixels are strongly correlated, so local connectivity should suffice. For example, an audio stream $$x^{(0)}[n]$$ is commonly processed by a linear time-invariant filter $$h$$, whose output is the [convolution](https://en.wikipedia.org/wiki/Convolution) of the input and the filter:

$$
x^{(1)}[n] = \sum_u h[u] \, x^{(0)} [n-u]
$$

The filter is local: $$h[u]=0$$ for $$\vert u \vert > r$$, for a small radius $$r$$. Different filters extract different features: an averaging filter smooths the signal, and a high-pass filter enhances the differences between neighbouring samples. The same holds for an image, with a 2D filter:

$$
x^{(1)}[n,m] = \sum_{u,v} h[u,v] \, x^{(0)} [n-u,m-v]
$$

The output at position $$[n, m]$$ depends only on the input at the positions close to $$[n, m]$$, so a locally connected layer suffices. Moreover, the same filter, i.e. the same weights, applies at every position, since a feature such as an edge looks the same wherever it appears in the image. This **weight sharing** reduces the number of parameters further. For a $$256 \times 256$$ image and an output of the same size, a fully connected layer has $$256^4 \approx 4.3 \times 10^9$$ weights, a locally connected layer with $$3 \times 3$$ windows has $$256^2 \times 9 \approx 5.9 \times 10^5$$, and a convolution with a $$3 \times 3$$ filter has $$9$$. The picture compares local (top) and full (bottom) connectivity:

{: .picture}
<img width="41.2%" height="41.2%" src="{{ site.assets }}/deep-neural-networks/fully_sparse_connectivity.svg">

{: .caption}
Sparse and full connectivity, viewed from below. Top: a locally connected layer with a kernel of width 3, where the input $$x_3$$ affects only the outputs $$s_2$$, $$s_3$$ and $$s_4$$. Bottom: a fully connected layer, where $$x_3$$ affects every output. Adapted from: [Deep Learning, I. Goodfellow, Y. Bengio and A. Courville, Figure 9.2](https://www.deeplearningbook.org/contents/convnets.html)

A single filter extracts a single feature, so a convolutional layer applies several filters to the same input, and each filter produces one output *channel*. For images, the input and the output of a layer are thus 3D: height, width and channels. A layer with $$C_{in}$$ input channels, $$C_{out}$$ output channels and filters of $$(2r+1) \times (2r+1)$$ has $$C_{in} C_{out} (2r+1)^2$$ weights, since each output channel combines all input channels, and computes as many multiply-accumulate operations per output position. Each weight is reused at every output position, which gives convolutions with many channels a high [arithmetic intensity](https://en.wikipedia.org/wiki/Roofline_model#Arithmetic_intensity), i.e. many operations per byte moved from memory. Deeper layers typically have more channels, and a smaller height and width. Stacking such layers gives a convolutional neural network (CNN), with the following structure for images:

{: .picture}
<img width="99.6%" height="99.6%" src="{{ site.assets }}/deep-neural-networks/cnn.svg">

{: .caption}
A CNN that classifies $$16 \times 16$$ RGB patches of a satellite image as road or non-road: two stages of convolution and pooling, two fully connected layers, and the output probabilities of the two classes. Each label gives the shape of the data after a layer for $$N$$ input images, with the number of channels or units in brackets.

In this example, each convolution keeps the height and width and increases the number of channels, from 3 to 32 and from 32 to 64, and each pooling halves the height and width. The fully connected layers then map the $$4 \times 4 \times 64 = 1024$$ features of each patch to 512 units, and to the 2 class probabilities. A CNN has three types of layers:
- convolutional layers apply the filters. Each output depends only on a small window of the input, its *receptive field*, as a neuron of the visual cortex responds only to a small region of the visual field. Deep learning libraries compute a [cross-correlation](https://en.wikipedia.org/wiki/Cross-correlation), $$\sum_{u} h[u] \, x[n+u]$$, which is a convolution with the flipped filter. The filters are learned, so the two are equivalent;
- pooling layers reduce each window of neighbouring outputs, e.g. of $$2 \times 2$$, to one value, e.g. the maximum (*max pooling*) or the average, which makes the features less sensitive to small shifts of the input;
- fully connected layers, an MLP at the end, map the features to the output, e.g. to the class scores.

The height and width decrease through the layers, by pooling, by strides larger than 1, or by convolutions without padding at the boundaries. Training uses backpropagation as before, treating each connection as if it had its own weight. The gradient of a shared weight is then the sum of the gradients of all connections that share it, since the loss depends on the shared weight through each of these connections.

## Skip connections

A deeper network should reach a training error at least as low as a shallower one, since its extra layers could learn the identity and copy the shallower network. In practice, however, deep *plain* networks, without skip connections, train worse as their depth grows: beyond some depth, both the training and the test errors increase. This *degradation problem* is not overfitting, since the training error increases too. Vanishing and exploding gradients are natural suspects, and normalized initialization and [normalization](https://arxiv.org/abs/1803.08494) layers largely address them, but not the degradation: the authors of [ResNet](https://arxiv.org/abs/1512.03385) (He et al., 2015) measured healthy gradient norms in degraded networks with batch normalization, and attribute the degradation to the difficulty of optimizing deep plain networks.

*Skip connections* solve this problem by feeding the output of a layer to later layers that are not directly connected to it. A residual block of ResNet computes $$y = F(x) + x$$, where $$F$$ is a stack of two or three layers. The block thus learns only the residual $$F(x) = y - x$$: if the identity is the best mapping, the block only has to push $$F$$ towards zero, which is easier than learning the identity with a stack of non-linear layers. The gradient also flows to the earlier layers through the identity path, since the derivative of $$y$$ with respect to $$x$$ includes the identity. [DenseNet](https://arxiv.org/abs/1608.06993) instead concatenates the outputs of all previous layers of a block as the input of each layer. Addition keeps the number of channels constant, while concatenation increases it with every layer, which costs memory, but gives each layer direct access to all earlier features. In the figure, each layer adds 4 feature maps, so the input of layer $$H_i$$ holds the feature maps of $$x_0$$ plus $$4(i-1)$$ more:

{: .picture}
<img width="82%" height="82%" src="{{ site.assets }}/deep-neural-networks/densenet.svg">

{: .caption}
A 5-layer dense block of DenseNet with a growth rate of $$k = 4$$: each layer $$H_i$$ (BN-ReLU-Conv) takes the feature maps $$x_0, \ldots, x_{i-1}$$ of all preceding layers as input and produces $$k$$ new feature maps $$x_i$$. The transition layer takes all of them. Adapted from: [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

Many CNNs use other layouts of skip connections. [U-net](https://arxiv.org/abs/1505.04597), for example, concatenates the features of each downsampling stage to the upsampling stage of the same resolution, which brings the fine details of the input back to the output of a segmentation network.

## Embeddings

An *embedding* is a dense, low-dimensional vector that represents a data item, e.g. a word, an image or a user, in a continuous vector space, such that similar items have close vectors. Other models can then compare, search and cluster the items with vector operations, such as dot products. A common way to obtain an embedding is to pass the input through a *trained* network, and to take the output of an intermediate layer, typically one before the output layer. Common embeddings per data type are:
- words: [Word2vec](https://en.wikipedia.org/wiki/Word2vec) learns one vector per word of the vocabulary, with one of two models ([Mikolov et al., 2013](https://arxiv.org/abs/1301.3781)). *Continuous bag-of-words* (CBOW) predicts a word from its neighbours, and *skip-gram* predicts the neighbours from the word. In the original paper, CBOW trained in about a day and skip-gram in about three days. Skip-gram is more accurate on semantic analogies, e.g. Athens : Greece = Oslo : Norway, and CBOW slightly more accurate on syntactic analogies, e.g. great : greater = tough : tougher. One-hot vectors, the usual input of these models, are not embeddings: their size is the size of the vocabulary, and all pairs of words are at the same distance;
- sentences and documents: the outputs of a Transformer encoder such as [BERT](https://en.wikipedia.org/wiki/BERT_(language_model)), e.g. at its `[CLS]` token, or averaged over all tokens;
- other sequences: the last hidden state of a recurrent encoder, e.g. an LSTM;
- vectors: their coordinates along the first principal components, from [principal component analysis]({{ site.baseurl }}{% post_url 2017-11-01-unsupervised-learning %}) (PCA);
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
