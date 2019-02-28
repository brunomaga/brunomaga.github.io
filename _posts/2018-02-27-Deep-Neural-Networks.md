---

layout: post
title:  "Deep Neural Networks"
date:   2018-02-27
categories: [machine learning, supervised learning, deep neural networks]
tags: [machinelearning]
---

##### Overview 

We have seen that simple linear classification schemes like logistic regression:

$$
p (y | x^T w) = \frac{e^{x^Tw_y}}{1+e^{x^Tw_y}}
$$

can work well but are limited. Alternative approaches like binning work well only for low input dimensionality. That's where neural networks fill the gap. Neural networks allows us to learn not only the *weights* but also the *useful features*. In practice, they are what we call an **[universal approximator](https://en.wikipedia.org/wiki/Universal_approximation_theorem)**, i.e. an approximator to any function in a bounded continuous domain. I will omit the proof, email me if you are interested in knowing more.

##### Structure

The structure of a simple NN is the following: one input layer of size $D$, $L$ hidden layers of size $K$, and one output layer. It is a **feedfoward network**: the computation performed by the network starts with the input from the left and flows to the right. There is no feedback loop. It can be used for regression (when input and output are provided) and for classification (when input are provided and new output is the classifier).

<p align="center">
<img width="30%" height="30%" src="/assets/2018-Deep-Neural-Networks/dnn.png"><br/>
<small>source: Machine Learning lecture notes, M Jaggi, EPFL</small>
</p>

Input and output layers work similarly to any previous regression method. Hidden layers are different. The output at the node $j$ in hidden layer $l$ is denoted by:

$$
x_j^{(l)} = \phi ( \sum_i w_{i,j}^{(l)} x_i^{(l-1)}+b_j^{(l)}). 
$$

where $b$ is the bias term, trained equally to any other weight. Note that, in order to compute the output we first compute the weighted sum of the inputs and then apply an **activation function** $\phi$ to this sum. We can equally describe the function that is implemented by each layer in the form:

$$
x^{(l)} = f^{(l)} (x^{(l-1)}) = \phi ((W^{(l)})^T x^{(l-1)}+b^{(l)})
$$

where $f$ is the regression function of neurons, as previously. Some popular choices of activation functions are:
- [sigmoid function](https://www.wolframalpha.com/input/?i=1%2F(1%2Be%5E%7B-x%7D)): $$ \phi(x) = \frac{1}{1+e^{-x}} $$, positive and bounded;
- [*tanh*](https://www.wolframalpha.com/input/?i=%5Cfrac%7Be%5Ex+-+e%5E%7B-x%7D%7D%7Be%5Ex%2Be%5E%7B-x%7D%7D) = $$ \phi(x) = \frac{e^x - e^{-x}}{e^x+e^{-x}} $$, balanced (positive and negative) and  bounded;
- [Rectified linear Unit – ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)): $$\phi(x) = [x]_+ = maxo\{0,x\}$$, always positive and unbounded; often used as it's computationally *cheap*; 
- Leaky ReLU: $$\phi(x) = max{\alpha x,x}$$. Fixes the problem of *dying* in Non-leaky rectified linear units;
  - In brief, if the dot product of the input to a ReLU with its weights is negative, the output is 0. The gradient of $$ max\{ 0,x \} $$ is $0$ when the output is $0$. If for any reason the output is consistently $0$ (for example, if the ReLU has a large negative bias), then the gradient will be consistently 0. The error signal backpropagated from later layers gets multiplied by this 0, so no error signal ever passes to earlier layers. The ReLU has died.  With leaky ReLUs, the gradient is never 0, and this problem is avoided.  

##### Back-propagation

Similarly to previous regression use cases, we are required to minimize the loss in the system, using e.g. Stochastig Gradient Descent. As always when dealing with gradient descent we compute the gradient of the cost function for a particular input sample (with respect to all weights of the net and all bias terms) and then we take a small step in the direction opposite to this gradient. Computing the derivative with respect to a particular parameter is really just applying the chain rule of calculus. The cost function can be written as (details omitted):

$$
L = \frac{1}{N} \sum_{n=1}^N ( y_n - f^{(L+1)} \circ ... \circ f^{(2)} \circ f^{(1)} (x_n^{(0)}) ) ^2
$$

Since in general there are many parameters it would not be efficient to do this for each parameter individually. However, the algorithm of **back propagation** alows us to do it more efficently. Our aim is to compute:

$$
\frac{\partial L_n}{\partial w_{i,j}^{(l)}}, l=1, ..., L+1 \text {, and } \frac{\partial L_n}{\partial b_{j}^{(l)}}, l=1, ..., L+1 
\label{eq_gradient}
$$

For simplicity, we will represent the total input computed at the $l$-th layer as

$$
z^{(l)} = (W^{(l)})^T x^{(l-1)}+b^{(l)} .
\label{eq_z}
$$

These quantities are easy to compute by a forward pass in the network, starting at $$ x^{(0)} = x_n $$ and then applying this recursion for $$ l=1,...,L+1 $$.

Now, let

$$
\delta_j^{(l)} = \frac{\delta L_n}{\delta z_j^{(l)}}.
\label{eq_delta}
$$

These quantities are easier to compute with a backward pass:

$$
\delta_j^{(l)} = \frac{\delta L_n}{\delta z_j^{(l)}} = \sum \frac{\partial L_n}{\partial z_k^{(l+1)}} \frac{\partial z_k^{(l+1)}}{\delta z_j^{(l)}} = \sum_k \delta_k^{(l+1)} W_{j,k}^{(l+1)} \phi '(z_j^{(l)})
$$

We used the chain rule on the second equality operation. Going back to equation \ref{eq_gradient}:

$$
\frac{\partial L_n}{\partial w_{i,j}^{(l)}} = \sum \frac{\partial L_n}{\partial z_k^{(l)}} \frac{\partial z_k^{(l)}}{\partial w_{i,j}^{(l)}} = \delta_j^{(l)} x_j^{(l-1)}
\label{eq_w}
$$

Here we used chain rule, equation \ref{eq_delta} on the definition of $\delta$, and the partial derivative of equation \ref{eq_z} over $w$. In a similar manner:

$$
\frac{\partial L_n}{\partial b_{j}^{(l)}} = \sum \frac{\partial L_n}{\partial z_k^{(l)}} \frac{\partial z_k^{(l)}}{\partial b_{j}^{(l)}} = \delta_j^{(l)} \cdot 1 = \delta_j^{(l)}
\label{eq_b}
$$

The complete back propagation is then summarized as:
- Forward pass: set $$ x^{(0)} = x_n $$. Compute $$ z^{(l)} $$ for $$ l=1,...,l+1 $$;
- Backward pass: set $$ \delta ^{(L+1)} $$ using the appropriate loss and activation function. Compute $$ \delta ^{(L+1)} $$ for $$ l = L, ..., 1 $$;
- Final computation: For all parameters compute $$ \frac{\partial L_n}{\partial w_{i,j}^{(l)}} $$ (eq. \ref{eq_w}) and $$ \frac{\partial L_n}{\partial b_{j}^{(l)}} $$ (eq. \ref{eq_b});

##### Regularization via Dropout

[Dropout](https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5) is a method to *drop out* (ignore) neurons in neural network  and retrieving the final model as an average of models. The rationale is that the processing of the same problem in different neural networks prevents complex co-adaptations on training data. It performs model averaging, and reduces overfitting.

From [wikipedia](https://en.wikipedia.org/wiki/Convolutional_neural_network#Dropout):  "Because a fully connected layer occupies most of the parameters, it is prone to overfitting. [...] At each training stage, individual nodes are either *dropped out* of the net with probability $$ 1-p $$ or kept with probability $p$, so that a reduced network is left; incoming and outgoing edges to a dropped-out node are also removed. Only the reduced network is trained on the data in that stage. The removed nodes are then reinserted into the network with their original weights. "

<p align="center">
<img width="35%" height="35%" src="/assets/2018-Deep-Neural-Networks/dropout.png">
</p>

##### Convolutional Neural Networks
 
<div class="alert alert-warning" role="alert">
15.02.2019: I will update this chapter soon
</div>
