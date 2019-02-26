---

layout: post
title:  "Supervised Learning: an overview of methods"
date:   2018-06-17
categories: [machine learning, supervised learning]
tags: [machinelearning]
---

<div class="alert alert-primary" role="alert">
Unlike most posts that provide a thorough study on a particular subject, this post provides a very high level description of several supervised learning methods that I came across. Its content may not be entirely correct and is continuously updated.
</div>

Supervised Learning is the field of machine learning that learns through supervision, or in practice, learns with the help of an external agent (human or automatic) that provides the solution for a given training set, in order to *provide an approximator* for the mechanics that relates the given input to the outputs (labels).

### Foreword: Regression and Basic Stuff

Regression relates an input variable to an output, to either predict new outputs, or understand the effect of the input in the output. A regression dataset consists of a set of pairs $(x_n, y_n)$ of size $N$ with input $x_n$ and output/label $y_n$. For a new input $x_n$, the goal of regression is to find $f$ such that $$ y_n \approx f(x_n) $$. It does so by finding **weights** $w$ that approximate:
- linear regression: $$ y_n \approx f(x_n) = w_0 + w_1 x_{n1} $$
- multilinear regression: $$ y_n \approx f(x_n) = w_0 + w_1 x_{n1} + ... + + w_D x_{nD} $$

If $$ D \gt N $$ is called the problem is [under-determined]().

##### Convexity

A function is **convex** if a line joining two points never intersects with the function anywhere else, ie:

$$
f (\lambda u + (1-\lambda) v) \le \lambda f(u) + (1-\lambda)f(v)
$$

 A convex function has only one global minimum value. A strictly convex function has a unique global minimum, ie the inequality $\le$ is strict ($\lt$). Sums of convex functions are also convex. An alternative definition of convexity (*for differentiable functions*) is that the function must always lie above its [linearization](https://en.wikipedia.org/wiki/Linearization), i.e.

$$
L(u) \ge L(w) + \triangledown L(w)^T (u-w) \text{, for all } u,w
$$

A set $C$ is convex the line segment between any two points of $C$ lies in $C$, i'e does not touch outside the limits of $C$. Here's a picture of a convex and a non-convex set:

<p align="center">
<img width="30%" height="30%" src="/assets/2018-Supervised-Learning/convex-set.jpg"><br/>
<small>source: britannica</small>
</p>

Intersections of convex sets are convex. 

##### Loss

The quality of the approximation is provided by the **loss** or **cost function**. Examples:
- Mean Square Error: $$ MSE(w) = \frac{1}{N} \sum_{n=1}^N [ y_n - f(x_n)]^2 $$ , not good for outliers
- Mean Absolute Error: $$ MAE(w) = \frac{1}{N} \sum_{n=1}^N \mid y_n - f(x_n) \mid $$ 
- Huber: $$ Huber(w) = 
\begin{cases}
    \frac{1}{2}e^2 & \text{, if } \mid e \mid \le \delta\\
    \delta \mid e \mid - \frac{1}{2}\delta^2 & \text{, if } \mid e \mid \gt \delta
\end{cases}
$$
  - convex, differentiable, and also robust to outliers. The hard bit is to set $\delta$
- Tukey’s bisquare loss (de⇢fined in terms of the gradient): $$ \frac{\partial L}{\partial e} =
\begin{cases}
    e (1-\frac{e^2}{\delta^2})^2 & \text{, if } \mid e \mid \le \delta\\
    0 & \text{, if } \mid e \mid \gt \delta\\
\end{cases}
$$
  - non-convex, but robust to outliers;

For personal amusement, in this [website](https://lossfunctions.tumblr.com/) we find a funny compilation of loss functions gone *wrong*.

##### Optimization

Given a cost function $L(w)$ we want to find the weights that mimimizes the cost, via:
- Grid Search (brute-force);
- Gradient Descent: $$ w^{t+1} = w^{t} - \gamma \triangledown L (w^t) $$, for step size $\gamma$, and gradient $$ \triangledown L (w) = [ \frac{\partial L(w)}{\partial w_1}, ... , \frac{\partial L(w)}{\partial w_D}  ]^T $$;
- Stochastic Gradient Descent: $$ w^{t+1} = w^{t} - \gamma \triangledown L_n (w^t) $$, for a random choice of inputs $n$. Computationally cheap but unbiased estimate of gradient;
- Mini-batch SGD: $$ w^{t+1} = w^{t} - \gamma \frac{1}{\mid B \mid} \sum_{n \in B} \triangledown L_n (w^t) $$, for a random subset $ B \subseteq [N] $. For each sample $n$ , we compute the gradient at the same current point $w^{(t)}$;
- Subgradient Descent: $$ w^{t+1} = w^{t} - \gamma g $$, where $g$ is a [subgradient](https://en.wikipedia.org/wiki/Subderivative#The_subgradient) of $L$ (useful when function is not differentiable at $w^{(t)}$; 
- Projected Gradient: when we solve a constrained optimization problem $min_w L(w)$ subject to $w \in C$, where $C \subset R^D$ is the constraint set;
  - Note: Projections onto convex sets are unique;

