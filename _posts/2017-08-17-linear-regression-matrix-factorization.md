---
layout: post
title:  "Linear Regression, Matrix Factorization and Loss Functions"
categories: ["Machine Learning"]
tags: ["supervised learning", "linear regression", "matrix factorization", "loss functions"]
---

Regression relates input variables to an output, either to predict new outputs or to understand the effect of the inputs on the output. A regression dataset holds $$N$$ pairs $$(x_n, y_n)$$, each made of an input $$x_n \in \mathbb{R}^D$$ and an output, or label, $$y_n \in \mathbb{R}$$, and regression looks for a function $$f$$ such that $$y_n \approx f(x_n)$$. Fitting a hyperplane to the data is the problem of [linear regression][linear-regression], which finds the weights $$w = (w_0, w_1, \ldots, w_D)^T$$ such that:

$$
y_n \approx f(x_n) = w_0 + w_1 x_{n1} + \ldots + w_D x_{nD} \text{, or, for all samples, } y \approx Xw,
$$

where $$y = (y_1, \ldots, y_N)^T$$, and row $$n$$ of the $$N \times (D+1)$$ matrix $$X$$ is $$(1, x_{n1}, \ldots, x_{nD})$$. The column of ones multiplies the bias $$w_0$$, so that the product $$Xw$$ includes it.

[linear-regression]: https://en.wikipedia.org/wiki/Linear_regression

A loss function $$L(w)$$ measures how well the weights fit the data. The [mean squared error][mse] (MSE), $$L(w) = \frac{1}{N} \sum_{n=1}^N (y_n - f(x_n))^2$$, squares the errors, so it penalizes large errors much more than small ones. The [mean absolute error][mae] (MAE), $$L(w) = \frac{1}{N} \sum_{n=1}^N \vert y_n - f(x_n) \vert$$, grows only linearly with the error, which makes it more robust to outliers.

[mse]: https://en.wikipedia.org/wiki/Mean_squared_error
[mae]: https://en.wikipedia.org/wiki/Mean_absolute_error

Training looks for the weights that minimize the loss. A naive approach is grid search, a brute-force method that evaluates the loss on a grid of weights. With $$M$$ values per weight, it needs $$M^{D+1}$$ evaluations, so it only works for problems with a few dimensions. Gradient methods scale much better: they move $$w$$ in small steps along the negative gradient of the loss, the direction in which the loss decreases fastest, with a step size, or learning rate, $$\eta$$. The loss is a mean over the samples, $$L(w) = \frac{1}{N} \sum_{n=1}^N L_n(w)$$, where $$L_n$$ is the loss of sample $$n$$, and the three main methods differ in the number of samples that each step reads:
- [gradient descent][gd]: $$w^{t+1} = w^t - \eta \nabla L(w^t)$$, with the gradient $$\nabla L(w) = \left[ \frac{\partial L}{\partial w_0}, \ldots, \frac{\partial L}{\partial w_D} \right]^T$$. Each step reads all $$N$$ samples, at a cost of $$O(ND)$$ operations for linear regression;
- [stochastic gradient descent][sgd] (SGD): $$w^{t+1} = w^t - \eta \nabla L_n(w^t)$$, for a sample $$n$$ drawn at random. A step costs only $$O(D)$$, and $$\nabla L_n$$ is an unbiased estimate of $$\nabla L$$. The estimate is noisy, however, so SGD needs a decreasing step size to converge;
- mini-batch SGD: $$w^{t+1} = w^t - \frac{\eta}{\vert B \vert} \sum_{n \in B} \nabla L_n(w^t)$$, for a random subset $$B \subseteq \{1, \ldots, N\}$$, with the gradients of all samples computed at the same point $$w^t$$. A step costs $$O(\vert B \vert D)$$. For samples drawn independently, the variance of the estimate falls as $$1 / \vert B \vert$$, and the batch runs as one vectorized operation, so mini-batches combine the stability of gradient descent with the low cost per step of SGD.

[gd]: https://en.wikipedia.org/wiki/Gradient_descent
[sgd]: https://en.wikipedia.org/wiki/Stochastic_gradient_descent

For linear regression with the MSE, no iterations are needed: this is the [least squares][least-squares] problem, which has a closed-form solution. The factor $$1/N$$ does not move the minimum, so it can be dropped, and the loss expands as:

  $$
  \begin{align*}
  \| y - Xw \|^2 = \, & (y-Xw)^T(y-Xw) \\
  = \, & y^Ty - y^TXw - (Xw)^Ty + (Xw)^TXw \\
  = \, & y^Ty - w^TX^Ty - w^TX^Ty + w^TX^TXw \\
  = \, & y^Ty - 2 w^TX^Ty + w^TX^TXw\\
  \end{align*}
  $$

[least-squares]: https://en.wikipedia.org/wiki/Least_squares

This is a quadratic function of $$w$$, and its Hessian, $$2X^TX$$, is positive semidefinite, since $$v^TX^TXv = \Vert Xv \Vert^2 \geq 0$$ for every vector $$v$$. The loss is therefore convex, and every $$w$$ at which its derivative is zero is a minimum:

  $$
  \begin{align*}
    & \frac{\partial}{\partial w} \left( y^Ty - 2 w^TX^Ty + w^TX^TXw \right) = 0 \\
  \Leftrightarrow \, & 0 - 2 X^Ty + 2 X^TXw = 0 \\
  \Leftrightarrow \, & 2 X^TXw = 2X^Ty \\
  \Leftrightarrow \, & w = (X^TX)^{-1} X^Ty\\
  \end{align*}
  $$

The derivation uses $$\frac{\partial w^Ta}{\partial w} = \frac{\partial a^Tw}{\partial w} = a$$ and $$\frac{\partial w^TAw}{\partial w} = (A + A^T)w$$, which is $$2Aw$$ for a symmetric matrix $$A$$ such as $$X^TX$$ (<a href="{{ site.assets }}/resources/the_matrix_cookbook.pdf">The Matrix Cookbook</a>, equations 69 and 81). The solution is unique if the [Gram matrix][gram] $$X^TX$$ is invertible. This holds if and only if $$X$$ has full column [rank][rank], $$\text{rank}(X) = D+1$$, i.e. if its columns are linearly independent, which requires at least as many samples as weights, $$N \geq D+1$$. The identity above shows why: if $$X^TXv = 0$$, then $$\Vert Xv \Vert^2 = v^TX^TXv = 0$$, so $$X^TX$$ and $$X$$ have the same null space, which holds only $$v = 0$$ when the columns of $$X$$ are linearly independent.

[gram]: https://en.wikipedia.org/wiki/Gram_matrix
[rank]: https://en.wikipedia.org/wiki/Rank_(linear_algebra)

How expensive is the closed form? Forming $$X^TX$$ costs $$O(ND^2)$$ operations and $$O(D^2)$$ memory, and solving the $$(D+1) \times (D+1)$$ system costs $$O(D^3)$$. Solvers do not compute the inverse: they factorize $$X^TX$$ with a [Cholesky decomposition][cholesky], or $$X$$ with a [QR decomposition][qr]. For $$N \gg D$$, QR costs about twice as much, but it never forms $$X^TX$$, whose [condition number][condition-number] is the square of that of $$X$$, so it loses fewer digits to rounding. A gradient descent step, in turn, costs $$O(ND)$$ and needs only $$O(D)$$ memory besides the data, which makes iterative methods the better choice when $$D$$ is large.

[cholesky]: https://en.wikipedia.org/wiki/Cholesky_decomposition
[qr]: https://en.wikipedia.org/wiki/QR_decomposition
[condition-number]: https://en.wikipedia.org/wiki/Condition_number

### Regularization

A model with many weights can fit the noise of the training data, and then predict new data poorly. A common remedy for this [overfitting][overfitting] is to add to the loss a penalty on the weights, which favors simple models. A parameter $$\lambda > 0$$ sets the strength of the penalty, and choosing it is a **model selection** problem, usually solved with [cross-validation][cross-validation]. The two common penalties are the following, with the squared error written as a sum ($$\lambda$$ absorbs the factor $$1/N$$):
- L1: $$\min_w \Vert y - Xw \Vert^2 + \lambda \Vert w \Vert_1$$, with $$\Vert w \Vert_1 = \sum_{i=1}^D \vert w_i \vert$$, so the bias $$w_0$$ is not penalized. This is the [LASSO][lasso], for **L**east **A**bsolute **S**hrinkage and **S**election **O**perator. It encourages sparse models, with many weights equal to 0, so it also selects the input variables, which automates a part of model selection. It has no closed-form solution in general, but coordinate descent, below, solves it. With highly correlated inputs ([multicollinearity][multicollinearity]), the LASSO tends to keep one of them, chosen arbitrarily; ridge regression, or the [elastic net][elastic-net], which combines the L1 and L2 penalties, is then the better choice.
- L2: $$\min_w \Vert y - Xw \Vert^2 + \lambda \Vert w \Vert_2^2$$, with $$\Vert w \Vert_2^2 = \sum_{i=1}^D w_i^2$$. Large weights are penalized, as they are considered unlikely. With the MSE, this is [ridge regression][ridge], which keeps a closed form: setting the gradient to zero gives $$w = (X^TX + \lambda I)^{-1} X^Ty$$. The matrix $$X^TX + \lambda I$$ is positive definite for $$\lambda > 0$$, so the solution exists and is unique even when $$X$$ does not have full column rank. This form also penalizes $$w_0$$; excluding the bias sets the first diagonal entry of $$I$$ to 0.

[overfitting]: https://en.wikipedia.org/wiki/Overfitting
[cross-validation]: https://en.wikipedia.org/wiki/Cross-validation_(statistics)
[lasso]: https://en.wikipedia.org/wiki/Lasso_(statistics)
[multicollinearity]: https://en.wikipedia.org/wiki/Multicollinearity
[elastic-net]: https://en.wikipedia.org/wiki/Elastic_net_regularization
[ridge]: https://en.wikipedia.org/wiki/Ridge_regression

Both penalties shrink the weights toward 0. In statistics, this [shrinkage][shrinkage] reduces the effect of the sampling variation on the estimate, which varies less from one training set to another, at the cost of a bias. They shrink in different ways, which is easiest to see when the columns of $$X$$ are orthonormal, $$X^TX = I$$, and the penalty covers all weights. With $$\hat{w} = X^Ty$$, the least-squares weights, the loss then splits, up to the constant $$y^Ty$$, into one term per weight: $$(1 + \lambda) w_i^2 - 2 w_i \hat{w}_i$$ for ridge regression, whose minimum divides $$\hat{w}_i$$ by $$1 + \lambda$$, and $$w_i^2 - 2 w_i \hat{w}_i + \lambda \vert w_i \vert$$ for the LASSO, whose minimum is:

$$
w_i = \text{sign}(\hat{w}_i) \max \left( \vert \hat{w}_i \vert - \frac{\lambda}{2}, 0 \right)
$$

In other words, ridge regression makes every weight smaller, but never 0, while the LASSO reduces the magnitude of each weight by $$\lambda / 2$$, and sets it to 0 when it is below $$\lambda / 2$$. This *soft thresholding* is the source of the sparsity of the LASSO.

[shrinkage]: https://en.wikipedia.org/wiki/Shrinkage_(statistics)

Neural networks also use two techniques that improve generalization or training without a penalty in the loss:
- [dropout](https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5): at each training step, each neuron is *dropped out*, i.e. its output is set to 0, with a fixed probability. Inference uses all neurons, with their outputs scaled by the probability of keeping them. Since any neuron may be missing, the network cannot rely on specific neurons, which improves generalization.
- [layer, instance, batch and group normalization](https://arxiv.org/abs/1803.08494): each method normalizes the activations of a layer to zero mean and unit variance, over a different set of values: each channel over the batch (batch normalization), all channels of a sample (layer normalization), each channel of a sample (instance normalization), or groups of channels of a sample (group normalization). A learned scale and shift per channel then set the standard deviation and the mean of the output. Normalization stabilizes training, and allows larger learning rates.

## Matrix Factorization

[Matrix factorization][mf] discovers the latent factors of a matrix, and predicts its missing entries. In a movie [recommender system][recommender], $$X$$ is the $$D \times N$$ matrix of the ratings of $$D$$ movies by $$N$$ users, of which most are missing. The factorization $$X \approx WZ^T$$ uses two tall matrices, $$W \in \mathbb{R}^{D \times K}$$ and $$Z \in \mathbb{R}^{N \times K}$$, with $$K$$ latent features, such as how much a movie is a comedy, and how much a user likes comedies. Row $$d$$ of $$W$$ holds the features of movie $$d$$, row $$n$$ of $$Z$$ those of user $$n$$, and their dot product $$(WZ^T)_{dn}$$ predicts the rating $$x_{dn}$$. [Koren et al. (2009)][koren2009] review the method, which outperformed the classic nearest-neighbor methods in the [Netflix Prize][netflix-prize]. Only the observed ratings, with indices in $$\Omega \subseteq \{1, \ldots, D\} \times \{1, \ldots, N\}$$, enter the loss:

[mf]: https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems)
[recommender]: https://en.wikipedia.org/wiki/Recommender_system
[koren2009]: {{ site.assets }}/linear-regression-matrix-factorization/recommender-systems-netflix.pdf
[netflix-prize]: https://en.wikipedia.org/wiki/Netflix_Prize

$$
\min_{W,Z} L (W,Z) = \frac{1}{2} \sum_{(d,n) \in \Omega} [ x_{dn} - (WZ^T)_{dn}]^2
$$

{: .picture}
<img width="44.1%" height="44.1%"  src="{{ site.assets }}/linear-regression-matrix-factorization/matrix-factorization.svg">

{: .caption}
Matrix factorization $$X \approx WZ^T$$ of the ratings of $$D = 7$$ movies by $$N = 7$$ users, with 11 observed ratings and $$K = 3$$ latent features. The rating $$x_{dn}$$ of movie $$d$$ by user $$n$$ is approximated by the dot product of row $$d$$ of $$W$$, the features of movie $$d$$, and column $$n$$ of $$Z^T$$, the features of user $$n$$.

In the example of the figure, the loss sums over the 11 observed ratings, and the product $$WZ^T$$ fills in the other 38 entries. This loss is harder to minimize than least squares. It is not convex in $$(W, Z)$$, since the prediction multiplies the two unknowns, and it is not [identifiable](https://en.wikipedia.org/wiki/Identifiability): for any invertible $$K \times K$$ matrix $$A$$, the factors $$WA$$ and $$ZA^{-T}$$ give the same product. A large $$K$$ also leads to overfitting, which a regularizer limits, as in ridge regression:

$$
\min_{W,Z} L (W,Z) = \frac{1}{2} \sum_{(d,n) \in \Omega} [ x_{dn} - (WZ^T)_{dn}]^2 + \frac{\lambda_w}{2} \| W \|_F^2 +  \frac{\lambda_z}{2} \| Z \|_F^2
$$

with $$\lambda_w, \lambda_z > 0$$ and the [Frobenius norm][frobenius] $$\Vert \cdot \Vert_F$$.

[frobenius]: https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm

Two methods minimize this loss. The first one, SGD, visits one rating at a time. Without the regularizer, the objective is a mean of $$\vert \Omega \vert$$ terms, one per rating:

$$
\frac{1}{\vert \Omega \vert} \sum_{(d,n) \in \Omega} \frac{1}{2} [x_{dn} - (WZ^T)_{dn}]^2
$$

For one rating $$(d,n)$$, the gradient entries for $$W$$ and $$Z$$ are:

$$
\frac{\partial}{\partial w_{d',k}} \frac{1}{2} [x_{dn} - (WZ^T)_{dn}]^2 =
\begin{cases}
    -[x_{dn} - (WZ^T)_{dn}] z_{n,k} & \text{, if } d'=d\\
    0              & \text{, otherwise}
\end{cases}
$$

and

$$
\frac{\partial}{\partial z_{n',k}} \frac{1}{2} [x_{dn} - (WZ^T)_{dn}]^2 =
\begin{cases}
    -[x_{dn} - (WZ^T)_{dn}] w_{d,k} & \text{, if } n'=n\\
    0              & \text{, otherwise}
\end{cases} 
$$

Recall that $$w_{d,k}$$ are the features of the movie and $$z_{n,k}$$ those of the user: a step moves each of the two rows along the other one, in proportion to the error of the prediction. The gradient has $$(D+N)K$$ entries, but only the $$2K$$ entries of row $$d$$ of $$W$$ and row $$n$$ of $$Z$$ are nonzero, so an SGD step costs $$O(K)$$, and a pass over all ratings $$O(\vert \Omega \vert K)$$.

The second method, **alternating least squares** (ALS), minimizes the same loss in turns: it fixes $$Z$$ and solves for $$W$$, then fixes $$W$$ and solves for $$Z$$, and repeats. The idea is that with one factor fixed, the loss is a regularized least-squares problem in the other, with a closed-form solution, so that no step increases the loss.

- Without missing ratings, $$\Omega = \{1, \ldots, D\} \times \{1, \ldots, N\}$$, and:

  $$
  \begin{align*}
  \min_{W,Z} L(W,Z) = & \frac{1}{2} \sum_{d=1}^D \sum_{n=1}^N [x_{dn} - (WZ^T)_{dn}]^2 + \text{ (regularizer) } \\
  = & \frac{1}{2} \| X - WZ^T \|_F^2 + \frac{\lambda_w}{2} \| W \|_F^2 +  \frac{\lambda_z}{2} \| Z \|_F^2
  \end{align*}
  $$

  Setting the gradient to zero gives $$W = XZ (Z^TZ + \lambda_w I_K)^{-1}$$ for a fixed $$Z$$, and $$Z = X^TW (W^TW + \lambda_z I_K)^{-1}$$ for a fixed $$W$$.
- With missing ratings, the problem is harder, because only the observed ratings enter the loss:

  $$
  \min_{W,Z} L(W,Z) = \frac{1}{2} \sum_{(d,n) \in \Omega} [x_{dn} - (WZ^T)_{dn}]^2 + ...
  $$

  Each row $$w_d$$ of $$W$$ then has its own system, built from the ratings of movie $$d$$: $$w_d = (\sum_{n:(d,n) \in \Omega} z_n z_n^T + \lambda_w I_K)^{-1} \sum_{n:(d,n) \in \Omega} x_{dn} z_n$$, and each row $$z_n$$ of $$Z$$ is computed in the same way.

Which method is the better choice? With missing ratings, an ALS sweep builds and solves one $$K \times K$$ system per row of $$W$$ and of $$Z$$, at a cost of $$O(\vert \Omega \vert K^2 + (D+N)K^3)$$, against $$O(\vert \Omega \vert K)$$ for a pass of SGD. SGD is thus in general simpler and faster, but ALS is better in two cases ([Koren et al., 2009][koren2009]). The first one is parallel hardware: for a fixed $$Z$$, the rows of $$W$$ are independent of each other, and so are the rows of $$Z$$ for a fixed $$W$$, while two SGD updates on ratings of the same movie or user conflict. The second one is implicit feedback, such as views or clicks, where all entries of $$X$$ count as observations: SGD would visit all $$DN$$ of them, while ALS handles them efficiently.

### Coordinate Descent 

ALS is an instance of a more general method. [Coordinate descent](https://en.wikipedia.org/wiki/Coordinate_descent) minimizes a function along one coordinate at a time, with the other coordinates fixed, and cycles over the coordinates; ALS is a block coordinate descent with two blocks, $$W$$ and $$Z$$. The figure shows coordinate descent on $$f(x, y) = 5x^2 - 6xy + 5y^2$$. The minimum along $$y$$ is at $$y = 0.6x$$, and the minimum along $$x$$ at $$x = 0.6y$$, so the path goes from $$(-0.5, -1)$$ to $$(-0.5, -0.3)$$, then to $$(-0.18, -0.3)$$, and zigzags toward the minimum at the origin. The zigzag comes from the term $$-6xy$$, which couples the two coordinates; without it, one exact minimization per coordinate would reach the minimum.

{: .picture}
<img width="36.4%" height="36.4%" src="{{ site.assets }}/linear-regression-matrix-factorization/coordinate_descent.svg">

{: .caption}
Coordinate descent on $$f(x, y) = 5x^2 - 6xy + 5y^2$$ from $$(-0.5, -1)$$: each step minimizes $$f$$ along one coordinate, alternating $$y$$ and $$x$$, and the iterates converge to the minimum at the origin. Adapted from: [Coordinate descent](https://en.wikipedia.org/wiki/Coordinate_descent)

The method is simple to implement. Each step solves a problem of one variable, often in closed form, and needs no gradient of the whole function, which helps for functions whose derivatives are much more expensive to compute than the function itself. For the LASSO, the minimum along one weight, with the others fixed, is a soft thresholding, which makes coordinate descent a standard LASSO solver. However, the steps are sequential, since each one uses the latest values of the other coordinates, so the method is not a good candidate for parallelism: only coordinates that do not interact, such as the rows of $$W$$ in ALS, can be updated in parallel. On non-smooth functions, it can also get stuck at a non-stationary point ([Wikipedia](https://en.wikipedia.org/wiki/Coordinate_descent#Limitations)), although it still converges when the non-smooth part is a sum of terms of one coordinate each, such as the L1 norm of the LASSO.

## Risk and Loss Functions

The right loss depends on the task, and two principles explain where the common losses come from. The first one is [empirical risk minimization][erm], which non-probabilistic models follow to choose the parameters $$\theta$$ of a model $$f$$, such as the weights $$w$$ of linear regression. The **risk** is the expected loss over the distribution of the data,

$$
R(\theta) = \mathbb{E}_{(x,y)} \left[ \ell(f(x, \theta), y) \right],
$$

where the per-sample loss $$\ell$$ compares the prediction $$\hat{y} = f(x, \theta)$$ with the true output $$y$$. The distribution of the data is unknown, so training minimizes the empirical risk instead, the mean loss over the training set, $$\frac{1}{N} \sum_{n=1}^N \ell(f(x_n, \theta), y_n)$$.

[erm]: https://en.wikipedia.org/wiki/Empirical_risk_minimization

The second principle applies to probabilistic models, which define the probability $$p(y \mid x, \theta)$$ of the label. [Maximum likelihood estimation][mle] chooses the parameters that minimize the negative log-likelihood of the training set, which, for independent samples, is a sum over the samples:

$$
L(\theta) = - \log p(y_1, \ldots, y_N \mid x_1, \ldots, x_N, \theta) = - \sum_{n=1}^N \log p(y_n \mid x_n, \theta)
$$

This is an empirical risk with the loss $$\ell = -\log p(y \mid x, \theta)$$, up to the factor $$1/N$$, so the two principles meet, and each choice of $$p$$ gives a loss.

[mle]: https://en.wikipedia.org/wiki/Maximum_likelihood_estimation

### Mean Squared Error

Most regression tasks use the MSE or the MAE, defined above. If the labels have Gaussian noise of fixed variance $$\sigma^2$$ around the prediction, $$p(y \mid x, \theta) = \mathcal{N}(y; f(x, \theta), \sigma^2)$$, the negative log-likelihood is

$$
L(\theta) = \frac{1}{2\sigma^2} \sum_{n=1}^N (y_n - f(x_n, \theta))^2 + \frac{N}{2} \log (2 \pi \sigma^2),
$$

where only the first term depends on $$\theta$$, so minimizing the MSE gives the maximum likelihood estimate. In the same way, noise with a [Laplace distribution][laplace], which has heavier tails than the Gaussian, gives the MAE.

[laplace]: https://en.wikipedia.org/wiki/Laplace_distribution

### Cross-Entropy

Most classification tasks use the cross-entropy loss. A classifier predicts a probability $$p_c = p(c \mid x, \theta)$$ for each of the $$C$$ classes, usually with the [softmax][softmax] of its outputs $$z_c$$, the logits: $$p_c = e^{z_c} / \sum_{j=1}^C e^{z_j}$$. The [cross-entropy][cross-entropy] between a target distribution $$q$$ over the classes and the prediction $$p$$ is $$H(q, p) = -\sum_{c=1}^C q_c \log p_c$$. For a label of class $$y$$, the target is [one-hot][one-hot], $$q_y = 1$$, and the loss is $$-\log p_y$$: with one-hot labels, the cross-entropy is the negative log-likelihood of the label. With two classes, the predicted probability $$p$$ of class 1 and a label $$y \in \{0, 1\}$$, it is the binary cross-entropy:

$$
\ell = -(y \log p + (1-y) \log (1-p))
$$

which is $$-\log p$$ for $$y = 1$$, and $$-\log(1-p)$$ for $$y = 0$$.

Computing the softmax first and its logarithm afterwards can overflow in $$e^{z_j}$$, or underflow to $$p_y = 0$$ and $$\log p_y = -\infty$$. Implementations therefore compute $$\log p_y = z_y - \log \sum_j e^{z_j}$$ directly from the logits, in one fused operation, with the [log-sum-exp][logsumexp] trick $$\log \sum_j e^{z_j} = z_{\max} + \log \sum_j e^{z_j - z_{\max}}$$, where $$z_{\max} = \max_j z_j$$. After the shift, the largest term of the sum is $$e^0 = 1$$, so the sum can neither overflow nor underflow to 0.

[softmax]: https://en.wikipedia.org/wiki/Softmax_function
[cross-entropy]: https://en.wikipedia.org/wiki/Cross-entropy
[one-hot]: https://en.wikipedia.org/wiki/One-hot
[logsumexp]: https://en.wikipedia.org/wiki/LogSumExp

### Triplet Loss

The **[triplet loss](https://en.wikipedia.org/wiki/Triplet_loss)** learns an embedding $$f$$ that maps similar inputs to nearby points, and dissimilar inputs to distant points. It suits tasks with many classes, or with classes unknown at training time, such as the identities of people in face recognition. FaceNet ([Schroff et al., 2015](https://arxiv.org/abs/1503.03832)) trains it on triplets of face images: an anchor $$x$$, a positive $$x^+$$, another image of the same person, and a negative $$x^-$$, an image of another person. The loss pulls the positive toward the anchor and pushes the negative away, until the negative is farther from the anchor than the positive by a margin $$m$$:

$$
\ell(x, x^+, x^-) = \max\left( \Vert f(x) - f(x^+) \Vert^2 - \Vert f(x) - f(x^-) \Vert^2 + m, 0 \right)
$$

FaceNet normalizes the embeddings to unit length and uses $$m = 0.2$$. Triplets that already satisfy the margin have a zero loss and do not help training, so FaceNet selects hard triplets, which violate it. The embedding then serves face verification (are these two images of the same person?), recognition (who is this person?) and clustering (which faces belong together?).

### Contrastive Loss

The **[contrastive loss](https://en.wikipedia.org/wiki/Siamese_neural_network)** ([Hadsell et al., 2006](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)) is similar to, and often confused with, the triplet loss: the triplet loss considers triplets of an anchor, a positive and a negative, while the contrastive loss considers pairs. It trains a siamese network, which maps the two inputs of a pair with the same weights, on pairs labelled similar ($$y = 1$$) or dissimilar ($$y = 0$$); the paper uses the opposite labels. With the distance $$\delta = \Vert f(x) - f(x') \Vert$$ between the embeddings of a pair $$(x, x')$$, the loss is:

$$
\ell = \frac{1}{2} y \delta^2 + \frac{1}{2} (1-y) \max(0, m - \delta)^2
$$

Similar pairs are pulled together, and dissimilar pairs are pushed apart until their distance reaches the margin $$m$$. The two losses therefore constrain the distances differently. The contrastive loss sets absolute targets, 0 for similar pairs and at least $$m$$ for dissimilar pairs, while the triplet loss only orders the distances, which lets the embeddings of one class spread out, as long as they stay closer to each other than to the other classes. A summary of the paper is in the <a href="{{ site.publications_permalink }}">publications bookmark</a>.

### Connectionist Temporal Classification loss

**[Connectionist temporal classification](https://en.wikipedia.org/wiki/Connectionist_temporal_classification)** (CTC) is a loss for labelling sequences without an alignment ([Graves et al., 2006](https://www.cs.toronto.edu/~graves/icml_2006.pdf)). In speech recognition, for example, the input has $$T$$ frames and the label sequence has $$U \leq T$$ labels, but the frames of each label are unknown, so a loss per frame cannot be computed. A [recurrent neural network][rnn], such as an [LSTM][lstm], outputs at each frame a softmax over the labels and a *blank*. A path picks one output per frame, and removing first the repeated labels and then the blanks maps it to a label sequence: with `-` for the blank, the paths `a-ab-` and `aa-ab` both give `aab`. The probability of a label sequence is the sum of the probabilities of all its paths, and the loss is its negative log. The number of paths grows exponentially with $$T$$, but the [forward-backward algorithm][forward-backward] computes the sum in $$O(TU)$$ operations, by dynamic programming over the label sequence with a blank between the labels and at both ends, i.e. over $$2U+1$$ states.

[rnn]: https://en.wikipedia.org/wiki/Recurrent_neural_network
[lstm]: https://en.wikipedia.org/wiki/Long_short-term_memory
[forward-backward]: https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm

### Focal Loss

The **focal loss** ([Lin et al., 2017](https://arxiv.org/abs/1708.02002)) extends the cross-entropy to class imbalance, which is common, for example in medical image segmentation, where the background is much more frequent than the lesions. It was introduced for dense [object detection][object-detection], where a one-stage detector scores about 100,000 candidate locations per image, with about 1 object location for 1,000 background ones. The many easy background examples then dominate the cross-entropy, so the focal loss down-weights the well-classified examples:

$$
\mathcal{L}_{\text{focal}} = -\alpha_y (1 - p_y)^\gamma \log(p_y)
$$

where $$p_y$$ is the predicted probability of the true class $$y$$, $$\alpha_y$$ is a weight per class ($$\alpha$$ for class 1 and $$1 - \alpha$$ for class 0 with two classes), and $$\gamma \geq 0$$ is the focusing parameter. The factor $$(1 - p_y)^\gamma$$ is close to 0 for confident correct predictions. With $$\gamma = 0$$, the loss is the weighted cross-entropy. With $$\gamma = 2$$, the best value in the paper (with $$\alpha = 0.25$$), an example with $$p_y = 0.9$$ has a 100 times smaller loss than with the cross-entropy, while an example with $$p_y = 0.1$$ keeps 81% of its loss.

[object-detection]: https://en.wikipedia.org/wiki/Object_detection

### Dice Loss

The **[Dice loss](https://arxiv.org/abs/1707.03237)** is a loss for [image segmentation][segmentation]. It measures the overlap of the set of predicted pixels $$P$$ and the set of ground truth pixels $$G$$ with the [Dice coefficient][dice], which ranges from 0 (no overlap) to 1 (equal sets):

$$
\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \vert P \cap G \vert}{\vert P \vert + \vert G \vert}
$$

For training, a differentiable form replaces the sets by the predicted probability $$p_i \in [0, 1]$$ and the ground truth value $$g_i \in \{0, 1\}$$ of each pixel $$i$$:

$$
\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_{i} p_i g_i}{\sum_{i} p_i + \sum_{i} g_i}
$$

V-Net ([Milletari et al., 2016](https://arxiv.org/abs/1606.04797)) introduced this loss for segmentation networks, with the squares $$p_i^2$$ and $$g_i^2$$ in the denominator. The loss depends only on the foreground pixels of the prediction and of the ground truth, not on the number of correct background pixels. It thus suits segmentation with a small foreground and precise boundaries, such as a tumor or an organ in a medical image, where a per-pixel loss is dominated by the background, and a model can classify most pixels correctly while it misses the tumor. For $$C$$ classes, the loss averages the Dice coefficients of the classes:

$$
\mathcal{L}_{\text{Dice}} = 1 - \frac{1}{C} \sum_{c=1}^{C} \frac{2 \sum_{i} p_{i,c} \, g_{i,c}}{\sum_{i} p_{i,c} + \sum_{i} g_{i,c}}
$$

where $$p_{i,c}$$ and $$g_{i,c}$$ are the predicted and ground truth values of pixel $$i$$ for class $$c$$. The generalized Dice loss of Sudre et al. (2017), the paper linked above, instead sums the numerators and the denominators over the classes, with the weight $$\alpha_c = 1 / (\sum_i g_{i,c})^2$$ of each class, the inverse of the squared number of its ground truth pixels, so that the large classes do not dominate the small ones:

$$
\mathcal{L}_{\text{GDL}} = 1 - \frac{2 \sum_{c=1}^{C} \alpha_c \sum_{i} p_{i,c} \, g_{i,c}}{\sum_{c=1}^{C} \alpha_c \sum_{i} (p_{i,c} + g_{i,c})}
$$

[segmentation]: https://en.wikipedia.org/wiki/Image_segmentation
[dice]: https://en.wikipedia.org/wiki/Dice-S%C3%B8rensen_coefficient
