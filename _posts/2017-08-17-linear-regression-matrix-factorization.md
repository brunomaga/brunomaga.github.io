---
layout: post
title:  "Linear Regression and Matrix Factorization"
categories: ["Machine Learning"]
tags: ["supervised learning", "linear regression", "least squares", "gradient descent", "regularization", "matrix factorization", "alternating least squares", "empirical risk minimization", "maximum likelihood"]
description: "Linear regression with gradient descent, least squares and regularization (LASSO, ridge), matrix factorization with SGD and alternating least squares, and the squared loss as empirical risk minimization and maximum likelihood."
---

On 1 January 1801, the astronomer Giuseppe Piazzi discovered the dwarf planet [Ceres](https://en.wikipedia.org/wiki/Ceres_(dwarf_planet)). Piazzi tracked its path for 40 days, until it was lost in the glare of the Sun. To find it again, astronomers needed its orbit, computed from these few observations. The prediction that allowed them to find Ceres, on 31 December 1801, came from the 24-year-old Carl Friedrich Gauss, who fitted the orbit to the observations with the method of least squares.

The method of least squares is the earliest form of [regression](https://en.wikipedia.org/wiki/Regression_analysis). It was published by Legendre in 1805 and by Gauss in 1809, to determine the orbits of comets and minor planets from astronomical observations, and Gauss claimed to have used it since 1795. This post derives the closed-form solution of linear regression, presents gradient descent, regularization and matrix factorization, and relates the squared loss to the empirical risk and to the maximum likelihood.

## Linear Regression

Regression relates an input variable to an output, either to predict new outputs or to understand the effect of the input on the output. A regression dataset consists of a set of $$N$$ pairs $$(x_n, y_n)$$, with an input $$x_n$$ of $$D$$ features and an output (or label) $$y_n$$. The goal of regression is to find a function $$f$$ such that $$y_n \approx f(x_n)$$, which then predicts the output of a new input. Fitting a multidimensional plane to the dataset means solving the (multiple) [linear regression](https://en.wikipedia.org/wiki/Linear_regression) problem, by finding the weights $$w$$ such that:

$$
y_n \approx f(x_n) = w_0 + w_1 x_{n1} + ... + w_D x_{nD} \hspace{1cm}\text{, or simply}\hspace{1cm} y \approx f(X) = Xw.
$$

Here, $$w_0$$ is the bias (or intercept), and the $$N \times (D+1)$$ matrix $$X$$ holds the inputs in its rows, after a first column of ones that multiplies $$w_0$$. As a running example, take the three points $$(0, 1)$$, $$(1, 2)$$ and $$(2, 4)$$, with one input each ($$D = 1$$). No line passes through all three, and linear regression looks for the line $$f(x) = w_0 + w_1 x$$ closest to them, in a sense that the loss function defines.

An error or **loss function** $$L(w)$$ indicates how good the estimation is. As an example, the [Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error) (MSE) loss is defined as $$L(w) = \frac{1}{N} \sum_{n=1}^N [y_n - f(x_n)]^2$$, and penalizes predictions with larger errors more heavily, due to the square term. Another loss function, the [Mean Absolute Error](https://en.wikipedia.org/wiki/Mean_absolute_error) (MAE), with $$L(w) = \frac{1}{N} \sum_{n=1}^N \vert y_n - f(x_n) \vert$$, is more robust to outliers. For three predictions with errors of 1, 1 and 10, the MSE is $$(1 + 1 + 100)/3 = 34$$, of which the outlier makes 98%, while the MAE is $$(1 + 1 + 10)/3 = 4$$, of which it makes 83%. A model trained with the MSE is thus pulled much more strongly by the outlier.

### Gradient Descent

In brief, the goal is to find the weight values that minimize the loss function. A naive approach is **Grid Search**, a brute-force method that tests all combinations of $$w$$ in a given interval. This is usually inefficient and only works on very low-dimensional problems, as the number of combinations grows exponentially with the number of weights: with 10 values per weight, a model of 10 weights already needs $$10^{10}$$ evaluations of the loss.

An alternative is to optimize the loss function by computing its gradient, which points in the direction in which the loss increases the fastest, and stepping $$w$$ in the opposite direction, towards the values that minimize the loss. The most common methods are [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent), whose convergence is discussed in the post [Mathematics for ML]({{ site.baseurl }}{% post_url 2017-08-10-algebra-for-ml %}#gradient-descent), and its variants:
- **Gradient Descent**: $$w^{t+1} = w^{t} - \eta \nabla L(w^t)$$, for the step size (or [learning rate](https://en.wikipedia.org/wiki/Learning_rate)) $$\eta$$ and the gradient $$\nabla L(w) = [\frac{\partial L(w)}{\partial w_0}, ..., \frac{\partial L(w)}{\partial w_D}]^T$$;
- **[Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)** (SGD): $$w^{t+1} = w^{t} - \eta \nabla L_n(w^t)$$, for a randomly chosen input $$n$$ with loss $$L_n$$, where $$L = \frac{1}{N} \sum_{n=1}^N L_n$$. A computationally cheap but unbiased estimate of the gradient, as the mean of $$\nabla L_n$$ over all inputs is $$\nabla L$$;
- **Mini-batch SGD**: $$w^{t+1} = w^{t} - \eta \frac{1}{\vert B \vert} \sum_{n \in B} \nabla L_n(w^t)$$, for a random subset $$B \subseteq [N]$$ of the inputs, where $$[N] = \{1, ..., N\}$$. For each sample $$n$$, the gradient is computed at the same current point $$w^t$$. This is a *compromise* between the two: less noisy than SGD, and cheaper per step than gradient descent.

### Least Squares

For the particular case of linear regression with the Mean Squared Error loss function, the problem is called **[Least Squares](https://en.wikipedia.org/wiki/Least_squares)** and has a closed-form solution, $$w = (X^TX)^{-1} X^Ty$$, derived as follows. In practice, the following is minimized (the factor $$\frac{1}{N}$$ does not change the minimum):

  $$
  \begin{align*}
  \| y - Xw \|^2 = \, & (y-Xw)^T(y-Xw) \\
  = \, & y^Ty - y^TXw - (Xw)^Ty + (Xw)^TXw \\
  = \, & y^Ty - w^TX^Ty - w^TX^Ty + w^TX^TXw \\
  = \, & y^Ty - 2 w^TX^Ty + w^TX^TXw\\
  \end{align*}
  $$

This requires computing the $$w$$ that minimizes the loss or, equivalently (as the loss is a [convex function](https://en.wikipedia.org/wiki/Convex_function) of $$w$$), the $$w$$ where its derivative is 0. So $$w$$ is computed as:

  $$
  \begin{align*}
    & \frac{\partial}{\partial w} \left( y^Ty - 2 w^TX^Ty + w^TX^TXw \right) = 0 \\
  \Leftrightarrow \, & 0 - 2 X^Ty + 2 X^TXw = 0 \\
  \Leftrightarrow \, & 2 X^TXw = 2X^Ty \\
  \Leftrightarrow \, & w = (X^TX)^{-1} X^Ty\\
  \end{align*}
  $$

Note that the derivation used the trick $$\frac{\partial w^Ta}{\partial w} = \frac{\partial a^Tw}{\partial w} = a$$ (<a href="{{ site.assets }}/resources/the_matrix_cookbook.pdf">The Matrix Cookbook</a>, section 2.4.1, eq. 69) and $$\frac{\partial w^TAw}{\partial w} = (A + A^T)w$$ (section 2.4.2, eq. 81). The third line of this derivation, once divided by 2, gives the **normal equations** $$X^TXw = X^Ty$$. The result shows that **if $$X^TX$$ is invertible**, this minimization problem has a unique closed-form solution, given by that final form.

As a side note, the **[Gram matrix](https://en.wikipedia.org/wiki/Gram_matrix) $$X^TX$$** is invertible if **$$X$$ has full column rank**, i.e. $$\text{rank}(X)=D+1$$, as its columns are the $$D$$ inputs and the column of ones for the bias $$w_0$$. The [rank](https://en.wikipedia.org/wiki/Rank_(linear_algebra)) of a matrix is defined as the maximum number of linearly independent column vectors in the matrix, so this assumes that all columns are linearly independent. Then, for any vector $$v$$, $$X^TXv = 0$$ implies $$\lVert Xv \rVert^2 = v^TX^TXv = 0$$, i.e. $$Xv = 0$$, which only holds for $$v = 0$$, so $$X^TX$$ is invertible.

For the three points of the example, the rows of $$X$$ are $$(1, 0)$$, $$(1, 1)$$ and $$(1, 2)$$, and $$y = (1, 2, 4)^T$$, so that:

$$
X^TX = \begin{bmatrix} 3 & 3 \\ 3 & 5 \end{bmatrix}, \hspace{1cm} X^Ty = \begin{bmatrix} 7 \\ 10 \end{bmatrix}, \hspace{1cm} w = \frac{1}{6} \begin{bmatrix} 5 & -3 \\ -3 & 3 \end{bmatrix} \begin{bmatrix} 7 \\ 10 \end{bmatrix} = \begin{bmatrix} 5/6 \\ 3/2 \end{bmatrix}
$$

The best line is thus $$f(x) = 0.83 + 1.5x$$. Its predictions, 0.83, 2.33 and 3.83, leave the errors 0.17, −0.33 and 0.17, which sum to 0, and no other line has a smaller sum of squared errors.

For the sake of comparison, the closed-form solution takes $$O(ND^2 + D^3)$$ operations, to compute $$X^TX$$ and to solve the normal equations, against $$O(ND)$$ for a step of gradient descent. The step computes the gradient of the MSE, $$\nabla L(w) = -\frac{2}{N} X^T(y - Xw)$$, with two products of a matrix and a vector. For $$N = 10^6$$ samples of $$D = 10^3$$ features, the closed form takes about $$10^{12}$$ operations, as many as 1,000 steps of gradient descent. The closed form is thus preferred for a small $$D$$, and gradient descent for a large one.

### Regularization

It is common practice to add to the loss a **regularization** term $$\mathcal{R}(w)$$ that penalizes complex models. Such models tend to fit the noise of the training data and to predict new data poorly, a problem called [overfitting](https://en.wikipedia.org/wiki/Overfitting). The loss minimization problem then becomes:

$$
\min_w L(w) + \mathcal{R}(w)
$$

Common regularization approaches are:
- **L1 regularization**: $$\mathcal{R}(w) = \lambda \lVert w \rVert_1$$, where $$\lVert w \rVert_1 = \sum_{i=1}^D \vert w_i \vert$$ (the bias $$w_0$$ is not penalized). Linear regression with this term is called **[LASSO](https://en.wikipedia.org/wiki/Lasso_(statistics)) Regression**, as in **L**east **A**bsolute **S**hrinkage and **S**election **O**perator;
- **L2 regularization** (standard Euclidean norm): $$\mathcal{R}(w) = \lambda \lVert w \rVert_2^2$$, where $$\lVert w \rVert_2^2 = \sum_{i=1}^D w_i^2$$. Large weights are penalized, as they are *considered unlikely*. If $$L$$ is the MSE, this is called **[Ridge Regression](https://en.wikipedia.org/wiki/Ridge_regression)**.

In both, the parameter $$\lambda \gt 0$$ can be tuned to reduce overfitting. This is the **model selection** problem.

LASSO encourages simple, sparse models (i.e. models with fewer parameters). This particular type of regression is well-suited to automate certain parts of model selection, like variable selection/parameter elimination. However, with highly correlated features ([multicollinearity](https://en.wikipedia.org/wiki/Multicollinearity)), LASSO tends to keep one of them arbitrarily, and Ridge Regression or the [elastic net](https://en.wikipedia.org/wiki/Elastic_net_regularization) (L1 plus L2) are preferred.

Similarly to the Least Squares problem, the MSE loss with Ridge Regression has an analytical solution. It is computed by minimizing $$(y - Xw)^T(y-Xw) + \lambda w^Tw$$: differentiating with respect to $$w$$ leads to the regularized normal equations $$(X^TX + \lambda I)w = X^Ty$$, with the solution $$w = (X^TX + \lambda I)^{-1} X^Ty$$. This form also penalizes $$w_0$$; excluding the bias sets the first diagonal entry of $$I$$ to 0. The solution exists even when $$X^TX$$ is not invertible, as $$v^T(X^TX + \lambda I)v = \lVert Xv \rVert^2 + \lambda \lVert v \rVert^2 > 0$$ for any $$v \neq 0$$, so $$X^TX + \lambda I$$ is invertible.

On the three points of the example, with $$\lambda = 1$$ and the bias not penalized, Ridge Regression gives:

$$
w = \left( \begin{bmatrix} 3 & 3 \\ 3 & 5 \end{bmatrix} + \begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix} \right)^{-1} \begin{bmatrix} 7 \\ 10 \end{bmatrix} = \begin{bmatrix} 4/3 \\ 1 \end{bmatrix}
$$

The slope shrinks from 1.5 to 1.0, and the bias grows from 0.83 to 1.33. The line still passes through the mean of the points, $$(1, 2.33)$$: as $$w_0$$ is not penalized, the derivative of the loss with respect to $$w_0$$ still sets the sum of the errors to 0.

Note: in statistics, **[shrinkage](https://en.wikipedia.org/wiki/Shrinkage_(statistics))** is the reduction in the effects of sampling variation on the estimated parameters (e.g. a model overfitting the training data and not performing well on unseen data). Regularization attempts to do this by "shrinking" model parameters towards 0: LASSO sets some of them exactly to 0, while Ridge Regression makes all of them smaller, without setting them to 0.

The difference shows on a single weight $$w$$, whose loss $$(w - \hat{w})^2$$ has its minimum at the least squares value $$\hat{w}$$. Minimizing $$(w - \hat{w})^2 + \lambda \vert w \vert$$ gives $$w = \text{sign}(\hat{w}) \max(\vert \hat{w} \vert - \lambda/2, 0)$$, while minimizing $$(w - \hat{w})^2 + \lambda w^2$$ gives $$w = \hat{w}/(1 + \lambda)$$. With $$\lambda = 1$$, LASSO moves $$\hat{w} = 2$$ to 1.5 and $$\hat{w} = 0.4$$ to 0, while Ridge Regression moves them to 1.0 and 0.2. In practice, LASSO subtracts a constant from the magnitude of the weight and stops at 0, while Ridge Regression divides it by a constant and never reaches 0. The following picture illustrates the difference with two weights $$w_1$$ and $$w_2$$:

{: .picture}
<img width="65.6%" height="65.6%" src="{{ site.assets }}/linear-regression-matrix-factorization/l1_l2_regularization.svg">

{: .caption}
The contours of a squared loss (black ellipses) around its least squares solution, and the regions $$\lVert w \rVert_1 \le 1$$ of L1 regularization (left) and $$\lVert w \rVert_2 \le 1$$ of L2 regularization (right). Minimizing the regularized loss is equivalent to minimizing the loss inside such a region, of a size that depends on $$\lambda$$: the solution is the point where the smallest contour touches the region, often a corner for L1 regularization, where a weight is 0. Inspired by: [The Elements of Statistical Learning, figure 3.11](https://web.stanford.edu/~hastie/ElemStatLearn/)

Other techniques that do not add a regularization term to the loss function, but help improve model accuracy:
- [dropout](https://jmlr.org/papers/v15/srivastava14a.html): a method to *drop out* (ignore) neurons in a (deep) neural network at every batch during training, while using all neurons at validation. Dropout was shown to increase generalization by forcing the model to train without relying on specific weights.
- [layer, instance, batch, and group normalization](https://arxiv.org/abs/1803.08494): methods to reduce internal covariate shift by normalizing the output of activation layers to the mean and variance of a standard normal distribution. They can also learn scale and shift parameters, which scale the standard deviation and shift the mean, for improved performance over the standard normal distribution.

## Matrix Factorization

[Matrix factorization](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems)) can be used to discover underlying latent factors and/or to predict missing values of the matrix. It became widely known during the [Netflix Prize](https://en.wikipedia.org/wiki/Netflix_Prize) competition, started in 2006, where it outperformed classic nearest-neighbor techniques ([Koren et al., 2009]({{ site.assets }}/linear-regression-matrix-factorization/recommender-systems-netflix.pdf)). The training set of the competition held more than 100 million ratings, from 1 to 5 stars, of more than 17,000 movies by about 500,000 users: only about 1.2% of the entries of the matrix of ratings were known. The [singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) does not apply, as it is undefined when entries of the matrix are missing, and fitting only the few known entries is highly prone to overfitting (Koren et al., 2009).

The goal is to find $$W$$ and $$Z$$ such that $$X \approx WZ^T$$, i.e. to predict $$x_{dn}$$, where $$d$$ indexes the rows of $$W$$ and $$n$$ the rows of $$Z$$. For movie ratings, $$Z$$ could be users, $$W$$ could be movies, and $$x_{dn}$$ the star rating. Each movie and each user is then described by a vector of $$K$$ numbers, a row of $$W$$ or of $$Z$$, where $$K$$ is the number of columns of both factors. The rating of a movie by a user is predicted by the dot product of their two vectors. Note that, in this section, $$X$$ is the $$D \times N$$ matrix of the ratings of $$D$$ movies by $$N$$ users, and no longer the data matrix of linear regression.

The aim is to optimize:

$$
\min_{W,Z} L (W,Z) = \frac{1}{2} \sum_{(d,n) \in \Omega} [ x_{dn} - (WZ^T)_{dn}]^2
$$

where $$ W \in \mathbb{R}^{D \times K} $$ and $$ Z \in \mathbb{R}^{N \times K} $$ are tall matrices, and $$ \Omega \subseteq [D] \times [N] $$ collects the indices of the observed ratings of the input matrix $$X$$. The following picture illustrates the factorization:

{: .picture}
<img width="44.1%" height="44.1%"  src="{{ site.assets }}/linear-regression-matrix-factorization/matrix-factorization.svg">

{: .caption}
Matrix factorization $$X \approx WZ^T$$ of the ratings of $$D = 7$$ movies by $$N = 7$$ users, with 11 observed ratings and $$K = 3$$ latent features. The rating $$x_{dn}$$ of movie $$d$$ by user $$n$$ is approximated by the dot product of row $$d$$ of $$W$$, the features of movie $$d$$, and column $$n$$ of $$Z^T$$, the features of user $$n$$.

This cost function is neither convex nor [identifiable](https://en.wikipedia.org/wiki/Identifiability): for any invertible $$K \times K$$ matrix $$Q$$, the factors $$WQ$$ and $$Z(Q^{-1})^T$$ give the same product $$WZ^T$$. $$K$$ is the number of **latent features** (e.g. gender, age group, citizenship, etc.). A large $$K$$ facilitates overfitting, as the factors hold $$(D+N)K$$ parameters: for the Netflix Prize, $$K = 200$$ already gives about 100 million parameters, as many as the known ratings. A regularizer can be added to the function:

$$
\min_{W,Z} L (W,Z) = \frac{1}{2} \sum_{(d,n) \in \Omega} [ x_{dn} - (WZ^T)_{dn}]^2 + \frac{\lambda_w}{2} \| W \|^2 +  \frac{\lambda_z}{2} \| Z \|^2
$$ 

where (again) $$\lambda_z$$ and $$\lambda_w$$ are scalars, and the norms are [Frobenius norms](https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm).

### Stochastic Gradient Descent

Stochastic gradient descent can be used to optimize this problem. Without the regularizer, the training objective is an average of $$\vert \Omega \vert$$ terms (one per observed rating), and each step of SGD takes one of them at random:

$$
\frac{1}{| \Omega |} \sum_{(d,n) \in \Omega} \frac{1}{2} [x_{dn} - (WZ^T)_{dn}]^2
$$

For one fixed element $$(d,n)$$, the gradient entries $$(d',k)$$ for $$W$$ and $$(n',k)$$ for $$Z$$ are derived as:

$$
(d',k) = \frac{\partial}{\partial w_{d',k}} \frac{1}{2} [x_{dn} - (WZ^T)_{dn}]^2 =
\begin{cases}
    -[x_{dn} - (WZ^T)_{dn}] z_{n,k} & \text{, if } d'=d\\
    0              & \text{, otherwise}
\end{cases}
$$

and

$$
(n',k) = \frac{\partial}{\partial z_{n',k}} \frac{1}{2} [x_{dn} - (WZ^T)_{dn}]^2 =
\begin{cases}
    -[x_{dn} - (WZ^T)_{dn}] w_{d,k} & \text{, if } n'=n\\
    0              & \text{, otherwise}
\end{cases} 
$$

Reminder: $$z_{n,k}$$ are user features, $$w_{d,k}$$ are movie features. The gradient has $$(D+N)K$$ entries, of which only $$2K$$ are non-zero: those of row $$d$$ of $$W$$, written $$w_d$$, and of row $$n$$ of $$Z$$, written $$z_n$$. Thus, with the prediction error $$e_{dn} = x_{dn} - (WZ^T)_{dn}$$, an SGD step on the rating $$(d, n)$$ updates only these two rows, as:

$$
w_d \leftarrow w_d + \eta \, e_{dn} \, z_n, \hspace{1cm} z_n \leftarrow z_n + \eta \, e_{dn} \, w_d
$$

where both updates use the values before the step, and $$\eta$$ is the step size. In practice, when the rating is underestimated ($$e_{dn} > 0$$), the step moves the features of the movie towards those of the user, and vice versa, which increases their dot product. With the regularizer, Koren et al. (2009) also subtract $$\eta \lambda_w w_d$$ and $$\eta \lambda_z z_n$$ from the two updates, which shrink both rows towards 0. Another approach minimizes the cost over one factor at a time, a form of coordinate descent.

### Coordinate Descent

The [**coordinate descent**](https://en.wikipedia.org/wiki/Coordinate_descent) optimization method iterates over individual coordinates (keeping the others fixed) to minimize the loss function, as illustrated below:

{: .picture}
<img width="36.4%" height="36.4%" src="{{ site.assets }}/linear-regression-matrix-factorization/coordinate_descent.svg">

{: .caption}
Coordinate descent on $$f(x, y) = 5x^2 - 6xy + 5y^2$$ from $$(-0.5, -1)$$: each step minimizes $$f$$ along one coordinate, alternating $$y$$ and $$x$$, and the iterates converge to the minimum at the origin. Adapted from: [Coordinate descent](https://en.wikipedia.org/wiki/Coordinate_descent)

For the function of the picture, $$f(x, y) = 5x^2 - 6xy + 5y^2$$, the minimum along $$y$$ for a fixed $$x$$ is where $$\frac{\partial f}{\partial y} = 10y - 6x = 0$$, i.e. at $$y = 0.6x$$. Similarly, the minimum along $$x$$ for a fixed $$y$$ is at $$x = 0.6y$$. From $$(-0.5, -1)$$, the iterates are $$(-0.5, -0.3)$$, $$(-0.18, -0.3)$$, $$(-0.18, -0.108)$$, and so on: each pair of steps multiplies both coordinates by 0.36, and the iterates converge to the minimum at the origin.

The main advantage is that it is extremely simple to implement and does not require any knowledge of the derivative of the function: the minimum along one coordinate can be found by a search along that coordinate (a *line search*). It is useful for extremely complicated functions or functions whose derivatives are far more expensive to compute than the function itself. However, due to its sequential nature, it is not a good candidate for parallelism, unless several coordinates can be minimized independently, as the rows of each factor in ALS, described next. Another issue arises with non-smooth multivariable functions: it may get stuck at a non-stationary point if the level curves of the function are not smooth (source: [Wikipedia](https://en.wikipedia.org/wiki/Coordinate_descent#Limitations)).

### Alternating Least Squares

**Alternating Least Squares (ALS)** can also be used. ALS factorizes the same matrix $$X$$ into the two factors $$W$$ and $$Z$$ such that $$X \approx WZ^T$$, with a **block coordinate descent**. It minimizes the cost over all the entries of $$W$$ for a fixed $$Z$$, then over all the entries of $$Z$$ for a fixed $$W$$, and so on. With one factor fixed, the cost is quadratic in the other one, as in least squares, so each step has a closed-form solution and decreases the cost, until convergence (Koren et al., 2009):

- If there are **no** missing ratings in the matrix, i.e. $$ \Omega = [D] \times [N] $$, then:

  $$
  \begin{align*}
  \min_{W,Z} L(W,Z) = & \frac{1}{2} \sum_{d=1}^D \sum_{n=1}^N [x_{dn} - (WZ^T)_{dn}]^2 + \text{ (regularizer) } \\
  = & \frac{1}{2} \| X - WZ^T \| ^2 + \frac{\lambda_w}{2} \| W \|^2 +  \frac{\lambda_z}{2} \| Z \|^2
  \end{align*}
  $$

  - the closed-form updates are $$W = XZ (Z^TZ + \lambda_w I_K)^{-1}$$ for a fixed $$Z$$, and $$Z = X^TW (W^TW + \lambda_z I_K)^{-1}$$ for a fixed $$W$$, where $$I_K$$ is the $$K \times K$$ identity matrix;
- If there are missing entries, the problem is harder, as only the ratings $$ (d,n) \in \Omega $$ contribute to the cost, i.e.:

  $$
  \min_{W,Z} L(W,Z) = \frac{1}{2} \sum_{(d,n) \in \Omega} [x_{dn} - (WZ^T)_{dn}]^2 + ...
  $$

  - each row $$w_d$$ of $$W$$ is then computed from its observed ratings only, as $$w_d = (\sum_{n:(d,n) \in \Omega} z_n z_n^T + \lambda_w I_K)^{-1} \sum_{n:(d,n) \in \Omega} x_{dn} z_n$$, and similarly for each row $$z_n$$ of $$Z$$.

SGD is in general easier and faster than ALS. However, ALS is favorable in two cases (source: [Koren et al., 2009]({{ site.assets }}/linear-regression-matrix-factorization/recommender-systems-netflix.pdf)). The first is parallel computing, as each row of $$W$$ (or of $$Z$$) is computed independently of the other rows. The second is implicit feedback, such as purchases or views instead of ratings, where the matrix records whether each user bought or watched each item. As this matrix is dense, looping over each of its entries, as SGD does, is not practical.

## Risk and Loss Functions

In ML, for non-probabilistic models, the principle of [empirical risk minimization](https://en.wikipedia.org/wiki/Empirical_risk_minimization) is followed to find good parameters. The **risk** (or *expected* risk) of a predictor $$f$$ is its expected loss over the distribution of the data, between the true output $$y$$ and the predicted value $$f(x)$$, i.e.

$$
r(f) = \mathbb{E}_{x,y}[l(f(x), y)]
$$

where $$l(\hat{y}, y)$$ is the loss of the prediction $$\hat{y}$$ for the true output $$y$$, e.g. $$(y - \hat{y})^2$$. As the distribution of the data is unknown, the average loss on the $$N$$ training samples, the **empirical risk** $$\frac{1}{N} \sum_{n=1}^N l(\hat{y}_n, y_n)$$, is minimized instead (MML book, section 8.2.2). The predictor $$\hat{y}_n = f(x_n, \theta^*)$$ represents the output of the model $$f$$ with input data $$x_n$$ and the parameters $$\theta^*$$ that minimize the empirical risk. For linear regression with the squared loss, $$\theta$$ is $$w$$ and the empirical risk is the MSE.

In the case of a statistical model, [maximum likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) finds the parameters for which the model fits the data well, by minimizing the negative log-likelihood:

$$
L_x(\theta) = -\log p(x \mid \theta) .
$$

For linear regression with Gaussian noise, this minimization is equivalent to the Least Squares problem above. If $$y_n = f(x_n) + \epsilon_n$$, with independent Gaussian noise $$\epsilon_n$$ of mean 0 and variance $$\sigma^2$$, the negative log-likelihood of the outputs is $$\frac{1}{2\sigma^2} \lVert y - Xw \rVert^2 + \frac{N}{2} \log(2\pi\sigma^2)$$, whose minimum over $$w$$ is the least squares solution.

Most regression tasks use the Mean Squared Error loss function:

$$
MSE(\hat{y}_n) = \mathbb{E}\left[ (y_n - \hat{y}_n)^2 \right],
$$

or the Mean Absolute Error, which is similar to the MSE but uses the absolute value of the difference between $$y$$ and $$\hat{y}$$ instead of its square. The MSE is the negative log-likelihood of Gaussian noise, as shown above, and the MAE the one of noise with a [Laplace distribution](https://en.wikipedia.org/wiki/Laplace_distribution), whose heavier tails make large errors more likely. For a constant prediction, the MSE is minimized by the mean of the outputs, which one outlier can move far, and the MAE by their median, which an outlier hardly moves. Thus, the MSE suits data with Gaussian noise, and the MAE data with outliers.

Classification tasks use the cross-entropy, presented in the post [Deep Neural Networks, Backpropagation and Optimization]({{ site.baseurl }}{% post_url 2018-03-27-dnn-backprop %}#loss-functions), and the losses of other tasks, such as learning embeddings and segmenting images, are presented in the post [Dropout, CNNs, Skip Connections and Embeddings]({{ site.baseurl }}{% post_url 2018-04-01-dropout-cnns-skip-connections-embeddings %}#task-specific-loss-functions).

To find more about the topic, the book [Mathematics for Machine Learning](https://mml-book.github.io/) by Deisenroth, Faisal and Ong covers empirical risk minimization and parameter estimation (chapter 8), and the maximum likelihood and maximum a posteriori estimators of linear regression (chapter 9).


