---
layout: post
title:  "Linear Regression, Matrix Factorization and Loss Functions"
categories: ["Machine Learning"]
tags: ["supervised learning", "linear regression", "least squares", "gradient descent", "regularization", "matrix factorization", "alternating least squares", "loss functions"]
description: "Linear regression with gradient descent, least squares and regularization (LASSO, ridge), matrix factorization with SGD and alternating least squares, and the common loss functions, from cross-entropy to the focal and Dice losses."
---

On 1 January 1801, the astronomer Giuseppe Piazzi discovered the dwarf planet [Ceres](https://en.wikipedia.org/wiki/Ceres_(dwarf_planet)). Piazzi tracked its path for 40 days, until it was lost in the glare of the Sun. To find it again, astronomers needed its orbit, computed from these few observations. The prediction that allowed them to find Ceres, on 31 December 1801, came from the 24-year-old Carl Friedrich Gauss, who fitted the orbit to the observations with the method of least squares.

The method of least squares is the earliest form of [regression](https://en.wikipedia.org/wiki/Regression_analysis). It was published by Legendre in 1805 and by Gauss in 1809, to determine the orbits of comets and minor planets from astronomical observations, and Gauss claimed to have used it since 1795. This post derives the closed-form solution of linear regression, presents gradient descent, regularization and matrix factorization, and reviews the most common loss functions.

## Linear Regression

Regression relates an input variable to an output, either to predict new outputs or to understand the effect of the input on the output. A regression dataset consists of a set of $$N$$ pairs $$(x_n, y_n)$$, with an input $$x_n$$ of $$D$$ features and an output (or label) $$y_n$$. The goal of regression is to find a function $$f$$ such that $$y_n \approx f(x_n)$$, which then predicts the output of a new input. Fitting a multidimensional plane to the dataset means solving the (multiple) [linear regression](https://en.wikipedia.org/wiki/Linear_regression) problem, by finding the weights $$w$$ such that:

$$
y_n \approx f(x_n) = w_0 + w_1 x_{n1} + ... + w_D x_{nD} \hspace{1cm}\text{, or simply}\hspace{1cm} y \approx f(X) = Xw.
$$

Here, $$w_0$$ is the bias (or intercept), and the $$N \times (D+1)$$ matrix $$X$$ holds the inputs in its rows, after a first column of ones that multiplies $$w_0$$. As a running example, take the three points $$(0, 1)$$, $$(1, 2)$$ and $$(2, 4)$$, with one input each ($$D = 1$$). No line passes through all three, and linear regression looks for the line $$f(x) = w_0 + w_1 x$$ closest to them, in a sense that the loss function defines.

An error or **loss function** $$L(w)$$ indicates how good the estimation is. As an example, the [Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error) (MSE) loss is defined as $$L(w) = \frac{1}{N} \sum_{n=1}^N [y_n - f(x_n)]^2$$, and penalizes predictions with larger errors more heavily, due to the square term. Another loss function, the [Mean Absolute Error](https://en.wikipedia.org/wiki/Mean_absolute_error) (MAE), with $$L(w) = \frac{1}{N} \sum_{n=1}^N \vert y_n - f(x_n) \vert$$, is more robust to outliers. For three predictions with errors of 1, 1 and 10, the MSE is $$(1 + 1 + 100)/3 = 34$$, of which the outlier makes 98%, while the MAE is $$(1 + 1 + 10)/3 = 4$$, of which it makes 83%. A model trained with the MSE is thus pulled much more strongly by the outlier.

### Gradient Descent

In brief, the goal is to find the weight values that minimize the loss function. A naive approach is **Grid Search**, a brute-force method that tests all combinations of $$w$$ in a given interval. This is usually inefficient and only works on very low-dimensional problems, as the number of combinations grows exponentially with the number of weights: with 10 values per weight, a model of 10 weights already needs $$10^{10}$$ evaluations of the loss.

An alternative is to optimize the loss function by computing its gradient, which points in the direction in which the loss increases the fastest, and stepping $$w$$ in the opposite direction, towards the values that minimize the loss. The most common methods are [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) and its variants:
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

The difference shows on a single weight $$w$$, whose loss $$(w - \hat{w})^2$$ has its minimum at the least squares value $$\hat{w}$$. Minimizing $$(w - \hat{w})^2 + \lambda \vert w \vert$$ gives $$w = \text{sign}(\hat{w}) \max(\vert \hat{w} \vert - \lambda/2, 0)$$, while minimizing $$(w - \hat{w})^2 + \lambda w^2$$ gives $$w = \hat{w}/(1 + \lambda)$$. With $$\lambda = 1$$, LASSO moves $$\hat{w} = 2$$ to 1.5 and $$\hat{w} = 0.4$$ to 0, while Ridge Regression moves them to 1.0 and 0.2. In practice, LASSO subtracts a constant from the magnitude of the weight and stops at 0, while Ridge Regression divides it by a constant and never reaches 0.

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

The choice of the loss function depends on the task. The most common ones are the MSE for regression, the cross-entropy for classification, the contrastive and triplet losses to learn [embeddings](https://en.wikipedia.org/wiki/Embedding_(machine_learning)), the CTC loss for unsegmented sequences, and the focal and Dice losses for imbalanced classes and image segmentation. The following sections present them in this order, and a table at the end of the post compares them.

### Mean Squared Error

Most regression tasks use the Mean Squared Error loss function:

$$
MSE(\hat{y}_n) = \mathbb{E}\left[ (y_n - \hat{y}_n)^2 \right],
$$

or the Mean Absolute Error, which is similar to the MSE but uses the absolute value of the difference between $$y$$ and $$\hat{y}$$ instead of its square. The MSE is the negative log-likelihood of Gaussian noise, as shown above, and the MAE the one of noise with a [Laplace distribution](https://en.wikipedia.org/wiki/Laplace_distribution), whose heavier tails make large errors more likely. For a constant prediction, the MSE is minimized by the mean of the outputs, which one outlier can move far, and the MAE by their median, which an outlier hardly moves. Thus, the MSE suits data with Gaussian noise, and the MAE data with outliers.

### Cross-Entropy

With [one-hot](https://en.wikipedia.org/wiki/One-hot) labels, the [cross-entropy](https://en.wikipedia.org/wiki/Cross-entropy) is equivalent to the negative log-likelihood (of the [softmax](https://en.wikipedia.org/wiki/Softmax_function), when the output of the model is not a distribution). However, to avoid numerical instabilities and overflows, implementations compute it directly from the output of the model, with the [log-sum-exp](https://en.wikipedia.org/wiki/LogSumExp) trick. For the outputs (*logits*) $$a_1, ..., a_C$$ of the model, the loss of a sample of class $$c$$ is $$-\log \text{softmax}(a)_c = -a_c + a_{\max} + \log \sum_{j=1}^C \exp(a_j - a_{\max})$$, where $$a_{\max} = \max_j a_j$$, so that no exponential overflows. Most classification tasks use the binary or multi-class cross-entropy loss functions:
- binary: $$H(p) = -(y \log p(x) + (1-y) \log (1-p(x)))$$, where $$p(x)$$ is the predicted probability that the input $$x$$ is of class 1, i.e. it simplifies to $$-\log p(x)$$ for true label $$y=1$$, and to $$-\log (1-p(x))$$ for true label $$y=0$$;
- multi-class: $$H(q,p) = -\sum_{c=1}^{C} q_c \log p_c$$, over the $$C$$ classes, for the true distribution $$q$$ and the predicted distribution $$p$$. With a one-hot label, only the term of the true class remains.

As an example, a prediction of 0.9 for the true class costs $$-\log 0.9 = 0.105$$, while a prediction of 0.1 costs $$-\log 0.1 = 2.303$$, 22 times more: the cross-entropy penalizes confident mistakes heavily.

### Contrastive Loss

The **[contrastive loss](https://en.wikipedia.org/wiki/Siamese_neural_network)** trains siamese networks, two copies of one network with shared weights, against pairs of inputs labeled as similar or dissimilar (1 or 0). It is formulated in terms of the similarity label $$y$$ (1 for similar, 0 for dissimilar) and the Euclidean distance $$\delta$$ between the embeddings of the two inputs, as:

$$
L = \frac{1}{2} ⋅y⋅\delta^2 + \frac{1}{2} ⋅(1−y)⋅\max(0,m-\delta)^2
$$

where $$m$$ is a hyper-parameter (the margin). In practice, the loss pulls similar pairs together, towards a distance of 0, and pushes dissimilar pairs apart, and a dissimilar pair contributes to the loss only while its distance is below $$m$$. The original paper is [Dimensionality Reduction by Learning an Invariant Mapping](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf), summarized in the <a href="{{ site.publications_permalink }}">publications bookmark</a>.

### Triplet Loss

**[Triplet loss](https://en.wikipedia.org/wiki/Triplet_loss)** is used in classification tasks where the number of classes is very large. Triplet loss helps by learning distributed embedding representations of data points such that, in the high-dimensional vector space, contextually similar data points are projected nearby, while dissimilar data points are projected far away from each other.

The network is trained against triplets of images: an **anchor image** of a person, a **positive image**, which is another image of the same person, and a **negative image**, which is a picture of another person. The Triplet Loss minimizes the Euclidean distance between the anchor and the positive, and maximizes the distance between the anchor and the negative. Tasks that can be performed: face recognition ("who is this person?"), verification ("are these the same person?") and clustering ("find the most similar people"). The formulation is:

$${\displaystyle {\mathcal {L}}\left(x,x^{+},x^{-}\right)=\operatorname {max} \left({\|\operatorname {f} \left(x\right)-\operatorname {f} \left(x^{+}\right)\|}^{2}-{\|\operatorname {f} \left(x\right)-\operatorname {f} \left(x^{-}\right)\|}^{2}+m ,0\right)}$$

where $$x$$ is an anchor input, $$x^{+}$$ is a positive input of the same class as $$x$$, $$x^{-}$$ is a negative input of a different class from $$x$$, $$m$$ is a margin between positive and negative pairs, and $$f$$ is an embedding. More details are given in the original paper, [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832).

The triplet loss is similar to, and often confused with, the contrastive loss, as both learn embeddings from examples of similar and dissimilar inputs. Yet, they constrain different distances. The contrastive loss constrains *absolute* distances: each similar pair is pulled towards a distance of 0, and each dissimilar pair is pushed beyond the margin $$m$$. The triplet loss constrains *relative* distances: the squared distance of the anchor to the negative must exceed the one to the positive by the margin $$m$$, whatever the distances themselves. In practice, the triplet loss considers the anchor-neighbor-distant triplets, while the contrastive loss deals with the anchor-neighbor and anchor-distant pairs of samples. As the FaceNet paper notes, a loss that pulls the faces of one person together "encourages all faces of one identity to be projected onto a single point in the embedding space", while the triplet loss "allows the faces for one identity to live on a manifold".

### Connectionist Temporal Classification Loss

**[Connectionist Temporal Classification (CTC)](https://en.wikipedia.org/wiki/Connectionist_temporal_classification)** is a classifier and loss function for noisy, unsegmented sequential input data, used to train [recurrent neural networks](https://en.wikipedia.org/wiki/Recurrent_neural_network) (RNNs) such as [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) networks to tackle sequence problems where the timing is variable. CTC was published by Graves et al. in 2006, in [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://www.cs.toronto.edu/~graves/icml_2006.pdf).

At each time step, the network outputs the probability of each label and of an extra *blank* label (no label). The probability of a target sequence is the sum of the probabilities of all the alignments (one output per time step) that give the target once repeated labels and blanks are removed. As an example, with the blank written "−", 7 of the 256 sequences of 4 outputs over the labels c, a, t and the blank give the target "cat": "ccat", "caat", "catt", "cat−", "ca−t", "c−at" and "−cat". The blank also separates repeated labels: "l−l" gives "ll", while "ll" gives "l". The CTC loss is the negative logarithm of this probability, computed by dynamic programming, as the number of alignments grows quickly with the number of time steps (28 for 5 time steps).

### Focal Loss

**[Focal Loss](https://arxiv.org/abs/1708.02002)** (Lin et al., 2017) is an extension of the Cross-Entropy Loss designed to address class imbalance, a common issue in e.g. dense [object detection](https://en.wikipedia.org/wiki/Object_detection) and medical image segmentation, where the background (non-lesion regions) is much more prevalent than the foreground (lesion regions). In dense detection, a detector evaluates about 100,000 candidate locations per image, and the paper gives an imbalance of 1:1000 between foreground and background as an example. Focal Loss focuses on hard-to-classify examples and down-weights the loss contribution from easy examples. For binary classification, it is defined as (Eq. 5 in the paper):

$$
\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

where:
- $$p_t$$ is the model’s estimated probability for the true class.
- $$\alpha_t$$ is a balancing factor, equal to $$α$$ for the positive class and to $$1-α$$ for the negative class (especially useful for imbalanced datasets).
- $$γ$$ is the focusing parameter, usually set to 2, which adjusts the focus on hard-to-classify examples.

In practice, the factor $$(1 - p_t)^\gamma$$ is close to 0 for well-classified examples, where $$p_t$$ is close to 1, and close to 1 for misclassified examples. With $$\gamma = 2$$, an example classified with $$p_t = 0.9$$ has a loss 100 times lower than with the cross-entropy, as $$(1 - 0.9)^2 = 0.01$$, and one classified with $$p_t \approx 0.968$$ a loss 1,000 times lower (section 3.2 of the paper). A misclassified example with $$p_t = 0.1$$ keeps 81% of its loss. Thus, the many easy examples of the background no longer dominate the loss. With $$\gamma = 0$$, the focal loss is the cross-entropy weighted by $$\alpha_t$$ (Eq. 3 in the paper). For multi-class classification, the same formula applies, with one balancing factor per class, $$\alpha_t$$ being the one of the true class.

### Dice Loss

**[Dice loss](https://arxiv.org/abs/1707.03237)** (or Dice Similarity Coefficient Loss) is a loss function used specifically for [image segmentation](https://en.wikipedia.org/wiki/Image_segmentation) tasks, to evaluate the overlap between predicted and ground truth segmentation masks, and was first used to train neural networks in [V-Net](https://arxiv.org/abs/1606.04797). It is based on the [Dice coefficient](https://en.wikipedia.org/wiki/Dice-S%C3%B8rensen_coefficient), which measures the similarity between two sets, with values ranging from 0 (no overlap) to 1 (perfect overlap). Dice Loss is typically used when precise object boundaries are important, such as segmenting organs, tumors, or any region of interest in medical imaging where the goal is to accurately delineate the object of interest from the surrounding tissue. Key benefits of Dice Loss:
- Direct Focus on Overlap: Dice loss directly optimizes the overlap between predicted and ground truth masks, which is crucial for segmentation tasks where precise boundary delineation matters.
- Effective for Imbalanced Classes: It works well when there is a significant class imbalance between foreground and background, as it evaluates segmentation performance based on the overlap of the foreground regions rather than pixel accuracy.
- Works Well with Small Objects: Dice loss is particularly effective in medical imaging tasks where the objects of interest (e.g. tumors, organs) might occupy a small portion of the image.

$$
\mathcal{L}_{\text{Dice}} = 1 - \frac{2 |P \cap G|}{|P| + |G|}
$$

where:
- $$P$$ is the set of predicted pixels.
- $$G$$ is the set of ground truth pixels.
- $$\vert P \cap G \vert$$ is the size of the intersection of the predicted and ground truth sets, representing the overlap.

As an example, take an image of 100 pixels, of which 4 belong to an object. A prediction of background everywhere has a pixel accuracy of 96%, but a Dice coefficient of 0, i.e. a Dice loss of 1. A prediction that finds 2 of the 4 object pixels, plus 2 false positives, has the same pixel accuracy of 96%, but a Dice coefficient of $$2 \cdot 2 / (4 + 4) = 0.5$$. Unlike the pixel accuracy, the Dice loss rewards the overlap with the small object.

For practical implementation, as sets are not differentiable, Dice Loss is often computed as:

$$
\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_{i=1}^{M} p_i g_i}{\sum_{i=1}^{M} p_i + \sum_{i=1}^{M} g_i}
$$

where:
- $$p_i$$ is the predicted value for the $$i$$-th pixel.
- $$g_i$$ is the ground truth value for the $$i$$-th pixel.
- $$M$$ is the number of pixels of the image.

For multi-class segmentation, a generalized form of Dice Loss is:

$$
\mathcal{L}_{\text{Dice}} = 1 - \frac{1}{C} \sum_{c=1}^{C} \frac{2 \sum_{i=1}^{M} p_{i,c} \, g_{i,c}}{\sum_{i=1}^{M} p_{i,c} + \sum_{i=1}^{M} g_{i,c}}
$$

where $$C$$ is the number of classes, and $$p_{i,c}$$ and $$g_{i,c}$$ are the predicted and ground truth values of the $$i$$-th pixel for class $$c$$. The Generalised Dice Loss of the paper linked above (Sudre et al., 2017) instead sums the terms of all classes in a single fraction. It weights those of each class $$c$$ by $$w_c = 1/(\sum_{i=1}^{M} g_{i,c})^2$$, the inverse of the square of its number of ground truth pixels, so that small structures count as much as large ones.

### Comparison of the Loss Functions

The following table sums up the loss functions of this post:

| Loss | Task | Penalizes | When to use it |
|---|---|---|---|
| MSE | Regression | The squared error | Gaussian noise: large errors are rare |
| MAE | Regression | The absolute error | Data with outliers |
| Cross-entropy | Classification | A low probability of the true class | Most classification tasks |
| Contrastive | Embeddings | Similar pairs far apart, and dissimilar pairs closer than $$m$$ | Pairs labeled as similar or dissimilar |
| Triplet | Embeddings | A negative not farther from the anchor than the positive by $$m$$ | Many classes, as the identities of face recognition |
| CTC | Sequence labeling | A low probability of the target, summed over all alignments | An unknown alignment of the target to the input, as in speech recognition |
| Focal | Classification | As the cross-entropy, scaled down for easy examples | Many easy examples, as the background of dense detection |
| Dice | Segmentation | A small overlap of the predicted and true masks | Small objects in large images |

To find more about the topic, the book [Mathematics for Machine Learning](https://mml-book.github.io/) by Deisenroth, Faisal and Ong covers empirical risk minimization and parameter estimation (chapter 8), and the maximum likelihood and maximum a posteriori estimators of linear regression (chapter 9).


