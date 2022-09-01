---
layout: post
title:  "Closed-form Linear Regression and Matrix Factorization"
categories: [machine learning, supervised learning]
tags: [machinelearning]
---

Regression relates an input variable to an output, to either predict new outputs, or understand the effect of the input in the output. A regression dataset consists of a set of pairs $(x_n, y_n)$ of size $N$ with input $x_n$ and output/label $y_n$. For a new input $x_n$, the goal of regression is to find $f$ such that $$ y_n \approx f(x_n) $$. If we wan't to fit the dataset to a multidimensional plane, we are solving the (multivariate) linear regression problem, by finding the weights $w$ that approximate $y_n$ i.e:

$$
y_n \approx f(x_n) = w_0 + w_1 x_{n1} + ... + + w_D x_{nD} \hspace{1cm}\text{, or simply}\hspace{1cm} y_n \approx f(X) = Xw.
$$

We use an error or loss function $L(w)$ to indicate how good our estimation is. As as example, the Mean Square Error (MSE) loss is described as $ L(w) = \frac{1}{N} \sum_{n=1}^N [ y_n - f(x_n)]^2$, and has the property of penalizing predictions with larger errors due to the square term. Another loss function, the Mean Absolute Error (MAE), with $ L(w) = \frac{1}{N} \sum_{n=1}^N \mid y_n - f(x_n) \mid $, enforces sparsity in the weight values.

In brief, we want to find the weight values that mimimize the loss function. A naive approach is the Grid Search, a brute-force approach that tests all combinations of $w$ in a given interval. This is usually a dull approach and only works on very low dimensionality problems. An alternative is to optimize the loss function by computing the gradient of the derivative and stepping $w$ to the values that minimizes the loss. The most common methods are the gradient descent and its variants:
- Gradient Descent: $$ w^{t+1} = w^{t} - \gamma \triangledown L (w^t) $$, for step size $\gamma$, and gradient $$ \triangledown L (w) = [ \frac{\partial L(w)}{\partial w_1}, ... , \frac{\partial L(w)}{\partial w_D}  ]^T $$;
- Stochastic Gradient Descent: $$ w^{t+1} = w^{t} - \gamma \triangledown L_n (w^t) $$, for a random choice of an inputs $n$. Computationally cheap but unbiased estimate of gradient;
- Mini-batch SGD: $$ w^{t+1} = w^{t} - \gamma \frac{1}{\mid B \mid} \sum_{n \in B} \triangledown L_n (w^t) $$, for a random subset $ B \subseteq [N] $. For each sample $n$ , we compute the gradient at the same current point $w^{(t)}$;

For the particular example of linear regression and the Mean Square Error loss function, the problem is called **Least Squares** and has a closed-form solution. In practice, we want to minimize the following:

  $$
  \begin{align*}
  (y - Xw)^2 = & (y-Xw)^T(y-Xw) \\
  = & y^Ty - y^TXw - (Xw)^Ty + (Xw)^TXw \\
  = & y^Ty - w^TX^Ty - w^TX^Ty + w^TX^TXw \\
  = & y^Ty - 2 w^TX^Ty + w^TX^TXw;\\
  \end{align*}
  $$

We want to compute the $w$ that minimizes the loss, or equally, where its derivative is 0. So we compute:

  $$
  \begin{align*}
    & \frac{\partial}{\partial w} y^Ty - 2 w^TX^Ty + w^TX^TXw = 0 \\
  \Leftrightarrow & 0 - 2 X^Ty + 2 X^TXw = 0 \\
  \Leftrightarrow & 2 X^TXw = 2X^Ty \\
  \Leftrightarrow & w = (X^TX)^{-1} X^Ty;\\
  \end{align*}
  $$

Note that we used the trick $\frac{\partial w^Ta}{\partial w} = \frac{\partial a^Tw}{\partial w} = a$ (<a href="{{ site.assets }}/resources/the_matrix_cookbook.pdf">The Matrix Cookbook</a>, eq 2.4.1). The results tells us that **if $X^TX$ is invertible**, this minimization problem has an unique closed-form solution given by that final form. As a side note, the **Gram matrix $X^TX$** is invertible if **X has full column rank**, i.e. $rank(X)=D$ (we'll ommit the proof). The rank of a matrix is defined as the maximum number of linearly independent column vectors in the matrix, therefore we assume that all columns are linearly independent.


##### Regularization

It's a common practice to add to the loss a regulatization term $\Omega$ that penalizes complex models. Therefore our loss minimization problem becomes:

$$
min_w L(w) + \Omega (w)
$$

Common regularizer approaches are:
- L1 regularization: $$ \Omega(w) = \lambda \| w \| $$, where $$
 \| w \| = \sum_{i=0}^M | w_i | $$ 
  - keeping L1-norm small forces some elements in $w$ to be strictly 0, thus enforcity sparcity. Some features will not be used since their weight is 0.
  - If $L$ is the MAE, this is called **Lasso Regression**;
  - the parameter $ \lambda \gt 0 $ can be tuned to reduce overfitting. This is the **model selection** problem;
- L2 regularization (standard Euclidean norm): $$ \Omega(w) = \lambda \| w \|^2 $$, where $$ \| w \|^2 = \sum_{i=0}^M w_i^2 $$
  - Large weights will be penalized, as they are *considered unlikely*;
  - If $L$ is the MSE, this is called **Ridge Regression** ;
  - similarly to the Least Squares problem, the loss of the MSE with Rigde Regression has an analytical solution:
    - computed by minimizing $(y - Xw)^T(y-Wx) + \lambda w^Tw$, and;
    - derivating with respect to $w$ leads to the normal equation $w = (X^TX - \lambda I)^{-1} X^Ty$.

Other techniques that do not add the regularizer term to the loss function but helps in improving model accuracy: 
- [dropout](https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5): a method to *drop out* (ignore) neurons in a (deep) [neural network]({{ site.baseurl }}{% post_url 2018-03-27-Deep-Neural-Networks %}) at every batch during training and using all neurons at validation. Shown to increase generalization by forcing the model to train wihtout relying on specific weights. (see [separate post]({{ site.baseurl }}{% post_url 2018-03-27-Deep-Neural-Networks %}) for more details.
- [layer, instance, batch, and group normalization](https://arxiv.org/abs/1803.08494): methods to reduce internal covariance shift by fitting the output of activation layers to a standard normal distributed. Learns optional scale and shift parameters that scale of standard deviation and shift of mean for improved performance over the standard normal distribution.  

### Matrix Factorization

Matrix factorization can be used to discover underling latent factors and/or to predict missing values of the matrix. We aim to find $W$, $Z$ such that $$ W \approx WZ^T $$. I.e. we aim to predict $x_{dn}$, where $d$ is an element in $Z$, and $n$ is an element in $W$. For movie rating, $Z$ could be users, $W$ could be movies, and $x_{dn}$ the star rating.

We aim at optimizing:

$$
min_{W,Z} L (W,Z) = \frac{1}{2} \sum_{(d,n) \in \Omega} [ x_{dn} - (WZ^T)_{dn}]^2
$$

where $$ D \in R^{D \times K} $$ and $$ Z \in R^{N \times K} $$ are tall matrices, and $$ \Omega \subseteq [D] \times [N] $$ collects the indices of the observed ratings of the input matrix $X$.

<p align="center">
<img width="60%"  src="/assets/Linear-Regression-and-Matrix-Factorization/matrix-factorization.png">
</p>

This cost function is not convex and not [identifiable](https://en.wikipedia.org/wiki/Identifiability). $K$ is the number of latent features (e.g. gender, age group, citizenship, etc). Large $K$ facilitates overfitting. We can add a regularizer to the function:

$$
min_{W,Z} L (W,Z) = \frac{1}{2} \sum_{(d,n) \in \Omega} [ x_{dn} - (WZ^T)_{dn}]^2 + \frac{\lambda_w}{2} \| W \|^2 +  \frac{\lambda_z}{2} \| Z \|^2
$$ 

where (again), $\lambda_z$ and  $\lambda_w$ are scalars. 

We use stochastic gradient descent to optimize this problem. The training objective is a function over \|$\Omega$\| terms (one per rating):

$$
\frac{1}{| \Omega |} \sum_{(d,n) \in \Omega} \frac{1}{2} [x_{dn} - (WZ^T)_{dn}]^2
$$

For one fixed element $(d,n)$, we derive the gradient $(d',k)$ for $W$ and  $(n',k)$ for $Z$:

$$
(d',k) = \frac{\partial}{\partial w_{d',k}} \frac{1}{2} [x_{dn} - (WZ^T)_{dn}]^2 =
\begin{cases}
    -[x_n - (WZ^T)_{dn}] z_{n,k} & \text{, if } d'=d\\
    0,              & \text{, otherwise}
\end{cases}
$$

and

$$
(n',k) = \frac{\partial}{\partial w_{n',k}} \frac{1}{2} [x_{dn} - (WZ^T)_{dn}]^2 =
\begin{cases}
    -[x_n - (WZ^T)_{dn}] w_{d,k} & \text{, if } n'=n\\
    0,              & \text{, otherwise}
\end{cases} 
$$

Reminder: $z_{n,k}$ are user features, $w_{d,k}$ are movie features. The gradient has $(D+N)K$ entries.

We can also use **Alternating Least Squares (ALS)**. The ALS factorizes a given matrix $R$ into two factors $U$ and $V$ such that $$ R \approx U^TV $$.

- If there are **no** missing ratings in the matrix ie $$ \Omega = [D] \times [N] $$, then:

  $$
  min_{W,Z} L(W,Z) = \frac{1}{2} \sum_{d=1}^D \sum_{n=1}^N [x_{dn} - (WZ^T)_{dn}]^2 + \text{ (regularizer) } =  \frac{1}{2} \| X - WZ^T \| ^2 + \frac{\lambda_w}{2} \| W \|^2 +  \frac{\lambda_z}{2} \| Z \|^2
  $$

  - We can use coordinate descent (minimize $W$ for fixed $Z$ and vice-versa) to minimize cost plus regularizer;
- If there are missing entries, the problem is harder, as only the ratings $$ (d,n) \in \Omega $$ contribute to the cost, i.e..;

  $$
  min_{W,Z} L(W,Z) = \frac{1}{2} \sum_{(d,n) \in \Omega} [x_{dn} - (WZ^T)_{dn}]^2 + ...
  $$


##### Coordinate Descent 

The [**coordinate descent**](https://en.wikipedia.org/wiki/Coordinate_descent) optimization method iterates over individual coordinates (keeping others fixed), in order to minimize the loss function. 

<p align="center">
<img width="40%" height="40%" src="/assets/Linear-Regression-and-Matrix-Factorization/coordinate_descent.svg.png"><br/>
<small>source: wikipedia</small>
</p>

The main advantage is that it is extremely simple to implement and doesn’t require any knowledge of the derivative of the function. It’s really useful for extremely complicated functions or functions whose derivatives are far more expensive to compute than the function itself. However, due to its iterative nature, it's not a good candidate for parallelism. Another issue is that it has a a non-smooth multivariable function, thus it may be stuck in non-stationary point if the level curves of a function are not smooth (source: [wikipedia](https://en.wikipedia.org/wiki/Coordinate_descent#Limitations)).

