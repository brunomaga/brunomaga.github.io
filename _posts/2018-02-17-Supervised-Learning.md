---

layout: post
title:  "Supervised Learning: basic concepts and methods"
date:   2018-02-17
categories: [machine learning, supervised learning]
tags: [machinelearning]
---

<div class="alert alert-primary" role="alert">
This post is largely inspired by the lecture notes for Machine Learning, by M. E. Khan, M. Jaggi and R. Urbanke at EPFL. Its content is continuously updated.
</div>

Supervised Learning is the field of machine learning that learns through supervision, or in practice, learns with the help of an external agent (human or automatic) that provides the solution for a given training set, in order to *provide an approximator* for the mechanics that relates the given input to the outputs (labels). We present the basic contents in the following sections. Advanced topics will be covered in individual posts.

##### Regression

Regression relates an input variable to an output, to either predict new outputs, or understand the effect of the input in the output. A regression dataset consists of a set of pairs $(x_n, y_n)$ of size $N$ with input $x_n$ and output/label $y_n$. For a new input $x_n$, the goal of regression is to find $f$ such that $$ y_n \approx f(x_n) $$. It does so by finding **weights** $w$ that approximate:
- linear regression: $$ y_n \approx f(x_n) = w_0 + w_1 x_{n1} $$
- multilinear regression: $$ y_n \approx f(x_n) = w_0 + w_1 x_{n1} + ... + + w_D x_{nD} $$
- logistic regression: $$ y_n \approx f(x_n) = \frac{e^{x_n}}{1+e^{x_n}} = \frac{1}{1+ e^{-x_n}} $$ 

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
- Mean Square Error: $$ MSE(w) = \frac{1}{N} \sum_{n=1}^N [ y_n - f(x_n)]^2 $$ , large errors have relatively greater influence, so it is not good for outliers;
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
- Least Squares: analytical solution for linear MSE ( $$ \triangledown L(w) = 0 $$ is a system of D equations ); 
- Gradient Descent: $$ w^{t+1} = w^{t} - \gamma \triangledown L (w^t) $$, for step size $\gamma$, and gradient $$ \triangledown L (w) = [ \frac{\partial L(w)}{\partial w_1}, ... , \frac{\partial L(w)}{\partial w_D}  ]^T $$;
- Stochastic Gradient Descent: $$ w^{t+1} = w^{t} - \gamma \triangledown L_n (w^t) $$, for a random choice of inputs $n$. Computationally cheap but unbiased estimate of gradient;
- Mini-batch SGD: $$ w^{t+1} = w^{t} - \gamma \frac{1}{\mid B \mid} \sum_{n \in B} \triangledown L_n (w^t) $$, for a random subset $ B \subseteq [N] $. For each sample $n$ , we compute the gradient at the same current point $w^{(t)}$;

With some variants:
- Subgradient Descent: $$ w^{t+1} = w^{t} - \gamma g $$, where $g$ is a [subgradient](https://en.wikipedia.org/wiki/Subderivative#The_subgradient) of $L$ (useful when function is not differentiable at $w^{(t)}$; 
- Projected Gradient: useful to solve a constrained optimization problem $min_w L(w)$ subject to $w \in C$, where $C \subset R^D$ is the constraint set;
  - projected gradient descent minimizes a function subject to a constraint. At each step we move in the direction of the negative gradient, and then "project" onto the feasible set.
  - Projections onto convex sets are unique;
  - Definition of projection: $$ P_c(w') = argmin_{v \in C} \| v - w' \| $$;
  - Update rule: $$ w^{(t+1)} = P_C [ w^{(t)} - \gamma \triangledown L (w^{(t)}) ] $$
  - alternatively, one can use penalty functions instead of projections, solving $$ min_{w \in R^D} L(w) + I_c(w) $$ where $I_c$ is the penalty function;

<p align="center">
<img width="30%" height="30%" src="/assets/2018-Supervised-Learning/projection.png"><br/>
<small>source: Machine Learning lecture notes, M Jaggi, EPFL</small>
</p>

When $$ \triangledown L (w) $$ is close to zero, we are close to an optimum. If second derivative is positive (semi-definite), then it may be a local minimum. If the function is convex, we are at a global minimum. 

Gradient descent is very sensitive to ill-conditioning. Normalizing input dimensions allows the step size to properly adjust to high differences in features dimensionalities. Choosing step size can be tricky. See [MIT_252_Lecture04.pdf](/assets/2018-Supervised-Learning/MIT_252_Lecture04.pdf) and [optpart2.pdf](/assets/2018-Supervised-Learning/optpart2.pdf) for some guidance, and the famous **Armijos Rule**.

Gradient descent is a first-order method (only looks at the gradient).  We get a more powerful optimization algorithm if we use also the second order terms. We need fewer steps to converge if we use second order terms, on the other hand every iteration is more costly. The **Newton's method** make use of the second order terms and takes steps in the direction that minimizes a quadratic approximation. We define the **Hessian** as 

$$
H_{i,j} = \frac{\partial^2 L(w)}{\partial w_{i} \partial w_j} 
$$  

and the update function of Newton's method as:

$$
w^{(t+1)} = w^{(t)} - \gamma ^{(t)} ( H^{(t)})^{-1} \triangledown L(w^{(t)}).
$$

[comment]: <>  This term comes from the second order Taylor approximation of a function around a point $$ w^{\star} $$:
[comment]: <> 
[comment]: <> $$
[comment]: <> L(w) \approx L(w^{\star}) + \triangledown L(w^{\star})^T (w - w^{\star}) + \frac{1}{2}(w - w^{\star})^T H(w^{\star})(w - w^{\star}).
[comment]: <> $$ 

I'll ommit the explanation of this formula, but email me if you are interested. As an important remark, this method is not commonly used. Although it can be faster when the second derivative is known and easy to compute, the analytic expression for the second derivative is often complicated or intractable, requiring a lot of computation. Numerical methods for computing the second derivative also require a lot of computation -- if *N* values are required to compute the first derivative, $N^2$ are required for the second derivative (source: [stack exchange](https://stats.stackexchange.com/questions/253632/why-is-newtons-method-not-widely-used-in-machine-learning)).

Finally, another method is the [**coordinate descent**](https://en.wikipedia.org/wiki/Coordinate_descent) that iterates over individual coordinates (keeping others fixed), in order to minimize the loss function. 

<p align="center">
<img width="30%" height="30%" src="/assets/2018-Supervised-Learning/coordinate_descent.svg.png"><br/>
<small>source: wikipedia</small>
</p>

The main advantage is that it is extremely simple to implement and doesn’t require any knowledge of the derivative of the function. It’s really useful for extremely complicated functions or functions whose derivatives are far more expensive to compute than the function itself. However, due to its iterative nature, it's not a good candidate for parallelism. Another issue is that it has a a non-smooth multivariable function, thus it may be stuck in non-stationary point if the level curves of a function are not smooth (source: [wikipedia](https://en.wikipedia.org/wiki/Coordinate_descent#Limitations)).

##### Probability Distributions 

A [probability distribution](https://en.wikipedia.org/wiki/Probability_distribution) is a mathematical function that provides the probabilities of occurrence of different possible outcomes in an experiment. Examples:
- Gaussian: 
 $$ p(y | \mu, \sigma^2 ) = \frac{1}{ \sqrt{2 \pi \sigma^2} } exp [ - \frac{(y - \mu)^2}{2 \sigma^2} ] $$, or in vector form $$ p(y | \mu, \Sigma ) = \frac{1}{ \sqrt{ (2 \pi)^D det(\Sigma)} } exp [ - \frac{1}{2}(y - \mu)^T \Sigma^{-1} (y-\mu)] $$

- Laplace:
 $$ p( y_n | x_n, w ) = \frac{1}{2b} e^{-\frac{1}{b} | y_n - X_n^T w | } $$

- [TODO complete]

The **log-likelihood** is the log of the [likelihood](https://www.statisticshowto.datasciencecentral.com/likelihood-function/) of observing --- given observed data --- a parameter value  of the statistical model used to describe that data. I.e.:

$$
L_{lik} (w) = log p (y | X, w)
$$

This can be used to estimate the cost. The log-likelihood if (typically?) convex in the weight vector $w$, as it's a sum of convex functions. The **Maximum likelihood estimator (MLE)** states that:

$$
argmin_w L_{MSE}(w) = argmax_w L_{lik} (w)
$$

MLE is a sample approximation of the *expected* log-likelihood i.e.

$$
L_{lik}(w) \approx E_{p(x,y)} [ log p(y | x,w ) ]
$$

##### Overfitting and Underfitting

Overfitting is fitting the noise in ad- dition to the signal. Underfitting is not fitting the signal well. To reduce overfitting, increasing data *may help*. 

<p align="center">
<img width="35%" height="35%" src="/assets/2018-Supervised-Learning/overfitting.png"><br/>
<small>source: Machine Learning lecture notes, M Jaggi, EPFL</small>
</p>

We can also use regularization, forcing the model to be not too complex.

##### Regularization

Occam’s Razor: "Plurality is not to be posited without necessity" or *simple models are better. To simplify the model, we can add a regulatization term $\Omega$ to penalize complex models:

$$
min_w L(w) + \Omega (w)
$$

Techniques:
- L2 regularization (standard Euclidean norm): $$ \Omega(w) = \lambda \| w \|^2 $$, where $$ \| w \|^2 = \sum_{i=0}^M w_i^2 $$
  - Large weights will be penalized, as they are *considered unlikely*;
  - If $L$ is the MSE, this is called **Ridge Regression** ;
  - the parameter $ \lambda \gt 0 $ can be tuned to reduce overfitting. This is the **model selection** problem: 
- L1 regularization: $$ \Omega(w) = \lambda \| w \| $$, where $$
 \| w \| = \sum_{i=0}^M | w_i | $$ 
  - keeping L1-norm small forces some elements in $w$ to be strictly 0, thus enforcity sparcity. Some features will not be used since their weight is 0.
  - If $L$ is the MSE, this is called **Lasso Regression**;
- [shrinkage](https://en.wikipedia.org/wiki/Shrinkage_estimator); 
- [dropout](https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5): a method to *drop out* (ignore) neurons in a (deep) [neural network]({{ site.baseurl }}{% post_url 2018-02-27-Deep-Neural-Networks %}) and retrieving the final model as an average of models (see separate [post]({{ site.baseurl }}{% post_url 2018-02-27-Deep-Neural-Networks %}) in Deep Neural Networks for details.

As a final note, Linear models can be made more powerful, by constructing better fea- tures for your input data. One way is to use nonlinear **basis functions** $$ \phi(x_j) $$, which are nonlinear transformations applied to input variables $x_j$.

[Tensorflow playground](https://playground.tensorflow.org) is a very useful resource to visualize the effects of regularization.

##### Cross-validation

**K-fold cross-validation** allows us to efficiently compute the error and are a better alternative to random data splits. We randomly partition the data into $K$ groups. We train on $K − 1$ groups and test on the remaining group. We repeat this until we have tested on all $K$ sets. We then average the results.

##### Classification

A **classifier** will divide the input space into a collection of regions belonging to each class. The boundaries of these regions are called **decision boundaries**. The aim of the classification problem is to build a predictor based on a training set and apply it to predict on *new* data. 

A common method is the **nearest neighbor**, assigning the label of the majority of the closest $k$ datapoints. A natural prediction for the *k-NN* is:

$$
f_{S_t,k} = \frac{1}{k} \sum_{n : x_n \in neighbors_{S_t},k(x)} y_n
$$

where $$ neighbors_{S_t},k(x) $$ is the set of $k$ input points in $S_t$ that are closest to $x$. The classification is performed by assigning the label with the most *votes* across the neighbors.
Similarly to the unsupervised learning methods described [previously]({{ site.baseurl }}{% post_url 2017-11-01-Unsupervised-Learning %}), the kNN method struggles with high dimensionality. This relates to the side-length required to capture a fraction of the data. In the picture below, when dimensionality is 10, we need 80% of volume to reach 10% of data. 

<p align="center">
<img width="35%" height="35%" src="/assets/2018-Supervised-Learning/curse_dimensionality.png"><br/>
<small>source: Machine Learning lecture notes, R Urbanke, EPFL</small>
</p>

##### Support Vector Machines

The Support Vector Machine (SVM) is obtained by changing the loss function to Hinge, optimizing:

$$
min_w \sum_{n=1}^N [1 - y_n x_n^T w]_{+} + \frac{\lambda}{2} \| w \|^2
$$

where the **Hinge Loss** is defined by $$ [z]_{+} = max \{ 0, z \} $$

<p align="center">
<img width="30%" height="30%" src="/assets/2018-Supervised-Learning/hinge_loss.png"><br/>
<small>source: Machine Learning lecture notes, M Jaggi, EPFL</small>
</p>

SVM is a **maximum margin method**, it aims at maximizing the minimum distance between labels. Maximizing the margin seems good because points near the decision surface represent very uncertain classification decisions:


<p align="center">
<img width="30%" height="30%" src="/assets/2018-Supervised-Learning/svm.png"><br/>
<small>source:  Introduction to Information Retrieval, Manning et al.,  Cambridge University Press. 2008</small>
</p>

The SVM is not differentiable. The concept of **duality** allows one to optimize $$ min_w L(w) $$, by creating a function $$ G(w,\alpha) $$ such that $$ L(w) = max_\alpha G(w,\alpha) $$ so that we can choose to optimize either of

$$
min_w max_\alpha G(w, \alpha) = max_\alpha min_w G(w,\alpha)
$$ 

In the SVM problem, $$ [z]_{+} = max \{ 0, z \} = max_\alpha \text{ } \alpha \text{ } z$$ where $$ \alpha \in [0,1] $$, so we can rewite the problem as:

$$
min_w max_{\alpha \in [0,1]^N} \sum_{n=1}^N \alpha [1 - y_n x_n^T w]_{+} + \frac{\lambda}{2} \| w \|^2
$$

which is differentiable, convex in $w$ and **concave** in $\alpha$ (a **concave** or **concave downwards** function is the negative of a convex function).

<p align="center">
<img width="30%" height="30%" src="/assets/2018-Supervised-Learning/svm_convex_concave.png"><br/>
<small>source: Machine Learning lecture notes (annotated), M Jaggi, EPFL</small>
</p>

Finally:

$$
\triangledown_w G(w,\alpha) = - \sum_{n=1}^N \alpha_n y_n x_n + \lambda w
$$

where we can find $a^\star \in R^N$ maximizing $g(n)$ using coordinate descent.

The **representer theorem** generalizes this problem: for a $$ w^\star $$ minimizing the following function:

$$
min_w \sum_{n=1}^N L_n (y_n, x_n^Tw) + \frac{\lambda}{2} \| w \|^2
$$

there exists $$ \alpha^\star $$ such that $$ w^\star = X \alpha^\star $$.

##### Kernel Function

A kernel is defined by $$ K = X^TX $$, or $$ K = \phi^T \phi $$ for basis functions. A big advantage of using kernels is that we do not need to specify $$ \phi(X) $$ as we can work with $K$ directly.

If $$ K_{ij} = k(x_i, x_j) $$, then $$ k(x,x') = \phi(x)^T \phi(x') $$. Examples of kernels:
- linear kernel: $$  k(x,x') = x^Tx' $$
- polynomial kernel: $$ k(x,x') = x^2(x')^2 $$
- Radial Basis Function (RBF) kernel: $$ k(x,x') = exp [-\frac{1}{2} (x-x')^T(x-x') ] $$

How can we ensure that there exists a $$\phi$$ corresponding to a given kernnel $K$?  The answer is: as long as the kernel satisfies certain properties:
- A kernel must be an inner product in some feature space. To ensure it, these properties must hold:
  - $K$ should be symmetric: $$ k(x,x') = k(x',x) $$
  - $K$ should be positive semi-definite. 

<div class="alert alert-warning" role="alert">
I did not come across any exercise using kernels, so I will continue this topic once I have a better grasp on the subject
</div>

##### Matrix Factorization

Matrix factorization can be used to discover underling latent factors and/or to predict missing values of the matrix. We aim to find $W$, $Z$ such that $$ W \approx WZ^T $$. I.e. we aim to predict $x_{dn}$, where $d$ is an element in $Z$, and $n$ is an element in $W$. For movie rating, $Z$ could be users, $W$ could be movies, and $x_{dn}$ the star rating.

We aim at optimizing:

$$
min_{W,Z} L (W,Z) = \frac{1}{2} \sum_{(d,n) \in \Omega} [ x_{dn} - (WZ^T)_{dn}]^2
$$

where $$ D \in R^{D \times K} $$ and $$ Z \in R^{N \times K} $$ are tall matrices, and $$ \Omega \subseteq [D] \times [N] $$ collects the indices of the observed ratings of the input matrix $X$.

<p align="center">
<img width="60%"  src="/assets/2018-Supervised-Learning/matrix-factorization.png">
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

To finalize, the article [Matrix Factorization Techniques for Recommender Systems](/assets/2018-Supervised-Learning/Recommender-Systems-Netflix.pdf) might be of your interest. 

##### Text Representation

Several techniques to represent text:
- [Bag of words](https://en.wikipedia.org/wiki/Bag-of-words_model). Each word is represented by an index in a vocabulary of words.
- The Co-occurence Matrix, where $n_{ij}$ holds contexts where word $i$ is used along word $j$;
  - We can use the previous method of matrix factorization to predict words co-occurence;
- [word2vec](https://skymind.ai/wiki/word2vec): word is represented by the hidden layer of a trained 2-layer neural network;
  - Very good explanation [here](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/);
- [GLOVE](https://nlp.stanford.edu/projects/glove/): similar as Word2Vec. While Word2Vec is a *predictive* model that predicts context given word, GLOVE learns by constructing a co-occurrence matrix (words $\times$ context) that counts how frequently a word appears in a context. Since it's going to be a gigantic matrix, we factorize this matrix to achieve a lower-dimension representation.
  - Another good resource [here](https://towardsdatascience.com/emnlp-what-is-glove-part-i-3b6ce6a7f970);
- Skip-Gram model: we train a neural network with pairs of words. We the utilized the network to retrieve the probability of every word in the dictionary to appear near the input word;
  - Another good resource [here](https://towardsdatascience.com/word2vec-skip-gram-model-part-1-intuition-78614e4d6e0b)
- [FastText (from facebook)](https://fasttext.cc/): no idea!

<div class="alert alert-warning" role="alert">
I'll ommit details as I plan to write an entire post dedicated to text representation, Natural Language Processing and context retrieving.
</div>

##### Binning

We have seen that simple linear classification schemes like logistic regression can work well but are limited. For some data representations that are non-linear, some classifiers (e.g. polynomial) may still be good, particularly when we know a priori which features are useful. A possible solution is to add as many features as possible i.e. add all polynomial terms up to some order, leading to overfitting.

Another possibility is to use **binning**, a technique for data preparation where we represent input as combinations of input data intervals, allowing a non-linear combination of inputs. As a simple example, take a neural network to detect the house prices inSwitzerland for a given latitude and longitude. A regular neural net would be trained for a given *latitude* and *longitude* input, against an output price. This could only represent linear combinations of both parameters. Alternatively we can discretize the parameters space, and represent as input all the combinations of latitude and longitude intervals, allowing non-linearity in the model:

<p align="center">
<img width="80%" height="80%" src="/assets/2018-Supervised-Learning/binning.png"><br/>
</p>

This approach obviously falls short for high input dimensionality. For such use cases, deep neural networks is the solution, as it provides a non-linear approximator.

##### (Deep) Neural Networks

Moved to a separate [post]({{ site.baseurl }}{% post_url 2018-02-27-Deep-Neural-Networks %}).

##### Graphical Models, Bayesian Networks and Belief Propagation

<div class="alert alert-warning" role="alert">
25.02.2019: will be added soon.
</div>
