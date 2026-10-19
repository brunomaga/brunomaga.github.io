---
layout: post
title:  "Linear Regression, Matrix Factorization and Loss Functions"
categories: ["Machine Learning"]
tags: ["supervised learning", "linear regression", "matrix factorization", "loss functions"]
---

The earliest form of [regression](https://en.wikipedia.org/wiki/Regression_analysis) is the method of least squares, published by Legendre in 1805 and by Gauss in 1809. Both applied it to determine the orbits of comets and minor planets around the Sun from astronomical observations. This post derives the closed-form solution of linear regression, presents regularization and matrix factorization, and reviews the most common loss functions.

Regression relates an input variable to an output, either to predict new outputs or to understand the effect of the input on the output. A regression dataset consists of a set of $$N$$ pairs $$(x_n, y_n)$$, with input $$x_n$$ and output/label $$y_n$$. For a new input $$x_n$$, the goal of regression is to find $$f$$ such that $$ y_n \approx f(x_n) $$. Fitting a multidimensional plane to the dataset means solving the (multiple) [linear regression](https://en.wikipedia.org/wiki/Linear_regression) problem, by finding the weights $$w$$ such that:

$$
y_n \approx f(x_n) = w_0 + w_1 x_{n1} + ... + w_D x_{nD} \hspace{1cm}\text{, or simply}\hspace{1cm} y \approx f(X) = Xw.
$$

An error or **loss function** $$L(w)$$ is used to indicate how good the estimation is. As an example, the [Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error) (MSE) loss is defined as $$ L(w) = \frac{1}{N} \sum_{n=1}^N [ y_n - f(x_n)]^2$$, and penalizes predictions with larger errors more heavily, due to the square term. Another loss function, the [Mean Absolute Error](https://en.wikipedia.org/wiki/Mean_absolute_error) (MAE), with $$ L(w) = \frac{1}{N} \sum_{n=1}^N \mid y_n - f(x_n) \mid $$, is more robust to outliers.

In brief, the goal is to find the weight values that minimize the loss function. A naive approach is **Grid Search**, a brute-force method that tests all combinations of $$w$$ in a given interval. This is usually inefficient and only works on very low-dimensional problems, as the number of combinations grows exponentially with the number of weights.

An alternative is to optimize the loss function by computing its gradient and stepping $$w$$ towards the values that minimize the loss. The most common methods are [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) and its variants:
- Gradient Descent: $$ w^{t+1} = w^{t} - \gamma \triangledown L (w^t) $$, for step size $$\gamma$$, and gradient $$ \triangledown L (w) = [ \frac{\partial L(w)}{\partial w_0}, ... , \frac{\partial L(w)}{\partial w_D}  ]^T $$;
- [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent): $$ w^{t+1} = w^{t} - \gamma \triangledown L_n (w^t) $$, for a randomly chosen input $$n$$ (with loss $$L_n$$). A computationally cheap but unbiased estimate of the gradient;
- Mini-batch SGD: $$ w^{t+1} = w^{t} - \gamma \frac{1}{\mid B \mid} \sum_{n \in B} \triangledown L_n (w^t) $$, for a random subset $$ B \subseteq [N] $$. For each sample $$n$$, the gradient is computed at the same current point $$w^t$$. This is a *compromise* between the two: less noisy than SGD, and cheaper per step than gradient descent.

For the particular case of linear regression with the Mean Squared Error loss function, the problem is called **[Least Squares](https://en.wikipedia.org/wiki/Least_squares)** and has a closed-form solution. In practice, the following is minimized (the factor $$\frac{1}{N}$$ does not change the minimum):

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

Note that the derivation used the trick $$\frac{\partial w^Ta}{\partial w} = \frac{\partial a^Tw}{\partial w} = a$$ (<a href="{{ site.assets }}/resources/the_matrix_cookbook.pdf">The Matrix Cookbook</a>, section 2.4.1, eq. 69) and $$\frac{\partial w^TAw}{\partial w} = (A + A^T)w$$ (section 2.4.2, eq. 81). The result shows that **if $$X^TX$$ is invertible**, this minimization problem has a unique closed-form solution, given by that final form. As a side note, the **[Gram matrix](https://en.wikipedia.org/wiki/Gram_matrix) $$X^TX$$** is invertible if **$$X$$ has full column rank**, i.e. $$rank(X)=D+1$$ (the proof is omitted), as its columns are the $$D$$ inputs and a column of ones for the bias $$w_0$$. The [rank](https://en.wikipedia.org/wiki/Rank_(linear_algebra)) of a matrix is defined as the maximum number of linearly independent column vectors in the matrix, so this assumes that all columns are linearly independent.

For the sake of comparison, the closed-form solution takes $$O(ND^2 + D^3)$$ operations, against $$O(ND)$$ for a step of gradient descent, so it is preferred for a small $$D$$.


### Regularization

It is common practice to add to the loss a **regularization** term $$\Omega$$ that penalizes complex models. The loss minimization problem then becomes:

$$
min_w L(w) + \Omega (w)
$$

Common regularization approaches are:
- L1 regularization: $$ \Omega(w) = \lambda \mid w \mid $$, where $$\mid w \mid = \sum_{i=1}^D \mid w_i \mid $$ (the bias $$w_0$$ is not penalized)
  - this is called **[LASSO](https://en.wikipedia.org/wiki/Lasso_(statistics)) Regression** as in **L**east **A**bsolute **S**hrinkage and **S**election **O**perator.
  - the parameter $$ \lambda \gt 0 $$ can be tuned to reduce [overfitting](https://en.wikipedia.org/wiki/Overfitting). This is the **model selection** problem.
  - encourages simple, sparse models (i.e. models with fewer parameters). This particular type of regression is well-suited to automate certain parts of model selection, like variable selection/parameter elimination. However, with highly correlated features ([multicollinearity](https://en.wikipedia.org/wiki/Multicollinearity)), LASSO tends to keep one of them arbitrarily, and Ridge Regression or the [elastic net](https://en.wikipedia.org/wiki/Elastic_net_regularization) (L1 plus L2) are preferred.
- L2 regularization (standard Euclidean norm): $$ \Omega(w) = \lambda \| w \|^2 $$, where $$ \| w \|^2 = \sum_{i=1}^D w_i^2 $$
  - Large weights will be penalized, as they are *considered unlikely*;
  - If $$L$$ is the MSE, this is called **[Ridge Regression](https://en.wikipedia.org/wiki/Ridge_regression)**;
  - similarly to the Least Squares problem, the MSE loss with Ridge Regression has an analytical solution, which exists even when $$X^TX$$ is not invertible:
    - computed by minimizing $$(y - Xw)^T(y-Xw) + \lambda w^Tw$$, and
    - differentiating with respect to $$w$$ leads to the normal equation $$w = (X^TX + \lambda I)^{-1} X^Ty$$. This form also penalizes $$w_0$$; excluding the bias sets the first diagonal entry of $$I$$ to 0.

Note: in statistics, **[shrinkage](https://en.wikipedia.org/wiki/Shrinkage_(statistics))** is the reduction in the effects of sampling variation on the estimated parameters (e.g. a model overfitting the training data and not performing well on unseen data). Regularization attempts to do this by "shrinking" model parameters towards 0 in LASSO, or by making them smaller in Ridge Regression.

Other techniques that do not add a regularization term to the loss function, but help improve model accuracy:
- [dropout](https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5): a method to *drop out* (ignore) neurons in a (deep) neural network at every batch during training, while using all neurons at validation. Dropout was shown to increase generalization by forcing the model to train without relying on specific weights.
- [layer, instance, batch, and group normalization](https://arxiv.org/abs/1803.08494): methods to reduce internal covariate shift by normalizing the output of activation layers to the mean and variance of a standard normal distribution. They can also learn scale and shift parameters, which scale the standard deviation and shift the mean, for improved performance over the standard normal distribution.

## Matrix Factorization

[Matrix factorization](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems)) can be used to discover underlying latent factors and/or to predict missing values of the matrix. It became widely known during the [Netflix Prize](https://en.wikipedia.org/wiki/Netflix_Prize) competition, started in 2006, where it predicted the ratings of movies by users better than classic nearest-neighbor techniques ([Koren et al., 2009]({{ site.assets }}/linear-regression-matrix-factorization/recommender-systems-netflix.pdf)). The goal is to find $$W$$ and $$Z$$ such that $$ X \approx WZ^T $$, i.e. to predict $$x_{dn}$$, where $$d$$ indexes the rows of $$W$$ and $$n$$ the rows of $$Z$$. For movie ratings, $$Z$$ could be users, $$W$$ could be movies, and $$x_{dn}$$ the star rating.

The aim is to optimize:

$$
min_{W,Z} L (W,Z) = \frac{1}{2} \sum_{(d,n) \in \Omega} [ x_{dn} - (WZ^T)_{dn}]^2
$$

where $$ W \in R^{D \times K} $$ and $$ Z \in R^{N \times K} $$ are tall matrices, and $$ \Omega \subseteq [D] \times [N] $$ collects the indices of the observed ratings of the input matrix $$X$$. The following picture illustrates the factorization:

{: .picture}
<img width="44.1%" height="44.1%"  src="{{ site.assets }}/linear-regression-matrix-factorization/matrix-factorization.svg">

{: .caption}
Matrix factorization $$X \approx WZ^T$$ of the ratings of $$D = 7$$ movies by $$N = 7$$ users, with 11 observed ratings and $$K = 3$$ latent features. The rating $$x_{dn}$$ of movie $$d$$ by user $$n$$ is approximated by the dot product of row $$d$$ of $$W$$, the features of movie $$d$$, and column $$n$$ of $$Z^T$$, the features of user $$n$$.

This cost function is neither convex nor [identifiable](https://en.wikipedia.org/wiki/Identifiability): for any invertible $$K \times K$$ matrix $$Q$$, the factors $$WQ$$ and $$Z(Q^{-1})^T$$ give the same product $$WZ^T$$. $$K$$ is the number of **latent features** (e.g. gender, age group, citizenship, etc.). A large $$K$$ facilitates overfitting. A regularizer can be added to the function:

$$
min_{W,Z} L (W,Z) = \frac{1}{2} \sum_{(d,n) \in \Omega} [ x_{dn} - (WZ^T)_{dn}]^2 + \frac{\lambda_w}{2} \| W \|^2 +  \frac{\lambda_z}{2} \| Z \|^2
$$ 

where (again) $$\lambda_z$$ and $$\lambda_w$$ are scalars, and the norms are [Frobenius norms](https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm).

Stochastic gradient descent can be used to optimize this problem. Without the regularizer, the training objective is a sum of \|$$\Omega$$\| terms (one per rating):

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

Reminder: $$z_{n,k}$$ are user features, $$w_{d,k}$$ are movie features. The gradient has $$(D+N)K$$ entries, of which only $$2K$$ are non-zero.

**Alternating Least Squares (ALS)** can also be used. ALS factorizes the same matrix $$X$$ into the two factors $$W$$ and $$Z$$ such that $$X \approx WZ^T$$:

- If there are **no** missing ratings in the matrix, i.e. $$ \Omega = [D] \times [N] $$, then:

  $$
  \begin{align*}
  min_{W,Z} L(W,Z) = & \frac{1}{2} \sum_{d=1}^D \sum_{n=1}^N [x_{dn} - (WZ^T)_{dn}]^2 + \text{ (regularizer) } \\
  = & \frac{1}{2} \| X - WZ^T \| ^2 + \frac{\lambda_w}{2} \| W \|^2 +  \frac{\lambda_z}{2} \| Z \|^2
  \end{align*}
  $$

  - coordinate descent (minimize over $$W$$ for a fixed $$Z$$, and vice versa) can be used to minimize the cost plus the regularizer, with the closed-form updates $$W = XZ (Z^TZ + \lambda_w I_K)^{-1}$$ for a fixed $$Z$$, and $$Z = X^TW (W^TW + \lambda_z I_K)^{-1}$$ for a fixed $$W$$;
- If there are missing entries, the problem is harder, as only the ratings $$ (d,n) \in \Omega $$ contribute to the cost, i.e.:

  $$
  min_{W,Z} L(W,Z) = \frac{1}{2} \sum_{(d,n) \in \Omega} [x_{dn} - (WZ^T)_{dn}]^2 + ...
  $$

  - each row $$w_d$$ of $$W$$ is then computed from its observed ratings only, as $$w_d = (\sum_{n:(d,n) \in \Omega} z_n z_n^T + \lambda_w I_K)^{-1} \sum_{n:(d,n) \in \Omega} x_{dn} z_n$$, and similarly for each row $$z_n$$ of $$Z$$.

SGD is in general easier and faster than ALS. However, ALS is better suited to parallel computing, as each row of $$W$$ (or of $$Z$$) is computed independently of the other rows (source: [Koren et al., 2009]({{ site.assets }}/linear-regression-matrix-factorization/recommender-systems-netflix.pdf)).


### Coordinate Descent 

The [**coordinate descent**](https://en.wikipedia.org/wiki/Coordinate_descent) optimization method iterates over individual coordinates (keeping the others fixed) to minimize the loss function, as illustrated below:


{: .picture}
<img width="36.4%" height="36.4%" src="{{ site.assets }}/linear-regression-matrix-factorization/coordinate_descent.svg">

{: .caption}
Coordinate descent on $$f(x, y) = 5x^2 - 6xy + 5y^2$$ from $$(-0.5, -1)$$: each step minimizes $$f$$ along one coordinate, alternating $$y$$ and $$x$$, and the iterates converge to the minimum at the origin. Adapted from: [Coordinate descent](https://en.wikipedia.org/wiki/Coordinate_descent)


The main advantage is that it is extremely simple to implement and does not require any knowledge of the derivative of the function. It is useful for extremely complicated functions or functions whose derivatives are far more expensive to compute than the function itself. However, due to its sequential nature, it is not a good candidate for parallelism, unless several coordinates can be minimized independently, as the rows of each factor in ALS. Another issue arises with non-smooth multivariable functions: it may get stuck at a non-stationary point if the level curves of the function are not smooth (source: [Wikipedia](https://en.wikipedia.org/wiki/Coordinate_descent#Limitations)).


## Risk and Loss Functions

In ML, for non-probabilistic models, the principle of [empirical risk minimization](https://en.wikipedia.org/wiki/Empirical_risk_minimization) is followed to find good parameters. The **risk** is the expected loss between the true output $$y_n$$ and the predicted value $$\hat{y}_n$$, i.e.

$$
r(f) = \mathbb{E}[l(\hat{y}_n, y_n)]
$$

where the predictor $$\hat{y}_n = f(x_n, \theta^*)$$ represents the output of the model $$f$$ with input data $$x_n$$ and parameters $$\theta^*$$. In practice, the average loss on the training data (the *empirical* risk) is minimized.

In the case of a statistical model, [maximum likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) finds the parameters for which the model fits the data well, by minimizing the negative log-likelihood:

$$
L_x(θ) = −log p(x | θ) .
$$

For linear regression with Gaussian noise, minimizing the negative log-likelihood is equivalent to the Least Squares problem above, as derived in the [Bayesian linear regression post]({{ site.baseurl }}{% post_url 2018-08-20-bayesian-linear-regression %}).

The choice of the loss function depends on the task. The following sections present the most common ones: the MSE for regression, the cross-entropy for classification, the triplet and contrastive losses to learn [embeddings](https://en.wikipedia.org/wiki/Embedding_(machine_learning)), the CTC loss for unsegmented sequences, and the focal and Dice losses for imbalanced classes and image segmentation.

### Mean Squared Error

Most regression tasks use the Mean Squared Error loss function:

$$
MSE(\hat{y}_n) = \mathbb{E}\left[ (y_n - \hat{y}_n)^2 \right],
$$

or the Mean Absolute Error, which is similar to the MSE but uses the absolute value of the difference between $$y$$ and $$\hat{y}$$ instead of its square.

### Cross-Entropy

With [one-hot](https://en.wikipedia.org/wiki/One-hot) labels, the [cross-entropy](https://en.wikipedia.org/wiki/Cross-entropy) is equivalent to the negative log-likelihood (of the [softmax](https://en.wikipedia.org/wiki/Softmax_function), when the output of the model is not a distribution). However, to avoid numerical instabilities and overflows, implementations compute it directly from the output of the model, with the [log-sum-exp](https://en.wikipedia.org/wiki/LogSumExp) trick. Most classification tasks use the binary or multi-class cross-entropy loss functions:
- binary: $$H(p) = -(y \log p(x) + (1-y) \log (1-p(x)))$$, i.e. it simplifies to $$-\log p(x)$$ for true label $$y=1$$, and to $$-\log (1-p(x))$$ for true label $$y=0$$.
- multi-class: $$H(q,p) = -\sum_x q(x) \log(p(x))$$, for the true distribution $$q$$ and the predicted distribution $$p$$.

### Triplet Loss

**[Triplet loss](https://en.wikipedia.org/wiki/Triplet_loss)** is used in classification tasks where the number of classes is very large. Triplet loss helps by learning distributed embedding representations of data points such that, in the high-dimensional vector space, contextually similar data points are projected nearby, while dissimilar data points are projected far away from each other.

The network is trained against triplets of images: an **anchor image** of a person, a **positive image**, which is another image of the same person, and a **negative image**, which is a picture of another person. The Triplet Loss minimizes the Euclidean distance between the anchor and the positive, and maximizes the distance between the anchor and the negative. Tasks that can be performed: face recognition ("who is this person?"), verification ("are these the same person?") and clustering ("find the most similar people"). The formulation is:

$${\displaystyle {\mathcal {L}}\left(x,x^{+},x^{-}\right)=\operatorname {max} \left({\|\operatorname {f} \left(x\right)-\operatorname {f} \left(x^{+}\right)\|}^{2}-{\|\operatorname {f} \left(x\right)-\operatorname {f} \left(x^{-}\right)\|}^{2}+m ,0\right)}$$

where $$x$$ is an anchor input, $$x^{+}$$ is a positive input of the same class as $$x$$, $$x^{-}$$ is a negative input of a different class from $$x$$, $$m$$ is a margin between positive and negative pairs, and $$f$$ is an embedding. More details are given in the original paper, [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832).

### Contrastive Loss

**[Contrastive loss](https://en.wikipedia.org/wiki/Siamese_neural_network)** is similar to, and often confused with, the triplet loss. Yet, they solve different problems: for known similarity relationships, Contrastive loss is used. For only negative/positive relationships (like in face recognition, where people's identity is the anchor), Triplet loss is used. In practice, the triplet loss considers the anchor-neighbor-distant triplets, while the contrastive loss deals with the anchor-neighbor and anchor-distant pairs of samples.

The contrastive loss trains siamese networks against pairs of inputs labelled as similar or dissimilar (1 or 0). Contrastive loss is formulated in terms of the similarity label $$y$$ (1 for similar, 0 for dissimilar) and the Euclidean distance $$\delta$$ between the embeddings of two images, as:

$$
L = \frac{1}{2} ⋅y⋅\delta^2 + \frac{1}{2} ⋅(1−y)⋅max(0,m−\delta)^2
$$

where $$m$$ is a hyper-parameter (the margin). The original paper is [Dimensionality Reduction by Learning an Invariant Mapping](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf), summarized in the <a href="{{ site.publications_permalink }}">publications bookmark</a>.


### Connectionist Temporal Classification loss

**[Connectionist Temporal Classification (CTC)](https://en.wikipedia.org/wiki/Connectionist_temporal_classification)** is a classifier and loss function for noisy, unsegmented sequential input data, used to train [recurrent neural networks](https://en.wikipedia.org/wiki/Recurrent_neural_network) (RNNs) such as [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) networks to tackle sequence problems where the timing is variable. CTC was published by Graves et al. in 2006, in [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://www.cs.toronto.edu/~graves/icml_2006.pdf).

At each time step, the network outputs the probability of each label and of an extra *blank* label (no label). The probability of a target sequence is the sum of the probabilities of all the alignments (one output per time step) that give the target once repeated labels and blanks are removed. The CTC loss is the negative logarithm of this probability, computed efficiently by dynamic programming.

### Focal Loss

**[Focal Loss](https://arxiv.org/abs/1708.02002)** is an extension of the Cross-Entropy Loss designed to address class imbalance, a common issue in e.g. dense [object detection](https://en.wikipedia.org/wiki/Object_detection) and medical image segmentation, where the background (non-lesion regions) is much more prevalent than the foreground (lesion regions). Focal Loss focuses on hard-to-classify examples and down-weights the loss contribution from easy examples. For binary classification, it is defined as (Eq. 5 in the paper):

$$
\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

where:
- $$p_t$$ is the model’s estimated probability for the true class.
- $$\alpha_t$$ is a balancing factor, equal to $$α$$ for the positive class and to $$1-α$$ for the negative class (especially useful for imbalanced datasets).
- $$γ$$ is the focusing parameter, usually set to 2, which adjusts the focus on hard-to-classify examples.

In practice, the factor $$(1 - p_t)^\gamma$$ is close to 0 for well-classified examples, where $$p_t$$ is close to 1, so their loss is down-weighted, and close to 1 for misclassified examples. With $$\gamma = 0$$, the focal loss is the cross-entropy weighted by $$\alpha_t$$ (Eq. 3 in the paper). For multi-class classification, the same formula applies, with one balancing factor $$\alpha_t$$ per class $$t$$.

### Dice Loss

**[Dice loss](https://arxiv.org/abs/1707.03237)** (or Dice Similarity Coefficient Loss) is a loss function used specifically for [image segmentation](https://en.wikipedia.org/wiki/Image_segmentation) tasks, to evaluate the overlap between predicted and ground truth segmentation masks, and was first used to train neural networks in [V-Net](https://arxiv.org/abs/1606.04797). It is based on the [Dice coefficient](https://en.wikipedia.org/wiki/Dice-S%C3%B8rensen_coefficient), which measures the similarity between two sets, with values ranging from 0 (no overlap) to 1 (perfect overlap). Dice Loss is typically used when precise object boundaries are important, such as segmenting organs, tumors, or any region of interest in medical imaging where the goal is to accurately delineate the object of interest from the surrounding tissue. Key benefits of Dice Loss:
- Direct Focus on Overlap: Dice loss directly optimizes the overlap between predicted and ground truth masks, which is crucial for segmentation tasks where precise boundary delineation matters.
- Effective for Imbalanced Classes: It works well when there is a significant class imbalance between foreground and background, as it evaluates segmentation performance based on the overlap of the foreground regions rather than pixel accuracy.
- Works Well with Small Objects: Dice loss is particularly effective in medical imaging tasks where the objects of interest (e.g. tumors, organs) might occupy a small portion of the image.

$$
\mathcal{L}_{\text{Dice}} = 1 - \frac{2 |A \cap B|}{|A| + |B|}
$$

where:
- $$A$$ is the set of predicted pixels.
- $$B$$ is the set of ground truth pixels.
- $$ \mid A \cap B \mid$$ is the size of the intersection of the predicted and ground truth sets, representing the overlap.

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

where $$C$$ is the number of classes, and $$p_{i,c}$$ and $$g_{i,c}$$ are the predicted and ground truth values of the $$i$$-th pixel for class $$c$$.

To find more about the topic, the book [Mathematics for Machine Learning](https://mml-book.github.io/) by Deisenroth, Faisal and Ong covers empirical risk minimization and parameter estimation (chapter 8), and the maximum likelihood and maximum a posteriori estimators of linear regression (chapter 9).


