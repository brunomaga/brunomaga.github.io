---
layout: post
title:  "Closed-form Linear Regression and Matrix Factorization, and loss functions"
categories: ["Machine Learning"]
tags: ["supervised learning", "linear regression", "matrix factorization", "loss functions"]
---

Regression relates an input variable to an output, either to predict new outputs or to understand the effect of the input on the output. A regression dataset consists of $$N$$ pairs $$(x_n, y_n)$$, with input $$x_n$$ and output/label $$y_n$$. For a new input $$x_n$$, the goal of regression is to find $$f$$ such that $$ y_n \approx f(x_n) $$. Fitting a multidimensional plane to the dataset is the (multivariate) linear regression problem, which consists of finding the weights $$w$$ such that:

$$
y_n \approx f(x_n) = w_0 + w_1 x_{n1} + ... + w_D x_{nD} \hspace{1cm}\text{, or simply}\hspace{1cm} y \approx f(X) = Xw.
$$

An error or loss function $$L(w)$$ measures the quality of the estimation. For example, the Mean Squared Error (MSE) loss is defined as $$ L(w) = \frac{1}{N} \sum_{n=1}^N [ y_n - f(x_n)]^2$$. Due to the square term, it penalizes predictions with larger errors more heavily. The Mean Absolute Error (MAE) loss, $$ L(w) = \frac{1}{N} \sum_{n=1}^N \mid y_n - f(x_n) \mid $$, is more robust to outliers.

The goal is to find the weight values that minimize the loss function. A naive approach is Grid Search, a brute-force method that tests all combinations of $$w$$ in a given interval. It is usually inefficient and only works on very low-dimensional problems. An alternative is to compute the gradient of the loss function and step $$w$$ towards the values that minimize the loss. The most common methods are gradient descent and its variants:
- Gradient Descent: $$ w^{t+1} = w^{t} - \gamma \triangledown L (w^t) $$, for step size $$\gamma$$ and gradient $$ \triangledown L (w) = [ \frac{\partial L(w)}{\partial w_1}, ... , \frac{\partial L(w)}{\partial w_D}  ]^T $$;
- Stochastic Gradient Descent: $$ w^{t+1} = w^{t} - \gamma \triangledown L_n (w^t) $$, for a randomly chosen input $$n$$. It gives a computationally cheap but unbiased estimate of the gradient;
- Mini-batch SGD: $$ w^{t+1} = w^{t} - \gamma \frac{1}{\mid B \mid} \sum_{n \in B} \triangledown L_n (w^t) $$, for a random subset $$ B \subseteq [N] $$. The gradient of every sample $$n$$ is computed at the same current point $$w^t$$.

For linear regression with the Mean Squared Error loss function, the problem is called **Least Squares** and has a closed-form solution. The quantity to minimize is:

  $$
  \begin{align*}
  \| y - Xw \|^2 = \, & (y-Xw)^T(y-Xw) \\
  = \, & y^Ty - y^TXw - (Xw)^Ty + (Xw)^TXw \\
  = \, & y^Ty - w^TX^Ty - w^TX^Ty + w^TX^TXw \\
  = \, & y^Ty - 2 w^TX^Ty + w^TX^TXw\\
  \end{align*}
  $$

The solution is the $$w$$ that minimizes the loss or, equivalently, the $$w$$ at which its derivative is 0:

  $$
  \begin{align*}
    & \frac{\partial}{\partial w} \left( y^Ty - 2 w^TX^Ty + w^TX^TXw \right) = 0 \\
  \Leftrightarrow \, & 0 - 2 X^Ty + 2 X^TXw = 0 \\
  \Leftrightarrow \, & 2 X^TXw = 2X^Ty \\
  \Leftrightarrow \, & w = (X^TX)^{-1} X^Ty\\
  \end{align*}
  $$

This derivation uses the identity $$\frac{\partial w^Ta}{\partial w} = \frac{\partial a^Tw}{\partial w} = a$$ (<a href="{{ site.assets }}/resources/the_matrix_cookbook.pdf">The Matrix Cookbook</a>, section 2.4.1, eq. 69). The result shows that **if $$X^TX$$ is invertible**, the minimization problem has a unique closed-form solution, given by the final expression above. The **Gram matrix $$X^TX$$** is invertible if **$$X$$ has full column rank**, i.e. $$rank(X)=D$$ (the proof is omitted). The rank of a matrix is the maximum number of linearly independent column vectors in the matrix, so this assumes that all columns are linearly independent.


### Regularization

A common practice is to add to the loss a regularization term $$\Omega$$ that penalizes complex models. The loss minimization problem then becomes:

$$
min_w L(w) + \Omega (w)
$$

Two common regularization approaches are:
- L1 regularization: $$ \Omega(w) = \lambda \mid w \mid $$, where $$\mid w \mid = \sum_{i=1}^D \mid w_i \mid $$ (the bias $$w_0$$ is not penalized) 
  - this is called **LASSO Regression**, for **L**east **A**bsolute **S**hrinkage and **S**election **O**perator;
  - the parameter $$ \lambda \gt 0 $$ is tuned to reduce overfitting. This is the **model selection** problem;
  - it encourages simple, sparse models (i.e. models with fewer parameters). This type of regression is well suited for automating parts of model selection, such as variable selection/parameter elimination. With highly correlated features (multicollinearity), LASSO tends to keep one of them arbitrarily; Ridge Regression or the elastic net (L1 plus L2) are preferred in that case.
- L2 regularization (standard Euclidean norm): $$ \Omega(w) = \lambda \| w \|^2 $$, where $$ \| w \|^2 = \sum_{i=1}^D w_i^2 $$
  - large weights are penalized, as they are *considered unlikely*;
  - if $$L$$ is the MSE, this is called **Ridge Regression**;
  - as in the Least Squares problem, the MSE loss with Ridge Regression has an analytical solution:
    - minimize $$(y - Xw)^T(y-Xw) + \lambda w^Tw$$, and
    - differentiate with respect to $$w$$ to obtain the normal equation $$w = (X^TX + \lambda I)^{-1} X^Ty$$. This form also penalizes $$w_0$$; excluding the bias sets the first diagonal entry of $$I$$ to 0.

Note: in statistics, **shrinkage** is the reduction of the effects of sampling variation on the estimated parameters. Without it, a model can overfit the training data and perform poorly on unseen data. Regularization reduces this effect by "shrinking" model parameters towards 0 in LASSO, or by making them smaller in Ridge Regression.

Other techniques improve model accuracy without adding a regularization term to the loss function:
- [dropout](https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5): a method that *drops out* (ignores) neurons of a (deep) neural network at every batch during training, and uses all neurons at validation. It improves generalization by forcing the model to train without relying on specific weights (see the [separate post]({{ site.baseurl }}{% post_url 2018-03-27-deep-neural-networks %}) for more details).
- [layer, instance, batch, and group normalization](https://arxiv.org/abs/1803.08494): methods that reduce internal covariate shift by fitting the output of activation layers to a standard normal distribution. They can also learn scale and shift parameters, which scale the standard deviation and shift the mean, for better performance than the standard normal distribution.

## Matrix Factorization

Matrix factorization can discover underlying latent factors and/or predict missing values of a matrix. The goal is to find $$W$$ and $$Z$$ such that $$ X \approx WZ^T $$, i.e. to predict $$x_{dn}$$, where $$d$$ indexes the rows of $$W$$ and $$n$$ the rows of $$Z$$. For movie ratings, $$Z$$ could be users, $$W$$ could be movies, and $$x_{dn}$$ the star rating.

The optimization problem is:

$$
min_{W,Z} L (W,Z) = \frac{1}{2} \sum_{(d,n) \in \Omega} [ x_{dn} - (WZ^T)_{dn}]^2
$$

where $$ W \in R^{D \times K} $$ and $$ Z \in R^{N \times K} $$ are tall matrices, and $$ \Omega \subseteq [D] \times [N] $$ contains the indices of the observed ratings of the input matrix $$X$$.

{: .picture}
<img width="60%"  src="{{ site.assets }}/linear-regression-matrix-factorization/matrix-factorization.svg">

This cost function is neither convex nor [identifiable](https://en.wikipedia.org/wiki/Identifiability). $$K$$ is the number of latent features (e.g. gender, age group, citizenship, etc.). A large $$K$$ facilitates overfitting. A regularizer can be added to the function:

$$
min_{W,Z} L (W,Z) = \frac{1}{2} \sum_{(d,n) \in \Omega} [ x_{dn} - (WZ^T)_{dn}]^2 + \frac{\lambda_w}{2} \| W \|^2 +  \frac{\lambda_z}{2} \| Z \|^2
$$ 

where $$\lambda_z$$ and $$\lambda_w$$ are again scalars.

Stochastic gradient descent optimizes this problem. The training objective is a sum of \|$$\Omega$$\| terms (one per rating):

$$
\frac{1}{| \Omega |} \sum_{(d,n) \in \Omega} \frac{1}{2} [x_{dn} - (WZ^T)_{dn}]^2
$$

For one fixed element $$(d,n)$$, the gradient entries $$(d',k)$$ for $$W$$ and $$(n',k)$$ for $$Z$$ are:

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

Recall that $$z_{n,k}$$ are user features and $$w_{d,k}$$ are movie features. The gradient has $$(D+N)K$$ entries.

**Alternating Least Squares (ALS)** is another option. ALS is the alternating minimization of the same loss: it fixes $$Z$$ and solves for $$W$$, then fixes $$W$$ and solves for $$Z$$, and repeats. Each step is a regularized least-squares problem with a closed-form solution.

- If the matrix has **no** missing ratings, i.e. $$ \Omega = [D] \times [N] $$, then:

  $$
  \begin{align*}
  min_{W,Z} L(W,Z) = & \frac{1}{2} \sum_{d=1}^D \sum_{n=1}^N [x_{dn} - (WZ^T)_{dn}]^2 + \text{ (regularizer) } \\
  = & \frac{1}{2} \| X - WZ^T \| ^2 + \frac{\lambda_w}{2} \| W \|^2 +  \frac{\lambda_z}{2} \| Z \|^2
  \end{align*}
  $$

  - setting the gradient to zero gives the updates $$W = XZ (Z^TZ + \lambda_w I_K)^{-1}$$ for a fixed $$Z$$, and $$Z = X^TW (W^TW + \lambda_z I_K)^{-1}$$ for a fixed $$W$$;
- If there are missing entries, the problem is harder, because only the ratings $$ (d,n) \in \Omega $$ contribute to the cost, i.e.:

  $$
  min_{W,Z} L(W,Z) = \frac{1}{2} \sum_{(d,n) \in \Omega} [x_{dn} - (WZ^T)_{dn}]^2 + ...
  $$

  - each row $$w_d$$ of $$W$$ is then solved separately, using only its observed entries: $$w_d = (\sum_{n:(d,n) \in \Omega} z_n z_n^T + \lambda_w I_K)^{-1} \sum_{n:(d,n) \in \Omega} x_{dn} z_n$$, and each row $$z_n$$ of $$Z$$ in the same way;


### Coordinate Descent 

The [**coordinate descent**](https://en.wikipedia.org/wiki/Coordinate_descent) optimization method minimizes the loss function by iterating over individual coordinates while keeping the others fixed.


{: .picture}
<img width="40%" height="40%" src="{{ site.assets }}/linear-regression-matrix-factorization/coordinate_descent.svg">

{: .caption}
source: Wikipedia


Its main advantage is that it is very simple to implement and does not require the derivative of the function. It is useful for very complicated functions, or for functions whose derivatives are much more expensive to compute than the function itself. However, due to its sequential nature, it is not a good candidate for parallelism. It also has an issue with non-smooth multivariable functions: it may get stuck at a non-stationary point if the level curves of the function are not smooth (source: [Wikipedia](https://en.wikipedia.org/wiki/Coordinate_descent#Limitations)).


## Risk and Loss Functions

In ML, non-probabilistic models follow the principle of empirical risk minimization to find good parameters. The **risk** is the expected loss between the true output $$y_n$$ and the predicted value $$\hat{y}_n$$, i.e.

$$
r(f) = \mathbb{E}[l(\hat{y}_n, y_n)]
$$

where the predictor $$\hat{y}_n = f(x_n, \theta^*)$$ is the output of the model $$f$$ for input data $$x_n$$ and parameters $$\theta^*$$.

For a statistical model, maximum likelihood estimation finds the parameters for which the model fits the data well, by minimizing the negative log-likelihood:

$$
L_x(θ) = −log p(x | θ) .
$$

Likelihood estimators are covered in a [different post]({{ site.baseurl }}{% post_url 2018-08-20-bayesian-linear-regression %}).

### Mean Squared Error

Most regression tasks use the Mean Squared Error loss function:

$$
MSE(\hat{y}_n) = \mathbb{E}\left[ (y_n - \hat{y}_n)^2 \right],
$$

or the Mean Absolute Error, which is similar to the MSE but uses the absolute difference between $$y$$ and $$\hat{y}$$ instead of the squared difference.

### Cross-Entropy

The cross-entropy with one-hot labels equals the negative log-likelihood (of the softmax, when the output of the model is not a distribution). Implementations compute it directly from the logits, fusing the log and the softmax (log-sum-exp). This avoids the overflows and underflows of computing the softmax first and its log afterwards. Most classification tasks use the binary or multi-class cross-entropy loss functions:
- binary: $$H(p) = -(y \log p(x) + (1-y) \log (1-p(x)))$$. It simplifies to $$-\log p(x)$$ for true label $$y=1$$, and to $$-\log (1-p(x))$$ for true label $$y=0$$.
- multi-class: $$H(p,q) = -\sum_x p(x) \log(q(x))$$, where $$p$$ is the true distribution and $$q$$ the predicted distribution.

### Triplet Loss

**[Triplet loss](https://en.wikipedia.org/wiki/Triplet_loss)** is used in classification tasks with a very large number of classes. It learns distributed embedding representations of data points such that, in the high-dimensional vector space, contextually similar data points are projected nearby, and dissimilar data points are projected far apart. The network is trained on triplets of images: an **anchor image** of a person, a **positive image**, which is another image of the same person, and a **negative image**, which is a picture of another person. The Triplet Loss minimizes the Euclidean distance between the anchor and the positive, and maximizes the distance between the anchor and the negative. Typical tasks are face recognition ("who's this person?"), verification ("are these the same person?") and clustering ("find the most similar people"). The formulation is:

$${\displaystyle {\mathcal {L}}\left(A,P,N\right)=\operatorname {max} \left({\|\operatorname {f} \left(A\right)-\operatorname {f} \left(P\right)\|}^{2}-{\|\operatorname {f} \left(A\right)-\operatorname {f} \left(N\right)\|}^{2}+\alpha ,0\right)}$$

where $$A$$ is an anchor input, $$P$$ is a positive input of the same class as $$A$$, $$N$$ is a negative input of a different class from $$A$$, $$\alpha$$ is a margin between positive and negative pairs, and $$f$$ is an embedding. The details are in the original paper, [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832).

### Contrastive Loss

**[Contrastive loss](https://en.wikipedia.org/wiki/Siamese_neural_network)** is similar to, and often confused with, the triplet loss. However, the two solve different problems. Contrastive loss is used for known similarity relationships. Triplet loss is used for only negative/positive relationships (as in face recognition, where the identity of a person is the anchor). The triplet loss considers anchor-neighbor-distant triplets, while the contrastive loss considers anchor-neighbor and anchor-distant pairs of samples. The contrastive loss trains siamese networks on pairs of inputs labelled as similar or dissimilar (1 or 0). It is formulated in terms of the similarity label $$y$$ (1 for similar, 0 for dissimilar) and the Euclidean distance $$D$$ between two images:

$$
L = \frac{1}{2} ⋅y⋅D^2 + \frac{1}{2} ⋅(1−y)⋅max(0,m−D)^2
$$

where $$m$$ is a hyper-parameter. The original paper is [Dimensionality Reduction by Learning an Invariant Mapping](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf), and a summary is available in the <a href="{{ site.publications_permalink }}">publications bookmark</a>.


### Connectionist Temporal Classification loss

**[Connectionist Temporal Classification (CTC)](https://en.wikipedia.org/wiki/Connectionist_temporal_classification)** is a classifier and loss function for noisy, unsegmented sequential input data. It is used to train recurrent neural networks (RNNs), such as LSTM networks, on sequence problems where the timing is variable. It was published in [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://www.cs.toronto.edu/~graves/icml_2006.pdf).

### Focal Loss

**[Focal Loss](https://arxiv.org/abs/1708.02002)** is an extension of the Cross-Entropy Loss that addresses class imbalance. Class imbalance is common in e.g. medical image segmentation, where the background (non-lesion regions) is much more prevalent than the foreground (lesion regions). Focal Loss focuses on hard-to-classify examples and down-weights the loss contribution of easy examples.

$$
\mathcal{L}_{\text{focal}} = -\alpha (1 - p_t)^\gamma \log(p_t)
$$

where:
- $$p_t$$ is the estimated probability of the model for the true class.
- $$α$$ is a balancing factor that adjusts the weight of the positive class (especially useful for imbalanced datasets).
- $$γ$$ is the focusing parameter, usually set to 2, which controls the focus on hard-to-classify examples.

For multi-class classification, Focal Loss generalizes to:

$$
\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

where $$\alpha_t$$ is the balancing factor for class $$t$$.

### Dice Loss

**[Dice loss](https://arxiv.org/abs/1707.03237)** (or Dice Similarity Coefficient Loss) is a loss function for image segmentation tasks. It evaluates the overlap between predicted and ground truth segmentation masks. It is based on the Dice coefficient, which measures the similarity between two sets, with values from 0 (no overlap) to 1 (perfect overlap). Dice Loss is typically used when precise object boundaries are important, e.g. when segmenting organs, tumors, or any region of interest in medical imaging, where the object of interest must be accurately separated from the surrounding tissue. Its main benefits are:
- Direct Focus on Overlap: Dice loss directly optimizes the overlap between predicted and ground truth masks, which is crucial for segmentation tasks that require precise boundaries.
- Effective for Imbalanced Classes: it performs well under a significant class imbalance between foreground and background, as it measures segmentation performance by the overlap of the foreground regions rather than by pixel accuracy.
- Works Well with Small Objects: Dice loss is particularly effective in medical imaging tasks where the objects of interest (e.g., tumors, organs) occupy a small portion of the image.

$$
\mathcal{L}_{\text{Dice}} = 1 - \frac{2 |A \cap B|}{|A| + |B|}
$$

where:
- $$A$$ is the set of predicted pixels.
- $$B$$ is the set of ground truth pixels.
- $$ \mid A \cap B \mid$$ is the size of the intersection of the predicted and ground truth sets, i.e. the overlap.

In practice, Dice Loss is often computed as:

$$
\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_{i=1}^{N} p_i g_i}{\sum_{i=1}^{N} p_i + \sum_{i=1}^{N} g_i}
$$

where:
- $$p_i$$ is the predicted value for the $$i$$-th pixel.
- $$g_i$$ is the ground truth value for the $$i$$-th pixel.

For multi-class segmentation, a generalized form of Dice Loss is:

$$
\mathcal{L}_{\text{Dice}} = 1 - \frac{1}{C} \sum_{c=1}^{C} \frac{2 \sum_{i=1}^{N} p_{i,c} \, g_{i,c}}{\sum_{i=1}^{N} p_{i,c} + \sum_{i=1}^{N} g_{i,c}}
$$

where $$C$$ is the number of classes, and $$p_{i,c}$$ and $$g_{i,c}$$ are the predicted and ground truth values of the $$i$$-th pixel for class $$c$$.


