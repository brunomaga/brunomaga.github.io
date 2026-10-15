---
layout: post
title:  "Closed-form Linear Regression and Matrix Factorization, and loss functions"
categories: ["ML Foundations"]
tags: ["supervised learning", "linear regression", "matrix factorization", "loss functions"]
---

Regression relates an input variable to an output, either to predict new outputs or to understand the effect of the input on the output. A regression dataset consists of a set of $$N$$ pairs $$(x_n, y_n)$$, with input $$x_n$$ and output/label $$y_n$$. For a new input $$x_n$$, the goal of regression is to find $$f$$ such that $$ y_n \approx f(x_n) $$. If we want to fit a multidimensional plane to the dataset, we are solving the (multivariate) linear regression problem, by finding the weights $$w$$ such that:

$$
y_n \approx f(x_n) = w_0 + w_1 x_{n1} + ... + w_D x_{nD} \hspace{1cm}\text{, or simply}\hspace{1cm} y \approx f(X) = Xw.
$$

We use an error or loss function $$L(w)$$ to indicate how good our estimation is. As an example, the Mean Squared Error (MSE) loss is defined as $$ L(w) = \frac{1}{N} \sum_{n=1}^N [ y_n - f(x_n)]^2$$, and penalizes predictions with larger errors more heavily, due to the square term. Another loss function, the Mean Absolute Error (MAE), with $$ L(w) = \frac{1}{N} \sum_{n=1}^N \mid y_n - f(x_n) \mid $$, is more robust to outliers.

In brief, we want to find the weight values that minimize the loss function. A naive approach is Grid Search, a brute-force method that tests all combinations of $$w$$ in a given interval. This is usually inefficient and only works on very low-dimensional problems. An alternative is to optimize the loss function by computing its gradient and stepping $$w$$ towards the values that minimize the loss. The most common methods are gradient descent and its variants:
- Gradient Descent: $$ w^{t+1} = w^{t} - \gamma \triangledown L (w^t) $$, for step size $$\gamma$$, and gradient $$ \triangledown L (w) = [ \frac{\partial L(w)}{\partial w_1}, ... , \frac{\partial L(w)}{\partial w_D}  ]^T $$;
- Stochastic Gradient Descent: $$ w^{t+1} = w^{t} - \gamma \triangledown L_n (w^t) $$, for a randomly chosen input $$n$$. A computationally cheap but unbiased estimate of the gradient;
- Mini-batch SGD: $$ w^{t+1} = w^{t} - \gamma \frac{1}{\mid B \mid} \sum_{n \in B} \triangledown L_n (w^t) $$, for a random subset $$ B \subseteq [N] $$. For each sample $$n$$, we compute the gradient at the same current point $$w^t$$.

For the particular case of linear regression with the Mean Squared Error loss function, the problem is called **Least Squares** and has a closed-form solution. In practice, we want to minimize the following:

  $$
  \begin{align*}
  \| y - Xw \|^2 = \, & (y-Xw)^T(y-Xw) \\
  = \, & y^Ty - y^TXw - (Xw)^Ty + (Xw)^TXw \\
  = \, & y^Ty - w^TX^Ty - w^TX^Ty + w^TX^TXw \\
  = \, & y^Ty - 2 w^TX^Ty + w^TX^TXw\\
  \end{align*}
  $$

We want to compute the $$w$$ that minimizes the loss or, equivalently, the $$w$$ where its derivative is 0. So we compute:

  $$
  \begin{align*}
    & \frac{\partial}{\partial w} \left( y^Ty - 2 w^TX^Ty + w^TX^TXw \right) = 0 \\
  \Leftrightarrow \, & 0 - 2 X^Ty + 2 X^TXw = 0 \\
  \Leftrightarrow \, & 2 X^TXw = 2X^Ty \\
  \Leftrightarrow \, & w = (X^TX)^{-1} X^Ty\\
  \end{align*}
  $$

Note that we used the trick $$\frac{\partial w^Ta}{\partial w} = \frac{\partial a^Tw}{\partial w} = a$$ (<a href="{{ site.assets }}/resources/the_matrix_cookbook.pdf">The Matrix Cookbook</a>, section 2.4.1, eq. 69). The result tells us that **if $$X^TX$$ is invertible**, this minimization problem has a unique closed-form solution, given by that final form. As a side note, the **Gram matrix $$X^TX$$** is invertible if **$$X$$ has full column rank**, i.e. $$rank(X)=D$$ (we'll omit the proof). The rank of a matrix is defined as the maximum number of linearly independent column vectors in the matrix, so we are assuming that all columns are linearly independent.


### Regularization

It's common practice to add to the loss a regularization term $$\Omega$$ that penalizes complex models. Our loss minimization problem then becomes:

$$
min_w L(w) + \Omega (w)
$$

Common regularization approaches are:
- L1 regularization: $$ \Omega(w) = \lambda \mid w \mid $$, where $$\mid w \mid = \sum_{i=0}^D \mid w_i \mid $$ 
  - this is called **LASSO Regression** as in **L**east **A**bsolute **S**hrinkage and **S**election **O**perator.
  - the parameter $$ \lambda \gt 0 $$ can be tuned to reduce overfitting. This is the **model selection** problem.
  - encourages simple, sparse models (i.e. models with fewer parameters). This particular type of regression is well-suited for models showing high levels of multicollinearity or when you want to automate certain parts of model selection, like variable selection/parameter elimination.
- L2 regularization (standard Euclidean norm): $$ \Omega(w) = \lambda \| w \|^2 $$, where $$ \| w \|^2 = \sum_{i=0}^D w_i^2 $$
  - Large weights will be penalized, as they are *considered unlikely*;
  - If $$L$$ is the MSE, this is called **Ridge Regression**;
  - similarly to the Least Squares problem, the MSE loss with Ridge Regression has an analytical solution:
    - computed by minimizing $$(y - Xw)^T(y-Xw) + \lambda w^Tw$$, and
    - differentiating with respect to $$w$$ leads to the normal equation $$w = (X^TX + \lambda I)^{-1} X^Ty$$.

Note: in statistics, **shrinkage** is the reduction in the effects of sampling variation, e.g. a model overfitting the training data and not performing well on unseen data. Regularization attempts to do this by "shrinking" model parameters towards 0 in LASSO, or by making them smaller in Ridge Regression.

Other techniques that do not add a regularization term to the loss function, but help improve model accuracy: 
- [dropout](https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5): a method to *drop out* (ignore) neurons in a (deep) neural network at every batch during training, while using all neurons at validation. Shown to increase generalization by forcing the model to train without relying on specific weights (see the [separate post]({{ site.baseurl }}{% post_url 2018-03-27-deep-neural-networks %}) for more details).
- [layer, instance, batch, and group normalization](https://arxiv.org/abs/1803.08494): methods to reduce internal covariate shift by fitting the output of activation layers to a standard normal distribution. They can also learn scale and shift parameters, which scale the standard deviation and shift the mean, for improved performance over the standard normal distribution.  

## Matrix Factorization

Matrix factorization can be used to discover underlying latent factors and/or to predict missing values of the matrix. We aim to find $$W$$ and $$Z$$ such that $$ X \approx WZ^T $$, i.e. we aim to predict $$x_{dn}$$, where $$d$$ indexes the rows of $$W$$ and $$n$$ the rows of $$Z$$. For movie ratings, $$Z$$ could be users, $$W$$ could be movies, and $$x_{dn}$$ the star rating.

We aim at optimizing:

$$
min_{W,Z} L (W,Z) = \frac{1}{2} \sum_{(d,n) \in \Omega} [ x_{dn} - (WZ^T)_{dn}]^2
$$

where $$ W \in R^{D \times K} $$ and $$ Z \in R^{N \times K} $$ are tall matrices, and $$ \Omega \subseteq [D] \times [N] $$ collects the indices of the observed ratings of the input matrix $$X$$.

{: style="text-align:center; font-size: small;"}
<img width="60%"  src="{{ site.assets }}/linear-regression-matrix-factorization/matrix-factorization.png">

This cost function is neither convex nor [identifiable](https://en.wikipedia.org/wiki/Identifiability). $$K$$ is the number of latent features (e.g. gender, age group, citizenship, etc.). A large $$K$$ facilitates overfitting. We can add a regularizer to the function:

$$
min_{W,Z} L (W,Z) = \frac{1}{2} \sum_{(d,n) \in \Omega} [ x_{dn} - (WZ^T)_{dn}]^2 + \frac{\lambda_w}{2} \| W \|^2 +  \frac{\lambda_z}{2} \| Z \|^2
$$ 

where (again) $$\lambda_z$$ and $$\lambda_w$$ are scalars. 

We use stochastic gradient descent to optimize this problem. The training objective is a sum of \|$$\Omega$$\| terms (one per rating):

$$
\frac{1}{| \Omega |} \sum_{(d,n) \in \Omega} \frac{1}{2} [x_{dn} - (WZ^T)_{dn}]^2
$$

For one fixed element $$(d,n)$$, we derive the gradient entries $$(d',k)$$ for $$W$$ and $$(n',k)$$ for $$Z$$:

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

Reminder: $$z_{n,k}$$ are user features, $$w_{d,k}$$ are movie features. The gradient has $$(D+N)K$$ entries.

We can also use **Alternating Least Squares (ALS)**. ALS factorizes a given matrix $$R$$ into two factors $$U$$ and $$V$$ such that $$ R \approx U^TV $$.

- If there are **no** missing ratings in the matrix, i.e. $$ \Omega = [D] \times [N] $$, then:

  $$
  \begin{align*}
  min_{W,Z} L(W,Z) = & \frac{1}{2} \sum_{d=1}^D \sum_{n=1}^N [x_{dn} - (WZ^T)_{dn}]^2 + \text{ (regularizer) } \\
  = & \frac{1}{2} \| X - WZ^T \| ^2 + \frac{\lambda_w}{2} \| W \|^2 +  \frac{\lambda_z}{2} \| Z \|^2
  \end{align*}
  $$

  - We can use coordinate descent (minimize over $$W$$ for a fixed $$Z$$, and vice versa) to minimize the cost plus the regularizer;
- If there are missing entries, the problem is harder, as only the ratings $$ (d,n) \in \Omega $$ contribute to the cost, i.e.:

  $$
  min_{W,Z} L(W,Z) = \frac{1}{2} \sum_{(d,n) \in \Omega} [x_{dn} - (WZ^T)_{dn}]^2 + ...
  $$


### Coordinate Descent 

The [**coordinate descent**](https://en.wikipedia.org/wiki/Coordinate_descent) optimization method iterates over individual coordinates (keeping the others fixed) to minimize the loss function. 


{: style="text-align:center; font-size: small;"}
<img width="40%" height="40%" src="{{ site.assets }}/linear-regression-matrix-factorization/coordinate_descent.svg.png">

{: style="text-align:center; font-size: small;"}
source: Wikipedia


The main advantage is that it is extremely simple to implement and doesn’t require any knowledge of the derivative of the function. It’s really useful for extremely complicated functions or functions whose derivatives are far more expensive to compute than the function itself. However, due to its sequential nature, it's not a good candidate for parallelism. Another issue arises with non-smooth multivariable functions: it may get stuck at a non-stationary point if the level curves of the function are not smooth (source: [Wikipedia](https://en.wikipedia.org/wiki/Coordinate_descent#Limitations)).


## Risk and Loss Functions

In ML, for non-probabilistic models, we follow the principle of empirical risk minimization to find good parameters. The **risk** is the expected loss between the true output $$y_n$$ and the predicted value $$\hat{y}_n$$, i.e.

$$
r(f) = \mathbb{E}[l(\hat{y}_n, y_n)]
$$

where the predictor $$\hat{y}_n = f(x_n, \theta∗)$$ represents the output of the model $$f$$ with input data $$x_n$$ and parameters $$\theta*$$.

In the case of a statistical model, maximum likelihood estimation finds the parameters for which the model fits the data well, by minimizing the negative log-likelihood:

$$
L_x(θ) = −log p(x | θ) .
$$

We'll omit details on likelihood estimators, as they are covered in a [different post]({{ site.baseurl }}{% post_url 2018-08-20-bayesian-linear-regression %}). 

### Mean Squared Error

Most regression tasks use the Mean Squared Error loss function:

$$
MSE(\hat{y}_n) = \mathbb{E}\left[ (y_n - \hat{y}_n)^2 \right],
$$

or the Mean Absolute Error, which is similar to the MSE but uses the absolute value of the difference between $$y$$ and $$\hat{y}$$ instead of its square.

### Cross-Entropy

The cross-entropy is equivalent to the negative log-likelihood (of the softmax, when the output of the model is not a distribution). However, for classification tasks, we use the cross-entropy formulation to avoid numerical instabilities and overflows. Most classification tasks use the binary or multi-class cross-entropy loss functions:
- binary: $$H(p) = -(y \log p(x) + (1-y) \log (1-p(x)))$$, i.e. it simplifies to $$-\log p(x)$$ for true label $$y=1$$, and to $$-\log (1-p(x))$$ for true label $$y=0$$.
- multi-class: $$H(p,q) = -\sum_x p(x) \log(q(x))$$, for the true distribution $$p$$ and the predicted distribution $$q$$.

### Triplet Loss

**[Triplet loss](https://en.wikipedia.org/wiki/Triplet_loss)** is used in classification tasks where the number of classes is very large. Triplet loss helps by learning distributed embedding representations of data points such that, in the high-dimensional vector space, contextually similar data points are projected nearby, while dissimilar data points are projected far away from each other. The network is trained against triplets of images: an **anchor image** of a person, a **positive image**, which is another image of the same person, and a **negative image**, which is a picture of another person. The Triplet Loss minimizes the Euclidean distance between the anchor and the positive, and maximizes the distance between the anchor and the negative. Tasks that we can perform: face recognition ("who's this person?"), verification ("are these the same person?") and clustering ("find the most similar people"). The formulation is:

$${\displaystyle {\mathcal {L}}\left(A,P,N\right)=\operatorname {max} \left({\|\operatorname {f} \left(A\right)-\operatorname {f} \left(P\right)\|}^{2}-{\|\operatorname {f} \left(A\right)-\operatorname {f} \left(N\right)\|}^{2}+\alpha ,0\right)}$$

where $$A$$ is an anchor input, $$P$$ is a positive input of the same class as $$A$$, $$N$$ is a negative input of a different class from $$A$$, $$\alpha$$ is a margin between positive and negative pairs, and $$f$$ is an embedding. For more details, see the original paper [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832).

### Contrastive Loss

**[Contrastive loss](https://en.wikipedia.org/wiki/Siamese_neural_network)** is similar to, and often confused with, the triplet loss. Yet, they solve different problems: for known similarity relationships, we use Contrastive loss. For only negative/positive relationships (like in face recognition, where people's identity is the anchor), Triplet loss is used. In practice, the triplet loss considers the anchor-neighbor-distant triplets, while the contrastive loss deals with the anchor-neighbor and anchor-distant pairs of samples. The contrastive loss trains siamese networks against pairs of inputs labelled as similar or dissimilar (1 or 0). Contrastive loss is formulated in terms of the similarity label $$y$$ (1 for similar, 0 for dissimilar) and the Euclidean distance $$D$$ between two images, as:

$$
L = \frac{1}{2} ⋅y⋅D^2 + \frac{1}{2} ⋅(1−y)⋅max(0,m−D)^2
$$

where $$m$$ is a hyper-parameter. For the original paper, refer to [Dimensionality Reduction by Learning an Invariant Mapping](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf) or my summary in the <a href="{{ site.publications_permalink }}">publications bookmark</a>.


### Connectionist Temporal Classification loss

**[Connectionist Temporal Classification (CTC)](https://en.wikipedia.org/wiki/Connectionist_temporal_classification)** is a classifier and loss function for noisy, unsegmented sequential input data, used to train recurrent neural networks (RNNs) such as LSTM networks to tackle sequence problems where the timing is variable. Published in [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://www.cs.toronto.edu/~graves/icml_2006.pdf). Already summarized in the <a href="{{ site.publications_permalink }}">publications bookmark</a> section.

### Focal Loss

**[Focal Loss](https://arxiv.org/abs/1708.02002)** is an extension of the Cross-Entropy Loss designed to address class imbalance, a common issue in e.g. medical image segmentation, where the background (non-lesion regions) is much more prevalent than the foreground (lesion regions). Focal Loss focuses on hard-to-classify examples and down-weights the loss contribution from easy examples.

$$
\mathcal{L}_{\text{focal}} = -\alpha (1 - p_t)^\gamma \log(p_t)
$$

where:
- $$p_t$$ is the model’s estimated probability for the true class.
- $$α$$ is a balancing factor to adjust the weight of the positive class (especially useful for imbalanced datasets).
- $$γ$$ is the focusing parameter, usually set to 2, which adjusts the focus on hard-to-classify examples.

For multi-class classification, Focal Loss can be generalized as:

$$
\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

where $$\alpha_t$$ is the balancing factor for class $$t$$.

### Dice Loss

**[Dice loss](https://arxiv.org/abs/1707.03237)** (or Dice Similarity Coefficient Loss) is a loss function used specifically for image segmentation tasks, to evaluate the overlap between predicted and ground truth segmentation masks. It is based on the Dice coefficient, which measures the similarity between two sets, with values ranging from 0 (no overlap) to 1 (perfect overlap). Dice Loss is typically used when precise object boundaries are important, such as segmenting organs, tumors, or any region of interest in medical imaging where the goal is to accurately delineate the object of interest from the surrounding tissue. Key benefits of Dice Loss:
- Direct Focus on Overlap: Dice loss directly optimizes the overlap between predicted and ground truth masks, which is crucial for segmentation tasks where precise boundary delineation matters.
- Effective for Imbalanced Classes: It works well when there is a significant class imbalance between foreground and background, as it evaluates segmentation performance based on the overlap of the foreground regions rather than pixel accuracy.
- Works Well with Small Objects: Dice loss is particularly effective in medical imaging tasks where the objects of interest (e.g., tumors, organs) might occupy a small portion of the image.

$$
\mathcal{L}_{\text{Dice}} = 1 - \frac{2 |A \cap B|}{|A| + |B|}
$$

where:
- $$A$$ is the set of predicted pixels.
- $$B$$ is the set of ground truth pixels.
- $$ \mid A \cap B \mid$$ is the size of the intersection of the predicted and ground truth sets, representing the overlap.

For practical implementation, Dice Loss is often computed as:

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


