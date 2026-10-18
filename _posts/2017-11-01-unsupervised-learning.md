---
layout: post
title:  "Unsupervised Learning basics and Principal Component Analysis"
categories: ["Machine Learning"]
tags: ["unsupervised learning", "PCA", "clustering", "dimensionality reduction"]
---

Unsupervised learning is the field of machine learning that learns from the data alone. Unlike supervised learning, which trains on inputs paired with labels, and [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning), which trains with reward signals, an unsupervised method receives only the data, and must find the structure hidden in it. This setting matters because unlabelled data is far more abundant than labelled data. Common applications are image compression, dimensionality reduction, clustering, visualization, and learning representations of the data that other models then build on.

## Foreword: Hebbian Learning

[Hebbian learning][hebbian] ([Donald Hebb](https://en.wikipedia.org/wiki/Donald_O._Hebb), 1949) is one of the oldest *biologically inspired* learning models, and its principle underlies several of the unsupervised learning rules that follow, from Oja's rule to self-organizing maps. A synapse strengthens when its pre-synaptic neuron repeatedly takes part in firing its post-synaptic neuron, an idea often summarized as "*cells that fire together wire together*". Spike-timing-dependent plasticity ([STDP, Markram et al., Frontiers](https://www.frontiersin.org/articles/10.3389/fnsyn.2012.00002/full)) refines the principle: the change of the synapse depends on the time difference between the pre- and post-synaptic spikes. In its classic form, the synapse strengthens when the pre-synaptic spike comes shortly before the post-synaptic spike, so that the first neuron may have caused the second one to fire, and weakens when the order is reversed.

[hebbian]: https://en.wikipedia.org/wiki/Hebbian_theory

In Hebb's own words (1949):

> When an axon of cell A is near enough to excite a cell B and repeatedly or persistently takes part in firing it, some growth process or metabolic change takes place in one or both cells such that A's efficiency, as one of the cells firing B, is increased.

Artificial neurons follow the same principle, with the weight of a connection in the role of the strength of a synapse. The general Hebbian rule describes the change over time of the weight $$w_{ij}$$ from a pre-synaptic neuron $$i$$, with activity $$x_i$$, to a post-synaptic neuron $$j$$, with output $$y_j$$:

$$
\frac{d}{dt}w_{ij} = F (w_{ij}; x_i, y_j)
$$

The simplest Hebbian learning rule is:

$$
\frac{d}{dt}w_{ij} = \eta \, x_i \, y_j
$$

where $$\eta$$ is the learning rate. Hebbian learning is **local**: it only looks at the two neurons involved, and ignores the activity of the rest of the system. It also needs no external signal or measure of accuracy, which makes it a form of **unsupervised learning**. However, when both activities are positive, the simple rule above can only increase the weights, and they grow without bound. A common way to keep the weights bounded is [Oja's learning rule](http://www.scholarpedia.org/article/Oja_learning_rule):

$$
\frac{d}{dt}w_{ij} = \eta \, (x_i y_j - y_j^2 w_{ij})
$$

The second term, a decay, grows with the squared output $$y_j^2$$, so it balances the Hebbian term more strongly as the output grows. For a linear neuron (below) and inputs of zero mean, Oja's rule converges to a weight vector of unit norm along the first principal component of the inputs, the topic of the next section. Another common formulation is the covariance rule, which uses the fluctuations of $$x_i$$ and $$y_j$$ around their mean values $$\langle x_i \rangle$$ and $$\langle y_j \rangle$$:

$$
\frac{d}{dt}w_{ij} = \eta \, (x_i - \langle x_i \rangle) ( y_j - \langle y_j \rangle )
$$

The weight then also decreases when one activity is above its mean and the other below, and on average it follows the covariance of the two activities. Another option is the Bienenstock-Cooper-Munro rule ([BCM][bcm], 1982), a model of learning in the visual cortex:

$$
\frac{d}{dt}w_{ij} = \eta \, x_i \, y_j \, (y_j - \theta_j)
$$

[bcm]: https://en.wikipedia.org/wiki/BCM_theory

The weight increases, a [long-term potentiation](https://en.wikipedia.org/wiki/Long-term_potentiation) (LTP), when the post-synaptic activity is above the threshold $$\theta_j$$, and decreases, a [long-term depression](https://en.wikipedia.org/wiki/Long-term_depression) (LTD), when it is below. The threshold slides with the time-averaged post-synaptic activity, e.g. $$\theta_j = \langle y_j^2 \rangle$$: it rises after a period of high activity, which favours depression, and falls after a period of low activity, which favours potentiation. This feedback stabilizes the weights.

The output of a neuron is typically a function of the weighted sum of its inputs:

$$
y_j = g \left( \sum_i w_{ij} x_i \right)
$$

where $$g$$ is the [activation function](https://en.wikipedia.org/wiki/Activation_function). The simplest case is the **linear neuron**:

$$
y_j = \sum_i w_{ij} x_i
$$

**Sigmoidal functions** ([examples](https://en.wikipedia.org/wiki/Sigmoid_function#Examples)) are commonly used instead: they bound the output and make the neuron non-linear. In general, unsupervised learning follows $$\Delta w \propto F(\text{pre}, \text{post})$$. Adding a third factor, a reward signal, to each update gives rules of the form $$\Delta w \propto F(\text{pre}, \text{post}, \text{reward})$$, a model of **reinforcement learning**. Hebbian rules model the synaptic changes of LTP and LTD in the brain, while reward-modulated rules model the effect of its dopamine [reward system](https://en.wikipedia.org/wiki/Reward_system), which rewards the learning of a behaviour.

## Principal Component Analysis

High-dimensional data is often redundant: its features are correlated, so most of its variation lies along a few directions. [Principal component analysis][pca] (PCA) is a dimensionality (feature) reduction method that exploits this redundancy. It rotates the data towards the directions of largest variance, the *principal components*, so that the data can be analysed along its most meaningful dimensions, and the dimensions of low variance can be discarded. Gerstner's lecture notes on unsupervised learning define it as:

[pca]: https://en.wikipedia.org/wiki/Principal_component_analysis

<blockquote class="blockquote text-right">
Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components.
</blockquote>

Given $$n$$ data points $$x_1, \ldots, x_n$$ of dimension $$D$$, the algorithm is:
- subtract the mean to center the data: $$x_i \leftarrow x_i - \bar{x}$$, with $$\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$$;
- compute the $$D \times D$$ [covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix) of the centered data, $$C = \frac{1}{n-1} \sum_{i=1}^n x_i \, x_i^T$$;
- compute the [eigenvalues and eigenvectors](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors) of $$C$$:
  - solve $$\det(C - \lambda I)=0$$ to obtain the eigenvalues $$\lambda$$;
  - for each eigenvalue $$\lambda_m$$, compute its eigenvector $$v_m$$ by solving $$C v_m = \lambda_m v_m \, \Leftrightarrow \, (C- \lambda_m I) v_m = 0$$;
- build the feature matrix $$F$$, with the eigenvectors as columns, sorted by decreasing eigenvalue;
- obtain the transformed data by multiplying the transposed feature matrix $$F^T$$, whose first row is the most significant eigenvector, by the matrix $$X$$ whose columns are the centered data points: $$Y = F^T X$$. The columns of $$Y$$ are the coordinates of the points along the principal components.

Why the eigenvectors? The variance of the centered data along a unit vector $$v$$ is $$\frac{1}{n-1} \sum_{i} (v^T x_i)^2 = v^T C v$$, which is largest for the eigenvector of the largest eigenvalue, where it equals that eigenvalue. Each eigenvalue $$\lambda_m$$ is thus the variance of the data along its eigenvector $$v_m$$. The figure below shows an example, where PCA rotates the data towards the new axes of the cloud of points:

{: .picture}
<img width="26.4%" height="26.4%" src="{{ site.assets }}/unsupervised-learning/gaussianscatterpca.svg">

{: .caption}
PCA of a 2D Gaussian centred at $$(1, 3)$$, with a standard deviation of 3 roughly along the direction $$(0.866, 0.5)$$ and of 1 in the orthogonal direction. The arrows are the eigenvectors of the covariance matrix, scaled by the square roots of their eigenvalues, with their tails at the mean. Adapted from: [Principal component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis)

The least significant dimensions of the transformed data can now be discarded, which reduces the number of features of the model. The number $$r$$ of eigenvectors kept in $$F$$ sets the amount of compression, since each point then keeps $$r$$ coordinates instead of $$D$$. The first $$r$$ components explain a fraction $$\sum_{m=1}^r \lambda_m / \sum_{m=1}^D \lambda_m$$ of the variance of the data, a common criterion to choose $$r$$.

Inverting the transformation recovers the original data, or an approximation of it if $$F$$ does not include all eigenvectors. The inversion multiplies the transformed data by the inverse of $$F^T$$, which is easy to find: $$C$$ is symmetric, so its eigenvectors can be chosen orthonormal, and $$F^TF = I$$. When $$F$$ keeps all eigenvectors, it is square and orthogonal, so $$F^{-T} = F$$ and:

$$
X = F^{-T} Y = F Y
$$

When $$F$$ keeps only $$r$$ eigenvectors, $$F Y$$ is the projection of the data onto the first $$r$$ principal components. This projection is the best approximation of $$X$$ of rank $$r$$, in the least-squares sense ([Eckart-Young theorem](https://en.wikipedia.org/wiki/Low-rank_approximation)).

In practice, PCA is usually computed with the [singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) (SVD) of the centered data, without forming $$C$$. Since $$X X^T = (n-1) C$$, the right singular vectors of $$X^T$$ are the eigenvectors of $$C$$, and the eigenvalues of $$C$$ are the squared singular values divided by $$n-1$$. Forming $$C$$ squares the [condition number](https://en.wikipedia.org/wiki/Condition_number) of the data, so the SVD is numerically more stable, although not cheaper: both approaches cost $$O(nD^2 + D^3)$$ operations. For large datasets, iterative and randomized methods compute only the first $$r$$ components, in about $$O(nDr)$$ operations.

## Kernel PCA

PCA only finds linear structure, so it cannot summarize data that lies along a curve. [Kernel PCA](https://en.wikipedia.org/wiki/Kernel_principal_component_analysis) (Schölkopf et al., 1998) applies PCA in a high-dimensional feature space $$\phi(x)$$ instead, where non-linear structure of the input space may become linear. Computing $$\phi$$ would be expensive, or impossible when the feature space has infinite dimensions, but the [kernel trick](https://en.wikipedia.org/wiki/Kernel_method) avoids it, because the method only needs the dot products $$\kappa(x_i, x_j) = \phi(x_i)^T \phi(x_j)$$. A kernel function provides these dot products, e.g. the Gaussian kernel $$\kappa(x_i, x_j) = \exp \left( - \Vert x_i - x_j \Vert^2 / 2 \sigma^2 \right)$$ or the polynomial kernel $$\kappa(x_i, x_j) = (x_i^T x_j + c)^d$$ of degree $$d$$, with a constant $$c \geq 0$$. The algorithm is:
- build the $$n \times n$$ kernel matrix $$K$$, with $$K_{ij} = \kappa(x_i, x_j)$$;
- center the data in the feature space, using $$K$$ alone: $$\tilde{K} = K - 1_n K - K 1_n + 1_n K 1_n$$, where $$1_n$$ is the $$n \times n$$ matrix with all entries equal to $$1/n$$;
- compute the eigenvalues $$\lambda_m$$ and eigenvectors $$\alpha^{(m)}$$ of $$\tilde{K}$$, and scale each eigenvector so that $$\lambda_m \, \alpha^{(m)T} \alpha^{(m)} = 1$$. This gives the corresponding direction in the feature space, whose squared norm is $$\alpha^{(m)T} \tilde{K} \alpha^{(m)} = \lambda_m \, \alpha^{(m)T} \alpha^{(m)}$$, a unit norm;
- the $$m$$-th principal component of a data point $$x_j$$ is $$\sum_{i=1}^n \alpha_i^{(m)} \tilde{K}_{ij}$$, which is the $$j$$-th entry of $$\tilde{K} \alpha^{(m)} = \lambda_m \alpha^{(m)}$$.

Kernel PCA stores the $$n \times n$$ matrix $$\tilde{K}$$, in $$O(n^2)$$ memory, and its eigendecomposition costs $$O(n^3)$$ operations, against $$O(nD^2 + D^3)$$ for PCA. This limits kernel PCA to datasets of moderate size. In addition, mapping a point back to the input space, the *pre-image* problem, has only approximate solutions, because most points of the feature space are not the image $$\phi(x)$$ of any input.

## Independent Component Analysis

This section is largely based on the lecture notes of Tatsuya Yokota (Tokyo Institute of Technology, 2012).

The principal components of PCA are uncorrelated, but uncorrelated signals may still depend on each other. [Independent component analysis](https://en.wikipedia.org/wiki/Independent_component_analysis) (ICA) is a related method for [blind source separation](https://en.wikipedia.org/wiki/Blind_signal_separation) that looks for statistically independent components instead. It separates a multidimensional signal into additive subcomponents, under two assumptions:
- the source signals $$s_1, \ldots, s_M$$ are statistically independent: $$p(s_1, \ldots, s_M) = p(s_1) \cdots p(s_M)$$;
- the sources are not Gaussian. At most one of them may be Gaussian; otherwise, ICA cannot separate them.

ICA builds on the [central limit theorem](https://en.wikipedia.org/wiki/Central_limit_theorem): the distribution of a sum of independent random variables with finite variance approaches a Gaussian as the number of terms grows. For example, a single die throw has a uniform distribution from 1 to 6, but the sum of two throws peaks at 7, and the distribution of the sum of many throws is close to a Gaussian. A mixture of independent signals is therefore *more Gaussian* than the signals themselves, and the most non-Gaussian directions of the data recover the independent signals. In short, *non-Gaussianity is a measure of independence*.

A well-known example is the [cocktail party](https://en.wikipedia.org/wiki/Cocktail_party_effect) problem. Three guests speak at the same time, and three microphones at different places of the room record different mixtures $$x$$ of their voices $$s$$:

$$
\begin{align*}
x_1(t) = a_{11}s_1(t) + a_{12}s_2(t) + a_{13}s_3(t)\\
x_2(t) = a_{21}s_1(t) + a_{22}s_2(t) + a_{23}s_3(t)\\
x_3(t) = a_{31}s_1(t) + a_{32}s_2(t) + a_{33}s_3(t)
\end{align*}
$$

ICA estimates the independent components $$s(t)$$ from the observations

$$
x(t) = As(t)
$$

where $$A$$ is an invertible mixing matrix. Both $$A$$ and $$s(t)$$ are unknown, so the problem is to find the unmixing matrix $$B = A^{-1}$$ such that the components of $$s(t) = Bx(t)$$ are independent.

Signals $$z$$ are *white* if $$\mathbb{E}[z] = 0$$ and $$\mathbb{E}[z z^T] = I$$. [Whitening](https://en.wikipedia.org/wiki/Whitening_transformation) is a common pre-processing step of ICA: after centering $$x$$, $$z(t) = Vx(t)$$, with the whitening matrix $$V = \Lambda^{-1/2} F^T$$, where the diagonal matrix $$\Lambda$$ holds the eigenvalues and $$F$$ the eigenvectors of the covariance matrix of $$x$$, as in PCA. In other words, $$F^T$$ rotates the data onto its principal components, and $$\Lambda^{-1/2}$$ scales each of them to unit variance. ICA cannot recover the scale of the sources, since scaling a source and inversely scaling the matching column of $$A$$ gives the same observations, so the sources are taken with unit variance, and the model becomes:

$$
s(t) = U z (t) = U V x (t) = B x (t)
$$

where $$U$$ is orthogonal, since both $$s$$ and $$z$$ are white: $$I = \mathbb{E}[s s^T] = U \, \mathbb{E}[z z^T] \, U^T = U U^T$$. **Whitening simplifies ICA**: only $$U$$ remains to be estimated, with $$M(M-1)/2$$ degrees of freedom instead of the $$M^2$$ of $$B$$. This also shows why the sources must not be Gaussian: white Gaussian data have the same distribution after any rotation, so no choice of $$U$$ is better than another. The figures below show the source, observed and whitened data:

{: .picture}
<img width="20.1%" height="20.1%" src="{{ site.assets }}/unsupervised-learning/ica_source.svg">
<img width="19.1%" height="19.1%" src="{{ site.assets }}/unsupervised-learning/ica_observed.svg"> 
<img width="17.9%" height="17.9%" src="{{ site.assets }}/unsupervised-learning/ica_whitening.svg">

{: .caption}
Independent component analysis of two mixed signals: (a) the sources $$s_1$$, $$s_2$$, independent and uniformly distributed; (b) the observed signals $$x_1$$, $$x_2$$, mixed as $$x = As$$; (c) the whitened signals $$z_1$$, $$z_2$$, $$z = Vx$$, with zero mean and identity covariance, from which only the orthogonal matrix $$U$$ remains to be found. Adapted from: [T. Yokota, Independent Component Analysis for Blind Source Separation, Tokyo Institute of Technology, 2012](https://www.slideshare.net/slideshow/independent-component-analysis-11359849/11359849)

The independent uniform sources fill a square in (a). The mixing stretches and shears it into a flat parallelogram in (b), and whitening turns it back into a square in (c), but a rotated one: the rotation that aligns its sides with the axes is the matrix $$U$$.

[Kurtosis](https://en.wikipedia.org/wiki/Kurtosis) measures non-Gaussianity. For a variable $$y$$ of zero mean, ICA uses:

$$
\text{kurt}(y) = \mathbb{E}[y^4] - 3 (\mathbb{E}[y^2])^2
$$

It is zero for a Gaussian, whose fourth moment is $$3 (\mathbb{E}[y^2])^2$$, positive for distributions with heavier tails, and negative for flatter ones, such as the uniform sources of the figure. For a white $$y$$, with $$\mathbb{E}[y^2]=1$$, it simplifies to $$\text{kurt}(y) = \mathbb{E}[y^4]-3$$. Each row $$b^T$$ of $$B$$ recovers one source, $$s_m = b^T x$$. ICA finds it by maximizing the non-Gaussianity, whatever the sign of the kurtosis:

$$
\hat{b} = \arg\max_b \vert \text{kurt}(b^T x(t)) \vert
$$

where $$b^T x(t)$$ is constrained to unit variance. On the whitened data, $$b^T x(t) = w^T z(t)$$, where $$w^T = b^T V^{-1}$$ is a row of $$U$$, and the constraint becomes $$\Vert w \Vert = 1$$, since $$\mathbb{E}[(w^T z)^2] = w^T \mathbb{E}[z z^T] w = \Vert w \Vert^2$$. ICA therefore maximizes $$\vert \text{kurt}(w^T z) \vert$$ subject to $$\Vert w \Vert = 1$$. The gradient is:

$$
\begin{align*}
\frac{\partial \vert \text{kurt}(w^T z) \vert}{\partial w} & = \frac{\partial}{\partial w} \left\vert \mathbb{E}[(w^T z)^4] - 3 \left(\mathbb{E}[(w^T z)^2]\right)^2 \right\vert \\
& = \frac{\partial}{\partial w} \left\vert \mathbb{E}[(w^T z)^4] - 3 \Vert w \Vert^4 \right\vert \quad \text{(because } \mathbb{E}[z z^T] = I) \\
& = 4 \, \text{sign}[\text{kurt}(w^T z)] \left( \mathbb{E}[z (w^T z)^3] - 3 w \Vert w \Vert^2 \right)
\end{align*}
$$

The last line differentiates each term, and the absolute value contributes the sign of the kurtosis. Gradient ascent then iterates, with $$\Vert w \Vert = 1$$:
- $$\Delta w \propto \text{sign} [ \text{kurt}(w^T z)] \, ( \mathbb{E} [ z(w^T z)^3 ] - 3w )$$;
- $$w \leftarrow w + \Delta w$$, followed by the normalization $$w \leftarrow w / \Vert w \Vert$$.

At convergence, the update no longer changes the direction of $$w$$, so $$w \propto \Delta w$$, and $$w$$ and $$-w$$ give the same component. The fixed-point iteration of [FastICA](https://en.wikipedia.org/wiki/FastICA) (Hyvärinen and Oja, 1997) therefore sets $$w$$ directly to the direction of the update:
- $$w \leftarrow \mathbb{E} [ z ( w^T z)^3 ] - 3w$$, followed by $$w \leftarrow w / \Vert w \Vert$$.

FastICA needs no step size, and converges in far fewer iterations than gradient ascent. The remaining components are found in the same way, with each new $$w$$ kept orthogonal to the previous ones, since the rows of $$U$$ are orthonormal.

Kurtosis, a fourth-order moment, is sensitive to outliers. Negentropy, the difference between the entropy of a Gaussian of the same variance, which is the largest possible, and the entropy of the signal, is a more robust measure of non-Gaussianity. The [original lecture notes](https://www.slideshare.net/yokotatsuya/independent-component-analysis-11359849) cover it.

## Competitive Learning and k-means

Competitive learning and k-means are not to be confused with the [k-nearest neighbours algorithm (KNN)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm), a supervised method.

In [competitive learning](https://en.wikipedia.org/wiki/Competitive_learning), neurons compete to respond to each input, and only the winner, or the winner and its neighbours, adapts. It models the self-organization of the brain, whose wiring is not fixed but adapts to its inputs. A common competitive learning technique for data reduction is **prototype clustering**. It represents the data points $$x_i$$ by a smaller set of prototypes $$w_1, \ldots, w_k$$, each representing a cluster of points. A point $$x$$ belongs to the closest prototype $$j$$:

$$
\Vert w_j - x \Vert \leq \Vert w_l - x \Vert \text{, for all } l
$$

The prototypes thus partition the space into the cells of a [Voronoi diagram](https://en.wikipedia.org/wiki/Voronoi_diagram). In the 2D example below, the points of each colored cell belong to its prototype, the black point:

{: .picture}
<img width="36.6%" height="36.6%" src="{{ site.assets }}/unsupervised-learning/voronoi.svg">

{: .caption}
The Voronoi diagram of 20 prototypes (black dots) in 2D: each colored cell holds the points closer to its prototype than to any other, in Euclidean distance. Adapted from: [Voronoi diagram](https://en.wikipedia.org/wiki/Voronoi_diagram)

An application is image compression by [color quantization](https://en.wikipedia.org/wiki/Color_quantization), which replaces each RGB pixel, with values in $$[0, 255]^3$$, by the closest color of a small palette: with 16 colors, each pixel needs 4 bits instead of 24. The squared reconstruction error sums the errors of all points over all clusters $$C_j$$:

$$
E = \sum_{j=1}^k \sum_{x_i \in C_j} \Vert w_j - x_i \Vert^2
$$

**[k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering)** seeks the partition of the points into $$k$$ clusters $$C_1, \ldots, C_k$$ that minimizes $$E$$, with each prototype $$w_j$$, or mean, at the center of its cluster. Finding the best partition is NP-hard, so k-means alternates between two simpler steps:
- assignment step: with the means fixed, assign each point to the closest mean, using e.g. the [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance);
- update step: with the clusters fixed, move each mean to the centroid of the points of its new cluster:
  $$ w_j^{(t+1)} = \frac{1}{\vert C^{(t)}_j \vert} \sum_{x_i \in C_j^{(t)}} x_i $$

The algorithm stops when the assignments no longer change. The means are then the centers of mass of their clusters: with the batch update $$\Delta w_j = \eta \sum_{x_i \in C_j} (x_i - w_j)$$, a gradient descent step on $$E$$, convergence means $$\Delta w_j = 0$$, which requires:

$$
\eta \sum_{x_i \in C_j} (x_i - w_j) = 0 \rightarrow \sum_{x_i \in C_j} x_i - \vert C_j \vert w_j =0 \rightarrow w_j = \frac{1}{\vert C_j \vert} \sum_{x_i \in C_j} x_i
$$

Neither step increases $$E$$: the assignment step moves points only to closer means, and the update step moves each mean to the centroid, the point with the smallest sum of squared distances to the points of its cluster. Since the number of partitions is finite, k-means always converges, but possibly to a local minimum. It is therefore run several times from different initial means, e.g. with [k-means++](https://en.wikipedia.org/wiki/K-means%2B%2B), which spreads the initial means over the data. Each iteration computes the $$nk$$ distances between points and means, in $$O(nkD)$$ operations. The figure below illustrates the algorithm:

{: .picture}
<img width="78.4%" height="78.4%" src="{{ site.assets }}/unsupervised-learning/k-means.svg">

{: .caption}
**1.** $$k = 3$$ initial means (colored circles) are chosen at random within the data domain. **2.** Assignment step: each point joins the cluster of the closest mean; the colored regions are the Voronoi cells of the means. **3.** Update step: each mean moves to the centroid of its cluster (dashed circles: the previous means; lines: the points of each cluster). **4.** Assignment step with the new means and their Voronoi cells: two points move from the green to the red cluster. The two steps repeat until the assignments no longer change. Adapted from: [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering)

In step 4, the means have moved, and their Voronoi borders with them, so the points near a border can change cluster. In 2D, a full run with 3 clusters looks as follows:

{: .picture}
<img width="30%" height="30%" src="{{ site.assets }}/unsupervised-learning/k-means_convergence.gif">

{: .caption}
Iterations of k-means with $$k = 3$$ on 2D data, with the Voronoi borders of the current means, until the assignments no longer change. Source: [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering)

The update rule above is a **batch learning rule**: it updates the means from a set (*batch*) of inputs. The **online** rule updates the mean $$w_j$$ of the winning prototype for every new input $$x_i$$:

$$
\Delta w_j = \eta (x_i - w_j)
$$

This is Oja's rule for a winner-take-all neuron, with output $$y_j = 1$$ for the winner and $$0$$ for all other neurons: with $$y_j = 1$$, the Hebbian term of Oja's rule becomes the input and its decay term the weight itself.

k-means has two main limitations:
- it assumes spherical clusters, while elliptical clusters are sometimes more appropriate;
- each point belongs to exactly one cluster, even a point halfway between two clusters.

Gaussian [mixture models](https://en.wikipedia.org/wiki/Mixture_model) address both.

## Gaussian Mixture Models and Expectation-Maximization

A Gaussian mixture model (GMM) describes the data as drawn from $$k$$ Gaussians. Each data point picks a component $$j$$ with probability $$\pi_j$$, and is then drawn from the Gaussian $$\mathcal{N}(\mu_j, \Sigma_j)$$, so that:

$$
p(x) = \sum_{j=1}^k \pi_j \, \mathcal{N}(x \mid \mu_j, \Sigma_j), \text{ with } \sum_{j=1}^k \pi_j = 1.
$$

The covariance matrices $$\Sigma_j$$ give each cluster an elliptical shape, and each point belongs to every cluster with some probability. The [expectation-maximization (EM)](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm) algorithm learns the parameters by maximizing the log-likelihood $$\sum_{i=1}^n \log p(x_i)$$ of the $$n$$ data points. If the component of each point were known, each Gaussian could be fitted to its own points. EM therefore alternates, like k-means, between estimating the components and fitting the Gaussians:
- E-step: compute the *responsibility* of each component $$j$$ for each point $$x_i$$, the posterior probability that $$x_i$$ was drawn from that component:

$$
\gamma_{ij} = \frac{\pi_j \, \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}{\sum_{l=1}^k \pi_l \, \mathcal{N}(x_i \mid \mu_l, \Sigma_l)}
$$

- M-step: update the parameters, with the responsibilities as weights, where $$n_j = \sum_{i} \gamma_{ij}$$:

$$
\mu_j = \frac{1}{n_j} \sum_{i} \gamma_{ij} \, x_i, \quad
\Sigma_j = \frac{1}{n_j} \sum_{i} \gamma_{ij} \, (x_i - \mu_j)(x_i - \mu_j)^T, \quad
\pi_j = \frac{n_j}{n}
$$

No iteration decreases the log-likelihood, and EM converges to a stationary point of it, in practice a local maximum. The log-likelihood has no upper bound, however: it grows without limit when a component collapses onto a single data point and its covariance tends to $$0$$, so implementations add a small constant to the diagonal of each $$\Sigma_j$$. As with k-means, EM is usually run from several initializations, or initialized with the result of k-means. In fact, k-means is a limit case of EM: for components with the shared covariance $$\sigma^2 I$$, the responsibilities become hard assignments to the closest mean as $$\sigma \rightarrow 0$$.

The flexibility of the GMM has a cost. With full covariance matrices, an EM iteration costs $$O(nkD^2)$$ operations to evaluate the $$nk$$ densities, and $$O(kD^3)$$ to factorize the $$k$$ covariance matrices, against $$O(nkD)$$ for an iteration of k-means. The covariance matrices also add $$kD(D+1)/2$$ parameters, which need more data. Diagonal covariance matrices, which give ellipses aligned with the axes, reduce the cost to $$O(nkD)$$ and the covariance parameters to $$kD$$. The number of components $$k$$ can be chosen with a model selection criterion, such as the [Bayesian information criterion](https://en.wikipedia.org/wiki/Bayesian_information_criterion).

## Hierarchical Clustering

k-means and GMMs need the number of clusters as input. [Hierarchical clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering) avoids this choice: it builds a tree of clusters, a *dendrogram*, instead of a single partition. The common *agglomerative* (bottom-up) variant starts with one cluster per data point and repeatedly merges the two closest clusters, until a single cluster remains. The *linkage* criterion defines the distance between two clusters $$A$$ and $$B$$, from a distance $$d$$ between points:
- single linkage: the distance between their closest points, $$\min_{a \in A, b \in B} d(a,b)$$. It finds elongated clusters, but a few close points can chain distinct clusters together;
- complete linkage: the distance between their farthest points, $$\max_{a \in A, b \in B} d(a,b)$$. It favours compact clusters;
- average linkage: the average distance over all pairs of points, $$\frac{1}{\vert A \vert \vert B \vert} \sum_{a \in A} \sum_{b \in B} d(a,b)$$, a compromise between the two;
- [Ward's linkage](https://en.wikipedia.org/wiki/Ward%27s_method): the increase of the k-means error $$E$$ caused by merging $$A$$ and $$B$$. For cluster means $$\mu_A$$ and $$\mu_B$$, this increase is $$\frac{\vert A \vert \vert B \vert}{\vert A \vert + \vert B \vert} \Vert \mu_A - \mu_B \Vert^2$$, so small clusters with close means merge first.

Cutting the dendrogram at a given height yields a partition, so the number of clusters can be chosen afterwards, by inspecting the tree. The *divisive* (top-down) variant works in the opposite direction: it starts with all points in one cluster and splits clusters recursively. Standard implementations of agglomerative clustering store the $$n \times n$$ matrix of pairwise distances, in $$O(n^2)$$ memory, and run in $$O(n^2)$$ to $$O(n^3)$$ time, depending on the linkage and the algorithm. Agglomerative clustering is therefore suited to small and medium datasets, while k-means, at $$O(nkD)$$ operations per iteration, scales to large ones.

## DBSCAN

The methods above assign every point to a cluster, outliers included. [DBSCAN](https://en.wikipedia.org/wiki/DBSCAN) (Density-Based Spatial Clustering of Applications with Noise, Ester et al., 1996) instead defines clusters as dense regions separated by sparse regions. It takes two parameters, a radius $$\varepsilon$$ and a minimum number of points $$\text{minPts}$$, and labels each point as one of:
- a *core point*, if at least $$\text{minPts}$$ points, itself included, lie within distance $$\varepsilon$$ of it;
- a *border point*, if it is not a core point but lies within distance $$\varepsilon$$ of a core point;
- *noise*, in all other cases.

Two core points within distance $$\varepsilon$$ of each other belong to the same cluster. Each cluster is thus a connected group of core points, plus the border points around them. Unlike k-means, DBSCAN does not need the number of clusters as input, finds clusters of arbitrary shapes, and labels outliers as noise. However, a single radius $$\varepsilon$$ cannot separate clusters with large differences in density: a radius small enough for the dense clusters labels the sparse ones as noise. The extensions [OPTICS](https://en.wikipedia.org/wiki/OPTICS_algorithm) and [HDBSCAN](https://hdbscan.readthedocs.io/) address this limitation by considering a range of radii. The cost is one neighbourhood query per point: $$O(n^2)$$ distances in total without an index. In low dimensions, a spatial index, such as a [k-d tree](https://en.wikipedia.org/wiki/K-d_tree), answers each query without computing all pairwise distances. DBSCAN needs only $$O(n)$$ memory, since it does not store the distances.

## Spectral Clustering

k-means separates clusters with the straight borders of Voronoi cells, so it cannot separate clusters that are curved or nested in each other. [Spectral clustering](https://en.wikipedia.org/wiki/Spectral_clustering) treats clustering as a graph partitioning problem instead. It builds a graph of the data points, with a similarity $$W_{ij}$$ for each pair of points, e.g. $$W_{ij} = \exp ( - \Vert x_i - x_j \Vert^2 / 2\sigma^2 )$$ between each point and its nearest neighbours, and $$0$$ otherwise. It then looks for clusters connected by edges of small weight. With the degree matrix $$G$$, a diagonal matrix with $$G_{ii} = \sum_j W_{ij}$$, the graph [Laplacian](https://en.wikipedia.org/wiki/Laplacian_matrix) is $$L = G - W$$, and for any vector $$f$$:

$$
f^T L f = \frac{1}{2} \sum_{i,j} W_{ij} (f_i - f_j)^2
$$

Therefore, $$f^T L f$$ is small when $$f$$ takes similar values on strongly connected points, and the eigenvectors of $$L$$ with the smallest eigenvalues vary little within clusters. In particular, the multiplicity of the eigenvalue $$0$$ equals the number of connected components of the graph, since the indicator vector of each component, equal to 1 on its points and 0 elsewhere, gives $$f^T L f = 0$$. Clusters joined by a few weak edges give eigenvectors close to such indicator vectors. The algorithm, for $$k$$ clusters, is:
- compute the eigenvectors $$u_1, \ldots, u_k$$ of the $$k$$ smallest eigenvalues of $$L$$, or of the normalized Laplacian $$L_{sym} = I - G^{-1/2} W G^{-1/2}$$;
- stack these eigenvectors as the columns of an $$n \times k$$ matrix $$U$$. Its $$i$$-th row is the new $$k$$-dimensional representation of point $$x_i$$. With $$L_{sym}$$, each row is first normalized to unit length;
- cluster the rows of $$U$$ with k-means.

This is a relaxation of the NP-hard problem of finding the *ratio cut* (with $$L$$) or the *normalized cut* (with $$L_{sym}$$) of the graph: the discrete indicator vectors of the clusters are replaced by real eigenvectors, and k-means turns these back into a partition. Spectral clustering therefore separates non-convex clusters, e.g. two concentric rings. A dense eigendecomposition of the $$n \times n$$ Laplacian costs $$O(n^3)$$ operations and $$O(n^2)$$ memory. With a nearest-neighbour graph, the Laplacian is sparse, and iterative eigensolvers, such as the [Lanczos algorithm](https://en.wikipedia.org/wiki/Lanczos_algorithm), compute only the $$k$$ eigenvectors needed, from sparse matrix-vector products.

## Self-Organizing Maps and Kohonen Maps

A [self-organizing map](https://en.wikipedia.org/wiki/Self-organizing_map) (SOM), or Kohonen map, is an unsupervised artificial neural network, typically a 2D grid of neurons, that represents a discretization of the input space, or of the training set. Each neuron represents a localized region of the input space, an agglomerate of inputs. Unlike the prototypes of k-means, the neurons are ordered: neighbouring neurons in the grid represent nearby regions of the input space:

{: .picture}
<img width="45.8%" height="45.8%" src="{{ site.assets }}/unsupervised-learning/kohonen.svg">
<img width="24.5%" height="24.5%" src="{{ site.assets }}/unsupervised-learning/kohonen2.svg">

{: .caption}
**Left:** the neuron layer of Kohonen's model. The input fibres running horizontally carry the input signal (the stimulus) and excite the neurons of the layer through synaptic connections. Lateral interactions between the neurons confine the response to a spatially bounded local excitation zone. The layer acts as a topographic feature map if the position $$j^*$$ of the excitation zone varies continuously with the stimulus features of interest. **Right:** the adaptation step. The input $$x_i$$ in the input space $$V$$ selects the winner neuron $$j^*$$ in the lattice of neurons $$A$$ ($$\phi$$ is the map from $$V$$ to $$A$$). The winner and its neighbours shift their weight vectors towards $$x_i$$, by an amount that decreases with the distance from $$j^*$$, shown by the sizes and gray levels of the neurons. Only the shift $$\Delta w_{j^*}$$ of the winner's weights $$w_{j^*}$$ is drawn. Adapted from: [Neural Computation and Self-Organizing Maps, H. Ritter, T. Martinetz and K. Schulten, 1992, Figures 4.1 and 4.2](https://tcbg.illinois.edu/Services/Class/PHYS498TBP/spring2002/neuro_4.pdf)

The algorithm repeats the adaptation step on the right:
- place the neurons at random in the input space, i.e. initialize their weights $$w_j$$ at random;
- at every iteration, choose an input $$x_i$$;
- find the *winner* neuron $$j^*$$, or *best matching unit* (BMU), the neuron closest to the input:

$$
\Vert w_{j^*} - x_i \Vert \leq \Vert w_j - x_i \Vert \text{, for all } j
$$

- update the weights of the winner and of its neighbours, pulling them closer to the input:

$$
\Delta w_j = \eta \, \Lambda (j,j^*) \, (x_i - w_j)
$$

where $$\Lambda (j,j^*)$$ is the *neighbourhood function*. It decreases with the distance between neurons $$j$$ and $$j^*$$ in the grid, not in the input space, e.g. as a Gaussian. $$\eta$$ and the width of $$\Lambda$$ commonly decrease as the iterations progress, so that the map first unfolds over the data and then refines its details;
- repeat until stabilization: until a maximum number of iterations, or until the change of the weights falls below a threshold.

The online k-means rule above is the special case where only the winner adapts. The neighbourhood is what orders the map, since neighbouring neurons move together towards the same inputs. Two videos show the iterations of a Kohonen map: [video 1](https://www.youtube.com/watch?v=IixbH1gDhsg) and [video 2](https://www.youtube.com/watch?v=zeOtwCAI3Fs).

A good application is the recognition of the handwritten digits of the [MNIST database](http://yann.lecun.com/exdb/mnist/). Each digit is a 28x28 (=784) pixel array, a data point in a 784-dimensional space. Each pixel is a gray level, so each dimension takes values between 0 and 255. The task is to find, from a database of 5000 handwritten digits, the 6x6 SOM that best represents 4 of the digits from 0 to 9, and to study how the neighbourhood width and the learning rate affect the Kohonen map.

The input dataset and the `python` implementation are available in <a href="{{ site.assets }}/unsupervised-learning/kohonen_mnist.zip">kohonen_mnist.zip</a>. The weights of each neuron form a point of the input space, which can be drawn as a 28x28 image, so a good solution gives a visual perception of the digits being identified, here 0, 1, 2 and 3. The figure below shows three possible outputs:
- left: a well-tuned SOM, where each quadrant of neurons represents a digit;
- center: a SOM with a high learning rate. The large step size makes the neurons *step* over long distances during training. This mixes digits on nearby neurons, and may produce a *tangled* SOM, in which neighbouring neurons no longer represent nearby inputs;
- right: a SOM with a large neighbourhood width. The update reaches almost all neurons, which then move together and end with similar weights.

{: .picture}
<img width="29.4%" height="29.4%" src="{{ site.assets }}/unsupervised-learning/kohonen_1.svg">
<span style="display:inline-block; width:1cm;"></span>
<img width="29.8%" height="29.8%" src="{{ site.assets }}/unsupervised-learning/kohonen_2.svg">
<span style="display:inline-block; width:1cm;"></span>
<img width="29.3%" height="29.3%" src="{{ site.assets }}/unsupervised-learning/kohonen_3.svg">
 
 

More examples are listed in the [Wikipedia entry for self-organizing maps](https://en.wikipedia.org/wiki/Self-organizing_map#Examples).

## t-SNE and UMAP

PCA preserves the global, linear structure of the data. This is often not the structure of interest when visualizing high-dimensional data in 2D or 3D. *Neighbour embedding* methods instead place similar points close together, which preserves local neighbourhoods.

[t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) (t-distributed Stochastic Neighbor Embedding, van der Maaten and Hinton, 2008) converts distances into probabilities. In the input space, the similarity of point $$x_j$$ to point $$x_i$$ is the probability that $$x_i$$ would pick $$x_j$$ as its neighbour, under a Gaussian centred at $$x_i$$:

$$
p_{j \mid i} = \frac{\exp \left( - \Vert x_i - x_j \Vert^2 / 2 \sigma_i^2 \right)}{\sum_{l \neq i} \exp \left( - \Vert x_i - x_l \Vert^2 / 2 \sigma_i^2 \right)}
$$

where each $$\sigma_i$$ is set so that the distribution $$p_{\cdot \mid i}$$ has a given *perplexity*, roughly the number of effective neighbours, typically 5 to 50, which gives points in dense regions a small $$\sigma_i$$. The symmetric similarity of the pair is $$p_{ij} = (p_{j \mid i} + p_{i \mid j}) / 2n$$. In the low-dimensional embedding $$y_1, \ldots, y_n$$, the similarities follow a Student t-distribution with one degree of freedom:

$$
q_{ij} = \frac{\left(1 + \Vert y_i - y_j \Vert^2 \right)^{-1}}{\sum_{l \neq m} \left( 1 + \Vert y_l - y_m \Vert^2 \right)^{-1}}
$$

The embedding minimizes the [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) $$KL(P \Vert Q) = \sum_{i \neq j} p_{ij} \log (p_{ij} / q_{ij})$$ with gradient descent, where:

$$
\frac{\partial KL}{\partial y_i} = 4 \sum_{j} (p_{ij} - q_{ij}) (y_i - y_j) \left( 1 + \Vert y_i - y_j \Vert^2 \right)^{-1}
$$

Each term of the gradient acts like a spring between $$y_i$$ and $$y_j$$, which pulls the points together when $$p_{ij} > q_{ij}$$ and pushes them apart when $$p_{ij} < q_{ij}$$.

Why a t-distribution in the embedding? A high-dimensional space has much more room than a plane: in 10 dimensions, 11 points can all be at the same distance from each other, which is impossible in 2D. With a Gaussian in the embedding, the many moderately distant neighbours of a point would be crushed together, the *crowding problem*. The heavy tails of the t-distribution let them move far apart instead, which produces well-separated clusters. There are two caveats. First, the sizes of the clusters and the distances between clusters in a t-SNE plot carry little meaning. Second, the result depends on the perplexity and on the random initialization. Each iteration costs $$O(n^2)$$, or $$O(n \log n)$$ with the Barnes-Hut approximation, which treats groups of distant points as single points. Its trees grow exponentially with the dimension of the embedding, which limits it to embeddings in 2 or 3 dimensions.

[UMAP](https://umap-learn.readthedocs.io/) (Uniform Manifold Approximation and Projection, McInnes et al., 2018) follows the same idea with a different construction. It builds a weighted graph that connects each point to its $$h$$ nearest neighbours, with weights:

$$
v_{j \mid i} = \exp \left( - \frac{d(x_i, x_j) - \rho_i}{\sigma_i} \right)
$$

where $$\rho_i$$ is the distance from $$x_i$$ to its nearest neighbour, which therefore gets the weight 1, and $$\sigma_i$$ is set so that $$\sum_j v_{j \mid i} = \log_2 h$$, as the perplexity sets $$\sigma_i$$ in t-SNE. The weights are symmetrized as $$v_{ij} = v_{j \mid i} + v_{i \mid j} - v_{j \mid i} \, v_{i \mid j}$$. In the embedding, the similarities are $$w_{ij} = \left( 1 + a \Vert y_i - y_j \Vert^{2b} \right)^{-1}$$, where $$a$$ and $$b$$ are fitted to the minimum distance between embedded points, a parameter. The embedding minimizes the cross-entropy between the two graphs:

$$
\sum_{i \neq j} v_{ij} \log \frac{v_{ij}}{w_{ij}} + (1 - v_{ij}) \log \frac{1 - v_{ij}}{1 - w_{ij}}
$$

with stochastic gradient descent and negative sampling. The first term pulls together the points that are neighbours in the graph, and the second term pushes apart the points that are not. UMAP runs faster than t-SNE, its cost grows only linearly with the dimension of the embedding, and it can embed new points, so it also serves as a general dimensionality reduction method, not only for visualization. UMAP is often reported to preserve more of the global structure than t-SNE; with the same initialization, both preserve it similarly, and the difference comes from the spectral initialization of UMAP ([Kobak and Linderman, 2021][kobak2021]).

[kobak2021]: https://www.nature.com/articles/s41587-020-00809-z

## Autoencoders and Self-Supervised Learning

An [autoencoder](https://en.wikipedia.org/wiki/Autoencoder) is a neural network trained to reproduce its input. An encoder $$f$$ maps the input $$x$$ to a lower-dimensional *code* $$z = f(x)$$, and a decoder $$g$$ maps the code back to a reconstruction $$\hat{x} = g(z)$$. Both are trained jointly to minimize the reconstruction error, e.g.:

$$
\min_{f, g} \sum_{i} \Vert x_i - g(f(x_i)) \Vert^2
$$

The narrow code, the *bottleneck*, forces the network to keep only the most informative features of the data, since a code with fewer dimensions than the input cannot copy it. A linear autoencoder trained with the squared error learns the same subspace as PCA, although not necessarily the principal components themselves, because any invertible linear transformation of the code, undone by the decoder, gives the same reconstruction. An autoencoder with non-linear activations is therefore a non-linear generalization of PCA. Common variants are:
- *sparse autoencoders* add a penalty on the code activations, e.g. their L1 norm, so that each input activates only a few features, even when the code is larger than the input;
- *denoising autoencoders* reconstruct the clean input $$x$$ from a corrupted version $$\tilde{x}$$, e.g. with added noise or masked entries. This forces them to learn the structure of the data instead of the identity function;
- *variational autoencoders* learn a distribution over the codes, which turns the decoder into a generative model.

Denoising autoencoders are an early example of [self-supervised learning](https://en.wikipedia.org/wiki/Self-supervised_learning), where the training targets come from the data itself: part of the input is hidden, and the model predicts it. Most large models are pretrained this way today: language models predict the next token of a text, BERT ([Devlin et al., 2018](https://arxiv.org/abs/1810.04805)) predicts masked words, and masked autoencoders ([He et al., 2021](https://arxiv.org/abs/2111.06377)) reconstruct masked image patches. *Contrastive learning* is another self-supervised approach. Methods like SimCLR ([Chen et al., 2020](https://arxiv.org/abs/2002.05709)) create two augmented views of each image, e.g. with random cropping and color distortion. They train an encoder so that the embeddings $$z_i$$ and $$z_j$$ of two views of the same image are close to each other, and far from the embeddings of the other images in the batch. The loss is:

$$
\ell_{i,j} = - \log \frac{\exp \left( \text{sim}(z_i, z_j) / \tau \right)}{\sum_{l \neq i} \exp \left( \text{sim}(z_i, z_l) / \tau \right)}
$$

where $$\text{sim}$$ is the [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) and $$\tau$$ is a temperature. The loss is small when $$z_j$$ is the most similar to $$z_i$$ among all the other embeddings of the batch, so the encoder learns features that the augmentations do not change. The learned representations can then be used for clustering or retrieval, or fine-tuned on a supervised task with few labels.

## The Curse of Dimensionality

Most of the methods above compare points by their distances, and they struggle with high-dimensional data, in particular with Euclidean distances. This is due to the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality): as the dimension $$D$$ grows, the distances between any two points become similar, and the distance metric loses its meaning. For points with independent and identically distributed coordinates, the relative difference between the distances from a point to its farthest and to its nearest neighbour, $$(d_{max} - d_{min}) / d_{min}$$, tends to 0 as $$D$$ grows ([Beyer et al., 1999][beyer1999]; this [blog post](https://towardsdatascience.com/that-cursing-dimensionality-ac317fb0fdcc) covers the details). The reason is that a squared Euclidean distance is a sum of $$D$$ independent terms, one per coordinate: its mean grows in proportion to $$D$$, but its standard deviation only in proportion to $$\sqrt{D}$$, so all distances concentrate around the same value. This is one more reason to reduce the dimensionality of the data, e.g. with PCA, before clustering it.

[beyer1999]: https://link.springer.com/chapter/10.1007/3-540-49257-7_15
