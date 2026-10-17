---
layout: post
title:  "Unsupervised Learning basics and Principal Component Analysis"
categories: ["Machine Learning"]
tags: ["unsupervised learning", "PCA", "clustering", "dimensionality reduction"]
---

Unsupervised learning is the field of machine learning that learns without supervision or reward signals. Unlike [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) and [supervised learning]({{ site.baseurl }}{% post_url  2017-08-17-linear-regression-matrix-factorization %}), the method receives only the data. These methods are still very powerful. Common applications are image compression, dimensionality reduction, clustering, visualization, and learning representations of the data that other models then build on.

## Foreword: Hebbian Learning

Hebbian learning ([Donald Hebb](https://en.wikipedia.org/wiki/Donald_O._Hebb), 1949) is probably the oldest of the many *biologically-inspired* learning models. Its principle underlies many unsupervised learning rules, including several in this post. It is based on the synaptic changes between neurons that spike at close instants in time (motto: "*if they fire together, they wire together*"). Spike-Timing-Dependent Plasticity ([STDP, Markram et al., Frontiers](https://www.frontiersin.org/articles/10.3389/fnsyn.2012.00002/full)) shows that the strength of a synapse increases or decreases depending on the time difference between the spikes of the pre- and post-synaptic neurons. In essence, if a neuron spikes and causes the output neuron to fire, the synapse between them is strengthened.

In Hebb's own words (1949):

> When an axon of cell j repeatedly or persistently takes part in firing cell i, then j's efficiency as one of the cells firing i is increased.

Artificial neurons follow the same principle: the weight of the connection between two neurons adapts in the same way. The general Hebb's rule describes the change over time of the weight $$w_{ij}$$, from a pre-synaptic neuron $$i$$ with activity $$x_i$$ to a post-synaptic neuron $$j$$ with output $$y_j$$:

$$
\frac{d}{dt}w_{ij} = F (w_{ij}; x_i, y_j)
$$

The simplest Hebbian learning rule is:

$$
\frac{d}{dt}w_{ij} = \eta \, x_i \, y_j
$$

where $$\eta$$ is the learning rate. Hebbian learning is **local**: it only looks at the two neurons involved and ignores the activity of the rest of the system. It is a form of **unsupervised learning**, i.e. learning from the data alone, without any external signal or measure of accuracy. When both activities are positive, the simple rule above only increases the weights. A common way to keep the weights bounded is [Oja's learning rule](http://www.scholarpedia.org/article/Oja_learning_rule):

$$
\frac{d}{dt}w_{ij} = \eta \, (x_i y_j - y_j^2 w_{ij}).
$$

The squared output $$y_j^2$$ makes the balancing effect stronger as the output of the neuron grows. For a linear neuron (below) and inputs of zero mean, Oja's rule converges to a weight vector of unit norm along the first principal component of the inputs. Principal components are the topic of the next section.
Another common formulation is the covariance rule

$$
\frac{d}{dt}w_{ij} = \eta \, (x_i - \langle x_i \rangle) ( y_j - \langle y_j \rangle )
$$

which assumes that $$x$$ and $$y$$ fluctuate around their mean values $$ \langle x \rangle $$ and $$ \langle y \rangle $$. Another option is the Bienenstock-Cooper-Munro (BCM, 1982) rule:

$$
\frac{d}{dt}w_{ij} = \eta \, x_i \, y_j \, (y_j - \theta_j)
$$

which models learning in the visual cortex. The weight increases (long-term potentiation, LTP) when the post-synaptic activity is above the threshold $$\theta_j$$, and decreases (long-term depression, LTD) when it is below. The threshold slides with the time-averaged post-synaptic activity (e.g. $$\theta_j = \langle y_j^2 \rangle$$), which stabilizes synaptic plasticity.

The output of a neuron is typically a function of the weighted sum of its inputs:

$$
y_j = g \left( \sum_i w_{ij} x_i \right)
$$

where $$g$$ is the [activation function](https://en.wikipedia.org/wiki/Activation_function). The simplest case is the **linear neuron**:

$$
y_j = \sum_i w_{ij} x_i
$$

**Sigmoidal functions** are commonly used instead (see [here](https://en.wikipedia.org/wiki/Sigmoid_function#Examples) for a list of examples), as they describe an output more *powerfully* than a linear function. In general, unsupervised learning follows $$\Delta w \propto F(pre,post)$$. Adding a reward signal to each update, i.e. $$\Delta w \propto F(pre, post, reward)$$, leads to **reinforcement learning**. Unsupervised learning relates to the synaptic weight changes in the brain caused by [Long-Term Potentiation (LTP)](https://en.wikipedia.org/wiki/Long-term_potentiation) and [Long-Term Depression (LTD)](https://en.wikipedia.org/wiki/Long-term_depression). Reinforcement learning relates more to the dopamine [reward system](https://en.wikipedia.org/wiki/Reward_system) of the brain, an external factor that rewards the learning of a behaviour.

## Principal Component Analysis

Principal Component Analysis (PCA) is a dimensionality (feature) reduction method. It rotates the data towards the directions of largest variance, so that the data can be analysed along its most meaningful dimensions, and the dimensions of low variance can be discarded. Gerstner's lecture notes on unsupervised learning define it as:

<blockquote class="blockquote text-right">
Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components.
</blockquote>

Given $$n$$ data points $$x_1, ..., x_n$$, the algorithm is:
- subtract the mean to center the data: $$x_i \leftarrow x_i - \bar{x}$$, with $$\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$$;
- compute the [covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix) of the centered data: $$ C = \frac{1}{n-1} \sum_{i=1}^n x_i \, x_i^{\intercal}  $$;
- compute the [eigenvalues and eigenvectors](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors) of the covariance matrix:
  - solve $$\det(C - \lambda I)=0$$ to obtain the eigenvalues $$\lambda$$;
  - for each eigenvalue $$\lambda_i$$, compute its eigenvector $$v_i$$ by solving $$C v_i = \lambda_i v_i \, \Leftrightarrow \, (C- \lambda_i I) v_i = 0$$;
- build the feature matrix $$F$$, with the eigenvectors as columns, sorted by decreasing eigenvalue;
- obtain the transformed data by multiplying the transposed feature matrix ($$F^T$$, i.e. with the most significant eigenvector on top) by the matrix $$D_c$$ whose columns are the centered data points: $$D_n = F^T D_c$$.

The figure below shows an example, where PCA rotates the data towards the new axes of the cluster of data points:

{: style="text-align:center; font-size: small;"}
<img width="30%" height="30%" src="{{ site.assets }}/unsupervised-learning/gaussianscatterpca.svg">

{: style="text-align:center; font-size: small;"}
adapted from: wikipedia

The least significant dimensions of the transformed data can now be discarded, which reduces the number of features of the model. The number of eigenvectors kept in $$F$$ sets the amount of compression. The first $$k$$ components explain a fraction $$\sum_{i=1}^k \lambda_i / \sum_{i} \lambda_i$$ of the variance of the data, a common criterion to choose $$k$$.

Inverting the transformation recovers the original data, or an approximation of it if $$F$$ does not include all eigenvectors. The inversion multiplies the transformed data by the inverse of $$F^T$$. The eigenvectors of the symmetric matrix $$C$$ are orthonormal, so $$F^TF = I$$. When $$F$$ keeps all eigenvectors, it is square and orthogonal, so $$F^{-T} = F$$ and:

$$
D_c = F^{-T} D_n = F D_n
$$

When $$F$$ keeps only $$k$$ eigenvectors, $$F D_n$$ is the projection of the data onto the first $$k$$ principal components. This projection is the best approximation of $$D_c$$ in $$k$$ dimensions, in the least-squares sense.

In practice, PCA is usually computed with the [singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) of the centered data: if $$D_c^T = U \Sigma V^T$$, the columns of $$V$$ are the eigenvectors of $$C$$, and its eigenvalues are $$\lambda_i = \sigma_i^2 / (n-1)$$. This avoids computing $$C$$, and is faster and numerically more stable.

## Kernel PCA

PCA only finds linear structure. [Kernel PCA](https://en.wikipedia.org/wiki/Kernel_principal_component_analysis) (Schölkopf et al., 1998) applies PCA in a high-dimensional feature space $$\phi(x)$$, where non-linear structure of the input space may become linear. The *kernel trick* avoids computing $$\phi$$, because the method only needs the dot products $$k(x_i, x_j) = \phi(x_i)^T \phi(x_j)$$. A kernel function provides these dot products, e.g. the Gaussian kernel $$k(x_i, x_j) = \exp \left( - \| x_i - x_j \|^2 / 2 \sigma^2 \right)$$ or the polynomial kernel $$k(x_i, x_j) = (x_i^T x_j + c)^d$$. The algorithm is:
- build the $$n \times n$$ kernel matrix $$K$$, with $$K_{ij} = k(x_i, x_j)$$;
- center the data in the feature space: $$\tilde{K} = K - 1_n K - K 1_n + 1_n K 1_n$$, where $$1_n$$ is the $$n \times n$$ matrix with all entries equal to $$1/n$$;
- compute the eigenvalues $$\lambda_m$$ and eigenvectors $$\alpha^{(m)}$$ of $$\tilde{K}$$, and scale each eigenvector so that $$\lambda_m \, \alpha^{(m)T} \alpha^{(m)} = 1$$. This gives the corresponding direction in the feature space a unit norm;
- the $$m$$-th principal component of a data point $$x_j$$ is $$\sum_{i=1}^n \alpha_i^{(m)} \tilde{K}_{ij}$$, which is the $$j$$-th entry of $$\tilde{K} \alpha^{(m)} = \lambda_m \alpha^{(m)}$$.

Kernel PCA requires the eigendecomposition of an $$n \times n$$ matrix, which limits it to datasets of moderate size. In addition, mapping a point back to the input space (the *pre-image* problem) is only approximate.

## Independent Component Analysis

This section is largely based on the lecture notes of Tatsuya Yokota, Assistant Professor at the Nagoya Institute of Technology.

Independent Component Analysis (ICA) is a related method for [blind source separation](https://en.wikipedia.org/wiki/Blind_signal_separation). It separates a multidimensional signal into additive subcomponents, under two assumptions:
- the source signals $$s_1$$, ..., $$s_n$$ are statistically independent: $$p(s_1, s_2, ..., s_n) = p(s_1)\,\,p(s_2) \,\, ... \,\, p(s_n)$$;
- the sources are not Gaussian (at most one of them may be Gaussian); otherwise, ICA is impossible.

ICA builds on the [Central Limit Theorem](https://en.wikipedia.org/wiki/Central_limit_theorem): the distribution of the sum of $$N$$ independent random variables approaches a Gaussian as $$N \rightarrow \infty$$. For example, a single die throw has a uniform distribution from 1 to 6, but the sum of several throws does not. As the number of throws increases, the distribution of the sum gets closer to a Gaussian. A mixture of independent signals is therefore *more Gaussian* than the signals themselves, and the most non-Gaussian directions recover the independent signals. In short, *non-Gaussianity is independence*.

A well-known example is the [cocktail party](https://en.wikipedia.org/wiki/Cocktail_party_effect) problem. A guest hears observed signals $$x$$ that are mixtures of the original signals $$s$$:

$$
\begin{align*}
x_1(t) = a_{11}s_1(t) + a_{12}s_2(t) + a_{13}s_3(t)\\
x_2(t) = a_{21}s_1(t) + a_{22}s_2(t) + a_{23}s_3(t)\\
x_3(t) = a_{31}s_1(t) + a_{32}s_2(t) + a_{33}s_3(t)
\end{align*}
$$

The signals $$s_1$$, $$s_2$$ and $$s_3$$ are assumed to be statistically independent. ICA estimates the independent components $$s(t)$$ from $$x(t)$$, with

$$
x(t) = As(t)
$$

where $$A$$ is an invertible mixing matrix. The problem is therefore written as $$s(t) = Bx(t)$$, where $$B = A^{-1}$$. Only $$B$$ must be estimated, such that the $$s_i$$ are independent.

Signals $$z$$ are *white* if they satisfy $$E [z] = 0$$ and $$E [ z z^T ] = I$$. Whitening is a common pre-processing step for ICA. The whitened signals are $$z (t) = V x(t)$$, where $$V$$ is the whitening matrix (e.g. $$V = \Lambda^{-1/2} E^T$$, with the eigenvalues $$\Lambda$$ and the eigenvectors $$E$$ of the covariance matrix of $$x$$, as in PCA). The model becomes:

$$
s(t) = U z (t) = U V x (t) = B x (t)
$$

where $$U$$ is an orthogonal matrix. **Whitening simplifies ICA**, since only $$U$$ remains to be estimated. The figures below show the source, observed and whitened data:

{: style="text-align:center; font-size: small;"}
<img width="20%" height="20%" src="{{ site.assets }}/unsupervised-learning/ica_source.svg">
<img width="20%" height="20%" src="{{ site.assets }}/unsupervised-learning/ica_observed.svg"> 
<img width="20%" height="20%" src="{{ site.assets }}/unsupervised-learning/ica_whitening.svg">

{: style="text-align:center; font-size: small;"}
adapted from: T. Yokota, Lecture notes, Nagoya Institute of Technology

**Non-Gaussianity is a measure of independence**. Kurtosis measures non-Gaussianity and is defined as:

$$
k(y) =  E [y^4] - 3 (E[y^2])^2.
$$

Kurtosis is zero for a Gaussian. If $$y$$ is white, i.e. $$E[y]=0$$ and $$E[y^2]=1$$, it simplifies to

$$
k(y) = E[y^4]-3.
$$

Each row $$b^T$$ of $$B$$ recovers one source, $$s_i = b^T x$$. The ICA problem is then solved with

$$
\hat{b} = \arg\max_b | k(b^T x(t)) |
$$

where $$b^T x(t)$$ is constrained to unit variance. On the whitened data, $$b^T x(t) = w^T z(t)$$, where $$w^T = b^T V^{-1}$$ is a row of $$U$$, and the unit-variance constraint becomes $$w^T w=1$$. ICA therefore maximizes the absolute value of the *kurtosis* $$\mid k(w^Tz) \mid$$, subject to $$w^T w=1$$. The gradient is:

$$
\begin{align*}
\frac{\partial |\mathrm{kurt}(\boldsymbol{w}^T \boldsymbol{z})|}{\partial \boldsymbol{w}} & = \frac{\partial}{\partial \boldsymbol{w}} \left| \mathrm{E}\{(\boldsymbol{w}^T \boldsymbol{z})^4\} - 3\mathrm{E}\{(\boldsymbol{w}^T \boldsymbol{z})^2\}^2 \right| \\
& = \frac{\partial}{\partial \boldsymbol{w}} \left| \mathrm{E}\{(\boldsymbol{w}^T \boldsymbol{z})^4\} - 3\{||\boldsymbol{w}||^2\}^2 \right| \quad \text{(because } \mathrm{E}(\boldsymbol{z} \boldsymbol{z}^T) = I) \\
& = 4\mathrm{sign}[\mathrm{kurt}(\boldsymbol{w}^T \boldsymbol{z})] \left[ \mathrm{E}\{\boldsymbol{z} (\boldsymbol{w}^T \boldsymbol{z})^3\} - 3\boldsymbol{w} ||\boldsymbol{w}||^2 \right]
\end{align*}
$$

Gradient ascent then solves the problem:
- $$\Delta w \propto \text{sign} [ k(w^T z)] \, ( E [ z(w^T z)^3 ] - 3w )$$;
- $$w \leftarrow w + \Delta w$$, followed by the normalization $$w \leftarrow w / \| w \|$$.

Alternatively, since the algorithm converges when $$w \propto \Delta w$$, and $$w$$ and $$-w$$ are equivalent, the fixed-point iteration of **FastICA** finds the solution:
- $$w \leftarrow E [ z ( w^T z)^3 ] - 3w$$, followed by $$w \leftarrow w / \| w \|$$.

The remaining components are found in the same way, keeping each new $$w$$ orthogonal to the previous ones.

Note that *kurtosis is very sensitive to outliers, because it is a fourth-order function*. The negentropy is a more robust alternative. It is omitted here for brevity; the details are in the [original lecture notes](https://www.slideshare.net/yokotatsuya/independent-component-analysis-11359849).

## Competitive Learning and k-means

Competitive learning and k-means are not to be confused with the [k-nearest neighbours algorithm (KNN)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm).

Competitive learning underlies the self-organization of the brain. Neural network wiring in the brain is adaptive, not fixed, and competition drives the wiring of all brain systems (motor cortex, neocortex, etc.).
A common competitive learning technique for data reduction is **prototype clustering**. It represents the data points $$x_u$$ by a smaller set of prototypes $$w_k$$, each representing a cluster of points. The general rule is that *a new point $$x$$ belongs to the closest prototype $$k$$*, i.e.

$$
\| w_k - x \| \leq \| w_i - x \| \text{, for all } i
$$

This type of partitioning underlies problems such as the [Voronoi diagram](https://en.wikipedia.org/wiki/Voronoi_diagram) for spatial data partitioning. In the 2D example below, the points of each region (colored cells) belong to its prototype (black points):


{: style="text-align:center; font-size: small;"}
<img width="30%" height="30%" src="{{ site.assets }}/unsupervised-learning/voronoi.svg">

{: style="text-align:center; font-size: small;"}
adapted from: wikipedia


These methods apply to e.g. image compression, where every RGB pixel in the range $$[0-255, 0-255, 0-255]$$ is reduced to the closest color of a smaller set. The (squared) reconstruction error is the sum of the errors of all data points over all clusters $$C_k$$:

$$
E = \sum_k \sum_{u \in C_k} \| w_k - x_u \|^2
$$

**k-means clustering** is an iterative method for prototype clustering. It finds the partition $$S = \{ S_1, ..., S_k \}$$ of the data points into $$k$$ clusters that minimizes the error above:

$$
\arg\min_S \sum_{i=1}^{k} \sum_{x \in S_i} \| x - w_i \|^2
$$

where $$w_i$$ is the mean of the points in $$S_i$$. The algorithm alternates between 2 steps:
- assignment step: assign each observation to the closest mean, using e.g. the [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance);
- update step: set the new means to the centroids of the observations in each new cluster:
  $$ w_i^{(t+1)} = \frac{1}{\mid C^{(t)}_i \mid} \sum_{x_j \in C_i^{(t)}} x_j $$

The algorithm converges when the assignments no longer change. The means are then the centers of mass of their clusters. With the update $$\Delta w_i = \eta \sum_{u \in C_i} (x_u - w_i)$$ over the $$N$$ points of cluster $$C_i$$, convergence means $$\Delta w_i = 0$$, which requires:

$$
\eta \sum_{u \in C_i} (x_u - w_i) = 0 \rightarrow \sum_{u \in C_i} x_u - N w_i =0 \rightarrow w_i = \frac{1}{N} \sum_{u \in C_i} x_u
$$

which is the definition of the center of mass of cluster $$i$$. The error decreases at every step and the number of partitions is finite, so k-means always converges, but possibly to a local minimum. For this reason, it is usually run several times from different initial means, e.g. means picked far from each other with [k-means++](https://en.wikipedia.org/wiki/K-means%2B%2B). The figure below illustrates the algorithm:


{: style="text-align:center; font-size: small;"}
<img width="70%" height="70%" src="{{ site.assets }}/unsupervised-learning/k-means.svg">

{: style="text-align:center; font-size: small;"}
adapted from: wikipedia


In a 2D space, a run with 3 clusters looks as follows:

{: style="text-align:center; font-size: small;"}
<img width="30%" height="30%" src="{{ site.assets }}/unsupervised-learning/k-means_convergence.gif">

{: style="text-align:center; font-size: small;"}
source: wikipedia


The update rule above is a **batch learning rule**, as it updates the weights based on a set (*batch*) of inputs. The alternative is the **online update** rule, which updates the weights of the winning prototype $$i$$ for every new input $$x_u$$:

$$
\Delta w_i = \eta (x_u - w_i)
$$

which is Oja's Hebbian learning rule for a winner-take-all neuron, with output $$y_i = 1$$ for the winner and $$0$$ for all other neurons.

K-means has 2 main limitations:
- it forces clusters to be spherical, although elliptical clusters are sometimes desirable;
- each element belongs to exactly one cluster, which is not always appropriate.

Gaussian [mixture models](https://en.wikipedia.org/wiki/Mixture_model) address both limitations. A mixture model is a mixture of distributions (here, Gaussians) that represents the probability distribution of the observations in the overall population.

## Gaussian Mixture Models and Expectation-Maximization

A Gaussian mixture model (GMM) describes the data as drawn from $$K$$ Gaussians. Each data point picks a component $$k$$ with probability $$\pi_k$$, and is then drawn from the Gaussian $$\mathcal{N}(\mu_k, \Sigma_k)$$, so that:

$$
p(x) = \sum_{k=1}^K \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k), \text{ with } \sum_{k=1}^K \pi_k = 1.
$$

The covariance matrices $$\Sigma_k$$ give each cluster an elliptical shape, and each point belongs to every cluster with some probability. The parameters are learned by maximizing the log-likelihood $$\sum_{n=1}^N \log p(x_n)$$ of the $$N$$ data points with the [Expectation-Maximization (EM)](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm) algorithm. Like k-means, EM alternates between two steps:
- E-step: compute the *responsibility* of each component $$k$$ for each point $$x_n$$, i.e. the posterior probability that $$x_n$$ was drawn from that component:

$$
\gamma_{nk} = \frac{\pi_k \, \mathcal{N}(x_n \mid \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \, \mathcal{N}(x_n \mid \mu_j, \Sigma_j)}
$$

- M-step: update the parameters with the responsibilities as weights, where $$N_k = \sum_{n} \gamma_{nk}$$:

$$
\mu_k = \frac{1}{N_k} \sum_{n} \gamma_{nk} \, x_n, \quad
\Sigma_k = \frac{1}{N_k} \sum_{n} \gamma_{nk} \, (x_n - \mu_k)(x_n - \mu_k)^T, \quad
\pi_k = \frac{N_k}{N}
$$

No iteration decreases the log-likelihood, and the algorithm converges to a local maximum. As with k-means, EM is therefore usually run from several initializations, or initialized with the result of k-means. In fact, k-means is a limit case of EM: for a GMM whose components share the covariance $$\sigma^2 I$$, the responsibilities become hard assignments to the closest mean as $$\sigma \rightarrow 0$$. The number of components $$K$$ can be chosen with a model selection criterion, such as the [Bayesian information criterion](https://en.wikipedia.org/wiki/Bayesian_information_criterion). The Bayesian treatment of GMMs with variational inference is covered in [another post]({{ site.baseurl }}{% post_url 2019-07-01-variational-inference-gmm %}).

## Hierarchical Clustering

[Hierarchical clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering) builds a tree of clusters (a *dendrogram*) instead of a single partition. The common *agglomerative* (bottom-up) variant starts with one cluster per data point and repeatedly merges the two closest clusters, until a single cluster remains. The *linkage* criterion defines the distance between two clusters $$A$$ and $$B$$:
- single linkage: the distance between their closest points, $$\min_{a \in A, b \in B} d(a,b)$$. It finds elongated clusters, but a few close points can chain distinct clusters together;
- complete linkage: the distance between their farthest points, $$\max_{a \in A, b \in B} d(a,b)$$. It favours compact clusters;
- average linkage: the average distance over all pairs of points, $$\frac{1}{\mid A \mid \mid B \mid} \sum_{a \in A} \sum_{b \in B} d(a,b)$$;
- Ward's linkage: the increase of the k-means error (the sum of squared distances to the cluster means) caused by merging $$A$$ and $$B$$. For cluster means $$\mu_A$$ and $$\mu_B$$, this increase is $$\frac{\mid A \mid \mid B \mid}{\mid A \mid + \mid B \mid} \| \mu_A - \mu_B \|^2$$.

Cutting the dendrogram at a given height yields a partition, so the number of clusters can be chosen afterwards by inspecting the tree. The *divisive* (top-down) variant works in the opposite direction: it starts with all points in one cluster and splits clusters recursively. Agglomerative clustering needs the $$n \times n$$ matrix of pairwise distances, and runs in $$O(n^2)$$ to $$O(n^3)$$ time, depending on the linkage and the implementation. It is therefore suited to small and medium datasets.

## DBSCAN

[DBSCAN](https://en.wikipedia.org/wiki/DBSCAN) (Density-Based Spatial Clustering of Applications with Noise, Ester et al., 1996) defines clusters as dense regions separated by sparse regions. It takes two parameters, a radius $$\varepsilon$$ and a minimum number of points $$m$$, and labels each point as one of:
- a *core point*, if at least $$m$$ points (itself included) lie within distance $$\varepsilon$$ of it;
- a *border point*, if it is not a core point but lies within distance $$\varepsilon$$ of a core point;
- *noise*, in all other cases.

Two core points within distance $$\varepsilon$$ of each other belong to the same cluster. Each cluster is thus a connected group of core points, plus the border points around them. Unlike k-means, DBSCAN does not need the number of clusters as input, finds clusters of arbitrary shapes, and labels outliers as noise. However, a single radius $$\varepsilon$$ cannot capture clusters of very different densities. The extensions [OPTICS](https://en.wikipedia.org/wiki/OPTICS_algorithm) and [HDBSCAN](https://hdbscan.readthedocs.io/) address this limitation. In low dimensions, a spatial index finds the neighbours of each point without computing all pairwise distances.

## Spectral Clustering

[Spectral clustering](https://en.wikipedia.org/wiki/Spectral_clustering) treats clustering as a graph partitioning problem. It builds a graph of the data points, with a similarity $$W_{ij}$$ for each pair of points, e.g. $$W_{ij} = \exp ( - \| x_i - x_j \|^2 / 2\sigma^2 )$$ between each point and its nearest neighbours, and $$0$$ otherwise. It then looks for clusters connected by edges of small weight. With the degree matrix $$D$$, a diagonal matrix with $$D_{ii} = \sum_j W_{ij}$$, the graph [Laplacian](https://en.wikipedia.org/wiki/Laplacian_matrix) is $$L = D - W$$, and for any vector $$f$$:

$$
f^T L f = \frac{1}{2} \sum_{i,j} W_{ij} (f_i - f_j)^2
$$

Therefore, $$f^T L f$$ is small when $$f$$ takes similar values on strongly connected points, and the eigenvectors of $$L$$ with the smallest eigenvalues vary little within clusters. In particular, the multiplicity of the eigenvalue $$0$$ equals the number of connected components of the graph. The algorithm is:
- compute the eigenvectors $$u_1, ..., u_k$$ of the $$k$$ smallest eigenvalues of $$L$$, or of the normalized Laplacian $$L_{sym} = I - D^{-1/2} W D^{-1/2}$$;
- stack these eigenvectors as the columns of an $$n \times k$$ matrix $$U$$. Its $$i$$-th row is the new $$k$$-dimensional representation of point $$x_i$$ (with $$L_{sym}$$, each row is first normalized to unit length);
- cluster the rows of $$U$$ using k-means.

This is a relaxation of the NP-hard problem of finding the *ratio cut* (with $$L$$) or the *normalized cut* (with $$L_{sym}$$) of the graph. Spectral clustering separates non-convex clusters (e.g. two concentric rings) that k-means cannot separate. The cost is the eigendecomposition of an $$n \times n$$ (sparse) matrix.

## Self-Organizing Maps and Kohonen Maps

A self-organizing map (SOM) is an unsupervised artificial neural network, typically a 2D grid of neurons, that represents a discretization of the input space (or of the training set). Its main purpose is to represent the training data by localized regions (neurons), each representing an agglomerate of inputs. Neighbouring neurons in the grid represent nearby regions of the input space:

{: style="text-align:center; font-size: small;"}
<img width="40%" height="40%" src="{{ site.assets }}/unsupervised-learning/kohonen.svg">
<img width="43%" height="43%" src="{{ site.assets }}/unsupervised-learning/kohonen2.svg">

{: style="text-align:center; font-size: small;"}
adapted from: Unsupervised and Reinforcement Learning lecture notes from W. Gerstner at EPFL


The algorithm is:
- place the neurons randomly in the multidimensional space;
- at every iteration, choose an input $$x_u$$;
- find the *winner* neuron $$k$$ (or *best matching unit*, BMU), i.e. the closest neuron in space:

$$
\| w_k - x_u \| \leq \| w_i - x_u \| \text{, for all } i
$$

- update the weights of the winner and of its neighbours, pulling them closer to the input vector:

$$
\Delta w_j = \eta \, \Lambda (j,k) \, (x_u - w_j)
$$

where $$\Lambda (j,k)$$ is the *neighbourhood function*. It decreases with the distance between neurons $$j$$ and $$k$$ in the grid (e.g. a Gaussian). It is also common to decrease $$\eta$$ and the width of $$\Lambda$$ as the iterations progress;
- repeat until stabilization, i.e. until the maximum number of iterations is reached, or until the change in weights falls below a given threshold.

Videos of an iterative execution of the Kohonen map are available [in this link](https://www.youtube.com/watch?v=IixbH1gDhsg) and [this link](https://www.youtube.com/watch?v=zeOtwCAI3Fs).

A good application is the recognition of handwritten digits from the [MNIST database](http://yann.lecun.com/exdb/mnist/).
Each digit is a 28x28 (=784) pixel array, represented as a data point in a 784-dimensional space. Each pixel is monochromatic, so each dimension takes values between 0 and 255. The task is to find, from a database of 5000 handwritten digits, the 6x6 SOM that best represents 4 of the digits from 0 to 9. The experiment studies how the neighbourhood width and the learning rate affect the performance of the Kohonen map.

The input dataset and the `python` implementation are available in <a href="{{ site.assets }}/unsupervised-learning/kohonen_mnist.zip">kohonen_mnist.zip</a>. The goal is to tune the neighbourhood width and the learning rate $$\eta$$. A good solution gives a visual perception of the digits being identified (in this case 0, 1, 2 and 3). The figure below shows three possible outputs:
- left: a well-tuned SOM, where each quadrant of neurons represents a digit;
- center: a SOM with a high learning rate. The large step size makes neurons *step* over long distances during training. This mixes digits on nearby neurons, and may produce a *tangled* SOM;
- right: a SOM with a large neighbourhood width. Due to the very wide range of the update function, all neurons behave almost synchronously, which leads to very similar outputs per neuron.


{: style="text-align:center; font-size: small;"}
<img width="25%" height="25%" src="{{ site.assets }}/unsupervised-learning/kohonen_1.svg">
<span style="display:inline-block; width:1cm;"></span>
<img width="25%" height="25%" src="{{ site.assets }}/unsupervised-learning/kohonen_2.svg">
<span style="display:inline-block; width:1cm;"></span>
<img width="25%" height="25%" src="{{ site.assets }}/unsupervised-learning/kohonen_3.svg">
 
 
More examples are listed in the [Wikipedia entry for self-organizing maps](https://en.wikipedia.org/wiki/Self-organizing_map#Examples).

## t-SNE and UMAP

PCA preserves the global, linear structure of the data. This is often not the structure of interest when visualizing high-dimensional data in 2D or 3D. *Neighbour embedding* methods instead place similar points close together, which preserves local neighbourhoods.

[t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) (t-distributed Stochastic Neighbor Embedding, van der Maaten and Hinton, 2008) converts distances into probabilities. In the input space, the similarity of point $$x_j$$ to point $$x_i$$ is:

$$
p_{j \mid i} = \frac{\exp \left( - \| x_i - x_j \|^2 / 2 \sigma_i^2 \right)}{\sum_{k \neq i} \exp \left( - \| x_i - x_k \|^2 / 2 \sigma_i^2 \right)}
$$

where each $$\sigma_i$$ is set so that the distribution $$p_{\cdot \mid i}$$ has a given *perplexity* (roughly the number of effective neighbours, typically 5 to 50). The symmetric similarity of the pair is $$p_{ij} = (p_{j \mid i} + p_{i \mid j}) / 2n$$. In the low-dimensional embedding $$y_1, ..., y_n$$, similarities follow a Student t-distribution with one degree of freedom:

$$
q_{ij} = \frac{\left(1 + \| y_i - y_j \|^2 \right)^{-1}}{\sum_{k \neq l} \left( 1 + \| y_k - y_l \|^2 \right)^{-1}}
$$

The embedding is found by minimizing the Kullback-Leibler divergence $$KL(P \| Q) = \sum_{i \neq j} p_{ij} \log (p_{ij} / q_{ij})$$ with gradient descent, where:

$$
\frac{\partial KL}{\partial y_i} = 4 \sum_{j} (p_{ij} - q_{ij}) (y_i - y_j) \left( 1 + \| y_i - y_j \|^2 \right)^{-1}
$$

The heavy tails of the t-distribution allow moderately distant points to move far apart in the embedding. This avoids the *crowding problem* that a Gaussian would cause in low dimensions, and produces well-separated clusters. There are two caveats. First, cluster sizes and distances between clusters in a t-SNE plot carry little meaning. Second, the result depends on the perplexity and the random initialization. Each iteration costs $$O(n^2)$$, or $$O(n \log n)$$ with the Barnes-Hut approximation.

[UMAP](https://umap-learn.readthedocs.io/) (Uniform Manifold Approximation and Projection, McInnes et al., 2018) follows the same idea with a different construction. It builds a weighted graph that connects each point to its $$k$$ nearest neighbours, with weights:

$$
v_{j \mid i} = \exp \left( - \frac{d(x_i, x_j) - \rho_i}{\sigma_i} \right)
$$

where $$\rho_i$$ is the distance from $$x_i$$ to its nearest neighbour, and $$\sigma_i$$ is set so that $$\sum_j v_{j \mid i} = \log_2 k$$. The weights are symmetrized as $$v_{ij} = v_{j \mid i} + v_{i \mid j} - v_{j \mid i} \, v_{i \mid j}$$. In the embedding, the similarities are $$w_{ij} = \left( 1 + a \| y_i - y_j \|^{2b} \right)^{-1}$$, where $$a$$ and $$b$$ depend on the minimum distance between embedded points. The embedding minimizes the cross-entropy between the two graphs:

$$
\sum_{i \neq j} v_{ij} \log \frac{v_{ij}}{w_{ij}} + (1 - v_{ij}) \log \frac{1 - v_{ij}}{1 - w_{ij}}
$$

using stochastic gradient descent with negative sampling. The second term pushes apart points that are not neighbours, so UMAP tends to preserve more of the global structure than t-SNE. UMAP is also faster, scales to millions of points and can embed new points. It is therefore also used as a general dimensionality reduction method, not only for visualization.

## Autoencoders and Self-Supervised Learning

An [autoencoder](https://en.wikipedia.org/wiki/Autoencoder) is a neural network trained to reproduce its input. An encoder $$f$$ maps the input $$x$$ to a lower-dimensional *code* $$z = f(x)$$, and a decoder $$g$$ maps the code back to a reconstruction $$\hat{x} = g(z)$$. Both are trained jointly to minimize the reconstruction error, e.g.:

$$
\min_{f, g} \sum_{n} \| x_n - g(f(x_n)) \|^2
$$

The narrow code (the *bottleneck*) forces the network to keep only the most informative features of the data. A linear autoencoder trained with the squared error learns the same subspace as PCA, although not necessarily the principal components themselves. An autoencoder with non-linear activations is therefore a non-linear generalization of PCA. Common variants are:
- *sparse autoencoders* add a penalty on the code activations (e.g. their L1 norm), so that each input activates only a few features, even when the code is larger than the input;
- *denoising autoencoders* reconstruct the clean input $$x$$ from a corrupted version $$\tilde{x}$$ (e.g. with added noise or masked entries). This forces them to learn the structure of the data instead of the identity function;
- [variational autoencoders]({{ site.baseurl }}{% post_url 2019-10-12-vae-gan %}) learn a distribution over the codes, which turns the decoder into a generative model.

Denoising autoencoders are an early example of **self-supervised learning**, where the training targets come from the data itself: part of the input is hidden, and the model predicts it. Most large models are pretrained this way today. Language models predict the next token of a text (see the [GPT post]({{ site.baseurl }}{% post_url 2023-02-28-gptlite %})), BERT predicts masked words (see the [Transformers post]({{ site.baseurl }}{% post_url 2020-02-28-learning-from-sequences %})), and masked autoencoders reconstruct masked image patches. *Contrastive learning* is another self-supervised approach. Methods like [SimCLR](https://arxiv.org/abs/2002.05709) create two augmented views of each image. They train an encoder so that the embeddings $$z_i$$ and $$z_j$$ of two views of the same image are close to each other, and far from the embeddings of the other images in the batch. The loss is:

$$
\ell_{i,j} = - \log \frac{\exp \left( \text{sim}(z_i, z_j) / \tau \right)}{\sum_{k \neq i} \exp \left( \text{sim}(z_i, z_k) / \tau \right)}
$$

where $$\text{sim}$$ is the cosine similarity and $$\tau$$ is a temperature. The learned representations can then be used for clustering or retrieval, or fine-tuned on a supervised task with few labels.

## The Curse of Dimensionality

Distance-based methods struggle with high-dimensional data, in particular with Euclidean distances. This is due to the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality): as the number of dimensions increases, the distances between any two points become similar, and the distance metric becomes almost irrelevant. For example, for points with independent and identically distributed coordinates, the relative difference between the distances from a point to its farthest and to its nearest neighbour, $$(d_{max} - d_{min}) / d_{min}$$, tends to 0 as the number of dimensions grows. The proof is omitted here; this [blog post](https://towardsdatascience.com/that-cursing-dimensionality-ac317fb0fdcc) covers the details. This is one more reason to reduce the dimensionality of the data, e.g. with PCA, before clustering it.

