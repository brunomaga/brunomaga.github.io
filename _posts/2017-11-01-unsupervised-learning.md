---
layout: post
title:  "Unsupervised Learning basics and Principal Component Analysis"
categories: ["Machine Learning"]
tags: ["unsupervised learning", "clustering", "k-means", "KNN", "DBSCAN", "GMM", "self-organizing maps", "dimensionality reduction", "PCA", "t-SNE", "ICA", "association rules", "Apriori", "FP-growth", "anomaly detection", "isolation forest", "LOF", "curse of dimensionality", "nearest neighbor search", "RAG"]
description: "Unsupervised learning: clustering with k-means, KNN, DBSCAN, Gaussian mixtures and self-organizing maps, dimensionality reduction with PCA, t-SNE and ICA, association rules, anomaly detection, and nearest-neighbor search for RAG."
---

In 1901, Karl Pearson described the method known today as principal component analysis, in the paper [On lines and planes of closest fit to systems of points in space](https://doi.org/10.1080/14786440109462720). In 1957, Stuart Lloyd proposed the algorithm known today as k-means, to quantize signals, although his paper [Least squares quantization in PCM](https://doi.org/10.1109/TIT.1982.1056489) was only published in 1982.

Unsupervised learning is the field of machine learning dedicated to learning without supervision or reward signals. Unlike in [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) and [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning), the data is the only information given to the method. Still, these methods are very powerful, and are commonly used for image compression, dimensionality reduction, clustering, visualization and the detection of anomalies. This post starts with the clustering methods (k-means, the k-nearest neighbors, hierarchical clustering, DBSCAN, spectral clustering, Gaussian mixtures and self-organizing maps). It then covers dimensionality reduction, with principal component analysis (PCA) and kernel PCA, t-SNE and independent component analysis (ICA), association rule mining (Apriori, ECLAT and FP-growth), and anomaly detection (isolation forests and the local outlier factor). It ends with the curse of dimensionality, and with the nearest-neighbor search in high dimensions that retrieval-augmented generation relies on.

## Clustering

**[Clustering](https://en.wikipedia.org/wiki/Cluster_analysis)** (or cluster analysis) groups the data points so that the points of a group, a *cluster*, are more similar to each other than to the points of other clusters. Since no labels define the clusters, the methods of this section differ in what they call a cluster, from the points closest to a prototype (k-means) to a dense region of points (DBSCAN). A table at the end of the section compares them.

### K-Means Clustering

[Competitive learning](https://en.wikipedia.org/wiki/Competitive_learning) underlies the self-organization of the brain. In practice, neural network wiring is adaptive, not fixed, and competition drives wiring across all brain systems (motor cortex, neocortex, etc.).
A common technique for data reduction from competitive learning is **prototype clustering**, where the data points $$x_u$$ are represented by a smaller set of prototypes $$w_i$$ that represent clusters of points. The general rule is that *a new point $$x$$ belongs to the closest prototype $$i^*$$*, i.e.

$$
\| w_{i^*} - x \| \leq \| w_i - x \| \text{, for all } i
$$

This kind of partitioning underlies problems such as the [Voronoi diagram](https://en.wikipedia.org/wiki/Voronoi_diagram) for spatial data partitioning. A 2D example follows, where the points of each region (colored cells) belong to its prototype (black points):


{: .picture}
<img width="36.7%" height="36.7%" src="{{ site.assets }}/unsupervised-learning/voronoi.svg">

{: .caption}
The Voronoi diagram of 20 prototypes (black dots) in 2D: each colored cell holds the points closer to its prototype than to any other, in Euclidean distance. Adapted from: [Voronoi diagram](https://en.wikipedia.org/wiki/Voronoi_diagram)

Such methods can be applied to e.g. image compression, where one may reduce every RGB pixel in the range $$[0, 255]^3$$ to the closest color of a smaller set, a *palette*, a process called [color quantization](https://en.wikipedia.org/wiki/Color_quantization). With a palette of 16 colors, the prototypes, each pixel is stored as a 4-bit index into the palette instead of its 24 bits: the image takes 6 times less memory, plus the 48 bytes of the palette.

The reconstruction (squared) error is the sum of the errors of all data points over all clusters $$C_i$$:

$$
E = \sum_{i=1}^{k} \sum_{u \in C_i} \| w_i - x_u \|^2
$$

**[k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering)** is an iterative method for prototype clustering. Its objective is to find the partition of the data points into $$k$$ clusters $$C_1, ..., C_k$$ that minimizes the error above:

$$
\arg\min_{C_1, ..., C_k} \sum_{i=1}^{k} \sum_{x \in C_i} \| x - w_i \|^2
$$

where $$w_i$$ is the mean of the points in $$C_i$$. The standard algorithm, also called **Lloyd's algorithm**, iterates over 2 steps:
- assignment step: assign each observation to the closest mean, based on e.g. the [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance);
- update step: compute the new means as the centroids of the observations in each new cluster:
  $$ w_i^{(t+1)} = \frac{1}{\mid C^{(t)}_i \mid} \sum_{x_j \in C_i^{(t)}} x_j $$

The algorithm converges when the assignments no longer change, and the means are then the centers of mass of their clusters. With the update $$\Delta w_i = \eta \sum_{u \in C_i} (x_u - w_i)$$ over the $$\vert C_i \vert$$ points of cluster $$C_i$$, convergence means $$\Delta w_i = 0$$, so it must hold that:

$$
\eta \sum_{u \in C_i} (x_u - w_i) = 0 \rightarrow \sum_{u \in C_i} x_u - \vert C_i \vert \, w_i =0 \rightarrow w_i = \frac{1}{\vert C_i \vert} \sum_{u \in C_i} x_u
$$

which is the definition of the center of mass of cluster $$i$$. Neither step increases the error. The assignment step moves each point to its closest mean, and the update step moves each mean to the center of mass of its cluster, which minimizes the sum of the squared distances to its points. As there is a finite number of partitions, k-means always converges, but possibly to a local minimum. So it is usually run several times from different initial means, e.g. picked far from each other with [k-means++](https://en.wikipedia.org/wiki/K-means%2B%2B). An illustration of the algorithm follows:

{: .picture}
<img width="78.7%" height="78.7%" src="{{ site.assets }}/unsupervised-learning/k-means.svg">

{: .caption}
**1.** $$k = 3$$ initial means (colored circles) are chosen at random within the data domain. **2.** Assignment step: each point joins the cluster of the closest mean; the colored regions are the Voronoi cells of the means. **3.** Update step: each mean moves to the centroid of its cluster (dashed circles: the previous means; lines: the points of each cluster). **4.** Assignment step with the new means and their Voronoi cells: two points move from the green to the red cluster. The two steps repeat until the assignments no longer change. Adapted from: [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering)


In a 2D space, one can imagine a 3-cluster run as:

{: .picture}
<img width="30%" height="30%" src="https://upload.wikimedia.org/wikipedia/commons/e/ea/K-means_convergence.gif">

{: .caption}
Iterations of k-means with $$k = 3$$ on 2D data, with the Voronoi borders of the current means, until the assignments no longer change. Source: [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering)


The update rule above is a **batch learning rule**, as it updates the weights based on a set (*batch*) of inputs. The converse rule is the **online update** rule, which updates the weights of the winning prototype $$i$$ at the introduction of every new input $$x_u$$:

$$
\Delta w_i = \eta (x_u - w_i)
$$

which is [Oja's rule](https://en.wikipedia.org/wiki/Oja%27s_rule), a form of [Hebbian learning](https://en.wikipedia.org/wiki/Hebbian_theory) ("*neurons that fire together wire together*"), for a winner-take-all neuron, i.e. with an output $$y_i = 1$$ for the winner and $$0$$ for all other neurons.

### K-Nearest Neighbors

The **[k-nearest neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)** algorithm (KNN, [Cover and Hart, 1967](https://doi.org/10.1109/TIT.1967.1053964)) is not to be confused with k-means. It is not a clustering method either: it is a supervised classifier, as it needs the labels of the training points. It is covered here because its core operation, finding the $$k$$ points closest to a query, is the building block of many unsupervised methods of this post. DBSCAN counts the neighbors of each point within a radius, spectral clustering connects each point to its nearest neighbors, t-SNE and UMAP compute their similarities from the neighbors of each point, and the local outlier factor compares the density around a point with the density around its neighbors.

Given a query point $$x$$, KNN finds the $$k$$ training points closest to it, and predicts the most common label among them (classification), or the average of their values (regression). It has no training step: the model is the training set itself, which is why KNN is called a *lazy* learner. As an example, take the points 0, 1 and 2 of class A and the points 3.5, 6 and 7 of class B on a line, and the query point 3. With $$k = 1$$, its nearest neighbor is 3.5, at a distance of 0.5, and the query is labeled B. With $$k = 3$$, its neighbors are 3.5, 2 and 1, at the distances 0.5, 1 and 2, and the vote gives A, by 2 to 1. A small $$k$$ follows the training data closely and is sensitive to noise, such as a single point of B among the points of A, while a large $$k$$ gives smoother boundaries between classes, but blurs the small classes. In practice, $$k$$ is chosen by [cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)), and an odd $$k$$ avoids ties between two classes.

A brute-force search computes the distances from the query to all $$n$$ training points, i.e. $$O(nd)$$ operations per query for $$d$$ dimensions. A spatial index, such as the [k-d tree](https://en.wikipedia.org/wiki/K-d_tree) ([Bentley, 1975](https://doi.org/10.1145/361002.361007)), finds the neighbors in about $$O(\log n)$$ time in low dimensions, but not in high dimensions, as the last section of this post explains. Also, the distances must be meaningful: the features are usually normalized first, so that a feature with a large range does not dominate the distance.

### Hierarchical Clustering

**[Hierarchical clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering)** builds a tree of clusters (a *[dendrogram](https://en.wikipedia.org/wiki/Dendrogram)*) instead of a single partition. Its common *agglomerative* (bottom-up) variant starts with one cluster per data point, and repeatedly merges the two closest clusters, until a single cluster remains. The distance between two clusters $$A$$ and $$B$$ is given by the *linkage* criterion:
- single linkage: the distance between their closest points, $$\min_{a \in A, b \in B} d(a,b)$$. It finds elongated clusters, but can chain distinct clusters together through a few close points;
- complete linkage: the distance between their farthest points, $$\max_{a \in A, b \in B} d(a,b)$$, which favors compact clusters;
- average linkage: the average distance between all pairs of points, $$\frac{1}{\mid A \mid \mid B \mid} \sum_{a \in A} \sum_{b \in B} d(a,b)$$;
- Ward's linkage: the increase of the k-means error (the sum of squared distances to the cluster means) caused by merging $$A$$ and $$B$$, which is $$\frac{\mid A \mid \mid B \mid}{\mid A \mid + \mid B \mid} \| \mu_A - \mu_B \|^2$$ for cluster means $$\mu_A$$ and $$\mu_B$$.

As an example, take the five points 0, 1, 3, 4 and 8 on a line. With single linkage, the pairs $$\{0, 1\}$$ and $$\{3, 4\}$$ merge first, at a distance of 1. The two clusters then merge at a distance of 2 (from 1 to 3), and the point 8 joins last, at a distance of 4 (from 4 to 8). With complete linkage, the same merges happen at the heights 1, 1, 4 and 8, as the distance between two clusters is then the one between their farthest points.

Cutting the dendrogram at a given height yields a partition, so the number of clusters can be chosen after the fact, by looking at the tree. In the single linkage example, a cut at a height of 3 gives the two clusters $$\{0, 1, 3, 4\}$$ and $$\{8\}$$, and a cut at 1.5 gives three clusters. The *divisive* (top-down) variant does the opposite: it starts with all points in one cluster, and splits clusters recursively. Agglomerative clustering needs the $$n \times n$$ matrix of pairwise distances, and runs in $$O(n^2)$$ to $$O(n^3)$$ time depending on the linkage and the implementation, so it suits small to medium datasets.

### DBSCAN

**[DBSCAN](https://en.wikipedia.org/wiki/DBSCAN)** (Density-Based Spatial Clustering of Applications with Noise, [Ester et al., 1996](https://cdn.aaai.org/KDD/1996/KDD96-037.pdf)) defines clusters as dense regions separated by sparse ones. It takes two parameters, a radius $$\varepsilon$$ and a minimum number of points $$m$$, and labels each point as:
- a *core point*, if at least $$m$$ points (itself included) lie within distance $$\varepsilon$$ of it;
- a *border point*, if it is not a core point but lies within distance $$\varepsilon$$ of a core point;
- *noise*, otherwise.

Two core points within distance $$\varepsilon$$ of each other belong to the same cluster, so each cluster is a connected group of core points, plus the border points around them. Unlike k-means, DBSCAN does not need the number of clusters, finds clusters of arbitrary shapes, and labels outliers as noise. On the other hand, a single radius $$\varepsilon$$ cannot capture clusters of very different densities, a limitation addressed by its extensions [OPTICS](https://en.wikipedia.org/wiki/OPTICS_algorithm) and [HDBSCAN](https://hdbscan.readthedocs.io/). In low dimensions, a spatial index finds the neighbors of each point without computing all pairwise distances.

### Spectral Clustering

**[Spectral clustering](https://en.wikipedia.org/wiki/Spectral_clustering)** treats clustering as a graph partitioning problem. It builds a graph of the data points, with a similarity $$W_{ij}$$ for each pair of points, e.g. $$W_{ij} = \exp ( - \| x_i - x_j \|^2 / 2\sigma^2 )$$ between each point and its nearest neighbors and $$0$$ otherwise, and looks for clusters connected by edges of little weight. With the degree matrix $$D$$, a diagonal matrix with $$D_{ii} = \sum_j W_{ij}$$, the graph [Laplacian](https://en.wikipedia.org/wiki/Laplacian_matrix) is $$L = D - W$$, and for any vector $$f$$:

$$
f^T L f = \frac{1}{2} \sum_{i,j} W_{ij} (f_i - f_j)^2
$$

So $$f^T L f$$ is small when $$f$$ takes similar values on strongly connected points, and the eigenvectors of $$L$$ with the smallest eigenvalues vary little within clusters. In particular, the multiplicity of the eigenvalue $$0$$ is the number of connected components of the graph. The algorithm is:
- compute the eigenvectors $$u_1, ..., u_k$$ of the $$k$$ smallest eigenvalues of $$L$$, or of the normalized Laplacian $$L_{sym} = I - D^{-1/2} W D^{-1/2}$$;
- stack them as the columns of an $$n \times k$$ matrix $$H$$, and take its $$i$$-th row as the new $$k$$-dimensional representation of point $$x_i$$ (with $$L_{sym}$$, after normalizing each row to unit length);
- cluster the rows of $$H$$ with k-means.

This is a relaxation of the NP-hard problem of finding the *ratio cut* (with $$L$$) or the *normalized cut* (with $$L_{sym}$$) of the graph. Spectral clustering separates non-convex clusters (e.g. two concentric rings) that k-means cannot, at the cost of an eigendecomposition of an $$n \times n$$ (sparse) matrix.

For more details, [A Tutorial on Spectral Clustering](https://arxiv.org/abs/0711.0189) (von Luxburg, 2007) derives these algorithms from graph cuts, random walks and perturbation theory, and discusses which graph Laplacian to use.

### Gaussian Mixture Models (GMM)

K-means has two main problems:
- it forces the clusters to be spherical, while elliptical clusters are sometimes a better fit;
- each point belongs to one cluster only, which is not always a good choice for the points between two clusters.

Both problems are solved by Gaussian [mixture models](https://en.wikipedia.org/wiki/Mixture_model). A mixture model represents the probability distribution of the observations of the whole population as a mixture of distributions, in this case Gaussians. A **Gaussian mixture model** (GMM) describes the data as drawn from $$K$$ Gaussians: each data point picks a component $$k$$ with probability $$\pi_k$$, and is then drawn from the Gaussian $$\mathcal{N}(\mu_k, \Sigma_k)$$, so that:

$$
p(x) = \sum_{k=1}^K \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k), \text{ with } \sum_{k=1}^K \pi_k = 1.
$$

The covariance matrices $$\Sigma_k$$ give each cluster an elliptical shape, and each point belongs to every cluster with some probability. The parameters are learned by maximizing the log-likelihood $$\sum_{i=1}^n \log p(x_i)$$ of the $$n$$ data points, with the [Expectation-Maximization (EM)](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm) algorithm ([Dempster et al., 1977](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x)), which alternates two steps, like k-means:
- E-step: compute the *responsibility* of each component $$k$$ for each point $$x_i$$, i.e. the posterior probability that $$x_i$$ was drawn from it:

$$
\gamma_{ik} = \frac{\pi_k \, \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \, \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}
$$

- M-step: update the parameters, using the responsibilities as weights, with $$n_k = \sum_{i} \gamma_{ik}$$, the effective number of points of component $$k$$:

$$
\mu_k = \frac{1}{n_k} \sum_{i} \gamma_{ik} \, x_i, \quad
\Sigma_k = \frac{1}{n_k} \sum_{i} \gamma_{ik} \, (x_i - \mu_k)(x_i - \mu_k)^T, \quad
\pi_k = \frac{n_k}{n}
$$

As an example, take two components in one dimension, with the means 0 and 4, unit variances and equal weights. A point at $$x = 3$$ has the responsibility $$e^{-1/2} / (e^{-9/2} + e^{-1/2}) = 1/(1 + e^{-4}) = 0.982$$ for the component of mean 4, and 0.018 for the other one, while a point at $$x = 2$$, halfway, has a responsibility of 0.5 for each. In the M-step, the point at 3 thus counts for 0.982 of a point in the mean of the second component, and for 0.018 in the first, where k-means would assign it fully to the second cluster.

Each iteration never decreases the log-likelihood, and the algorithm converges to a local maximum, so, as with k-means, it is usually run from several initializations, or initialized with the result of k-means. In fact, k-means is a limit case of EM: for a GMM whose components share the covariance $$\sigma^2 I$$, the responsibilities become hard assignments to the closest mean as $$\sigma \rightarrow 0$$. The number of components $$K$$ can be chosen with a model selection criterion, such as the [Bayesian information criterion](https://en.wikipedia.org/wiki/Bayesian_information_criterion). In practice, k-means is cheaper per iteration and is preferred for large datasets with compact clusters, while a GMM suits elliptical or overlapping clusters. The post on [variational inference]({{ site.baseurl }}{% post_url 2019-07-01-variational-inference-gmm %}) fits a Bayesian GMM, with a prior over its means, by variational inference instead of EM.

### Self-Organizing Maps and Kohonen Maps

A **self-organizing map** (SOM) is an unsupervised artificial neural network, typically a 2D grid of neurons, that represents a discretization of the input space (or of the training set). It was introduced by Teuvo Kohonen in the paper [Self-organized formation of topologically correct feature maps](https://link.springer.com/article/10.1007/BF00337288) (1982), hence its other name, the Kohonen map. Its main purpose is to represent the training data by localized regions (neurons) that represent agglomerates of inputs, with neighboring neurons of the grid representing nearby regions of the input space. Like k-means, a SOM is a prototype method, whose prototypes are the weights of its neurons; unlike k-means, its prototypes are ordered on a grid. The following picture illustrates Kohonen's model (top) and one step of its learning algorithm (bottom):

{: .picture}
<img width="46%" height="46%" src="{{ site.assets }}/unsupervised-learning/kohonen.svg"><br>
<img width="37%" height="37%" src="{{ site.assets }}/unsupervised-learning/kohonen2.svg">

{: .caption}
**Top:** the neuron layer of Kohonen's model. The input fibres running horizontally carry the input signal (the stimulus) and excite the neurons of the layer through synaptic connections. Lateral interactions between the neurons confine the response to a spatially bounded local excitation zone. The layer acts as a topographic feature map if the position $$s$$ of the excitation zone varies continuously with the stimulus features of interest. **Bottom:** the adaptation step. The input $$v$$ in the input space $$V$$ selects the winner neuron $$s$$ in the lattice of neurons $$A$$ ($$\phi$$ is the map from $$V$$ to $$A$$). The winner and its neighbours shift their weight vectors towards $$v$$, by an amount that decreases with the distance from $$s$$, shown by the sizes and gray levels of the neurons. Only the shift $$\Delta w_s$$ of the winner's weights $$w_s$$ is drawn. In the algorithm below, $$v$$ is the input $$x_u$$ and $$s$$ is the winner $$j^*$$. Adapted from: [Neural Computation and Self-Organizing Maps, H. Ritter, T. Martinetz and K. Schulten, 1992, Figures 4.1 and 4.2](https://tcbg.illinois.edu/Services/Class/PHYS498TBP/spring2002/neuro_4.pdf)


The algorithm is the following:
- neurons are randomly placed in the multidimensional space;
- at every iteration, choose an input $$x_u$$;
- determine which neuron $$j^*$$ is the *winner* (or *best matching unit*, BMU), i.e. the closest in space:

$$
\| w_{j^*} - x_u \| \leq \| w_j - x_u \| \text{, for all } j
$$

- update the weights of the winner and of its neighbors, by pulling them closer to the input vector:

$$
\Delta w_j = \eta \, h(j, j^*) \, (x_u - w_j)
$$

where $$h(j, j^*)$$ is the *neighborhood function*, which decreases with the distance between neurons $$j$$ and $$j^*$$ in the grid (e.g. a Gaussian). If $$h$$ is $$1$$ for the winner and $$0$$ for all other neurons, the update above is the online update rule of k-means. It is also common to decrease $$\eta$$ and the width of $$h$$ as the iterations progress;
- repeat until stabilization: either when reaching the maximum number of iterations, or when the change in weights falls below a given threshold.

Very nice videos demonstrating an iterative execution of the Kohonen map can be found [in this link](https://www.youtube.com/watch?v=IixbH1gDhsg) and [this link](https://www.youtube.com/watch?v=zeOtwCAI3Fs).

A good example is the recognition of handwritten digits from the [MNIST database](http://yann.lecun.com/exdb/mnist/).
Each digit is a 28x28 (=784) pixel array, represented as a data point in a 784-dimensional space. Each pixel is monochromatic, so each dimension takes values between 0 and 255.

The task is, given a database of 5,000 handwritten digits, to find the 6x6 SOM that best represents 4 of the digits from 0 to 9. This example studies how the neighborhood width and the learning rate affect the performance of the Kohonen map.

The input dataset and the `python` implementation are available in <a href="{{ site.assets }}/unsupervised-learning/kohonen_mnist.zip">kohonen_mnist.zip</a>. The goal is to fine-tune the neighborhood width and the learning rate $$\eta$$. A good solution gives a visual perception of the digits being identified (in this case 0, 1, 2 and 3), as the weights of each neuron form a 28x28 image. Three possible outputs are displayed in the pictures below:
- left: a nicely tuned SOM, where each quadrant of neurons represents a digit;
- center: a SOM with a high learning rate. The large step size makes neurons *step* over long distances during training, leading to a mixing of digits on nearby neurons, and possibly a *tangled* SOM;
- right: a SOM with a large neighborhood width. The large width causes all neurons to behave almost synchronously, due to the very wide range of the update function. This leads to very similar outputs per neuron.

{: .picture}
<img width="27.1%" height="27.1%" src="{{ site.assets }}/unsupervised-learning/kohonen_1.svg">
<span style="display:inline-block; width:1cm;"></span>
<img width="27.2%" height="27.2%" src="{{ site.assets }}/unsupervised-learning/kohonen_2.svg">
<span style="display:inline-block; width:1cm;"></span>
<img width="26.7%" height="26.7%" src="{{ site.assets }}/unsupervised-learning/kohonen_3.svg">
 
 
More examples can be found in the [Wikipedia entry for self-organizing maps](https://en.wikipedia.org/wiki/Self-organizing_map#Examples).

### Comparison of the Clustering Methods

The following table sums up the clustering methods of this section:

| Method | A cluster is | Number of clusters | Cost | When to use it |
|---|---|---|---|---|
| k-means | The points closest to a mean | Given, $$k$$ | Low: the distances from each point to the $$k$$ means, at each iteration | Large datasets with compact, spherical clusters |
| Hierarchical | A branch of the dendrogram | Chosen after the fact, by cutting the tree | The $$n \times n$$ distances, and $$O(n^2)$$ to $$O(n^3)$$ time | Small to medium datasets, nested clusters |
| DBSCAN | A connected group of core points, plus their border points | Found by the method | The neighbors of each point, fast with a spatial index in low dimensions | Arbitrary shapes and outliers, with similar densities |
| Spectral | A group of strongly connected points of the similarity graph | Given, $$k$$ | An eigendecomposition of an $$n \times n$$ sparse matrix | Non-convex clusters, e.g. two concentric rings |
| Gaussian mixture | A Gaussian of the mixture, with soft assignments | Given, $$K$$, or chosen with a criterion such as the BIC | Higher than k-means, with a covariance matrix per component | Elliptical or overlapping clusters |
| Self-organizing map | The inputs closest to a neuron of the grid | The neurons of the grid | The distances from each input to all neurons | A 2D map of the data, where nearby neurons represent nearby inputs |

All these methods cluster the data in the features they are given. When the data has many features, the methods of the next section reduce it to fewer dimensions first.

## Dimensionality Reduction

Data often has many more dimensions than its structure needs. A 28x28 image of a handwritten digit, as in the example of self-organizing maps above, is a point in a space of 784 dimensions. Yet the images of digits fill only a small part of that space. On the 5,000 digits of that example, 118 of the 784 pixels are 0 in every image, and the first 50 principal components (defined next) explain 83% of the variance.

**[Dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction)** maps the data to fewer dimensions that keep most of its structure, to compress it, to visualize it in 2D or 3D, or to prepare it for other methods, such as clustering. The methods of this section differ in what they keep: PCA and kernel PCA keep the directions of largest variance, t-SNE and UMAP the neighborhoods of the points, and ICA the directions of independent signals.

### Principal Component Analysis (PCA) and Kernel PCA

**[Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis)** (PCA) is a dimensionality (feature) reduction method that rotates the data towards the directions of largest variance, so that one can analyze the data along its most meaningful dimensions and discard the dimensions of low variance. In the words of an earlier version of the Wikipedia article on PCA:

<blockquote class="blockquote text-right">
Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components.
</blockquote>

For $$n$$ data points $$x_1, ..., x_n$$, the algorithm is the following:
- center the data by subtracting the mean: $$x_i \leftarrow x_i - \bar{x}$$, with $$\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$$;
- compute the [covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix) of the centered data: $$ C = \frac{1}{n-1} \sum_{i=1}^n x_i \, x_i^{\intercal}  $$;
- compute the [eigenvalues and eigenvectors](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors) of the covariance matrix:
  - solve $$\det(C - \lambda I)=0$$, where $$\lambda$$ are the eigenvalues;
  - compute the eigenvector $$v_i$$ of each eigenvalue $$\lambda_i$$ by solving $$C v_i = \lambda_i v_i \, \Leftrightarrow \, (C- \lambda_i I) v_i = 0$$;
- build the feature matrix $$F$$, whose columns are the eigenvectors in the order of decreasing eigenvalues (each eigenvalue is the variance of the data along its eigenvector);
- the transformed data is obtained by multiplying the transposed feature matrix ($$F^T$$, i.e. with the most significant eigenvector on top) by the matrix $$D_c$$ whose columns are the centered data points: $$D_n = F^T D_c$$.

A visual example is displayed below, where PCA rotates the data towards the new axes of the cluster of data points:

{: .picture}
<img width="26.7%" height="26.7%" src="{{ site.assets }}/unsupervised-learning/gaussianscatterpca.svg">

{: .caption}
PCA of a 2D Gaussian centred at $$(1, 3)$$, with a standard deviation of 3 roughly along the direction $$(0.866, 0.5)$$ and of 1 in the orthogonal direction. The arrows are the eigenvectors of the covariance matrix, scaled by the square roots of their eigenvalues, with their tails at the mean. Adapted from: [Principal component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis)

The least significant dimensions of the transformed data can now be discarded, reducing the number of features of the model. The number of eigenvectors kept in $$F$$ dictates the amount of compression achieved. The first $$k$$ components explain a fraction $$\sum_{i=1}^k \lambda_i / \sum_{i} \lambda_i$$ of the variance of the data, a common criterion to choose $$k$$.

For the Gaussian of the picture, with standard deviations of 3 and 1, the eigenvalues of the covariance matrix are $$3^2 = 9$$ and $$1^2 = 1$$, so the first principal component explains a fraction $$9 / (9 + 1)$$ of the variance, i.e. 90%. Keeping only this component describes each point by one number instead of two, and loses the 10% of the variance across it.

To recover the original data (or an approximation, if $$F$$ does not include all eigenvectors), the transformation is inverted, i.e. the transformed data is multiplied by the inverse of $$F^T$$. This gives the centered data $$D_c$$, to which the mean $$\bar{x}$$ is added back, column by column. The eigenvectors of the symmetric matrix $$C$$ are orthonormal, so $$F^TF = I$$. When $$F$$ keeps all eigenvectors, it is square and orthogonal, so $$F^{-T} = F$$ and:

$$
D_c = F^{-T} D_n = F D_n
$$

When $$F$$ keeps only $$k$$ eigenvectors, $$F D_n$$ is the projection of the data onto the first $$k$$ principal components, the best approximation of $$D_c$$ by $$k$$ dimensions in the least-squares sense.

In practice, PCA is usually computed with the [singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) of the centered data instead: if $$D_c^T = U \Sigma V^T$$, the columns of $$V$$ are the eigenvectors of $$C$$, and its eigenvalues are $$\lambda_i = \sigma_i^2 / (n-1)$$. This avoids computing $$C$$, whose [condition number](https://en.wikipedia.org/wiki/Condition_number) is the square of that of $$D_c$$, and is numerically more stable.

PCA only finds linear structure. **[Kernel PCA](https://en.wikipedia.org/wiki/Kernel_principal_component_analysis)** ([Schölkopf et al., 1998](https://doi.org/10.1162/089976698300017467)) applies PCA in a high-dimensional feature space $$\phi(x)$$, where non-linear structure of the input space may become linear. The [*kernel trick*](https://en.wikipedia.org/wiki/Kernel_method) avoids computing $$\phi$$, as the method only needs the dot products $$k(x_i, x_j) = \phi(x_i)^T \phi(x_j)$$, given by a kernel function such as the Gaussian kernel $$k(x_i, x_j) = \exp \left( - \| x_i - x_j \|^2 / 2 \sigma^2 \right)$$ or the polynomial kernel $$k(x_i, x_j) = (x_i^T x_j + c)^d$$. The algorithm is:
- compute the $$n \times n$$ kernel matrix $$K$$, with $$K_{ij} = k(x_i, x_j)$$;
- center the data in the feature space: $$\tilde{K} = K - 1_n K - K 1_n + 1_n K 1_n$$, where $$1_n$$ is the $$n \times n$$ matrix with all entries equal to $$1/n$$;
- compute the eigenvalues $$\lambda_m$$ and eigenvectors $$\alpha^{(m)}$$ of $$\tilde{K}$$, and scale each eigenvector so that $$\lambda_m \, \alpha^{(m)T} \alpha^{(m)} = 1$$, which gives the corresponding direction in the feature space a unit norm;
- the $$m$$-th principal component of a data point $$x_j$$ is $$\sum_{i=1}^n \alpha_i^{(m)} \tilde{K}_{ij}$$, i.e. the $$j$$-th entry of $$\tilde{K} \alpha^{(m)} = \lambda_m \alpha^{(m)}$$.

The eigendecomposition of an $$n \times n$$ matrix limits kernel PCA to datasets of moderate size, and mapping a point back to the input space (the *pre-image* problem) is only approximate. For the sake of comparison, the cost of PCA grows only linearly with the number of data points, as the size of its covariance matrix depends only on the number of features.

### t-SNE (t-Distributed Stochastic Neighbor Embedding)

PCA preserves the global, linear structure of the data, which is often not what one wants to see when visualizing high-dimensional data in 2D or 3D. *Neighbor embedding* methods instead place similar points close together, preserving local neighborhoods.

**[t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)** ([van der Maaten and Hinton, 2008](https://www.jmlr.org/papers/v9/vandermaaten08a.html)) turns distances into probabilities. In the input space, the similarity of point $$x_j$$ to point $$x_i$$ is:

$$
p_{j \mid i} = \frac{\exp \left( - \| x_i - x_j \|^2 / 2 \sigma_i^2 \right)}{\sum_{k \neq i} \exp \left( - \| x_i - x_k \|^2 / 2 \sigma_i^2 \right)}
$$

where each $$\sigma_i$$ is set so that the distribution $$p_{\cdot \mid i}$$ has a given *[perplexity](https://en.wikipedia.org/wiki/Perplexity)* (roughly, the number of effective neighbors, typically 5 to 50), and the symmetric similarity of the pair is $$p_{ij} = (p_{j \mid i} + p_{i \mid j}) / 2n$$. In the low-dimensional embedding $$y_1, ..., y_n$$, similarities follow a [Student t-distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution) with one degree of freedom:

$$
q_{ij} = \frac{\left(1 + \| y_i - y_j \|^2 \right)^{-1}}{\sum_{k \neq l} \left( 1 + \| y_k - y_l \|^2 \right)^{-1}}
$$

and the embedding minimizes the [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) $$KL(P \| Q) = \sum_{i \neq j} p_{ij} \log (p_{ij} / q_{ij})$$ by gradient descent, with:

$$
\frac{\partial KL}{\partial y_i} = 4 \sum_{j} (p_{ij} - q_{ij}) (y_i - y_j) \left( 1 + \| y_i - y_j \|^2 \right)^{-1}
$$

The heavy tails of the t-distribution let moderately distant points move far apart in the embedding. This avoids the *crowding problem* that a Gaussian would have in low dimensions, and produces well-separated clusters. Two caveats: the sizes of the clusters and the distances between them in a t-SNE plot carry little meaning, and the result depends on the perplexity and the random initialization. Each iteration costs $$O(n^2)$$, or $$O(n \log n)$$ with the [Barnes-Hut approximation](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation).

**[UMAP](https://umap-learn.readthedocs.io/)** (Uniform Manifold Approximation and Projection, [McInnes et al., 2018](https://arxiv.org/abs/1802.03426)) follows the same idea with a different construction. It builds a weighted graph of the $$k$$ nearest neighbors of each point, with weights:

$$
v_{j \mid i} = \exp \left( - \frac{d(x_i, x_j) - \rho_i}{\sigma_i} \right)
$$

where $$\rho_i$$ is the distance from $$x_i$$ to its nearest neighbor, and $$\sigma_i$$ is set so that $$\sum_j v_{j \mid i} = \log_2 k$$. The weights are made symmetric as $$v_{ij} = v_{j \mid i} + v_{i \mid j} - v_{j \mid i} \, v_{i \mid j}$$. In the embedding, the similarities are $$w_{ij} = \left( 1 + a \| y_i - y_j \|^{2b} \right)^{-1}$$, with $$a$$ and $$b$$ set by the minimum distance between embedded points, and the embedding minimizes the [cross-entropy](https://en.wikipedia.org/wiki/Cross-entropy) between the two graphs:

$$
\sum_{i \neq j} v_{ij} \log \frac{v_{ij}}{w_{ij}} + (1 - v_{ij}) \log \frac{1 - v_{ij}}{1 - w_{ij}}
$$

with stochastic gradient descent and negative sampling. The second term pushes apart the points that are not neighbors, and UMAP tends to preserve more of the global structure than t-SNE, although mostly due to its spectral initialization ([Kobak and Linderman, 2021](https://www.nature.com/articles/s41587-020-00809-z)). It is also faster, scales to millions of points and can embed new points, so it is used as a general dimensionality reduction method too, and not only for visualization.

### Independent Component Analysis (ICA)

This section was largely inspired by the lecture notes of Tatsuya Yokota (Tokyo Institute of Technology, 2012).

Closely related to PCA is **[Independent Component Analysis](https://en.wikipedia.org/wiki/Independent_component_analysis)** (ICA), a method for [blind source separation](https://en.wikipedia.org/wiki/Blind_signal_separation): it separates a multidimensional signal into additive subcomponents, under the assumptions that:
- the source signals $$s_1$$, ..., $$s_n$$ are statistically independent, i.e. $$p(s_1, s_2, ..., s_n) = p(s_1)\,\,p(s_2) \,\, ... \,\, p(s_n)$$;
- the sources are not Gaussian (at most one of them may be), otherwise ICA is impossible.

ICA builds on the [Central Limit Theorem](https://en.wikipedia.org/wiki/Central_limit_theorem), which states that the distribution of the sum of $$N$$ independent random variables approaches a Gaussian as $$N \rightarrow \infty$$. As an example, the throw of a die has a uniform distribution from 1 to 6, but the distribution of the sum of several throws is better approximated by a Gaussian as the number of throws increases. A mixture of independent signals is thus *more Gaussian* than the signals themselves, and the most non-Gaussian directions recover the independent signals. In short, *non-Gaussianity is independence*.

A famous example is the [cocktail party](https://en.wikipedia.org/wiki/Cocktail_party_effect) problem. Three guests speak at the same time, and three microphones at different places of the room record the observed signals $$x$$, each a different mixture of the original signals $$s$$, the three voices, as in:

$$
\begin{align*}
x_1(t) = a_{11}s_1(t) + a_{12}s_2(t) + a_{13}s_3(t)\\
x_2(t) = a_{21}s_1(t) + a_{22}s_2(t) + a_{23}s_3(t)\\
x_3(t) = a_{31}s_1(t) + a_{32}s_2(t) + a_{33}s_3(t)
\end{align*}
$$

Assume that $$s_1$$, $$s_2$$ and $$s_3$$ are statistically independent. The goal of ICA is to estimate the independent components $$s(t)$$ from $$x(t)$$, with

$$
x(t) = As(t)
$$

where $$A$$ is an invertible mixing matrix. Thus, the problem can be written as $$s(t) = Bx(t)$$, where $$B = A^{-1}$$. It is only necessary to estimate $$B$$ so that the $$s_i$$ are independent.

Signals $$z$$ are *white* if they satisfy $$E [z] = 0$$ and $$E [ z z^T ] = I$$. [Whitening](https://en.wikipedia.org/wiki/Whitening_transformation) is a useful pre-processing step for ICA. The whitened signals are $$z (t) = V x(t)$$, where $$x$$ is centered and $$V$$ is the whitening matrix (e.g. $$V = \Lambda^{-1/2} F^T$$, with the eigenvalues $$\Lambda$$ and the eigenvectors $$F$$ of the covariance matrix of $$x$$, as in PCA). The model becomes:

$$
s(t) = U z (t) = U V x (t) = B x (t)
$$

where $$U$$ is an orthogonal matrix. **Whitening simplifies ICA**, as it is then only necessary to estimate $$U$$: the whitened signals are already uncorrelated, like the principal components of PCA, and $$U$$ is the rotation that makes them independent. An illustration of the source, observed and whitened data is displayed below, where the mixing turns the square of the sources into a parallelogram, and whitening turns it back into a square, rotated:

{: .picture}
<img width="19.2%" height="19.2%" src="{{ site.assets }}/unsupervised-learning/ica_source.svg">
<img width="19.2%" height="19.2%" src="{{ site.assets }}/unsupervised-learning/ica_observed.svg"> 
<img width="18.2%" height="18.2%" src="{{ site.assets }}/unsupervised-learning/ica_whitening.svg">

{: .caption}
Independent component analysis of two mixed signals: (a) the sources $$s_1$$, $$s_2$$, independent and uniformly distributed; (b) the observed signals $$x_1$$, $$x_2$$, mixed as $$x = As$$; (c) the whitened signals $$z_1$$, $$z_2$$, $$z = Vx$$, with zero mean and identity covariance, from which only the orthogonal matrix $$U$$ remains to be found. Adapted from: [T. Yokota, Independent Component Analysis for Blind Source Separation, Tokyo Institute of Technology, 2012](https://www.slideshare.net/slideshow/independent-component-analysis-11359849/11359849)

**Non-Gaussianity is a measure of independence**. [Kurtosis](https://en.wikipedia.org/wiki/Kurtosis) is a measure of non-Gaussianity, defined for a variable $$y$$ of zero mean by:

$$
\mathrm{kurt}(y) =  E [y^4] - 3 (E[y^2])^2.
$$

It is zero for a Gaussian. Assume that $$y$$ is white, i.e. $$E[y]=0$$ and $$E[y^2]=1$$, thus

$$
\mathrm{kurt}(y) = E[y^4]-3.
$$

As an example, the sources of the picture are uniform. A uniform variable of unit variance has $$E[y^4] = 9/5$$, i.e. a kurtosis of −1.2, while a variable with a [Laplace distribution](https://en.wikipedia.org/wiki/Laplace_distribution) of unit variance has $$E[y^4] = 6$$, i.e. a kurtosis of +3. Variables of negative kurtosis, flatter than a Gaussian, are called *sub-Gaussian*, and those of positive kurtosis, with a sharper peak and heavier tails, *super-Gaussian*. The kurtosis of a sum of independent variables is the sum of their kurtoses, and a factor $$1/\sqrt{2}$$ divides it by 4. Thus, the mixture $$(s_1 + s_2)/\sqrt{2}$$ of two uniform sources of unit variance has a kurtosis of −0.6, halfway to the 0 of a Gaussian. Mixing makes the signals more Gaussian, and maximizing the absolute value of the kurtosis undoes the mixing.

Each row $$b^T$$ of $$B$$ recovers one source, $$s_i = b^T x$$. The ICA problem is solved with

$$
\hat{b} = \arg\max_b | \mathrm{kurt}(b^T x(t)) |
$$

where $$b^T x(t)$$ is constrained to unit variance. On the whitened data, $$b^T x(t) = w^T z(t)$$, where $$w^T = b^T V^{-1}$$ is a row of $$U$$, and the unit-variance constraint becomes $$w^T w=1$$. The absolute value of the *kurtosis* $$\mid \mathrm{kurt}(w^Tz) \mid$$ is then maximized, subject to $$w^T w=1$$. Its gradient is:

$$
\begin{align*}
\frac{\partial \vert \mathrm{kurt}(w^T z) \vert}{\partial w} & = \frac{\partial}{\partial w} \left\vert E[(w^T z)^4] - 3 (E[(w^T z)^2])^2 \right\vert \\
& = \frac{\partial}{\partial w} \left\vert E[(w^T z)^4] - 3 (\| w \|^2)^2 \right\vert \quad \text{(because } E[z z^T] = I) \\
& = 4 \, \text{sign}[\mathrm{kurt}(w^T z)] \left( E[z (w^T z)^3] - 3 w \| w \|^2 \right)
\end{align*}
$$

The problem can now be solved by gradient ascent (with $$\| w \| = 1$$):
- $$\Delta w \propto \text{sign} [ \mathrm{kurt}(w^T z)] \, ( E [ z(w^T z)^3 ] - 3w )$$;
- $$w \leftarrow w + \Delta w$$, followed by the normalization $$w \leftarrow w / \| w \|$$.

Or, because the algorithm converges when $$w \propto \Delta w$$, and $$w$$ and $$-w$$ are equivalent, with the fixed-point iteration of **[FastICA](https://en.wikipedia.org/wiki/FastICA)** ([Hyvärinen and Oja, 1997](https://doi.org/10.1162/neco.1997.9.7.1483)):
- $$w \leftarrow E [ z ( w^T z)^3 ] - 3w$$, followed by $$w \leftarrow w / \| w \|$$.

Unlike gradient ascent, FastICA has no learning rate to tune, and its convergence is cubic, so it is usually much faster. The other components are found in the same way, keeping each new $$w$$ orthogonal to the previous ones, as the rows of $$U$$ are orthonormal.

As a final remark, *kurtosis is very sensitive to outliers, because it is a fourth-order function*. A more robust alternative is the [negentropy](https://en.wikipedia.org/wiki/Negentropy). It is omitted here for brevity, but the details are in the [original lecture notes](https://www.slideshare.net/yokotatsuya/independent-component-analysis-11359849).

To find more about the topic, the book [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) by Christopher Bishop covers k-means, Gaussian mixtures and the EM algorithm (chapter 9), and PCA, kernel PCA and ICA (chapter 12).

## Association Rule Mining

**[Association rule learning](https://en.wikipedia.org/wiki/Association_rule_learning)** finds the items that often appear together in a database of transactions. It was introduced by [Agrawal, Imieliński and Swami (1993)](https://doi.org/10.1145/170035.170072) for *market basket analysis*, i.e. finding the products that the customers of a supermarket buy together, and it is also used in web usage mining, recommendation systems and bioinformatics. A rule $$X \Rightarrow Y$$, e.g. $$\{\text{butter}\} \Rightarrow \{\text{eggs}\}$$, says that the transactions that hold the itemset $$X$$ often hold the itemset $$Y$$ too, with $$X \cap Y = \emptyset$$. For a set of transactions $$T$$, the quality of a rule is measured by:
- the **support** of an itemset, the fraction of the transactions that contain it: $$\text{supp}(X) = \vert \{ t \in T : X \subseteq t \} \vert / \vert T \vert$$;
- the **confidence** of a rule, the fraction of the transactions with $$X$$ that also contain $$Y$$: $$\text{conf}(X \Rightarrow Y) = \text{supp}(X \cup Y) / \text{supp}(X)$$;
- the **lift** of a rule, its confidence divided by the support of $$Y$$: $$\text{lift}(X \Rightarrow Y) = \text{conf}(X \Rightarrow Y) / \text{supp}(Y)$$. A lift above 1 means that $$X$$ and $$Y$$ appear together more often than if they were independent.

The rules are mined in two steps. First, find all the *frequent itemsets*, whose support is at least a minimum support. Then, split each frequent itemset into rules whose confidence is at least a minimum confidence. The second step is cheap, as all the supports it needs were computed in the first one. The first step is the hard one: $$m$$ items form $$2^m - 1$$ itemsets, too many to count one by one. The algorithms of this section differ in how they search this space.

### Apriori Algorithm

The **[Apriori algorithm](https://en.wikipedia.org/wiki/Apriori_algorithm)** ([Agrawal and Srikant, 1994](https://www.vldb.org/conf/1994/P487.PDF)) is based on the *Apriori property*: every subset of a frequent itemset is frequent too, or, equivalently, an itemset that has an infrequent subset cannot be frequent. The algorithm searches the itemsets breadth-first, by size:
- count the support of each item, and keep the frequent ones, the itemsets $$L_1$$;
- at step $$k$$, build the candidate itemsets $$C_k$$ of size $$k$$ by joining the pairs of itemsets of $$L_{k-1}$$ that differ in one item, and remove (*prune*) every candidate that has a subset of size $$k-1$$ not in $$L_{k-1}$$;
- scan the database to count the support of each candidate, and keep the frequent ones, $$L_k$$;
- stop when $$L_k$$ is empty.

As an example, take five transactions: {bread, milk}, {bread, butter, eggs}, {milk, butter, eggs, cola}, {bread, milk, butter, eggs} and {bread, milk, butter, cola}, with a minimum support of 3 transactions (60%). Bread, milk and butter appear in 4 transactions, eggs in 3 and cola in 2, so $$L_1$$ holds all items but cola. Among the 6 pairs of the 4 frequent items, {bread, milk}, {bread, butter}, {milk, butter} and {butter, eggs} appear 3 times, and {bread, eggs} and {milk, eggs} only twice. Of the 3 candidates of size 3, {bread, butter, eggs} and {milk, butter, eggs} are pruned without a scan, as {bread, eggs} and {milk, eggs} are not frequent, and {bread, milk, butter} appears in only 2 transactions, so the search stops. The rule $$\{\text{eggs}\} \Rightarrow \{\text{butter}\}$$ has a confidence of 3/3 = 100%, as every transaction with eggs has butter, and a lift of 1/0.8 = 1.25. The rule $$\{\text{bread}\} \Rightarrow \{\text{milk}\}$$ has a confidence of 3/4 = 75%, but a lift of 0.75/0.8 = 0.94: milk is in 80% of the transactions anyway, and bread makes it slightly less likely.

Apriori scans the database once per itemset size, and the number of candidates grows very large when the minimum support is low: $$10^4$$ frequent items give about $$5 \times 10^7$$ candidate pairs.

### ECLAT and FP-growth

**ECLAT** (Equivalence Class Transformation, [Zaki, 2000](https://doi.org/10.1109/69.846291)) changes the layout of the database. Instead of a list of items per transaction (the *horizontal* layout), it stores, for each item, the set of the transactions that hold it, its *tidset* (the *vertical* layout). The support of an itemset is then the size of the intersection of the tidsets of its items, with no scan of the database. In the example above, the tidsets of butter and eggs are {2, 3, 4, 5} and {2, 3, 4}, and their intersection {2, 3, 4} gives the support 3 of {butter, eggs}. ECLAT searches the itemsets depth-first, extending each itemset with one item at a time and intersecting their tidsets, which is fast as long as the tidsets fit in memory.

**FP-growth** (Frequent Pattern growth, [Han, Pei and Yin, 2000](https://doi.org/10.1145/342009.335372)) generates no candidates at all. It scans the database twice: first, to count the support of each item; then, to insert each transaction, with its frequent items sorted by decreasing support, into a prefix tree, the *FP-tree*. Transactions that share a prefix share a path in the tree, and each node counts the transactions that pass through it, so the tree is a compressed copy of the database. Then, for each item, from the least frequent one, FP-growth collects the paths of the tree that end at that item (its *conditional pattern base*), builds a smaller FP-tree from them, and mines it recursively. In the example above, with the order bread, milk, butter, eggs, the 15 frequent items of the five transactions fit in an FP-tree of 9 nodes. The conditional pattern base of eggs is {bread, milk, butter}, {bread, butter} and {milk, butter}, one transaction each, in which only butter reaches the minimum support of 3, which gives the frequent itemset {butter, eggs}.

In practice, FP-growth is usually faster than Apriori, as it scans the database only twice and generates no candidates, while ECLAT is fast on dense datasets whose tidsets fit in memory. Apriori is the simplest of the three, and suits sparse data with a high minimum support.

## Anomaly and Outlier Detection

**[Anomaly detection](https://en.wikipedia.org/wiki/Anomaly_detection)** (or outlier detection) finds the data points that differ much from the rest of the data, such as fraudulent transactions, failing machines or network intrusions. Some methods of this post already detect outliers: DBSCAN labels them as noise, a GMM gives them a low likelihood $$p(x)$$, and a point far from its $$k$$-th nearest neighbor is likely an outlier. The two methods of this section are dedicated to the task, and give each point an *anomaly score*.

### Isolation Forest

An **[isolation forest](https://en.wikipedia.org/wiki/Isolation_forest)** ([Liu, Ting and Zhou, 2008](https://doi.org/10.1109/ICDM.2008.17)) is based on the observation that anomalies are *few and different*, so they are easier to isolate than normal points. An *isolation tree* splits the data recursively: each node picks a random feature and a random split value between the minimum and the maximum of that feature in the node, until each point is alone in a leaf, or a maximum depth is reached. An anomaly, far from the other points, is isolated by few splits, close to the root, while a point in a dense region needs many. The anomaly score of a point $$x$$ is computed from its path length $$h(x)$$, the number of edges from the root to its leaf, averaged over the trees of the forest:

$$
s(x, m) = 2^{- \mathbb{E}[h(x)] / c(m)}
$$

where $$m$$ is the number of points used to build each tree, and $$c(m) = 2 H(m-1) - 2(m-1)/m$$ is the average path length of an unsuccessful search in a [binary search tree](https://en.wikipedia.org/wiki/Binary_search_tree) of $$m$$ points, which normalizes $$h(x)$$, with the [harmonic number](https://en.wikipedia.org/wiki/Harmonic_number) $$H(i) \approx \ln(i) + 0.5772$$. A score close to 1 marks an anomaly, a score well below 0.5 a normal point, and when all points have a score close to 0.5, the data has no distinct anomalies. As an example, for trees of $$m = 256$$ points, $$c(256) = 10.24$$: a point isolated after 2 splits on average has a score of $$2^{-2/10.24} = 0.87$$, and a point isolated after 15 splits a score of 0.36.

Each tree is built on a small random subsample of the data, e.g. the 256 points and 100 trees of the paper, and its depth can be limited to about $$\log_2 m$$, as only the short paths matter. Thus, an isolation forest computes no distances, its cost grows linearly with the number of points, and it scales to large and high-dimensional datasets.

### Local Outlier Factor (LOF)

The **[local outlier factor](https://en.wikipedia.org/wiki/Local_outlier_factor)** (LOF, [Breunig et al., 2000](https://doi.org/10.1145/342009.335388)) compares the density around a point with the density around its neighbors. A point in a sparse region is not an outlier when its neighbors lie in a sparse region too, so LOF finds outliers relative to their local neighborhood, also in data with clusters of different densities, where a single distance threshold fails. With the $$k$$ nearest neighbors $$N_k(p)$$ of a point $$p$$, and the distance $$d_k(o)$$ from a point $$o$$ to its $$k$$-th nearest neighbor:
- the *reachability distance* of $$p$$ from $$o$$ is $$\text{reach}_k(p, o) = \max \{ d_k(o), d(p, o) \}$$, i.e. the distance between them, but at least $$d_k(o)$$, which smooths the distances within a dense region;
- the *local reachability density* of $$p$$ is the inverse of its average reachability distance from its neighbors: $$\text{lrd}_k(p) = \left( \frac{1}{k} \sum_{o \in N_k(p)} \text{reach}_k(p, o) \right)^{-1}$$;
- the LOF of $$p$$ is the average ratio between the densities of its neighbors and its own: $$\text{LOF}_k(p) = \frac{1}{k} \sum_{o \in N_k(p)} \text{lrd}_k(o) / \text{lrd}_k(p)$$.

A LOF close to 1 means that the point is as dense as its neighbors, and a LOF much larger than 1 that it is an outlier. As an example, take the points 0, 1, 2, 3 and 10 on a line, with $$k = 2$$. The points 0 to 3 each have an average reachability distance of 1.5, i.e. a local reachability density of 2/3, and a LOF of 1. The point 10 has the neighbors 3 and 2, at the reachability distances 7 and 8, so its density is 1/7.5, and its LOF is (2/3) / (1/7.5) = 5.

LOF needs the $$k$$ nearest neighbors of each point, i.e. $$O(n^2)$$ distances with a brute-force search, and its scores are relative, so the threshold that separates the outliers depends on the data. For the sake of comparison, an isolation forest is faster and scales to large datasets, while LOF finds the local outliers next to a dense cluster that a global method may miss.

## The Curse of Dimensionality and Nearest-Neighbor Search

Distance-based methods, such as k-means, KNN, DBSCAN and LOF, struggle with high-dimensional data, in particular with Euclidean distances. This is due to a phenomenon called the **[curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)**: as the number of dimensions increases, the distances between any two points become similar, which makes the distance metric almost irrelevant. For example, for points with independent and identically distributed coordinates, the relative difference between the distances from a point to its farthest and nearest neighbors, $$(d_{max} - d_{min}) / d_{min}$$, tends to 0 as the number of dimensions grows ([Beyer et al., 1999](https://link.springer.com/chapter/10.1007/3-540-49257-7_15)).

For 1,000 points drawn uniformly in the unit cube, this relative difference has a median of 67 in 2 dimensions, 2.8 in 10, 0.43 in 100 and 0.12 in 1,000, over 100 random query points. The reason is that the squared distance between two such points is a sum of one independent term per dimension, the squared difference of two uniform coordinates, of mean 1/6 and variance 7/180. Its mean grows with the number of dimensions, while its standard deviation grows only with the square root of that number. Thus, the standard deviation relative to the mean is $$\sqrt{7/5}$$ divided by the square root of the number of dimensions: 84% in 2 dimensions, 12% in 100 and 3.7% in 1,000.

The general proof of the claim is not detailed here, but this [blog post](https://medium.com/towards-data-science/that-cursing-dimensionality-ac317fb0fdcc) covers the curse of dimensionality in more detail. This phenomenon is one more reason to reduce the dimensionality of the data, e.g. with PCA, before clustering it.

The curse of dimensionality also hits the search for nearest neighbors, the core operation of KNN, DBSCAN, LOF, t-SNE and UMAP. In low dimensions, [spatial databases](https://en.wikipedia.org/wiki/Spatial_database) answer such queries with spatial indexes, such as the k-d tree, which splits the space with one axis-aligned plane per node, and the [R-tree](https://en.wikipedia.org/wiki/R-tree) ([Guttman, 1984](https://doi.org/10.1145/602259.602266)), which groups nearby objects in nested bounding rectangles. A query then visits only the few nodes whose region may hold a neighbor, and skips the rest of the data. In high dimensions, this no longer works: when all distances are similar, the ball around a query overlaps almost every node, and [Weber, Schek and Blott (1998)](https://www.vldb.org/conf/1998/p194.pdf) showed that tree indexes then do worse than a sequential scan of the data, from about 10 dimensions.

This matters today, as [retrieval-augmented generation](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) (RAG, [Lewis et al., 2020](https://arxiv.org/abs/2005.11401)) is built on the nearest-neighbor search in high dimensions. A RAG system splits a collection of documents into chunks, maps each chunk to an *embedding*, a vector of hundreds to thousands of dimensions computed by a neural network, and stores the vectors in a [vector database](https://en.wikipedia.org/wiki/Vector_database). At inference time, the question is embedded in the same way, the database returns the chunks whose embeddings are closest to it, usually by [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity), and the [large language model](https://en.wikipedia.org/wiki/Large_language_model) generates its answer from the question and the retrieved chunks. The search thus gives the model knowledge that is not in its weights, such as private or recent documents, and makes its answers traceable to their sources.

An exact search over $$n$$ vectors of $$d$$ dimensions costs $$O(nd)$$ operations per query: $$10^9$$ embeddings of 1,024 dimensions in FP32 take 4 TB, read in full at every query. Therefore, vector databases use **[approximate nearest neighbor](https://en.wikipedia.org/wiki/Nearest_neighbor_search)** (ANN) search, which trades a small loss of *recall* (the fraction of the true nearest neighbors that are returned) for orders of magnitude less work. The main families of methods are:
- **[locality-sensitive hashing](https://en.wikipedia.org/wiki/Locality-sensitive_hashing)** (LSH, [Indyk and Motwani, 1998](https://doi.org/10.1145/276698.276876)): random hash functions that map nearby vectors to the same bucket with high probability, e.g. the signs of random projections for the cosine similarity;
- **inverted file indexes** (IVF): the vectors are clustered with k-means, and a query scans only the clusters whose centroids are closest to it, i.e. the few Voronoi cells of the k-means section around the query;
- **product quantization** (PQ, [Jégou, Douze and Schmid, 2011](https://doi.org/10.1109/TPAMI.2010.57)): each vector is split into sub-vectors, and each sub-vector is replaced by the index of its closest centroid in a small k-means codebook. Split into 64 sub-vectors with 256 centroids each, an embedding of 1,024 FP32 values (4 KB) takes 64 bytes, and its distances to a query are computed with table lookups;
- **graph indexes**, such as [HNSW](https://en.wikipedia.org/wiki/Hierarchical_navigable_small_world) (Hierarchical Navigable Small World, [Malkov and Yashunin, 2016](https://arxiv.org/abs/1603.09320)): each vector is a node connected to its approximate nearest neighbors, in a hierarchy of graphs of increasing density, and a query walks greedily from node to node towards its nearest neighbors, in about $$O(\log n)$$ steps.

Libraries such as FAISS ([Johnson, Douze and Jégou, 2017](https://arxiv.org/abs/1702.08734)) combine these methods, e.g. IVF with PQ, and run them on GPUs. Embeddings escape the curse of dimensionality only in part: the embedding model is trained so that related texts are close, so the embeddings lie near a manifold of much lower intrinsic dimension than their 1,024 coordinates, and the nearest neighbors of a query still stand out from the rest. Thus, the ideas of this post (k-means clustering, Voronoi cells, quantization and the graphs of nearest neighbors) are what makes the search over billions of embeddings fast enough for RAG.
