# TODO

Open items that the text rewrite and the picture work found but did not fix. Each item names the post, the problem
and a proposed fix. The user picks the items to fix before the next pass.

## Code issues

Bugs in the code and data files of the posts (under `assets/`).

### Distributed Sorting Algorithms (2014-06-21-distributed-sort.md)

- `assets/distributed-sort/distributedmemorysorter.cxx`, `oddEvenSort_MPI`: the loop runs exactly `mpiSize` phases,
  which sorts the data only when every rank holds the same number of elements. Its own test, `test_OddEvenSort`,
  gives rank r `1000000 + 3r` elements. A simulation with unequal counts fails to sort in about a third of the
  cases after p phases. Fix: balance the counts first, or run phases until an even and an odd phase in a row change
  nothing (an all-reduce of a "changed" flag per phase).

### Distributed Load Balancing and Spatial Slicing (2015-12-01-slicing.md)

- `assets/slicing/slicing.tar.gz`, `DistributedLoadBalancing.cxx` line 20 and `DistributedSorting.cxx` line 28: the
  checksum test is inverted. It warns "checksum ... failed" when the sums before and after are equal
  (`std::abs(pre_sum - post_sum) < 0.000001`), and stays silent when they differ. The warnings are off by default
  (`outputWarnings = false`). Fix: warn when the difference is larger than a relative tolerance, e.g.
  `std::abs(pre_sum - post_sum) > 1e-9 * std::abs(pre_sum)`, since the sum of many doubles changes with the order
  of the additions.

### Unsupervised Learning basics and Principal Component Analysis (2017-11-01-unsupervised-learning.md)

- `assets/unsupervised-learning/kohonen_mnist.zip`, `kohonen.py`, `name2digits`: the function loads `hash.mat`,
  which is not in the zip, so the script stops with a `FileNotFoundError` before the training starts. Fix: add
  `hash.mat` to the zip, or set the digits directly, e.g. `targetdigits = [0, 1, 2, 3]`, the digits of the post's
  figure.

### Bayesian Linear Regression, Maximum Likelihood and Maximum a Posteriori (2018-08-20-bayesian-linear-regression.md)

- `assets/bayesian-linear-regression/bayesian_linear_regression.ipynb`, predictive distribution cell: the data
  have a noise standard deviation of 0.25 (`create_toy_data(sinusoidal, 25, 0.25)`), but the model uses
  `BayesianRegression(alpha=1e-3, beta=2.)`, i.e. a noise standard deviation of 0.71. The predictive band is thus
  three times too wide. Fix: `beta=16.` (1 / 0.25^2).
- Same notebook, posterior cell: `plt.contour(w0, w1, multivariate_normal.pdf(...))` uses the default levels, and
  draws the contour at density 0 where the pdf underflows. Fix: pass
  `levels=pdf.max() * np.exp(-0.5 * np.array([9, 4, 1]))`, or draw the contours of the log-density.
- Same notebook, `GaussianFeature` docstring: "exp(-0.5 * (x - m) / v)" misses the square that the code applies.
  Fix: "exp(-0.5 * (x - m)^2 / v)". `predict(..., sample_size)` says it samples "from the predictive
  distribution", but it returns the noise-free lines `X @ w` for weights drawn from the posterior. Fix: "samples of
  the mean function X @ w, with w drawn from the posterior".

## Diagram issues

Pictures that are wrong or disagree with the post text (under `assets/`). Items marked "Optional" improve a picture
that is not wrong.

### Distributed Matrix Multiplication and Database Joins (2014-04-02-distributed-matmul-dbs.md)

- `distributed-matmul-dbs/joins.svg`, `layout_columns.svg`: the `NULL` cells are filled with `#f5f5f5`, which reads
  as white on the page, while the caption of `joins.svg` says "Gray cells are `NULL`". Fix: a visible gray from the
  palette for `NULL` cells in all join pictures.

### Distributed Sorting Algorithms (2014-06-21-distributed-sort.md)

- `distributed-sort/odd_even_sort.svg`: the ranks hold 3, 5, 3, 4 and 5 elements. Odd-even sort is only guaranteed
  to finish in p phases when all ranks hold the same number of elements (a random test with unequal counts fails
  in about a third of the cases); this example happens to finish in 5 phases. Fix: 4 elements per rank.
- `distributed-sort/odd_even_sort.svg`, `sample_sort.svg`: the labels say "processor", "CPU", "cpu" and "root node",
  and "N" for the number of processors, while the text says "rank" and "p". Fix: "rank", "rank 0" and "p" in
  both pictures.
- `distributed-sort/sample_sort.svg`, step 3: "takes N samples" while only the first N-1 split the value range (the
  last is the maximum). Fix: "takes every N-th sample as a splitter".

### Distributed Matrix Transpose Algorithms (2015-04-21-matrix-transpose.md)

- `matrix-transpose/xcrs_transpose_3.svg`: the brackets under the arrays say "CPU 0", "CPU 1", "CPU 2", while every
  other picture of the post says "rank". Fix: "rank 0", "rank 1", "rank 2".
- `matrix-transpose/crs_format.svg`, `xcrs_format.svg`: the array of column indices is labeled `disps`, MPI's name
  for displacements, which reads as offsets. Optional fix: `col_idx`, as in the matmul post.

### Distributed Load Balancing and Spatial Slicing (2015-12-01-slicing.md)

- `slicing/balance.svg`: the observation says "2 collective operations (MPI_Alltoall, MPI_Alltoallv)", but step 1
  (every rank sends its count to all ranks) is an `MPI_Allgather`. The code (`DistributedLoadBalancing.cxx`) calls
  `MPI_Allgather`, `MPI_Alltoall` and `MPI_Alltoallv`. Fix: "2 collective operations (MPI_Allgather,
  MPI_Alltoallv)", since the receive counts follow from the gathered counts, or list the three calls of the code.
- `slicing/balance.svg`: the extra element goes to the last rank (6, 6, 7), while the code gives it to the first
  ranks (7, 6, 6). Optional fix: 7, 6, 6.
- `slicing/split.svg`: the data is unsorted, with 2, 2, 3, 2, 3, 2, 1 and 3 elements per rank. In SBS, the split
  follows the sort and the balance, so the data is sorted and the counts differ by at most one. The K=3 case splits
  8 ranks into 3, 3 and 2, which SBS does not allow: the number of subnetworks must divide the number of ranks (the
  code warns otherwise). Fix: sorted data with 2 or 3 elements per rank, and K=2 and K=4 only.
- `slicing/str_steps.svg`: the key numbers the ranks 1 to 4, while `balance.svg`, `split.svg` and `sbs.svg` number
  them from 0. Fix: ranks 0 to 3.
- `slicing/sbs.svg`: the labels say "CPU" and "N" for the number of CPUs, while the text says "rank" and "p". Fix:
  "rank" and "p". All data and cut points of the 18-object example are correct.
- `slicing/mini-*.svg`: the four layouts have 10 (single axis), 35 (NUG, 5 x 7), 16 (ORB) and 35 (STR, 5 x 7)
  slices. Optional fix: 16 slices in all four (4 x 4 for NUG and STR), so that they compare on equal terms.

### The Leaky Integrate-and-Fire Neuron and the Brunel Network (2016-02-08-lif-brunel.md)

- `lif-brunel/raster_plot_g4.5_f0.9.svg`, `raster_plot_g5_f2.svg`, `raster_plot_g3_f2.svg`: the plots do not name
  their state. The post used to call them "slow", "regular" and "fast" dynamics, but their parameters are those of
  Brunel's states SI slow (g = 4.5, eta = 0.9), AI (g = 5, eta = 2) and SR (g = 3, eta = 2); the text now uses these
  names. Fix: a title per plot, e.g. "SI slow (g = 4.5, eta = 0.9)".
- Same three plots: the label "time interval (ms)" overlaps the bars at the right end of the lower panel, and
  "spikes count" reads better as "spike count (per ms)". Fix: move the label below the axis.
- `lif-brunel/neuron_plot_equations.svg`: the legend says "Analytic", the text says "exact"; the legend text is too
  small to read at page width. Optional fix: "Exact (fixed dt = 0.1 ms)", "Exact (variable dt)", and a larger font.
- `lif-brunel/action_potential.gif`: a "MakeAGIF.com" watermark in the bottom right corner. Optional fix: crop it.

### Numerical Simulation of Detailed Neuron Models (2016-08-15-neuron-electrical-activity.md)

- `neuron-electrical-activity/action_potential.svg`: the caption says "Adapted from: unknown". Fix: name the source of
  the figure, or redraw it in the site style (rest at -70 mV, threshold at -55 mV, peak at +30 mV, as now).
- `neuron-electrical-activity/neuron_model_scales.svg`, reaction-diffusion panel: the arrow from PDE to cAMP reads as
  "PDE produces cAMP", but phosphodiesterase (PDE) breaks cAMP down. Optional fix: an inhibition arrow (flat head)
  from PDE to cAMP.

### Variable Timestep Simulation of the Electrical Activity of Neurons (2017-02-17-neuron-variable-timestep.md)

- `neuron-variable-timestep/cvode_pulse_currents_table.svg`, `cvode_dependency_on_variation_solution.svg` and
  `cvode_dependency_on_events_arrival.svg`: the currents are in mA and µA (2.7 mA, "threshold current (0.206mA)",
  "0.1µA" to "1mA"), as in the preprint. NEURON's `IClamp` amplitude is in nA, and the threshold current of a cortical
  pyramidal cell model is of the order of 0.1 to 1 nA, so 0.206 mA is likely 0.206 nA. The text gives no current
  values. Fix, after a check of the simulation scripts: mA to nA and µA to pA in the labels.
- Same pictures, and `l5_neuron_pulse1_3ma_100ms_cvode_vs_backward_euler.svg`, `l5_neuron_pulse_1000ms_results.svg`
  and the six `neuron_*.svg`: a dot is the thousands separator ("48.001 steps", "9.750 steps", "1.512 steps",
  "100.001 steps", "200.001 steps", "6.240 steps", axis ticks "1.000" and "1.100"). English text reads these as
  decimals. Fix: a comma or no separator ("48,001", "1,000").
- `neuron_208223.svg` and `neuron_87784.svg`: the histogram axis says "interval events (ms)"; the other four
  `neuron_*.svg` say "events interval (ms)". Fix: "interval between events (ms)" in all six.
- `lvardt_exec_time_comparison.svg`: the curve labels "dt=0.0025" and "dt=0.025" have no unit. Fix: "dt = 0.0025 ms"
  and "dt = 0.025 ms". Its caption "source: Lytton et al. 2005" names a two-author paper. Fix: "Source: Lytton and
  Hines (2005)".
- `benchmark_two_rows_2.svg`: the legend entry "(2a) BSP Deriv. Implicit†" has no key for the dagger. The preprint's
  caption defines it: Newton iterations solve each mechanism implicitly, but the coupling between mechanisms stays
  explicit. Fix: a key line in the legend, "† implicit per mechanism, explicit across mechanisms".
- `local_variable_timestep_six_cell_example.svg`: the caption cites the NEURON book; the figure matches Fig. 1 of
  Lytton and Hines (2005). Optional fix: cite the paper in the caption.

### Linear Regression, Matrix Factorization and Loss Functions (2017-08-17-linear-regression-matrix-factorization.md)

- `linear-regression-matrix-factorization/matrix-factorization.svg`: W and Z^T have swapped orientations. W is drawn as
  a horizontal strip with the movie index d along it, and Z^T as a vertical strip with the user index n along it. In
  X, d indexes the rows (movies) and n the columns (users), so W (D x K) is a tall block left of X, with row d
  aligned to movie d, and Z^T (K x N) a wide block above X, with column n aligned to user n. Fix: redraw W as a 7 x K
  block left of X and Z^T as a K x 7 block above X, with K = 2 or 3 and K marked on both. Optional: "x_dn" in lower
  case, as in the text.
- `linear-regression-matrix-factorization/coordinate_descent.svg`: the steps stop short of the minimum along each
  coordinate. From (-0.5, -1), the minimum along y is at y = 0.6x = -0.3, but the path stops at y = -0.4; the next
  minimum along x is at x = 0.6y, but the path stops at x = -0.34. The text describes exact minimization along each
  coordinate. Fix: draw the exact path (-0.5, -1), (-0.5, -0.3), (-0.18, -0.3), (-0.18, -0.108), (-0.065, -0.108),
  (-0.065, -0.039), ..., where each sweep scales the point by 0.36.

### Unsupervised Learning basics and Principal Component Analysis (2017-11-01-unsupervised-learning.md)

- `unsupervised-learning/ica_whitening.svg`: the panel label is "(d) whitening (z1, z2)", after "(a) source" and
  "(b) observed"; no panel is labelled (c). Fix: "(c) whitening (z1, z2)".
- `unsupervised-learning/kohonen2.svg` and its caption: the picture uses v for the input, s for the winner neuron and
  w_s for its weights; the text uses x_i, j* and w_j. The caption defines its own symbols, so the picture is not
  wrong. Optional fix: relabel v, s and w_s as x, j* and w_j*.

### Deep Neural Networks, Backpropagation, Dropout, CNNs and Embeddings (2018-03-27-deep-neural-networks.md)

- `deep-neural-networks/fully_sparse_connectivity.svg`: the picture has no caption, and it reproduces Figure 9.2 of
  Goodfellow, Bengio and Courville, Deep Learning (2016). It uses s for the outputs; the text uses x^(1). Fix: a
  caption "Sparse (top) and full (bottom) connectivity; the outputs that depend on x3 are highlighted. Adapted from:
  Goodfellow et al., Deep Learning (2016), Fig. 9.2". Optional: relabel s as x^(1).
- `deep-neural-networks/densenet.svg`: the caption says "An example of a CNN with skip connections (DenseNet)", but
  the picture shows one dense block of 5 layers with a growth rate of 4, not a full CNN. Fix: "A dense block of
  DenseNet, with 5 layers and a growth rate of 4: each layer takes the feature maps of all preceding layers as input.
  Adapted from: Densely Connected Convolutional Networks".
- `deep-neural-networks/dropout.svg`: the picture has no caption. Optional fix: "A network without dropout (left) and
  with a dropout rate of 0.5 (right); crossed units are dropped."

### Mathematics for ML (2018-04-20-algebra-for-ml.md)

- `algebra-for-ml/newton_method.gif`: the legend is in German ("Funktion", "Tangente"). Fix: "function" and
  "tangent".
- `algebra-for-ml/svd_example.svg`: the right half is the "Lenna" test image, which IEEE no longer allows in its
  publications since April 1, 2024. Optional fix: the same experiment (all 256, then 20, 50 and 85 singular values)
  on another test image.
- `algebra-for-ml/mml_eigenvalues_and_eigenvectors.svg`: the right third of the picture is the book's caption text,
  in a small font at 65% of the page width. Optional fix: remove the text from the picture, widen the plots, and put
  a short description in the page caption, e.g. "Five linear mappings of 400 points (left to right: input,
  eigenvectors scaled by their eigenvalues, output)".

### Statistics for ML (2018-06-15-statistics-for-ml.md)

- `statistics-for-ml/bivariate_gaussian.svg`: the caption says "Green: joint density p(x,y)", but the green ellipse
  is the 3σ contour of the joint density, the dots are samples, and the bars are histograms of the samples. Fix:
  "Samples of a bivariate Gaussian (dots), the 3σ contour of the joint density p(x,y) (green), and the marginal
  densities p(x) (blue) and p(y) (red) with histograms of the samples. A conditional distribution is a normalized
  slice of the joint density at a fixed x or y, and it is also Gaussian. Adapted from: Wikipedia page for
  Multivariate normal distribution".
- `statistics-for-ml/confidence_interval.svg`: the picture has no caption and no source. Fix: the caption "95%
  confidence interval for the mean: the limits X̄ ± 1.96σ/√N leave a probability of 0.025 in each tail", and the
  source of the drawing. Optional: the curve is centered at X̄, as in many textbooks, but the sampling distribution
  of X̄ is centered at μ; a curve centered at μ, with the interval drawn around one sample mean X̄, is exact.

### Bayesian Linear Regression, Maximum Likelihood and Maximum a Posteriori (2018-08-20-bayesian-linear-regression.md)

- `bayesian-linear-regression/linear_bayesian_posterior_5.svg`: the posterior panel shows a large jagged purple
  curve far from the posterior. It is the contour at density 0: it traces where the pdf underflows to 0 in double
  precision, at a Mahalanobis distance of about 39. Fix: draw the contours at fixed fractions of the maximum
  density, e.g. `levels = pdf.max() * np.exp(-0.5 * np.array([9, 4, 1]))` for the 3σ, 2σ and 1σ ellipses.
- Caption of the four `linear_bayesian_posterior_*.svg` pictures: it does not say what the rows, the cross and the
  lines are. Fix: "Online learning for y = w0 + w1 x after 1, 2, 3 and 20 samples (rows). Left: the posterior over
  (w0, w1); the cross marks the true weights (-0.3, 0.5). Right: the samples and six lines with weights drawn from
  the posterior. Inspired by Fig. 3.7 of Pattern Recognition and Machine Learning, Chris Bishop". Optional: the
  panel title "prior/posterior" becomes "posterior", since no row shows the prior.
- `bayesian-linear-regression/linear_bayesian_predictive_distribution_*.svg`: the model assumes a noise standard
  deviation of 0.71 (noise precision beta = 2), but the data have a noise standard deviation of 0.25. After 25
  samples, the band is thus 0.76 to 0.88 wide on each side of the mean, three times the scatter of the samples.
  Fix: redraw with beta = 16 (band 0.27 to 0.32), see the code issues above. Caption: "Predictive distribution of a
  model with 9 Gaussian basis functions and a bias, after 1, 2, 4 and 25 noisy samples (circles) of sin(2 pi x)
  (blue): mean (orange) and mean plus and minus one predictive standard deviation (light orange)".

### Exponential Family of Distributions (2019-03-20-exponential-family.md)

- `exponential-family/common_distributions.svg`: the names of the distributions overlap the bars and the curves in
  their boxes, e.g. "Uniform", "Negative Binomial", "Normal (Gaussian)" and "Chi-Squared". Fix: place each name
  above the plot in its box, or give the names a white outline.
- `exponential-family/common_distributions.svg`: the text says that the binomial approaches the normal for large n
  (de Moivre-Laplace theorem), as the source post by Sean Owen does, but the figure has no arrow from Binomial to
  Normal; it has the arrows from Hypergeometric and from Poisson. Fix: add the arrow from Binomial to Normal.
- Optional: the caption starts with "The 15 most common probability distributions", but the source post only calls
  them common. Fix: "Fifteen common probability distributions and their relationships", with the same source.

## Content issues

Content problems that need the author's decision, such as overlap between posts.

### Mathematics for ML (2018-04-20-algebra-for-ml.md)

- Overlap with the deep neural networks post (2018-03-27-deep-neural-networks.md): both posts explain
  back-propagation as the chain rule and introduce automatic differentiation. Proposed fix: this post keeps the math
  (the chain rule through K layers and the worked example with its computation graph); the deep neural networks
  post keeps the algorithm, its costs and the PyTorch details.

### Statistics for ML (2018-06-15-statistics-for-ml.md)

- Overlap with the exponential family post (2019-03-20-exponential-family.md), which also covers the canonical form,
  the Bernoulli example, the derivatives of the log-partition function and maximum likelihood, and with the
  variational inference post (2019-07-01-variational-inference-gmm.md), which also derives the ELBO. This post keeps
  a short version of each and links to the dedicated post. Proposed fix: keep the short versions as a reference
  sheet and the full derivations in the dedicated posts; the alternative is to keep only the links here.

### Bayesian Linear Regression, Maximum Likelihood and Maximum a Posteriori (2018-08-20-bayesian-linear-regression.md)

- The rewrite removed three parts that repeat other posts: the refresher on the Gaussian pdf (in the statistics
  post), the section "Adding regularization", whose ridge solution is the MAP estimate of the next section, and the
  appendix of linear algebra rules (the derivations now state the two gradient rules they use; the mathematics for
  ML post and The Matrix Cookbook cover the rest). Proposed fix: keep them removed; restore the appendix if it
  should stay as a refresher.

## Title issues

A new title keeps the URL, which comes from the file name (`permalink: /:title` in `_config.yml`). A title, with the
category that the home page puts before it, fits on one line of the home page on a desktop screen (under 750 px).

### All posts

- The capitalization is mixed: the posts up to 2019 use title case, most posts since 2023 use sentence case, and
  a few titles mix both. Fix: one style for all titles.

### Unsupervised Learning basics and Principal Component Analysis (2017-11-01-unsupervised-learning.md)

- To do later. The title names only PCA, but the post also covers ICA, k-means, Gaussian mixtures, hierarchical
  clustering, DBSCAN, spectral clustering, self-organizing maps, t-SNE, UMAP and self-supervised learning.
  Proposed: "Dimensionality Reduction, Clustering and Self-Supervised Learning" (652 px).
