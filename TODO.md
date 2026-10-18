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

### Building a GPT model in PyTorch from scratch (2023-02-28-gptlite.md)

- Post, section "Multi-Head Masked Attention": the uniform attention matrix is created on `device`, but `x` and
  `tril` on the CPU, so on a GPU machine `wei @ x` and `wei.masked_fill(tril==0, ...)` fail with a device mismatch.
  The alternative notation applies the mask and the softmax to the `wei` of the previous snippet, which gives the
  uniform matrix only because the unmasked entries of each row are equal. Fix: drop `device=device` from the first
  snippet, and start the second one with `wei = torch.zeros((S,S))`.
- Post, section "Training Loop": `eval_loss` runs with dropout on, since the model never switches to eval mode, and
  the loop defines the function again at every step. Fix: define `eval_loss` once before the loop, and call
  `m.eval()` at its start and `m.train()` before it returns, as `assets/gptlite/main.py` does.
- Post, section "Final Model": the comment "one layer normalization layer after transformer blocks and one before
  linear layer that outputs the vocabulary" describes the single `self.ln` as two layers. Fix: "one layer
  normalization after the transformer blocks, before the linear layer that outputs the vocabulary".
- Post, section "Tokenization": the code block that lists the 65 characters is marked `python`, so the highlighter
  marks "!" as an error. Fix: remove the language tag `python` from the opening fence.
- `assets/gptlite/gptlite.py`, `GPTlite.__init__`: the comment `# Add this` after `self.ln = nn.LayerNorm(d_model)`
  is a leftover editing note. Fix: remove it.
- `assets/gptlite/utils.py`, `sin_cos_positional_embeddings`: no code calls it, and it differs from the formula of
  the transformer: the base is 1000 instead of 10000, and the odd dimension 2i+1 uses the exponent (2i+1)/E instead
  of the 2i/E of its pair. Fix: remove the function, or set `div_term = 10000 ** (2 * (torch.arange(E) // 2) / E)`.
- `assets/gptlite/utils.py`, `get_gpt3_small_model_parameters`: the docstring dates the GPT-3 paper to 2021 instead
  of 2020, and `batch_size = 500_000` counts tokens in the paper (0.5M tokens, about 244 sequences of 2048 tokens),
  while `get_batch` reads it as a number of sequences. The only caller overrides it. Fix: "Brown et al, 2020", and
  `batch_size = 500_000 // 2048`, or a comment that the value counts tokens.

## Diagram issues

Pictures that are wrong or disagree with the post text (under `assets/`). Items marked "Optional" improve a picture
that is not wrong.

### All pictures

- The text rewrite (PR #92) changes the text of the posts, and the picture restyle (branch
  `claude/project-thread-56yymq`) changes their pictures, img tags and captions, so both change the same Markdown
  files. The restyled pictures already use the symbols of the text rewrite: ranks and p in the sorting pictures, x_i,
  j* and w_j* in the Kohonen pictures, and "Exact" in the legend of the LIF solutions. Fix, in the order the user
  chose: merge the picture branch first, then merge the main branch into the text rewrite, keeping its text and the
  pictures, img tags and captions of the main branch, and check each caption against the new text.
- The two kernel fusion posts (2026-08-18-kernel-fusion-tensor-parallelism.md and
  2026-10-09-kernel-fusion-expert-parallelism.md): their 17 pictures, `inference/imgs/tp_*.svg` and `ep_*.svg`, are
  not restyled. 16 are copies of the figures of the paper repositories, and `tp_expert_sharding.svg` comes from a tab
  of `inference/inference.drawio` without a figure group. All show at widths set by hand (40% to 100%), so their text
  sizes differ from the other pictures. Fix: redraw them at page scale in tabs of `inference/inference.drawio` named
  after the posts, from the draw.io or TikZ sources of the paper repositories (CLAUDE.md, Pictures, section 2), and
  check their captions.
- Style details that the restyle left, from an automated check of the 231 draw.io figures: 384 rounded boxes in 39
  figures have a corner radius of 1 to 4 px instead of 5 px, and the 12 chat bubbles of `publications/llama2_gatt.svg`
  15 px; 147 arrows in 19 figures are curved but are neither pointers nor loops (for example in
  `deep-neural-networks/densenet.svg`, `lif-brunel/diagram_class.svg` and `publications/graphconvnets.svg`); 176
  filled cells of `distributed-training/imgs/mixture_of_depths.svg`, and a few in other figures, have no outline;
  `learning-from-sequences/transformer_attention.svg` (#97afd2, a heat map shade) and `diffusion/sora_vae.svg`
  (#333333) use colors outside the palette; 8 figures have lines thinner than 0.7 px;
  `publications/attention_mech.svg` has 8 arrow heads smaller than size 6; `vae-gan/gan.svg` has a gradient glow
  around the prior. Fix: set them to the values of CLAUDE.md, then run the check again.

### Distributed Matrix Transpose Algorithms (2015-04-21-matrix-transpose.md)

- `matrix-transpose/crs_format.svg`, `xcrs_format.svg`: the array of column indices is labeled `disps`, MPI's name
  for displacements, which reads as offsets. Optional fix: `col_idx`, as in the matmul post.

### Distributed Load Balancing and Spatial Slicing (2015-12-01-slicing.md)

- `slicing/mini-*.svg`: the four layouts have 10 (single axis), 35 (NUG, 5 x 7), 16 (ORB) and 35 (STR, 5 x 7)
  slices. Optional fix: 16 slices in all four (4 x 4 for NUG and STR), so that they compare on equal terms.

### The Leaky Integrate-and-Fire Neuron and the Brunel Network (2016-02-08-lif-brunel.md)

- `lif-brunel/raster_plot_g4.5_f0.9.svg`, `raster_plot_g5_f2.svg`, `raster_plot_g3_f2.svg`: the plots do not name
  their state. The post used to call them "slow", "regular" and "fast" dynamics, but their parameters are those of
  Brunel's states SI slow (g = 4.5, eta = 0.9), AI (g = 5, eta = 2) and SR (g = 3, eta = 2); the text now uses these
  names. Fix: a caption under each plot, e.g. "SI slow (g = 4.5, eta = 0.9)", since CLAUDE.md puts titles in captions.
- Same three plots: the label "time interval (ms)" overlaps the bars at the right end of the lower panel, and
  "spikes count" reads better as "spike count (per ms)". Fix: move the label below the axis.
- `lif-brunel/action_potential.gif`: a "MakeAGIF.com" watermark in the bottom right corner, and the arrows that show
  the direction of the current are missing in the first and last frames. Fix, asked by the user for after the merge: a
  static SVG in which the arrows are always visible.

### Numerical Simulation of Detailed Neuron Models (2016-08-15-neuron-electrical-activity.md)

- `neuron-electrical-activity/brain_scales.svg`: the LadyofHats drawing is not restyled and not at page scale. It is
  716 px wide and shows at 50%, so its sans-serif panel letters (a to d) and ion labels show at 61% of their size.
  Fix: keep the drawing, scale it to about 440 px, redraw the letters and labels in Times New Roman (14 px and 12 px),
  and set P = W/880.

### Variable Timestep Simulation of the Electrical Activity of Neurons (2017-02-17-neuron-variable-timestep.md)

- `neuron-variable-timestep/cvode_pulse_currents_table.svg`, `cvode_dependency_on_variation_solution.svg` and
  `cvode_dependency_on_events_arrival.svg`: the currents are in mA and µA (2.7 mA, "threshold current (0.206mA)",
  "0.1µA" to "1mA"), as in the preprint. NEURON's `IClamp` amplitude is in nA, and the threshold current of a cortical
  pyramidal cell model is of the order of 0.1 to 1 nA, so 0.206 mA is likely 0.206 nA. The text gives no current
  values. Fix, after a check of the simulation scripts: mA to nA and µA to pA in the labels.
- `benchmark_two_rows_2.svg`: the legend entry "(2a) BSP Deriv. Implicit†" has no key for the dagger. The preprint's
  caption defines it: Newton iterations solve each mechanism implicitly, but the coupling between mechanisms stays
  explicit. Fix: a key line in the legend, "† implicit per mechanism, explicit across mechanisms".
- Optional, `benchmark_two_rows_2.svg`: the orange "region of similar runtime growth" does not follow from the plotted
  runtimes. Fix: check it against the benchmark data, or remove it.
- `neuron_203076.svg`, `neuron_138083.svg`, `neuron_208223.svg` and `neuron_87784.svg`: the last 10 ms bar of each
  event histogram leaves out the events after 7.5 s, and the burst bars near 0 s are cut at the top of the plot (198,
  203, 44 and 47 events) without a mark. Fix: redraw the histograms from the event data of the simulations, with the
  count on each cut bar.
- `cvode_pulse_currents_table.svg`: the CVODE step counts cover 1.1 s of simulation, while the 48,001 steps of
  backward Euler cover 1.2 s. The values are those of the table of the preprint. Fix, after a check of the simulation
  scripts: step counts for the same time span.

### Linear Regression, Matrix Factorization and Loss Functions (2017-08-17-linear-regression-matrix-factorization.md)

- Optional, `linear-regression-matrix-factorization/matrix-factorization.svg`: the rating 6 (movie 1, user 7) is
  outside a scale of 1 to 5 stars. Fix: a rating from 1 to 5.

### Unsupervised Learning basics and Principal Component Analysis (2017-11-01-unsupervised-learning.md)

- `unsupervised-learning/gaussianscatterpca.svg`: the arrow heads end past the scaled eigenvectors, at 3.75 and 1.75
  instead of 3 and 1, the lengths in the Wikipedia figure it follows. Fix: arrow heads that end at the tips of the
  vectors.
- Optional, `unsupervised-learning/k-means_convergence.gif`: an animation of 15 frames. Fix: an SVG with a few
  iterations side by side, computed by running k-means on the same data.

### Mathematics for ML (2018-04-20-algebra-for-ml.md)

- `algebra-for-ml/newton_method.gif`: the legend is in German ("Funktion", "Tangente"), and the animation numbers the
  first guess x1, while the text starts at x0. The caption translates both words and explains the numbering. Fix: one
  SVG with the function, the first three tangents and the guesses x0 to x3, labeled in English.
- `algebra-for-ml/svd_example.svg`: the panels are crops of the "Lenna" test image, which IEEE no longer allows in its
  publications since April 1, 2024, and the panels with 50 and 85 singular values look like ranks 25 to 35. Optional
  fix: the same experiment (all 256, then 20, 50 and 85 singular values) on another test image, with the computed
  results.
- Optional, `algebra-for-ml/mml_eigenvalues_and_eigenvectors.svg`: the plots have unequal axis scales. Fix: the same
  scale on both axes.

### Statistics for ML (2018-06-15-statistics-for-ml.md)

- `statistics-for-ml/bivariate_gaussian.svg`: the picture labels the marginals p(X) and p(Y), while the text and the
  caption write p(x) and p(y), and the caption calls a conditional distribution "a slice" of the joint density. Fix:
  p(x) and p(y) in the picture, and "a normalized slice" in the caption.
- `statistics-for-ml/confidence_interval.svg`: the drawing has no source, and the label "0.95" is crossed by the
  center line. Fix: name the source of the drawing, or redraw it, and move the label. Optional: the curve is centered
  at X̄, as in many textbooks, but the sampling distribution of X̄ is centered at μ; a curve centered at μ, with the
  interval drawn around one sample mean X̄, is exact.

### Bayesian Linear Regression, Maximum Likelihood and Maximum a Posteriori (2018-08-20-bayesian-linear-regression.md)

- `bayesian-linear-regression/linear_bayesian_predictive_distribution_*.svg`: the model assumes a noise standard
  deviation of 0.71 (noise precision beta = 2, as the caption states), but the data have a noise standard deviation of
  0.25. After 25 samples, the band is thus 0.76 to 0.88 wide on each side of the mean, three times the scatter of the
  samples. Fix: redraw with beta = 16 (band 0.27 to 0.32), see the code issues above, and change the caption.

### Exponential Family of Distributions (2019-03-20-exponential-family.md)

- `exponential-family/common_distributions.svg`: the text says that the binomial approaches the normal for large n
  (de Moivre-Laplace theorem), as the source post by Sean Owen does, but the figure has no arrow from Binomial to
  Normal; it has the arrows from Hypergeometric and from Poisson. Fix: add the arrow from Binomial to Normal.
- Optional: the caption starts with "The 15 most common probability distributions", but the source post only calls
  them common. Fix: "Fifteen common probability distributions and their relationships", with the same source.

### Variational Inference (ELBO, CAVI) and Gaussian Mixture Models (2019-07-01-variational-inference-gmm.md)

- `variational-inference-gmm/gmm_iterations.svg`: the caption calls the gray ellipses "2σ contours of the variational
  approximating factors", as the source (Blei et al., Figure 3) does, and says that the figure uses the 2D version of
  the model. But the ellipses are elongated and, at iteration 50, as large as the clusters. In the model of the text,
  the factor of each mean is an isotropic Gaussian whose variance s_k² = 1/(1/σ² + Σ_i φ_ik) shrinks as the cluster
  gains points, so the figure comes from a model with unknown covariances. Fix: "The gray ellipses are the fitted
  components at initialization and after 20, 28, 35 and 50 iterations" in the caption, or redraw the figure with the
  model of the text in 2D: unit-variance components, drawn as circles of radius 2 around each m_k.

### Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs) (2019-10-12-vae-gan.md)

- `vae-gan/gan.svg`: the discriminator has two output arrows, "1 (real)" and "0 (generated)", but D outputs one
  number, the probability that its input is real, as the caption says. Fix: one output arrow labeled "D(x):
  probability that x is real (1) or generated (0)".
- Optional, `vae-gan/vae.svg`: the encoder box shared by the AE and the VAE paths is labeled q_φ(z|x), a distribution,
  but the AE encoder is a deterministic function. Fix: label the encoder "Encoder φ".

### Learning from sequences with Encoder-Decoder, Transformers and BERT (2020-02-28-learning-from-sequences.md)

- Optional, `learning-from-sequences/encoder_decoder.svg`: the output layer has as many neurons as the input
  embedding and the hidden state, but the decoder outputs a softmax over the target vocabulary, which is much
  larger. Fix: draw the output layer wider than the other layers, with "softmax over the vocabulary" next to it.
- Caption of `learning-from-sequences/transformer_attention.svg`: it writes the sentence length as N, while the text
  rewrite writes n. Fix: n, at the merge with the text rewrite.

### Diffusion models, from DDPMs to DiTs (2025-02-25-diffusion.md)

- `diffusion/masked_autoencoders_cropped.svg` and `masked_autoencoders.svg` are now the same picture, shown twice in
  the post. Fix: keep one file and link both places to it, or show the picture once.
- Optional, `diffusion/dit_moe.svg`: the dashed residual connections start after the first AdaLN, while the code of
  the post adds the input of the block, and three experts are gated, while the code routes each token to 2 experts.
  The picture follows the figure of the paper, which draws both this way. Fix: check both against the code, and say in
  the caption where the code differs.

### Faster attention (GQA, FlashAttention, sparse, linear and latent attention) (2025-12-30-inference-attention.md)

- Optional, `inference/imgs/mla.svg`: the absorb = True panel shows a whole sequence, while weight absorption serves
  decode, with one new token (H x 1 x (P+1) scores over P+1 latent vectors). The panels are labeled "usually prefill"
  and "usually decode". Fix: redraw the absorb = True panel for one decode token.

### Publications bookmark (publications.md)

- Optional, `publications/llama_3_multi_modal.svg`: the connectors "Every 4th layer" and "Every ~3rd layer" have arrow
  heads at both ends, as in Fig. 28 of the paper, while the forward pass sends data one way. Fix: one arrow head,
  toward the cross-attention layers.
- `publications/comet_2.svg`: GEMM #0 and #1 mix the rows of two experts after a global sort by source rank. The
  picture follows Fig. 5 of the paper, whose text does not say whether the sort is global or per expert. Fix: check
  the code of Comet, and sort per expert if the code does.
- Caption of `publications/performers2.gif`: no credit. Fix: "Source: [Rethinking Attention with
  Performers](https://research.google/blog/rethinking-attention-with-performers/)". Optional: a static SVG of the
  prefix sums instead of the animation.

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

### Variational Inference (ELBO, CAVI) and Gaussian Mixture Models (2019-07-01-variational-inference-gmm.md)

- The rewrite replaced the derivation of the CAVI update with the chain rule of probability, from the CMU 10-708
  lecture notes, by the shorter derivation of section 10.1.1 of Bishop's PRML: as a function of one factor, the ELBO
  is a negative KL divergence. It also removed the recaps of conditional probability and of the chain rule. The CMU
  notes stay linked. Proposed fix: keep the shorter derivation.

### Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs) (2019-10-12-vae-gan.md)

- The rewrite removed the section "Detour: why maximize the Expected Value?", which explained with games of chance
  that the law of large numbers makes the average of many trials approach the expected value. The losses in the
  post are expectations estimated with mini-batch averages, which the VI and statistics posts cover. Proposed fix:
  keep it removed.

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
