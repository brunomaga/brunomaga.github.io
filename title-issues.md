# Title issues

Proposed changes to post titles. A new title keeps the URL, which comes from the file name (`permalink: /:title` in
`_config.yml`).

## All posts

- The capitalization is mixed: the posts up to 2019 use title case, most posts since 2023 use sentence case, and
  three titles mix both. Fix: one style for all titles. The proposals below keep the style of each post.
- Five titles do not fit on one line of the home page on a desktop screen with a margin: the line "Category:
  Title" must stay under 750 px of the 815 px column at 16 px. Widths measured in Inter (close to the macOS system
  font); with DejaVu Sans, a common Linux font, the VAE and GAN title wraps:
  - "ML Architectures: Variational Autoencoders (VAEs) and Generative Adversarial Neural Networks (GANs)": 784 px;
  - "Computational Neuroscience: Numerical Resolution of the Electrical Activity of Detailed Neuron Models": 779 px;
  - "Machine Learning: Deep Neural Networks, backpropagation, autodiff, dropout, CNNs and embeddings": 772 px;
  - "Efficient Inference: Prefill vs decode, roofline, KV cache, continuous/ragged batching, disaggregation": 766 px;
  - "Computational Neuroscience: The Leaky Integrate-and-Fire Neuron Model and The Brunel Network": 751 px.

  Fix: the shorter titles proposed below, which measure at most 729 px.

## The Leaky Integrate-and-Fire Neuron Model and The Brunel Network (2016-02-08-lif-brunel.md)

- "The" is capitalized in the middle of the title, and the line is 751 px wide. Proposed: "The Leaky
  Integrate-and-Fire Neuron and the Brunel Network" (694 px).

## Numerical Resolution of the Electrical Activity of Detailed Neuron Models (2016-08-15-neuron-electrical-activity.md)

- "Numerical resolution" reads as the resolution of a grid; the post solves the model equations numerically. The
  line is 779 px wide. Proposed: "Numerical Simulation of Detailed Neuron Models" (595 px).

## Closed-form Linear Regression and Matrix Factorization, and loss functions (2017-08-17-linear-regression-matrix-factorization.md)

- The title mixes title and sentence case, and the post covers more than the closed form (regularization and
  coordinate descent). Proposed: "Linear Regression, Matrix Factorization and Loss Functions" (590 px).

## Unsupervised Learning basics and Principal Component Analysis (2017-11-01-unsupervised-learning.md)

- The title names only PCA, but the post also covers ICA, k-means, Gaussian mixtures, hierarchical clustering,
  DBSCAN, spectral clustering, self-organizing maps, t-SNE, UMAP and self-supervised learning. Proposed:
  "Dimensionality Reduction, Clustering and Self-Supervised Learning" (652 px).

## Deep Neural Networks, backpropagation, autodiff, dropout, CNNs and embeddings (2018-03-27-deep-neural-networks.md)

- The title mixes title and sentence case, and the line is 772 px wide. Proposed: "Deep Neural Networks,
  Backpropagation, Dropout, CNNs and Embeddings" (708 px); backpropagation covers the autodiff section.

## Variational Autoencoders (VAEs) and Generative Adversarial Neural Networks (GANs) (2019-10-12-vae-gan.md)

- The method is named "generative adversarial network" (Goodfellow et al., 2014), and the line is 784 px wide.
  Proposed: "Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs)" (729 px).

## Prefill vs decode, roofline, KV cache, continuous/ragged batching, disaggregation (2025-06-07-inference-prefill-decode-roofline-cache.md)

- The line is 766 px wide. Proposed: "Prefill vs decode, roofline, KV cache, batching and disaggregation" (646 px).

## Bayesian Linear Regression, Maximum Likelihood and Maximum-A-Posteriori (2018-08-20-bayesian-linear-regression.md)

- "Maximum-A-Posteriori" is not the usual spelling of the estimate. Proposed: "Bayesian Linear Regression, Maximum
  Likelihood and Maximum a Posteriori" (718 px).
