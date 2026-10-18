# Content issues

Content problems that need the author's decision, found by the text rewrite. Each item proposes a fix.

## Mathematics for ML (2018-04-20-algebra-for-ml.md)

- Overlap with the deep neural networks post (2018-03-27-deep-neural-networks.md): both posts explain
  back-propagation as the chain rule and introduce automatic differentiation. Proposed fix: this post keeps the math
  (the chain rule through K layers and the worked example with its computation graph); the deep neural networks
  post keeps the algorithm, its costs and the PyTorch details.

## Statistics for ML (2018-06-15-statistics-for-ml.md)

- Overlap with the exponential family post (2019-03-20-exponential-family.md), which also covers the canonical form,
  the Bernoulli example, the derivatives of the log-partition function and maximum likelihood, and with the
  variational inference post (2019-07-01-variational-inference-gmm.md), which also derives the ELBO. This post keeps
  a short version of each and links to the dedicated post. Proposed fix: keep the short versions as a reference
  sheet and the full derivations in the dedicated posts; the alternative is to keep only the links here.

## Bayesian Linear Regression, Maximum Likelihood and Maximum-A-Posteriori (2018-08-20-bayesian-linear-regression.md)

- The rewrite removed three parts that repeat other posts: the refresher on the Gaussian pdf (in the statistics
  post), the section "Adding regularization", whose ridge solution is the MAP estimate of the next section, and the
  appendix of linear algebra rules (the derivations now state the two gradient rules they use; the mathematics for
  ML post and The Matrix Cookbook cover the rest). Proposed fix: keep them removed; restore the appendix if it
  should stay as a refresher.
