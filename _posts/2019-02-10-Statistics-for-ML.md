---
layout: post
title:  "Statistics for ML Engineers"
categories: [machine learning, algebra]
tags: [machinelearning]
---

Just a follow up of the post [Algebra for ML Engineers]({{ site.baseurl }}{% post_url 2018-02-20-Algebra-for-ML %}), with basic concepts of probabilities and statistics that are relevant to ML engineers.

**Basic definitions**:
- **sample space $$Ω$$**  is the set of all *possible* outcomes of an experiment;
- **event space $$A$$** is the space of *potential* results of the experiment;
- **probability of A**, or $$P(A)$$ is the degree of belief that the event $$A$$ will occur, in the interval $$[0,1]$$;
- **random variable** is a target space function $$X : Ω → T$$ that takes an outcome/event in $$Ω$$ (an outcome) and returns a particular quantity of interest $$x \in T$$
  - For example, in the case of tossing two coins and counting the number of heads or tails, a random variable $$X$$ maps to the three possible outcomes: $$X(hh) = 2$$, $$X(ht) = 1$$, $$X(th) = 1$$, and $$X(tt) = 0$$.
  - Note: the name "random variable" creates confusion because it's neither random nor a variable: it's a function!

**Statistical distributions** can be either <a href="{{ site.statistics_distributions | replace: 'XXX', 'CONTINUOUS' }}"> continuous </a> or <a href="{{ site.statistics_distributions | replace: 'XXX', 'DISCRETE' }}"> discrete </a>. Most parametric continuous distributions belong to the [exponential family of distributions]({{ site.baseurl }}{% post_url 2019-03-20-Exponential-Family-Distributions %}) .
- $$P(X=x)$$ is called a **probabilistic mass function (pmf)** or a **probability density function (pdf)** for a discrete or continuous variable $$x$$, respectively; 
  - Any discrete (continuous) domain can be a probability as long as it only has non-negative values and all values sum (integrate) to 1; 
  - the probability of a subset/range of values is the sum of all probabilities of it occurring ie $$P( a \le X \le b) = \int^b_a p(x) dx$$, or the equivalent sum for the discrete use case;
- $$P(X \le x)$$ is the **cumulative distribution function (cdf)**
  - there are CDFs which do not have a corresponding PDF;
- for discrete probabilities:
  - **joint probability** of $$x \in X$$ and $$y \in Y$$ (not independent) is $$P(X=x_i, Y=y_i) = \frac{n_{ij}}{N}$$, where $$n_{ij}$$ can be taken as the $$ij$$-cell in the confusion matrix;

**Sum rule** (or **marginalization property**) tells that the probability of $$x$$ is the sum of all joint probabilities of $$x$$ and another variable $$y$$:
- discrete: $$p(x) = \sum_{y \in Y} p(x,y)$$
- continuous: $$p(x) = \int_Y p(x,y) dy$$

**Product rule** relates the joint distribution and the conditions distribution, and tells that every joint distribution can be factorized into two other distributions:
- ie $$p(x,y) = p(y \mid x) p(x) = p( x \mid y) p(y)$$

**Bayes rule** describes the relationship between some prior knowledge $$p(x)$$ about an unobserved random variable x and some relationship $$p(y | x)$$ between $$x$$ and a second variable $$y$$:
- $$p(x \mid y) = \frac{ p(y \mid x) p(x)}{p(y)} = \text{posterior} = \frac{\text{likelihood} \times \text{prior}}{\text{evidence}}$$. 
- derived from the sum and product rules: $$p(y \mid x) p(y) = p(x \mid y) p(x) \Leftrightarrow p(x \mid y) = \frac{p(y \mid x) p(x)}{p(y)}$$.
- the posterior is the quantity of interest as it tells us what we know about $$x$$ after observing $$y$$.
- $$p(y) = \int p(y \mid x) p(x) dx = \mathbb{E}[ p(y \mid x)]$$ is the **marginal likelihood** or **evidence**

**Expected Value** of a function $$g$$ given:
- an univariate random variable $$X$$ is: $$\mathbb{E}_X [g(x)] = \int_X g(x) p(x) dx$$, or the equivalent sum for the discrete use case 
- multivariate vector as a finite set of univariate ie $$X = \left[X_1, X_2, X_D \right]$$ is:  $$\mathbb{E}_X [g(x)] = \left[ \mathbb{E}_{X_1} [g(x_1)], \mathbb{E}_{X_2} [g(x_2)], ..., \mathbb{E}_{X_D} [g(x_D)] \right] \in \mathbb{R}^D$$  

**Covariance** is the expected product of their two deviations from the respective means, ue:
- univariate: $$Cov_{X,Y}[x,y] = \mathbb{E}_{X,Y} \left[ (x-\mathbb{E}_X[x]) (y-\mathbb{E}_Y[y]) \right] = Cov_{Y,X}[y,x]$$
- multivariate r.v. $$X$$ and $$Y$$ with states $$x \in \mathbb{R}^D$$ and $$y \in \mathbb{R}^E$$: $$Cov_{X,Y}[x,y] = \mathbb{E}[xy^{\intercal}] - \mathbb{E}[x] \mathbb{E}[y]^{\intercal} = Cov[y,y]^{\intercal} \in \mathbb{R}^{D \times E}$$

**Variance**:
- univariate: $$\mathbb{V}_X[x] = Cov_X[x,x] = \mathbb{E}_{X} \left[ (x-\mathbb{E}_X[x])^2 \right] = \mathbb{E}_X[x^2] - \mathbb{E}_X[x]^2$$
- multivariate: $$\mathbb{V}_X[x] = Cov_X[x,x] = \mathbb{E}_X[(x-\mu)(x-\mu)^{\intercal} ] = \mathbb{E}_X[xx^{\intercal}] - \mathbb{E}_X[x] \mathbb{E}_X[x]^{\intercal}$$ 
  - this is a $$D \times D$$ matrix also called the **Covariance Matrix** of the multivariate r.v. $$X$$.
    - it is symmetric and positive semidefinite;
    - the diagonal terms are 1, i.e. no covariance between the same 2 random variables;
    - the off-diagonals are $$Cov[x_i, x_j]$$ for $$i,j = 1, ..., D$$ and $$i \neq j$$. 

**Correlation** between random variables is a measure of covariance standardized to a limited interval $$[-1,1]$$:
- computed from the Covariance (matrix) as $$corr[x,y] = \frac{Cov[x,y]}{\sqrt{\mathbb{V}[x] \mathbb{V}[y]}}  \in [-1, 1]$$
- positive correlation means $y$ increases as $x$ increases (1=perfect correlation). Negative means $$y$$ decreases as $$x$$ increases. Zero means no correlation.

**Rules of transformations of Random Variables**:
- $$\mathbb{E}[x+y] = \mathbb{E}[x] + \mathbb{E}[y]$$.
- $$\mathbb{E}[x-y] = \mathbb{E}[x] - \mathbb{E}[y]$$.
- $$\mathbb{V}[x+y] = \mathbb{V}[x] + \mathbb{V}[y] + Cov[x,y] + Cov[y,x]$$.
- $$\mathbb{V}[x-y] = \mathbb{V}[x] + \mathbb{V}[y] - Cov[x,y] - Cov[y,x]$$.
- $$\mathbb{E}[Ax+b] = A \mathbb{E}[x] = A \mu$$, for the affine transformation $$y = Ax + b$$ of $$x$$, where $$\mu$$ is the mean.
- $$\mathbb{V}[Ax+b] = \mathbb{V}[Ax] = A \mathbb{V}[x] A^{\intercal} = A \Sigma A^{\intercal}$$, where $$\Sigma$$ is the covariance.

**Statistical Independence** of random variables $$X$$ and $$Y$$ iff $$p(x,y)=p(x)p(y)$$. Independence means:
- $$p(y \mid x) = p(x)$$.
- $$p(x \mid y) = p(y)$$.
- $$\mathbb{E}[x,y] = \mathbb{E}[x] \mathbb{E}[y]$$, thus
- $$Cov_{X,Y}[x,y] = \mathbb{E}[x,y] - \mathbb{E}[x] \mathbb{E}[y] = 0$$, thus
- $$\mathbb{V}_{X,Y}[x+y] = \mathbb{V}_X[x] + \mathbb{V}_X[y]$$.
- important: $$Cov_{X,Y}[x,y]=0$$ does not hold in converse, i.e. zero covariance does not mean independence!

**Conditional Independence** of rvs $$X$$ and $$Y$$ given $$Z$$ ie $$X \perp Y \mid Z$$ iff $$p(x,y \mid z) = p(x \mid z) \, p(y \mid z)$$ for all $$z \in Z$$

**Gaussian Distribution** (see the post about the [Exponential Family of Distributions]({{ site.baseurl }}{% post_url 2019-03-20-Exponential-Family-Distributions %}) for more distributions): 
- univariate with mean $$\mu$$ and variance $$\sigma^2$$: $$\,\,\,p(x \mid \mu, \sigma^2 ) = \frac{1}{ \sqrt{2 \pi \sigma^2} } \text{ } exp \left( - \frac{(x - \mu)^2}{2 \sigma^2} \right)$$
- multivariate with mean vector $$\mu \in \mathbb{R}^D$$ and covariance matrix $$\Sigma$$: $$p(x \mid \mu, \Sigma ) = (2 \pi)^{-\frac{D}{2}} \mid\Sigma\mid^{-\frac{1}{2}} \exp \left( -\frac{1}{2}(x-\mu)^{\intercal} \Sigma^{-1}(x-\mu) \right)$$
- marginals and conditional of multivariate Gaussians are also Gaussians:
  <p align="center">
  <img width="45%" height="45%" src="/assets/Statistics-for-ML/bivariate_gaussian.png"/><br/>
  <br/><small>A bivariate Gaussian. <b>Green:</b> joint density p(x,y). <b>Blue:</b> marginal density p(x). <b>Red:</b> marginal density p(y). The conditional disribution is a slice accross the X or Y dimension and is also a Gaussian. <b>Source:</b> post <a href="https://en.wikipedia.org/wiki/Multivariate_normal_distribution">Wikipedia "Multivariate normal distribution"</a></small>
  </p>
  - bivariate Gassian distribution of two Gaussian random variables $$X$$ and $$Y$$: $$p(x,y) = \mathcal{N} \left( \begin{bmatrix} \mu_x \\ \mu_y  \end{bmatrix}, \begin{bmatrix} \Sigma_{xx} & \Sigma_{xy} \\ \Sigma_{yx} & \Sigma_{yy} \end{bmatrix}   \right)$$. 
  - the conditional is also Gaussian: $$p(x \mid y) = \mathcal{N} ( \mu_{x \mid y} , \Sigma_{x \mid y})$$. 
  - the marginal $$p(x)$$ of $$p(x,y)$$: $$p(x) = \int p(x,y) dy = \mathcal{N} ( x \mid \mu_x, \Sigma_{xx})$$.
- the product of two gaussians $$\mathcal{N} (x \mid a, A) \, \mathcal{N}(x \mid b, B)$$ is a Gaussian scaled by a $$c \in \mathbb{R}$$.
- if $$X,Y$$ are independent univariate Gaussian random variables:
  - $$p(x,y)=p(x) p(y)$$, and
  - $$p(x+y) = \mathcal{N}(\mu_x + \mu_y, \Sigma_x + \Sigma_y)$$.
    - weighted sum $$p(ax + by) = \mathcal{N}(a\mu_x + b\mu_y, a^2 \Sigma_x + b^2 \Sigma_y)$$.
- any linear/affine transformation of a Gaussian random variable is also Guassian. Take $$y=Ax$$ being the transformed version of $$x$$:
  - $$\mathbb{E}[y] = \mathbb{E}[Ax] = A \mathbb{E}[x] = A\mu$$, and
  - $$\mathbb{V}[y] = \mathbb{V}[Ax] = A \mathbb{V}[x]A^T = A \Sigma A^{\intercal}$$, thus
  - $$p(y) = \mathcal{N}(y \mid A\mu, A \Sigma A^{\intercal})$$.


**Conjugacy**

- If the posterior distribution $$p(\theta \mid x)$$ is in the same probability distribution *family* as the prior probability distribution $$p(\theta )$$:
  - the prior and posterior are then called **conjugate distributions**, and
  - the prior is called a **conjugate prior** for the likelihood function $$p(x\mid \theta )$$.
- A conjugate prior is an algebraic convenience, giving a closed-form expression for the posterior; otherwise, numerical integration may be necessary.
- Every member of the exponential family has a conjugate prior.

**Exponential Family** are all distributions that can be expressed in the form $$p(x \mid \theta) = h(x) \exp \left(\eta(\theta)^{\intercal} ϕ(x) -A(\theta)\right)$$
- $$θ$$ are the the **natural parameters** of the family
- $$A(θ)$$ is the **log-partition function**, a normalization constant that ensures that the distribution sums up or integrates to one.
- $$ϕ(x)$$ is a **sufficient statistic** of the distribution
  - we can capture information about data in $$ϕ(x)$$.
  - sufficient statistics carry all the information needed to make inference about the population, that is, they are the statistics that are sufficient to represent the distribution:
  - **Fischer-Neyman theorem**: Let $$X$$ have probability density function $$p(x \mid θ)$$. Then the statistics $$ϕ(x)$$ are sufficient for $$θ$$ if and only if $$p(x \mid θ)$$ can be written in the form $$p(x \mid \theta) = h(x) g_{\theta}(\theta(x))$$, where $$h(x)$$ is a distribution independent of $$θ$$ and $$g_θ$$ captures all the dependence on $$θ$$ via sufficient statistics $$ϕ(x)$$. 
  - Note that the form of the exponential family is essentially a particular expression of $$g_θ(ϕ(x))$$ in the Fisher-Neyman theorem. 
- $$\eta$$ is the **natural parameter**,
- for optimization purposes, we use $$ p (x \mid \theta) \propto \exp (\theta^{\intercal} \eta(x)) $$
- Why use exponential family:
  - they have finite-dimensional sufficient statistics;
  - conjugate distributions are easy to write down, and the conjugate distributions also come from an exponential family;
  - Maximum Likelihood Estimation behaves nicely because empirical estimates of sufficient statistics are optimal estimates of the population values of sufficient statistics (recall the mean and covariance of a Gaussian);
  - From an optimization perspective, the log-likelihood function is concave, allowing for efficient optimization approaches to be applied;
- Alternative notation in **natural form**: $$p(x \mid \eta) = h(x) \exp \left(\eta^T T(x) -A(\eta)\right)$$.



