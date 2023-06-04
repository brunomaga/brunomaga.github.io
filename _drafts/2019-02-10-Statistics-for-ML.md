---
layout: post
title:  "[Draft] Statistics for ML Engineers"
categories: [machine learning, algebra]
tags: [machinelearning]
---

### Chapter 6 MML book: Probabilities and Distributions

Basic definitions:
- **sample space $$Ω$$**  is the set of all *possible* outcomes of an experiment;
- **event space $$A$$** is the space of *potential* results of the experiment;
- **probability of A**, or $$P(A)$$ is the degree of belief that the event $$A$$ will occur, in the interval $$[0,1]$$;
- **random variable** is a target space function $$X : Ω → T$$ that takes an outcome/event in $$Ω$$ (an outcome) and returns a particular quantity of interest $$x \in T$$
  - For example, in the case of tossing two coins and counting the number of heads or tails, a random variable $$X$$ maps to the three possible outcomes: $$X(hh) = 2$$, $$X(ht) = 1$$, $$X(th) = 1$$, and $$X(tt) = 0$$.
  - Note: the name "random variable" creates confusion because it's neither random nor a variable: it's a function!
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
- read as $$\text{posterior} = \frac{\text{likelihood} \times \text{prior}}{\text{evidence}}$$ 
- computed as $$p(x \mid y) = \frac{ p(y \mid x) p(x)}{p(y)} $$ 
  - this is derived directly from the sum and product rules
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
- $$\mathbb{E}[Ax+b] = A \mathbb{E}[x] = A \mu$$, where $$\mu$$ is the mean.
- $$\mathbb{V}[Ax+b] = \mathbb{V}[Ax] = A \mathbb{V}[x] A^{\intercal} = A \Sigma A^{\intercal}$$, where $$\Sigma$$ is the covariance.

**Statistical Independence** of random variables $$X$$ and $$Y$$: $$p(x,y)=p(x)p(y)

### Basics of Probability

There are two frameworks for statistical modelling, the explanatory model framework and the predictive framework. We cannot always model the resolution of the observed data, so we introduce **stochasticity** to our model. Stochasticity refers to the property of being well described by a random probability distribution. Although stochasticity and **randomness** are distinct in that the former refers to a modeling approach and the latter refers to phenomena themselves, these two terms are often used synonymously.  The data is viewed as observations from that model.

Statistical distributions can be either <a href="{{ site.statistics_distributions | replace: 'XXX', 'CONTINUOUS' }}"> continuous </a> or <a href="{{ site.statistics_distributions | replace: 'XXX', 'DISCRETE' }}"> discrete </a>, while most continuous distributions belong to the [exponential family of distributions]({{ site.baseurl }}{% post_url 2019-03-20-Exponential-Family-Distributions %}) .

- the **union** of two subsets is written as $$F_1 \cup F_2 = \{ ω ∈ Ω : ω ∈ F1 \text{ or } ω ∈ F2 \}$$;
- the  **intersection** is $$F1 ∩ F2 = \{ ω ∈ Ω : ω ∈ F1 \text{ and } ω ∈ F2 \}$$;
- two events F1 and F2 are **disjoint** if they have no elements in common, or $F_1 ∩ F_2 = ∅$;
- the **complement** of $F$ is written as $F^C$ and contains all elements of $\Omega$ which are not in $F$ ie $F^c = \{ ω ∈ Ω : ω \not\in F \}$. 
  - From this we write $F ∪ F^c = Ω$;
- a **partition** $$\{ F_n \}$$  for $n \ge 1$ is a collection of events such that $F_i ∩ F_j = ∅$ for all $i \neq j$ and $\cup_{n≥1} F_n = Ω$;
- the **difference** between $F_1 and F_2$ is defined as $F1 \backslash F2 = F1 ∩ F_2^C$;
- the following **properties** hold:
  - associativity: $(F1 ∪ F2) ∪ F3 = F1 ∪ (F2 ∪ F3) = F1 ∪ F2 ∪ F3$
  - associativity: $(F1 ∩ F2) ∩ F3 = F1 ∩ (F2 ∩ F3) = F1 ∩ F2 ∩ F3$
  - distributivity: $F1 ∩ (F2 ∪ F3) = (F1 ∩ F2) ∪ (F1 ∩ F3)$
  - distributivity: $F1 ∪ (F2 ∩ F3) = (F1 ∪ F2) ∩ (F1 ∪ F3)$
  - De Morgan's Laws: $(F1 ∪ F2)^c = F^c_1 ∩ F^c_2$  and  $(F1 ∩ F2)^c = F^c_1 ∪ F^c_2$


A **probability measure** $\mathbb{P}$ is a real function defined over the events in Ω, that provides the probability of an event. Three constraints hold: always positive ($\mathbb{P}(F) \ge 0$), sum to 1 ($\mathbb{P}(\Omega) = 1$); and $\mathbb{P}(G) \sum_{n \ge 1} \mathbb{P}(F_n)$ for the union G of the disjoint events $$\{ F_n \}$$. 

Using the previous axioms we can show that:
- We can show that $Pr(F1 ∪ F2) = Pr(F1) − Pr(F1 ∩ F2) + Pr(F2)$;
- $$Pr(F1 ∩ F2) ≤ min\{Pr(F1), Pr(F2)\}$$;
- $F ∪ F^c = Ω$, $1 = Pr(Ω) = Pr(F) + Pr(F^c)$, thus $Pr(F^c) = 1 − Pr(F)$

### Conditional Probability and Independence

The events $$\{ G_n \}$$ are called **independent** if $$ Pr(G_{1} ∩ ··· ∩ G_{K}) = Pr(G_1) × Pr(G_2) × ··· × Pr(G_K) $$. This includes random varibles, numerical summaries of the outcome of a random experiment. A random variable is a real function $X : Ω → \mathbb{R}$. 

We write $$\{a ≤ X ≤ b\}$$ to denote the event $$\{ω ∈ Ω : a ≤ X(ω) ≤ b\}$$. If $A ⊂ \mathbb{R}$ is a more general subset, we write $$\{X ∈ A\}$$ to denote the event $$\{ω ∈ Ω : X(ω) ∈ A\}$$

 
- **Conditional probability** of $F_1$ given $F_2$: $$Pr(F_1 \mid F_2) = \frac{Pr(F_1 ∩ F_2)}{Pr(F_2)}$$;
- **Law of Total Probability**: $$ Pr(G) = \sum_{n=1}^{\infty} Pr(G\mid F_n) Pr(F_n) $$;
- **Bayes' Theorem**:  $$ Pr(F_j \mid G) = \frac{Pr(F_j∩G)}{P_r(G)} = \frac{Pr(F_j∩G)}{\sum_n  Pr(F_n∩G)} $$


The **cumulative distribution function (CDF)** $$F_X : \mathbb{R} → [0, 1]$$ of a random variable $X$ (or the law of $X$) is described by $$ F_X (x) = Pr(X ≤ x)$$. A distribution functions satisfies the properties:
- if $x ≤ y$ then $F_X (x) ≤ F_X (y)$;
- $$\lim x → ∞$$ $$F_X (x) = 1$$ and $$\lim x → −∞$$ $$F_X (x) = 0$$;
- $F_X (x)$ is right continuous and left limited;
- $Pr(a < X ≤ b) = F_X (b) − F_X (a)$;
- $ Pr(X > a) = 1 − F(a)$;

The **quantile function** of the random variable $X$ with the distribution function $$F_{\overline{X}} : (0, 1) → \mathbb{R}$$ is defined as $$ F_{\overline{X}}(α) = inf\{t ∈ R : F_X (t) ≥ α\} $$. The **α-quantile** of $X$ is the real number $$q_α = F_\overline{X}(α)$$.

A **continuous random variable** $X$ has **probability density function (PDF)** $f_X$ if:

$$
F_X(b) − F_X (a) = \int_a^b f_X (t) dt.
$$

A PDF satisfies:
- $$F_X (x) = \int_{-∞}^{∞}  f_X (t) dt$$,
- $$f_X (x) = F'(x)$$ whenever $f_X (x)$ is continuous,

Note that $$f_X (x) \neq Pr(X = x) = 0$$. Also $$f_X (x) > 1$$ may be possible and $$f_X (x)$$ can be unbounded.

A **discrete random variable** $X$ has a **probability mass function (PMF)** defined as $$f_X(x) = Pr(X=x)$$. $F_X (x)$ is a "stair-shaped" function i.e. piecewise constant with jumps at the points in $X$.

Instead of using random variables, we cna use the output of a function applied it to random vars. This introduces the concept of **Transformed Mass Functions$$.  Let $X$ be discrete taking values in $X$ and let $Y = g(X)$. Then $Y$ takes values in $Y = g(X)$, and :

$$
F_Y(y) = Pr(g(X) ≤ y) = \sum_{x∈X} f_X (x) I \{g(x) ≤ y\}, y ∈ Y
$$

$$
f_Y(y) = Pr(g(X) = y) = \sum_{x∈X} f_X (x)I \{g(x) = y\}, y ∈ Y
$$

### Random Vectors

A **random vector** for a fixed positive integer $d$ is $$X =(X_1, ..., X_d)^T$$ is a finite collection of random variables.
- The **joint distribution of a random vector** $X$ is defined as $$F_X(x_1, ..., x_d ) = Pr(X_1 ≤ x_1, ..., X_d ≤ x_d )$$;
- The **joint mass function** of the discrete $$\{X_i\}$$ is $$f_X(x_1, ..., x_d ) = Pr(X_1 = x_1, ..., X_d = x_d )$$;
- The **joint density functio** of $$f_X : R_d → [0, ∞)$$ (when it exists) is $$F_X(x_1, ..., x_d ) = \int_{−∞}^{x_1} ... \int_{−∞}^{x_d} f_X(u_1, ..., u_d ) du_1 ... du_d $$;
  - When $f_X(x_1, ..., x_d )$ is continuous at $x$, $$f_X(x_1, ..., x_d)= \frac{∂^d}{∂_{x_1} ... ∂_{x_d}} F_X(x_1, ..., x_d)$$;
- In the *discrete* case, the **marginal mass function** of $X_i$ is given by $$ f_{X_i} (x_i) = Pr(X_i = x_i) = \sum_{x_1} .. \sum_{x_{i-1}} \sum_{x_{i+1}} ... \sum_{x_d} f_X (x_1, ..., x_d)$$;
- In the *continuous* case, the **marginal density function** of $X_i$ is given by $$ f_{X_i} (x_i) = Pr(X_i = x_i) = \int_{-∞}^{∞} \int_{-∞}^{∞} f_X (y_1, ..., y_{i-1}, x_i, y_{i+1}, ..., y_d) d_{y_1} ... d_{y_{i-1}} d_{y_{i+1}} ... d_{y_d}$$;

