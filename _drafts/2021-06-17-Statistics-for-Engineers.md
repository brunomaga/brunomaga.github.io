---
layout: post
title:  "Statistics for Engineers"
categories: [machine learning, statistics]
tags: [machinelearning]
---

### Basics of Probability

There are two frameworks for statistical modelling, the explanatory model framework and the predictive framework.

The explanatory framework starts from assuming a model to describe the observations. The predictive framework starts from assuming you can find a function $f(x)$ which maps from input $x$ to an output $f(x)$.

We cannot always model the resolution of the observed data, so we introduce **stochasticity** to our model. Stochasticity refers to the property of being well described by a random probability distribution. Although stochasticity and **randomness** are distinct in that the former refers to a modeling approach and the latter refers to phenomena themselves, these two terms are often used synonymously.  The data is viewed as observations from that model.

Probabilistic models can be caracterized  <a href="{{ site.statistics_distributions | replace: 'XXX', 'CONTINUOUS' }}"> continuous </a> and <a href="{{ site.statistics_distributions | replace: 'XXX', 'DISCRETE' }}"> discrete </a> input interval. I've covered different families of distributions in a previous [post]({{ site.baseurl }}{% post_url 2019-11-20-Exponential-Family-Distributions %}) .

The outcome of an experiment, or *what is observed* is called an observation or a realisation. 
Models are modelled by a set of parameters $\theta$. So we model the distribution $F(y_1, ..., y_n; \theta)$ where $y_i \in Y$. We typically assume that $F(y_1, ..., y_n; \theta)$ is known, but $\theta$ in unknown. So we observe a realisation of $Y=(Y_1, ..., Y_b)^T \in \mathcal{Y}^n$, and use them to assert the true value of $\theta$ and quantify our uncertainty.

When $F( \cdot , \theta)$ if known, we have a **parametric** proglem. When $F( \cdot )$ is unknown the problem is **non-parametric**.
We shall model **outcomes** of experiments. The set of outcomes will be written as $\Omega$. An **event** is a subset $F \subset \Omega$ of $Omega$. An event is **realised** when the outcome of the experiment is an element of $F$. 


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

