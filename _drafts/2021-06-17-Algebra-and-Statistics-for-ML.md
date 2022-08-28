---
layout: post
title:  "Algebra and Statistics for ML Engineers"
categories: [machine learning, statistics]
tags: [machinelearning]
---

All ML engineers have an exhaustive training on algebra and statistics. Because it's easy to forget some basic content after some time without practice, I decided to compile a summary of related topics. Most information is collected from the books in the <a href="{{ site.resources_permalink }}">resources</a> section.  

### Mathematics and Algebra Foundations

A brief review of algebrainc definitions and properties of matrices:
- Matrices have the properties of associativity $$(AB)C = A(BC)$$ and distributivity $$(A+B)C = AC + BC$$ and $$A(B+C)=AB+AC$$;
- **Inverse**: not every matrix $$A$$ contains an inverse $$A^{-1}$$. If it exists, $$A$$ is called regular/invertible/non-singular. Otherwise it is called singular/non-invertible;
  - $$AA^{-1} = I = A^{-1}A$$,  $$(AB)^{-1}=B^{-1}A^{-1}$$;
  - two matrics $$A$$ and $$B$$ are inverse to each other if $$AB=I=BA$$;
- **Transpose**: $${(A^T)}^{T}=A$$, $$(A+B)^T=A^T+B^T$$, $$(AB)^T=B^TA^T$$;
- **Symmetric** iff $$A=A^T$$. Thus $$A$$ is square. Also, if $$A$$ is invertible then $$A^T$$ is also invertible and $$A^T=A^{-1}$$. Sum of symmetric matrices is a symmetric matrix, but usually not their product;

A **System of Linear Equations** with equations of the type $$a_1x_1 + ...+ a_nx_n = b$$ for constants $$a_1$$ to $$a_n$$ and $$b$$ and unkownn $$x$$ can be defined as $$A x=b$$;
- We can have none, one or infinitely many solutions to such system (when there are more unknowns than equations). When no solution exists for $$Ax=b$$ we wave to resort to approximate solutions; 
- The solution represents the interception of all lines (defined by diff. equations) in a geometric representation;

The general solution of a SLE is found with **Gaussian Elimination** of the augmented matrix $$[A \mid b]$$;
- The result of the forward pass of the Gaussian Elimination puts the matrix in the **Row-Echelon** form i.e. a staircase structure;
  - A row-echelon matrix is in **reduced row-echelon** format if the leading entries of each row (the **pivot**) is 1 and the pivot is the only nonzero entry in its *column*;   
- To compute the inverse we find the matrix that satisfies $$AX=I$$, so that $$X=A^{-1}$$. We use Gaussian Elimination to solve the SLE $$[A \mid I]$$ and turn it into $$[I \mid A^{-1}]$$; 
  - When $$A$$ is square and invertible, the solution for $$Ax=b$$ is $$x=A^{-1}b$$;
  - Otherwise, $$Ax = b \Leftrightarrow A^T Ax = A^Tb \Leftrightarrow x = (A^TA)^{−1}A^Tb$$, which is also the **least-squares** solution; 
- GE is not feasible for large matrices because of its cubic computational complexity. In practice, these are solved iteratively with e.g. the Jacobi method, Richardson method, etc. The main idea is:
  - to solve $$Ax=b$$ iteratively, we set up an iteration of the form $$x^{(k+1)} = Cx^{(k)} + d$$, for a suitable $$C$$ and $$d$$ that minimized the residual error $$\mid x^{k+1}-x_* \mid$$ in every iteration and converges to $$x_*$$; 

Vector spaces: 
- the term "vector multiplication" is not defined. Theoretically, it could be an element-wise multiplication $$c_j = a_j b_j$$, or most commonly **outer product** $$ab^T$$ or **inner/scaler/dot product** $$a^Tb$$;
- a **linear combination** $$v$$ of vectors $$x_1, ..., x_n$$ is defined by the sum of a scaled set of vectors, ie $$v = \sum_{i=1}^k \lambda_i x_i \in V$$, for constants $$\lambda_i$$; 
- if there is a linear combination of vectors $$x_1, ..., x_n$$ such that $$\sum_{i=1}^k \lambda_i x_i=0$$ with all $$\lambda_i \neq 0$$, then vectors $$x$$ are **linearly dependent**. If only the trivial solution exists with all $$\lambda_i=0$$ then they are **linearly independent** ;  
  - Intuitively, a set of linearly independent vectors consists of vectors that have no redundancy, i.e., if we remove any of those vectors from
the set, we will lose something in our representation;
  - To find out if a set of vectors are linearly independent, is to write all vectors as columns of a matrix and perform GE until the row echelon form. All column vectors are linearly independent iff all columns are pivot columns; 
- for a given vector space, if every vector can be expressed as a linear combination of a set of vectors $$A=\{ x_1, ..., x_n\}$$, then $$A$$ is the **generating set** of that vector space. The set of all linear combinations of $$A$$ is called the **span** of $$A$$. Moreover, $$A$$ is called minimal if there exists no smaller set that spans V. Every linearly independent generating set of V is minimal and is called a **basis** of V;
  - a basis is a minimal generating set and a maximal linearly independent set of vectors;
  - the dimension of a vector space corresponds to the number of its basis vectors;
  - a basis of a subspace can be found by the row-echelon form of a matrix with the spanning vectors as columns. The spanning vectors of the pivot columns are *a* basis of U;  


---

### Basics of Probability

There are two frameworks for statistical modelling, the explanatory model framework and the predictive framework.

The explanatory framework starts from assuming a model to describe the observations. The predictive framework starts from assuming you can find a function $f(x)$ which maps from input $x$ to an output $f(x)$.

We cannot always model the resolution of the observed data, so we introduce **stochasticity** to our model. Stochasticity refers to the property of being well described by a random probability distribution. Although stochasticity and **randomness** are distinct in that the former refers to a modeling approach and the latter refers to phenomena themselves, these two terms are often used synonymously.  The data is viewed as observations from that model.

Probabilistic models can be caracterized  <a href="{{ site.statistics_distributions | replace: 'XXX', 'CONTINUOUS' }}"> continuous </a> and <a href="{{ site.statistics_distributions | replace: 'XXX', 'DISCRETE' }}"> discrete </a> input interval. I've covered different families of distributions in a previous [post]({{ site.baseurl }}{% post_url 2019-03-20-Exponential-Family-Distributions %}) .

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

