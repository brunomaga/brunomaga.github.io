---
layout: post
title:  "[Draft] Statistics for ML Engineers"
categories: [machine learning, algebra]
tags: [machinelearning]
---

**Statistics** is the study of uncertainty. **Probability** is the fraction of times that an event occurs. A **sample space** is the set of all possible outcomes of the experiment. 

In discrete probability, the joint probability is given by the confusion matrix:
$$
P(X = x_i, Y = y_j) = \frac{n{ij}}{N}
$$

All ML engineers have an exhaustive training on algebra and statistics. Because it's easy to forget some basic content after some time without practice, I decided to compile a summary of related topics. Most information is collected from the books in the <a href="{{ site.resources_permalink }}">resources</a> section.  

## Mathematics and Algebra Foundations

##### Properties of Matrices

- Matrices have the properties of associativity $$(AB)C = A(BC)$$ and distributivity $$(A+B)C = AC + BC$$ and $$A(B+C)=AB+AC$$;
- **Inverse**: not every matrix $$A$$ contains an inverse $$A^{-1}$$. If it exists, $$A$$ is called regular/invertible/non-singular. Otherwise it is called singular/non-invertible;
  - $$AA^{-1} = I = A^{-1}A$$,  $$(AB)^{-1}=B^{-1}A^{-1}$$;
  - two matrics $$A$$ and $$B$$ are inverse to each other if $$AB=I=BA$$;
- **Transpose**: $${(A^T)}^{T}=A$$, $$(A+B)^T=A^T+B^T$$, $$(AB)^T=B^TA^T$$;
- **Symmetric** iff $$A=A^T$$. Thus $$A$$ is square. Also, if $$A$$ is invertible then $$A^T$$ is also invertible and $$A^T=A^{-1}$$. Sum of symmetric matrices is a symmetric matrix, but usually not their product;


##### Systems of Linear Equations

A **System of Linear Equations** with equations of the type $$a_1x_1 + ...+ a_nx_n = b$$ for constants $$a_1$$ to $$a_n$$ and $$b$$ and unkownn $$x$$ can be defined as $$A x=b$$;
- We can have none, one or infinitely many solutions to such system (when there are more unknowns than equations). When no solution exists for $$Ax=b$$ we wave to resort to approximate solutions; 
- The solution represents the interception of all lines (defined by diff. equations) in a geometric representation;

The general solution of a SLE is found with **Gaussian Elimination** of the augmented matrix $$[A \mid b]$$;
- The result of the forward pass of the Gaussian Elimination puts the matrix in the **Row-Echelon** form i.e. a staircase structure;
  - A row-echelon matrix is in **reduced row-echelon** format if the leading entries of each row (the **pivot**) is 1 and the pivot is the only nonzero entry in its *column*;   
- To compute the inverse we find the matrix that satisfies $$AX=I$$, so that $$X=A^{-1}$$. We use Gaussian Elimination to solve the SLE $$[A \mid I]$$ and turn it into $$[I \mid A^{-1}]$$; 
  - When $$A$$ is square and invertible, the solution for $$Ax=b$$ is $$x=A^{-1}b$$;
  - Otherwise, $$Ax = b \Leftrightarrow A^T Ax = A^Tb \Leftrightarrow x = (A^TA)^{−1}A^Tb$$, which is also the **least-squares** solution;
  - $$(A^TA)^{−1}A^T$$ is also called the **pseudo-inverse** of $$A$$, which can be computed for non-square matrices $$A$$. It only requires that $$A^A$$ is positive definite, which is the case if $$A$$ is full rank; 
- GE is not feasible for large matrices because of its cubic computational complexity. In practice, these are solved iteratively with e.g. the Jacobi method, Richardson method, etc. The main idea is:
  - to solve $$Ax=b$$ iteratively, we set up an iteration of the form $$x^{(k+1)} = Cx^{(k)} + d$$, for a suitable $$C$$ and $$d$$ that minimized the residual error $$\mid x^{k+1}-x_* \mid$$ in every iteration and converges to $$x_*$$; 

##### Vector Spaces
 
- the term "vector multiplication" is not defined. Theoretically, it could be an element-wise multiplication $$c_j = a_j b_j$$, or most commonly **outer product** $$ab^T$$ or **inner product**, **scalar product** or **dot product** $$a^Tb$$;
- a **linear combination** $$v$$ of vectors $$x_1, ..., x_n$$ is defined by the sum of a scaled set of vectors, ie $$v = \sum_{i=1}^k \lambda_i x_i \in V$$, for constants $$\lambda_i$$; 
- if there is a linear combination of vectors $$x_1, ..., x_n$$ such that $$\sum_{i=1}^k \lambda_i x_i=0$$ with all $$\lambda_i \neq 0$$, then vectors $$x$$ are **linearly dependent**. If only the trivial solution exists with all $$\lambda_i=0$$ then they are **linearly independent** ;  
  - Intuitively, a set of linearly independent vectors consists of vectors that have no redundancy, i.e., if we remove any of those vectors from
the set, we will lose something in our representation;
  - To find out if a set of vectors are linearly independent, is to write all vectors as columns of a matrix and perform GE until the row echelon form. All column vectors are linearly independent iff all columns are pivot columns; 
- for a given vector space, if every vector can be expressed as a linear combination of a set of vectors $$A=\{ x_1, ..., x_n\}$$, then $$A$$ is the **generating set** of that vector space. The set of all linear combinations of $$A$$ is called the **span** of $$A$$. Moreover, $$A$$ is called minimal if there exists no smaller set that spans V. Every linearly independent generating set of V is minimal and is called a **basis** of V;
  - a basis is a minimal generating set and a maximal linearly independent set of vectors;
  - the dimension of a vector space corresponds to the number of its basis vectors;
  - a basis of a vector space can be found by the row-echelon form of a matrix with the spanning vectors of the space as columns. The vector defined by the columns with a pivot are *a* basis of U;  
- the **rank** of a matrix is the number of *linearly independent* columns/rows. Thus, $$rk(A)=rk(A^T)$$;
  - a matrix $$A \in \mathbb{R}^{m \times n}$$ is invertible iff $$rk(A)=n$$;
  - a SLE $$Ax=b$$ can only be solved if $$rk(A) = rk (A \mid b)$$;
  - a matrix has **ful rank** if its rank equals the largest possible rank for a matrix of its dimensions;
  - note: **dimension** of a matrix is the number of vectors in *any* basis for the space to be spanned. Rank of a matrix is the dimension of the column space;


##### Linear Mappings 

A **Linear Mapping** is a maping $$\phi: V \rightarrow W$$ such that: $$\phi (\lambda x + \psi y) = \lambda \phi(x) + \lambda \psi(y)$$, for constants $$\lambda, \psi$$ and $$x, y \in V$$.
- it can be classified as **Injective** if $$\phi(x)=\phi(y) \rightarrow x = y$$; $$\,$$ **Surjective** if $$\phi(V)=W$$; $$\,$$ **Bijective** if both;
- for $$y=Ax$$, where $$x$$ are a set coordinates, $$A$$ is the **transformation matrix** of $$x$$ into the new coordinates $$y$$;
- two matrices $$A, B \in \mathbb{R}^{m\ times n}$$ are **equivalent** if there exist regular matrices $$S \in \mathbb{R}^{n \times n}$$ and  $$S \in \mathbb{R}^{m \times m}$$, such that: $$B = T^{-1}A S$$; 
- two matrices $$A, B \in \mathbb{R}^{m\ times n}$$ are **similarent** if there exist regular matrices $$S \in \mathbb{R}^{n \times n}$$ and  $$S \in \mathbb{R}^{m \times m}$$, such that: $$B = S^{-1}A S$$. Similarity implies equivalence, no the other way around;
- for $$\phi : V \rightarrow W$$, a **kernel** is the set of vectors $$v \in V$$ that $$\phi$$ maps onto the neutral element $$0_W \in W$$, ie $$\{ v \in V : \phi(v)=0_W \}$$; 
- the **image** is the set of vectors $$w \in W$$ that can be reached by $$\phi$$ from any vector in $$V$$;
- the **kernel space** or **null space** is the solution to the SLE $$Ax=0$$;

An **affine mapping** is defined as $$x \Rightarrow a + \phi(x)$$ for linear mapping $$\phi$$ and mapping $$a$$ (the translation vector). Affine mappings keep the geometric **structure invariant**, and preserve dimension and parallelism. It's a transformation in terms of translation, scaling and shearing.

##### Analytic Geometry

- a bilinear mapping $$\Omega : V \times V \rightarrow V$$ is called **symmetric** if order of arguments doesnt matter for the final result, **positive definite** if $$∀x \in V \setminus {0} : Ω(x, x) > 0 , Ω(0, 0) = 0$$;
- a positive, symmetric, definite, bilinear mapping  $$\Omega : V \times V \rightarrow V$$ is called an inner product on $$V$$ and is typically written as $$<x,y>$$. Symmetric, positive definite matrices are important in ML as they are defined by the inner product;
- Note: some inner products are not dot products (example in page 73 in <a href="{{ site.resources_permalink }}">MML book</a>.);
- a symmetric matrix $$A$$ that satisfies $$∀x ∈ V\setminus {0} : x^TAx > 0$$ is called **symmetric, positive definite** or just positive definite. In this case $$<x,y> = \hat{x}^TA\hat{y}$$ defines an **inner product** where $$\hat{x}, \hat{y}$$ are the coordinate representations of $$x, y \in V$$; 
- a norm of vector a vector $$x$$ is represented by $$\| x \|$$. Examples of norms: manhattan and euclidian;
- The **Length of a vector** is $$\| x \| = \sqrt{<x,x>}$$ and the **distance** between two vectors is $$d(x,y) = \| x-y \| = \sqrt{ <x-y, x-y>}$$;
- **Cauchy-Schwarz Inequality**: $$\mid <x, y> \mid \, \le \, \| x \| \, \| y \|$$;
- any **metric** of distance must be positive definite, symmetric, and respect triangle inequality: $$d(x, z) \le d(x, y) + d(y, z)$$ for all $$x, y, z ∈ V$$;
- the angle $$w$$ between two vectors is computed as $$\cos w = \frac{<x,y>}{\|x\| \|y\|}$$, where $$ <x,y> $$ is typically the dot product $$x^Ty$$;
  - if $$<x,y>=0$$, both vectors are **orthogonal**, and we write it as $$x \perp y$$. If $$\|x\|=\|y\|=1$$ they are **orthonormal**;
  - a square matrix $$A$$ is an **orthogonal matrix** iff $$AA^T = I = A^TA$$ ie $$A^{-1}=A^T$$; 
- the **inner product** of two functions $$u$$ and $$v$$ defined in $$\mathbb{R}$$ is computed as $$ <u,v> = \int_a^b u(x) v(x) dx $$ for lower and upper limits $$a, b < \infty$$; 
- a **projection** is a linear transformation $$P$$ from a vector space to itself (ie an endomorphism): $$P^2 = P \circ P = P$$, ie $$P$$ is idempotent.  

##### Matrix Decompositions

- a matrix is **invertible** if determinant is not 0;
- determinant rules: $$det(AB) = det(A) det(B)$$;\, $$det(A) = det(A^T)$$;\, $$det(A^{-1})=\frac{1}{det(A)}$$;\, *Similar* matrices have the same determinant;
- the determinant can also be computed as the product of the diagonal on a matrix in row-echelon form;
  - thus, a square matrix $$A \in \mathbb{R}^{n \times n}$$ has $$det(A) \neq 0$$ iff $$rank(A)=n$$.
  - I.e. $$A$$ is invertible iff it has full rank; 
- the **trace** of a square matrix is the sum of its diagonal terms, and the sum of its eigenvalues;
- for a square matrix $$A \in \mathbb{R}^{n \times n}$$, $$λ ∈ \mathbb{R}$$ is an **eigenvalue** of $$A$$ and $$x ∈ \mathbb{R}^n \setminus {0}$$ is the corresponding **eigenvector** of $$A$$ if  $$Ax = λx$$; 
  - to determine the eigenvalues of $$A$$, solve $$Ax = \lambda x$$ or equivalently solve $$det(A - \lambda I)=0$$ for $$x$$;
    - $$p(\lambda) = det(A - \lambda I)$$ is called the **characteristic polynomial**;
  - to determine the eigenvectors $$x_i$$, solve $$Ax_i=\lambda_i x_i$$ for each eigenvalue $$\lambda_i$$ found previously;
    - eigenvectors corresponding to distinct eigenvalues are linearly independent;
    - therefore, eigenvectors of a matrix with $$n$$ distinct eigenvectors form a basis of $$\mathbb{R}^n$$;
- a square matrix with less (linearly independent) eigenvectors than its dimension is called **defective**; 
  - Two vectors are called **codirected** if they point in the same direction and **collinear** if they point in opposite directions. All vectors that are collinear to x are also eigenvectors of A;
  - a matrix and its transpose have the same eigenvalues, but not necessarily the same eigenvectors;
  - similar matrices possess the same eigenvalues;
  - symmetric, positive definite matrices always have positive, real eigenvalues;
  - Graphical intuition: the direction of the two eigenvectors correspond to the canonical basis vectors i.e., to cardinal axes. Each axis is scaled by a factor equivalent to its eigenvalue. 
- Given a matrix  $$A \in \mathbb{R}^{m \times n}$$, we can always obtain a symmetric positive semidefinite matrix $$S=A^TA$$. If $$rk(A)=n$$, then $$A^TA$$ is symmetric positive definite;
- Note: covariance matrices are positive semidefinite :)

<p align="center"><img width="65%" height="65%" src="/assets/publications/MML_eigenvalues_and_eigenvectors.png"/><br/>
<small>image source: <a href="{{ site.resources_permalink }}">Mathematics for Machine Learning book</a></small></p>

**Cholesky Decomposition**: a symmetric, positive definite matrix $$A$$ can be factorized into a product $$A = LL^T$$, where $$L$$ is a *lower-triangular matrix* with positive diagonal elements. $$L$$ is unique. This can be solved normally as a SLE.

**Diagonalizable**: a matrix $$A \in \mathbb{R}^{n \times n}$$ is diagonalizable (ie made into a diagonal matrix) if there is an invertible square matrix $$P \in \mathbb{R}^{n \times n}$$ such that $$D=P^{-1}AP$$. A symmetric matrix is always diagonalizable.

**Eigendecomposition**: a matrix $$A \in \mathbb{R}^{n \times n}$$ can be factored into $$A = PDP^{-1}$$, where $$P \in \mathbb{R}^{n \times n}$$ and $$D$$ is a diagonal matrix whose diagonal entries are the eigenvalues of $$A$$, iff the eigenvectors of $$A$$ form a basis of $$\mathbb{R}^n$$;


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

