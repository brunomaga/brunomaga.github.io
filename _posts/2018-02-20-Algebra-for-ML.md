---
layout: post
title:  "Algebra for ML Engineers"
categories: [machine learning, algebra]
tags: [machinelearning]
---

I decided to compile a summary of topics in algebra that are relevant to ML. Most information is extracted from books in the <a href="{{ site.resources_permalink }}">resources</a> section.

### Properties of Matrices

- Matrices have the properties of associativity $$(AB)C = A(BC)$$ and distributivity $$(A+B)C = AC + BC$$ and $$A(B+C)=AB+AC$$;
- **Inverse**: not every matrix $$A$$ contains an inverse $$A^{-1}$$. If it exists, $$A$$ is called regular/invertible/non-singular. Otherwise it is called singular/non-invertible;
  - $$AA^{-1} = I = A^{-1}A$$,  $$(AB)^{-1}=B^{-1}A^{-1}$$;
  - two matrics $$A$$ and $$B$$ are inverse to each other if $$AB=I=BA$$;
- **Transpose**: $${(A^T)}^{T}=A$$, $$(A+B)^T=A^T+B^T$$, $$(AB)^T=B^TA^T$$;
- **Symmetric** iff $$A=A^T$$. Thus $$A$$ is square. Also, if $$A$$ is invertible then $$A^T$$ is also invertible and $$A^T=A^{-1}$$. Sum of symmetric matrices is a symmetric matrix, but usually not their product;


### Systems of Linear Equations

A **System of Linear Equations** with equations of the type $$a_1x_1 + ...+ a_nx_n = b$$ for constants $$a_1$$ to $$a_n$$ and $$b$$ and unkownn $$x$$ can be defined as $$A x=b$$;
- We can have none, one or infinitely many solutions to such system (when there are more unknowns than equations). When no solution exists for $$Ax=b$$ we wave to resort to approximate solutions; 
- The solution represents the interception of all lines (defined by diff. equations) in a geometric representation;

The general solution of a SLE is found with **Gaussian Elimination** of the augmented matrix $$[A \mid b]$$;
- The result of the forward pass of the Gaussian Elimination puts the matrix in the **Row-Echelon** form i.e. a staircase structure;
  - A row-echelon matrix is in **reduced row-echelon** format if the leading entries of each row (the **pivot**) is 1 and the pivot is the only nonzero entry in its *column*;   
- To compute the inverse we find the matrix that satisfies $$AX=I$$, so that $$X=A^{-1}$$. We use Gaussian Elimination to solve the SLE $$[A \mid I]$$ and turn it into $$[I \mid A^{-1}]$$; 
  - When $$A$$ is square and invertible, the solution for $$Ax=b$$ is $$x=A^{-1}b$$;
  - Otherwise, $$Ax = b \Leftrightarrow A^T Ax = A^Tb \Leftrightarrow x = (A^TA)^{−1}A^Tb$$, which is also the **least-squares** solution;
  - $$(A^TA)^{−1}A^T$$ is also called the **pseudo-inverse** of $$A$$, which can be computed for non-square matrices $$A$$. It only requires that $$A^T$$ is positive definite, which is the case if $$A$$ is full rank; 
- Gaussian Elimination is not feasible for large matrices because of its cubic computational complexity. In practice, these are solved iteratively with e.g. the Jacobi method, Richardson method, etc. The main idea is:
  - to solve $$Ax=b$$ iteratively, we set up an iteration of the form $$x^{(k+1)} = Cx^{(k)} + d$$, for a suitable $$C$$ and $$d$$ that minimized the residual error $$\mid x^{k+1}-x_* \mid$$ in every iteration and converges to $$x_*$$; 

### Vector Spaces
 
- the term "vector multiplication" is not well defined. Theoretically, it could be an element-wise multiplication $$c_j = a_j b_j$$, or most commonly **outer product** $$ab^T$$ or **inner product**, **scalar product** or **dot product** $$a^Tb$$;
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
  - a matrix has **full rank** if its rank equals the largest possible rank for a matrix of its dimensions;
  - note: **dimension** of a matrix is the number of vectors in *any* basis for the space to be spanned. Rank of a matrix is the dimension of the column space;


### Linear Mappings 

A **Linear Mapping** is a maping $$\phi: V \rightarrow W$$ such that: $$\phi (\lambda x + \psi y) = \lambda \phi(x) + \lambda \psi(y)$$, for constants $$\lambda, \psi$$ and $$x, y \in V$$.
- it can be classified as **Injective** if $$\phi(x)=\phi(y) \rightarrow x = y$$; $$\,$$ **Surjective** if $$\phi(V)=W$$; $$\,$$ **Bijective** if both;
- for $$y=Ax$$, where $$x$$ are a set coordinates, $$A$$ is the **transformation matrix** of $$x$$ into the new coordinates $$y$$;
- two matrices $$A, B \in \mathbb{R}^{m\ times n}$$ are **equivalent** if there exist regular matrices $$S \in \mathbb{R}^{n \times n}$$ and  $$S \in \mathbb{R}^{m \times m}$$, such that: $$B = T^{-1}A S$$; 
- two matrices $$A, B \in \mathbb{R}^{m\ times n}$$ are **similarent** if there exist regular matrices $$S \in \mathbb{R}^{n \times n}$$ and  $$S \in \mathbb{R}^{m \times m}$$, such that: $$B = S^{-1}A S$$. Similarity implies equivalence, no the other way around;
- for $$\phi : V \rightarrow W$$, a **kernel** is the set of vectors $$v \in V$$ that $$\phi$$ maps onto the neutral element $$0_W \in W$$, ie $$\{ v \in V : \phi(v)=0_W \}$$; 
- the **image** is the set of vectors $$w \in W$$ that can be reached by $$\phi$$ from any vector in $$V$$;
- the **kernel space** or **null space** is the solution to the SLE $$Ax=0$$;

An **affine mapping** is defined as $$x \Rightarrow a + \phi(x)$$ for linear mapping $$\phi$$ and mapping $$a$$ (the translation vector). Affine mappings keep the geometric **structure invariant**, and preserve dimension and parallelism. It's a transformation in terms of translation, scaling and shearing.

### Analytic Geometry

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

### Matrix Decompositions

- a matrix is **invertible** if determinant is not 0;
- determinant properties:
  - $det(AB) = det(A) det(B)$; 
  - $det(A) = det(A^T)$; 
  - $det(A^{-1})=\frac{1}{det(A)}$;
  - *Similar* matrices have the same determinant;
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

<p align="center"><img width="65%" height="65%" src="/assets/Algebra-for-ML/MML_eigenvalues_and_eigenvectors.png"/><br/>
<small>image source: <a href="{{ site.resources_permalink }}">Mathematics for Machine Learning book</a></small></p>

**Cholesky Decomposition**: a symmetric, positive definite matrix $$A$$ can be factorized into a product $$A = LL^T$$, where $$L$$ is a *lower-triangular matrix* with positive diagonal elements. $$L$$ is unique. This can be solved normally as a SLE.

**Diagonalizable**: a matrix $$A \in \mathbb{R}^{n \times n}$$ is diagonalizable (ie made into a diagonal matrix) if there is an invertible square matrix $$P \in \mathbb{R}^{n \times n}$$ such that $$D=P^{-1}AP$$. A symmetric matrix is always diagonalizable. The inverse of a diagonal matrix is the matrix replacing all diagonals by their reciprocal, i.e. replace $$a_{ii}$$ by $$\frac{1}{a_{ii}}$$.

**Eigendecomposition**: a matrix $$A \in \mathbb{R}^{n \times n}$$ can be factored into $$A = PDP^{-1}$$, where $$P \in \mathbb{R}^{n \times n}$$ and $$D$$ is a diagonal matrix whose diagonal entries are the eigenvalues of $$A$$, iff the eigenvectors of $$A$$ form a basis of $$\mathbb{R}^n$$.
- When this eigendecomposition exists, then $$det(A) = det(PDP^{−1}) = det(P) \, det(D) \, det(P^{−1})$$, and $$A^k = (PDP^{−1})^k = PD^kP^{−1}$$.

**Singular Value Decomposition** is a decomposition of the form $$S = U \Sigma V^T$$:
<p align="center"><img width="45%" height="45%" src="/assets/Algebra-for-ML/MML_SVD.png"/><br/>
<small>image source: <a href="{{ site.resources_permalink }}">Mathematics for Machine Learning book</a></small></p>
- applicable to all (not only square) matrices, and it always exists;
- $$\Sigma$$ is the **singular value matrix**, a (non-square) matrix with only positive diagonal entries $$\Sigma_{ii} = \sigma_{i} \ge 0$$, and zero otherwise. $$\sigma_{i}$$ are the **singular values**. $$\Sigma$$ is unique. By convention $$\sigma_{i}$$ are ordered by value with largest at $$i=0$$;
- $$U$$ is orthogonal and its column vectors $$u_i$$ are called **left-singular values**;
- $$V$$ is orthogonal and its column vectors $$v_i$$ are called **right-singular values**;
- the SVD intuition follows superficially a similar structure to our eigendecomposition intuition;
  - the left-singular vectors of $$A$$ are eigenvectors of $$AA^T$$;
  - the right-singular vectors of $$A$$ are eigenvectors of $$A^TA$$;
  - the nonzero singular values of $$A$$ are the square roots of the nonzero eigenvalues of both $$AA^T$$ and $$A^TA$$;
- computing the SVD of $$A \in \mathbb{R}^{m \times n}$$ is equivalent to finding two sets of orthonormal bases $$U = (u_1,... , u_m)$$ and $$V = (v_1,... , v_n)$$ of the codomain $$\mathbb{R}_m$$ and the domain $$\mathbb{R}_n$$, respectively. From these ordered bases, we construct the matrices $$U$$ and $$V$$;
- matrix approximation and compression can be achieving by reconstructing the original matrix using less singular values $$\sigma$$;
  - In practice, for a rankg-$$k$$ approximation: $$\hat{A(k)} = \sum_{i=1}^{k} \sigma_i u_i v_i^T = \sum_{i=1}^k \sigma_i A_i$$.

<img width="43%" height="43%" src="/assets/Algebra-for-ML/SVD_example_2.png"/> $$\,\,\,\,$$ <img width="50%" height="50%" src="/assets/Algebra-for-ML/SVD_example.png"/><br/>
<small>**Left:** intuition behind the SVD of a matrix $$A \in \mathbb{R}^{3 \times 2}$$ as a sequential transformations. TThe SVD of a matrix can be interpreted as a decomposition of a corresponding linear mapping into three operations. Top-left to bottom-left: $$V^T$$ performs a basis change in $$\mathbb{R}^2$$. Bottom-left to bottom-right: $$\Sigma$$ scales and maps from $$\mathbb{R}^2$$ to $$\mathbb{R}^3$$. The ellipse in the bottom-right lives in $$\mathbb{R}^3$$. The third dimension is orthogonal to the surface of the elliptical disk. Bottom-right to top-right: $$U$$ performs a basis change within $$\mathbb{R}^3$$. **Right**: example of application of SVD. Image sources: <a href="{{ site.resources_permalink }}">Mathematics for Machine Learning book</a> and <a href="https://scholarworks.gsu.edu/math_theses/52/">Workalemahu, Tsegaselassie, "Singular Value Decomposition in Image Noise Filtering and Reconstruction." Thesis, Georgia State University, 2008</a></small>

**Eigenvalue Decomposition vs. Singular Value Decomposition**:
- The SVD always exists for any matrix. The ED is only defined for square matrixes and only exists if we can find a bases of the eigenvectors in $$\mathbb{R}^n$$;
- in SVD, domain and codomain can be vector spaces of different dimensions;
- the vectors of $$P$$ in ED are not necessarily orthogonal. The vectors in the matrices $$U$$ and $$V$$ in the SVD are orthonormal, so they do represent rotations;

### Vector Calculus

- A **Taylor Polynomial** of degree $$n$$ of a function $$f$$ at $$x_0$$ is: $$T_n(x) = \sum_{k=0}^n \frac{f^{(k)}(x_0)}{k!}(x-x_0)^k$$, where $$f^{(k)}$$ is the $$k$$-th derivative;
  - the **Taylor series** is a representation of a function f as an infinite sum of terms. These terms are determined using derivatives of $$f$$ evaluated at $$x_0$$. It is the Taylor Polinomial of infinite order i.e. $$T_{\infty}(x)$$. if $$T_{\infty}(x)=f(x)$$ it is called **analytic**;
  - Relevance: in ML we often need to compute expectations, i.e., we need to solve integrals of the form:
$$ \mathbb{E}[f(x)] = \int f(x) p(x) dx$$;
  - Even for parametric $$p(x)$$, this integral typically cannot be solved analytically. The Taylor series expansion of $$f$$ is one way of finding an approximate solution.

- the **gradient** of **Jacobian** or $$\triangledown_x f(x)$$ of function $$f : \mathbb{R}^n \rightarrow \mathbb{R}^n$$ is the $$m \times n$$ matrix of partial derivatives per variable $$\frac{df_i(x)}{x_j}$$ for row and column iterators $$i$$ and $$j$$, respectively. Useful rules:
  - Product: $$ (f(x) g(x))' = f'(x)g(x) + f(x)g'(x)$$;
  - Quotient: $$ \left(\frac{f(x)}{g(x)}\right)' = \frac{f'(x)g(x) - f(x)g'(x)}{g(x)^2}$$;
  - Sum: $$ (f(x) + g(x))' = f'(x)+ g'(x)$$;
  - Chain: $$ (g(f(x))' = (g \circ f)'(x) = g'(f(x))f'(x) = \frac{df}{dg} \frac{dg}{dx}$$;
- the **Hessian** or $$\triangledown^2_{x,y} f(x,y)$$ is a collection of all second-order partial derivatives, defined by the symmetric matrix:
  <img width="40%" height="40%" src="/assets/Algebra-for-ML/Hessian.png"/>
 - the gradient of a function is often used to locally (linearly) approximate $$f$$ around $$x_0$$: $$f(x) ≈ f(x_0) + (∇_xf)(x_0)(x − x_0)$$. I.e. t's a Taylor series of two terms. multivariate Taylor series can be used for higher-order approximations (page 166, def 5.7 in <a href="{{ site.resources_permalink }}">MML book</a>);

- In deep neural networks, the function value $$y$$ of a $$K$$-deep DNN is computed as: $$ y = (f_K ◦ f^{K−1} ◦ ... ◦ f_1)(x) = f_K(f_{K−1}( ... (f_1(x)) ... )) $$
- The **chain rule** allows us to describe the partial derivatives as the following:  
  <img width="33%" height="33%" src="/assets/Algebra-for-ML/DNN_partial_derivatives.png"/>
  - the orange terms are partial derivatives of a layer with respect to its inputs; 
  - the blue terms are partial derivatives of a layer with respect to its parameters; 
- Backpropagation is a special case of the **automatic differentiation** algorithm, a techniques to evaluate the gradient of a function by working with intermediate variables and dependencies and applying the chain rule;

### Optimization

- **Gradient Descent**: an optimization method to minimize an $$f$$ function iteratively. For iteration $$i$$ and step-size $$\gamma$$:
  - $$x_{i+1} = x_t − γ((∇f)(x_0))^T$$, 
- **Gradient Descent with Momentum**: stores the value of the update $$\Delta x_i$$ at each iteration $$i$$ to determine the next update as a linear combination of the current and previous gradients:
   - $x_{i+1} = x_i − γ_i((∇f)(x_i))^T + α∆x_i$
   - where $$∆x_i = x_i − x_{i−1} = α∆x_{i−1} − γ_{i−1}((∇f)(x_{i−1}))^T$$ and $$\alpha \in [0,1]$$;
      - $$\alpha$$ is a hyper-parameter (user defined), close to $$1$$. If $$\alpha=0$$, this performs regular Gradient Descent'
- **Constrained gradients**: find $$min_x f(x)$$ subject to $$g_i(x) \le 0$$, for all $$i=1,...,m$$;
  - solved by converting from a constrained to an unconstrained problem of minimizing $$J$$ where $$J(x) = f(x) + \sum_{i=1}^m \mathbb{1} (g_i(x))$$; 
    - where $$1(z)$$ is an infinite step function: $$1(z)=0$$ if $$z \le 0$$, and $$1(z)=\infty$$ otherwise;
  - and replacing this step function (difficult to optimize) by **Lagrange multipliers** $$\lambda$$;
    - the new function to minimize is now $$L(x) = f(x) + \sum_{i=1}^m \lambda_i (g_i(x)) = f(x) + \lambda^T g(x)$$
  - **Lagrange duality** is the method of converting an optimization problem in one set of variables x (the **primal variables**), into another optimization problem in a different set of variables λ (**dual variables**);
    - further details in section 7.2 in <a href="{{ site.resources_permalink }}">MML book</a>:
- **Adam Optimizer** (ADAptive Moment estimation): uses estimations of the first and second moments of the gradient (the "curvature") to adapt the learning rate for each weight of the neural network;  
  - in some cases Adam doesn't converge to the optimal solution, but SGD does. According to the authors, switching to SGD in some cases show better generalizing performance than Adam alone;
  - calculates the exponential moving average of gradients and square gradients. Parameters $$\beta_1$$ and $$\beta_2$$ are used to control the decay rates of these moving averages. Adam is a combination of two gradient descent methods, Momentum, and RMSP
