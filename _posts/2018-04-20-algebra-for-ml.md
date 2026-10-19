---
layout: post
title:  "Mathematics for ML"
redirect_from: /algebra
categories: ["Machine Learning"]
tags: ["linear algebra", "matrix decompositions", "vector calculus", "optimization"]
---

A brief summary of algebra topics that ML engineers use on a daily basis, extracted from the books in the <a href="{{ site.resources_permalink }}">resources</a> section.


## Properties of Matrices

Matrix multiplication is associative, $$(AB)C = A(BC)$$, and distributive, $$(A+B)C = AC + BC$$ and $$A(B+C)=AB+AC$$. Division between matrices is not defined. Its equivalent is multiplication by the inverse (when it exists), i.e. $$A/B = AB^{-1}$$.

**Inverse**: not every matrix $$A$$ has an inverse $$A^{-1}$$. If the inverse exists, $$A$$ is called regular/invertible/non-singular; otherwise, it is called singular/non-invertible. The inverse satisfies $$AA^{-1} = I = A^{-1}A$$ and $$(AB)^{-1}=B^{-1}A^{-1}$$. Two matrices $$A$$ and $$B$$ are inverses of each other if $$AB=I=BA$$.

**Transpose**: $${(A^{\intercal})^{\intercal}}=A$$, $$(A+B)^{\intercal}=A^{\intercal}+B^{\intercal}$$, $$(AB)^{\intercal}=B^{\intercal}A^{\intercal}$$.

**Symmetric** iff $$A=A^{\intercal}$$, which implies that $$A$$ is square. If $$A$$ is invertible, then $$A^{\intercal}$$ is also invertible and $$(A^{\intercal})^{-1}=(A^{-1})^{\intercal}$$. The sum of symmetric matrices is symmetric, but their product usually is not.


## Systems of Linear Equations

A **System of Linear Equations** (SLE) is a set of equations of the type $$a_1x_1 + ...+ a_nx_n = b$$, with constants $$a_1$$ to $$a_n$$ and $$b$$ and unknown $$x$$. It can be written as $$A x=b$$. The general solution of an SLE is found analytically or with **Gaussian Elimination** of the augmented matrix $$[A \mid b]$$.

Such a system has no solution, exactly one solution, or infinitely many solutions (e.g. when there are more unknowns than equations). When $$Ax=b$$ has no solution, an approximate solution is used instead. Geometrically, the solution is the intersection of all the lines defined by the equations.

Analytically, when $$A$$ is square and invertible, the solution of $$Ax=b$$ is $$x=A^{-1}b$$. Otherwise, $$Ax = b \Rightarrow A^{\intercal} Ax = A^{\intercal}b$$ and, when $$A^{\intercal}A$$ is invertible:

$$
x = (A^{\intercal}A)^{−1}A^{\intercal}b
$$

which is also the solution to the **Least Squares** problem. The matrix $$(A^{\intercal}A)^{−1}A^{\intercal}$$ is called the **pseudo-inverse** of $$A$$, and it can be computed for non-square matrices $$A$$. It only requires $$A^{\intercal}A$$ to be positive definite, which holds iff $$A$$ has full column rank. Let $$\hat{x}$$ be the solution above and $$x$$ the true value. Then:

$$
\mathbb{E}[\hat{x}] = (A^{\intercal}A)^{−1}A^{\intercal} \, \mathbb{E}[b] = (A^{\intercal}A)^{−1}A^{\intercal}(Ax) = x
$$

Let $$P=(A^{\intercal}A)^{−1}A^{\intercal}$$. Then:

$$
\mathbb{Var}[\hat{x}] = \mathbb{Var}[Pb] = P\, \mathbb{Var}[b] P^{\intercal} = \sigma^2 P P^{\intercal} = \sigma^2 \, (A^{\intercal}A)^{−1}A^{\intercal} \, A (A^{\intercal}A)^{-1} = \sigma^2 \, (A^{\intercal}A)^{−1}
$$

This derivation assumes uncorrelated noise with equal variance, i.e. $$\mathbb{Var}[b] = \sigma^2 I$$. It uses $$\mathbb{Var}[Pb] = P \, \mathbb{Var}[b] P^{\intercal}$$ (formula 375 in the Matrix Cookbook) and $$A^{\intercal} \, A (A^{\intercal}A)^{-1}=I$$.

In Gaussian Elimination on $$[A \mid b]$$, the forward pass brings the matrix to **Row-Echelon** form, a staircase structure. A row-echelon matrix is in **reduced row-echelon** form if the leading entry of each row (the **pivot**) is 1 and the pivot is the only nonzero entry in its *column*.

The **inverse** of $$A$$ is the matrix $$X$$ that satisfies $$AX=I$$, so that $$X=A^{-1}$$. Gaussian Elimination on the SLE $$[A \mid I]$$ turns it into $$[I \mid A^{-1}]$$. Gaussian Elimination is not feasible for large matrices, due to its cubic computational complexity. In practice, large systems are solved iteratively, e.g. with the Jacobi or Gauss-Seidel methods. The main idea is to solve $$Ax=b$$ with an iteration of the form $$x^{(k+1)} = Cx^{(k)} + d$$, with suitable $$C$$ and $$d$$, that reduces the error $$\mid x^{(k+1)}-x_* \mid$$ in every iteration and converges to the solution $$x_*$$.


## Vector Spaces

The **norm** of a vector $$x$$ is written as $$\| x \|$$ or $$\mid x \mid$$. Examples of norms are the Manhattan and the Euclidean norms. The **length of a vector** is $$\| x \| = \sqrt{<x,x>} = \sqrt{x^{\intercal}x}$$, and the **distance** between two vectors is $$d(x,y) = \| x-y \| = \sqrt{ <x-y, x-y>}$$. The **Cauchy-Schwarz Inequality** states that $$\mid <x, y> \mid \,\, \le \,\, \| x \| \, \| y \|$$.

The term *vector multiplication* is ambiguous. It can refer to:
- an element-wise **multiplication** $$c_j = a_j b_j$$,
- the **outer product** $$ab^{\intercal}$$, or 
- the **inner product**, **scalar product** or **dot product** $$<a, b> = a \cdot b = a^{\intercal}b$$, with the geometric interpretation $$a \cdot b = \vert a \vert \, \vert b \vert \, \cos \, \theta$$, where $$\vert a \vert$$ is the magnitude (norm) of $$a$$ and $$\theta$$ is the angle between $$a$$ and $$b$$. The dot product is defined specifically for Euclidean spaces, while the inner product is defined for a generic vector space. It is commutative: $$a \cdot b = b \cdot a$$.
- the **cross product**, with the geometric interpretation $$a \times b = \vert a \vert \, \vert b \vert \, \sin \, \theta \, n$$, where $$n$$ is the unit vector perpendicular to $$a$$ and $$b$$, given by the right-hand rule. Unlike the dot product, it returns a vector instead of a scalar. It is not commutative: $$a \times b \neq b \times a$$.

A **linear combination** $$v$$ of vectors $$x_1, ..., x_k$$ is a sum of the scaled vectors, i.e. $$v = \sum_{i=1}^k \lambda_i x_i \in V$$, for constants $$\lambda_i$$. The vectors $$x_1, ..., x_k$$ are **linearly dependent** if there is a linear combination $$\sum_{i=1}^k \lambda_i x_i=0$$ with at least one $$\lambda_i \neq 0$$. If only the trivial solution exists, with all $$\lambda_i=0$$, they are **linearly independent**.

Intuitively, a set of linearly independent vectors has no redundancy: removing any vector from the set loses information in the representation. To test whether a set of vectors is linearly independent, write the vectors as columns of a matrix and perform the forward pass of Gaussian Elimination, up to the row-echelon form. All column vectors are linearly independent iff all columns are pivot columns. Otherwise, every column without a pivot is a linear combination of the pivot columns (which form a *basis* of that vector space).

Given a vector space $$V$$, if every vector can be expressed as a linear combination of a set of vectors $$A=\{ x_1, ..., x_n\}$$, then $$A$$ is a **generating set** of $$V$$. The set of all linear combinations of $$A$$ is called the **span** of $$A$$. $$A$$ is called minimal if no smaller set spans $$V$$. Every linearly independent generating set of $$V$$ is minimal and is called a **basis** of $$V$$. A basis is both a minimal generating set and a maximal linearly independent set of vectors. The dimension of a vector space is the number of its basis vectors. A basis of a vector space $$U$$ can be found from the row-echelon form of a matrix whose columns are the spanning vectors of $$U$$. The spanning vectors in the pivot columns form *a* basis of $$U$$.

The **rank** of a matrix is the number of *linearly independent* columns/rows. Thus, $$rk(A)=rk(A^{\intercal})$$. A matrix $$A \in \mathbb{R}^{n \times n}$$ is invertible iff $$rk(A)=n$$, and an SLE $$Ax=b$$ has a solution only if $$rk(A) = rk (A \mid b)$$. A matrix has **full rank** if its rank equals the largest possible rank for its dimensions; otherwise, it is **rank deficient**. Note that the **dimension** of a vector space is the number of vectors in *any* of its bases, and the rank of a matrix is the dimension of its column space.


## Linear Mapping

A **Linear Mapping** (also called linear transformation or map) is a function $$\phi: V \rightarrow W$$ between vector spaces $$V$$ and $$W$$, such that $$\phi (\lambda x + \psi y) = \lambda \phi(x) + \psi \phi(y)$$, for constants $$\lambda, \psi$$ and $$x, y \in V$$. A linear mapping can be represented as a function or as a matrix.

$$\phi$$ is called **injective** if $$\phi(x)=\phi(y) \rightarrow x = y$$; $$\,\,$$ **surjective** if $$\phi(V)=W$$; $$\,\,$$ **bijective** if both. Surjective means that every element in $$W$$ can be reached from $$V$$. Bijective means that the mapping can be undone, i.e. there is a mapping $$\Psi : W \rightarrow V$$ such that $$\Psi \circ \phi(x) = x$$. Special cases:
- **Isomorphism** $$\phi: V \rightarrow W$$ if linear and bijective. Finite-dimensional vector spaces $$V$$ and $$W$$ are isomorphic iff $$dim(V) = dim(W)$$, i.e. there is a linear, bijective mapping between any two vector spaces of the same dimension;
- **Endomorphism** $$\phi : V \rightarrow V$$ if linear;
- **Automorphism** $$\phi : V \rightarrow V$$ if linear and bijective.

For linear mappings $$Φ : V → W$$ and $$Ψ : W → X$$, the mapping $$Ψ ◦ Φ : V → X$$ is also linear. If $$Φ : V → W$$ is an isomorphism, then $$Φ^{−1} : W → V$$ is also an isomorphism.

For $$y=Ax$$, where $$x$$ is a vector of coordinates, $$A$$ is the **transformation matrix** that maps $$x$$ to the new coordinates $$y$$. In other words, a transformation matrix maps coordinates from one ordered basis to another ordered basis.

Two matrices $$A, B \in \mathbb{R}^{m \times n}$$ are **equivalent** if there exist regular matrices $$S \in \mathbb{R}^{n \times n}$$ and  $$T \in \mathbb{R}^{m \times m}$$ such that $$B = T^{-1}A S$$, i.e. one matrix can be transformed into the other by a combination of elementary row and column operations. Two matrices $$A, B \in \mathbb{R}^{n \times n}$$ are **similar** if there exists a regular matrix $$S \in \mathbb{R}^{n \times n}$$ such that $$B = S^{-1}A S$$. Thus, similar matrices are always equivalent, but the converse does not hold.

For $$\phi : V \rightarrow W$$, the **kernel** is the set of vectors $$v \in V$$ that $$\phi$$ maps onto the neutral element $$0_W \in W$$, i.e. $$\{ v \in V : \phi(v)=0_W \}$$. The **kernel space** ($$ker$$ or **null space**, a subspace of the **domain**) of a linear mapping is the solution set of the SLE $$Ax=0$$, i.e. all linear combinations of the elements in $$R^n$$ that produce $$0$$. The **image** ($$Im$$, a subspace of the **codomain**) of a linear mapping $$Φ : V → W$$ is the set of vectors $$w \in W$$ that can be reached from any vector in $$V$$. For a linear mapping $$Φ : V →W$$, it holds that $$\,\,ker(Φ) \subseteq V$$ and $$Im(Φ) \subseteq W$$.

The **Rank-Nullity theorem** states that $$dim(ker(Φ)) + dim(Im(Φ)) = dim(V)$$ for vector spaces $$V$$, $$W$$ and a linear mapping $$Φ : V →W$$.

An **affine mapping / transformation** is a linear transformation followed by a translation. It can represent translation, scaling, rotation and shearing. Affine transformations keep the structure invariant and preserve parallelism and dimension.


## Analytic Geometry

A bilinear mapping $$\Omega : V \times V \rightarrow \mathbb{R}$$ is called **symmetric** if the order of its arguments does not change the result, and **positive definite** if $$∀x \in V \setminus \{0\} : Ω(x, x) > 0 , Ω(0, 0) = 0$$.

A symmetric matrix $$A$$ is called **symmetric, positive definite**, or just positive definite, iff $$∀x ∈ V\setminus \{0\} : x^{\intercal}Ax > 0$$. If only $$\ge$$ holds, it is **positive semidefinite**. Informally, positive definite matrices are the matrix analogue of positive numbers. As an example, the covariance matrix is positive semidefinite.

A symmetric, positive definite, bilinear mapping  $$\Omega : V \times V \rightarrow \mathbb{R}$$ is called an inner product on $$V$$ and is typically written as $$<x,y>$$. If $$A$$ is s.p.d., then $$<x,y> = \hat{x}^{\intercal}A\hat{y}$$ defines an inner product, where $$\hat{x}, \hat{y}$$ are the coordinate representations of $$x, y \in V$$. If $$A \in \mathbb{R}^{n \times n}$$ is s.p.d., then:
- the null space (kernel) of $$A$$ contains only $$0$$, because $$x^{\intercal} A x \gt 0$$ for all $$x \neq 0$$. This implies $$Ax \neq 0$$ if $$x \neq 0$$;
- the diagonal elements $$a_{ii}$$ of $$A$$ are positive, because $$a_{ii} = e^{\intercal}_i A e_i > 0$$, where $$e_i$$ is the $$i$$-th vector of the standard basis in $$\mathbb{R}^n$$.

In optimization, quadratic forms $$x^{\intercal}Ax$$ of positive definite matrices are positive for any non-zero $$x$$, and they are convex. *Important:* all eigenvalues of a positive definite matrix are positive. Since the determinant is the product of the eigenvalues, the determinant of a positive definite matrix is positive. A non-zero determinant implies that an s.p.d. matrix is invertible.

A **metric** is a function $$d:V ×V → R$$, with $$(x, y) → d(x, y)$$, that satisfies the following properties:
- $$d$$ is positive definite, i.e. $$d(x, y) ⩾ 0$$ for all $$x, y ∈ V$$ and $$d(x, y) = 0 \Leftrightarrow x = y$$;
- $$d$$ is symmetric, i.e. $$d(x, y) = d(y, x)$$ for all $$x, y ∈ V$$;
- triangle inequality: $$d(x, z) ⩽ d(x, y) + d(y, z)$$ for all $$x, y, z ∈ V$$.

The angle $$w$$ between two vectors is given by $$\cos w = \frac{<x,y>}{\|x\| \|y\|}$$, where $$ <x,y> $$ is typically the dot product $$x^{\intercal}y$$. Thus:

$$
\cos w = \frac{<x,y>}{\sqrt{<x,x> <y,y>}} =  \frac{x^{\intercal}y}{\sqrt{x^{\intercal}xy^{\intercal}y}}
$$

If $$<x,y>=0$$, the two vectors are **orthogonal**, written $$x \perp y$$ (x and y are **perpendicular**). If, in addition, $$\|x\|=\|y\|=1$$, they are **orthonormal**.

A square matrix $$A$$ is an **orthogonal matrix** iff its columns are orthonormal. Then $$AA^{\intercal} = I = A^{\intercal}A$$, which implies $$A^{-1}=A^{\intercal}$$.

The **inner product** of two functions $$u : \mathbb{R} \rightarrow \mathbb{R} $$ and $$v: \mathbb{R} \rightarrow \mathbb{R}$$ is defined as $$ <u,v> = \int_a^b u(x) v(x) dx $$, with lower and upper limits $$a, b < \infty$$.

Let $$V$$ be a vector space and $$U ⊆ V$$ a subspace of $$V$$. A linear mapping $$π : V → U$$ is a **projection** if $$π^2 = π ◦ π = π$$, i.e. if $$\pi$$ is idempotent. For matrices, $$P$$ is a projection iff $$P^2 = P$$.

## Matrix Decompositions

### Determinant

Properties of the determinant:
- $$\det(AB) = \det(A) \det(B)$$; 
- $$\det(A) = \det(A^{\intercal})$$; 
- $$\det(A^{-1})=\frac{1}{\det(A)}$$;
- for matrices with more than 2 columns or rows: $$\det(A) = \sum_{k=1}^{n} (−1)^{k+j}a_{kj} \det(A_{k,j})$$.
- *Similar* matrices have the same determinant;
- a matrix $$A$$ is **invertible** if $$\det(A) \neq 0$$;
- for a diagonal matrix, the determinant is the product of the diagonal elements.

The determinant is the signed volume of the parallelepiped spanned by the columns of the matrix. It can also be computed as the product of the diagonal elements of the row-echelon form, if this form is obtained without scaling rows, with a sign change for each row swap. Important: $$A$$ is invertible *iff* it has full rank, and a square matrix $$A \in \mathbb{R}^{n \times n}$$ has $$\det(A) \neq 0$$ iff $$rank(A)=n$$.

### Trace

The **trace** of a square matrix is the sum of its diagonal elements, which equals the sum of its eigenvalues. For a square matrix $$A \in \mathbb{R}^{n \times n}$$, $$λ ∈ \mathbb{R}$$ is an **eigenvalue** of $$A$$ and $$x ∈ \mathbb{R}^n \setminus \{0\}$$ is a corresponding **eigenvector** of $$A$$ if  $$Ax = λx$$.

### Eigenvalues and eigenvectors

The **eigenvalues** of $$A$$ are found by solving $$Ax = \lambda x$$, or equivalently $$\det(A - \lambda I)=0$$, for $$\lambda$$, where $$p(\lambda) = \det(A - \lambda I)$$ is called the **characteristic polynomial**. The eigenvectors $$x_i$$ are then found by solving $$Ax_i=\lambda_i x_i$$ for each eigenvalue $$\lambda_i$$. Eigenvectors of distinct eigenvalues are linearly independent. Therefore, the eigenvectors of a matrix with $$n$$ distinct eigenvalues form *a* basis of $$\mathbb{R}^n$$. A square matrix with fewer linearly independent eigenvectors than its dimension is called **defective**.

Two vectors are **codirected** if they point in the same direction, and **collinear** if they point in the same or opposite directions. All vectors collinear to $$x$$ are also eigenvectors of $$A$$:
- non-uniqueness of eigenvectors: if $$x$$ is an eigenvector of $$A$$ with eigenvalue $$λ$$, then $$cx$$ is also an eigenvector of $$A$$ with the same eigenvalue, for any $$c ∈ \mathbb{R}\setminus\{0\}$$;
- a matrix and its transpose have the same eigenvalues, but not necessarily the same eigenvectors;
- symmetric, positive definite matrices, such as a non-singular covariance matrix, always have positive, real eigenvalues. Since the determinant is the product of the eigenvalues, it is not zero, and the matrix is invertible;
- any matrix $$A \in \mathbb{R}^{m \times n}$$ yields a symmetric positive semidefinite matrix $$S=A^{\intercal}A$$. If $$rk(A)=n$$, then $$A^{\intercal}A$$ is symmetric positive definite. By the spectral theorem, $$S$$ has eigenvectors with real eigenvalues, so $$S$$ is positive definite iff all its eigenvalues are positive.

Graphical intuition: a linear mapping scales space along the directions of its eigenvectors, each by its eigenvalue. For a diagonal matrix, the eigenvectors are the canonical basis vectors, i.e. the cardinal axes:

{: .picture}
<img width="41.1%" height="41.1%" src="{{ site.assets }}/algebra-for-ml/mml_eigenvalues_and_eigenvectors.svg"/>

{: .caption}
Determinants and eigenspaces. Overview of five linear mappings and their associated transformation matrices $$A_i \in \mathbb{R}^{2 \times 2}$$ projecting 400 color-coded points $$x \in \mathbb{R}^2$$ (left column) onto target points $$A_i x$$ (right column). The central column depicts the first eigenvector, stretched by its associated eigenvalue $$\lambda_1$$, and the second eigenvector stretched by its eigenvalue $$\lambda_2$$. Each row depicts the effect of one of five transformation matrices $$A_i$$ with respect to the standard basis, and gives $$\lambda_1$$, $$\lambda_2$$ and $$\det(A_i)$$. Adapted from: [Mathematics for Machine Learning, Figure 4.4](https://mml-book.github.io/)

### Cholesky Decomposition

A symmetric, positive definite matrix $$A$$ can be factorized into a product $$A = LL^{\intercal}$$, where $$L$$ is a *lower-triangular matrix* with positive diagonal elements. $$L$$ is unique, and its entries can be computed one at a time by expanding $$LL^{\intercal}$$ and comparing the result with $$A$$. The Cholesky decomposition is used, e.g., to sample from Gaussian distributions and to compute determinants efficiently, since $$\det(A) = \det(L) \det(L^⊤) = \det(L)^2$$.

### Diagonalizable

A matrix $$A \in \mathbb{R}^{n \times n}$$ is diagonalizable if it is similar to a diagonal matrix, i.e. if there is an invertible matrix $$P \in \mathbb{R}^{n \times n}$$ such that $$D=P^{-1}AP$$. The inverse of a diagonal matrix is obtained by replacing each diagonal element $$a_{ii}$$ by its reciprocal $$\frac{1}{a_{ii}}$$.

A matrix $$A \in \mathbb{R}^{n \times n}$$ can be factored into $$A = PDP^{-1}$$, where $$P \in \mathbb{R}^{n \times n}$$ and $$D$$ is a diagonal matrix with the eigenvalues of $$A$$ on its diagonal, iff the eigenvectors of $$A$$ form a basis of $$\mathbb{R}^n$$. When this eigendecomposition exists:

$$
\det(A) = \det(PDP^{−1}) = \det(P) \, \det(D) \, \det(P^{−1})
$$

and $$A^k = (PDP^{−1})^k = PD^kP^{−1}$$. A symmetric matrix can always be diagonalized.

### Singular Value Decomposition

The Singular Value Decomposition (SVD) decomposes a matrix as $$A = U \Sigma V^{\intercal}$$:

{: .picture}
<img width="40.7%" height="40.7%" src="{{ site.assets }}/algebra-for-ml/mml_svd.svg"/>

{: .caption}
The SVD of a matrix $$A \in \mathbb{R}^{m \times n}$$, $$A = U \Sigma V^{\intercal}$$, with the orthogonal matrices $$U \in \mathbb{R}^{m \times m}$$ and $$V \in \mathbb{R}^{n \times n}$$, and $$\Sigma \in \mathbb{R}^{m \times n}$$, whose only non-zero entries are the singular values on its diagonal. Adapted from: [Mathematics for Machine Learning, Section 4.5](https://mml-book.github.io/)

The SVD exists for every matrix, not only for square ones:
- $$\Sigma$$ is the **singular value matrix**, a (possibly non-square) matrix with non-negative diagonal entries $$\Sigma_{ii} = \sigma_{i} \ge 0$$ and zeros elsewhere. The values $$\sigma_{i}$$ are the **singular values**. $$\Sigma$$ is unique. By convention, the $$\sigma_{i}$$ are sorted by value, with the largest at $$i=1$$;
- $$U$$ is orthogonal, i.e. $$(UU^{\intercal}=U^{\intercal}U=I)$$, so its columns are orthonormal. Its column vectors $$u_i$$ are called **left-singular vectors**. The left-singular vectors of $$A$$ are the eigenbasis (i.e. the eigenvectors) of $$AA^{\intercal}$$;
- $$V$$ is orthogonal, and its column vectors $$v_i$$ are called **right-singular vectors**. The right-singular vectors of $$A$$ are the eigenvectors of $$A^{\intercal}A$$.

Computing the SVD of $$A \in \mathbb{R}^{m \times n}$$ is equivalent to finding two sets of orthonormal bases, $$U = (u_1,... , u_m)$$ and $$V = (v_1,... , v_n)$$, of the codomain $$\mathbb{R}^m$$ and the domain $$\mathbb{R}^n$$, respectively. These ordered bases form the matrices $$U$$ and $$V$$. For an s.p.d. matrix, the eigendecomposition ($$S = S^⊤ = PDP^⊤$$) and the SVD ($$S = UΣV^⊤$$) are equivalent, with $$U = P = V$$ and $$D = Σ$$.

Matrix approximation (compression) reconstructs the original matrix from fewer singular values $$\sigma$$. A rank-$$k$$ approximation is:

$$
\hat{A}(k) = \sum_{i=1}^{k} \sigma_i u_i v_i^{\intercal} = \sum_{i=1}^k \sigma_i A_i
$$

{: .picture}
<img width="39.8%" height="39.8%" src="{{ site.assets }}/algebra-for-ml/svd_example_2.svg"/> $$\,\,\,\,$$ <img width="57.9%" height="57.9%" src="{{ site.assets }}/algebra-for-ml/svd_example.svg"/>

{: .caption}
**Left:** the SVD of a matrix $$A \in \mathbb{R}^{3 \times 2}$$ as a sequence of three transformations. Top-left to bottom-left: $$V^{\intercal}$$ performs a basis change in $$\mathbb{R}^2$$, which maps the right-singular vectors $$v_1$$, $$v_2$$ to the standard basis vectors $$e_1$$, $$e_2$$. Bottom-left to bottom-right: $$\Sigma$$ scales by the singular values $$\sigma_1$$, $$\sigma_2$$ and maps from $$\mathbb{R}^2$$ to $$\mathbb{R}^3$$. The ellipse in the bottom-right lives in $$\mathbb{R}^3$$, and the third dimension is orthogonal to the surface of the elliptical disk. Bottom-right to top-right: $$U$$ performs a basis change within $$\mathbb{R}^3$$, to the left-singular vectors $$u_1$$, $$u_2$$. Top-left to top-right: $$A = U \Sigma V^{\intercal}$$ maps the unit circle directly. **Right:** an application of the SVD to image compression: (a) a $$256 \times 256$$ gray image, and its approximations with its (b) 20, (c) 50 and (d) 85 largest singular values. Adapted from: [Mathematics for Machine Learning, Figure 4.8](https://mml-book.github.io/) (left) and [T. Workalemahu, Singular Value Decomposition in Image Noise Filtering and Reconstruction, Georgia State University, 2008](https://scholarworks.gsu.edu/math_theses/52/) (right)

### Eigenvalue Decomposition vs. Singular Value Decomposition

- the SVD exists for any matrix. The ED is defined only for square matrices, and exists only if there is a basis of eigenvectors in $$\mathbb{R}^n$$;
- in the SVD, the domain and codomain can be vector spaces of different dimensions;
- the vectors of $$P$$ in the ED are not necessarily orthogonal. The vectors in the matrices $$U$$ and $$V$$ of the SVD are orthonormal, so they represent rotations, possibly combined with a reflection.


## Vector Calculus

### Derivative

The **derivative** of $$f$$ at $$x$$ is defined as the following limit, for $$h > 0$$:

$$
\frac{df}{dx} = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h}
$$

In the limit, this yields the $$tangent$$.

### Taylor series

The **Taylor series** represents a function $$f$$ as an infinite sum of terms, computed from the derivatives of $$f$$ at a single point $$x_0$$. The **Taylor Polynomial** of degree $$n$$ of $$f$$ at $$x_0$$ is:

$$
T_n(x) = \sum_{k=0}^n \frac{f^{(k)}(x_0)}{k!}(x-x_0)^k
$$

where $$f^{(k)}$$ is the $$k$$-th derivative of $$f$$. The Taylor series is the Taylor polynomial of infinite order, $$T_{\infty}(x)$$. If $$T_{\infty}(x)=f(x)$$, $$f$$ is called **analytic**.

Relevance to ML:
- ML often requires expectations, i.e. integrals of the form $$ \mathbb{E}[f(x)] = \int f(x) p(x) dx$$. Even for a parametric $$p(x)$$, this integral typically has no analytical solution. The Taylor series expansion of $$f(x)$$ gives an approximate solution $$\mathbb{E}[f(x)] \approx \mathbb{E}[T_k(x)]$$.
- in Bayesian statistics, computing the posterior often involves integrals: $$p(x \mid y) = \frac{ p(y \mid x) p(x)}{p(y)}$$, where $$p(y) = \int p(y \mid x) p(x) dx$$. In realistic problems, these integrals are often multidimensional and analytically intractable (except in a few special cases, such as with conjugate priors), so approximate inference is required.
- much of non-Bayesian statistics is based on maximum likelihood, i.e. finding the maximum of a (usually multidimensional) function. This requires its derivatives, i.e. differentiation. Numerical methods are not always required, but they are used because they are simpler (faster). In short, differentiation is more tractable than integration.
- more details in [Taylor expansions for the moments of functions of random variables](https://en.wikipedia.org/wiki/Taylor_expansions_for_the_moments_of_functions_of_random_variables).

### Newton's method

**Newton's method** is a root-finding algorithm that produces successively better approximations to the roots (or zeroes) of a real-valued function, i.e. it solves equations of the form $$f(x)=0$$ by successive approximation. The iterations are:

$$
x_{n+1}=x_{n}-{\frac {f(x_{n})}{f'(x_{n})}}
$$

The initial value $$x_0$$ should be as close as possible to the zero. The line tangent to $$y=f(x)$$ at $$x=x_0$$ crosses the $$x$$ axis at a new, improved guess $$x_1$$. A stationary point of a function (a minimum, maximum or saddle point) can be found by computing the root of its derivative, i.e. $$x_{n+1}=x_{n}-{\frac {f'(x_{n})}{f''(x_{n})}} $$. For multivariate functions, division is not defined, so it is replaced by multiplication by the inverse of the Hessian, i.e. $$x_{n+1}=x_{n}- \left(\triangledown^2 f(x_{n})\right)^{-1} \triangledown f(x_{n})$$.

{: .picture}
<img width="45%" height="45%" src="https://upload.wikimedia.org/wikipedia/commons/e/e0/NewtonIteration_Ani.gif"/>

{: .caption}
Newton's method: the tangent of $$f$$ at each guess crosses the $$x$$ axis at the next, improved guess, and the guesses converge to the zero of $$f$$. The animation numbers the initial guess $$x_1$$ (the text's $$x_0$$), and its legend reads \"Funktion\" (the function $$f$$) and \"Tangente\" (the tangent). Source: [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method)

### Gradient and Hessian

The **gradient** or **Jacobian** $$\triangledown_x f(x)$$ of a function $$f : \mathbb{R}^n \rightarrow \mathbb{R}^m$$ is the $$m \times n$$ matrix of *partial* derivatives $$\frac{\partial f_i(x)}{\partial x_j}$$, for each function $$i$$ and variable $$j$$. Useful rules of partial differentiation:

$$
\begin{align*}
& \text{Product:} & (f(x) g(x))' & = f'(x)g(x) + f(x)g'(x) \\
& \text{Quotient:} & \left(\frac{f(x)}{g(x)}\right)' & = \frac{f'(x)g(x) - f(x)g'(x)}{g(x)^2} \\
& \text{Sum:} & (f(x) + g(x))' & = f'(x)+ g'(x) \\
& \text{Chain:} & (g(f(x)))' & = (g \circ f)'(x) = g'(f(x))f'(x) = \frac{dg}{df} \frac{df}{dx}
\end{align*}
$$

The **Hessian** $$\triangledown^2_{x,y} f(x,y)$$ of a function is the matrix of all second-order partial derivatives of all variables with respect to all other variables. It is *symmetric* if the second-order partial derivatives are continuous:

$$
\mathbf{H}_f =
\begin{bmatrix}
  \dfrac{\partial^2 f}{\partial x_1^2} & \dfrac{\partial^2 f}{\partial x_1\,\partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_1\,\partial x_n} \\[2.2ex]
  \dfrac{\partial^2 f}{\partial x_2\,\partial x_1} & \dfrac{\partial^2 f}{\partial x_2^2} & \cdots & \dfrac{\partial^2 f}{\partial x_2\,\partial x_n} \\[2.2ex]
  \vdots & \vdots & \ddots & \vdots \\[2.2ex]
  \dfrac{\partial^2 f}{\partial x_n\,\partial x_1} & \dfrac{\partial^2 f}{\partial x_n\,\partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}
$$


## Deep Neural Networks

The output $$y$$ of a $$K$$-deep DNN is:

$$
y = (f_K ◦ f_{K−1} ◦ ... ◦ f_1)(x) = f_K(f_{K−1}( ... (f_1(x)) ... ))
$$

The **chain rule** gives the partial derivatives as follows:

$$
\begin{align*}
\frac{\partial L}{\partial \boldsymbol{\theta}_{K-1}} & = \frac{\partial L}{\partial \boldsymbol{f}_K} {\color{#0000FF} \frac{\partial \boldsymbol{f}_K}{\partial \boldsymbol{\theta}_{K-1}}} \\
\frac{\partial L}{\partial \boldsymbol{\theta}_{K-2}} & = \frac{\partial L}{\partial \boldsymbol{f}_K} \boxed{ {\color{#FF7300} \frac{\partial \boldsymbol{f}_K}{\partial \boldsymbol{f}_{K-1}}} {\color{#0000FF} \frac{\partial \boldsymbol{f}_{K-1}}{\partial \boldsymbol{\theta}_{K-2}}} } \\
\frac{\partial L}{\partial \boldsymbol{\theta}_{K-3}} & = \frac{\partial L}{\partial \boldsymbol{f}_K} {\color{#FF7300} \frac{\partial \boldsymbol{f}_K}{\partial \boldsymbol{f}_{K-1}}} \boxed{ {\color{#FF7300} \frac{\partial \boldsymbol{f}_{K-1}}{\partial \boldsymbol{f}_{K-2}}} {\color{#0000FF} \frac{\partial \boldsymbol{f}_{K-2}}{\partial \boldsymbol{\theta}_{K-3}}} } \\
\frac{\partial L}{\partial \boldsymbol{\theta}_i} & = \frac{\partial L}{\partial \boldsymbol{f}_K} {\color{#FF7300} \frac{\partial \boldsymbol{f}_K}{\partial \boldsymbol{f}_{K-1}}} \cdots \boxed{ {\color{#FF7300} \frac{\partial \boldsymbol{f}_{i+2}}{\partial \boldsymbol{f}_{i+1}}} {\color{#0000FF} \frac{\partial \boldsymbol{f}_{i+1}}{\partial \boldsymbol{\theta}_i}} }
\end{align*}
$$

where the orange terms are partial derivatives of a layer with respect to its inputs, and the blue terms are partial derivatives of a layer with respect to its parameters.

Backpropagation is a special case of **automatic differentiation**, a technique that evaluates the gradient of a function through intermediate variables and their dependencies, by applying the chain rule. Example: 

Consider the function

$$
f(x) = \sqrt{x^2 + \exp(x^2)} + \cos\left(x^2 + \exp(x^2)\right) \, .
$$

By application of the chain rule, and noting that differentiation is linear, we compute the gradient

$$
\begin{align*}
\frac{\mathrm{d}f}{\mathrm{d}x} & = \frac{2x + 2x \exp(x^2)}{2 \sqrt{x^2 + \exp(x^2)}} - \sin\left(x^2 + \exp(x^2)\right) \left(2x + 2x \exp(x^2)\right) \\
& = 2x \left( \frac{1}{2 \sqrt{x^2 + \exp(x^2)}} - \sin\left(x^2 + \exp(x^2)\right) \right) \left(1 + \exp(x^2)\right) \, .
\end{align*}
$$

(Source: section 5.6, <a href="{{ site.resources_permalink }}">Mathematics for Machine Learning</a>)

"Writing out the gradient in this explicit way is often impractical since it often results in a very lengthy expression for a derivative. In practice, it means that, if we are not careful, the implementation of the gradient could be significantly more expensive than computing the function, which imposes unnecessary overhead."

Automatic differentiation instead performs an efficient *forward* pass by reusing intermediate results, and breaks the *back propagation* algorithm into partial derivatives that propagate backwards through the graph:

Consider the function

$$
f(x) = \sqrt{x^2 + \exp(x^2)} + \cos\left(x^2 + \exp(x^2)\right)
$$

If we were to implement a function $$f$$ on a computer, we would be able to save some computation by using *intermediate variables*:

$$
\begin{align*}
a & = x^2 \, , \\
b & = \exp(a) \, , \\
c & = a + b \, , \\
d & = \sqrt{c} \, , \\
e & = \cos(c) \, , \\
f & = d + e \, .
\end{align*}
$$

{: .picture}
<img width="64.9%" height="64.9%" src="{{ site.assets }}/algebra-for-ml/autodiff.svg"/>

{: .caption}
Computation graph of $$f(x) = \sqrt{x^2 + \exp(x^2)} + \cos(x^2 + \exp(x^2))$$, with the intermediate variables $$a = x^2$$, $$b = \exp(a)$$, $$c = a + b$$, $$d = \sqrt{c}$$ and $$e = \cos(c)$$, and $$f = d + e$$. Adapted from: [Mathematics for Machine Learning, Example 5.14](https://mml-book.github.io/)

This is the same kind of thinking process that occurs when applying the chain rule. Note that the preceding set of equations requires fewer operations than a direct implementation of the function $$f(x)$$.


## Continuous Optimization

A stationary point is a minimum or a maximum depending on the sign of the second derivative at that point. The second derivative $$\frac{d^2f(x)}{dx^2}$$ is evaluated at all stationary points: if $$f′′(x)>0$$, the function is concave up and the point is a minimum. If $$<0$$, it is concave down and the point is a maximum.

### Gradient Descent

Gradient Descent is an iterative method to minimize a function $$f$$. For iteration $$i$$ and step size $$\gamma$$:

$$
x_{i+1} = x_i − γ((∇f)(x_i))^{\intercal}
$$

**Gradient Descent with Momentum** stores the update $$\Delta x_i$$ of each iteration $$i$$, and computes the next update as a linear combination of the current and previous gradients:

$$
x_{i+1} = x_i − γ_i((∇f)(x_i))^{\intercal} + α∆x_i
$$

where $$∆x_i = x_i − x_{i−1} = α∆x_{i−1} − γ_{i−1}((∇f)(x_{i−1}))^{\intercal}$$ and $$\alpha \in [0,1]$$. $$\alpha$$ is a user-defined hyper-parameter, close to $$1$$. With $$\alpha=0$$, the method reduces to regular Gradient Descent.

**Stochastic Gradient Descent** is a computationally cheaper and noisy approximation of gradient descent that uses only a subset of the inputs at each iteration.

### Constrained optimization

The goal is to find $$min_x f(x)$$ subject to $$g_i(x) \le 0$$, for all $$i=1,...,m$$. The constrained problem is converted into an unconstrained problem of minimizing $$J$$, where:

$$
J(x) = f(x) + \sum_{i=1}^m \mathbb{1} (g_i(x))
$$

where $$\mathbb{1}(z)$$ is an infinite step function: $$\mathbb{1}(z)=0$$ if $$z \le 0$$, and $$\mathbb{1}(z)=\infty$$ otherwise. This step function is difficult to optimize, so it is replaced by **Lagrange multipliers** $$\lambda$$. The new function to minimize is:

$$
L(x) = f(x) + \sum_{i=1}^m \lambda_i (g_i(x)) = f(x) + \lambda^{\intercal} g(x)
$$

**Lagrange duality** converts an optimization problem in one set of variables x (the **primal variables**) into another optimization problem in a different set of variables λ (the **dual variables**). Further details are in section 7.2 of the <a href="{{ site.resources_permalink }}">MML book</a>.

### Adam Optimizer

Adam (ADAptive Moment estimation) uses estimates of the first and second moments of the gradient (the second moment is the exponential moving average of the squared gradients, a per-parameter estimate of the gradient scale) to adapt the learning rate of each weight of the neural network:
- the second moment is not the second derivative (the curvature), which would be too expensive to compute;
- "The method computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients."
- in some cases, Adam does not converge to the optimal solution, while SGD does. According to Keskar and Socher (2017), switching to SGD in some cases yields better generalization performance than Adam alone;
- it computes the exponential moving averages of the gradients and of the squared gradients. Parameters $$\beta_1$$ and $$\beta_2$$ control the decay rates of these moving averages. Adam combines two gradient descent methods: Momentum and RMSP.

### Convex sets and functions

**Convex sets** are sets in which the straight line connecting any two elements lies inside the set. Similarly, a **[convex function](https://en.wikipedia.org/wiki/Convex_function)** is a function where the line segment between any two distinct points on its graph lies on or above the graph between those points. A convex function is concave up. If it is twice differentiable, its second derivative is non-negative on its entire domain. A strictly convex function has at most one global minimum.

### Linear programming

**Linear programming**, or **linear optimization**, is a method to achieve the best outcome (maximum profit or lowest cost) in a mathematical model whose requirements are linear relationships, e.g. of the form $$f(x) = ax+b$$. In algebraic notation, a linear program is: find a vector $$x$$ that maximizes/minimizes $$c^{\intercal} x$$, subject to $$Ax \le b$$ and $$x \ge 0$$, where $$A \in \mathbb{R}^{m \times d}$$ and $$b \in \mathbb{R}^m$$. Solvers include Simplex, HiGHS, etc.
