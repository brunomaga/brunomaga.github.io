---
layout: post
title:  "Mathematics for ML"
redirect_from: /algebra
categories: ["Machine Learning"]
tags: ["linear algebra", "matrix decompositions", "vector calculus", "optimization"]
---

Linear algebra, vector calculus and optimization are the daily tools of machine learning: data and parameters are vectors and matrices, and training follows the derivatives of a loss towards a minimum. The sections below summarize the results used most often, taken from the books in the <a href="{{ site.resources_permalink }}">resources</a> section, mainly *Mathematics for Machine Learning* (MML) by Deisenroth, Faisal and Ong.


## Properties of Matrices

Matrix multiplication is associative, $$(AB)C = A(BC)$$, and distributive, $$(A+B)C = AC + BC$$ and $$A(B+C) = AB + AC$$, but, unlike the multiplication of numbers, it is not commutative: in general, $$AB \neq BA$$. There is no matrix division either. The multiplication with an inverse, when it exists, takes its role, and since the order of the factors matters, $$AB^{-1}$$ and $$B^{-1}A$$ differ in general.

**Inverse**: a square matrix $$A \in \mathbb{R}^{n \times n}$$ is *invertible* (or *regular*, *non-singular*) if a matrix $$A^{-1}$$ exists with $$AA^{-1} = I = A^{-1}A$$, and *singular* otherwise. For invertible $$A$$ and $$B$$, $$(AB)^{-1} = B^{-1}A^{-1}$$: the product applies $$B$$ first, so its inverse must undo $$A$$ first.

**Transpose**: $$(A^{\intercal})^{\intercal} = A$$, $$(A+B)^{\intercal} = A^{\intercal} + B^{\intercal}$$ and $$(AB)^{\intercal} = B^{\intercal}A^{\intercal}$$, again in reverse order. If $$A$$ is invertible, so is $$A^{\intercal}$$, with $$(A^{\intercal})^{-1} = (A^{-1})^{\intercal}$$.

**Symmetric**: $$A = A^{\intercal}$$, which requires a square $$A$$. The sum of symmetric matrices is symmetric, but their product is symmetric only if they commute, since $$(AB)^{\intercal} = B^{\intercal}A^{\intercal} = BA$$.


## Systems of Linear Equations

A *system of linear equations* (SLE) is a set of $$m$$ equations $$a_{i1}x_1 + \dots + a_{in}x_n = b_i$$, for $$i = 1, \dots, m$$, with known coefficients $$a_{ij}$$ and $$b_i$$, and unknowns $$x_1, \dots, x_n$$. In matrix form, it reads $$Ax = b$$, with $$A \in \mathbb{R}^{m \times n}$$, $$x \in \mathbb{R}^n$$ and $$b \in \mathbb{R}^m$$. Geometrically, each equation defines a hyperplane, and the solutions are the points common to all the hyperplanes. Since hyperplanes can meet in a single point, share a line or a larger subspace, or have no point in common, an SLE has exactly one solution, infinitely many, or none. With fewer equations than unknowns, it has infinitely many or none.

### Gaussian Elimination

When $$A$$ is square and invertible, the solution is $$x = A^{-1}b$$. In practice, however, the system is solved directly, with [Gaussian elimination](https://en.wikipedia.org/wiki/Gaussian_elimination), which also handles systems that are not square. It simplifies the system step by step, without changing its solutions, with three elementary row operations on the augmented matrix $$[A \mid b]$$: swapping two rows, scaling a row, and adding a multiple of a row to another. The forward pass brings the matrix to *row-echelon form*, a staircase structure in which the first non-zero entry of each row, its *pivot*, lies to the right of the pivot of the row above. The backward pass then brings it to *reduced row-echelon form*, in which each pivot is 1 and is the only non-zero entry of its column, so that the solutions can be read off. Since the inverse solves $$AA^{-1} = I$$, the same elimination on $$[A \mid I]$$ gives $$[I \mid A^{-1}]$$.

On a dense $$n \times n$$ matrix, Gaussian elimination, in the form of an [LU decomposition](https://en.wikipedia.org/wiki/LU_decomposition), costs $$\frac{2}{3}n^3$$ flops and $$O(n^2)$$ memory. This dense solve is the workload of the [HPL benchmark](https://en.wikipedia.org/wiki/LINPACK_benchmarks), which ranks the supercomputers of the TOP500 list.

### Least Squares

Systems built from data have more equations than unknowns, one per sample, and because of noise, they rarely have an exact solution. The [least-squares](https://en.wikipedia.org/wiki/Least_squares) solution is then the $$x$$ that minimizes the squared error $$\Vert Ax - b \Vert^2$$. Setting the gradient to zero gives the square system of the *normal equations*, $$A^{\intercal}Ax = A^{\intercal}b$$. $$A^{\intercal}A$$ is invertible iff the columns of $$A$$ are linearly independent, and then:

$$
\hat{x} = (A^{\intercal}A)^{-1}A^{\intercal}b = A^{+}b
$$

The matrix $$A^{+} = (A^{\intercal}A)^{-1}A^{\intercal}$$ is the [Moore–Penrose pseudo-inverse](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse) of $$A$$, the counterpart of the inverse for a non-square $$A$$, which the SVD, below, extends to any matrix. If the data follow $$b = Ax + \varepsilon$$, with zero-mean noise of covariance $$\mathrm{Var}[\varepsilon] = \sigma_\varepsilon^2 I$$, i.e. uncorrelated and with equal variances, then $$\hat{x}$$ is unbiased:

$$
\mathbb{E}[\hat{x}] = A^{+} \, \mathbb{E}[b] = (A^{\intercal}A)^{-1}A^{\intercal}Ax = x
$$

and its covariance is:

$$
\mathrm{Var}[\hat{x}] = A^{+} \, \mathrm{Var}[b] \, (A^{+})^{\intercal} = \sigma_\varepsilon^2 \, (A^{\intercal}A)^{-1}A^{\intercal}A(A^{\intercal}A)^{-1} = \sigma_\varepsilon^2 \, (A^{\intercal}A)^{-1}
$$

The derivation uses $$\mathrm{Var}[Mb] = M \, \mathrm{Var}[b] \, M^{\intercal}$$ for a constant matrix $$M$$ (equation 313 of [The Matrix Cookbook]({{ site.assets }}/resources/the_matrix_cookbook.pdf)), $$\mathrm{Var}[b] = \mathrm{Var}[\varepsilon]$$, and $$(A^{+})^{\intercal} = A(A^{\intercal}A)^{-1}$$, since $$A^{\intercal}A$$ is symmetric. The uncertainty of $$\hat{x}$$ thus grows with the noise and shrinks as samples are added.

Three methods compute $$\hat{x}$$ for $$A \in \mathbb{R}^{m \times n}$$ with $$m \ge n$$, and they trade speed for accuracy:
- the normal equations, solved with a Cholesky decomposition: about $$mn^2 + \frac{1}{3}n^3$$ flops, the cheapest. However, forming $$A^{\intercal}A$$ squares the [condition number](https://en.wikipedia.org/wiki/Condition_number) $$\kappa(A)$$, the ratio of the largest to the smallest singular value of $$A$$, so the error bound grows with $$\kappa(A)^2$$, and they suit only well-conditioned problems;
- a Householder QR decomposition $$A = QR$$, followed by the triangular solve $$R\hat{x} = Q^{\intercal}b$$: about $$2mn^2 - \frac{2}{3}n^3$$ flops, up to twice as many. It is backward stable, and the standard method for full-rank problems;
- the SVD: the most expensive, and the most robust when $$A$$ is rank deficient or ill-conditioned.

### Iterative Methods

Large sparse systems, such as those of discretized PDEs, need another approach, because elimination fills in the zero entries of the factors, and a dense factorization would need $$O(n^2)$$ memory and $$O(n^3)$$ time. Iterative methods instead improve a guess $$x^{(0)}$$ step by step with matrix-vector products only, at a cost of $$O(\text{nnz})$$ per step for $$\text{nnz}$$ non-zero entries, and with $$O(n)$$ extra memory. Stationary methods, such as the [Jacobi](https://en.wikipedia.org/wiki/Jacobi_method) and [Gauss–Seidel](https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method) methods, iterate $$x^{(k+1)} = Cx^{(k)} + d$$, with $$C$$ and $$d$$ derived from $$A$$ and $$b$$. The solution $$x_*$$ is a fixed point of this iteration, $$x_* = Cx_* + d$$, so each step multiplies the error by $$C$$, $$x^{(k+1)} - x_* = C(x^{(k)} - x_*)$$, and the iteration converges from any $$x^{(0)}$$ iff the [spectral radius](https://en.wikipedia.org/wiki/Spectral_radius) of $$C$$, its largest absolute eigenvalue, is below 1. For a symmetric positive definite $$A$$, the [conjugate gradient method](https://en.wikipedia.org/wiki/Conjugate_gradient_method) converges faster: it gains a digit of accuracy every $$O(\sqrt{\kappa(A)})$$ iterations.


## Vector Spaces

The *norm* $$\Vert x \Vert$$ of a vector $$x$$ is its length. Examples are the Manhattan norm $$\Vert x \Vert_1 = \sum_i \vert x_i \vert$$, the distance along a grid of streets, and the Euclidean norm $$\Vert x \Vert_2 = \sqrt{x^{\intercal}x}$$, the length of the straight line. An inner product $$\langle x, y \rangle$$, defined in the analytic geometry section, induces the norm $$\Vert x \Vert = \sqrt{\langle x, x \rangle}$$ and the distance $$d(x, y) = \Vert x - y \Vert$$, and the dot product $$\langle x, y \rangle = x^{\intercal}y$$ induces the Euclidean norm. The [Cauchy–Schwarz inequality](https://en.wikipedia.org/wiki/Cauchy%E2%80%93Schwarz_inequality) states that $$\vert \langle x, y \rangle \vert \le \Vert x \Vert \, \Vert y \Vert$$.

The term *vector multiplication* is ambiguous: the product of two vectors $$a, b \in \mathbb{R}^n$$ can mean:
- the element-wise or [Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)) $$a \odot b$$, with entries $$a_j b_j$$;
- the outer product $$ab^{\intercal} \in \mathbb{R}^{n \times n}$$, a matrix of rank 1 for non-zero $$a$$ and $$b$$;
- the dot product $$a \cdot b = a^{\intercal}b = \Vert a \Vert \, \Vert b \Vert \cos \omega$$, a scalar, where $$\omega$$ is the angle between $$a$$ and $$b$$. It is commutative, and it is the standard inner product of $$\mathbb{R}^n$$;
- the cross product $$a \times b = \Vert a \Vert \, \Vert b \Vert \sin(\omega) \, u$$, defined for $$n = 3$$, where $$u$$ is the unit vector perpendicular to $$a$$ and $$b$$ given by the right-hand rule. It is a vector, and it is anti-commutative: $$a \times b = -(b \times a)$$.

A *linear combination* of vectors $$x_1, \dots, x_k$$ of a vector space $$V$$ is the vector $$\sum_{i=1}^k \lambda_i x_i \in V$$, for scalars $$\lambda_i$$. The vectors are *linearly dependent* if a combination with at least one $$\lambda_i \neq 0$$ gives $$0$$, and *linearly independent* if only the trivial combination, with all $$\lambda_i = 0$$, does. Intuitively, independent vectors have no redundancy: none of them is a combination of the others. Gaussian elimination tests independence: it brings the matrix with the vectors as columns to row-echelon form, and the vectors are independent iff every column has a pivot. A column without a pivot is a linear combination of the pivot columns to its left.

The *span* of a set of vectors is the set of all their linear combinations, and the set *generates* $$V$$ if its span is $$V$$. A linearly independent generating set is a *basis*, which is both a minimal generating set and a maximal linearly independent set. All bases of $$V$$ have the same number of vectors, the *dimension* $$\dim(V)$$, and the columns of a matrix at the pivot positions of its row-echelon form are a basis of the span of all its columns.

The *rank* $$\text{rk}(A)$$ of a matrix is the number of its linearly independent columns, i.e. the dimension of the span of its columns. It also equals the number of linearly independent rows, so $$\text{rk}(A) = \text{rk}(A^{\intercal})$$. $$A \in \mathbb{R}^{m \times n}$$ has *full rank* if $$\text{rk}(A) = \min(m, n)$$, and it is *rank deficient* otherwise. The rank answers the questions of the previous section: a square $$A \in \mathbb{R}^{n \times n}$$ is invertible iff $$\text{rk}(A) = n$$, and the SLE $$Ax = b$$ has a solution iff $$\text{rk}(A) = \text{rk}([A \mid b])$$, i.e. iff $$b$$ is a combination of the columns of $$A$$, by the [Rouché–Capelli theorem](https://en.wikipedia.org/wiki/Rouch%C3%A9%E2%80%93Capelli_theorem).


## Linear Mappings

A *linear mapping* $$\Phi: V \rightarrow W$$ between vector spaces $$V$$ and $$W$$ preserves their structure: it maps sums to sums and scaled vectors to scaled vectors, $$\Phi(\lambda x + \psi y) = \lambda \Phi(x) + \psi \Phi(y)$$, for all $$x, y \in V$$ and scalars $$\lambda, \psi$$. For finite-dimensional spaces with fixed bases, every linear mapping is a matrix: the *transformation matrix* $$A_\Phi$$ maps the coordinates of $$x$$ in the basis of $$V$$ to the coordinates of $$\Phi(x)$$ in the basis of $$W$$.

$$\Phi$$ is *injective* if different inputs give different outputs, $$\Phi(x) = \Phi(y) \Rightarrow x = y$$, *surjective* if it reaches every element of $$W$$, $$\Phi(V) = W$$, and *bijective* if it is both. A bijective mapping can be undone: it has an inverse $$\Phi^{-1}: W \rightarrow V$$, with $$\Phi^{-1}(\Phi(x)) = x$$. Special cases are:
- *isomorphism*: $$\Phi: V \rightarrow W$$ linear and bijective. Finite-dimensional $$V$$ and $$W$$ are isomorphic iff $$\dim(V) = \dim(W)$$;
- *endomorphism*: $$\Phi: V \rightarrow V$$ linear;
- *automorphism*: $$\Phi: V \rightarrow V$$ linear and bijective.

The composition of linear mappings is linear, and the inverse of an isomorphism is an isomorphism.

Two matrices $$A, B \in \mathbb{R}^{m \times n}$$ are *equivalent* if $$B = T^{-1}AS$$ for invertible $$S \in \mathbb{R}^{n \times n}$$ and $$T \in \mathbb{R}^{m \times m}$$. They then represent the same linear mapping in other bases of $$V$$ and $$W$$, and elementary row and column operations turn one into the other. Two square matrices are [similar](https://en.wikipedia.org/wiki/Matrix_similarity) if $$B = S^{-1}AS$$: they represent the same endomorphism in another basis. Similar matrices are thus equivalent, but not the reverse, and since they describe the same mapping, they have the same rank, determinant, trace and eigenvalues.

The *kernel* (null space) $$\ker(\Phi) = \{x \in V : \Phi(x) = 0\}$$ is the set of inputs that the mapping sends to zero, a subspace of the domain $$V$$. For a matrix, it is the solution set of $$Ax = 0$$: the coefficients of all combinations of the columns of $$A$$ that give $$0$$. The *image* $$\text{Im}(\Phi) = \{\Phi(x) : x \in V\}$$ is the set of all outputs, a subspace of the codomain $$W$$, and for a matrix, it is the span of the columns, of dimension $$\text{rk}(A)$$. The [rank–nullity theorem](https://en.wikipedia.org/wiki/Rank%E2%80%93nullity_theorem) relates the two:

$$
\dim(\ker(\Phi)) + \dim(\text{Im}(\Phi)) = \dim(V)
$$

Each dimension of the input thus either survives in the image or is lost in the kernel, so a mapping to a space of lower dimension, $$\dim(W) < \dim(V)$$, always has a non-trivial kernel: different inputs give the same output.

An [affine mapping](https://en.wikipedia.org/wiki/Affine_transformation) $$x \mapsto Ax + t$$ is a linear mapping followed by a translation by $$t$$. It represents translations, scalings, rotations and shears. With an invertible $$A$$, it maps lines to lines and keeps parallel lines parallel, but it changes lengths and angles in general. A fully connected layer of a neural network, before its activation function, is an affine mapping, with the weight matrix as $$A$$ and the bias as $$t$$.


## Analytic Geometry

Lengths, distances and angles all derive from the inner product. A *bilinear mapping* $$\Omega: V \times V \rightarrow \mathbb{R}$$ is linear in each argument. It is *symmetric* if the order of its arguments does not matter, $$\Omega(x, y) = \Omega(y, x)$$, and *positive definite* if $$\Omega(x, x) > 0$$ for all $$x \neq 0$$. A symmetric, positive definite bilinear mapping is an *inner product* $$\langle x, y \rangle$$.

A symmetric matrix $$A \in \mathbb{R}^{n \times n}$$ is [positive definite](https://en.wikipedia.org/wiki/Definite_matrix), or s.p.d., if $$x^{\intercal}Ax > 0$$ for all $$x \neq 0$$, and *positive semidefinite* if $$x^{\intercal}Ax \ge 0$$. Positive definite matrices are the matrix analogue of positive numbers. The covariance matrix of a random vector $$x$$, for example, is positive semidefinite, since $$w^{\intercal} \, \mathrm{Var}[x] \, w = \mathrm{Var}[w^{\intercal}x] \ge 0$$ for any $$w$$.

For an s.p.d. $$A$$, $$\langle x, y \rangle = \hat{x}^{\intercal}A\hat{y}$$ is an inner product, where $$\hat{x}$$ and $$\hat{y}$$ are the coordinates of $$x$$ and $$y$$ in a basis of $$V$$, and every inner product on a finite-dimensional space has this form. An s.p.d. matrix:
- has a kernel that holds only $$0$$, since $$x^{\intercal}Ax > 0$$ implies $$Ax \neq 0$$ for $$x \neq 0$$. It is therefore invertible;
- has positive diagonal entries, $$a_{ii} = e_i^{\intercal}Ae_i > 0$$, where $$e_i$$ is the $$i$$-th vector of the standard basis;
- defines a quadratic form $$x^{\intercal}Ax$$ that is strictly convex, with its only minimum at $$x = 0$$.

A *metric* $$d: V \times V \rightarrow \mathbb{R}$$ formalizes the distance. It is:
- positive definite: $$d(x, y) \ge 0$$, and $$d(x, y) = 0 \Leftrightarrow x = y$$;
- symmetric: $$d(x, y) = d(y, x)$$;
- subject to the triangle inequality: $$d(x, z) \le d(x, y) + d(y, z)$$.

The distance $$\Vert x - y \Vert$$ induced by a norm is a metric. The angle $$\omega$$ between two vectors satisfies:

$$
\cos \omega = \frac{\langle x, y \rangle}{\Vert x \Vert \, \Vert y \Vert} = \frac{x^{\intercal}y}{\sqrt{x^{\intercal}x \; y^{\intercal}y}}
$$

where the second form uses the dot product. The Cauchy–Schwarz inequality keeps this ratio between $$-1$$ and $$1$$. In ML, it is the [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity), which compares embeddings by their direction and ignores their length. $$x$$ and $$y$$ are *orthogonal*, or perpendicular, $$x \perp y$$, if $$\langle x, y \rangle = 0$$, and *orthonormal* if, in addition, $$\Vert x \Vert = \Vert y \Vert = 1$$.

A square matrix $$Q$$ is an *orthogonal matrix* iff its columns are orthonormal. Then $$Q^{\intercal}Q = I = QQ^{\intercal}$$, so its inverse is its transpose, $$Q^{-1} = Q^{\intercal}$$. It preserves lengths and angles, $$\Vert Qx \Vert = \Vert x \Vert$$: it is a rotation, possibly combined with a reflection.

The inner product of two functions $$u, v: \mathbb{R} \rightarrow \mathbb{R}$$ on an interval $$[a, b]$$ is $$\langle u, v \rangle = \int_a^b u(x)v(x) \, dx$$, the continuous counterpart of the dot product, with an integral in place of the sum. For example, $$\sin$$ and $$\cos$$ are orthogonal on $$[-\pi, \pi]$$.

A linear mapping $$\pi: V \rightarrow U$$ onto a subspace $$U \subseteq V$$ is a *projection* if $$\pi^2 = \pi \circ \pi = \pi$$, i.e. if it is idempotent: projecting a second time changes nothing. A matrix $$P_\pi$$ is a projection iff $$P_\pi^2 = P_\pi$$. The orthogonal projection onto the span of the columns of $$A$$, for linearly independent columns, is $$P_\pi = A(A^{\intercal}A)^{-1}A^{\intercal}$$. This brings back least squares: the fit $$A\hat{x} = P_\pi b$$ is the orthogonal projection of $$b$$ onto the image of $$A$$, the point of the image closest to $$b$$, and the residual $$b - A\hat{x}$$ is orthogonal to that image.


## Matrix Decompositions

### Determinant

The *determinant* $$\det(A)$$ of a square matrix is the signed volume of the parallelepiped spanned by its columns, i.e. the signed factor by which the mapping scales volumes. Its properties are:
- $$\det(AB) = \det(A)\det(B)$$, $$\det(A^{\intercal}) = \det(A)$$ and $$\det(A^{-1}) = 1/\det(A)$$;
- similar matrices have the same determinant;
- $$A \in \mathbb{R}^{n \times n}$$ is invertible iff $$\det(A) \neq 0$$, iff $$\text{rk}(A) = n$$. A zero determinant means that the mapping flattens space into fewer dimensions, which cannot be undone;
- the determinant of a triangular matrix, such as a diagonal one, is the product of its diagonal entries.

The [Laplace expansion](https://en.wikipedia.org/wiki/Laplace_expansion) along column $$j$$ computes it recursively, $$\det(A) = \sum_{k=1}^{n} (-1)^{k+j} a_{kj} \det(A_{k,j})$$, where $$A_{k,j}$$ is $$A$$ without row $$k$$ and column $$j$$, but at a cost of $$O(n!)$$ operations. Gaussian elimination needs only $$O(n^3)$$: adding a multiple of a row to another leaves the determinant unchanged, and a row swap flips its sign, so without row scaling, the determinant is the product of the diagonal entries of the row-echelon form, up to these sign changes.

### Trace

The *trace* $$\text{tr}(A)$$ of a square matrix is the sum of its diagonal entries. It is linear, and invariant under cyclic permutations of a product, $$\text{tr}(ABC) = \text{tr}(BCA) = \text{tr}(CAB)$$, which often simplifies derivations. For example, $$x^{\intercal}Ax = \text{tr}(Axx^{\intercal})$$.

### Eigenvalues and Eigenvectors

$$\lambda$$ is an [eigenvalue](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors) of $$A \in \mathbb{R}^{n \times n}$$, and $$x \in \mathbb{R}^n \setminus \{0\}$$ a corresponding *eigenvector*, if $$Ax = \lambda x$$: instead of changing the direction of an eigenvector, the matrix only scales it. The equation $$(A - \lambda I)x = 0$$ has a non-zero solution only if $$A - \lambda I$$ is singular, so the eigenvalues are the roots of the *characteristic polynomial* $$p(\lambda) = \det(A - \lambda I)$$, and they can be complex for a real $$A$$. The eigenvectors of $$\lambda_i$$ are the non-zero vectors of $$\ker(A - \lambda_i I)$$. They are not unique: every vector $$cx$$ collinear to an eigenvector $$x$$, for $$c \neq 0$$, is an eigenvector too. Other properties:
- eigenvectors of distinct eigenvalues are linearly independent, so a matrix with $$n$$ distinct eigenvalues has a basis of eigenvectors. A matrix with fewer than $$n$$ linearly independent eigenvectors is *defective*;
- $$A$$ and $$A^{\intercal}$$ have the same eigenvalues, but not necessarily the same eigenvectors;
- $$\det(A)$$ is the product of the eigenvalues, and $$\text{tr}(A)$$ is their sum;
- by the [spectral theorem](https://en.wikipedia.org/wiki/Spectral_theorem), a symmetric matrix has real eigenvalues and an orthonormal basis of eigenvectors. It is positive definite iff all its eigenvalues are positive, and then $$\det(A) > 0$$;
- for any $$A \in \mathbb{R}^{m \times n}$$, $$A^{\intercal}A$$ is symmetric and positive semidefinite, since $$x^{\intercal}A^{\intercal}Ax = \Vert Ax \Vert^2 \ge 0$$. It is positive definite iff $$\text{rk}(A) = n$$.

Numerical libraries do not solve $$p(\lambda) = 0$$, since the roots of a polynomial are sensitive to rounding errors in its coefficients. The [QR algorithm](https://en.wikipedia.org/wiki/QR_algorithm) computes all eigenvalues in $$O(n^3)$$. [Power iteration](https://en.wikipedia.org/wiki/Power_iteration) and the [Lanczos algorithm](https://en.wikipedia.org/wiki/Lanczos_algorithm) compute a few extreme eigenvalues with matrix-vector products only. They suit large sparse matrices, and problems that need only the leading eigenvectors, such as PCA.

The picture below shows the geometric intuition, with five matrices $$A_i \in \mathbb{R}^{2 \times 2}$$ applied to 400 points. A matrix with a basis of real eigenvectors stretches space along each eigenvector by its eigenvalue, and it scales areas by its determinant. The first matrix is diagonal, so its eigenvectors are the axes: it stretches the vertical axis by 2 and halves the horizontal one, which keeps the area, with a determinant of 1. The second, a shear, is defective, with a single eigenvector direction, and the third, a rotation, has complex eigenvalues, since it changes the direction of every vector. The fourth has the eigenvalue 0 and maps the plane onto a line, so its determinant is 0. The fifth is symmetric, with orthogonal eigenvectors.

{: .picture}
<img width="41.1%" height="41.1%" src="{{ site.assets }}/algebra-for-ml/mml_eigenvalues_and_eigenvectors.svg"/>

{: .caption}
Determinants and eigenspaces. Overview of five linear mappings and their associated transformation matrices $$A_i \in \mathbb{R}^{2 \times 2}$$ projecting 400 color-coded points $$x \in \mathbb{R}^2$$ (left column) onto target points $$A_i x$$ (right column). The central column depicts the first eigenvector, stretched by its associated eigenvalue $$\lambda_1$$, and the second eigenvector stretched by its eigenvalue $$\lambda_2$$. Each row depicts the effect of one of five transformation matrices $$A_i$$ with respect to the standard basis, and gives $$\lambda_1$$, $$\lambda_2$$ and $$\det(A_i)$$. Adapted from: [Mathematics for Machine Learning, Figure 4.4](https://mml-book.github.io/)

### Cholesky Decomposition

An s.p.d. matrix has a square root of sorts: it factorizes uniquely as $$A = LL^{\intercal}$$, where $$L$$ is lower triangular with positive diagonal entries. The entries of this [Cholesky decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition) can be computed one at a time, by expanding $$LL^{\intercal}$$ and comparing the result with $$A$$. Since it exploits the symmetry of $$A$$, it costs $$\frac{1}{3}n^3$$ flops, half the cost of an LU decomposition, and it solves s.p.d. systems such as the normal equations. It also samples from a Gaussian $$\mathcal{N}(\mu, A)$$, as $$\mu + Lz$$ with $$z \sim \mathcal{N}(0, I)$$, because the covariance of $$Lz$$ is $$LL^{\intercal} = A$$, and it gives the determinant $$\det(A) = \det(L)^2 = \prod_i L_{ii}^2$$.

### Eigendecomposition

Diagonal matrices are easy to work with: their powers, inverse and determinant involve only the diagonal entries, and the inverse, for example, replaces each diagonal entry $$a_{ii}$$ by $$1/a_{ii}$$. A matrix $$A \in \mathbb{R}^{n \times n}$$ is *diagonalizable* if it is similar to a diagonal matrix, $$D = P^{-1}AP$$, for an invertible $$P$$. This holds iff the eigenvectors of $$A$$ form a basis of $$\mathbb{R}^n$$, and then $$A = PDP^{-1}$$ is the *eigendecomposition* of $$A$$, with the eigenvectors as the columns of $$P$$ and the eigenvalues in $$D$$. Powers then become $$A^k = PD^kP^{-1}$$, since the inner factors $$P^{-1}P$$ cancel; when no eigenvalue is 0, the inverse is $$A^{-1} = PD^{-1}P^{-1}$$; and the determinant is $$\det(A) = \det(P)\det(D)\det(P^{-1}) = \det(D)$$, the product of the eigenvalues. A symmetric matrix is always diagonalizable, with an orthogonal $$P$$: $$A = PDP^{\intercal}$$.

### Singular Value Decomposition

The eigendecomposition exists only for some square matrices. The [singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) (SVD) has no such limit: it factorizes any matrix $$A \in \mathbb{R}^{m \times n}$$ as $$A = U \Sigma V^{\intercal}$$:

{: .picture}
<img width="40.7%" height="40.7%" src="{{ site.assets }}/algebra-for-ml/mml_svd.svg"/>

{: .caption}
The SVD of a matrix $$A \in \mathbb{R}^{m \times n}$$, $$A = U \Sigma V^{\intercal}$$, with the orthogonal matrices $$U \in \mathbb{R}^{m \times m}$$ and $$V \in \mathbb{R}^{n \times n}$$, and $$\Sigma \in \mathbb{R}^{m \times n}$$, whose only non-zero entries are the singular values on its diagonal. Adapted from: [Mathematics for Machine Learning, Section 4.5](https://mml-book.github.io/)

- $$U \in \mathbb{R}^{m \times m}$$ is orthogonal. Its columns $$u_i$$, the *left-singular vectors*, are eigenvectors of $$AA^{\intercal}$$;
- $$V \in \mathbb{R}^{n \times n}$$ is orthogonal. Its columns $$v_i$$, the *right-singular vectors*, are eigenvectors of $$A^{\intercal}A$$;
- $$\Sigma \in \mathbb{R}^{m \times n}$$ is zero except for its diagonal, which holds the *singular values* $$\sigma_1 \ge \sigma_2 \ge \dots \ge 0$$. The non-zero $$\sigma_i^2$$ are the non-zero eigenvalues of $$A^{\intercal}A$$ and of $$AA^{\intercal}$$. $$\Sigma$$ is unique; $$U$$ and $$V$$ are not.

The SVD thus finds orthonormal bases of the domain $$\mathbb{R}^n$$ and of the codomain $$\mathbb{R}^m$$ in which $$A$$ maps each $$v_i$$ onto a multiple of $$u_i$$, $$Av_i = \sigma_i u_i$$. The left part of the picture below follows the unit circle through the three factors: $$V^{\intercal}$$ turns the right-singular vectors into the standard basis, $$\Sigma$$ stretches the circle into an ellipse in $$\mathbb{R}^3$$, and $$U$$ turns the ellipse into place. For an s.p.d. matrix, the SVD and the eigendecomposition $$A = PDP^{\intercal}$$ coincide, with $$U = V = P$$ and $$\Sigma = D$$. The SVD also gives the pseudo-inverse of any matrix, $$A^{+} = V \Sigma^{+} U^{\intercal}$$, where $$\Sigma^{+} \in \mathbb{R}^{n \times m}$$ is the transpose of $$\Sigma$$ with each non-zero $$\sigma_i$$ replaced by $$1/\sigma_i$$.

The SVD writes $$A$$ as a sum of rank-1 matrices $$A_i = u_i v_i^{\intercal}$$, weighted by the singular values, and the terms with a small $$\sigma_i$$ contribute little. Keeping the $$k$$ largest singular values thus gives a compressed rank-$$k$$ approximation of $$A$$:

$$
\hat{A}(k) = \sum_{i=1}^{k} \sigma_i u_i v_i^{\intercal} = \sum_{i=1}^{k} \sigma_i A_i
$$

By the Eckart–Young theorem, it is the best [rank-$$k$$ approximation](https://en.wikipedia.org/wiki/Low-rank_approximation) of $$A$$ in the spectral and Frobenius norms, with a spectral-norm error $$\Vert A - \hat{A}(k) \Vert_2 = \sigma_{k+1}$$, and it stores $$k(m + n + 1)$$ numbers instead of $$mn$$. For the $$256 \times 256$$ image on the right of the picture below, 20 singular values take 16% of the storage, 50 take 39% and 85 take 67%.

{: .picture}
<img width="39.8%" height="39.8%" src="{{ site.assets }}/algebra-for-ml/svd_example_2.svg"/> $$\,\,\,\,$$ <img width="57.9%" height="57.9%" src="{{ site.assets }}/algebra-for-ml/svd_example.svg"/>

{: .caption}
**Left:** the SVD of a matrix $$A \in \mathbb{R}^{3 \times 2}$$ as a sequence of three transformations. Top-left to bottom-left: $$V^{\intercal}$$ performs a basis change in $$\mathbb{R}^2$$, which maps the right-singular vectors $$v_1$$, $$v_2$$ to the standard basis vectors $$e_1$$, $$e_2$$. Bottom-left to bottom-right: $$\Sigma$$ scales by the singular values $$\sigma_1$$, $$\sigma_2$$ and maps from $$\mathbb{R}^2$$ to $$\mathbb{R}^3$$. The ellipse in the bottom-right lives in $$\mathbb{R}^3$$, and the third dimension is orthogonal to the surface of the elliptical disk. Bottom-right to top-right: $$U$$ performs a basis change within $$\mathbb{R}^3$$, to the left-singular vectors $$u_1$$, $$u_2$$. Top-left to top-right: $$A = U \Sigma V^{\intercal}$$ maps the unit circle directly. **Right:** an application of the SVD to image compression: (a) a $$256 \times 256$$ gray image, and its approximations with its (b) 20, (c) 50 and (d) 85 largest singular values. Adapted from: [Mathematics for Machine Learning, Figure 4.8](https://mml-book.github.io/) (left) and [T. Workalemahu, Singular Value Decomposition in Image Noise Filtering and Reconstruction, Georgia State University, 2008](https://scholarworks.gsu.edu/math_theses/52/) (right)

### Eigenvalue Decomposition vs. Singular Value Decomposition

The two decompositions differ as follows:
- The SVD exists for any matrix. The eigendecomposition exists only for square matrices with a basis of eigenvectors.
- In the SVD, the domain and the codomain can have different dimensions.
- The eigenvectors in $$P$$ are not orthogonal in general, so $$P$$ is a general change of basis. $$U$$ and $$V$$ are orthogonal: rotations, possibly combined with reflections.
- Eigenvalues can be negative or complex. Singular values are real and non-negative.
- Both cost $$O(n^3)$$ for an $$n \times n$$ matrix. The SVD of an $$m \times n$$ matrix costs $$O(mn \min(m, n))$$.

The eigendecomposition is therefore the tool for powers of a square matrix, such as $$A^k$$, and the SVD the tool for rectangular data matrices, low-rank approximations and least squares.


## Vector Calculus

### Derivative

The derivative of $$f: \mathbb{R} \rightarrow \mathbb{R}$$ at $$x$$ is the limit of the slope between $$x$$ and a nearby point $$x + h$$:

$$
\frac{df}{dx} = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h}
$$

In the limit, the line through the two points becomes the tangent to the graph of $$f$$ at $$x$$, so the derivative is the slope of the tangent.

### Gradient, Jacobian and Hessian

A function of several variables has a partial derivative for each variable, which keeps the other variables fixed. The *gradient* of $$f: \mathbb{R}^n \rightarrow \mathbb{R}$$ is the column vector of its partial derivatives, $$\nabla f(x) = \left[ \frac{\partial f}{\partial x_1}, \dots, \frac{\partial f}{\partial x_n} \right]^{\intercal}$$, and it points in the direction of steepest ascent of $$f$$. The [Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) of $$f: \mathbb{R}^n \rightarrow \mathbb{R}^m$$ is the $$m \times n$$ matrix $$J_f = \frac{\partial f}{\partial x}$$, with entries $$\frac{\partial f_i}{\partial x_j}$$, and for $$m = 1$$, $$J_f = \nabla f^{\intercal}$$. The rules of differentiation, for $$f, g: \mathbb{R} \rightarrow \mathbb{R}$$, are:

$$
\begin{align*}
& \text{Product:} & (f(x) g(x))' & = f'(x)g(x) + f(x)g'(x) \\
& \text{Quotient:} & \left(\frac{f(x)}{g(x)}\right)' & = \frac{f'(x)g(x) - f(x)g'(x)}{g(x)^2} \\
& \text{Sum:} & (f(x) + g(x))' & = f'(x)+ g'(x) \\
& \text{Chain:} & (g(f(x)))' & = (g \circ f)'(x) = g'(f(x))f'(x) = \frac{dg}{df} \frac{df}{dx}
\end{align*}
$$

For vector functions, the chain rule multiplies Jacobians: $$J_{g \circ f}(x) = J_g(f(x)) \, J_f(x)$$.

The second-order derivatives describe the curvature of $$f$$. The [Hessian](https://en.wikipedia.org/wiki/Hessian_matrix) $$\mathbf{H}_f \in \mathbb{R}^{n \times n}$$ of $$f: \mathbb{R}^n \rightarrow \mathbb{R}$$ holds all of them, and it is symmetric if they are continuous, by [Schwarz's theorem](https://en.wikipedia.org/wiki/Symmetry_of_second_derivatives):

$$
\mathbf{H}_f =
\begin{bmatrix}
  \dfrac{\partial^2 f}{\partial x_1^2} & \dfrac{\partial^2 f}{\partial x_1\,\partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_1\,\partial x_n} \\[2.2ex]
  \dfrac{\partial^2 f}{\partial x_2\,\partial x_1} & \dfrac{\partial^2 f}{\partial x_2^2} & \cdots & \dfrac{\partial^2 f}{\partial x_2\,\partial x_n} \\[2.2ex]
  \vdots & \vdots & \ddots & \vdots \\[2.2ex]
  \dfrac{\partial^2 f}{\partial x_n\,\partial x_1} & \dfrac{\partial^2 f}{\partial x_n\,\partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}
$$

### Taylor Series

The [Taylor series](https://en.wikipedia.org/wiki/Taylor_series) approximates a function $$f$$ near a point $$x_0$$ by a polynomial built from the derivatives of $$f$$ at $$x_0$$. The Taylor polynomial of degree $$q$$ is:

$$
T_q(x) = \sum_{j=0}^{q} \frac{f^{(j)}(x_0)}{j!}(x-x_0)^j
$$

where $$f^{(j)}$$ is the $$j$$-th derivative of $$f$$. Each term matches one more derivative of $$f$$ at $$x_0$$, so the approximation is good near $$x_0$$. The Taylor series is the limit $$T_{\infty}$$, and $$f$$ is *analytic* at $$x_0$$ if $$T_{\infty}(x) = f(x)$$ for all $$x$$ near $$x_0$$. For $$f: \mathbb{R}^n \rightarrow \mathbb{R}$$, the second-order expansion uses the gradient and the Hessian:

$$
f(x) \approx f(x_0) + \nabla f(x_0)^{\intercal}(x - x_0) + \frac{1}{2}(x - x_0)^{\intercal} \, \mathbf{H}_f(x_0) \, (x - x_0)
$$

Polynomials are easy to integrate and to optimize, which makes the expansion useful in ML:
- expectations $$\mathbb{E}[f(x)] = \int f(x) p(x) \, dx$$ often have no closed form. Around the mean $$\mu$$ of $$x$$, the linear term of the expansion has zero mean, and the second order gives $$\mathbb{E}[f(x)] \approx f(\mu) + \frac{1}{2} f''(\mu) \, \mathrm{Var}[x]$$. More details are in [Taylor expansions for the moments of functions of random variables](https://en.wikipedia.org/wiki/Taylor_expansions_for_the_moments_of_functions_of_random_variables);
- the posterior $$p(\theta \mid \mathcal{D}) = p(\mathcal{D} \mid \theta) \, p(\theta) / p(\mathcal{D})$$ needs the evidence $$p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta) \, p(\theta) \, d\theta$$, an integral with a closed form only in special cases, such as conjugate priors. The [Laplace approximation](https://en.wikipedia.org/wiki/Laplace%27s_approximation) expands $$\log p(\theta \mid \mathcal{D})$$ to second order around its mode $$\theta^*$$, where the gradient is zero. The result is quadratic in $$\theta$$, so the posterior becomes a Gaussian with mean $$\theta^*$$ and covariance $$\mathbf{H}^{-1}$$, where $$\mathbf{H}$$ is the Hessian of $$-\log p(\theta \mid \mathcal{D})$$ at $$\theta^*$$;
- maximum likelihood estimation needs only derivatives, which are easier to compute than integrals, and Newton's method, below, uses the second-order expansion at each step.

### Newton's Method

Newton's method solves an equation $$f(x) = 0$$ by successive approximations. Each step replaces $$f$$ by its tangent at the current guess $$x^{(k)}$$, a first-order Taylor expansion, and takes the point where the tangent crosses the $$x$$ axis as the next guess:

$$
x^{(k+1)} = x^{(k)} - \frac{f(x^{(k)})}{f'(x^{(k)})}
$$

Near a simple root, the number of correct digits roughly doubles at each step. The starting point $$x^{(0)}$$ should be close to the root, however, because from a point far from it, the iteration can diverge. The animation below shows the first steps.

{: .picture}
<img width="45%" height="45%" src="{{ site.assets }}/algebra-for-ml/newton_method.gif"/>

{: .caption}
Newton's method: the tangent of $$f$$ at each guess crosses the $$x$$ axis at the next, improved guess, and the guesses converge to the zero of $$f$$. The animation numbers the initial guess $$x_1$$ (the text's $$x_0$$), and its legend reads \"Funktion\" (the function $$f$$) and \"Tangente\" (the tangent). Source: [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method)

A stationary point of $$f$$ (a minimum, maximum or saddle point) is a root of $$f'$$, which Newton's method finds with $$x^{(k+1)} = x^{(k)} - f'(x^{(k)}) / f''(x^{(k)})$$. In $$n$$ dimensions, the division becomes a multiplication with the inverse of the Hessian:

$$
x^{(k+1)} = x^{(k)} - \mathbf{H}_f(x^{(k)})^{-1} \, \nabla f(x^{(k)})
$$

This step goes to the stationary point of the second-order Taylor expansion of $$f$$ at $$x^{(k)}$$, which is its minimum when the Hessian is positive definite. It solves an $$n \times n$$ linear system, at $$O(n^3)$$ time and $$O(n^2)$$ memory, which rules out Newton's method for neural networks with billions of parameters. Quasi-Newton methods, such as [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS), instead approximate the inverse Hessian from the last few steps and gradients, at $$O(n)$$ memory per stored step.


## Backpropagation and Automatic Differentiation

A neural network of $$K$$ layers computes the composition:

$$
y = (f_K \circ f_{K-1} \circ \dots \circ f_1)(x) = f_K(f_{K-1}( \dots (f_1(x)) \dots ))
$$

where layer $$f_{i+1}$$ has the parameters $$\boldsymbol{\theta}_i$$. Training needs the derivatives of a loss $$L$$ with respect to all the parameters, and the chain rule gives them as products of Jacobians:

$$
\begin{align*}
\frac{\partial L}{\partial \boldsymbol{\theta}_{K-1}} & = \frac{\partial L}{\partial \boldsymbol{f}_K} {\color{#0000FF} \frac{\partial \boldsymbol{f}_K}{\partial \boldsymbol{\theta}_{K-1}}} \\
\frac{\partial L}{\partial \boldsymbol{\theta}_{K-2}} & = \frac{\partial L}{\partial \boldsymbol{f}_K} \boxed{ {\color{#FF7300} \frac{\partial \boldsymbol{f}_K}{\partial \boldsymbol{f}_{K-1}}} {\color{#0000FF} \frac{\partial \boldsymbol{f}_{K-1}}{\partial \boldsymbol{\theta}_{K-2}}} } \\
\frac{\partial L}{\partial \boldsymbol{\theta}_{K-3}} & = \frac{\partial L}{\partial \boldsymbol{f}_K} {\color{#FF7300} \frac{\partial \boldsymbol{f}_K}{\partial \boldsymbol{f}_{K-1}}} \boxed{ {\color{#FF7300} \frac{\partial \boldsymbol{f}_{K-1}}{\partial \boldsymbol{f}_{K-2}}} {\color{#0000FF} \frac{\partial \boldsymbol{f}_{K-2}}{\partial \boldsymbol{\theta}_{K-3}}} } \\
\frac{\partial L}{\partial \boldsymbol{\theta}_i} & = \frac{\partial L}{\partial \boldsymbol{f}_K} {\color{#FF7300} \frac{\partial \boldsymbol{f}_K}{\partial \boldsymbol{f}_{K-1}}} \cdots \boxed{ {\color{#FF7300} \frac{\partial \boldsymbol{f}_{i+2}}{\partial \boldsymbol{f}_{i+1}}} {\color{#0000FF} \frac{\partial \boldsymbol{f}_{i+1}}{\partial \boldsymbol{\theta}_i}} }
\end{align*}
$$

The orange terms are the derivatives of a layer with respect to its input, and the blue terms with respect to its parameters. Each line reuses the product of the line above and adds only the boxed terms. [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation) exploits this: it evaluates the products from the left, from the loss towards the input, so that each product is a row vector times a Jacobian, and the backward pass costs a small multiple of the forward pass, whatever the number of parameters.

Backpropagation is a special case of [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation), which computes the exact derivatives of a program by applying the chain rule to its elementary operations. Take, for example, the function:

$$
f(x) = \sqrt{x^2 + \exp(x^2)} + \cos\left(x^2 + \exp(x^2)\right)
$$

By the chain rule, and since differentiation is linear, its derivative is:

$$
\begin{align*}
\frac{\mathrm{d}f}{\mathrm{d}x} & = \frac{2x + 2x \exp(x^2)}{2 \sqrt{x^2 + \exp(x^2)}} - \sin\left(x^2 + \exp(x^2)\right) \left(2x + 2x \exp(x^2)\right) \\
& = 2x \left( \frac{1}{2 \sqrt{x^2 + \exp(x^2)}} - \sin\left(x^2 + \exp(x^2)\right) \right) \left(1 + \exp(x^2)\right) \, .
\end{align*}
$$

(Source: section 5.6, <a href="{{ site.resources_permalink }}">Mathematics for Machine Learning</a>)

Writing out the gradient in this explicit way is often impractical: the expression gets long, and it repeats subexpressions, such as $$x^2 + \exp(x^2)$$, so evaluating it as written costs more than evaluating $$f$$. Automatic differentiation instead names each intermediate result once, as an efficient implementation of $$f$$ would:

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

The picture shows the computation graph of these variables. The forward pass computes $$a$$ to $$f$$ in this order, with fewer operations than a direct implementation of $$f$$, and the local derivatives of the operations are:

$$
\frac{\partial a}{\partial x} = 2x, \quad
\frac{\partial b}{\partial a} = \exp(a), \quad
\frac{\partial c}{\partial a} = \frac{\partial c}{\partial b} = 1, \quad
\frac{\partial d}{\partial c} = \frac{1}{2\sqrt{c}}, \quad
\frac{\partial e}{\partial c} = -\sin(c), \quad
\frac{\partial f}{\partial d} = \frac{\partial f}{\partial e} = 1
$$

The backward pass then applies the chain rule from $$f$$ back to $$x$$, along the edges of the graph in reverse. Where a variable feeds several others, as $$c$$ feeds $$d$$ and $$e$$, its derivative sums over the paths:

$$
\begin{align*}
\frac{\partial f}{\partial c} & = \frac{\partial f}{\partial d} \frac{\partial d}{\partial c} + \frac{\partial f}{\partial e} \frac{\partial e}{\partial c} = \frac{1}{2\sqrt{c}} - \sin(c) \\
\frac{\partial f}{\partial b} & = \frac{\partial f}{\partial c} \frac{\partial c}{\partial b} = \frac{\partial f}{\partial c} \\
\frac{\partial f}{\partial a} & = \frac{\partial f}{\partial b} \frac{\partial b}{\partial a} + \frac{\partial f}{\partial c} \frac{\partial c}{\partial a} = \frac{\partial f}{\partial c} \left( \exp(a) + 1 \right) \\
\frac{\partial f}{\partial x} & = \frac{\partial f}{\partial a} \frac{\partial a}{\partial x} = 2x \, \frac{\partial f}{\partial c} \left( \exp(a) + 1 \right)
\end{align*}
$$

The result equals the explicit derivative above, but each step reuses the values of the forward pass and the derivatives already computed. The price is memory: the backward pass needs the intermediate values of the forward pass.


## Continuous Optimization

Training a model means finding the parameters that minimize a loss, and the candidates are its stationary points. A *stationary point* $$x_*$$ of $$f$$ has $$\nabla f(x_*) = 0$$, and the curvature at that point decides its type: it is a local minimum if $$\mathbf{H}_f(x_*)$$ is positive definite, a local maximum if it is negative definite, and a saddle point if it has both positive and negative eigenvalues. In one dimension, $$f''(x_*) > 0$$ gives a minimum, where $$f$$ is concave up, and $$f''(x_*) < 0$$ a maximum, where it is concave down.

### Gradient Descent

Since the gradient points uphill, a small step against it decreases $$f$$, and [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) repeats such steps, of size $$\gamma$$:

$$
x^{(k+1)} = x^{(k)} - \gamma \nabla f(x^{(k)})
$$

For a strongly convex $$f$$ and a well-chosen $$\gamma$$, the error shrinks by a factor of about $$1 - 1/\kappa$$ per step, where $$\kappa$$ is the ratio of the largest to the smallest eigenvalue of the Hessian. Badly conditioned problems, with a large $$\kappa$$, have narrow valleys that gradient descent crosses in a zigzag, so it converges slowly.

**Gradient descent with momentum** damps this zigzag by adding a fraction $$\alpha$$ of the previous update $$\Delta x^{(k)} = x^{(k)} - x^{(k-1)}$$ to the step:

$$
x^{(k+1)} = x^{(k)} - \gamma_k \nabla f(x^{(k)}) + \alpha \Delta x^{(k)}
$$

with $$\alpha \in [0, 1]$$, a hyper-parameter close to 1, such as $$0.9$$; $$\alpha = 0$$ gives back gradient descent. Since the previous update follows the same rule, $$\Delta x^{(k)} = \alpha \Delta x^{(k-1)} - \gamma_{k-1} \nabla f(x^{(k-1)})$$, the update is a moving average of the past gradients, with weights that decay as powers of $$\alpha$$. Gradient components that change sign between steps, across a valley, cancel out, while consistent components, along the valley, add up and speed up the descent. Momentum stores one extra vector.

**[Stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)** (SGD) reduces the cost of the gradient itself. Most ML losses are an average over $$N$$ samples, $$f(x) = \frac{1}{N} \sum_{i=1}^{N} f_i(x)$$, and SGD replaces their gradient by the average gradient of a random mini-batch of $$B$$ samples. This estimate is unbiased, its variance decreases as $$1/B$$, and a step costs $$B$$ instead of $$N$$ sample gradients. On large datasets, many cheap, noisy steps make faster progress than a few exact ones, and SGD and its variants train almost all deep networks.

### Adam Optimizer

The gradients of different parameters can have different scales, so a single step size can be too large for some parameters and too small for others. [Adam](https://arxiv.org/abs/1412.6980) (adaptive moment estimation, Kingma and Ba, ICLR 2015) therefore gives each parameter its own step size: in the words of its authors, it "computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients". With the mini-batch gradient $$g^{(k)}$$, Adam keeps exponential moving averages of the gradient (the first moment) and of its element-wise square (the second moment), with the decay rates $$\beta_1$$ and $$\beta_2$$. It corrects their bias towards their zero initialization, and divides the step of each parameter by the square root of its second moment:

$$
\begin{align*}
m^{(k)} & = \beta_1 m^{(k-1)} + (1 - \beta_1) \, g^{(k)} \\
v^{(k)} & = \beta_2 v^{(k-1)} + (1 - \beta_2) \, g^{(k)} \odot g^{(k)} \\
\hat{m}^{(k)} & = m^{(k)} / (1 - \beta_1^k), \quad \hat{v}^{(k)} = v^{(k)} / (1 - \beta_2^k) \\
x^{(k)} & = x^{(k-1)} - \gamma \, \hat{m}^{(k)} / \left( \sqrt{\hat{v}^{(k)}} + \epsilon \right)
\end{align*}
$$

with an element-wise square root and division, $$m^{(0)} = v^{(0)} = 0$$, and the defaults $$\gamma = 0.001$$, $$\beta_1 = 0.9$$, $$\beta_2 = 0.999$$ and $$\epsilon = 10^{-8}$$. Adam thus combines two methods: the first moment acts as momentum, and the second moment, the per-parameter scaling of RMSProp (Tieleman and Hinton, 2012), estimates the scale of each gradient component, not the curvature, which would be too expensive to compute. Adam stores two extra values per parameter. It can fail to converge, even on simple convex problems where SGD converges ([Reddi et al., 2018](https://arxiv.org/abs/1904.09237)), and switching from Adam to SGD during training can close the gap in generalization between the two ([Keskar and Socher, 2017](https://arxiv.org/abs/1712.07628)).

The optimizers compare as follows, for $$n$$ parameters:

| Method | Extra memory | Work per step | Idea | When to use |
|---|---|---|---|---|
| Gradient descent | none | one full gradient | steepest descent | small, smooth problems |
| SGD | none | one mini-batch gradient | many cheap, noisy steps | large datasets |
| Momentum | $$n$$ | as SGD | average of past gradients | badly conditioned problems, usually with SGD |
| Adam | $$2n$$ | as SGD | a step size per parameter | deep networks with badly scaled gradients, with little tuning |
| L-BFGS | a few vectors of size $$n$$ | one gradient, plus $$O(n)$$ per stored step | approximate inverse Hessian | smooth, deterministic problems |
| Newton | $$n^2$$ | $$O(n^3)$$ | minimum of the quadratic model | small problems that need high accuracy |

### Constrained Optimization

Many problems restrict the solution, for example to non-negative values. The problem $$\min_x f(x)$$ subject to $$g_i(x) \le 0$$, for $$i = 1, \dots, m$$, is equivalent to the unconstrained minimization of:

$$
f(x) + \sum_{i=1}^m \mathbb{1}(g_i(x))
$$

where $$\mathbb{1}(z) = 0$$ if $$z \le 0$$, and $$\mathbb{1}(z) = \infty$$ otherwise, which gives every infeasible point an infinite cost. This step function gives no gradient to follow, so [Lagrange multipliers](https://en.wikipedia.org/wiki/Lagrange_multiplier) $$\lambda_i \ge 0$$ replace it by a linear term, in the *Lagrangian*:

$$
\mathfrak{L}(x, \lambda) = f(x) + \sum_{i=1}^m \lambda_i g_i(x) = f(x) + \lambda^{\intercal} g(x)
$$

For a fixed $$x$$, the maximum of the Lagrangian over $$\lambda \ge 0$$ is $$f(x)$$ if all constraints hold, and infinite if one of them is violated, since its multiplier can then grow without bound. So $$\max_{\lambda \ge 0} \mathfrak{L}(x, \lambda)$$ equals the function above, and the original *primal* problem is $$\min_x \max_{\lambda \ge 0} \mathfrak{L}(x, \lambda)$$. [Lagrange duality](https://en.wikipedia.org/wiki/Duality_(optimization)) swaps the order, into the *dual* problem $$\max_{\lambda \ge 0} \mathfrak{D}(\lambda)$$, with $$\mathfrak{D}(\lambda) = \min_x \mathfrak{L}(x, \lambda)$$: an optimization over the *dual variables* $$\lambda$$ instead of the *primal variables* $$x$$. $$\mathfrak{D}$$ is concave even when $$f$$ and the $$g_i$$ are not convex, and the dual optimum is a lower bound of the primal one (weak duality). For convex problems that satisfy [Slater's condition](https://en.wikipedia.org/wiki/Slater%27s_condition), the two optima are equal (strong duality), so solving the dual solves the primal problem. Section 7.2 of the <a href="{{ site.resources_permalink }}">MML book</a> gives the details.

### Convex Sets and Functions

A **convex set** contains the line segment between any two of its points. A **[convex function](https://en.wikipedia.org/wiki/Convex_function)** lies on or below the line segment between any two points of its graph, i.e. it is concave up, and a twice-differentiable $$f$$ is convex iff its Hessian is positive semidefinite everywhere, i.e. $$f'' \ge 0$$ in one dimension. Convexity matters because every local minimum of a convex function is a global minimum, and a strictly convex function has at most one minimum. Convex problems, such as least squares, logistic regression and the training of SVMs, can thus be solved to global optimality. The losses of neural networks are not convex, so these guarantees do not hold for them.

### Linear Programming

A **[linear program](https://en.wikipedia.org/wiki/Linear_programming)** optimizes a linear objective under linear constraints: it finds the $$x \in \mathbb{R}^n$$ that maximizes $$c^{\intercal} x$$ subject to $$Ax \le b$$ and $$x \ge 0$$, with $$A \in \mathbb{R}^{m \times n}$$ and $$b \in \mathbb{R}^m$$. The constraints define a convex polytope of feasible points, and since the objective is linear, an optimum, when it exists, lies at one of its vertices. The [simplex method](https://en.wikipedia.org/wiki/Simplex_algorithm) moves between the vertices of the polytope; it is fast in practice, but exponential in the worst case. [Interior-point methods](https://en.wikipedia.org/wiki/Interior-point_method) cross the inside of the polytope, in polynomial time. Solvers such as HiGHS implement both.
