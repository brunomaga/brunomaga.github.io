---
layout: post
title:  "Bayesian Optimization: Complex use cases"
categories: [machine learning, supervised learning, probabilistic programming]
tags: [machinelearning]
---

### Stochastic Variational Inference for approximate posterior

advantage: faster than sampling methods; allows non-linear regression methods;
disadvantages: over-confident, works well only if we know the parametric distribution of the posterior, requires the posterior to follow a parametric distribution, otherwise we can use the sampling.

### Monte Carlo sampling for exact posteriors 

advantages: *real* posterior.
disadvantages: very high computation cost, what to do with an exact posterior which doesnt follow a parametric ditribution.

### Deep Neural Net

---

### Refresher: Linear Algebra

1. **Multiplication:**
	1. $A(BC) = (AB)C$;
	2. $A(B+C)=AB+AC$;
	3. $(B+C)A=BA+CA$;
	4. $r(AB)=(rA)B=A(rB)$ for a scalar $r$;
	5. $I_mA=AI=AI_n$;

2. **Transpose:**
	1. $(A^T)^T = A$;
	2. $(A+B)^T=A^T+B^T$;
	3. $(rA)^T = rA^T$ for a scalar $r$;
	4. $(AB)^T=B^TA^T$;

3. **Division:**
	1. if $rA=B$, then $r=BA^{-1}$, for a scalar $r$;
	2. if $Ar=B$, then $r=A^{-1}B$, for a scalar $r$;
	3. $Ax=b$ is the system of linear equations $a_{1,1}x_1 + a_{1,2}x_2 + ... + a_{1,n}x_n = b_1$ for row $1$, repeated for every row;
		- therefore, $x = A^{-1}b$, if matrix has $A$ an inverse;

4. **Inverse:** 
	1. $AA^{-1}=A^{-1}A=I$;
	2. If $A$ is invertible, its inverse is unique;
	3. If $A$ is invertible, then $Ax=b$ has an unique solution;
	4. If $A$ is invertible, $(A^{-1})^{-1}=A$;
	5. $rA^{-1} = (\frac{1}{r}A)^{-1}$ for a scalar $r$;
