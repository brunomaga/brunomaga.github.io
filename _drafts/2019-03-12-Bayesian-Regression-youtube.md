---
layout: post
title:  "[draft] Bayesian Linear Regression"
date:   2019-03-12 12:01:42 +0100
categories: [machine learning]
tags: [machinelearning]
---

Regular regression models, particularly linear regression, have a major problem: overfitting. An alternative would be to use Maxiumum a porteriori (MAP), however we have no representation of our uncertainty.

To that extent, the main advantadge of Bayesian regression is that provides a measure of uncertainty or "how sure we are", based on the seen data. Take the following example of few observations of $x$ in a linear space, plotted in pink, with a yellow line represeting a linear regression of the data:
  
<p align="center">
<img width="40%" height="40%" src="/assets/2019-Bayesian-Regression/linear_bayesian_1.png">
</p>

The value of $x$ can be estimated by the value in the regression model represented by a blue cross. However, because it is so different from other observations, we may not be certain of how accurate is our prediction. The Bayesian model helps in this decision by computing the uncertainty (or error) of our decision, as plotted in green:

<p align="center">
<img width="40%" height="40%" src="/assets/2019-Bayesian-Regression/linear_bayesian_2.png">
</p>

In practice, Bayesian uses decision theory to *optimize the loss function*. That is possible because Bayesian gives us what we need to optimize the loss function: the *predictive distribution* $p( y | x, D)$. 

Here is the setup: 
- Data is given by $D = ((x_1, y_1), ..., (x_n, y_n)), x+i \in R^d, y_i \in R$;
- $y$ are conditionally independent given $w$, where $y_i \approx \mathcal{N}(w^T x_i, a^{-1})$ for normal distribution $\mathcal{N}$ and $a$ being the precision $a = 1/ \delta^2$;
- $w$ is a multivariate normal with $w \approx \mathcal{N}(0, b^{-1} I)$, the mean given by zero-vector $0$, covariance matrix $b^{-1} I$, and identity matrix $I$. $b>0$ is also the precision value.
- $a$ and $b$ are known, so the only unknown is $w$.

We can replace $x_i$ by an activation function $\phi(x_i) = ( \phi_1(x_1), ..., \phi_n(x_n))$, allowing us to model non-linearities in $x$, as before.

##### Posterior for Linear Regression

The likelihood of our data is $p(D\|w)$ is proportional to $exp ( - \frac{a}{2} (y - Aw)^T (y-Aw))$, where A is the *design matrix* $A = ( -x_i^T, ..., -x_n^T)^T$ , for vectors $x$. The posterior of our data is $p(w\|D)$ proportional to $p(D\|w) p(w)$ given by:

\begin{equation}
\label{eq_1}
exp ( -\frac{a}{2} (y-Aw)^T (y-Aw) - \frac{b}{2} w^Tw). 
\end{equation}

Remember from before that we put a multivariance Gaussian prior on $w$, being proportional to $exp( -\frac{b}{2} w^Tw)$.

Our strategy is to show that the posterior distribution is a Gaussian. We can see that as the previous equation in 
\ref{eq_1} is quadratic in $w$, therefore the posterior is a Gaussian. For simplicity we pull out the $exp$ and multiply by $2$, leading to:

\begin{equation}
a(y-Aw)^T(y-Aw) + bw^Tw = a(y^Ty - 2w^TA^Ty + w^TA^TAw) + bw^Tw = ay^Ty - 2aw^TA^Ty+w^T(aA^TA+bI)w
\end{equation}

To solve this, we use a faily complicated trick: 
Imagine $\Lambda$ is the inverse of the covariance matrix. We know that $(w - \mu)^T \Lambda (w - \mu) = w^T \Lambda w - 2w \Lambda \mu +$ const. If we take $\Lambda = aA^TA+bI$ and $\mu = a \Lambda^{-1} A^T y$,  we see that $aw^TA^Ty = w^T \Lambda \mu$ and $a A^T y = \Lambda \mu$. This solves the previous equation.

Thus the posterior is given by a normal distribution $p(w \| D) = \mathcal{N}(w \| \mu, \Lambda^{-1})$. 

### Maximum a Posteriori and Maximum Likelihood Estimator of w

The MAP of w is given by $w_{MAP} = a ( aA^TA+bI)^{-1} A^T y = (A^TA + \frac{b}{a} I)^{-1}A^T y$.

The MLE of w is given by $(A^TA)^{-1}A^Ty$ where we remove $\frac{b}{a}I$ from the previous equation as the regularization parameter. 

The predictive probability is given by:

\begin{equation}
p (y\|x,D) = \int p(y\|x,D,w) p(w\|x,D) dw = \int \mathcal{N} (y \| w^Tx, a^{-1}) \mathcal{N} (w \| \mu, \Lambda^{-1}) dw
\end{equation}

which is proportional to:
\begin{equation}
\int exp( -\frac{a}{2}(y - w^Tx)^2) exp (\frac{1}{2}(w-\mu)^T \Lambda (w-\mu)) dw = \int exp (-\frac{a}{2} (y^2 - 2 (w^Tx) y + (w^Tx)^2 - \frac{1}{2} (w^T \Lambda w - 2w^T \Lambda \mu + \mu^T \Lambda \mu))
\end{equation}

STOPPED AT minute 10 in https://www.youtube.com/watch?v=xyuSiKXttxw

---

Reminder of some rules in matrices of linear algebra operations, in case the previous reductions cause some confusion:
- Multiplication: |
  - $A(BC) = (AB)C$;
  - $A(B+C)=AB+AC$;
  - $(B+C)A=BA+CA$;
  - $r(AB)=(rA)B=A(rB)$ for a scalar $r$;
  - $I_mA=AI=AI_n$;
- Transpose:
  - $(A^T)^T = A$;
  - $(A+B)^T=A^T+B^T$;
  - $(rA)^T = rA^T$ for a scalar $r$;
  - $(AB)^T=B^TA^T$;
- Inverse: $AA^{-1}=A^{-1}A=I$;
  - If $A$ is invertible, its inverse is unique;
  - If $A$ is invertible, then $Ax=b$ has an unique solution;
  - If $A$ is invertible, $(A^{-1})^{-1}=A$;
  - $rA^{-1} = (\frac{1}{r}A)^{-1}$ for a scalar $r$;
