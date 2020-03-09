---
layout: post
title:  "Bayesian Optimization"
date:   2019-10-12 01:00:00 +0100
categories: [machine learning, supervised learning, probabilistic programming]
tags: [machinelearning]
---

Bayesian methods allows us to perform modelling of an input to an output by providing a measure of uncertainty or "how sure we are", based on the seen data. Unlike most *frequentist* methods commonly used, where the outpt of the method is a set of best fit parameters (e.g. the values of the weight on linear regression), the output of a Bayesian regression is a probability distribution of each model parameter, called the *posterior distribution*. 
Unlike most *frequentist* methods commonly used, where the outpt of the method is a set of best fit parameters (e.g. the values of the weight on linear regression), the output of a Bayesian regression is a probability distribution of each model parameter, called the *posterior distribution*. 

For the sake of comparison, take the example of the simple linear regression $y = mx + b$. On the frequentist approach, one tries to find the constant that define the slope $m$ and bias $b$ values, with $m \in \mathbb{R}$ and $b \in \mathbb{R}$, using optimization methods. On the other hand, the Bayesian approach would also compute $y = mx + b$, however, $b$ and $m$ are not fixed values but drawn from probabilities that we *tune* during training. As an example, we can assume as prior knowledge that $m$ and $b$ are independent and and normally distributed, therefore $b \thicksim \mathbb{N}(\mu, \sigma^2)$ and $w \thicksim \mathbb{N}(\mu, \sigma^2)$, and the parameters to be learnt would then be all $\mu$ and $\sigma$. Additionally, we can also add the model of noise (or inverse precision) constant $\varepsilon$ that models as well *how noisy* our data may be, by computing $y = mx + b + \varepsilon$, with $\varepsilon \thicksim \mathbb{N}(0, \sigma^2)$.
 
Another benefit of Bayesian is the possibility of **online learning**, TODO

Take the following example of few observations of $x$ in a linear space, plotted in pink, with a yellow line represeting a linear regression of the data:
  
<p align="center">
<img width="40%" height="40%" src="/assets/2019-Bayesian-Optimization/linear_bayesian_1.png">
</p>

The value of $x$ can be estimated by the value in the regression model represented by a blue cross. However, because it is so different from other observations, we may not be certain of how accurate is our prediction. The Bayesian model helps in this decision by computing the uncertainty (or error) of our decision, as plotted in green:

<p align="center">
<img width="40%" height="40%" src="/assets/2019-Bayesian-Optimization/linear_bayesian_2.png">
</p>

In practice, Bayesian uses decision theory to *optimize the loss function*. That is possible because Bayesian gives us what we need to optimize the loss function: the *predictive distribution* of the output given the data i.e. $p( y \| x, D)$. 

In this post we will discuss how to perform Bayesian regression, with a particular emphasis on linear regression and normal distrutions.

### Basic concepts

From the field of probability, the **product rule** tells us that the **joint distribution** of two given events $A$ and $B$ can be written as the product of the distribution of $a$ and the **conditional distribution** of $B$ given a value of $A$, i.e: $P(A, B) = P(A) P(B\|A)$. By symmetry we have that $P(B,A) = P(B) P(A\|B)$. By equating both right hand sides of the equations and re-arranging the terms we obtain the **Bayes Theorem**:

\begin{equation}
P (A\|B) = \frac{P(B\|A) P(A)}{P(B)} \propto P(B\|A) P(A)
\end{equation}

This equation is commonly read as "the **posterior** $P(A\| B)$ is proportional to the product of the **prior** $P(A)$ and the **likelihood** $P(B\|A)$" -- note that we dropped the **normalizer** term $P(B)$ as it is constant, making the left-hand term proportional to $P (A\|B)$.
 
The posterior distribution describes how much the data has changed our *prior* beliefs. An important theoream called the *Bernstein-von Mises Theorem* states that:
- for a sufficiently large sample size, the posterior distribution becomes independent of the prior (as long as the prior is neither 0 or 1)
  - ie for a sufficiently large dataset, the prior just doesnt matter much anymore as the current data has enough information;
  - in other others, if we let the datapoints go to infitiny, the posterior distribution will go to normal distribution, where the mean will be the maximum likelihood estimator;
    - this is a restatement of the central limit theorem, where the posterior distribution becomes the likelihood function;
  - ie the effect of the prior decreases as the data increases;

The prior distribution $P(A)$ is a shorthand for $P(A \| I)$ where $I$ is all information we have before start collecting data. If we have no information about the parameters then $P(A\|I)$ is a constant --- called an *uninformative prior* or *objective prior* --- and the posterior equals the likelihood function. Otherwise, we call it a *substantive/informative* prior.

### Bayesian Linear Optimization


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
- Division:
  - if $rA=B$, then $r=BA^{-1}$, for a scalar $r$;
  - if $Ar=B$, then $r=A^{-1}B$, for a scalar $r$;
  - $Ax=b$ is the system of linear equations $a_{1,1}x_1 + a_{1,2}x_2 + ... + a_{1,n}x_n = b_1$ for row $1$, repeated for every row.
    - therefore, $x = A^{-1}b$, if matrix has $A$ an inverse.
- Inverse: $AA^{-1}=A^{-1}A=I$;
  - If $A$ is invertible, its inverse is unique;
  - If $A$ is invertible, then $Ax=b$ has an unique solution;
  - If $A$ is invertible, $(A^{-1})^{-1}=A$;
  - $rA^{-1} = (\frac{1}{r}A)^{-1}$ for a scalar $r$;
