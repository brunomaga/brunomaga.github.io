---
layout: post
title:  "Bayesian Optimization: Linear Regression"
categories: [machine learning, supervised learning, probabilistic programming]
tags: [machinelearning]
---

Bayesian methods allows us to perform modelling of an input to an output by providing a measure of *uncertainty* or "how sure we are", based on the seen data. Unlike most *frequentist* methods commonly used, where the outpt of the method is a set of best fit parameters, the output of a Bayesian regression is a probability distribution of each model parameter, called the **posterior distribution**. 
Unlike most *frequentist* methods commonly used, where the outpt of the method is a set of best fit parameters (e.g. the values of the weight on linear regression), the output of a Bayesian regression is a probability distribution of each model parameter, called the *posterior distribution*. 

For the sake of comparison, take the example of the simple linear regression $y = mx + b$. On the frequentist approach, one tries to find the constant that define the slope $m$ and bias $b$ values, with $m \in \mathbb{R}$ and $b \in \mathbb{R}$, using optimization methods. On the other hand, the Bayesian approach would also compute $y = mx + b$, however, $b$ and $m$ are not fixed values but drawn from probabilities that we *tune* during training. As an example, we can assume as prior knowledge that $m$ and $b$ are independent and normally distributed --- i.e. $b \thicksim \mathcal{N}(\mu, \sigma^2)$ and $w \thicksim \mathcal{N}(\mu, \sigma^2)$ --- and the parameters to be learnt would then be all $\mu$ and $\sigma$. Additionally, we can also add the model of the noise (or inverse of the precision) constant $\varepsilon$ that models as well *how noisy* our data may be, by computing $y = mx + b + \varepsilon$ instead, with $\varepsilon \thicksim \mathcal{N}(0, \sigma^2)$.
 
Take the following example of few observations of $x$ in a linear space, plotted in pink, with a yellow line representing a linear regression of the data. The value of $x$ can be estimated by the value in the regression model represented by a blue cross. However, because it is so different from other observations, we may not be certain of how accurate is our prediction. The Bayesian model helps in this decision by computing the uncertainty (or error) of our decision, as plotted in green:

<p align="center">
<img width="40%" height="40%" src="/assets/Bayesian-Optimization/linear_bayesian_2.png">
</p>

Apart from the uncertainty quantification, another benefit of Bayesian is the possibility of **online learning**, i.e. a continuous update of the trained model (from previously-seen data) by looking at only the new data. This is a handy feature for e.g. datasets that are purged periodically.

In this post we will discuss how to perform Bayesian optimization, with a particular emphasis on linear regression and normal distrutions.

### Basic concepts

From the field of probability, the **product rule** tells us that the **joint distribution** of two given events $A$ and $B$ can be written as the product of the distribution of $a$ and the **conditional distribution** of $B$ given a value of $A$, i.e: $P(A, B) = P(A) P(B\mid A)$. By symmetry we have that $P(B,A) = P(B) P(A\mid B)$. By equating both right hand sides of the equations and re-arranging the terms we obtain the **Bayes Theorem**:

\begin{equation}
P (A\mid B) = \frac{P(B\mid A) P(A)}{P(B)} \propto P(B\mid A) P(A)
\label{eq_prior_AB}
\end{equation}

This equation is commonly read as "the **posterior** $P(A\mid  B)$ is proportional to the product of the **prior** $P(A)$ and the **likelihood** $P(B\mid A)$" -- note that we dropped the **normalizer** term $P(B)$ as it is constant, making the left-hand term proportional to $P (A\mid B)$.
 
The prior distribution $P(A)$ is a shorthand for $P(A\mid I)$ where $I$ is all information we have before start collecting data. If we have no information about the parameters then $P(A\mid I)$ is a constant --- called an *uninformative prior* or *objective prior* --- and the posterior equals the likelihood function. Otherwise, we call it a *substantive/informative* prior.

The posterior distribution describes how much the data has changed our *prior* beliefs. An important theoream called the *Bernstein-von Mises Theorem* states that:
- for a sufficiently large sample size, the posterior distribution becomes independent of the prior (as long as the prior is neither 0 or 1)
  - ie for a sufficiently large dataset, the prior just doesnt matter much anymore as the current data has enough information;
  - in other others, if we let the datapoints go to infitiny, the posterior distribution will go to normal distribution, where the mean will be the maximum likelihood estimator;
    - this is a restatement of the central limit theorem, where the posterior distribution becomes the likelihood function;
  - ie the effect of the prior decreases as the data increases;


### Probability Distributions and the Exponential Family 

A [probability distribution](https://en.wikipedia.org/wiki/Probability_distribution) is a mathematical function that provides the probabilities of occurrence of different possible outcomes in an experiment. Examples:
- Gaussian distribution, for an input $y$:
  - $$p(y \mid \mu, \sigma^2 ) = \frac{1}{ \sqrt{2 \pi \sigma^2} } \text{ } exp \left( - \frac{(y - \mu)^2}{2 \sigma^2} \right)$$ for a distribution with mean $\mu$ and standard deviation $\sigma$, or
  - $$p(y \mid \mu, \Sigma ) = \left(\frac{1}{2 \pi}\right)^{-D/2} \text{ } det(\Sigma)^{-1/2} exp \left( - \frac{1}{2}(y - \mu)^T \Sigma^{-1} (y-\mu) \right)$$ with means-vector $\mu$ and covariance matrix $\Sigma$ on its multivariate notation
- Laplace:
 $$ p( y_n \mid x_n, w ) = \frac{1}{2b} e^{-\frac{1}{b} \mid y_n - X_n^T w \mid } $$

- TODO add more distributions

### Exponential Family of distributions

- TODO add explanation of exponential family
- TODO explain that because all are exponentials, the log makes it easier

### Maximum Likelihood (MLE) and Maximum-a-Posteriori (MAP)

The problem in hand is to find the parameters of the distribution that best represent the data. Adapting the previous equation \ref{eq_prior_AB} of the prior to the problem of regression in a Bayesian environment (finding the weights of our model given input,  labels, and parameters), we aim at computing:

$$
P (w\mid y, X, \sigma^2) = \frac{P(y\mid w, X, \sigma^2) P(w)}{P(y, X, \sigma^2)} \propto P(y\mid w, X, \sigma^2) P(w)
\label{eq_prior_w}
$$

for a given input set $X$, with labels $y$, and model parameters $\sigma$.

There are two main optimization problems that we discuss commonly on Bayesian methods:
- When we try to find *how likely is that an output $y$ belongs to a model defined by $X$, $w$ and $\sigma$*, or **maximize the likelihood $P(y\mid w, X, \sigma^2)$**, we perform a Maximum Likelihood Estimator (MLE);
- When we try to maximize the posterior, or the probability of the model parameters $w$ given the model $X$, $y$ and $\sigma$, or **maximize the posterior $P (w\mid y, X, \sigma^2)$**, we perform a Maximum-a-Posteriori (MAP); 


To compute that, we perform the *log-trick* and place the term to optimize into a log function. We can do this because $log$ is a monotonically-increasing function, thus applying it to any function won't change the input values where the minimum or maximum of the solution is found (ie where gradient is zero). Moreover, since most distributions are part of the **exponential family** of distributions, they can all be represented as an exponential, and applying the log will bring the power term *out* of the exponential, and make it computationally simpler are faster. 

The **log-likelihood** is the log of the [likelihood](https://www.statisticshowto.datasciencecentral.com/likelihood-function/) of observing --- given observed data --- a parameter value  of the statistical model used to describe that data. I.e.:

$$
L_{lik} (w) = log p (y \mid X, w)
$$

This can be used to estimate the cost. The log-likelihood is (typically?) convex in the weight vector $w$, as it's a sum of convex functions. The **Maximum likelihood estimator (MLE)** states that:

$$
argmin_w L_{MSE}(w) = argmax_w L_{lik} (w)
$$

i.e. the solution that minimizes the weights in the Mean Square Error problem, maximizes the log-likelihood of the data. MLE is a sample approximation of the *expected* log-likelihood i.e.

$$
L_{lik}(w) \approx E_{p(x,y)} [ log p(y \mid x,w ) ]
$$

[coordinate-ascent]: {{ site.baseurl }}{% post_url 2018-02-17-Supervised-Learning %}

When the distribution of the prior and posterior are computationally tractable, the optimization of the parameters that define the distribution can be performed using the [coordinate ascent][coordinate-ascent] method, detailed briefly previously. In practice we perform and iterative partial derivatives of the prior/posterior for the model parameters. For the sake of comparison, while we minimize the negation of the loss on $w$ when performing linear regression, on a Bayesian model with a Gaussian distribution on all priors, we'd maximize based on the coordinate ascent of the parameters mean $\mu$ and standard deviation $\sigma$.

### Bayesian Linear Optimization

<small>
(In case the following reductions cause some confusion, check <a href="{{ site.assets }}/the_matrix_cookbook.pdf">The Matrix Cookbook</a> for a simple overview of matrix operations.)
</small>

A particular exception of Bayesian optimization that requires non-iterative methods is the linear regression with normal priors and posterior. In this case, *the posterior has an analytical solution*. This approach is utilized very commonly, mainly due to two reasons:
1. defining the appropriate parametric distribution of the weights (i.e. the prior) is a *hard* problem and requires domain knowledge that many times is not easy to grasp;
2. the analytical solution for the posteriors is *extremelly fast* to compute even for very large datasets and dimensionality, compared to aternative methods that we will cover later;

Take the simplest form of linear regression with $y = Xw$. To penalize outliers, we use a loss function based on the mean square error. Thus, we want now to minimize

$$
(y - Xw)^2 = (y-Xw)^T(y-Xw).
\label{eq_E2}
$$

This minimization problem is called the Least Square Problem, and we showed in a [previous post]({{ site.baseurl }}{% post_url 2018-02-17-Supervised-Learning %}) that it has the closed-form solution $ w = (X^TX)^{-1} X^Ty$. 

In practice, minimizing the Least Squares problem is equivalent to determining the most likely $w$ under the assumption that $y$ contains gaussian noise  i.e. $y = wx + b + \varepsilon$ instead, with $\varepsilon \thicksim \mathcal{N}(0, \sigma^2)$. This is obivous since the MSE loss in $(y - Xw)^2$ increases quadratically with the difference between groundtruth $y$ and predicted $Xw$ values.

With that in mind, suppose that we want to compute the noise level in the output solution, ie, suppose $y$ has been generated through $y_d = w_0 + w_1x_1 + ... + w_dx_d + \varepsilon_d$ for a problem with dimensionlity $D$, with $\varepsilon_d \thicksim \mathcal{N}(0, \sigma^2)$. Applying the Gaussian distribution (with mean setto the $Xw$ sum of products), we get:
  - $$p(y_d \mid X_d, w, \sigma^2 ) = \frac{1}{ \sqrt{2 \pi \sigma^2} } exp \left( - \frac{1}{2 \sigma^2} \left( y_d - x_dw \right)^2 \right)$$ or
  - $$p(y \mid X, w, \sigma^2 ) = \frac{1}{(2 \pi \sigma^2)^{D/2}} exp \left( - \frac{1}{2 \sigma^2} (y-Xw)^T(y-Xw) \right)$$ in matrix form;

Going back to equation \ref{eq_prior_w}, we take the log of the previous equation to simplify the maths: 
$$
L = log P ( y \mid X, w, \sigma^2 ) = - \frac{D}{2} log ( 2 \pi \sigma^2) - \frac{1}{2 \sigma^2} (y-Xw)^T(y-Xw)
$$

The relevant message here is that minimizing the loss (or maximizing the negative log) --- i.e. by computing its derivative equal to zero --- depends only on the second term on the right-hand side sum, as the first term is independent of $w$. Moreover, $1/\sigma^2$ is a constant therefore this resolution is equivalent to miminizing the Least Squares (Equation \ref{eq_E2}) and scale it to $1/\sigma^2$.

##### Adding regularization

Regularizers can be added normally as in the non-Bayesian regression and may have as well an analytical solution. As an example, if we want ot add an $L_2$/Ridge regularizer:

$$
  \begin{align*}
 0 = & \frac{d}{dw} \left( L - \frac{\alpha}{2} w^Tw \right)\\
 0 = & \frac{d}{dw} \left( - \frac{D}{2} log ( 2 \pi \sigma^2) - \frac{1}{2 \sigma^2} (y-Xw)^T(y-Xw) - \frac{\alpha}{2} w^Tw \right)\\
 w = & (X^TX + \lambda I )^{-1} X^Ty
  \end{align*}
$$

where $\lambda = \alpha \sigma^2$. Note that we picked the regularizer constant $\alpha/2$ on the first step to simplify the maths and *cancel out* the 2 when doing the derivative of the $w^Tw$.

##### Analytical Solution

We assume all our weights are drawn from a gaussian distribution and may (or may not) be independent. In practice, we start with the prior $p(w) \thicksim \mathcal{N}(m_0, S_0)$, with mean vector $m_0$, and (positive semi-definite) covariance matrix $S_0$ (following the variable notation found on [Chris Bishop's PCML book]({{ site.resources_permalink }}).

We have then the following prior in multivariate notation:

$$
p(w \mid m_0, S_0 ) = (2 \pi)^{-\frac{D}{2}} \text{ } det(S_0)^{- \frac{1}{2}} \text{ } exp \left( - \frac{1}{2}(w - m_0)^T S_0^{-1} (y-m_0) \right)
$$ 

And the likelihood:

$$
p(y \mid w, X, \sigma^2) = \frac{1}{(2 \pi \sigma^2)^{D/2}} exp \left(  -\frac{1}{2 \sigma^2} (y-Xw)^T(y-Xw) \right)
$$

We now replace the terms on the the Bayes equation \ref{eq_prior_w} with the previous terms, leading to the final formulation:

$$
p(w \mid y, X, \sigma^2) \propto exp \left(  -\frac{1}{2 \sigma^2} (y-Xw)^T(y-Xw) - \frac{1}{2}(w-m_0)^TS_0^{-1}(w-m_0) \right)
$$

##### Exponent of Multivariate Normal Distribution

The value of $$d = \sqrt{ (x - \mu)^T \Sigma^{-1}(x-\mu)}$$ for a given observation $x_i$ is called the [Mahalonabis distance](https://en.wikipedia.org/wiki/Mahalanobis_distance).

We will note that the previous density function only depends on $w$ through the squared Mohalanobis distance:

$$
Q = -\frac{1}{2}(w - \mu)^T S^{-1}(w-\mu)
$$

which is the equation for a hyper-ellipse centered at the mean $\mu$, eg:

<p align="center">
<img width="30%" height="30%" src="/assets/Bayesian-Optimization/mahalanobis.gif"/>
<br/><br/><small>bivariate normal for 2 dimensions, showing an ellipse with center at mean $\mu$ and covariance $S$
<br/>source: <a href="https://online.stat.psu.edu/stat505/lesson/4/4.3">Applied Multivariate Statistical Analysis, PennState Eberly College of Science</a> </small>
</p>


There are several important properties on the exponent component defined by the Mahalobis distance $(w - \mu)^T S^{-1}(w-\mu)$:
- All values of $w$ such that $$(w - \mu)^T S^{-1}(w-\mu) = c$$  for any constant $c$ have the same value of density $p(w)$ and thus equal likelihood;
- As the value of $$(w - \mu)^T S^{-1}(w-\mu)$$ increases, the value of the density function decreases;
- The value of $$(w - \mu)^T S^{-1}(w-\mu)$$ increases as the distance bewtween $w$ and $\mu$ decreases;
- The variable $d^2$ has a chi-square distribution with $p$ degrees of freedom;

We will find the mean $\mu$ and covariance $S$ that fit that expression (also detailed on section 9.3.3 of the [Mathematics for Machine Learning book]({{ site.resources_permalink }})).

$$
  \begin{align}
   & - \frac{1}{2 \sigma^2} (y-Xw)^T(y-Xw) - \frac{1}{2}(w-m_0)^TS_0^{-1}(w-m_0) \\
 = & - \frac{1}{2} \left( \sigma^{-2} y^T y - 2\sigma^{-2} y^T Xw + \sigma^{-2} w^T X^T Xw + w^TS_0^{-1}w - 2 m_0^T S_0^{-1} w + m_0^T S_0^{-1} m_0 \right) \\
   \label{eq1_sq}
 = & - \frac{1}{2} \left( w^T ( \sigma^{-2} X^TX + S_0^{-1}) w \right)   & \hspace{2cm}\text{(terms quadratic in $w$)} \\
   \label{eq1_lin}
   & + \left( \sigma^{-2} X^Ty + S_0^{-1} m_0)^T w \right)               &  \hspace{2cm}\text{(terms linear in $w$)} \\
   & - \frac{1}{2} \left( \sigma^{-2} y^T y + m_0^T S_0^{-1} m_0 \right) & \hspace{2cm}\text{(const terms independent of $w$)} \\
  \end{align}
$$


Looking at the last term, we can see that this function is quadratic in $w$. Because the unnoormalized log-posterior distribution is negative, implies that *the posterior is Gaussian*, i.e. going back to Equation \ref{eq_prior_w}:

$$
  \begin{align*}
    & P (w\mid y, X, \sigma^2) \\
  = & exp ( log \text{ } P (w\mid y, X, \sigma^2) ) )  \\
\propto & exp ( log \text{ } P(y\mid w, X, \sigma^2) + log \text{ } P(w) ) \\
\propto & exp \left( -\frac{1}{2} \left( w^T ( \sigma^{-2} X^TX + S_0^{-1}) w \right) + \left( \sigma^{-2} X^Ty + S_0^{-1} m_0)^T w \right) \right) \\
  \end{align*}
$$

where we used the linear and quadratic terms in $w$ and ignored the $const$ terms as they do not change the proportional operation.

Finally, to bring this unnormalized Gaussian into the form proportional to $ \mathcal{N} ( w \mid m_N, S_N )$, we utilize the method for [completing the square](https://en.wikipedia.org/wiki/Completing_the_square). We want:

$$
  \begin{align}
  & \mathcal{N} ( w \mid m_N, S_N ) \\
= & -\frac{1}{2}(w - \mu)^T S^{-1}(w-\mu)    & \hspace{2cm}\text{(the Mohalanobis term?)} \\ 
   \label{eq2_sq}
 = & -\frac{1}{2} \left(w^T S_N^{-1}w \right)   & \hspace{2cm}\text{(terms quadratic in $w$)} \\
   \label{eq2_lin}
   & + m^T_N S^{-1}_N w                         &  \hspace{2cm}\text{(terms linear in $w$)} \\
   & - \frac{1}{2} \left( m_N^T S_N^{-1} m_N \right) & \hspace{2cm}\text{(const terms independent of $w$)} \\
  \end{align}
$$

Thus, by matching the squared (\ref{eq1_sq}, \ref{eq2_sq}) and linear (\ref{eq1_lin}, \ref{eq2_lin}) expressions on the computed vs desired equations, we can find $S_N$ and $m_N$ as:

$$
 \begin{align*}
                   & w^T S_N^{-1} w = \sigma^{-2} w^TX^TXw+w^TS_0^{-1} w \\
  \Leftrightarrow  & w^T S_N^{-1} w = w^T ( \sigma^{-2} X^TX+ S^{-1}_0) w \\
  \Leftrightarrow  & S_N^{-1} = \sigma^{-2} X^T X + S^{-1}_0 \\
  \Leftrightarrow  & S_N = ( \sigma^{-2} X^TX + S^{-1}_0)^{-1} \\
 \end{align*}
$$

and

$$
 \begin{align*}
                   & -2 m_N^T S_N^{-1}w = 2 (\sigma^{-2}X^Ty + S_0^{-1} m_0)^T w \\
  \Leftrightarrow  & m_N^T S_N^{-1} = (\sigma^{-2}X^Ty + S_0^{-1} m_0)^T \\
  \Leftrightarrow  & m_N^T = S_N (\sigma^{-2}X^Ty + S_0^{-1} m_0) \\
 \end{align*}
$$

---
---

##### Refresher: Linear Algebra

For further details, check <a href="{{ site.assets }}/the_matrix_cookbook.pdf">The Matrix Cookbook</a> for a more detailed list of matrix operations.

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
