---
layout: post
title:  "Statistics for Engineers"
categories: [machine learning, statistics]
tags: [machinelearning]
---


There are two frameworks for statistical modelling, the explanatory model framework and the predictive framework.

The explanatory framework starts from assuming a model to describe the observations. The predictive framework starts from assuming you can find a function $f(x)$ which maps from input $x$ to an output $f(x)$.

We cannot always model the resolution of the observed data, so we introduce **stochasticity** to our model. Stochasticity refers to the property of being well described by a random probability distribution. Although stochasticity and **randomness** are distinct in that the former refers to a modeling approach and the latter refers to phenomena themselves, these two terms are often used synonymously.  The data is viewed as observations from that model.

Probabilistic models can be caracterized  <a href="{{ site.statistics_distributions | replace: 'XXX', 'CONTINUOUS' }}"> continuous </a> and <a href="{{ site.statistics_distributions | replace: 'XXX', 'DISCRETE' }}"> discrete </a> input interval. I've covered different families of distributions in a previous [post]({{ site.baseurl }}{% post_url 2019-11-20-Exponential-Family-Distributions %}) .

The outcome of an experiment, or $what is observed$ is called an observation or a realisation. 
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

