---

layout: post
title:  "Reinforcement Learning: an overview of methods"
date:   2018-04-27 12:01:42 +0100
categories: [machine learning, reinforcement learning]
tags: [machinelearning]
---

<div class="alert alert-primary" role="alert">
Unlike most posts that provide a thorough study on a particular subject, this post provides a very high level description of several reinforcement learning methods that I came across. Its content is continuously updated.
</div>

Reinforcement learning is a field of machine learning that --- roughly speaking --- encompasses learning by reward or prediction of reward. It is a topic of high interest as it's claimed to best represent human behaviour, mostly driven by stimuli. We eat because we are hungry, and after eating we are satisfied (positive reward). We feel pain as a negative *reward* of hurting our bodies, and we learn from previous painful experiences, to avoid repetition. We gamble because we predict a possible positive reward from a lucky play (the dopamine effect on the brain). This phenomenon has been first studied by [Pavlov in 1927](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4116985/), and the list of examples is endless. Here I describe some methods of RL that may be useful.

### Restricted Boltzmann Machines

<small> This section is heavily inspired in lecture notes from Prof. Gerstner at EPFL and the wikipedia entry to Restricted Boltzmann Machine</small> 

A Restricted Boltzmann Machine (RBM) is a neural network that learns a probability distribution over a set of inputs. RBMs are a special case of [Boltzmann Machines](https://en.wikipedia.org/wiki/Boltzmann_machine) a.k.a. a stochastic [Hopfield network](https://en.wikipedia.org/wiki/Hopfield_network) with hidden units. A BM is a network of neurons with an energy define for the overall network, producing binary results. Contrarily to Hopfield networks, BMs have stochastic neurons. A RBM is a BM whose neurons must form a [Bipartite Graph](https://en.wikipedia.org/wiki/Bipartite_graph): 

<p align="center">
<img width="20%" height="20%" src="/assets/2018-Reinforcement-Learning/Restricted_Boltzmann_machine.png"><br/>
<small>source: wikipedia</small>
</p>

The standard RBM has binary-value (Boolean/[Bernoulli](https://en.wikipedia.org/wiki/Bernoulli_distribution)) hidden/visible units. It consists of a matrix $W = (w_{ij})$ associated with the connection between hidden units $y_j$ and visible unit $x_i$, and the bias (offset) $a_i$ for the visible units, and $b_j$ for the hidden units. Similarly to a Hopfield network, the energyy of some activation pattern $(x,y)$ is defined as:

$$
E (x,y) = - \sum_i a_i x_i - \sum_j b_j y_j - \sum_i \sum_j w_{i,j} x_i y_j
$$

The probability distribution is defined in terms of the energy function

$$
P(x,x) = \frac{1}{Z} exp(-E(x,y) )
$$

where $Z$ is the [partition function](https://en.wikipedia.org/wiki/Partition_function_(mathematics)). It is compute as the sum of $$ exp(-E(x,y) ) $$ over all possible configurations, i.e. is a normalizing term that ensure that the probability distribution sums to 1. Similarly, the [marginal probability](https://en.wikipedia.org/wiki/Marginal_distribution) of an input vector of booleans is:

$$
P(x) = \frac{1}{Z} \sum_y exp(-E(x,y) )
$$

As RBMs are bipartite graphs, hidden units are independent, and visible units are mutually independent. Thus, the [Conditional Probability](https://en.wikipedia.org/wiki/Conditional_probability) (the probability of an event occuring given that another event has occured) of $x$ given $y$ is:

$$
P (x| y) = \prod_{i=1}^m P (x_i | y)
$$

for $m$ hidden units. The converse holds for $$ P (y \mid x) $$. The individual activation probabilities are given by:

$$
P (x_i = 1 \mid y ) =  \sigma ( b_i + \sum_{i=1}^m w_{i,j} y_i )
$$

Now the important part: training. We aim at maximizing the product of all probabilities aasigned to a training set $V$, i.e.:

$$
argmax_W \prod_{v \in V} P(v)
$$

which is equivalent to maximise the expected log-likelihood of a training sample $v$ extracted from $V$:

$$
argmax_W E [ log P (v) ]
$$

The most common algorithm to optimize the weight vector $W$ is the [Gibbs sampling](https://en.wikipedia.org/wiki/Gibbs_sampling) with [Gradient descent](https://en.wikipedia.org/wiki/Gradient_descent). An example of an application and resolution is provided in two blog posts from a colleague of mine, and are available [here](https://sharkovsky.github.io/2017/06/23/boltzmann-machine.html) and [here](https://sharkovsky.github.io/2017/06/30/practical-boltzmann.html).

### Q-Learning and SARSA 

The goal of Q-learning is to learn a policy, which tells an agent what action to take under what circumstances. Q-learning maximizes the expected value of the total reward over successive states, starting from the current state. It involves an agent, a set of states $S$, and a set of $A$ of actions that may be taken. By performing an action $a \in A$, the agent transitions between states. The algorithm has a function that calculates the quality (*expected reward*) of a state-action combination:

$$
Q : S \times A \rightarrow R
$$

In practice, at any given state $s$, taking the action $a$ gives the expected reward of:

$$
Q(s,a) = \sum_{s'} P_{s \rightarrow s'}^a R_{s \rightarrow s'}^a
$$


<p align="center">
<img width="20%" height="20%" src="/assets/2018-Reinforcement-Learning/q-value.png"><br/>
<small>source: Lecture notes, Unsupervised and Reing. Learning, M.O. Gewaltig, EPFL</small>
</p>

At every iteration, we take the action that is more *promising*, i.e. with the highest expected reward. We choose an action $$ a^{\star} $$ such that $$ Q(s,a^{\star}) \gt Q(s,a_j) $$ for all actions $a_j \in A$. The iterative update of Q-values is provided by the rule

$$
\Delta Q(s,a) = \eta [ r - Q(s,a) ]
$$

where $\eta$ is the learning rate. Convergence is expected in this model, when $ < Q(s,a) > = 0 >$.
This models only has a single-step horizon as it only looks at the next possible state. A common multi-step horizon model is the **State–action–reward–state–action (SARSA)** and is formulated as:

$$
\Delta Q(s,a) = \eta [ r - Q(s,a)  - \gamma Q(s',a') ]
\label{sarsa}
$$

where $s'$ and $a'$ and the state and action at the next step possibly taken.

<p align="center">
<img width="20%" height="20%" src="/assets/2018-Reinforcement-Learning/q-value-2.png"><br/>
<small>source: Lecture notes, Unsupervised and Reing. Learning, M.O. Gewaltig, EPFL</small>
</p>

##### Theorem - Convergence in expectation of SARSA

If the mean update has converged ( $ < \Delta Q (s,a) > =0 > $) with the update rule \ref{sarsa}, then the resulting Q-values solve the [Bellman equation](https://en.wikipedia.org/wiki/Bellman_equation), a necessary condition for optimality associated with the mathematical optimization method known as dynamic programming:

$$
Q (s,a) =  \sum_{s'} P^a_{s \rightarrow s'} [ R^a_{s \rightarrow s'} + \gamma \sum_{a'} \pi (s',a') Q(s',a') ]
$$

I'll omit the details. For further details on optimality in policies, check this <a href="/assets/2018-Reinforcement-Learning/Ag4-4x.pdf">Ag4-4x.pdf</a>.


