---

layout: post
title:  "Reinforcement Learning: an overview of methods"
date:   2018-04-27
categories: [machine learning, reinforcement learning]
tags: [machinelearning]
---

<div class="alert alert-primary" role="alert">
Unlike most posts that provide a thorough study on a particular subject, this post provides a very high level description of several reinforcement learning methods that I came across. Its content may not be entirely correct and is continuously updated.
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

### TD-Learning, SARSA and Q-Learning 

[Temporal difference (TD) learning](https://en.wikipedia.org/wiki/Temporal_difference_learning) refers to a class of model-free reinforcement learning methods which learn by estimating the value function from the current state. These methods sample from the environment, and perform updates based on current estimates, like dynamic programming methods.

 The goal of TD-learning is to learn a policy, which tells an agent what action to take under what circumstances. TD-learning maximizes the expected value of the total reward over successive states, starting from the current state. It involves an agent, a set of states $S$, and a set of $A$ of actions that may be taken. By performing an action $a \in A$, the agent transitions between states. The algorithm has a function that calculates the quality (*expected reward*) of a state-action combination:

$$
Q : S \times A \rightarrow R
$$

In practice, at any given state $s$, taking the action $a$ gives the expected reward of:

$$
Q(s,a) = \sum_{s'} P_{s \rightarrow s'}^a R_{s \rightarrow s'}^a
\label{eq_q}
$$


<p align="center">
<img width="20%" height="20%" src="/assets/2018-Reinforcement-Learning/q-value.png"><br/>
<small>source: Lecture notes, Unsupervised and Reing. Learning, M.O. Gewaltig, EPFL</small>
</p>

At every iteration, we take the action that is more *promising*, i.e. with the highest expected reward. We choose an action $$ a^{\star} $$ such that $$ Q(s,a^{\star}) \gt Q(s,a_j) $$ for all actions $a_j \in A$. The iterative update of Q values is provided by the rule

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

where $\pi$ is the policy chosen --- covered next. I'll skip the details. A thorough cover on optimality in policies is provided by <a href="/assets/2018-Reinforcement-Learning/Ag4-4x.pdf">Ag4-4x.pdf</a>.

To summarize, the SARSA algorithm follows as:
- Being in stat $s$ choose action $a$ according to the policy $\pi$;
- Observe reward $r$ and next state $s'$;
- Choose action $a'$ in state $s'$ according to policy;

##### Q-Learning

Another method for TD learning is the Q-learning. It's very simillar to the SARSA method described:
- SARSA, an on-policy method: $$ \Delta Q(s,a) = \eta [ r - Q(s,a)  - \gamma Q(s',a') ] e_{aij} $$
  - The same policy is used to select the next action and to update the Q-values
- Q-Learning, an off-policy method: $$ \Delta Q(s,a) = \eta [ r - Q(s,a)  - \gamma \text{ max } Q(s',a') ] e_{aij} $$
  - the action is selected using policy A e.g. soft-max;
  - the Q-values are updated using policy C e.g. greedy;

##### Eligibility Trace

A third term called the **eligibility trace** can be added to the model. An eligibility trace is a temporary record of the occurrence of an event, such as the visiting of a state or the taking of an action. The trace marks the memory parameters associated with the event as eligible for undergoing learning changes. In practice, it's a decay term on the expected reward of each state, based on the number of steps taken from the actual step to the one where the eligibility trace is computed from. This leads to the updated formulation \ref{sarsa}:

$$
\Delta Q(s,a) = \eta [ r - Q(s,a)  - \gamma Q(s',a') ] e_t(s,a)
$$

where the elegibility trace $e$ is given by

$$
e_t(s,a) = 
\begin{cases}
    \lambda e_{t-1} (s,a) + 1 ,& \text{if action $a$ was taken in state $s$ }\\
    \lambda e_{t-1} (s,a),             & \text{otherwise}
\end{cases}
$$

In practice, the eligibility decays the expected reward with a factor $\lambda$ and acts as a forget mechanism. If we imagine a 4x4 space and an action of moving up/down/left/right, the with and without eligibility trace models are illustrated as (green arrow is the expected reward of taking the green-arrow action at each state):

<p align="center">
<img width="20%" height="20%" src="/assets/2018-Reinforcement-Learning/eligibility2.png">
<span style="display:inline-block; width:1cm;"></span>
<img width="20%" height="20%" src="/assets/2018-Reinforcement-Learning/eligibility1.png">
<br/>
<small>source: Unsupervised and Reinforcement Learning lecture notes, MO Gewaltig, EPFL</small>
</p>

##### Policies

Due to the existance of an eligibility trace, it's easy to see that once a first path is successfully taken to the state of reward, it will always be followed if we simply follow the history of steps with maximum reward. To overcome this, the problem of **exploration vs exploitation** of space relates how much a model sticks to the previous state analysis versus how much he's allowed to deviate and try new paths. Among several possible **policies**, common approaches are:
- greedy strategy: always take action $$ a^\star $$ such that $$ Q(s, a^\star) = Q(s,a_j) $$;
- Epsilon greedy: have a probability $$ P = 1 - \epsilon $$ of following the best known path, or $ \epsilon $ of exploring a new one;
- [Softmax strategy](https://www.sciencedirect.com/science/article/pii/S0925231217310974): see [here](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-7-action-selection-strategies-for-exploration-d3a97b7cceaf)
- Optimistic greedy: initializes the estimated reward for each action to a high value, in order to prevent locking to a sub-optimal action;

##### Examples

Take an extended version of the navigation problem mentioned above. We plan to probam a mouse in a dark room:
- The room is made of 6x6 tiles (possible states);
- The possible actions are: move up, down, left or right;
- Hitting the wall of the room leads to a negative reward;
- Stepping in the same tile as a cat leads to a highly negative reward, and a restart of the problem;
  - In this training model we must assum that agents (mice) have memory after each restart;
- Stepping on a tile that surrounds the cheese leads to a small positive (expectation of) reward;
- Funding cheese is the ultimate reward and success; 

<p align="center">
<img width="30%" height="30%" src="/assets/2018-Reinforcement-Learning/problem.png">
</p>

The `python` source code that resolves this problem is available in <a href="/assets/2018-Reinforcement-Learning/gridworld.py">gridworld.py</a>. If you are keen to understand more, you can play with the eligibility constant, exploration (epsilon) constant, andlearning rates to fine tune the problem. 

A more sophisticated example of TD-learning was provided by Gerry Tesauro for the backgammon game. The detailed explanation is available in <a href="/assets/2018-Reinforcement-Learning/td-gammon.pdf">td-gammon.pdf</a>. 

<p align="center">                                                      
<img width="30%" height="30%" src="/assets/2018-Reinforcement-Learning/gammon.png">
</p>       

##### Continuous State spaces

Notice that the list of states $S$ described so far are represented on a discrete dataset. To properly represent a continuous space, we alter the state-action value function to 

$$
Q(s,a) = \sum_j w_{aj} r_j(s) = \sum_j w_{aj} \phi (s-s_j)
$$

where $\phi$ is the basis function ($s_j$ represents the centers of the BF shape), an approximation of reward based on the known rewards of the proximal discrete states. The eligibility trace follows accordingly, with $$ e=\phi(s-s_j) $$ if action $a$ taken, and $0$ otherwise. For the sake of clarity, in the following figure we illustrate a problem with 3 states and 3 actions per state, with known $Q$ labelled by the height of the blue bars, and the expected $Q$ on the continous space as blue curves.

<p align="center">
<img width="30%" height="30%" src="/assets/2018-Reinforcement-Learning/continuous_space.png"><br/>
<small>source: Lecture notes, Unsupervised and Reing. Learning, M.O. Gewaltig, EPFL</small>
</p>


##### Limitations

A major limitation of the simple TD-learning methods we presented so far is that it requires memory to store the Q values for all combinations of states and actions. Moreover, it is a linear function approximation, therefore not commonly used. However, it is relevant to mention that its application with multiple neuron layer architectures provide an approximation for non-linear functions, such as the TD-gammon examples mentioned previously, and the deeply-reinforced  (deep Q-learning) learning model presented by [Google Deepmind ATARI player](https://www.nature.com/articles/nature14236) ( [pdf](/assets/2018-Reinforcement-Learning/MnihEtAlHassibis15NatureControlDeepRL.pdf) ). 

### Policy Gradient

<small>This section was largely inspired on the lecture notes of MO Gewaltig, EPFL</small>

The main idea of policy gradient (PL) is to associate actions with stimuli in a stochastic fashion. While in Q-learning we store the table that provides a $Q$ value per state and action, in the Policy gradient we store the reward for an action given a stimulus and observation. PL makes a lot of sense, since we can only learn from states we visited (therefore requiring lot of storage and training iterations for large state sets), but we can learn from few observations that are common to many states, therefore providing learning for unseen states. As an example, while one could train a racing car in TD learning by navigating all locations in all existing tracks, with Policy graidient we need only to train the car for the observations (traffic signs), which are common to all tracks. 

The main idea is to optimize the total expected reward:

$$
J = < < R >_a >_s = \sum_s P(s) \sum_a \pi (a \mid s) R (a \mid s)
$$

for a stimuli $s$ and actions $a$. Thus we compute the weight gradient as:

$$
\Delta w_i = \eta \frac{\partial}{\partial w_i} < < R >_a >_s \\
= \eta \frac{\partial}{\partial w_i} \sum_s P(s) \sum_a \pi (a \mid s) R (a \mid s) \\
= \eta \sum_s P(s) \sum_a \frac{\partial}{\partial w_i} \pi (a \mid s) R (a \mid s) \\
= \eta \sum_s P(s) \sum_a \pi (a \mid s) \frac{\partial}{\partial w_i} ln (\pi (a \mid s)) R (a \mid s) \text{, the log-likelihood trick} \\
= < < R \frac{\partial}{\partial w_i} ln (\pi (a \mid s)) > >
$$

Taking the sample average as Monte Carlo approximation of this expectation, by taking N trial histories, we get:

$$
\approx \eta \frac{1}{N} \sum_{n=1} \frac{\partial}{\partial w_i} \pi (s_n) R (a_n \mid s_n)
$$

which is a fast approximation of the policy gradient.

<div class="alert alert-warning" role="alert">
I am looking for a nice application of policy gradient, with coding examples. I will try to finish this section in the near future.
</div>
