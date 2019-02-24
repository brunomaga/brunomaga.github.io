---
layout: post
title:  "Unsupervised Learning: an overview"
date:   2017-11-01 12:01:42 +0100
categories: [machine learning, unsupervised learning]
tags: [machilelearning]
---

Among several *biologically-inspired* learning models, Hebbian learning is probably the oldest. Its rationale is based on the synaptic changes for neurons that spike in close instants in time (motto "*if they fire together, they wire together*"). It is knows from the Spike-Timing-Dependent Plasticity ([STDP, Markram et al. Frontiers](https://www.frontiersin.org/articles/10.3389/fnsyn.2012.00002/full)) that the increase/decrease of a given synaptic strenght is correlated with the time difference between a pre- and a post-synaptic neuron spike times. In essence, if a neuron spikes and if it leads to the firing of the output neuron, the synapse is strengthned.

Analogously, in artifical system, the weight of a connection between two artificial neurons adapts accordingly. Mathematically, we can describe the weight $w$ at a future time $n+1$ with Hebbian learning as:

$$
w_{ij}[n+1] = w_{ij}[n] + \eta x_i[n] x_j[n] 
$$

where $\eta$ is the learning rate, $x$ is the output, and $i$ and $j$ are the neuron ids. Hebbian is a **local** learning model i.e. only looks at the two involved neurons and does not take into account the activity in the overall system.  

