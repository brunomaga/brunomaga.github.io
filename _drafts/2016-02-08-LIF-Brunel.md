---
layout: post
title:  "The Leaky Integrate-and-Fire Neuron Model and The Brunel Network"
date:   2016-02-08
categories: [distributed algorithms, parallel algorithms]
tags: [algorithms]
---

The Leaky integrate-and-fire (IF) neuron model is described as an electrical model of a neuron with a resistance R and capacity C:

$$
\tau \frac{dV}{dt}(t) = -V(t) + RI(t) \text{ with } \tau = R C 
\label{equation_diff}
$$

where the network contribution $RI(t)$ from a neuron $j$ to a neuron $i$ is described as :

$$
RI(t)_i = \tau \sum_{j} J_{ij} \sum_{t^\prime} \delta (t-t^\prime _{j} -D)
\label{equation_current}
$$

where $J_{ij}$ is the postsynaptic potential amplitude, 
$\delta (t)$ the dirac function, $t^\prime _{j}$ the spiking time of pre-synaptic neuron $j$ at time $t^\prime$ and 
$D$ the transmission delay. For simplicity we represent $V(t)$ as $V_t$. \\

We consider that the neuron is able to steeply rise its potential when there is an input of current, thus discarding the exponential charge of the capacitance. The discharge follows an exponential decay with the time constant $\tau$. We compute two different stepping methods from the analytical solution: 
- Two step algorithm:
  - Calculate decay from previous step: $V_t \leftarrow V_{t- \Delta t} exp(\frac{-dt}{\tau})$  
  - Add voltage change of network current $RI$ between $t-\Delta t$ and $t$: $V_t \leftarrow  V_t + RI_{(t-\Delta t, t]} / \tau $
- fixed step interpolation assumes $\Delta t$ to be constant throughout the execution. The \textbf{Variable step} method advances the neuron with the smallest $t$ on the network, with a $\Delta t$ computed as the minimum of the two following values:
  - the time difference to the next incoming spike;
  - $t^\star + D$ where $t^\star$ is the time of the neuron with the second smallest $t$, i.e. the largest step that can be taken from neuron at $t$ so that it won't miss a spike from neuron at time $t^\star$, if it spikes.  

For efficiency purposes, we implemented two approximated fixed-step Euler methods:
- **Explicit Forward Euler:** $\tau \frac{dV_{t}}{dt} = -V_{t - \Delta t} + RI_{(t-\Delta t, t]} \Leftrightarrow V_t = V_{t - \Delta t} +  \frac{dV_{t- \Delta t}}{dt} \Delta t \Leftrightarrow  V_t = \frac{-V_{t - \Delta t} + RI_{(t-\Delta t, t]}}{\tau} \Delta t$
- **Implicit Backward Euler:** $\tau \frac{dV_{t}}{dt} = -V_{t} + RI_{(t-\Delta t, t]} \Leftrightarrow \frac{V_{t} - V_{t - \Delta t}}{\Delta t} = -V_{t} + RI_{(t-\Delta t, t]} \Leftrightarrow V_t = \frac{RI_{(t-\Delta t, t]} + \tau * V_{t-\Delta t}}{\Delta t + \tau}$

with $\frac{dV_{t}}{dt}$ and $RI_{(t-\Delta t, t]}$ computed from equations \ref{equation_diff} and \ref{equation_current} respectively\footnote{$RI_{(t-\Delta t, t]}$ is divided by $\Delta t$ so that it is expressed in voltage per time-unit instead of per time-step.}.

We present the potential over time of neuron $0$ for the aforementioned methods on a $100ms$ simulation of a Brunel network of $10000$ neurons with random seed $0$ and default neuron parameters:

<p align="center"><img width="30%" height="30%" src="/assets/2016-LIF-Brunel/network.png"></p>

<p align="center"><img width="30%" height="30%" src="/assets/2016-LIF-Brunel/RC.png"></p>

<p align="center"><img width="100%" height="100%" src="/assets/2016-LIF-Brunel/neuron_plot_equations.png"></p>

[lif-gerstner]:https://neuronaldynamics.epfl.ch/online/Ch1.S3.html
[nest-brunel]:https://link.springer.com/chapter/10.1007/978-94-007-3858-4_18
[brunel-paper]:https://link.springer.com/article/10.1023/A:1008925309027
[brunel-paper-pdf]:https://web.stanford.edu/group/brainsinsilicon/documents/BrunelSparselyConnectedNets.pdf
[nest]:http://www.nest-simulator.org/

##### Output

Slow dynamics:

<p align="center"><img width="100%" height="100%" src="/assets/2016-LIF-Brunel/raster_plot_G4.5_f0.9.png"></p>

Regular dynamics:

<p align="center"><img width="100%" height="100%" src="/assets/2016-LIF-Brunel/raster_plot_G5_f2.png"></p>

Fast dynamics:

<p align="center"><img width="100%" height="100%" src="/assets/2016-LIF-Brunel/raster_plot_G3_f2.png"></p>


##### Other resources
- For a thorough detail of the Leaky Integrate-And-Fire Model, check the relevant section in the online copy of the [Neuronal Dynamics Book from Wulfram Gerster at EPFL][lif-gerstner].
- For a `Python` implementation of a Brunel network using the [NEST][nest] simulator, check [NEST by Example: An Introduction to the Neural Simulation Tool NEST, Marc Oliver, Springer Link][nest-brunel]
- For the `C++` code for this exercise, download <a href="/assets/2016-LIF-Brunel/lif-brunel.tar.gz">lif-brunel.tar.gz</a>. For better code understanding, the diagram of classes is presented in <a href="/assets/2016-LIF-Brunel/diagram_class.png">diagram_class.png</a>.
