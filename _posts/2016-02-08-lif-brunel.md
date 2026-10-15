---
layout: post
title:  "The Leaky Integrate-and-Fire Neuron Model and The Brunel Network"
categories: [neural dynamics, leaky integrate-and-fire, brunel network]
tags: [neuraldynamics]
---

[lif-gerstner]:https://neuronaldynamics.epfl.ch/online/Ch1.S3.html
[nest-brunel]:https://link.springer.com/chapter/10.1007/978-94-007-3858-4_18
[brunel-paper]:https://link.springer.com/article/10.1023/A:1008925309027
[brunel-paper-pdf]:https://web.stanford.edu/group/brainsinsilicon/documents/BrunelSparselyConnectedNets.pdf
[nest]:http://www.nest-simulator.org/
[euler]: https://en.wikipedia.org/wiki/Euler_method

The Leaky Integrate-and-Fire neuron model (LIF, first described in 1907 by [Louis Lapicque](https://en.wikipedia.org/wiki/Louis_Lapicque)) simplifies the complexity of neurons and assumes the neuron structure to be a single leaky compartment (*point neuron model*) with a membrane described by an RC circuit, as:

{: style="text-align:center; font-size: small;"}
<img width="20%" height="20%" src="{{ site.assets }}/lif-brunel/rc.png">

{: style="text-align:center; font-size: small;"}
source: Neuronal Dynamics, Gerstner et al., Cambridge University Press

The main formulation of its electrical activity is given by the **current equation**, with the following terms:
- The current in the capacitor $$C$$ is given by $$I_C = \frac{dQ}{dt} = \frac{d CV}{dt} = C \frac{dV}{dt}$$, for charge $$Q$$ and capacitance $$C$$;
- The current in the resistance $$R$$ is given by Ohm's law as $$I_R = V/R$$, for voltage $$V$$ and resistance $$R$$. 

The law of conservation of electric charge states that the total charge of an isolated system remains constant. Thus, the sum of all currents from the capacitor $$I_C$$, the resistance $$I_R$$ and all remaining currents $$I$$ must be zero. Equivalently:

$$
C \frac{dV}{dt}(t) = -\frac{V(t)}{R} + I(t) 
$$

It is common to multiply all terms by $$R$$ to expose the time **constant** $$\tau = R C$$. This is a very important metric in the analysis of the behaviour of models, as it tells how *fast* the dynamics of the neuron are. The formulation then becomes:

$$
\tau \frac{dV}{dt}(t) = -V(t) + RI(t) 
\label{equation_diff}
$$

We add the external currents (such as synaptic activity) to $$I(t)$$. In practice, the network contribution $$RI(t)$$ is the sum of all currents delivered from any neuron $$j$$ to a given neuron $$i$$ over time, and is described as:

$$
RI(t)_i = \tau \sum_{j} J_{ij} \sum_{t^\prime} \delta (t-t^\prime _{j} -D)
\label{equation_current}
$$

where $$J_{ij}$$ is the **postsynaptic potential** (PSP) amplitude, 
$$\delta (t)$$ the Dirac function, $$t^\prime _{j}$$ the spiking times $$t^\prime$$ of the pre-synaptic neuron $$j$$, and 
$$D$$ the transmission delay. For simplicity, we represent $$V(t)$$ as $$V_t$$. 

We consider that the neuron's potential rises steeply when there is an input current, thus discarding the exponential charging of the capacitor. On the other hand, the discharge follows an exponential decay with the time constant $$\tau$$.

To simulate the [Action Potential](https://en.wikipedia.org/wiki/Action_potential) (or *spike*) of biological neurons, we assume that each neuron has a certain *fixed* firing threshold. When the potential reaches the firing threshold, the neuron *spikes* (or *fires*), and its potential immediately drops to a reset potential. Axonal branches of neurons are simulated by a **synaptic delay** that regulates the time a spike signal takes to reach the connected post-synaptic neurons. This propagation of spike and current, which dictates the synaptic delay, is illustrated in the following picture:

{: style="text-align:center; font-size: small;"}
<img width="40%" height="40%" src="{{ site.assets }}/lif-brunel/action_potential.gif">source: Wikipedia

Afterwards, for a given **refractory period**, the neuron potential stays constant at the reset value, to simulate the [refraction](https://en.wikipedia.org/wiki/Refractory_period_(physiology)) of biological neurons. This may not be a very detailed simulation of neuron activity, but it is a good starting point for a simple neuron model, before we move on to more complex models in future posts.

## Resolution

We compute two different stepping methods from the analytical solution of the first-order ODE, both based on a two-step algorithm:
- Calculate the decay from the previous step: $$V_t \leftarrow V_{t- \Delta t} \exp(\frac{-\Delta t}{\tau})$$  
- Add the voltage change due to the network current $$RI$$ between $$t-\Delta t$$ and $$t$$: $$V_t \leftarrow  V_t + RI_{(t-\Delta t, t]} / \tau $$

We can advance neurons with fixed-step or variable-step interpolation.
- **Fixed step** interpolation assumes the interpolation interval $$\Delta t$$ to be constant throughout the execution.
- **Variable step** methods advance the neuron with the smallest $$t$$ in the network, with a $$\Delta t$$ computed as the minimum of the following two values:
  - the time difference to the next incoming spike;
  - $$t^\star + D - t$$, where $$t^\star$$ is the time of the neuron with the second smallest $$t$$ and $$D$$ is the synaptic delay, i.e. the largest step that can be taken from the neuron at $$t$$ without missing a spike from the neuron at time $$t^\star$$ (if it spikes).  

We can also solve the fixed and variable step interpolations numerically, instead of using the analytical solution.
This is common practice for efficiency purposes, as the required exponential and division operations are computationally costly. The solution is not exact, but it approximates the exact solution *well enough* when the timestep $$\Delta t$$ is *small enough*. As a simple example, we can use two approximate fixed-step [Euler methods][euler]:
- **Explicit Forward Euler:** 

$$
\begin{align*}
& \tau \frac{dV_{t - \Delta t}}{dt} = -V_{t - \Delta t} + RI_{(t-\Delta t, t]} / \Delta t \\
\Leftrightarrow & V_t = V_{t - \Delta t} +  \frac{dV_{t- \Delta t}}{dt} \Delta t \\
\Leftrightarrow & V_t = V_{t - \Delta t} + \frac{-V_{t - \Delta t} + RI_{(t-\Delta t, t]} / \Delta t}{\tau} \Delta t
\end{align*}
$$

- **Implicit Backward Euler:**

$$
\begin{align*}
& \tau \frac{dV_{t}}{dt} = -V_{t} + RI_{(t-\Delta t, t]} / \Delta t \\
\Leftrightarrow & \tau \frac{V_{t} - V_{t - \Delta t}}{\Delta t} = -V_{t} + RI_{(t-\Delta t, t]} / \Delta t \\
\Leftrightarrow & V_t = \frac{RI_{(t-\Delta t, t]} + \tau * V_{t-\Delta t}}{\Delta t + \tau}
\end{align*}
$$

with $$\frac{dV_{t}}{dt}$$ and $$RI_{(t-\Delta t, t]}$$ computed from equations \ref{equation_diff} and \ref{equation_current}, respectively. Note that $$RI_{(t-\Delta t, t]}$$ is divided by $$\Delta t$$ so that it is expressed in voltage per time unit instead of per timestep.

For the sake of comparison, we illustrate below the potential over time of a given neuron in a network, for the aforementioned methods, on a $$100ms$$ simulation of a (Brunel) network of $$10000$$ neurons with default neuron parameters. Results are given for the Implicit and Explicit Euler methods with $$\Delta t = 0.1ms$$, and for the analytical solution with fixed ($$\Delta t=0.1ms$$) and variable timestepping:

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="{{ site.assets }}/lif-brunel/neuron_plot_equations.png">

Note the vertical straight lines, which refer to the voltage reset after a spike, and the horizontal straight lines, which refer to the refractory period after firing.

## Brunel Network

We've just mentioned network activity in the previous section. Indeed, a neuron without any kind of stimulus (either electric current injection or synaptic currents) will always remain at rest.

The most common model of network activity is called the Brunel Network Model ([Dynamics of Sparsely Connected Networks of Excitatory and Inhibitory Spiking Neurons, Brunel, Journal of Computational Neuroscience][brunel-paper], [pdf][brunel-paper-pdf]). It is highly relevant, as it provides a good estimation of network dynamics. The model proposes a network of 10000 simulated LIF neurons (80% excitatory, 20% inhibitory), with external stimuli provided by excitatory neurons outside the network, which connect to both the excitatory and the inhibitory neurons. Excitatory means that the PSP is positive; the converse holds for inhibitory neurons. The external neurons are not simulated, so their spiking is generated by a Poisson random generator (PG). The diagram is the following:

{: style="text-align:center; font-size: small;"}
<img width="30%" height="30%" src="{{ site.assets }}/lif-brunel/network.png">

{: style="text-align:center; font-size: small;"}
source: Brunel 2000

The tricky part here is to define the parameters of the network in terms of connectivity, synaptic weights, firing threshold, etc. The original [paper][brunel-paper] describes them in detail, but for simplicity, we summarize them in the table below: 

{: style="text-align:center; font-size: small;"}
<img width="50%" height="50%" src="{{ site.assets }}/lif-brunel/summary-of-the-network-parameters-for-the-model-proposed-by-brunel-2000.png">

{: style="text-align:center; font-size: small;"}
source: Marc-Oliver Gewaltig, ResearchGate

We can now adjust the stimuli parameters to regulate network activity. Brunel defines a model for slow, regular and fast network dynamics, which exhibit the following average neuron spiking rates (Hz) and global oscillation frequencies (Hz) of voltage trajectories:

{: style="text-align:center; font-size: small;"}
<img width="30%" height="30%" src="{{ site.assets }}/lif-brunel/frequencies.png">

{: style="text-align:center; font-size: small;"}
source: Brunel 2000

Notice how close the simulated network is to the theoretical predictions. For that reason, although it is a simple approximation, the Brunel Network is a powerful tool for studying network dynamics. The model is particularly useful in very long simulations of large networks, as it is computationally inexpensive compared to more detailed models such as the [Hodgkin-Huxley](https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model) model.

## Output

We implemented a `C++`-based Brunel network model of LIF neurons and injected the appropriate stimuli to reproduce the results of the original paper. The outputs match the theoretical model in terms of spike rate and trajectory oscillation frequency. Results are presented as a timeline (left to right) of neuron spike times (a.k.a. *spike trace*, top) and the number of neurons that spike at a given instant (bottom). Enjoy.
 
**Slow dynamics**:

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="{{ site.assets }}/lif-brunel/raster_plot_g4.5_f0.9.png">

**Regular dynamics**:

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="{{ site.assets }}/lif-brunel/raster_plot_g5_f2.png">

**Fast dynamics**:

{: style="text-align:center; font-size: small;"}
<img width="100%" height="100%" src="{{ site.assets }}/lif-brunel/raster_plot_g3_f2.png">


## Other resources

- For a thorough explanation of the Leaky Integrate-and-Fire model, check the relevant section in the online copy of the [Neuronal Dynamics book from Wulfram Gerstner at EPFL][lif-gerstner].
- For a `Python` implementation of a Brunel network using the [NEST][nest] simulator, check [NEST by Example: An Introduction to the Neural Simulation Tool NEST, Marc-Oliver Gewaltig et al., Springer Link][nest-brunel].
- For the `C++` code of this exercise, download <a href="{{ site.assets }}/lif-brunel/lif-brunel.tar.gz">lif-brunel.tar.gz</a>. To help understand the code, the class diagram is available in <a href="{{ site.assets }}/lif-brunel/diagram_class.png">diagram_class.png</a>.
