---
layout: post
title:  "The Leaky Integrate-and-Fire Neuron Model and The Brunel Network"
categories: ["Computational Neuroscience"]
tags: ["spiking neurons", "LIF", "Brunel network", "simulation"]
---

[lif-gerstner]:https://neuronaldynamics.epfl.ch/online/Ch1.S3.html
[nest-brunel]:https://link.springer.com/chapter/10.1007/978-94-007-3858-4_18
[brunel-paper]:https://link.springer.com/article/10.1023/A:1008925309027
[brunel-paper-pdf]:https://web.stanford.edu/group/brainsinsilicon/documents/BrunelSparselyConnectedNets.pdf
[nest]:http://www.nest-simulator.org/
[euler]: https://en.wikipedia.org/wiki/Euler_method

The Leaky Integrate-and-Fire neuron model (LIF, first described in 1907 by [Louis Lapicque](https://en.wikipedia.org/wiki/Louis_Lapicque)) simplifies the neuron to a single leaky compartment (*point neuron model*), with a membrane described by an RC circuit:

{: .picture}
<img width="22.5%" height="22.5%" src="{{ site.assets }}/lif-brunel/rc.svg">

{: .caption}
A neuron receives an input current $$I(t)$$. Its membrane acts as a capacitor $$C$$ in parallel with a resistor $$R$$, in series with a battery of potential $$u_{\text{rest}}$$. Adapted from: [Neuronal Dynamics, Fig. 1.6](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)

The electrical activity of the model is described by the **current equation**, with the following terms:
- The current in the capacitor $$C$$ is $$I_C = \frac{dQ}{dt} = \frac{d CV}{dt} = C \frac{dV}{dt}$$, for charge $$Q$$ and capacitance $$C$$;
- The current through the resistance $$R$$ follows Ohm's law, $$I_R = V/R$$, for voltage $$V$$ and resistance $$R$$. 

Kirchhoff's current law, a consequence of the conservation of electric charge, states that the currents entering a node sum to the currents leaving it. Thus, the input current $$I$$ splits into the capacitor current $$I_C$$ and the resistance current $$I_R$$, i.e. $$I = I_C + I_R$$. Equivalently:

$$
C \frac{dV}{dt}(t) = -\frac{V(t)}{R} + I(t) 
$$

Multiplying all terms by $$R$$ exposes the time **constant** $$\tau = R C$$. This constant is very important in the analysis of model behaviour, as it determines how *fast* the neuron dynamics are. The formulation becomes:

$$
\tau \frac{dV}{dt}(t) = -V(t) + RI(t) 
\label{equation_diff}
$$

External currents (such as synaptic activity) are added to $$I(t)$$. In practice, the network contribution $$RI(t)$$ is the sum of all currents delivered from any neuron $$j$$ to a given neuron $$i$$ over time:

$$
RI(t)_i = \tau \sum_{j} J_{ij} \sum_{t^\prime} \delta (t-t^\prime _{j} -D)
\label{equation_current}
$$

where $$J_{ij}$$ is the **postsynaptic potential** (PSP) amplitude, 
$$\delta (t)$$ the Dirac function, $$t^\prime _{j}$$ the spiking times $$t^\prime$$ of the pre-synaptic neuron $$j$$, and 
$$D$$ the transmission delay. For simplicity, $$V(t)$$ is written as $$V_t$$. 

The model assumes that the neuron's potential rises steeply when there is an input current, i.e. the exponential charging of the capacitor is ignored. The discharge, on the other hand, follows an exponential decay with time constant $$\tau$$.

To simulate the [Action Potential](https://en.wikipedia.org/wiki/Action_potential) (or *spike*) of biological neurons, each neuron has a *fixed* firing threshold. When the potential reaches this threshold, the neuron *spikes* (or *fires*), and its potential immediately drops to a reset potential. Axonal branches are simulated by a **synaptic delay**, i.e. the time a spike signal takes to reach the connected post-synaptic neurons. The following picture illustrates this propagation of spike and current, which determines the synaptic delay:

{: .picture}
<img width="40%" height="40%" src="{{ site.assets }}/lif-brunel/action_potential.gif">

{: .caption}
An action potential travelling along an axon. Source: [Action Potential.gif, Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Action_Potential.gif)

After a spike, the neuron potential stays constant at the reset value for a given **refractory period**, to simulate the [refraction](https://en.wikipedia.org/wiki/Refractory_period_(physiology)) of biological neurons. This is not a detailed simulation of neuron activity, but it is a good starting point for a simple neuron model, before more complex models in future posts.

## Resolution

Two stepping methods are derived from the analytical solution of the first-order ODE. Both are based on the following two-step algorithm:
- Compute the decay from the previous step: $$V_t \leftarrow V_{t- \Delta t} \exp(\frac{-\Delta t}{\tau})$$  
- Add the voltage change due to the network current $$RI$$ between $$t-\Delta t$$ and $$t$$: $$V_t \leftarrow  V_t + RI_{(t-\Delta t, t]} / \tau $$

Neurons are advanced with fixed-step or variable-step interpolation:
- **Fixed step** interpolation keeps the interpolation interval $$\Delta t$$ constant throughout the execution.
- **Variable step** methods advance the neuron with the smallest $$t$$ in the network, with a $$\Delta t$$ equal to the minimum of the following two values:
  - the time difference to the next incoming spike;
  - $$t^\star + D - t$$, where $$t^\star$$ is the time of the neuron with the second smallest $$t$$ and $$D$$ is the synaptic delay. This is the largest step the neuron at $$t$$ can take without missing a spike from the neuron at time $$t^\star$$ (if it spikes).  

The fixed and variable step interpolations can also be solved numerically, instead of with the analytical solution.
This is common practice for efficiency, as the exponential and division operations are computationally costly. The numerical solution is not exact, but it approximates the exact solution *well enough* when the timestep $$\Delta t$$ is *small enough*. Two simple examples are the approximate fixed-step [Euler methods][euler]:
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

with $$\frac{dV_{t}}{dt}$$ and $$RI_{(t-\Delta t, t]}$$ computed from equations \ref{equation_diff} and \ref{equation_current}, respectively. Note that $$RI_{(t-\Delta t, t]}$$ is divided by $$\Delta t$$ to express it in voltage per time unit instead of per timestep.

For comparison, the plot below shows the potential over time of one neuron in a network for each of these methods, on a $$100ms$$ simulation of a (Brunel) network of $$10000$$ neurons with default neuron parameters. Results are shown for the Implicit and Explicit Euler methods with $$\Delta t = 0.1ms$$, and for the analytical solution with fixed ($$\Delta t=0.1ms$$) and variable timestepping:

{: .picture}
<img width="99.9%" height="99.9%" src="{{ site.assets }}/lif-brunel/neuron_plot_equations.svg">

The vertical straight lines are the voltage resets after a spike, and the horizontal straight lines are the refractory periods after firing.

## Brunel Network

The previous section mentioned network activity. Indeed, a neuron without any stimulus (either injected electric current or synaptic currents) always remains at rest.

The most common model of network activity is the Brunel Network Model ([Dynamics of Sparsely Connected Networks of Excitatory and Inhibitory Spiking Neurons, Brunel, Journal of Computational Neuroscience][brunel-paper], [pdf][brunel-paper-pdf]). It is highly relevant, as it provides a good estimation of network dynamics. The model proposes a network of 10000 simulated LIF neurons (80% excitatory, 20% inhibitory). External stimuli come from excitatory neurons outside the network, which connect to both the excitatory and the inhibitory neurons. Excitatory means that the PSP is positive, and inhibitory means that it is negative. The external neurons are not simulated: their spikes are generated by a Poisson random generator (PG). The network is illustrated below:

{: .picture}
<img width="33.4%" height="33.4%" src="{{ site.assets }}/lif-brunel/network.svg">

{: .caption}
The Brunel network: $$N_E$$ excitatory neurons (E) and $$N_I$$ inhibitory neurons (I). Each neuron receives $$C_E$$ connections from excitatory neurons, $$C_I$$ from inhibitory neurons, and $$C_E$$ from Poisson generators (PG). Arrows are excitatory connections, and open circles inhibitory ones. Adapted from: [NEST by Example, Fig. 2](https://www.nest-simulator.org/wp-content/uploads/2015/02/NEST_by_Example.pdf)

The hard part is defining the network parameters, such as connectivity, synaptic weights and firing threshold. The original [paper][brunel-paper] describes them in detail, and the table below summarizes them: 

<table markdown="1">
  <thead>
    <tr><th colspan="2" style="text-align: left">Network Parameters</th></tr>
    <tr><th style="text-align: left">Parameter</th><th style="text-align: right">Value</th></tr>
  </thead>
  <tbody>
    <tr><td style="text-align: left" markdown="span">Number of excitatory neurons $$N_E$$</td><td style="text-align: right">8000</td></tr>
    <tr><td style="text-align: left" markdown="span">Number of inhibitory neurons $$N_I$$</td><td style="text-align: right">2000</td></tr>
    <tr><td style="text-align: left" markdown="span">Excitatory synapses per neuron $$C_E$$</td><td style="text-align: right">800</td></tr>
    <tr><td style="text-align: left" markdown="span">Inhibitory synapses per neuron $$C_I$$</td><td style="text-align: right">200</td></tr>
  </tbody>
</table>

<table markdown="1">
  <thead>
    <tr><th colspan="2" style="text-align: left">Neuron Parameters</th></tr>
    <tr><th style="text-align: left">Parameter</th><th style="text-align: right">Value</th></tr>
  </thead>
  <tbody>
    <tr><td style="text-align: left" markdown="span">Membrane time constant $$\tau_m$$</td><td style="text-align: right">20 ms</td></tr>
    <tr><td style="text-align: left" markdown="span">Refractory period $$\tau_{\text{rp}}$$</td><td style="text-align: right">2 ms</td></tr>
    <tr><td style="text-align: left" markdown="span">Firing threshold $$V_{\text{th}}$$</td><td style="text-align: right">20 mV</td></tr>
    <tr><td style="text-align: left" markdown="span">Membrane capacitance $$C_m$$</td><td style="text-align: right">1pF</td></tr>
    <tr><td style="text-align: left" markdown="span">Resting potential $$V_E$$</td><td style="text-align: right">0 mV</td></tr>
    <tr><td style="text-align: left" markdown="span">Reset potential $$V_{\text{reset}}$$</td><td style="text-align: right">10 mV</td></tr>
    <tr><td style="text-align: left" markdown="span">Excitatory PSP amplitude $$J_E$$</td><td style="text-align: right">0.1 mV</td></tr>
    <tr><td style="text-align: left" markdown="span">Inhibitory PSP amplitude $$J_I$$</td><td style="text-align: right">-0.5 mV</td></tr>
    <tr><td style="text-align: left" markdown="span">Synaptic delay $$D$$</td><td style="text-align: right">1.5 ms</td></tr>
    <tr><td style="text-align: left" markdown="span">Background rate $$\eta$$</td><td style="text-align: right">2.0</td></tr>
  </tbody>
</table>

{: .caption}
source: Marc-Oliver Gewaltig, ResearchGate

Network activity is regulated by adjusting the stimuli parameters. Brunel defines models for slow, regular and fast network dynamics, with the following average neuron spiking rates (Hz) and global oscillation frequencies (Hz) of voltage trajectories:

<table>
  <thead>
    <tr><th rowspan="2"></th><th colspan="2" style="text-align: center">Firing rate</th><th colspan="2" style="text-align: center">Global frequency</th></tr>
    <tr><th style="text-align: center">Simulation</th><th style="text-align: center">Theory</th><th style="text-align: center">Simulation</th><th style="text-align: center">Theory</th></tr>
  </thead>
  <tbody>
    <tr><td style="text-align: left">SI, fast</td><td style="text-align: center">60.7 Hz</td><td style="text-align: center">55.8 Hz</td><td style="text-align: center">180 Hz</td><td style="text-align: center">190 Hz</td></tr>
    <tr><td style="text-align: left">AI</td><td style="text-align: center">37.7 Hz</td><td style="text-align: center">38.0 Hz</td><td style="text-align: center">—</td><td style="text-align: center">—</td></tr>
    <tr><td style="text-align: left">SI, slow</td><td style="text-align: center">5.5 Hz</td><td style="text-align: center">6.5 Hz</td><td style="text-align: center">22 Hz</td><td style="text-align: center">29 Hz</td></tr>
  </tbody>
</table>

{: .caption}
source: Brunel 2000

The simulated network is very close to the theoretical predictions. For this reason, although it is a simple approximation, the Brunel Network is a powerful tool for studying network dynamics. It is particularly useful for very long simulations of large networks, as it is computationally inexpensive compared to more detailed models such as the [Hodgkin-Huxley](https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model) model.

## Output

A `C++` implementation of the Brunel network model of LIF neurons, with the appropriate stimuli, reproduces the results of the original paper. The outputs match the theoretical model in terms of spike rate and trajectory oscillation frequency. The results below show a timeline (left to right) of neuron spike times (a.k.a. *spike trace*, top) and the number of neurons that spike at each instant (bottom).
 
**Slow dynamics**:

{: .picture}
<img width="99.9%" height="99.9%" src="{{ site.assets }}/lif-brunel/raster_plot_g4.5_f0.9.svg">

**Regular dynamics**:

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/lif-brunel/raster_plot_g5_f2.svg">

**Fast dynamics**:

{: .picture}
<img width="99.9%" height="99.9%" src="{{ site.assets }}/lif-brunel/raster_plot_g3_f2.svg">


## Other resources

- A thorough explanation of the Leaky Integrate-and-Fire model is available in the relevant section of the online copy of the [Neuronal Dynamics book from Wulfram Gerstner at EPFL][lif-gerstner].
- A `Python` implementation of a Brunel network using the [NEST][nest] simulator is described in [NEST by Example: An Introduction to the Neural Simulation Tool NEST, Marc-Oliver Gewaltig et al., Springer Link][nest-brunel].
- The `C++` code of this exercise is available in <a href="{{ site.assets }}/lif-brunel/lif-brunel.tar.gz">lif-brunel.tar.gz</a>. The class diagram in <a href="{{ site.assets }}/lif-brunel/diagram_class.svg">diagram_class.svg</a> helps to understand the code.
