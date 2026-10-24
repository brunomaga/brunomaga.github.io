---
layout: post
title:  "The Leaky Integrate-and-Fire Neuron and the Brunel Network"
categories: ["Computational Neuroscience"]
tags: ["spiking neurons", "LIF", "Brunel network", "simulation"]
---

[lif-gerstner]:https://neuronaldynamics.epfl.ch/online/Ch1.S3.html
[nest-brunel]:https://link.springer.com/chapter/10.1007/978-94-007-3858-4_18
[brunel-paper]:https://link.springer.com/article/10.1023/A:1008925309027
[brunel-paper-pdf]:https://web.stanford.edu/group/brainsinsilicon/documents/BrunelSparselyConnectedNets.pdf
[nest]:http://www.nest-simulator.org/
[euler]: https://en.wikipedia.org/wiki/Euler_method

The Leaky Integrate-and-Fire [neuron model](https://en.wikipedia.org/wiki/Biological_neuron_model) (LIF) was first described in 1907 by [Louis Lapicque](https://en.wikipedia.org/wiki/Louis_Lapicque). This post derives the equations of the LIF neuron, compares exact and approximate methods to solve them, and simulates the Brunel network of excitatory and inhibitory LIF neurons.

The LIF model simplifies the complexity of neurons and assumes the neuron structure to be a single leaky compartment (*point neuron model*) with a membrane described by an [RC circuit](https://en.wikipedia.org/wiki/RC_circuit), as:

{: .picture}
<img width="22.5%" height="22.5%" src="{{ site.assets }}/lif-brunel/rc.svg">

{: .caption}
A neuron receives an input current $$I(t)$$. Its membrane acts as a capacitor $$C$$ in parallel with a resistor $$R$$, in series with a battery of potential $$u_{\text{rest}}$$. Adapted from: [Neuronal Dynamics, Fig. 1.6](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)

The main formulation of its electrical activity is given by the **current equation**, with the following terms:
- The current in the capacitor $$C$$ is given by $$I_C = \frac{dQ}{dt} = \frac{d CV}{dt} = C \frac{dV}{dt}$$, for charge $$Q$$ and capacitance $$C$$;
- The current in the resistance $$R$$ is given by [Ohm's law](https://en.wikipedia.org/wiki/Ohm%27s_law) as $$I_R = V/R$$, for voltage $$V$$ (relative to the resting potential $$u_{\text{rest}}$$) and resistance $$R$$. 

The law of conservation of electric charge states that the total charge of an isolated system remains constant. Thus, the currents entering a node of the circuit sum to the currents leaving it ([Kirchhoff's current law](https://en.wikipedia.org/wiki/Kirchhoff%27s_circuit_laws)), i.e. the input current is $$I = I_C + I_R$$. Equivalently:

$$
C \frac{dV}{dt}(t) = -\frac{V(t)}{R} + I(t) 
$$

It is common to multiply all terms by $$R$$ to expose the **[time constant](https://en.wikipedia.org/wiki/Time_constant)** $$\tau = R C$$. This is a very important metric in the analysis of the behaviour of models, as it tells how *fast* the dynamics of the neuron are. The formulation then becomes:

$$
\tau \frac{dV}{dt}(t) = -V(t) + RI(t) 
\label{equation_diff}
$$

The external currents (such as synaptic activity) are added to $$I(t)$$. In practice, the network contribution $$RI(t)$$ is the sum of all currents delivered from any neuron $$j$$ to a given neuron $$i$$ over time, and is described as:

$$
RI(t)_i = \tau \sum_{j} J_{ij} \sum_{t^\prime} \delta (t-t^\prime _{j} -D)
\label{equation_current}
$$

where $$J_{ij}$$ is the **[postsynaptic potential](https://en.wikipedia.org/wiki/Postsynaptic_potential)** (PSP) amplitude, 
$$\delta (t)$$ the [Dirac function](https://en.wikipedia.org/wiki/Dirac_delta_function), $$t^\prime _{j}$$ the spiking times $$t^\prime$$ of the pre-synaptic neuron $$j$$, and 
$$D$$ the transmission delay. For simplicity, $$V(t)$$ is represented as $$V_t$$. 

The neuron's potential is considered to rise steeply when there is an input current (by $$J_{ij}$$ at each input spike), thus discarding the exponential charging of the capacitor. On the other hand, the discharge follows an exponential decay with the time constant $$\tau$$.

To simulate the [Action Potential](https://en.wikipedia.org/wiki/Action_potential) (or *spike*) of biological neurons, the model assumes that each neuron has a certain *fixed* firing threshold. When the potential reaches the firing threshold, the neuron *spikes* (or *fires*), and its potential immediately drops to a reset potential. Axonal branches of neurons are simulated by a **synaptic delay** that regulates the time a spike signal takes to reach the connected post-synaptic neurons. This propagation of spike and current, which dictates the synaptic delay, is illustrated in the following picture:

{: .picture}
<img width="40%" height="40%" src="https://upload.wikimedia.org/wikipedia/commons/9/95/Action_Potential.gif">

{: .caption}
An action potential travelling along an axon, in the direction of the arrows. The membrane is positive outside and negative inside, except at the action potential, where the polarity is reversed. Source: [Action Potential.gif, Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Action_Potential.gif)

Afterwards, for a given **refractory period**, the neuron potential stays constant at the reset value, to simulate the [refraction](https://en.wikipedia.org/wiki/Refractory_period_(physiology)) of biological neurons. This may not be a very detailed simulation of neuron activity, but it is a good starting point for a simple neuron model, before moving on to more complex models in future posts.

## Resolution

Two different stepping methods are derived from the analytical solution of the first-order [ODE](https://en.wikipedia.org/wiki/Ordinary_differential_equation), both based on a two-step algorithm:
- Calculate the decay from the previous step: $$V_t \leftarrow V_{t- \Delta t} \exp(\frac{-\Delta t}{\tau})$$  
- Add the voltage change due to the network current $$RI$$ between $$t-\Delta t$$ and $$t$$: $$V_t \leftarrow  V_t + RI_{(t-\Delta t, t]} / \tau $$, where $$RI_{(t-\Delta t, t]}$$ is the integral of $$RI$$ over that interval.

Neurons can be advanced with fixed-step or variable-step integration:
- **Fixed step** integration assumes the time step $$\Delta t$$ to be constant throughout the execution.
- **Variable step** methods advance the neuron with the smallest $$t$$ in the network, with a $$\Delta t$$ computed as the minimum of the following two values:
  - the time difference to the next incoming spike, so that each spike is added at its exact time;
  - $$t^\star + D - t$$, where $$t^\star$$ is the time of the neuron with the second smallest $$t$$ and $$D$$ is the synaptic delay, i.e. the largest step that can be taken from the neuron at $$t$$ without missing a spike from the neuron at time $$t^\star$$ (if it spikes).  

The fixed and variable step integrations can also be computed numerically, instead of using the analytical solution.
This is common practice for efficiency purposes, as the required exponential and division operations are computationally costly when $$\Delta t$$ changes at every step. The solution is not exact, but it approximates the exact solution *well enough* when the timestep $$\Delta t$$ is *small enough*. As a simple example, two approximate fixed-step [Euler methods][euler] can be used:
- **Explicit Forward Euler:** 

$$
\begin{align*}
& \tau \frac{dV_{t - \Delta t}}{dt} = -V_{t - \Delta t} + RI_{(t-\Delta t, t]} / \Delta t \\
& V_t = V_{t - \Delta t} +  \frac{dV_{t- \Delta t}}{dt} \Delta t \\
& \qquad \text{(forward difference)} \\
\Leftrightarrow & V_t = V_{t - \Delta t} + \frac{-V_{t - \Delta t} + RI_{(t-\Delta t, t]} / \Delta t}{\tau} \Delta t
\end{align*}
$$

- **Implicit Backward Euler:**

$$
\begin{align*}
& \tau \frac{dV_{t}}{dt} = -V_{t} + RI_{(t-\Delta t, t]} / \Delta t \\
& \tau \frac{V_{t} - V_{t - \Delta t}}{\Delta t} = -V_{t} + RI_{(t-\Delta t, t]} / \Delta t \\
& \qquad \text{(backward difference)} \\
\Leftrightarrow & V_t = \frac{RI_{(t-\Delta t, t]} + \tau * V_{t-\Delta t}}{\Delta t + \tau}
\end{align*}
$$

with $$\frac{dV_{t}}{dt}$$ and $$RI_{(t-\Delta t, t]}$$ computed from equations \ref{equation_diff} and \ref{equation_current}, respectively. Note that $$RI_{(t-\Delta t, t]}$$ is divided by $$\Delta t$$ so that it is expressed per time unit instead of per timestep.

With a fixed step, however, the three methods have the same cost, as the factor $$\exp(\frac{-\Delta t}{\tau})$$ of the analytical solution is a constant, computed only once. Explicit Euler is stable only for $$\Delta t < 2 \tau$$, while implicit Euler is stable for any $$\Delta t$$.

The variable step method is an **event-driven** algorithm: it is the most accurate, as each spike is added at its exact time, but it advances one neuron at a time. The fixed step methods are **clock-driven**: they advance all neurons together at every step, but align the spike times to the time grid ([Brette et al., 2007](https://arxiv.org/abs/q-bio/0611089)). In practice, the fixed step suits large networks, as all neurons run the same operations at every step, while the variable step suits models where the exact timing of spikes matters.

For the sake of comparison, the picture below illustrates the potential over time of a given neuron in a network, for the aforementioned methods, on a $$100ms$$ simulation of a (Brunel) network of $$10000$$ neurons with default neuron parameters. Results are given for the Implicit and Explicit Euler methods with $$\Delta t = 0.1ms$$, and for the analytical solution with fixed ($$\Delta t=0.1ms$$) and variable timestepping:

{: .picture}
<img width="92.9%" height="92.9%" src="{{ site.assets }}/lif-brunel/neuron_plot_equations.svg">

Note the vertical straight lines, which refer to the voltage reset after a spike, and the horizontal straight lines, which refer to the refractory period after firing. The four solutions diverge after the first spikes of the network, at around 14 ms, as a small difference in a spike time changes the input of all post-synaptic neurons.

## Brunel Network

Network activity was mentioned in the previous section. Indeed, a neuron without any kind of stimulus (either electric current injection or synaptic currents) will always remain at rest.

The most common model of network activity is called the **Brunel Network Model** ([Dynamics of Sparsely Connected Networks of Excitatory and Inhibitory Spiking Neurons, Brunel, Journal of Computational Neuroscience, 2000][brunel-paper], [pdf][brunel-paper-pdf]). It is highly relevant, as it provides a good estimation of network dynamics.

The model proposes a network of 10000 simulated LIF neurons (80% excitatory, 20% inhibitory; the paper simulates 12500), with external stimuli from excitatory neurons outside the network, which connect to both the excitatory and the inhibitory neurons. Excitatory means that the PSP is positive; the converse holds for inhibitory neurons. The external neurons are not simulated, so their spiking is generated by a [Poisson](https://en.wikipedia.org/wiki/Poisson_point_process) random generator (PG). The diagram is the following:

{: .picture}
<img width="33.4%" height="33.4%" src="{{ site.assets }}/lif-brunel/network.svg">

{: .caption}
The Brunel network: $$N_E$$ excitatory neurons (E) and $$N_I$$ inhibitory neurons (I). Each neuron receives $$C_E$$ connections from excitatory neurons, $$C_I$$ from inhibitory neurons, and $$C_E$$ from Poisson generators (PG). Arrows are excitatory connections, and flat bars inhibitory ones. Adapted from: [NEST by Example, Fig. 2](https://www.nest-simulator.org/wp-content/uploads/2015/02/NEST_by_Example.pdf)

The tricky part here is to define the parameters of the network in terms of connectivity, synaptic weights, firing threshold, etc. The original [paper][brunel-paper] describes them in detail, but for simplicity, they are summarized in the tables below: 

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
    <tr><td style="text-align: left" markdown="span">Membrane time constant $$\tau$$</td><td style="text-align: right">20 ms</td></tr>
    <tr><td style="text-align: left" markdown="span">Refractory period $$\tau_{\text{rp}}$$</td><td style="text-align: right">2 ms</td></tr>
    <tr><td style="text-align: left" markdown="span">Firing threshold $$V_{\text{th}}$$</td><td style="text-align: right">20 mV</td></tr>
    <tr><td style="text-align: left" markdown="span">Membrane capacitance $$C$$</td><td style="text-align: right">1 pF</td></tr>
    <tr><td style="text-align: left" markdown="span">Resting potential $$u_{\text{rest}}$$</td><td style="text-align: right">0 mV</td></tr>
    <tr><td style="text-align: left" markdown="span">Reset potential $$V_{\text{reset}}$$</td><td style="text-align: right">10 mV</td></tr>
    <tr><td style="text-align: left" markdown="span">Excitatory PSP amplitude $$J_E$$</td><td style="text-align: right">0.1 mV</td></tr>
    <tr><td style="text-align: left" markdown="span">Inhibitory PSP amplitude $$J_I$$</td><td style="text-align: right">-0.5 mV</td></tr>
    <tr><td style="text-align: left" markdown="span">Synaptic delay $$D$$</td><td style="text-align: right">1.5 ms</td></tr>
    <tr><td style="text-align: left" markdown="span">Background rate $$\eta$$</td><td style="text-align: right">2.0</td></tr>
  </tbody>
</table>

{: .caption}
source: Marc-Oliver Gewaltig, ResearchGate

The stimuli parameters can now be adjusted to regulate network activity. The background rate $$\eta$$ is the rate of the Poisson generators, in units of the rate that brings a neuron to its threshold without the input of the network. The relative strength $$g$$ of the inhibitory synapses sets $$J_I = -g J_E$$. Brunel defines a model for slow, asynchronous and fast network dynamics, which exhibit the following average neuron spiking rates (Hz) and global oscillation frequencies (Hz) of the network activity, where SI stands for synchronous irregular and AI for asynchronous irregular:

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

Notice how close the simulated network is to the theoretical predictions. For that reason, although it is a simple approximation, the Brunel Network is a powerful tool for studying network dynamics. The model is particularly useful in very long simulations of large networks, as it is computationally inexpensive compared to more detailed models such as the [Hodgkin-Huxley](https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model) model.

## Output

A `C++`-based Brunel network model of LIF neurons was implemented, and the appropriate stimuli were injected to reproduce the results of the original paper. The outputs match the theoretical model in terms of spike rate and global oscillation frequency. Results are presented as a timeline (left to right) of neuron spike times (a.k.a. *spike trace*, top) and the number of neurons that spike at a given instant (bottom).
 
**Slow dynamics** (SI slow):

{: .picture}
<img width="92.9%" height="92.9%" src="{{ site.assets }}/lif-brunel/raster_plot_g4.5_f0.9.svg">

**Asynchronous dynamics** (AI):

{: .picture}
<img width="92.9%" height="92.9%" src="{{ site.assets }}/lif-brunel/raster_plot_g5_f2.svg">

**Synchronous regular dynamics** (SR):

{: .picture}
<img width="92.9%" height="92.9%" src="{{ site.assets }}/lif-brunel/raster_plot_g3_f2.svg">

In the pictures above, the global activity oscillates slowly in the SI slow state, where each neuron fires irregularly at a very low rate, and is stationary in the AI state. In the SR state, the network is almost fully synchronized, and the neurons fire regularly at high rates.

## Other resources

- A thorough explanation of the Leaky Integrate-and-Fire model is available in the relevant section of the online copy of the [Neuronal Dynamics book from Wulfram Gerstner at EPFL][lif-gerstner].
- The states of the Brunel network and its phase diagram are explained in [section 13.4 of the same book](https://neuronaldynamics.epfl.ch/online/Ch13.S4.html).
- A `Python` implementation of a Brunel network using the [NEST][nest] simulator is described in [NEST by Example: An Introduction to the Neural Simulation Tool NEST, Marc-Oliver Gewaltig et al., Springer Link][nest-brunel].
- The `C++` code of this exercise can be downloaded as <a href="{{ site.assets }}/lif-brunel/lif-brunel.tar.gz">lif-brunel.tar.gz</a>. To help understand the code, the include graph of `Simulation.cpp` is available in <a href="{{ site.assets }}/lif-brunel/diagram_class.svg">diagram_class.svg</a>.
