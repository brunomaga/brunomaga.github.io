---
layout: post
title:  "The Leaky Integrate-and-Fire Neuron and the Brunel Network"
categories: ["Computational Neuroscience"]
tags: ["spiking neurons", "LIF", "Brunel network", "simulation", "Euler method", "event-driven simulation", "mean-field theory"]
description: "The leaky integrate-and-fire (LIF) neuron: its RC circuit equation, its exact and Euler integration with fixed and variable steps, and a C++ simulation of the Brunel network of excitatory and inhibitory neurons."
---

[lif-gerstner]:https://neuronaldynamics.epfl.ch/online/Ch1.S3.html
[nest-brunel]:https://link.springer.com/chapter/10.1007/978-94-007-3858-4_18
[brunel-paper]:https://link.springer.com/article/10.1023/A:1008925309027
[brunel-paper-pdf]:https://web.stanford.edu/group/brainsinsilicon/documents/BrunelSparselyConnectedNets.pdf
[nest]:http://www.nest-simulator.org/
[euler]: https://en.wikipedia.org/wiki/Euler_method

The Leaky Integrate-and-Fire [neuron model](https://en.wikipedia.org/wiki/Biological_neuron_model) (LIF) was first described in 1907 by [Louis Lapicque](https://en.wikipedia.org/wiki/Louis_Lapicque). It describes a neuron by a single variable, its [membrane potential](https://en.wikipedia.org/wiki/Membrane_potential). Its name says what this potential does: it integrates the input current, leaks back to rest, and makes the neuron fire a spike when it reaches a threshold. This simplicity matters in the simulation of networks. The Brunel network of this post has 10,000 neurons and 10 million synapses, and one second of its activity, at a time step of 0.1 ms, takes 100 million updates of a neuron.

In 2000, Nicolas Brunel showed that a sparsely connected network of excitatory and inhibitory LIF neurons already has several states of activity, from synchronous and regular to asynchronous and irregular firing. His theory predicts these states from two parameters. This post derives the equations of the LIF neuron, compares exact and approximate methods to solve them, and simulates the Brunel network in three of its four states.

## The LIF neuron model

The LIF model simplifies the structure of a neuron to a single leaky compartment (*point neuron model*), with a membrane described by an [RC circuit](https://en.wikipedia.org/wiki/RC_circuit). The membrane separates electric charges as a capacitor does, and its ion channels let a current leak through it as a resistor does. The circuit is the following:

{: .picture}
<img width="22.5%" height="22.5%" src="{{ site.assets }}/lif-brunel/rc.svg">

{: .caption}
A neuron receives an input current $$I(t)$$. Its membrane acts as a capacitor $$C$$ in parallel with a resistor $$R$$, in series with a battery of potential $$u_{\text{rest}}$$. Adapted from: [Neuronal Dynamics, Fig. 1.6](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)

The main formulation of its electrical activity is given by the **current equation**, with the following terms:
- The current in the capacitor $$C$$ is given by $$I_C = \frac{dQ}{dt} = \frac{d(CV)}{dt} = C \frac{dV}{dt}$$, for charge $$Q$$ and capacitance $$C$$;
- The current in the resistance $$R$$ is given by [Ohm's law](https://en.wikipedia.org/wiki/Ohm%27s_law) as $$I_R = V/R$$, for the membrane potential $$V$$ (relative to the resting potential $$u_{\text{rest}}$$) and resistance $$R$$.

The law of conservation of electric charge states that the total charge of an isolated system remains constant. Thus, the currents entering a node of the circuit sum to the currents leaving it ([Kirchhoff's current law](https://en.wikipedia.org/wiki/Kirchhoff%27s_circuit_laws)), i.e. the input current is $$I = I_C + I_R$$. Equivalently:

$$
C \frac{dV}{dt}(t) = -\frac{V(t)}{R} + I(t) 
$$

It is common to multiply all terms by $$R$$ to expose the **[time constant](https://en.wikipedia.org/wiki/Time_constant)** $$\tau = R C$$. This is a very important metric in the analysis of the behavior of models, as it tells how *fast* the dynamics of the neuron are. The formulation then becomes:

$$
\tau \frac{dV}{dt}(t) = -V(t) + RI(t) 
\label{equation_diff}
$$

The external currents (such as synaptic activity) are added to $$I(t)$$. In practice, the network contribution $$R I_i(t)$$ to a neuron $$i$$ is the sum of all currents delivered to it by any neuron $$j$$ over time, and is described as:

$$
R I_i(t) = \tau \sum_{j} J_{ij} \sum_{t^\prime} \delta (t-t^\prime _{j} -D)
\label{equation_current}
$$

where $$J_{ij}$$ is the **[postsynaptic potential](https://en.wikipedia.org/wiki/Postsynaptic_potential)** (PSP) amplitude of the synapse from neuron $$j$$ to neuron $$i$$, $$\delta (t)$$ the [Dirac delta function](https://en.wikipedia.org/wiki/Dirac_delta_function), $$t^\prime _{j}$$ the spiking times $$t^\prime$$ of the presynaptic neuron $$j$$, and $$D$$ the synaptic (or transmission) delay. For simplicity, the index $$i$$ is dropped from here on, and $$V(t)$$ is represented as $$V_t$$.

The neuron's potential is considered to rise steeply when there is an input current (by $$J_{ij}$$ at each input spike), thus discarding the exponential charging of the capacitor. On the other hand, the discharge follows an exponential decay with the time constant $$\tau$$.

To simulate the [action potential](https://en.wikipedia.org/wiki/Action_potential) (or *spike*) of biological neurons, the model assumes that each neuron has a certain *fixed* firing threshold. When the potential reaches the firing threshold, the neuron *spikes* (or *fires*), and its potential immediately drops to a reset potential. Axonal branches of neurons are simulated by a **synaptic delay** that regulates the time a spike signal takes to reach the connected postsynaptic neurons. This propagation of spike and current, which dictates the synaptic delay, is illustrated in the following picture:

{: .picture}
<img width="40%" height="40%" src="https://upload.wikimedia.org/wikipedia/commons/9/95/Action_Potential.gif">

{: .caption}
An action potential travelling along an axon, in the direction of the arrows. The membrane is positive outside and negative inside, except at the action potential, where the polarity is reversed. Source: [Action Potential.gif, Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Action_Potential.gif)

Afterwards, for a given **refractory period** $$\tau_{\text{rp}}$$, the neuron potential stays constant at the reset value, to simulate the [refraction](https://en.wikipedia.org/wiki/Refractory_period_(physiology)) of biological neurons. Therefore, a neuron fires at most $$1/\tau_{\text{rp}}$$ times per second, 500 times for the 2 ms of the network of this post. This may not be a very detailed simulation of neuron activity, but it is a good starting point for a simple neuron model, before moving on to more complex models in future posts.

## Integration methods

### Exact integration

The jumps and the exponential decay described above are the exact solution of equation \ref{equation_diff}. Between two input spikes, the input current is zero, and the equation reduces to $$\tau \frac{dV}{dt} = -V$$. Its solution is an exponential decay towards the resting potential:

$$
V(t) = V(t_0) \exp\left(-\frac{t - t_0}{\tau}\right)
$$

where $$t_0$$ is the time of the last input spike. At an input spike, the potential jumps by $$J_{ij}$$. Indeed, when equation \ref{equation_diff} is integrated over an infinitely short interval around the arrival of the spike, the leak term vanishes, and the delta current of equation \ref{equation_current} gives $$\tau \Delta V = \tau J_{ij}$$.

As an example, take the time constant $$\tau = 20$$ ms of the Brunel network below and a step of $$\Delta t = 0.1$$ ms. At every step, the potential keeps $$\exp(-0.1/20) = 0.995$$ of its value, i.e. it decays by 0.5% per step, and to 37% ($$1/e$$) of its value after 20 ms. Each excitatory input adds only $$J_E = 0.1$$ mV, so a neuron at its reset potential of 10 mV needs at least 100 excitatory inputs to reach its firing threshold of 20 mV. Therefore, the neuron fires only when many inputs arrive within a short time, before the leak brings its potential back to rest.

Two different stepping methods are derived from the analytical solution of the first-order [ODE](https://en.wikipedia.org/wiki/Ordinary_differential_equation), both based on a two-step algorithm:
- Calculate the decay from the previous step: $$V_t \leftarrow V_{t- \Delta t} \exp(\frac{-\Delta t}{\tau})$$
- Add the voltage change due to the network current $$RI$$ between $$t-\Delta t$$ and $$t$$: $$V_t \leftarrow  V_t + RI_{(t-\Delta t, t]} / \tau $$, where $$RI_{(t-\Delta t, t]}$$ is the integral of $$RI$$ over that interval, i.e. $$\tau$$ times the sum of the amplitudes $$J_{ij}$$ of the spikes that arrive in it.

Neurons can be advanced with fixed-step or variable-step integration:
- **Fixed step** integration assumes the time step $$\Delta t$$ to be constant throughout the execution.
- **Variable step** methods advance the neuron with the smallest $$t$$ in the network, with a $$\Delta t$$ computed as the minimum of the following two values:
  - the time difference to the next incoming spike, so that each spike is added at its exact time;
  - $$t^\star + D - t$$, where $$t^\star$$ is the time of the neuron with the second smallest $$t$$ and $$D$$ is the synaptic delay, i.e. the largest step that can be taken from the neuron at $$t$$ without missing a spike from the neuron at time $$t^\star$$ (if it spikes).  

### Euler methods

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
\Leftrightarrow & V_t = \frac{RI_{(t-\Delta t, t]} + \tau V_{t-\Delta t}}{\Delta t + \tau}
\end{align*}
$$

with $$\frac{dV_{t}}{dt}$$ and $$RI_{(t-\Delta t, t]}$$ computed from equations \ref{equation_diff} and \ref{equation_current}, respectively. Note that $$RI_{(t-\Delta t, t]}$$ is divided by $$\Delta t$$ so that it is expressed per time unit instead of per timestep.

### Comparison of the methods

With a fixed step, however, the three methods have the same cost, as the factor $$\exp(\frac{-\Delta t}{\tau})$$ of the analytical solution is a constant, computed only once. They differ in the factor that multiplies the potential of the previous step. Rearranged, the three updates are:

$$
\begin{align*}
\text{exact:} \quad & V_t = \exp(\frac{-\Delta t}{\tau}) \, V_{t-\Delta t} + RI_{(t-\Delta t, t]} / \tau \\
\text{explicit Euler:} \quad & V_t = (1 - \frac{\Delta t}{\tau}) \, V_{t-\Delta t} + RI_{(t-\Delta t, t]} / \tau \\
\text{implicit Euler:} \quad & V_t = \frac{\tau}{\tau + \Delta t} \, (V_{t-\Delta t} + RI_{(t-\Delta t, t]} / \tau)
\end{align*}
$$

With $$\Delta t = 0.1$$ ms and $$\tau = 20$$ ms, the three factors are 0.995012, 0.995000 and 0.995025: the methods agree to about 0.001% per step. Implicit Euler also scales each input by 0.995. Explicit Euler is stable only for $$\Delta t < 2 \tau$$, i.e. for steps under 40 ms here, while implicit Euler is stable for any $$\Delta t$$.

The variable step method is an **event-driven** algorithm: it is the most accurate, as each spike is added at its exact time, but it advances one neuron at a time. The fixed step methods are **clock-driven** (or time-driven): they advance all neurons together at every step, but align the spike times to the time grid ([Brette et al., 2007](https://arxiv.org/abs/q-bio/0611089)). In practice, the fixed step suits large networks, as all neurons run the same operations at every step, while the variable step suits models where the exact timing of spikes matters.

For the sake of comparison, the picture below illustrates the potential over time of a given neuron in a network, for the aforementioned methods. The simulation runs for 100 ms on a Brunel network of 10,000 neurons, with the parameters of the next section. Results are given for the implicit and explicit Euler methods with $$\Delta t = 0.1$$ ms, and for the analytical solution with fixed ($$\Delta t = 0.1$$ ms) and variable time steps:

{: .picture}
<img width="92.9%" height="92.9%" src="{{ site.assets }}/lif-brunel/neuron_plot_equations.svg">

Note the vertical straight lines, which refer to the voltage reset after a spike, and the horizontal straight lines, which refer to the refractory period after firing. During the first 12 ms, the four curves overlap, as the factors of the methods differ by about 0.001% per step. They diverge once the first spikes of the network reach the neuron, as a small difference in a spike time changes the input of all postsynaptic neurons.

## The Brunel network

Network activity was mentioned in the previous section. Indeed, a neuron without any kind of stimulus (either electric current injection or synaptic currents) will always remain at rest.

One of the most common models of network activity is the **Brunel network model** ([Dynamics of Sparsely Connected Networks of Excitatory and Inhibitory Spiking Neurons, Brunel, Journal of Computational Neuroscience, 2000][brunel-paper], [pdf][brunel-paper-pdf]). It is highly relevant, as it comes with a theory that predicts its dynamics. A [mean-field](https://en.wikipedia.org/wiki/Mean-field_theory) analysis replaces the input that each neuron receives from the network by its average and its fluctuations. A [Fokker-Planck equation](https://en.wikipedia.org/wiki/Fokker%E2%80%93Planck_equation) for the distribution of the membrane potentials then gives the firing rates and the states of the network.

The model proposes a network of 10,000 simulated LIF neurons (80% excitatory, 20% inhibitory; the paper simulates 12,500). Its connections are sparse and random: each neuron receives inputs from 10% of the excitatory and 10% of the inhibitory neurons. External stimuli come from excitatory neurons outside the network, which connect to both the excitatory and the inhibitory neurons. Excitatory means that the PSP is positive; the converse holds for inhibitory neurons. The external neurons are not simulated, so their spiking is generated by a [Poisson](https://en.wikipedia.org/wiki/Poisson_point_process) random generator (PG). The diagram is the following:

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

### Network states

The stimuli parameters can now be adjusted to regulate network activity. The background rate $$\eta$$ sets the rate $$\nu_{\text{ext}}$$ of each Poisson generator. It is given in units of the threshold rate $$\nu_{\text{thr}}$$, at which the external input alone brings the mean potential of a neuron to its threshold:

$$
\nu_{\text{ext}} = \eta \, \nu_{\text{thr}}, \quad \text{with} \quad \nu_{\text{thr}} = \frac{V_{\text{th}}}{J_E C_E \tau}
$$

With the parameters above, $$\nu_{\text{thr}} = 12.5$$ Hz, so $$\eta = 2$$ sets each of the $$C_E = 800$$ external synapses of a neuron to 25 Hz: 20,000 input spikes per second, or 2 per step of 0.1 ms. This input alone would bring the mean potential to $$J_E C_E \tau \nu_{\text{ext}} = \eta V_{\text{th}} = 40$$ mV, twice the threshold. Starting from rest, the mean potential, $$40 (1 - e^{-t/\tau})$$ mV, reaches the threshold at $$t = \tau \ln 2 \approx 13.9$$ ms. Therefore, many neurons fire their first spike at about the same time: the network fires a first wave of about 7,000 spikes between 10 and 15 ms (the first bars of the AI plot below).

The relative strength $$g$$ of the inhibitory synapses sets $$J_I = -g J_E$$ (the table above has $$J_I = -0.5$$ mV, i.e. $$g = 5$$), and decides whether excitation or inhibition dominates. When the neurons of the network fire at a rate $$\nu$$, the mean input that a neuron receives from the network is computed as:

$$
\mu_{\text{rec}} = \tau \nu (C_E J_E + C_I J_I) = C_E J_E \tau \nu \left(1 - g \frac{C_I}{C_E}\right)
$$

Each neuron receives four times as many excitatory inputs as inhibitory ones ($$C_I / C_E = 1/4$$), so this input is zero at $$g = 4$$, the **balance of excitation and inhibition**. In Brunel's words, the line $$g = 4$$ "is where feedback excitation exactly balances inhibition". The network is excitation-dominated for $$g < 4$$, and inhibition-dominated for $$g > 4$$. The plot of the previous section shows it: at $$g = 5$$, the first wave of spikes reaches the neuron after the synaptic delay, at around 14.5 ms, and its potential falls by 10 to 14 mV instead of rising.

Depending on $$g$$ and $$\eta$$, Brunel describes four states of the network. They are named after the global activity of the network, synchronous when it [oscillates](https://en.wikipedia.org/wiki/Neural_oscillation) and asynchronous when it is stationary, and after the firing of each neuron, regular or irregular. In the **synchronous regular (SR)** state, excitation dominates ($$g = 3$$), and the neurons fire regularly at high rates, almost all together. Inhibition dominates in the three other states, where each neuron fires irregularly. The global activity is stationary in the **asynchronous irregular (AI)** state, and oscillates fast or slowly in the **synchronous irregular (SI)** states, SI fast and SI slow.

The following table gives the values of $$g$$ and $$\eta$$ of each state in the simulations of the paper (Fig. 8 in the paper). For the three inhibition-dominated states, it also gives the average firing rates and global oscillation frequencies of the simulations and of the theory (Table 1 in the paper):

<table markdown="1">
  <thead>
    <tr><th rowspan="2" style="text-align: left">State</th><th rowspan="2" style="text-align: center" markdown="span">$$g$$</th><th rowspan="2" style="text-align: center" markdown="span">$$\eta$$</th><th colspan="2" style="text-align: center">Firing rate</th><th colspan="2" style="text-align: center">Global frequency</th></tr>
    <tr><th style="text-align: center">Simulation</th><th style="text-align: center">Theory</th><th style="text-align: center">Simulation</th><th style="text-align: center">Theory</th></tr>
  </thead>
  <tbody>
    <tr><td style="text-align: left">SR</td><td style="text-align: center">3</td><td style="text-align: center">2</td><td colspan="4" style="text-align: center">not compared in the paper</td></tr>
    <tr><td style="text-align: left">SI, fast</td><td style="text-align: center">6</td><td style="text-align: center">4</td><td style="text-align: center">60.7 Hz</td><td style="text-align: center">55.8 Hz</td><td style="text-align: center">180 Hz</td><td style="text-align: center">190 Hz</td></tr>
    <tr><td style="text-align: left">AI</td><td style="text-align: center">5</td><td style="text-align: center">2</td><td style="text-align: center">37.7 Hz</td><td style="text-align: center">38.0 Hz</td><td style="text-align: center">—</td><td style="text-align: center">—</td></tr>
    <tr><td style="text-align: left">SI, slow</td><td style="text-align: center">4.5</td><td style="text-align: center">0.9</td><td style="text-align: center">5.5 Hz</td><td style="text-align: center">6.5 Hz</td><td style="text-align: center">22 Hz</td><td style="text-align: center">29 Hz</td></tr>
  </tbody>
</table>

{: .caption}
source: Brunel 2000

Notice how close the simulated network is to the theoretical predictions: within 1% for the firing rate of the AI state, and within 25% for the firing rates and global frequencies of the SI states. For that reason, although it is a simple approximation, the Brunel network is a powerful tool for studying network dynamics. The model is particularly useful in very long simulations of large networks, as it is computationally inexpensive compared to more detailed models such as the [Hodgkin-Huxley](https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model) model, which describes each neuron with four variables instead of one.

## Simulation results

A `C++`-based Brunel network model of LIF neurons was implemented, and the appropriate stimuli were injected to reproduce three states of the original paper. Results are presented as a **raster plot** (a.k.a. *spike trace*, top), a timeline (left to right) of the spike times of the neurons, one row per neuron. Below it, a bar chart gives the number of neurons that spike in each interval of 1 ms.

**Slow dynamics** (SI slow):

{: .picture}
<img width="92.9%" height="92.9%" src="{{ site.assets }}/lif-brunel/raster_plot_g4.5_f0.9.svg">

**Asynchronous dynamics** (AI):

{: .picture}
<img width="92.9%" height="92.9%" src="{{ site.assets }}/lif-brunel/raster_plot_g5_f2.svg">

**Synchronous regular dynamics** (SR):

{: .picture}
<img width="92.9%" height="92.9%" src="{{ site.assets }}/lif-brunel/raster_plot_g3_f2.svg">

In the pictures above, the global activity oscillates slowly in the SI slow state, where each neuron fires irregularly at a very low rate. The network fires in 19 bursts in one second, one every 51 ms on average, i.e. at a global frequency of about 20 Hz, close to the 22 Hz of the simulation of the paper. With $$\eta = 0.9$$, the external input alone brings the mean potential only to $$\eta V_{\text{th}} = 18$$ mV, under the threshold. The neurons fire only when the fluctuations of their input push them above it, so the first spikes come only at 58 ms.

The global activity is stationary in the AI state. In the SR state, the network is almost fully synchronized, and the neurons fire regularly at high rates: the spike count of the network repeats every 3 ms.

## Other resources

- A thorough explanation of the Leaky Integrate-and-Fire model is available in the relevant section of the online copy of the [Neuronal Dynamics book from Wulfram Gerstner at EPFL][lif-gerstner].
- The states of the Brunel network and its phase diagram are explained in [section 13.4 of the same book](https://neuronaldynamics.epfl.ch/online/Ch13.S4.html).
- A `Python` implementation of a Brunel network using the [NEST][nest] simulator is described in [NEST by Example: An Introduction to the Neural Simulation Tool NEST, Marc-Oliver Gewaltig et al., Springer Link][nest-brunel].
- The `C++` code of this exercise can be downloaded as <a href="{{ site.assets }}/lif-brunel/lif-brunel.tar.gz">lif-brunel.tar.gz</a>. To help understand the code, the include graph of `Simulation.cpp` is available in <a href="{{ site.assets }}/lif-brunel/diagram_class.svg">diagram_class.svg</a>.
