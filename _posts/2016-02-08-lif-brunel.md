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
[neuron-model]: https://en.wikipedia.org/wiki/Biological_neuron_model
[rc]: https://en.wikipedia.org/wiki/RC_circuit
[kirchhoff]: https://en.wikipedia.org/wiki/Kirchhoff%27s_circuit_laws
[ohm]: https://en.wikipedia.org/wiki/Ohm%27s_law
[dirac]: https://en.wikipedia.org/wiki/Dirac_delta_function
[backward-euler]: https://en.wikipedia.org/wiki/Backward_Euler_method
[poisson]: https://en.wikipedia.org/wiki/Poisson_point_process
[mean-field]: https://en.wikipedia.org/wiki/Mean-field_theory

The [leaky integrate-and-fire][neuron-model] (LIF) model, first described in 1907 by [Louis Lapicque](https://en.wikipedia.org/wiki/Louis_Lapicque), simplifies a neuron to a single leaky compartment without spatial extent, a *point neuron*, whose membrane behaves like an [RC circuit][rc]:

{: .picture}
<img width="22.5%" height="22.5%" src="{{ site.assets }}/lif-brunel/rc.svg">

{: .caption}
A neuron receives an input current $$I(t)$$. Its membrane acts as a capacitor $$C$$ in parallel with a resistor $$R$$, in series with a battery of potential $$u_{\text{rest}}$$. Adapted from: [Neuronal Dynamics, Fig. 1.6](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)

The figure writes the membrane potential as $$u$$, and the text uses $$V = u - u_{\text{rest}}$$, the potential relative to rest, since only this difference matters. The electrical activity of the model follows from [Kirchhoff's current law][kirchhoff], a consequence of the conservation of electric charge: the currents that enter a node sum to the currents that leave it. The input current $$I$$ therefore splits into two currents:
- the current that charges the capacitor, $$I_C = \frac{dQ}{dt} = C \frac{du}{dt} = C \frac{dV}{dt}$$, for the charge $$Q = Cu$$ of the capacitance $$C$$;
- the leak current through the resistance $$R$$, which follows [Ohm's law][ohm]: $$I_R = V/R$$, since $$V$$ is the voltage across the resistor.

With $$I = I_C + I_R$$, the **current equation** of the model is:

$$
C \frac{dV}{dt}(t) = -\frac{V(t)}{R} + I(t)
$$

Multiplying all terms by $$R$$ exposes the **membrane time constant** $$\tau = R C$$:

$$
\tau \frac{dV}{dt}(t) = -V(t) + RI(t)
$$

This constant sets the speed of the dynamics: without input, $$V$$ decays to 0 as $$e^{-t/\tau}$$.

In a network, the input current of neuron $$i$$ is the sum of the spikes that its presynaptic neurons $$j$$ deliver to it over time:

$$
RI_i(t) = \tau \sum_{j} J_{ij} \sum_{k} \delta (t - t_j^k - D)
$$

where $$J_{ij}$$ is the **postsynaptic potential** (PSP) amplitude, $$\delta$$ the [Dirac delta function][dirac], $$t_j^k$$ the time of the $$k$$-th spike of neuron $$j$$, and $$D$$ the **synaptic delay**. Since the input is a sum of Dirac pulses, the charge of each input spike arrives at once, and $$V$$ jumps by $$J_{ij}$$ instead of rising gradually.

To model the [action potential](https://en.wikipedia.org/wiki/Action_potential), or *spike*, of biological neurons, each neuron has a fixed **firing threshold** $$\theta$$. When $$V$$ reaches $$\theta$$, the neuron spikes, or *fires*, and $$V$$ drops at once to the **reset potential** $$V_r$$. The spike then travels along the axon to the postsynaptic neurons, as the picture below illustrates, and the synaptic delay $$D$$ models the time that this propagation takes:

{: .picture}
<img width="45.1%" height="45.1%" src="{{ site.assets }}/lif-brunel/action_potential.svg">

{: .caption}
An action potential travelling along an axon, in the direction of the arrow. The membrane is positive outside and negative inside, except at the action potential, where the polarity is reversed. Adapted from: [Action Potential.gif, Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Action_Potential.gif)

After a spike, $$V$$ stays at $$V_r$$ during the **refractory period** $$\tau_{\text{rp}}$$, which models the [refraction](https://en.wikipedia.org/wiki/Refractory_period_(physiology)) of biological neurons. The LIF model is not a detailed simulation of a neuron, since it ignores the shape of the action potential and the morphology of the neuron, but it is a good starting point before the detailed models of the post on [the electrical activity of detailed neuron models]({{ site.baseurl }}{% post_url 2016-08-15-neuron-electrical-activity %}).

## Time integration

How can a simulation advance this equation in time? Between input spikes, the solution of the LIF equation is an exponential decay, so the exact solution can be computed step by step. Writing $$V_t$$ for $$V(t)$$, a step from $$t-\Delta t$$ to $$t$$ takes two operations:
1. the decay since the previous step: $$V_t \leftarrow V_{t-\Delta t} \, e^{-\Delta t / \tau}$$;
2. the voltage change due to the input between $$t-\Delta t$$ and $$t$$: $$V_t \leftarrow V_t + RI_{(t-\Delta t, t]} / \tau$$.

Here, $$RI_{(t-\Delta t, t]} = \int_{t-\Delta t}^{t} RI(s) \, ds$$ is the input over the step, which is $$\tau$$ times the sum of the PSP amplitudes of the spikes that arrive in the step. The step size can be chosen in two ways:
- **fixed step**: $$\Delta t$$ stays constant during the whole simulation, and the spikes of a step are applied at its end, so all spike times are rounded up to a multiple of $$\Delta t$$;
- **variable step**: the neuron with the smallest time $$t$$ in the network advances, up to its next input spike or up to $$t^\star + D$$, whichever comes first, where $$t^\star$$ is the second smallest time. This is the largest step that the neuron can take without missing a spike: no other neuron can spike before $$t^\star$$, so no new spike can reach the neuron before $$t^\star + D$$. The spike times are therefore exact.

The variable step is exact, but it costs more in a network. Every input spike ends a step, and every step evaluates an exponential, since $$\Delta t$$ changes from step to step. A neuron of the network below receives tens of spikes per ms from its 1,000 synapses and its external input, so its steps are much shorter than a fixed step of 0.1 ms.

The LIF equation can also be solved numerically, as most neuron models must be. A numerical solution is not exact, but it approximates the exact solution well when the step $$\Delta t$$ is small. The two simplest examples are the fixed-step [Euler methods][euler], which replace the derivative with a finite difference over the step. Forward (explicit) Euler evaluates the right-hand side of the equation at the start of the step, so the update follows directly:

$$
\begin{align*}
& \tau \frac{V_t - V_{t-\Delta t}}{\Delta t} = -V_{t-\Delta t} + \frac{RI_{(t-\Delta t, t]}}{\Delta t} \\
\Leftrightarrow \; & V_t = V_{t-\Delta t} + \frac{\Delta t}{\tau} \left( -V_{t-\Delta t} + \frac{RI_{(t-\Delta t, t]}}{\Delta t} \right) \\
\Leftrightarrow \; & V_t = \left( 1 - \frac{\Delta t}{\tau} \right) V_{t-\Delta t} + \frac{RI_{(t-\Delta t, t]}}{\tau}
\end{align*}
$$

[Backward (implicit) Euler][backward-euler] evaluates the right-hand side at the end of the step instead. The unknown $$V_t$$ then appears on both sides, and each step solves the equation for it:

$$
\begin{align*}
& \tau \frac{V_t - V_{t-\Delta t}}{\Delta t} = -V_t + \frac{RI_{(t-\Delta t, t]}}{\Delta t} \\
\Leftrightarrow \; & \tau V_t - \tau V_{t-\Delta t} = -\Delta t \, V_t + RI_{(t-\Delta t, t]} \\
\Leftrightarrow \; & V_t = \frac{\tau V_{t-\Delta t} + RI_{(t-\Delta t, t]}}{\tau + \Delta t}
\end{align*}
$$

In both methods, the input is divided by $$\Delta t$$: $$RI_{(t-\Delta t, t]} / \Delta t$$ is the mean input per time unit over the step, the quantity that the derivative needs.

Do the Euler methods run faster, since they avoid the costly exponential? With a fixed step, they do not: all three methods compute $$V_t = a V_{t-\Delta t} + b \, RI_{(t-\Delta t, t]}$$ with constants $$a$$ and $$b$$, so they cost the same: the exponential of the exact method, $$a = e^{-\Delta t / \tau}$$, needs to be computed only once. Forward Euler replaces this factor by $$1 - \Delta t / \tau$$, and is stable only for $$\Delta t < 2 \tau$$. Backward Euler replaces it by $$1 / (1 + \Delta t / \tau)$$ and scales the inputs by the same factor, but is stable for any $$\Delta t$$. For $$\Delta t = 0.1$$ ms and $$\tau = 20$$ ms, the three factors agree to four decimals, and backward Euler reduces each PSP by 0.5%. The Euler methods thus bring no speedup to the LIF model. They matter instead for models without an exact solution, such as the [Hodgkin-Huxley](https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model) model.

The figure below compares the four methods on one neuron of the 10,000-neuron network of the next section, over 100 ms, with $$\Delta t = 0.1$$ ms for the fixed steps:

{: .picture}
<img width="99.9%" height="99.9%" src="{{ site.assets }}/lif-brunel/neuron_plot_equations.svg">

The vertical lines are the resets after a spike, and the horizontal lines at 10 mV are the refractory periods. The four solutions agree for the first 14 ms, and diverge after the first spikes of the network: from then on, a small difference in a spike time changes the input of all postsynaptic neurons.

## Brunel network

The model above assumed an input from a network, and a neuron needs one: without any stimulus, either an injected current or synaptic currents, a LIF neuron stays at rest. The [Brunel network][brunel-paper] (Brunel, 2000; [pdf][brunel-paper-pdf]) provides this input with a sparse network of LIF neurons, 80% excitatory and 20% inhibitory. Excitatory neurons have positive PSPs, and inhibitory neurons negative ones. Each neuron also receives excitatory input from $$C_E$$ external neurons, at a rate $$\nu_{\text{ext}}$$ each. These external neurons are not simulated: their spikes come from a [Poisson process][poisson], which the figure shows as Poisson generators (PG):

{: .picture}
<img width="33.4%" height="33.4%" src="{{ site.assets }}/lif-brunel/network.svg">

{: .caption}
The Brunel network: $$N_E$$ excitatory neurons (E) and $$N_I$$ inhibitory neurons (I). Each neuron receives $$C_E$$ connections from excitatory neurons, $$C_I$$ from inhibitory neurons, and $$C_E$$ from Poisson generators (PG). Arrows are excitatory connections, and open circles inhibitory ones. Adapted from: [NEST by Example, Fig. 2](https://www.nest-simulator.org/wp-content/uploads/2015/02/NEST_by_Example.pdf)

The hard part of the model is the choice of its parameters, such as the connectivity, the synaptic weights and the firing threshold. The paper describes them in detail, and the tables below list them for a network of 10,000 neurons:

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
    <tr><td style="text-align: left" markdown="span">Firing threshold $$\theta$$</td><td style="text-align: right">20 mV</td></tr>
    <tr><td style="text-align: left" markdown="span">Membrane capacitance $$C$$</td><td style="text-align: right">1 pF</td></tr>
    <tr><td style="text-align: left" markdown="span">Resting potential $$u_{\text{rest}}$$</td><td style="text-align: right">0 mV</td></tr>
    <tr><td style="text-align: left" markdown="span">Reset potential $$V_r$$</td><td style="text-align: right">10 mV</td></tr>
    <tr><td style="text-align: left" markdown="span">Excitatory PSP amplitude $$J_E$$</td><td style="text-align: right">0.1 mV</td></tr>
    <tr><td style="text-align: left" markdown="span">Inhibitory PSP amplitude $$J_I$$</td><td style="text-align: right">-0.5 mV</td></tr>
    <tr><td style="text-align: left" markdown="span">Synaptic delay $$D$$</td><td style="text-align: right">1.5 ms</td></tr>
    <tr><td style="text-align: left" markdown="span">External rate $$\eta = \nu_{\text{ext}} / \nu_{\text{thr}}$$</td><td style="text-align: right">2.0</td></tr>
  </tbody>
</table>

{: .caption}
Source: Marc-Oliver Gewaltig, ResearchGate

How does the network behave? Its state depends on two parameters: the relative strength of inhibition $$g = -J_I / J_E$$, here 5, and the external rate $$\eta = \nu_{\text{ext}} / \nu_{\text{thr}}$$, where $$\nu_{\text{thr}} = \theta / (J_E C_E \tau)$$ is the external rate at which the mean external input alone reaches the threshold. By varying them, Brunel finds four states:
- **SR**, synchronous regular ($$g = 3$$, $$\eta = 2$$): the neurons fire regularly, almost in synchrony, at high rates;
- **SI fast**, synchronous irregular ($$g = 6$$, $$\eta = 4$$): the global activity oscillates fast, and the neurons fire irregularly, at a lower rate than the oscillation;
- **AI**, asynchronous irregular ($$g = 5$$, $$\eta = 2$$): the global activity is stationary, and the neurons fire irregularly;
- **SI slow**, synchronous irregular ($$g = 4.5$$, $$\eta = 0.9$$): the global activity oscillates slowly, and the neurons fire irregularly, at low rates.

The table compares the firing rates and the global oscillation frequencies of his simulations with the predictions of his [mean-field theory][mean-field]:

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
Source: [Dynamics of Sparsely Connected Networks of Excitatory and Inhibitory Spiking Neurons][brunel-paper]

Simulation and theory agree well: the rates are within 5 Hz of the predictions, and the frequencies within 10 Hz. The model is also cheap to simulate. A LIF neuron has a single state variable, $$V$$, updated with one multiplication and one addition per step, while a Hodgkin-Huxley neuron has four state variables per compartment, the potential and three gating variables, with nonlinear rate functions evaluated at every step. LIF networks thus suit long simulations of large networks.

## Simulation results

A `C++` implementation reproduces three of the four states, with the network of the tables. Each figure shows, over 1 s, the spike times of the 10,000 neurons, one row per neuron (top), and the number of spikes per ms (bottom).

**SI slow**:

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/lif-brunel/raster_plot_g4.5_f0.9.svg">

The global activity oscillates with a period of about 50 ms, or 20 Hz, close to the 22 Hz of the paper.

**AI**:

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/lif-brunel/raster_plot_g5_f2.svg">

After a burst at the start, when all neurons leave the same initial state together, the activity is stationary: the neurons fire irregularly, and the number of spikes per ms fluctuates around a constant mean.

**SR**:

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/lif-brunel/raster_plot_g3_f2.svg">

The neurons fire regularly and almost in synchrony, at high rates, so the number of spikes per ms forms a regular train of peaks of about 6,000 spikes.

## Other resources

- [Neuronal Dynamics][lif-gerstner], the online book by Wulfram Gerstner et al., explains the LIF model in detail.
- [NEST by Example: An Introduction to the Neural Simulation Tool NEST][nest-brunel], by Marc-Oliver Gewaltig et al., implements the Brunel network in `Python` with the [NEST][nest] simulator.
- The `C++` code is in <a href="{{ site.assets }}/lif-brunel/lif-brunel.tar.gz">lif-brunel.tar.gz</a>, and the include graph of `Simulation.cpp` in <a href="{{ site.assets }}/lif-brunel/diagram_class.svg">diagram_class.svg</a> shows how the headers of the code include each other.
