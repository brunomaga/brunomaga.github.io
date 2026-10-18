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

The [leaky integrate-and-fire][neuron-model] (LIF) model, first described in 1907 by [Louis Lapicque](https://en.wikipedia.org/wiki/Louis_Lapicque), reduces a neuron to a single compartment without spatial extent, a *point neuron*. Its membrane is an [RC circuit][rc]:

{: .picture}
<img width="22.5%" height="22.5%" src="{{ site.assets }}/lif-brunel/rc.svg">

{: .caption}
A neuron receives an input current $$I(t)$$. Its membrane acts as a capacitor $$C$$ in parallel with a resistor $$R$$, in series with a battery of potential $$u_{\text{rest}}$$. Adapted from: [Neuronal Dynamics, Fig. 1.6](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)

The figure writes the membrane potential as $$u$$, and the text uses $$V = u - u_{\text{rest}}$$, the potential relative to the resting potential. By [Kirchhoff's current law][kirchhoff], the input current $$I$$ splits into two currents:
- the capacitor current $$I_C = \frac{dQ}{dt} = C \frac{dV}{dt}$$, for the charge $$Q = CV$$ of the capacitance $$C$$;
- the leak current through the resistance $$R$$, $$I_R = V/R$$, by [Ohm's law][ohm].

With $$I = I_C + I_R$$:

$$
C \frac{dV}{dt}(t) = -\frac{V(t)}{R} + I(t)
$$

Multiplying by $$R$$ gives the **membrane time constant** $$\tau = R C$$:

$$
\tau \frac{dV}{dt}(t) = -V(t) + RI(t)
$$

$$\tau$$ sets the speed of the dynamics: without input, $$V$$ decays as $$e^{-t/\tau}$$. In a network, the input of neuron $$i$$ is the sum of the spikes of its presynaptic neurons $$j$$:

$$
RI_i(t) = \tau \sum_{j} J_{ij} \sum_{k} \delta (t - t_j^k - D)
$$

where $$J_{ij}$$ is the **postsynaptic potential** (PSP) amplitude, $$\delta$$ the [Dirac delta function][dirac], $$t_j^k$$ the time of the $$k$$-th spike of neuron $$j$$, and $$D$$ the **synaptic delay**. Each input spike raises $$V$$ at once by $$J_{ij}$$. Between input spikes, $$V$$ decays exponentially to 0.

A spike, or [action potential](https://en.wikipedia.org/wiki/Action_potential), is modeled by a fixed **firing threshold** $$\theta$$. When $$V$$ reaches $$\theta$$, the neuron spikes, and $$V$$ drops to the **reset potential** $$V_r$$. The spike reaches the postsynaptic neurons after the delay $$D$$, which models its propagation along the axon:

{: .picture}
<img width="40%" height="40%" src="{{ site.assets }}/lif-brunel/action_potential.gif">

{: .caption}
An action potential travelling along an axon. Source: [Action Potential.gif, Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Action_Potential.gif)

After a spike, $$V$$ stays at $$V_r$$ during the **refractory period** $$\tau_{\text{rp}}$$, as in the [refractory period](https://en.wikipedia.org/wiki/Refractory_period_(physiology)) of biological neurons. The LIF model ignores the shape of the action potential and the morphology of the neuron. Detailed models are in the post on [the electrical activity of detailed neuron models]({{ site.baseurl }}{% post_url 2016-08-15-neuron-electrical-activity %}).

## Time integration

Between input spikes, the solution of the LIF equation is an exponential decay. With $$V_t = V(t)$$, a step from $$t-\Delta t$$ to $$t$$ takes two operations:
1. the decay: $$V_t \leftarrow V_{t-\Delta t} \, e^{-\Delta t / \tau}$$;
2. the input: $$V_t \leftarrow V_t + RI_{(t-\Delta t, t]} / \tau$$.

$$RI_{(t-\Delta t, t]} = \int_{t-\Delta t}^{t} RI(s) \, ds$$ is the input over the step: $$\tau$$ times the sum of the PSP amplitudes of the spikes that arrive in the step. Two step sizes are possible:
- **fixed step**: $$\Delta t$$ is constant, and the spikes of a step are applied at its end. All spike times are thus rounded up to a multiple of $$\Delta t$$;
- **variable step**: the neuron with the smallest time $$t$$ advances, up to its next input spike or up to $$t^\star + D$$, whichever comes first, where $$t^\star$$ is the second smallest neuron time. No other neuron can spike before $$t^\star$$, so no new spike reaches the neuron before $$t^\star + D$$. The spike times are exact.

The variable step is exact, but costs more in a network. Every input spike ends a step, and every step evaluates an exponential, since $$\Delta t$$ changes. A neuron of the network below receives tens of spikes per ms from its 1000 synapses and its external input, so its steps are much shorter than a fixed step of 0.1 ms.

Numerical methods approximate the solution with arithmetic only. The [Euler methods][euler] replace the derivative with a finite difference over a fixed step. Forward (explicit) Euler evaluates the leak $$-V$$ at the start of the step:

$$
\tau \frac{V_t - V_{t-\Delta t}}{\Delta t} = -V_{t-\Delta t} + \frac{RI_{(t-\Delta t, t]}}{\Delta t}
\;\; \Leftrightarrow \;\;
V_t = \Big( 1 - \frac{\Delta t}{\tau} \Big) V_{t-\Delta t} + \frac{RI_{(t-\Delta t, t]}}{\tau}
$$

[Backward (implicit) Euler][backward-euler] evaluates it at the end of the step:

$$
\tau \frac{V_t - V_{t-\Delta t}}{\Delta t} = -V_t + \frac{RI_{(t-\Delta t, t]}}{\Delta t}
\;\; \Leftrightarrow \;\;
V_t = \frac{\tau V_{t-\Delta t} + RI_{(t-\Delta t, t]}}{\tau + \Delta t}
$$

$$RI_{(t-\Delta t, t]} / \Delta t$$ is the mean input over the step. All three fixed-step methods compute $$V_t = a V_{t-\Delta t} + b \, RI_{(t-\Delta t, t]}$$ with constants $$a$$ and $$b$$, so they cost the same. The exact method has $$a = e^{-\Delta t / \tau}$$, computed once. Forward Euler replaces it by $$1 - \Delta t / \tau$$, and is stable only for $$\Delta t < 2 \tau$$. Backward Euler replaces it by $$1 / (1 + \Delta t / \tau)$$ and scales the inputs by the same factor, but is stable for any $$\Delta t$$. For $$\Delta t = 0.1$$ ms and $$\tau = 20$$ ms, the three factors agree to four decimals, and backward Euler reduces each PSP by 0.5%. The Euler methods thus bring no speedup to the LIF model. They matter for models without an exact solution, such as the [Hodgkin-Huxley](https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model) model.

The figure compares the four methods on one neuron of the 10,000-neuron network below, over 100 ms, with $$\Delta t = 0.1$$ ms for the fixed steps:

{: .picture}
<img width="99.9%" height="99.9%" src="{{ site.assets }}/lif-brunel/neuron_plot_equations.svg">

The vertical lines are the resets after a spike, and the horizontal lines at 10 mV are the refractory periods. The four solutions agree for the first 14 ms, and diverge after the first spikes of the network: a small change in a spike time changes the input of all postsynaptic neurons.

## Brunel network

Without input, a LIF neuron stays at rest. The [Brunel network][brunel-paper] ([pdf][brunel-paper-pdf]) provides the input with a sparse network of LIF neurons, 80% excitatory and 20% inhibitory. Excitatory neurons have positive PSPs, and inhibitory neurons negative ones. Each neuron also receives excitatory input from $$C_E$$ external neurons, at a rate $$\nu_{\text{ext}}$$ each. The external neurons are not simulated: their spikes come from a [Poisson process][poisson] (PG in the figure).

{: .picture}
<img width="33.4%" height="33.4%" src="{{ site.assets }}/lif-brunel/network.svg">

{: .caption}
The Brunel network: $$N_E$$ excitatory neurons (E) and $$N_I$$ inhibitory neurons (I). Each neuron receives $$C_E$$ connections from excitatory neurons, $$C_I$$ from inhibitory neurons, and $$C_E$$ from Poisson generators (PG). Arrows are excitatory connections, and open circles inhibitory ones. Adapted from: [NEST by Example, Fig. 2](https://www.nest-simulator.org/wp-content/uploads/2015/02/NEST_by_Example.pdf)

The tables list the parameters of a network of 10,000 neurons:

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
source: Marc-Oliver Gewaltig, ResearchGate

The state of the network depends on two parameters: the relative strength of inhibition $$g = -J_I / J_E$$, here 5, and the external rate $$\eta$$. $$\nu_{\text{thr}} = \theta / (J_E C_E \tau)$$ is the external rate at which the mean external input alone reaches the threshold. Brunel finds four states:
- **SR**, synchronous regular ($$g = 3$$, $$\eta = 2$$): the neurons fire regularly, almost in synchrony, at high rates;
- **SI fast**, synchronous irregular ($$g = 6$$, $$\eta = 4$$): the global activity oscillates fast, and the neurons fire irregularly, at a lower rate than the oscillation;
- **AI**, asynchronous irregular ($$g = 5$$, $$\eta = 2$$): the global activity is stationary, and the neurons fire irregularly;
- **SI slow**, synchronous irregular ($$g = 4.5$$, $$\eta = 0.9$$): the global activity oscillates slowly, and the neurons fire irregularly, at low rates.

The table compares the firing rates and global oscillation frequencies of his simulations with the predictions of his [mean-field theory][mean-field]:

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

The simulated rates are within 5 Hz of the predictions, and the frequencies within 10 Hz. The model is also cheap to simulate. A LIF neuron has one state variable, $$V$$, updated with one multiplication and one addition per step. A Hodgkin-Huxley neuron has four state variables per compartment, the potential and three gating variables, with nonlinear rate functions evaluated at every step. LIF networks thus suit long simulations of large networks.

## Simulation results

A `C++` implementation reproduces three of the four states, with the network of the tables. Each figure shows the spike times of the 10,000 neurons (top) and the number of spikes per ms (bottom), over 1 s.

**SI slow**:

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/lif-brunel/raster_plot_g4.5_f0.9.svg">

Its global oscillation has a period of about 50 ms, or 20 Hz, close to the 22 Hz of the paper.

**AI**:

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/lif-brunel/raster_plot_g5_f2.svg">

**SR**:

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/lif-brunel/raster_plot_g3_f2.svg">

## Other resources

- [Neuronal Dynamics][lif-gerstner], by Wulfram Gerstner et al., explains the LIF model in detail.
- [NEST by Example: An Introduction to the Neural Simulation Tool NEST][nest-brunel], by Marc-Oliver Gewaltig et al., implements the Brunel network in `Python` with the [NEST][nest] simulator.
- The `C++` code is in <a href="{{ site.assets }}/lif-brunel/lif-brunel.tar.gz">lif-brunel.tar.gz</a>, and its class diagram in <a href="{{ site.assets }}/lif-brunel/diagram_class.svg">diagram_class.svg</a>.
