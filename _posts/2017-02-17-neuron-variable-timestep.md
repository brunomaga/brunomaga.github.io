---
layout: post
title:  "Variable Timestep Simulation of the Electrical Activity of Neurons"
categories: ["Computational Neuroscience"]
tags: ["CVODE", "variable timestep", "numerical methods", "simulation", "HPC"]
---

[neuron-methods-post]: {{ site.baseurl }}{% post_url 2016-08-15-neuron-electrical-activity %}

The [previous post][neuron-methods-post] integrates detailed neuron models with a fixed time step $$\Delta t$$, usually 25 μs. The step must resolve the fastest change of the solution, the rise of a spike, which lasts a fraction of a millisecond. Between spikes, the voltages and states change slowly, and most fixed steps add no accuracy. A [variable time step][adaptive-step] method sizes each step from an estimate of its error: short steps during a spike, long steps between spikes. Its weakness is discontinuities: every synaptic event changes the input of a neuron at once, and restarts the method with short steps. Variable steps thus pay off on neurons that receive few events.

[adaptive-step]: https://en.wikipedia.org/wiki/Adaptive_step_size

Implicit methods matter for complex models. The fixed-step method of the previous post solves one linear system per step when each state ODE is linear in its own state. Some models have nonlinear and coupled state equations, such as the synaptic plasticity model of [Graupner and Brunel (2012)][graupner2012], a cubic ODE of the synaptic efficacy driven by the calcium concentration. With fixed steps, NEURON integrates such models with `derivimplicit`: backward Euler with Newton iterations, at first order, one mechanism at a time, so the coupling between mechanisms stays explicit. CVODE, NEURON's variable-step method, solves the voltages and all the states together, implicitly, at variable order.

[graupner2012]: https://www.pnas.org/doi/10.1073/pnas.1109359109

## CVODE

[cvode]: https://aip.scitation.org/doi/pdf/10.1063/1.4822377
[sundials]: https://computation.llnl.gov/projects/sundials

CVODE ([Cohen and Hindmarsh, 1996][cvode]) solves the [initial value problem][ivp]

$$
\dot{y} = f(t,y) \text{, } y(t_0) = y_0 \text{, with } y \in \mathbb{R}^N
$$

where $$\dot{y} = dy/dt$$. For neurons, $$y$$ holds the voltages of all compartments and the states of all mechanisms, of one cell or of a whole network. NEURON's `CVode` class wraps two solvers of the [SUNDIALS][sundials] suite: CVODE for ODEs, and IDA for differential-algebraic equations, which the `extracellular` and `LinearMechanism` mechanisms require.

[ivp]: https://en.wikipedia.org/wiki/Initial_value_problem

Neuron models are [stiff][stiff]: their time scales differ by orders of magnitude, and explicit methods are stable on them only with steps of the order of the fastest time scale. For stiff problems, CVODE uses the [backward differentiation formulas][bdf] (BDF), a family of implicit [linear multistep methods][lmm]. The BDF of order $$q$$ computes $$y_n$$, the approximation of $$y(t_n)$$, from the $$q$$ previous values and the derivative at $$t_n$$:

$$
y_n = \sum_{i=1}^{q} \alpha_{n,i} \, y_{n-i} + h_n \beta_{n,0} \, \dot{y}_n
$$

where $$h_n = t_n - t_{n-1}$$ is the step size, and the coefficients $$\alpha_{n,i}$$ and $$\beta_{n,0}$$ depend on the order and on the recent step sizes. With a constant step $$h$$, the BDFs of orders 1 to 6 are:

- BDF-1, the [backward Euler][backward-euler] method: $$ y_{n+1} - y_n = h f (t_{n+1}, y_{n+1})$$;
- BDF-2: $$ 3y_{n+2} - 4y_{n+1} + y_n = 2 h f (t_{n+2}, y_{n+2})$$;
- BDF-3: $$ 11y_{n+3} - 18y_{n+2} + 9y_{n+1} - 2y_n = 6 h f (t_{n+3}, y_{n+3})$$;
- BDF-4: $$ 25y_{n+4} - 48y_{n+3} + 36y_{n+2} - 16y_{n+1} + 3y_n = 12 h f (t_{n+4}, y_{n+4})$$;
- BDF-5: $$ 137y_{n+5} - 300y_{n+4} + 300y_{n+3} - 200y_{n+2} + 75y_{n+1} - 12y_n = 60 h f (t_{n+5}, y_{n+5})$$;
- BDF-6: $$ 147y_{n+6} - 360y_{n+5} + 450y_{n+4} - 400y_{n+3} + 225y_{n+2} - 72y_{n+1} + 10y_n = 60 h f(t_{n+6}, y_{n+6})$$.

[stiff]: https://en.wikipedia.org/wiki/Stiff_equation
[bdf]: https://en.wikipedia.org/wiki/Backward_differentiation_formula
[lmm]: https://en.wikipedia.org/wiki/Linear_multistep_method
[backward-euler]: https://en.wikipedia.org/wiki/Backward_Euler_method

A method is zero-stable when a perturbation $$\epsilon$$ of its initial values changes the numerical solution on a finite interval by at most $$K \epsilon$$, with $$K$$ independent of the step size. The BDFs above order 6 are not zero-stable, so they do not converge. Orders 1 and 2 are A-stable: on a decaying linear ODE, they are stable with any step. Orders 3 to 6 are stable only when $$h \lambda$$ lies in a region that shrinks as the order grows, where $$\lambda$$ is an eigenvalue of the [Jacobian][jacobian] $$J = \partial f / \partial y$$. CVODE uses the BDFs of orders 1 to 5.

[jacobian]: https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant

### Newton Iterations

Each step of CVODE solves the nonlinear system

$$
G(y_n) \equiv y_n - \gamma f(t_n, y_n) - a_n = 0 \text{, with } \gamma = h_n \beta_{n,0} \text{ and } a_n = \sum_{i=1}^{q} \alpha_{n,i} \, y_{n-i}
\label{eq_variable_time_step_bdf}
$$

The fixed-point iteration $$y_{n(m+1)} = \gamma f(t_n, y_{n(m)}) + a_n$$ converges when all eigenvalues of $$\gamma J$$ have a magnitude below 1. A stiff problem has eigenvalues of large magnitude, so this condition limits the step to the fastest time scale, as for an explicit method. CVODE uses [Newton's method][newton] instead. Starting from a prediction $$y_{n(0)}$$, extrapolated from the past values, iteration $$m$$ solves the linear system

$$
M \, (y_{n(m+1)} - y_{n(m)}) = - G(y_{n(m)}) \text{, with } M \approx I - \gamma J
$$

[newton]: https://en.wikipedia.org/wiki/Newton%27s_method

The solves with $$M$$ are the main cost of a step. NEURON offers three solvers, selected with `CVode.jacobian`:
- **dense** (`jacobian(1)`): $$J$$ from finite differences, and an [LU factorization][lu] of $$M$$ in $$O(N^3)$$ time and $$O(N^2)$$ memory. A detailed cell has $$10^4$$ or more states, so its matrix alone takes 800 MB or more;
- **diagonal** (`jacobian(2)`): the diagonal of $$J$$ only, in $$O(N)$$ time and memory. It ignores the axial coupling between compartments, the stiffest part of the system;
- **NEURON's own solver** (`jacobian(0)`, the default): $$J$$ keeps the axial coupling of the voltages and the derivatives of the state equations of each mechanism with respect to its own states. It drops the derivatives of the currents with respect to the states, and of the state equations with respect to the voltage. The voltage part is the tree matrix of the backward Euler step of the previous post, with $$\gamma$$ in place of $$\Delta t$$, which the Hines algorithm solves in $$O(N)$$ time. Each mechanism then solves for its own states.

[lu]: https://en.wikipedia.org/wiki/LU_decomposition

A poorer $$M$$ slows the convergence of the iterations, which then need more iterations or shorter steps. In the tests of Magalhães et al. (<a href="{{ site.assets }}/neuron-variable-timestep/neurox-vardt-arxiv.pdf">preprint, 2019</a>), the diagonal solver ran faster but gave inaccurate results, and the dense solver was accurate but too slow and too large. CVODE also has a banded solver and a matrix-free [Krylov][gmres] solver (SPGMR), which NEURON does not use.

[gmres]: https://en.wikipedia.org/wiki/Generalized_minimal_residual_method

### Error Control

The user sets no step size, but an absolute tolerance $$\text{atol}_i$$ per state and a relative tolerance $$\text{rtol}$$. CVODE measures errors and corrections with the weighted [root mean square][rms] norm

$$
\Vert v \Vert = \sqrt{ \frac{1}{N} \sum_{i=1}^N \left( \frac{v_i}{W_i} \right)^2 } \text{, with } W_i = \text{rtol} \, \vert y_i \vert + \text{atol}_i
$$

so that a vector of norm 1 is at the tolerance. The local truncation error of a step is estimated as $$C' \Delta_n$$, where $$\Delta_n = y_{n(m)} - y_{n(0)}$$ is the difference between the corrected and the predicted solution, and $$C'$$ depends on the order. The step passes the error test when $$\Vert \Delta_n \Vert \le \epsilon = 1 / \vert C' \vert$$. Otherwise, CVODE repeats it with a shorter step, and after three failures, at order 1. The next step size is $$\eta h_n$$, with

$$
\eta = \left( \frac{\epsilon}{6 \, \Vert \Delta_n \Vert} \right)^{1/(q+1)}
$$

capped at 10. After $$q+1$$ steps at order $$q$$, CVODE also computes $$\eta$$ for the orders $$q-1$$ and $$q+1$$, and moves to the order that allows the longest step.

[rms]: https://en.wikipedia.org/wiki/Root_mean_square

The Newton iterations stop when the iteration error is ten times below the error test: $$\Vert y_n - y_{n(m)} \Vert < 0.1 \epsilon$$, where $$y_n$$ solves equation \eqref{eq_variable_time_step_bdf} and the factor 0.1 can be changed. CVODE estimates this error from the convergence rate $$R$$ of the iterations. $$R$$ starts at 1, returns to 1 when $$M$$ is updated, and after each correction $$\delta_m = y_{n(m)} - y_{n(m-1)}$$ with $$m > 1$$ becomes

$$
R \leftarrow \max \left\{ 0.3 R, \frac{\Vert \delta_m \Vert}{\Vert \delta_{m-1} \Vert} \right\}
$$

Then $$\Vert y_n - y_{n(m)} \Vert \approx \Vert y_{n(m+1)} - y_{n(m)} \Vert \approx R \, \Vert \delta_m \Vert$$, and the convergence test is

$$
R \, \Vert \delta_m \Vert < 0.1 \epsilon
$$

A step allows at most three iterations. When they do not converge, CVODE updates $$M$$ and retries the step, or, if $$M$$ was up to date, retries it with a step four times shorter.

[neuron-book]: https://www.cambridge.org/core/books/neuron-book/7C8D9BD861D288E658BEB652F593F273

NEURON's defaults are $$\text{atol} = 10^{-2}$$ and $$\text{rtol} = 0$$: an absolute error control in the units of each state, scaled per state with `CVode.atolscale`. The tolerances are 10 μV for voltages in mV, and 0.1 nM for the internal calcium concentration ([Carnevale and Hines, 2006][neuron-book]). For a Hodgkin-Huxley action potential at 6.3 °C, they give an accuracy comparable to that of the second-order fixed-step method with $$\Delta t = 25$$ μs.

## Events and the Local Variable Time Step

[lytton2005independent]: https://www.mitpressjournals.org/doi/abs/10.1162/0899766053429453

An event, such as a synaptic input, changes the right-hand side $$f$$ at a given time. The BDF history before the event does not describe the solution after it. The integrator therefore stops at the event time, applies the event, and restarts as a new initial value problem, at order 1 and with a short step. Each event thus costs a series of short steps.

With one integrator for the whole network, every event restarts all cells, and all cells advance with the steps of the most active one. The **local variable time step** method (lvardt) of [Lytton and Hines (2005)][lytton2005independent] gives each cell its own CVODE integrator instead. An event restarts the integrator of its target cell only: an active cell takes short steps, while a quiet cell takes long ones. A coordinator advances the integrators:
- each integrator can interpolate its solution to any time of its last step, $$[t_a, t_b]$$, from its BDF history;
- an integrator queue orders the cells by $$t_b$$, and an event queue orders the events by delivery time. Both are [priority queues][priority-queue], with $$O(\log N_c)$$ time per operation for $$N_c$$ cells;
- at each iteration, the coordinator takes the earliest item of the two queues. A cell integrates one step forward. An event makes its target cell interpolate back to the event time, handle the event, and restart its integrator there.

[priority-queue]: https://en.wikipedia.org/wiki/Priority_queue

Since the coordinator always takes the earliest item, no $$t_b$$ and no event time ever falls behind any $$t_a$$: every event lies within the interpolation interval of its target cell, or ahead of it. A spike found in a step stays tentative until the earliest $$t_b$$ and event time reach it, since an earlier event to the spiking cell can restart the cell before the spike.

The picture shows lvardt on six cells. The black rectangles are the intervals $$[t_a, t_b]$$, the vertical line marks the earliest $$t_b$$ or event time, and the short thick bars are events. (A) Cell 0 has the smallest $$t_b$$ and integrates one step (hatched). Its voltage crosses the spike threshold during the step, which creates a tentative spike. (B) Cell 5 now has the smallest $$t_b$$ and integrates one step. (C) The spike of cell 0 is now the earliest item. Handling it creates three events, for cells 3, 4 and 5, with different delays, and the event of cell 3 becomes the earliest. (D) Cell 3 interpolates back to the event time, handles the event and restarts its integrator, so its interval shrinks to a point. Cell 3 integrates next:

{: .picture}
<img width="30%" height="30%" src="{{ site.assets }}/neuron-variable-timestep/local_variable_timestep_six_cell_example.svg">

{: .caption}
Adapted from: [The NEURON book][neuron-book]

## Single Neuron

The following tests of Magalhães et al. (<a href="{{ site.assets }}/neuron-variable-timestep/neurox-vardt-arxiv.pdf">preprint, 2019</a>) compare NEURON's backward Euler, with its default $$\Delta t = 25$$ μs, and CVODE, with $$\text{atol} = 10^{-3}$$, on a model of a layer 5 pyramidal cell of the Blue Brain Project (L5_TTPC2_cADpyr232_1), with a current injected at the soma. The first test follows one spike. Backward Euler takes equidistant steps. CVODE takes long steps while the voltage changes slowly, before the spike, and short steps on the rise and around the peak:

{: .picture}
<img width="60%" height="60%" src="{{ site.assets }}/neuron-variable-timestep/l5_neuron_soma_voltage_per_step_6ms.svg">

A constant current from 100 ms to 1,100 ms makes the neuron fire at a rate that grows with the current. Over the 1.2 s of simulation, backward Euler takes 48,001 steps for every current. CVODE takes 338 steps without spikes, and 9,750 at the highest current:

{: .picture}
<img width="60%" height="60%" src="{{ site.assets }}/neuron-variable-timestep/cvode_pulse_currents_table.svg">

The step size also sets the accuracy. Over 100 ms, backward Euler with $$\Delta t = 1$$ μs and CVODE with $$\text{atol} = 10^{-4}$$ give the same spike times, and serve as references. With longer steps, backward Euler spikes later. CVODE with $$\text{atol} = 10^{-2}$$ matches the accuracy of backward Euler with $$\Delta t = 25$$ μs in 7 times fewer steps (566 against 4,001), and CVODE with $$\text{atol} = 10^{-3}$$ matches that of $$\Delta t = 5$$ μs in 22 times fewer steps (910 against 20,001):

{: .picture}
<img width="60%" height="60%" src="{{ site.assets }}/neuron-variable-timestep/l5_neuron_pulse1_3ma_100ms_cvode_vs_backward_euler.svg">

The delay of backward Euler grows spike after spike. After 1 s, it is about 2 ms with $$\Delta t = 5$$ μs and 6.5 ms with $$\Delta t = 25$$ μs, a large error at the time scale of spikes. CVODE with $$\text{atol} = 10^{-3}$$ takes 6,240 steps, against 40,001 and 200,001 for backward Euler:

{: .picture}
<img width="60%" height="60%" src="{{ site.assets }}/neuron-variable-timestep/l5_neuron_pulse_1000ms_results.svg">

The next test injects a constant current for 1 s, from 0 to 5 times the threshold current, the smallest constant current that makes the neuron spike. The runs use an Intel Core i5 at 1.6 GHz. Below half the threshold current, the voltage settles to a steady state, and CVODE takes 434 times fewer steps than backward Euler and runs 98 times faster. At the threshold current, the gains are 62 and 11.6 times, and at 5 times the threshold current, 9.4 and 2.5 times. A CVODE step costs 4 to 5 times as much as a backward Euler step: it runs Newton iterations, each with an evaluation of $$f$$ and a linear solve, and an error test.

{: .picture}
<img width="60%" height="60%" src="{{ site.assets }}/neuron-variable-timestep/cvode_dependency_on_variation_solution.svg">

The last single-cell test replaces the constant current by pulses of 1 μs at a fixed rate, as a model of synaptic events. Each pulse is a discontinuity. The runtime of CVODE grows with the pulse rate, and depends little on the pulse amplitude. At 10 pulses per second, CVODE takes 153 to 322 times fewer steps than backward Euler and runs 51 to 100 times faster, from the strongest to the weakest pulse. For the strongest pulse, both methods run equally fast at about 1,000 pulses per second.

{: .picture}
<img width="60%" height="60%" src="{{ site.assets }}/neuron-variable-timestep/cvode_dependency_on_events_arrival.svg">

## Homogeneous Networks

[Lytton and Hines (2005)][lytton2005independent] tested lvardt on one compute node. Their worst case is a fully connected network of 100 inhibitory neurons, which synchronize through mutual inhibition: each population spike sends 9,900 events at almost the same time. A fixed step of 2.5 μs gave results close to those of lvardt with $$\text{atol} = 10^{-3}$$ or $$10^{-5}$$. Event delivery took about 12% of the CPU time of lvardt. A single integrator for the whole network was up to 60 times slower than lvardt, since every event restarted all cells.

The picture shows the CPU time of this model. With fixed steps (dashed lines), it grows linearly with the simulated time. lvardt is slow at the start, while the cells are not yet synchronized and its steps vary widely. Once the network synchronizes, lvardt alternates long steps between the population spikes and short steps during them, and its CPU time grows slowly:

{: .picture}
<img width="30%" height="30%" src="{{ site.assets }}/neuron-variable-timestep/lvardt_exec_time_comparison.svg">

{: .caption}
source: Lytton et al. 2005 

Their best case, a synfire chain of 100 single-compartment Hodgkin-Huxley cells, ran 20 times faster with lvardt than with a fixed step of similar accuracy. A thalamocortical model has four cell types: cortical pyramidal cells and interneurons, thalamocortical cells and thalamic reticular cells. The two thalamic types fire bursts separated by long silent intervals, which suits lvardt. For 150 ms of simulation at high precision, lvardt with $$\text{atol} = 10^{-6}$$ took 6 min 13 s, and a fixed step of 0.1 μs took 10 h 20 min, 100 times longer. At standard precision, with $$\text{atol} = 10^{-3}$$ and $$\Delta t = 10$$ μs, the runs took 2 min and 5 min 53 s, 3 times longer.

## A Heterogeneous Network

[markram2015cell]: https://www.cell.com/fulltext/S0092-8674(15)01191-5

Homogeneous networks give only a rough estimate of the speed-up, since the activity of a brain region varies across neurons and over time. The event rates of a realistic network come from a fixed-step simulation of a digital reconstruction of the rat neocortex by the Blue Brain Project ([Markram et al., 2015][markram2015cell]): 7.5 s of spontaneous activity of 219,247 neurons under tonic depolarization. The network shows slow oscillations, with population bursts that start in layer 5 and spread to the other layers.

The pictures show two neurons of the top 1% by number of incoming events: the events per 10 ms (left), and the histogram of the intervals between events (right). The neurons of the top 1% receive 3,040 to 6,146 events, 405 to 820 per second:

{: .picture}
<img width="60%" height="60%" src="{{ site.assets }}/neuron-variable-timestep/neuron_203076.svg">
<img width="60%" height="60%" src="{{ site.assets }}/neuron-variable-timestep/neuron_138083.svg">

The neurons of the median 1% receive 541 to 558 events, about 73 per second:

{: .picture}
<img width="60%" height="60%" src="{{ site.assets }}/neuron-variable-timestep/neuron_208223.svg">
<img width="60%" height="60%" src="{{ site.assets }}/neuron-variable-timestep/neuron_87784.svg">

The neurons of the bottom 1% receive fewer than 100 events:

{: .picture}
<img width="60%" height="60%" src="{{ site.assets }}/neuron-variable-timestep/neuron_132412.svg">
<img width="60%" height="60%" src="{{ site.assets }}/neuron-variable-timestep/neuron_2101.svg">

In total, the neurons receive about 155 million events, 707 per neuron on average, or 94 per second: an order of magnitude below the break-even rate of the pulse test. The median and bottom neurons have long silent intervals between events, which suits variable steps. The top neurons do not.

## Distributed Execution

Networks of this size need distributed memory: as in the [previous post][neuron-methods-post], each rank integrates its own neurons, and the ranks exchange spikes.

### Bulk-Synchronous Parallel

NEURON splits the simulation into intervals of $$\Delta t_{\text{comm}}$$, the minimum synaptic delay between neurons on different ranks. Within an interval, each rank integrates its neurons independently. At its end, a collective call exchanges the spikes, and acts as a barrier: no rank starts the next interval before all ranks finish the current one. This is the [bulk synchronous parallel][bsp] (BSP) model. With variable steps, a neuron may take one step past the barrier. When an event from another rank arrives for a time inside that step, the neuron interpolates back and restarts, which discards part of the step. The barriers also cost more with variable steps than with fixed steps. The work of a rank depends on the activity of its neurons, so the ranks finish each interval at different times, and all of them wait for the slowest one: a [load imbalance][load-balancing] at every interval.

[bsp]: https://en.wikipedia.org/wiki/Bulk_synchronous_parallel
[load-balancing]: https://en.wikipedia.org/wiki/Load_balancing_(computing)

### Fully-Asynchronous Parallel

The fully-asynchronous parallel (FAP) model of Magalhães et al. (<a href="{{ site.assets }}/neuron-variable-timestep/neurox-cache-efficient.pdf">ICCS 2019</a>) removes the barriers. Each neuron $$j$$ stores, for each pre-synaptic neuron $$i$$, the last time $$t_i$$ that $$i$$ reported, and the minimum delay $$d_{ij}$$ of the connections from $$i$$ to $$j$$. A spike of $$i$$ after $$t_i$$ reaches $$j$$ at $$t_i + d_{ij}$$ or later, so $$j$$ can integrate up to

$$
t_j^{\max} = \min_{i \in \text{pre}(j)} \left( t_i + d_{ij} \right)
$$

without missing an event. Each neuron reports its time to its post-synaptic neurons at intervals set per pair, from 0.1 ms to several milliseconds, with one message per target rank. A scheduler on each rank steps the neuron with the earliest time first. No step is ever undone, and no rank waits at a barrier. Most delays $$d_{ij}$$ exceed $$\Delta t_{\text{comm}}$$, so neurons integrate longer between messages, while their data stays in cache. With fixed steps, FAP alone runs 26% to 65% faster than BSP on one compute node, depending on the architecture, and 16% to 40% faster on 32 nodes. The gain shrinks as the network grows.

A speculative variant, which lets neurons step ahead and undo the steps that a late event invalidates, does not scale: an undone step can invalidate spikes already sent to other ranks, which then undo their own steps.

### Benchmark

[LIF-model-post]: {{ site.baseurl }}{% post_url 2016-02-08-lif-brunel %}

The <a href="{{ site.assets }}/neuron-variable-timestep/neurox-vardt-arxiv.pdf">preprint</a> compares fixed and variable steps, with BSP and FAP, on 64 Cray XE6 nodes (AMD Opteron 6380, 16 cores at 2.5 GHz), for networks of 1,024 to 65,536 neurons and 1 s of activity, with $$\Delta t = 25$$ μs and $$\text{atol} = 10^{-3}$$:
- simple models, with state ODEs linear in their own state: (1a) BSP with `cnexp`, (1b) BSP with explicit Euler for the states, and (1c) FAP with explicit Euler;
- complex models: (2a) BSP with `derivimplicit`, (2b) BSP with CVODE, and (2c) FAP with CVODE, with each event delivered at its exact time, or with event grouping (EG): the events of each window of $$\Delta t / 2$$ or $$\Delta t$$ are delivered together, as with fixed steps.

A constant current drives all neurons to one of five spiking rates:
- quiet, 0.25 Hz: neurons almost at rest. The studies that the preprint cites place about 90% of the neurons of the brain at or below this rate in regular activity;
- slow, 1.5 Hz: the lowest rate of active neurons;
- moderate, 6.5 Hz: the rate of the SI slow state of the [Brunel network][LIF-model-post], above the 0.005 to 5 Hz of the rat frontal cortex;
- fast, 38 Hz: the rate of the AI state of the Brunel network, and of cortical and thalamic neurons during high vigilance, 30 to 40 Hz;
- burst, 55.8 Hz: the rate of the SI fast state of the Brunel network, typical of strong depolarizing currents.

{: .picture}
<img width="90%" height="90%" src="{{ site.assets }}/neuron-variable-timestep/benchmark_two_rows_2.svg">

The runtimes of the fixed-step methods differ by about 2% across the five rates: each step costs the same with or without spikes, and spikes add little work. The runtimes of CVODE grow with the rate, since events arrive at any time, not only at the ends of fixed steps, and each one restarts its target cell with short steps. Against the fixed-step method for complex models (2a), FAP CVODE with grouping per $$\Delta t$$ runs 544 times faster on 1,024 neurons and 65 times faster on 65,536 neurons at the quiet rate, and 7.7 to 1.8 times faster at the moderate rate. At the fast rate, it runs 2 times faster on 1,024 neurons and as fast on 65,536. At the burst rate, it runs 1.5 times faster on 1,024 neurons and 1.5 times slower on 65,536. At the quiet rate, FAP CVODE, which solves the complex model with Newton iterations, also beats the fixed-step methods for simple models, which solve one linear system per step: the rare events allow long steps.

Event grouping matters on large networks: on 65,536 neurons at the quiet rate, the speed-up falls from 65 times to 47 with grouping per $$\Delta t / 2$$, and to 33 without grouping. The gap between BSP and FAP shrinks as the network grows, since more connections per neuron bring $$t_j^{\max}$$ closer to $$\Delta t_{\text{comm}}$$. On 65,536 neurons, both models run in similar times with fixed steps, and with CVODE at the moderate, fast and burst rates.

To estimate the speed-up on a realistic mix of rates, the preprint groups the 31,346 neurons of the central column of the 219,247-neuron network by firing rate, between 2 s and 4 s of activity: 31.43% quiet, 38.44% slow, 27.02% moderate, 3.10% fast and 0.01% burst. Weighting the runtimes of each rate gives a speed-up of FAP CVODE over the fixed-step method (2a) of 224.5 to 11.9 times with exact event delivery, and of 228.5 to 24.6 times with grouping per $$\Delta t$$, from 1,024 to 65,536 neurons. The quiet rate holds 31% of the neurons here, against about 90% in the literature, so these estimates are lower bounds. Over 95% of the neurons fall in the three slowest rates, where the runtimes of both methods grow at the same rate beyond 16,384 to 32,768 neurons, so the speed-up should hold on larger networks.
