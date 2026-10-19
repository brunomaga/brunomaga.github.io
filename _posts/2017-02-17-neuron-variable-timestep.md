---
layout: post
title:  "Variable Timestep Simulation of the Electrical Activity of Neurons"
categories: ["Computational Neuroscience"]
tags: ["CVODE", "variable timestep", "numerical methods", "simulation", "HPC"]
---

[neuron-methods-post]: {{ site.baseurl }}{% post_url 2016-08-15-neuron-electrical-activity %}

The [previous post][neuron-methods-post] detailed the mathematical background underlying neuron activity. It then detailed the simulation of the electrical activity of simple neuron models, and mentioned that an interleaved voltage-current resolution of linear, unrelated state ODEs allows the problem to be solved as a system of linear equations. However, such a resolution is only possible for those *simple* models. Neuron models with non-linear state variable ODEs and correlation between states require a fully-implicit resolution, which is possible with [Backward Euler](https://en.wikipedia.org/wiki/Backward_Euler_method) with Newton iterations. This is a common use case, and underlies the resolution of extra-cellular mechanisms, capacitors between nodes, and other mechanisms that connect neurons' evolution over time.

A fully-implicit method that is commonly utilised to resolve this problem is the Backward (Implicit) Euler, where:

$$
y_{n+1} - y_n = h f (t_{n+1}, y_{n+1})
$$

where $$h$$ is the timestep, $$y$$ is the array of states and $$f$$ is the right-hand side function of the ODE. Can one do better? This post presents a fully-implicit method based on variable-order variable timestepping.

## CVODE for fully-implicit adaptive-order adaptive-step interpolation
 
[cvode]: https://aip.scitation.org/doi/pdf/10.1063/1.4822377
[sundials]: https://computation.llnl.gov/projects/sundials

The main principle supporting [adaptive time stepping](https://en.wikipedia.org/wiki/Adaptive_step_size) is its particular efficiency, due to large step sizes during long periods of slowly varying states, e.g. interspike intervals. NEURON's adaptive algorithm, [CVODE][cvode], can use separate variable-step integrators for individual neurons in the network. CVODE is used by NEURON as an interface to a collection of methods built on top of the existing CVODE and IDA (Integrator for Differential-Algebraic problems) integrators of the [SUNDIALS package][sundials], a SUite of Nonlinear and DIfferential/ALgebraic Equation Solvers. NEURON employs the [Backward Differentiation Formula](https://en.wikipedia.org/wiki/Backward_differentiation_formula) (BDF), a family of implicit multi-step methods for the numerical integration of (stiff) ordinary differential equations. For a given function and time, the BDF approximates its derivative using information from previously computed times, thus increasing the accuracy of the approximation. BDF methods are implicit, and thus require computing the solution of nonlinear equations at every step. The BDF multistep method is suitable for stiff problems, such as those that define neuronal activity. The main idea behind a [stiff equation](https://en.wikipedia.org/wiki/Stiff_equation) is that the equation includes terms that may lead to rapid variation of the solution, making certain numerical methods for solving the equation numerically unstable, unless the step size is taken to be *extremely* small. CVODE is a solver for the [Initial Value Problem](https://en.wikipedia.org/wiki/Initial_value_problem) (IVP), an ODE together with an initial condition of the unknown function at a given point in the domain of the solution, that can be written abstractly as:

$$
\dot{y} = f(t,y) \text{, } y(t_0) = y_0 \text{ where } y \in \mathbb{R}^N
$$

where $$\dot{y}$$ denotes $$\frac{dy}{dt}$$. The solution $$y$$ is differentiated using past (backward) values, i.e. $$\dot{y}_n$$ is a linear combination of $$y_n$$, $$y_{n-1}$$, $$y_{n-2}, ...$$. $$y$$ is the array of all state variables, in this case the voltages $$V_j$$ of all compartments (governed by $$dV_j/dt$$) and the states $$x_i$$ (governed by $$dx_i/dt$$), where $$x_i$$ is a given state variable of an ion channel $$i$$. The variable-step BDF of order $$q$$ can be written as:

$$
y_n = \sum_{i=1}^{q} \alpha_{n,i} y_{n-i} + h_n \beta _{n,0} \dot{y}_n
$$

where $$y_n$$ are the computed approximations to $$y (t_n)$$, $$h_n = t_n - t_{n-1}$$ is the step size, $$q$$ denotes the order (number of previous steps used by the BDF), and $$\alpha$$ and $$\beta$$ are the unique BDF coefficients, dependent on the method order and on the recent step sizes. With a constant step size $$h$$, the resolution can be unfolded as:

- BDF-1: $$ y_{n+1} - y_n = h f (t_{n+1}, y_{n+1})$$, i.e. the Backward Euler;
- BDF-2: $$ 3y_{n+2} - 4y_{n+1} + y_n = 2 h f (t_{n+2}, y_{n+2})$$;
- BDF-3: $$ 11y_{n+3} - 18y_{n+2} + 9y_{n+1} - 2y_n = 6 h f (t_{n+3}, y_{n+3})$$;
- BDF-4: $$ 25y_{n+4} - 48y_{n+3} + 36y_{n+2} - 16y_{n+1} + 3y_n = 12 h f (t_{n+4}, y_{n+4})$$;
- BDF-5: $$ 137y_{n+5} - 300y_{n+4} + 300y_{n+3} - 200y_{n+2} + 75y_{n+1} - 12y_n = 60 h f (t_{n+5}, y_{n+5})$$;
- BDF-6: $$ 147y_{n+6} - 360y_{n+5} + 450y_{n+4} - 400y_{n+3} + 225y_{n+2} - 72y_{n+1} + 10y_n = 60 h f(t_{n+6}, y_{n+6})$$.

BDF with order $$q > 6$$ is not [zero-stable](https://en.wikipedia.org/wiki/Zero_stability), and therefore cannot be used. Zero-stability means that, if a perturbation $$\epsilon$$ happens on an initial value, then the difference between the perturbed and unperturbed results must be no bigger than $$k \epsilon$$, for some constant $$k \in \mathbb{R}$$ independent of the step size. CVODE allows for a maximum order of 5.

If the iterates are denoted by $$y_{n(m)}$$, the functional iteration of the BDF is given by:

$$
y_{n(m+1)} = h_n \beta _{n,0} f ( t_n, y_{n(m)} ) + a_n
\label{eq_newton_method_derivation}
$$

As with any implicit method, at every step one must (approximately) solve the nonlinear system:

$$
G(y_n) \equiv y_n - h_n \beta_{n,0} f(t_n, y_n) - a_n = 0 \text{ , where } a_n \equiv \sum_{i=1}^{q} \alpha_{n,i} y_{n-i}  
\label{eq_variable_time_step_bdf}
$$

To resolve this, CVODE applies Newton iterations. [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method) finds successively better approximations to the roots (or zeroes) of a function, and is implemented (for a function $$g$$ of one variable) as $$x_{m+1} = x_m - g(x_m) / g'(x_m)$$ until a sufficiently accurate value is reached. In practice, it requires finding the solution of the linear systems:

$$
M [y_{n(m+1)} - y_{n(m)}] = - G(y_{n(m)})
$$

where 

$$
M \approx I - \gamma J \text{ , } J = \partial f / \partial y \text{ , and } \gamma = h_n \beta_{n,0}
$$

For the Newton corrections, CVODE provides the following options: (a) a direct dense solver, (b) a direct banded solver, (c) a diagonal approximate Jacobian solver, or the SPGMR (Scaled Preconditioned [GMRES](https://en.wikipedia.org/wiki/Generalized_minimal_residual_method)) method. A preconditioner matrix is a transformation that conditions a given problem into a form that is more suitable for numerical solving methods. The SPGMR method uses a preconditioner matrix $$P$$ and performs an Inexact Newton iteration, in which $$M$$ is applied in a matrix-free manner, and the matrix-vector products $$Jv$$ are obtained by either difference quotients or a user-supplied routine. In practice, the Newton iteration transforms the nonlinear problem into a linear one by introducing a multiplication by the Jacobian (in vector calculus, the [Jacobian matrix](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) is the matrix of all first-order partial derivatives of a vector-valued function), and it involves computing the solution of the equation:

$$
\begin{align*}
y_{n(m+1)} = & y_{n(m)} -  M^{-1} G(y_{n(m)}) \\
\equiv & y_{n(m+1)} - y_{n(m)} = - M^{-1} G(y_{n(m)}) \\
\equiv & M [y_{n(m+1)} - y_{n(m)}] = - G(y_{n(m)}) \\
\equiv & P^{-1} M [y_{n(m+1)} - y_{n(m)}] = - P^{-1} G(y_{n(m)}) \\
\end{align*}
$$

where 

$$
 P \approx I - \gamma J \text{ , } J = \partial f / \partial y \text{ , and } \gamma = h_n \beta_{n,0}
$$

The previous equation requires a division by $$M$$, the Jacobian of $$G$$, i.e. a multiplication by its inverse. Due to the expensive computation of the inverse of the Jacobian, CVODE allows for a trade-off between the accuracy and the solution time of any equation, by requiring one to only supply a solver for $$P y = b$$ instead. NEURON does not use SPGMR: it supplies its own solver for $$P y = b$$, in which the *SOLVE* block of each mechanism solves for the states of that mechanism, and takes $$P$$ in place of $$M$$ in the Newton iteration. For this linear system, although NEURON supports a resolution with the Hines solver (default) or the (a) or (c) methods mentioned previously, there has been little exploration of the non-Hines approaches.

## Error Handling

One main difference between fixed and variable time stepping is that, from the user's perspective, one does not specify the time step but the relative (*rtol*) and absolute (*atol*) local error tolerances instead. The solver then adjusts $$\Delta t$$ based on the two values. The local error is based on the extrapolation process within a time step. CVODE uses a weighted [root-mean-square](https://en.wikipedia.org/wiki/Root_mean_square) norm for all error-like quantities, denoted as:

$$
|| v || = \sqrt{ \frac{1}{N} \sum_{i=1}^N (v_i / W_i)^2}
$$

where $$v_i$$ represents the deviations, and the weights $$W_i$$ are based on the current solution (with components $$y^i$$), with user-specified absolute tolerances $$atol_i$$ and relative tolerance $$rtol$$, as:

$$
W_i =  rtol \cdot | y^i | + atol_i
$$

If $$y_n$$ is the exact solution of equation \ref{eq_variable_time_step_bdf}, CVODE ensures that the iteration error $$y_n - y_{n(m)}$$ is small enough, such that $$\| y_n - y_{n(m)} \| < 0.1 \epsilon$$, where $$\epsilon$$ is the tolerance of the local error test in the weighted norm above (and $$0.1$$ is user-changeable). For this, it also estimates the linear convergence rate constant $$R$$, initialized at $$1$$ and reset to $$1$$ when $$M$$ or $$P$$ is updated. After computing a correction $$\delta_m = y_{n(m)} - y_{n(m-1)}$$, $$R$$ is updated if $$m > 1$$ as:

$$
R \leftarrow \text{max} \left\{ 0.3 R, \frac{|| \delta_m ||}{||\delta_{m-1} ||} \right\}
$$

It is used to estimate

$$
||y_n - y_{n(m)}|| \approx || y_{n(m+1)} - y_{n(m)}|| \approx R || y_{n(m)} - y_{n(m-1)}|| = R || \delta_m ||
$$

Thus, the convergence test is 

$$
R || \delta_m || < 0.1 \epsilon
$$

## Local and Global Time Stepping

NEURON's default error setting for CVODE is $$10 \mu V$$ for membrane potential and $$0.1 nM$$ for internal free calcium concentration, giving an HH-based action potential an accuracy equivalent to that of a fixed time step of $$\Delta t = 25 \mu s$$. The default maximum order for BDF is $$5$$, although *it is recommended* to let CVODE decide the best order.

[lytton2005independent]: https://www.mitpressjournals.org/doi/abs/10.1162/0899766053429453
[neuron-book]: https://www.cambridge.org/core/books/neuron-book/7C8D9BD861D288E658BEB652F593F273

Large networks of cells cannot be integrated efficiently with global adaptive time stepping, as a high frequency of events such as gating channels or synapses causes discontinuities in parameters or state variables, which require the reinitialization of the computation and force the integrator to start again with a new initial value problem. NEURON's workaround is a local variable time stepping scheme called *lvardt* ([Lytton and Hines, Independent Variable Time-Step Integration of Individual Neurons for Network Simulations, MIT Press][lytton2005independent]) that creates individual CVODE integrators per cell, and aims at short time step integration for active neurons and long time step integration for inactive neurons, while keeping the integration accuracy consistent. All cells are kept in an ordered queue (a [splay tree](https://en.wikipedia.org/wiki/Splay_tree)), where their position is given by their event delivery time or their current simulation time. Because the system moves forward in time with multiple, independent integrators, the overall computation is reduced when compared to the global counterpart. Nevertheless, the main issue is the coordination of the time advancement of each cell, i.e. all the state variables of a given cell should be correctly available to any incoming event. To solve that issue, every cell holds the information for all states between its current time instant and the previous time step. This allows for backward interpolation when required. The integration coordinator also ensures that the interpolation intervals of the connected cells overlap, so that no event between two cells is missed. The critical point of this approach is that the time step is tentative, and an event received by the cell in the interim may reinitialize the cell to a time prior to the tentatively calculated threshold time.

The following picture illustrates a timeline (from A to D) of six cells running the *lvardt* algorithm. The intervals of possible interpolation are displayed as black rectangles. The right end of a cell's rectangle, $$t_b$$, is the time up to which that cell has integrated. The system's earliest $$t_b$$ or event time is shown by the vertical line traversing all neurons. Triggers of discrete events on a particular neuron are displayed as short thick vertical bars. (A) The cell with the smallest $$t_b$$ (cell 0) advances by the length of the hashed rectangle, and crosses the spike threshold during that step, creating a tentative trigger event; (B) Cell 5 has the smallest $$t_b$$ and integrates forward; (C) The next earliest event in the system is the trigger of cell 0. This event trigger creates three new events to be delivered to cells 3, 4 and 5, at different delays; (D) The new earliest event forces cell 3 to back-interpolate, the event is handled, and cell 3 reinitializes. Cell 3 will be the next cell to integrate forward:

{: .picture}
<img width="40.4%" height="40.4%" src="{{ site.assets }}/neuron-variable-timestep/local_variable_timestep_six_cell_example.svg">

{: .caption}
Adapted from: [Lytton and Hines (2005), Independent variable time-step integration of individual neurons for network simulations, Fig. 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC2712447/)

## Single Neuron Performance

One can notice the difference in performance of the Backward Euler and Variable Timestep interpolators during a stiff trajectory of an Action Potential of a layer 5 pyramidal cell in the following picture.

{: .picture}
<img width="97%" height="97%" src="{{ site.assets }}/neuron-variable-timestep/l5_neuron_soma_voltage_per_step_6ms.svg">

Note the difference in interpolation instants: placed at equidistant intervals in the Backward Euler use case, and adapting to the solution gradient in the variable step counterpart. A current injection at the soma forces the neuron to spike at a rate that increases with the injected current. A simple test consists in enforcing different levels of stiffness and measuring the number of interpolation steps of both methods:

{: .picture}
<img width="60.3%" height="60.3%" src="{{ site.assets }}/neuron-variable-timestep/cvode_pulse_currents_table.svg">

The slow dynamics of the fixed step method, which end up accounting for the events at discrete intervals instead of at the event delivery times of its variable step counterpart, lead to a shift in the voltage trajectory that propagates over time (Backward Euler with $$\Delta t=1 \mu s$$ and CVODE with $$\text{atol}=10^{-4}$$ are taken as reference solutions):

{: .picture}
<img width="93.6%" height="93.6%" src="{{ site.assets }}/neuron-variable-timestep/l5_neuron_pulse1_3ma_100ms_cvode_vs_backward_euler.svg">

After a second of simulation, the spike time phase-shifting is substantial (about 2 ms for $$\Delta t = 5 \mu s$$ and 6.5 ms for $$\Delta t = 25 \mu s$$), demonstrating the gain in using variable step methods for this use case.

{: .picture}
<img width="95.2%" height="95.2%" src="{{ site.assets }}/neuron-variable-timestep/l5_neuron_pulse_1000ms_results.svg">

To study whether these advantages hold in practice, the simulation runtime of both methods is measured. The first test measures the impact of the stiffness in terms of simulation steps and time to solution on an Intel i5 at 1.6 GHz. The different current values are injected as a proportion of the *action potential threshold* current, i.e. the minimum current that must be continuously injected for the neuron to spike.

{: .picture}
<img width="98%" height="98%" src="{{ site.assets }}/neuron-variable-timestep/cvode_dependency_on_variation_solution.svg">

Results look promising: even at an extremely high current value, the speed-up is significant. A second test enforces discontinuities and resets of the IVP by artificially injecting current pulses of $$1 \mu s$$ at given frequencies.

{: .picture}
<img width="97.9%" height="97.9%" src="{{ site.assets }}/neuron-variable-timestep/cvode_dependency_on_events_arrival.svg">

In this scenario, the runtime shows a high dependency on the number of discontinuities, and a minor dependency on the discontinuity value (i.e. the amount of current injected): CVODE is faster than Backward Euler up to about 1,000 pulses per second for the strongest pulse, and 1,600 for the weakest. The holy grail is then: how does this apply to networks of neurons?

## Simulation of a network of homogeneous neurons

The performance of *lvardt*'s integration model was tested ([Lytton and Hines 2005][lytton2005independent]) on a connected homogeneous network of 100 inhibitory neurons on a single compute node. Due to their inhibitory features, the circuit presents rapid synchronization through mutual inhibition, where variable time step integrators are likely to encounter problems. A fixed time step of $$0.0025 ms$$ yielded *closely comparable* results to those of *lvardt* with an absolute error tolerance (atol) of $$10^{-3}$$ or $$10^{-5}$$. The event deliveries and the associated interpolations took about $$12\%$$ of the total execution time. When using global variable time stepping instead, the authors report that the large number of events leads to an increase in execution time of up to 60-fold. Another experiment featured a thalamocortical simulation with four cell types (cortical pyramidal neurons and interneurons, thalamocortical
cells, and thalamic reticular neurons). The two thalamic cell types produced bursts with prolonged interburst intervals, which makes them very suitable for the *lvardt* algorithm. A simulation of $$150 ms$$ at high precision, with an error tolerance (atol) of $$10^{-6}$$ for *lvardt* and $$\Delta t = 10^{-4} ms$$ for the fixed time step, took 6 minutes 13 seconds and 10 hours 20 minutes, respectively, a 100-fold speed-up. For a standard-precision run, with an atol of $$10^{-3}$$ and $$\Delta t = 10^{-2} ms$$, the simulations took 2 minutes for *lvardt* and 5 minutes 53 seconds for the fixed time step, a 3-fold speed-up.

The following picture presents the aforementioned results for the comparison of fixed and variable time step methods for the mutual-inhibition model. Execution time grows linearly with simulation time for the fixed-step integration (dashed lines). Variable-step methods show a reduction in CPU load at the onset of synchrony. The rationale behind the non-linearity of adaptive time stepping is that *lvardt* produces time steps that jump around during the initial presynchrony phase of the simulation (fast uprising phase), and then slowly stabilizes to an alternation between large steps in the long intervals separating the population spikes and extremely short time steps during the population spike itself:

{: .picture}
<img width="31.1%" height="31.1%" src="{{ site.assets }}/neuron-variable-timestep/lvardt_exec_time_comparison.svg">

{: .caption}
Adapted from: [Lytton and Hines (2005), Fig. 3A](https://pmc.ncbi.nlm.nih.gov/articles/PMC2712447/)

## Simulation of a network of heterogeneous neurons

[markram2015cell]: https://www.cell.com/fulltext/S0092-8674(15)01191-5

Although networks of similarly behaving neurons give an *estimation* of the achievable speed-up when using fixed and variable time step interpolations, it is a rough one. In practice, brain activity differs widely across space and time.

To measure the efficiency across the different spiking dynamics of neurons, the digital reconstruction of a [laboratory experiment that the Blue Brain Project published in the journal Cell][markram2015cell] was reproduced. The reproduction simulates 7.5 seconds of electrical activity, performing a fixed-step simulation of the spontaneous activity of 219,247 neurons during tonic depolarization. For the biology enthusiasts: the network exhibits spontaneous slow oscillatory population bursts, initiated in Layer 5, spreading down to L6, and then up to L4 and L2/3 with secondary bursts spreading back to L6. For further details, refer to the section *Simulating Spontaneous Activity* of that paper.
 
Two sample neurons are taken from the top 1% *busiest* neurons (in terms of incoming spikes, i.e. discontinuities), to visually analyse the number of spikes arriving per time interval (left, each time bin accounts for 10 ms) and the distribution of time intervals between spike arrivals (right):

{: .picture}
<img width="59.3%" height="59.3%" src="{{ site.assets }}/neuron-variable-timestep/neuron_203076.svg">
<img width="59.5%" height="59.5%" src="{{ site.assets }}/neuron-variable-timestep/neuron_138083.svg">

The number of spikes received in that top 1% was between 3040 and 6146 events. The same analysis is redone for two sample neurons collected from the median 1% of neurons in terms of spike arrivals:

{: .picture}
<img width="58.3%" height="58.3%" src="{{ site.assets }}/neuron-variable-timestep/neuron_208223.svg">
<img width="58.1%" height="58.1%" src="{{ site.assets }}/neuron-variable-timestep/neuron_87784.svg">

The neurons of the median 1% received 541 to 558 spiking events. And the bottom 1%:

{: .picture}
<img width="57.6%" height="57.6%" src="{{ site.assets }}/neuron-variable-timestep/neuron_132412.svg">
<img width="58%" height="58%" src="{{ site.assets }}/neuron-variable-timestep/neuron_2101.svg">

with fewer than 100 events.

The total number of events received across all neurons in the network was approximately 155 million, i.e. a mean of 707 incoming spikes per neuron, or 94 per second. In practice, this means that most brain regions (or activity patterns) fall within the interval of acceleration described in the previous section.

Is the theoretical speed-up guaranteed to hold in practice? As a saying often attributed to Einstein goes, *"In theory, there's no difference between theory and practice, but in practice, there is"*. And that is exactly right in this case, where two main issues arise.

### 1. Distributed CVODE executions

Simulations of such scale require parallel/distributed computing due to the memory/computation requirements.

However, in parallel computing environments, the synchronization of cells spread across different compute nodes presents a bottleneck, due to the required step synchronization between neurons. In practice, neurons held on different compute nodes have to be able to *communicate* their time instant, so that when they spike, the synaptic current is delivered to a second neuron that has not advanced beyond the delivery time (or the spike would be *missed*).

To tackle this issue, the simulation is split into equidistant time intervals equal to the shortest delay of any global event (typically the minimal synaptic delay across all synapses, or, when applicable, the update frequency of extra-cellular mechanisms or gap junctions). At every time interval, all the nodes in the network perform a collective communication call and share the events (spikes) to be sent in the next time frame. This approach, the [bulk synchronous parallel](https://en.wikipedia.org/wiki/Bulk_synchronous_parallel) (BSP) execution model, has previously been shown ([Lytton and Hines 2005][lytton2005independent]) to provide a good speed-up. It then guarantees:
1. Asynchronous computation: every core in every compute node computes its dataset (neurons) independently, and advances the neurons up to the time limit of the next synchronization interval;
2. Asynchronous communication: synaptic activity can be transmitted immediately after it occurs, or together with the synchronization information in an **all-to-all communication**;
3. Synchronous communication: neuron interpolation and spike communication for a stepping time window must be completed before the computation and communication for the next window start.
 
However, going to the *extreme* of the possible optimisation, this **collective synchronization of neurons** has three main downsides: (a) no compute node can ever perform a CVODE timestep that exceeds the limits of the current time frame, (b) all compute nodes must wait for the slowest one to reach the end of the time frame, and (c) all compute nodes restart CVODE at the initial instant of every time frame. A solution to this problem is the **fully-asynchronous parallel** (FAP) execution model, an execution model that guarantees asynchronous communication, computation and synchronization. Detailed information on the implementation (and its added cache-efficiency benefits) can be found <a href="{{ site.assets }}/neuron-variable-timestep/neurox-cache-efficient.pdf">in the original publication</a>, but the main concept is:
- Each neuron keeps a map of all its pre-synaptic dependencies, i.e. those neurons whose axons connect to its synapses, or in practice, those which may spike and deliver current into the neuron;
  - This map is actively updated by small stepping notifications from the pre-synaptic dependencies;
- Each neuron keeps a map of all its post-synaptic dependencies, i.e. those whose dendrites connect to its axon terminal, or in practice, those which receive current from the neuron when it spikes;
  - In addition to the spike communication, at a regular interval, the neuron must inform its dependencies that it *stepped*, or advanced in time;
- Collective synchronization barriers are removed. Instead, neurons advance in time as much as possible, taking into account their pre-synaptic dependencies, i.e. they advance up to the timestep that guarantees no missed event if any pre-synaptic dependency spikes.

This approach increases computational efficiency, as neurons can be stepped beyond the synchronization intervals of the other approaches, and perform more interpolation steps in a row, staying longer in the more efficient CPU cache levels.

### 2. Testing conditions alter spike rates

[LIF-model-post]: {{ site.baseurl }}{% post_url 2016-02-08-lif-brunel %}

Even though results sound promising, different mammals, brain regions and mental states can easily change the simulation conditions, leading to very different spiking rates. So the final test is to understand how these factors alter the simulation efficiency. To that end, the fully-asynchronous execution model was applied and tested on five different brain dynamics:
- a model of *quiet dynamics* with a mean spiking rate of 0.25 Hz per neuron, representing neurons almost at rest and/or with little activity. This model provides an upper bound of the runtime of circa 90% of neurons in the brain during regular activity;
- a model of *slow dynamics* at 1.5 Hz, representing the lower bound of active neurons, described next;
- a model of *moderate dynamics* with a spiking rate of 6.5 Hz, an approximation of the irregular regime of slow oscillations displayed by the [Brunel network][LIF-model-post]; also an upper limit to the spiking rate of the rat frontal cortex; and a loose approximation of the visual cortex of cats;
- a model of *fast dynamics* of 38 Hz, approximating the cortical and thalamic neuronal activity during periods of high vigilance, and the asynchronous irregular regime of the theoretical inhibition-dominated model of the Brunel network; and 
- a model of *burst dynamics* at 55.8 Hz, typically a by-product of depolarizing current injections, representative of the fast-spiking regime of the inhibition-dominated irregular dynamics of the Brunel network.

The comparison tested several fixed- and variable-step interpolation methods, whose details are available <a href="{{ site.assets }}/neuron-variable-timestep/neurox-vardt-arxiv.pdf">in the original publication preprint</a>. The benchmark results are displayed next:

{: .picture}
<img width="87.6%" height="87.6%" src="{{ site.assets }}/neuron-variable-timestep/benchmark_two_rows_2.svg">

Quoting the previous paper: "Fixed step methods do not yield significantly-different execution times across different spiking regimes. This is due to the homogeneous computation of neuron  state updates throughout time, and the light computation attached to synaptic events and collective communication not yielding a substantial increase of runtime. The difference in execution times measured across the five regimes was of about $$2\%$$, which we consider negligible. On the other hand, as expected, variable step executions are penalized on regimes with high discontinuity rates. It is noticeable that the runtimes of fixed- and variable-step solvers approximate as we increase the spiking rate, i.e. the increase of runtimes with the input size is steeper for variable timestep (2b$$\hspace{0.3mm}{\color{#b85450}\bullet}$$ and  2c$$\hspace{0.3mm}{\color{black}\bullet}{\color{#666666}\bullet}{\color{#b3b3b3}\bullet}$$) compared to fixed timestep methods (2a$$\hspace{0.3mm}{\color{#6c8ebf}\bullet}$$). This is due to discontinuities in variable-step being delivered throughout a continuous time line, compared to the discrete delivery instants of the fixed-step methods --- therefore increase the number of interpolation steps; and the iterative model of the variable timestep reinitializing the state computation with small step sizes on each IVP reset, compared to the constant-sized step of fixed step methods.  A remarkable performance is visible on the quiet dynamics use case, where our fully-implicit ODE solver of complex models (with Newton iterations), still runs faster than the simple solver resolving only a system of linear equations. The underlying rationale is that --- despite the inherent computation cost of Newton iterations in the variable step methods --- the low level of discontinuities allow for very long steps, that surpass the simulation throughput of simple solvers running on fixed step methods." 

For further details, including detailed comparisons of the performance of *Variable Step Event Grouping*, *Fully-Asynchronous vs Bulk-Synchronous Execution Models*, *Runtime Dependency on Input Size and Spike Activity*, and the *Overall Runtime Speed-up Estimation* of 228.5-24.6x (based on the distribution of neurons per spiking regime in the simulation of the laboratory experiment above), refer to the <a href="{{ site.assets }}/neuron-variable-timestep/neurox-vardt-arxiv.pdf">article preprint</a>.

