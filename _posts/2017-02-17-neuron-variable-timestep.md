---
layout: post
title:  "Variable Timestep Simulation of the Electrical Activity of Neurons"
categories: ["Computational Neuroscience"]
tags: ["CVODE", "variable timestep", "numerical methods", "simulation", "HPC"]
---

[neuron-methods-post]: {{ site.baseurl }}{% post_url 2016-08-15-neuron-electrical-activity %}

The [previous post][neuron-methods-post] covered the mathematical background of neuron activity and the simulation of the electrical activity of simple neuron models. It showed that an interleaved voltage-current resolution of linear, unrelated state ODEs allows the problem to be solved as a system of linear equations. However, such a resolution is only possible for *simple* models. Neuron models with non-linear state variable ODEs and correlation between states require a fully-implicit resolution, which is possible with Backward Euler with Newton iterations. This is a common use case. It underlies the resolution of extra-cellular mechanisms, capacitors between nodes, and other mechanisms that couple the evolution of neurons over time.

A fully-implicit method commonly used for this problem is the Backward (Implicit) Euler method:

$$
y_{n+1} - y_n = h f (t_{n+1}, y_{n+1})
$$

where $$h$$ is the timestep, $$y$$ is the array of states and $$f$$ is the right-hand side function of the ODE. This post presents a more efficient alternative: a fully-implicit method based on variable-order variable timestepping.

## CVODE for fully-implicit adaptive-order adaptive-step interpolation
 
[cvode]: https://aip.scitation.org/doi/pdf/10.1063/1.4822377
[sundials]: https://computation.llnl.gov/projects/sundials

The main argument for adaptive time stepping is its efficiency: it takes large steps during long periods of slowly varying states, e.g. interspike intervals. NEURON's adaptive algorithm, [CVODE][cvode], uses separate variable-step integrators for individual neurons in the network. NEURON uses CVODE as an interface to a collection of methods built on top of the existing CVODE and IDA (Integrator for Differential-Algebraic problems) integrators of the [SUNDIALS package][sundials], a SUite of Nonlinear and DIfferential/ALgebraic Equation Solvers. NEURON employs the Backward Differentiation Formula (BDF), a family of implicit multi-step methods for the numerical integration of (stiff) ordinary differential equations. For a given function and time, BDF approximates the derivative using information from previously computed times, which increases the accuracy of the approximation. BDF methods are implicit, so they require solving nonlinear equations at every step. The BDF multistep method is suitable for stiff problems, such as those that describe neuronal activity. A stiff equation includes terms that may lead to rapid variation of the solution. This makes certain numerical methods unstable unless the step size is *extremely* small. CVODE is a solver for the Initial Value Problem (IVP) --- an ODE together with an initial condition of the unknown function at a given point in the domain of the solution --- that can be written abstractly as:

$$
\dot{y} = f(t,y) \text{, } y(t_0) = y_0 \text{ where } y \in \mathbb{R}^N
$$

where $$\dot{y}$$ denotes $$\frac{dy}{dt}$$. The derivative of the solution $$y$$ is computed from past (backward) values, i.e. $$\dot{y}_n$$ is a linear combination of $$y_n$$, $$y_{n-1}$$, $$y_{n-2}, ...$$. Here, $$y$$ is the array of all state variables: the voltages $$V_n$$ of all compartments (governed by $$dV_n/dt$$) and the states $$x_i$$ (governed by $$dx_i/dt$$), where $$x_i$$ is a given state variable of an ion $$i$$. The variable-step BDF of order $$q$$ can be written as:

$$
y_n = \sum_{i=1}^{q} \alpha_{n,i} y_{n-i} + h_n \beta _{n,0} \dot{y}_n
$$

where $$y_n$$ are the computed approximations of $$y (t_n)$$, $$h_n = t_n - t_{n-1}$$ is the step size, $$q$$ is the order (number of previous steps used by the BDF), and $$\alpha$$ and $$\beta$$ are the unique BDF coefficients, which depend on the method order. For each order, the method is:

- BDF-1: $$ y_{n+1} - y_n = h f (t_{n+1}, y_{n+1})$$, i.e. the Backward Euler;
- BDF-2: $$ 3y_{n+2} - 4y_{n+1} + y_n = 2 h f (t_{n+2}, y_{n+2})$$;
- BDF-3: $$ 11y_{n+3} - 18y_{n+2} + 9y_{n+1} - 2y_n = 6 h f (t_{n+3}, y_{n+3})$$;
- BDF-4: $$ 25y_{n+4} - 48y_{n+3} + 36y_{n+2} - 16y_{n+1} + 3y_n = 12 h f (t_{n+4}, y_{n+4})$$;
- BDF-5: $$ 137y_{n+5} - 300y_{n+4} + 300y_{n+3} - 200y_{n+2} + 75y_{n+1} - 12y_n = 60 h f (t_{n+5}, y_{n+5})$$;
- BDF-6: $$ 147y_{n+6} - 360y_{n+5} + 450y_{n+4} - 400y_{n+3} + 225y_{n+2} - 72y_{n+1} + 10y_n = 60 h f(t_{n+6}, y_{n+6})$$;

BDF with order $$q > 6$$ is not zero-stable and therefore cannot be used. Zero-stability means that, if a perturbation $$\epsilon$$ is applied to an initial value, the difference between the perturbed and unperturbed results is no bigger than $$k \epsilon$$, for some constant $$k \in \mathbb{R}$$. CVODE allows a maximum order of 5.

Denoting the iterates by $$y_{n,m}$$, the functional iteration of the BDF is given by:

$$
y_{n,m+1} = h_n \beta _{n,0} f ( t_n, y_{n,m} ) + a_n
\label{eq_newton_method_derivation}
$$

As with any implicit method, every step must (approximately) solve the nonlinear system:

$$
G(y_n) \equiv y_n - h_n \beta_{n,0} f(t_n, y_n) - a_n = 0 \text{ , where } a_n \equiv \sum_{i=1}^{q} \alpha_{n,i} y_{n-i}  
\label{eq_variable_time_step_bdf}
$$

CVODE solves this system with Newton iterations. Newton's method finds successively better approximations to the roots (or zeroes) of a function. For one variable, it iterates $$x_{n+1} = x_n - f(x_n) / f'(x_n)$$ until a sufficiently accurate value is reached. In practice, it requires solving the linear systems:

$$
M [y_{n(m+1)} - y_{n(m)}] = - G(y_{n(m)})
$$

where 

$$
M \approx I - \gamma J \text{ , } J = \partial f / \partial y \text{ , and } \gamma = h_n \beta_{n,0}
$$

For the Newton corrections, CVODE provides the following options: (a) a direct dense solver, (b) a direct banded solver, (c) a diagonal approximate Jacobian solver, or the SPGMR (Scaled Preconditioned GMRES) method. A preconditioner matrix is a transformation that conditions a problem into a form more suitable for numerical solvers. NEURON applies the SPGMR method, which uses a matrix $$M$$ (or a preconditioner matrix $$P$$, in the SPGMR case) and performs an Inexact Newton iteration. In this iteration, $$M$$ is applied in a matrix-free manner, and the matrix-vector products $$Jv$$ are obtained either by difference quotients or by a user-supplied routine. In NEURON, $$Jv$$ is provided by the *SOLVE* block in the mechanisms. In practice, SPGMR transforms the nonlinear problem into a linear one by introducing a multiplication by the Jacobian (in vector calculus, the Jacobian matrix is the matrix of all first-order partial derivatives of a vector-valued function). It computes the solution of the equation:

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

The previous equation requires a division by $$M$$, the Jacobian of $$G$$, i.e. a multiplication by its inverse. Since computing the inverse of the Jacobian is expensive, CVODE allows a trade-off between accuracy and solution time by only requiring a solver for $$P y = b$$. For the linear case, NEURON supports a resolution with the Hines solver (default) or with the (a) or (c) methods mentioned previously, but the non-Hines approaches have been little explored.

## Error Handling

One main difference between fixed and variable time stepping is that the user does not specify the time step, but the relative (*rtol*) and absolute (*atol*) local error tolerances instead. The solver then adjusts $$\Delta t$$ based on these two values. The local error is based on the extrapolation process within a time step. CVODE uses a weighted root-mean-square norm for all error-like quantities:

$$
|| v || = \sqrt{ \frac{1}{N} \sum_{i=1}^N (v_i / W_i)^2}
$$

where $$v_i$$ are the deviations, and the weights $$W_i$$ are based on the current solution (with components $$y^i$$), the user-specified absolute tolerances $$atol_i$$ and the relative tolerance $$rtol$$:

$$
W_i =  rtol \cdot | y^i | + atol_i
$$

If $$y_n$$ is the exact solution of equation \ref{eq_variable_time_step_bdf}, the iteration error $$y_n - y_{n(m)}$$ must be small enough, such that $$\| y_n - y_{n(m)} \| < 0.1 \epsilon$$, where $$\epsilon$$ is the tolerance of the local error test in the weighted norm above (and $$0.1$$ is user-changeable). For this, the solver also estimates the linear convergence rate constant $$R$$, initialized at $$1$$ and reset to $$1$$ when $$M$$ or $$P$$ is updated. After computing a correction $$\delta_m = y_{n(m)} - y_{n(m-1)}$$, $$R$$ is updated (if $$m > 1$$) as:

$$
R \leftarrow \text{max} \left\{ 0.3 R, \frac{|| \delta_m ||}{||\delta_{m-1} ||} \right\}
$$

This rate is used to estimate

$$
||y_n - y_{n(m)}|| \approx || y_{n(m+1)} - y_{n(m)}|| \approx R || y_{n(m)} - y_{n(m-1)}|| = R || \delta_m ||
$$

Thus, the convergence test is 

$$
R || \delta_m || < 0.1 \epsilon
$$

## Local and Global Time Stepping

NEURON's default error setting for CVODE is $$10 \mu V$$ for membrane potential and $$0.1 nM$$ for internal free calcium concentration. This gives an HH-based action potential an accuracy equivalent to that of a fixed time step of $$\Delta t = 25 \mu s$$. The default maximum order for BDF is $$5$$, although *it is recommended* to let CVODE decide the best order.

[lytton2005independent]: https://www.mitpressjournals.org/doi/abs/10.1162/0899766053429453
[neuron-book]: https://www.cambridge.org/core/books/neuron-book/7C8D9BD861D288E658BEB652F593F273

Large networks of cells cannot be integrated with global adaptive time stepping. A high frequency of events such as gating channels or synapses causes discontinuities in parameters or state variables. Each discontinuity requires a reinitialization of the computation and forces the integrator to restart with a new initial value problem. NEURON's workaround is a local variable time stepping scheme called *lvardt* ([Lytton and Hines, Independent Variable Time-Step Integration of Individual Neurons for Network Simulations, MIT Press][lytton2005independent]). It creates one CVODE integrator per cell, and aims at short time step integration for active neurons and long time step integration for inactive neurons, while keeping the integration accuracy consistent. All cells are kept in an ordered queue, where their position is given by their event delivery time or their current simulation time. Since the system advances in time with multiple, independent integrators, the overall computation is reduced compared to the global counterpart. The main issue is the coordination of the time advancement of each cell: all the state variables of a given cell must be correctly available to any incoming event. To solve this, every cell holds the information of all states between its current time instant and the previous time steps. This allows backward interpolation when required. The integration coordinator also ensures that the interpolation intervals of connected cells overlap, so that no event between two cells is missed. The critical point of this approach is that the time step is tentative: an event received by the cell in the interim may reinitialize the cell to a time prior to the tentatively calculated threshold time.

The following picture illustrates a timeline (from A to D) of six cells running the *lvardt* algorithm. The intervals of possible interpolation are displayed as black rectangles. The right end of a cell's rectangle, $$t_b$$, is the time up to which that cell has integrated. The vertical line traversing all neurons marks the system's earliest event time. Triggers of discrete events on a neuron are displayed as short thick vertical bars. (A) The cell with the smallest $$t_b$$ advances by the length of the hashed rectangle; (B) Cell 5 has the smallest $$t_b$$ and integrates forward; (C) The next earliest event in the system is at neuron 3. This event trigger creates three new events to be delivered to cells 3, 4 and 5; (D) The new earliest event forces cell 3 to back-interpolate, the event is handled, and cell 3 reinitializes. Cell 3 will be the next cell to integrate forward:

{: style="text-align:center; font-size: small;"}
<img width="30%" height="30%" src="{{ site.assets }}/neuron-variable-timestep/local_variable_timestep_six_cell_example.svg">

{: style="text-align:center; font-size: small;"}
Adapted from: [The NEURON book][neuron-book]

## Single Neuron Performance

The following picture shows the difference in performance of the Backward Euler and Variable Timestep interpolators during a stiff trajectory of an Action Potential.

{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="{{ site.assets }}/neuron-variable-timestep/l5_neuron_soma_voltage_per_step_6ms.svg">

Note the difference in interpolation instants: equidistant in the Backward Euler case, and adapted to the solution gradient in the variable step case. A current injection at the soma forces the neuron to spike at a rate that increases with the injected current. A simple test enforces different levels of stiffness and measures the number of interpolation steps of both methods:

{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="{{ site.assets }}/neuron-variable-timestep/cvode_pulse_currents_table.svg">

The slow dynamics of the fixed step method --- which accounts for events at discrete intervals instead of at the event delivery times of its variable step counterpart --- lead to a shift in the voltage trajectory that propagates over time (Euler with $$\Delta t=1 \mu s$$ and CVODE with $$\text{atol}=10^{-4}$$ serve as reference solutions):

{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="{{ site.assets }}/neuron-variable-timestep/l5_neuron_pulse1_3ma_100ms_cvode_vs_backward_euler.svg">

After one second of simulation, the spike time phase shift is substantial, which demonstrates the benefit of variable step methods for this use case.

{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="{{ site.assets }}/neuron-variable-timestep/l5_neuron_pulse_1000ms_results.svg">

To study whether these advantages hold in practice, the simulation runtime of both methods was measured. The first test measures the impact of stiffness on the number of simulation steps and on the time to solution, on an Intel i5 at 2.6 GHz. The different current values are injected as a proportion of the *action potential threshold* current, i.e. the minimum current that must be continuously injected for the neuron to spike.

{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="{{ site.assets }}/neuron-variable-timestep/cvode_dependency_on_variation_solution.svg">

The results are promising: even at an extremely high current value, the speed-up is significant. A second test enforces discontinuities and resets of the IVP by artificially injecting current pulses of $$1 \mu s$$ at given frequencies.

{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="{{ site.assets }}/neuron-variable-timestep/cvode_dependency_on_events_arrival.svg">

In this scenario, the runtime depends strongly on the number of discontinuities, and only slightly on the discontinuity value (i.e. the amount of current injected). The key question is then how these results apply to networks of neurons.

## Simulation of a network of homogeneous neurons

The performance of *lvardt*'s integration model was tested ([Lytton and Hines 2005][lytton2005independent]) on a connected homogeneous network of 100 inhibitory neurons on a single compute node. Because the neurons are inhibitory, the circuit shows rapid synchronization through mutual inhibition, a case where variable time step integrators are likely to encounter problems. A fixed time step of $$0.0025 ms$$ yielded *closely comparable* results to those of *lvardt* with an absolute error tolerance (atol) of $$10^{-3}$$ or $$10^{-5}$$. The event deliveries and the associated interpolations took about $$12\%$$ of the total execution time. With global variable time stepping instead, the authors report that the large number of events leads to a roughly 60-fold increase in execution time. Another experiment featured a thalamocortical simulation with four cell types (cortical pyramidal neurons and interneurons, thalamocortical
cells, and thalamic reticular neurons). The two thalamic cell types produced bursts with long interburst intervals, which makes them very suitable for the *lvardt* algorithm. A simulation of $$150 ms$$ at high precision, with an error tolerance (atol) of $$10^{-6}$$ for *lvardt* and $$\Delta t = 10^{-4} ms$$ for the fixed time step, took 6 minutes 13 seconds and 10 hours 20 minutes, respectively: a 100-fold speed-up. For a standard-precision run, with an atol of $$10^{-3}$$ and $$\Delta t = 10^{-2} ms$$, the simulations took 2 minutes for *lvardt* and 5 minutes 53 seconds for the fixed time step: a 3-fold speed-up.

The following picture shows these results for the comparison of fixed and variable time step methods on the mutual-inhibition model. Execution time grows linearly with simulation time for the fixed-step integration (dashed lines). Variable-step methods show a reduction in CPU load at the onset of synchrony. The non-linearity of adaptive time stepping has the following cause: *lvardt* produces time steps that jump around during the initial presynchrony phase of the simulation (fast uprising phase), and then slowly stabilizes to an alternation between large steps in the long intervals separating the population spikes and extremely short time steps during the population spike itself:

{: style="text-align:center; font-size: small;"}
<img width="30%" height="30%" src="{{ site.assets }}/neuron-variable-timestep/lvardt_exec_time_comparison.svg">

{: style="text-align:center; font-size: small;"}
source: Lytton et al. 2005 

## Simulation of a network of heterogeneous neurons

[markram2015cell]: https://www.cell.com/fulltext/S0092-8674(15)01191-5

Networks of similarly behaving neurons give only a rough *estimation* of the speed-up achievable with fixed and variable time step interpolations. In practice, brain activity differs widely across space and time.

To measure the efficiency across the different spiking dynamics of neurons, the digital reconstruction of a [laboratory experiment that the Blue Brain Project published in the journal Cell][markram2015cell] was reproduced. The reproduction is a fixed-step simulation of 7.5 seconds of the spontaneous electrical activity of 219,247 neurons during tonic depolarization. For readers interested in the biology: the network exhibits spontaneous slow oscillatory population bursts, initiated in Layer 5, spreading down to L6, and then up to L4 and L2/3 with secondary bursts spreading back to L6. For further details, refer to the section *Simulating Spontaneous Activity* of that paper.
 
Two sample neurons are taken from the top 1% *busiest* neurons (in terms of incoming spikes, i.e. discontinuities). The plots show the number of spikes arriving per time interval (left, each time bin accounts for 10ms) and the distribution of time intervals between spike arrivals (right):

{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="{{ site.assets }}/neuron-variable-timestep/neuron_203076.svg">
<img width="60%" height="60%" src="{{ site.assets }}/neuron-variable-timestep/neuron_138083.svg">

The neurons in that top 1% received between 3040 and 6146 events. The same analysis for two sample neurons from the median 1% of neurons in terms of spike arrivals:

{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="{{ site.assets }}/neuron-variable-timestep/neuron_208223.svg">
<img width="60%" height="60%" src="{{ site.assets }}/neuron-variable-timestep/neuron_87784.svg">

These received 541 to 558 spiking events. Finally, the neurons in the bottom 1%

{: style="text-align:center; font-size: small;"}
<img width="60%" height="60%" src="{{ site.assets }}/neuron-variable-timestep/neuron_132412.svg">
<img width="60%" height="60%" src="{{ site.assets }}/neuron-variable-timestep/neuron_2101.svg">

received fewer than 100 events.

The total number of events received across all neurons in the network was approximately 155 million, i.e. a mean of 707 incoming spikes per neuron. In practice, this means that most brain regions (or activity patterns) fall within the interval of acceleration described in the previous section.

Is the theoretical speed-up guaranteed to hold in practice? A saying often attributed to Einstein states: *"In theory, there's no difference between theory and practice, but in practice, there is"*. This is the case here, where two main issues arise.

### 1. Distributed CVODE executions

Simulations of this scale require parallel/distributed computing due to their memory and computation requirements.

However, in parallel computing environments, the step synchronization of neurons spread across different compute nodes is a bottleneck. Neurons held on different compute nodes must *communicate* their time instant. When a neuron spikes, the synaptic current must be delivered to a second neuron that has not advanced beyond the delivery time, otherwise the spike is *missed*.

To address this, the simulation is split into equidistant time intervals equal to the shortest delay of any global event --- typically the minimal synaptic delay across all synapses, or (when applicable) the update frequency of extra-cellular mechanisms or gap junctions. At every time interval, all nodes in the network perform a collective communication call and share the events (spikes) to be sent in the next time frame. This approach has previously been shown ([Lytton and Hines 2005][lytton2005independent]) to provide a good speed-up. It guarantees:
1. Asynchronous computation: every core in every compute node computes its dataset (neurons) independently, and advances the neurons up to the time limit of the next synchronization interval;
2. Asynchronous communication: synaptic activity can be transmitted immediately after it occurs, or together with the synchronization information in an **all-to-all communication**;
3. Synchronous communication: neuron interpolation and spike communication for a stepping time window must be completed before the computation and communication for the next window start.
 
However, at the *extreme* of the possible optimisation, this **collective synchronization of neurons** has three main downsides: (a) no compute node can perform a CVODE timestep that exceeds the limits of the current time frame, (b) all compute nodes must wait for the slowest one to reach the end of the time frame, and (c) all compute nodes restart CVODE at the initial instant of every time frame. A solution to this problem is the **fully-asynchronous execution model**, which guarantees asynchronous communication, computation and synchronization. Detailed information on the implementation (and its added cache-efficiency benefits) is available <a href="{{ site.assets }}/neuron-variable-timestep/neurox-cache-efficient.pdf">in the original publication</a>. The main concept is:
- Each neuron keeps a map of all its pre-synaptic dependencies, i.e. the neurons whose axons connect to its synapses or, in practice, those that may spike and deliver current into the neuron;
  - This map is actively updated by small stepping notifications from the pre-synaptic dependencies;
- Each neuron keeps a map of all its post-synaptic dependencies, i.e. the neurons whose dendrites connect to its axon terminal or, in practice, those that receive current from the neuron when it spikes;
  - In addition to spike communication, the neuron must regularly inform its dependencies that it *stepped*, i.e. advanced in time;
- Collective synchronization barriers are removed. Instead, neurons advance in time as far as possible given their pre-synaptic dependencies, i.e. up to the timestep that guarantees no missed event if any pre-synaptic dependency spikes;

This approach increases computational efficiency: neurons can step beyond the synchronization intervals of the other approaches and perform more interpolation steps in a row, staying longer in the more efficient CPU cache levels.

### 2. Testing conditions alter spike rates

[LIF-model-post]: {{ site.baseurl }}{% post_url 2016-02-08-lif-brunel %}

Even though these results are promising, different mammals, brain regions and mental states can change the simulation conditions and lead to very different spiking rates. The final test therefore measures how these factors alter the simulation efficiency. To that end, the fully-asynchronous execution model was applied to five different brain dynamics:
- a model of *quiet dynamics* with a mean spiking rate of 0.25 Hz per neuron, representing neurons almost at rest and/or with little activity. This model gives an upper bound of the runtime of about 90% of neurons in the brain during regular activity;
- a model of *slow dynamics* at 1.5Hz, representing the lower bound of active neurons, described next;
- a model of *moderate dynamics* with a spiking rate of 6.5 Hz, an approximation of the irregular regime of slow oscillations displayed by the [Brunel network][LIF-model-post]; also an upper limit to the spiking rate of the rat frontal cortex; and a loose approximation of the visual cortex of cats;
- a model of *fast dynamics* of 38 Hz, approximating the cortical and thalamic neuronal activity during periods of high vigilance, and the regular regime of the theoretical inhibition-dominated model of the [Brunel network][LIF-model-post]; and 
- a model of *burst dynamics* at 55.8 Hz, typically a by-product of depolarizing current injections, representative of the fast-spiking regime of the inhibition-dominated irregular dynamics of the [Brunel network][LIF-model-post].

The comparison included several fixed- and variable-step interpolation methods, detailed <a href="{{ site.assets }}/neuron-variable-timestep/neurox-vardt-arxiv.pdf">in the original publication preprint</a>. The benchmark results are:

{: style="text-align:center; font-size: small;"}
<img width="90%" height="90%" src="{{ site.assets }}/neuron-variable-timestep/benchmark_two_rows_2.svg">

Quoting the previous paper: "Fixed step methods do not yield significantly-different execution times across different spiking regimes. This is due to the homogeneous computation of neuron  state updates throughout time, and the light computation attached to synaptic events and collective communication not yielding a substantial increase of runtime. The difference in execution times measured across the five regimes was of about $$2\%$$, which we consider negligible. On the other hand, as expected, variable step executions are penalized on regimes with high discontinuity rates. It is noticeable that the runtimes of fixed- and variable-step solvers approximate as we increase the spiking rate, i.e. the increase of runtimes with the input size is steeper for variable timestep (2b$$\hspace{0.3mm}{\color{#b85450}\bullet}$$ and  2c$$\hspace{0.3mm}{\color{black}\bullet}{\color{#666666}\bullet}{\color{#b3b3b3}\bullet}$$) compared to fixed timestep methods (2a$$\hspace{0.3mm}{\color{#6c8ebf}\bullet}$$). This is due to discontinuities in variable-step being delivered throughout a continuous time line, compared to the discrete delivery instants of the fixed-step methods --- therefore increase the number of interpolation steps; and the iterative model of the variable timestep reinitializing the state computation with small step sizes on each IVP reset, compared to the constant-sized step of fixed step methods.  A remarkable performance is visible on the quiet dynamics use case, where our fully-implicit ODE solver of complex models (with Newton iterations), still runs faster than the simple solver resolving only a system of linear equations. The underlying rationale is that --- despite the inherent computation cost of Newton iterations in the variable step methods --- the low level of discontinuities allow for very long steps, that surpass the simulation throughput of simple solvers running on fixed step methods." 

For further details, including detailed comparisons of the performance of *Variable Step Event Grouping*, *Fully-Asynchronous vs Bulk-Synchronous Execution Models*, *Runtime Dependency on Input Size and Spike Activity*, and the *Overall Runtime Speed-up Estimation* of 228.5-24.6x (based on the distribution of neurons per spiking regime in the brain), refer to the <a href="{{ site.assets }}/neuron-variable-timestep/neurox-vardt-arxiv.pdf">article preprint</a>.

