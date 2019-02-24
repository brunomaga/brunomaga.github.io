---
layout: post
title:  "Variable Timestep Resolution of the Electrical Activity of Neurons"
date:   2017-08-17 
categories: [numerical methods, variable timestep, simulation, biological neuron models]
tags: [simulation]
---

Im a previous post I covered the simulation of the electrical activity of simple neuron models, and mentioned that an interleaved voltage-current resolution of linear unrelated state ODEs allows for the resolution of the problem as a system of linear equations. However, such resolution is only possible for those *simple* models. Neuron models with non-linear state variable OPDEs and correlation between states require a fully-implicit resolution, and is possible with Backward Euler with Newton iterations. This is a common use case, and underlies the resolution of extra-cellular mechanisms, capacitors between nodes, and other mechanisms that connect neurons' evolution over time.

A fully-implicit method that is commonly utilised to resolve this problem is the Backward (Implicit) Euler, where:

$$
y_{n+1} - y_n = h f (t_{n+1}, y_{n+1})
$$

where $h$ is the timestep, $y$ is the array of states and $f$ is the Right-Hand side function of the ODE. Can we do better? In this post we present a fully-implicit method based on variable-order variable timestepping.

### CVODE for fully-implicit adaptive-order adaptive-step interpolation
 
[lytton2005independent]: https://aip.scitation.org/doi/pdf/10.1063/1.4822377
[sundials]: https://computation.llnl.gov/projects/sundials

The main principle supporting the adaptive time stepping is its particular efficiency due to a large step size upon large and low-varying states e.g. interspiking intervals. NEURON's adaptive algorithm, [CVODE][lytton2005independent] uses separate variable-step integrators for individual neurons in the network. CVODE is used by NEURON as an interface to a collection of methods built on top of existing CVODE and IDA (Integrator for Differential-Algebraic problems) integrators of the [SUNDIALS package][sundials], a SUite of Nonlinear and DIfferential/ALgebraic Equation Solvers. NEURON employs the Backward Differentiation Formula. The backward differentiation formula is a family of implicit multi-step methods for the numerical integration of (stiff) ordinary differential equations. For a given function and time, the BDF approximates its derivative using information from previously computed times, thus increasing the accuracy of the approximation. BDF methods are implicit thus require the computing the solution of nonlinear equations at every step. The BDF multistep method is suitable for stiff problems that define neuronal activity. The main idea behind a stiff equation is that the equation includes terms that may lead to rapid variation of the solution, therefore causing certain numerical methods for solving the equation numerically unstable, unless the step size is taken to be *extremely* small. CVODES is a solver for the Initial Value Problem --- the IVP is an ODE together with an initial condition of the unknown function at a given point in the domain of the solution --- that can be written abstractly as:

$$
\dot{y} = f(t,y) \text{, } y(t_0) = y_0 \text{ where } y \in \mathbb{R}^N
$$

where $\dot{y}$ denotes $\frac{dy}{dt}$. The solution $y$ is differentiated using past (backward) values, ie $$\dot{y}_n$$ is the linear combination of $$y_n$$, $$y_{n-1}$$, $$y_{n-2}, ...$$.  $$y$$ is the array of all state variables, in our case the $$dV_n/dt$$ equation of all compartments, and $$dx_i/dt$$ where $x_i$ is a given state variable of an ion $i$. The variable-step BDF with $q$-order can be written as:

$$
y_n = \sum_i^{q} \alpha_{n,i} y_{n-i} + h_n \beta _{n,0} \dot{y}_n
$$

where $y_n$ are computed approximations to $y (t_n)$, $h_n = t_n - t_{n-1}$ is the step size, $q$ denotes the order (number of previous steps used on BDF), and $\alpha$ and $\beta$ are the unique coefficients of BDF dependent on the method order. The resolution can be unfolded as:

- BDF-1: $$ y_{n+1} - y_n = h f (t_{n+1}, y_{n+1})$$, i.e. the Backward Euler;
- BDF-2: $$ 3y_{n+2} - 4y_{n+1} + y_n = 2 h f (t_{n+2}, y_{n+2})$$;
- BDF-3: $$ 11y_{n+3} - 8y_{n+2} + 9y_{n+1} - 2y_n = 6 h f (t_{n+3}, y_{n+3})$$;
- BDF-4: $$ 25y_{n+4} - 48y_{n+3} + 36y_{n+2} - 13y_{n+1} + 3y_n = 12 h f (t_{n+4}, y_{n+4})$$;
- BDF-5: $$ 137y_{n+5} - 300y_{n+4} + 300y_{n+3} - 200y_{n+2} + 75y_{n+1} - 12y_n = 60 h f (t_{n+5}, y_{n+5})$$;
- BDF-6: $$ 147y_{n+6} - 360y_{n+5} + 450y_{n+4} - 400y_{n+3} + 225y_{n+2} - 72y_{n+1} + 10y_n = 60 h f(t_{n+6}, y_{n+6})$$;

BDF with order $q > 7$ is not zero-stable therefore is unable to be used. If a perturbation $\epsilon$ happens on an initial value, then final result between perturbed and unperturbed results must be no bigger than $k \epsilon$, for any $k \in \mathbb{R}$} therefore cannot be used. CVODE allows for a maximum order or 5.

NEURON also allows a second-order threshold for adaptive time step, which uses the Backward Euler ($q=1$) to build the second-order approximation using Crank-Nicholson instead of BDF-2. This is simply because BDF-2 requires two previous step sizes, whilst Crank-Nicholson requires only one, and delivers the same second-order accuracy --- in practice the Crank-Nicholson also requires two steps if we consider the intermediate step, but this can be computed on the fly, while the BDF requires storage of the previous steps.
<div class="alert alert-warning" role="alert">
  Disclaimer: I still have to double check the previous paragraph with the NEURON developers
</div>

If we denote the Newton iterates by $y_{n,m}$, the functional iteration of the BDF is given by:

$$
y_{n,m+1} = h_n \beta _{n,0} f ( t_n, y_{n,m} ) + a_n
\label{eq_newton_method_derivation}
$$

As with any implicit method, at every step we must (approximately) solve the nonlinear system:

$$
G(y_n) \equiv y_n - h_n \beta_{n,0} f(t_n, y_n) - a_n = 0 \text{ , where } a_n \equiv \sum_{i>0} ( \alpha_{n,i} y_{n-1} + h_n \beta_{n,i} \dot{y}_{n-i} )  
\label{eq_variable_time_step_bdf}
$$

To resolve this, CVODE applies Newton iterations. The Newton's method allows to finding successively better approximations to the roots or zeroes of a function, and is implemented (for one variable) as $x_{n+1} = x_n - f(x_n) / f'(x_n)$ until a sufficiently accurate value is reached. In practice, it requires finding the solution of the linear systems:

$$
M [y_{n(m+1)} - y_{n(m)}] = - G(y_{n(m)})
$$

where 

$$
M \approx I - \gamma J \text{ , } J = \partial f / \partial y \text{ , and } \gamma = h_n \beta_{n,0}
$$

For the Newton corrections, CVODE provides the following options: (a) direct dense solver, (b) direct banded solver, (c) a diagonal approximate Jacobian solver, or the SPGMR (Scaled Preconditioned GMRES) method. A preconditioner matrix is a transformation that conditions a given problem into a form that is more suitable for numerical solving methods. NEURON applies the SPGMR method, that requires a matrix $M$ (or preconditioner matrix $P$ for the SPGMR case) to be an Inexact Newton iteration, in which $M$ is applied in a matrix-free manner, where the matrix-vector products $Jv$ are obtained by either difference quotients or an user-supplied routine. In NEURON, $J y$ is user-provided by the *SOLVE* block in the mechanisms. In practice, the SPGMR transforms the nonlinear problem into a linear one, by introducing a multiplication by the Jacobian (In vector calculus, the Jacobian matrix is the matrix of all first-order partial derivatives of a vector-valued function), and it involves computing the solution of the equation:

$$
y_{n(m+1)} = y_{n(m)} -  J^{-1} G(y_{n(m)}) \\
\equiv y_{n(m+1)} - y_{n(m)} = - J^{-1} G(y_{n(m)}) \\
\equiv P [y_{n(m+1)} - y_{n(m)}] = - P J^{-1} G(y_{n(m)}) \\
\equiv P [y_{n(m+1)} - y_{n(m)}] = - G(y_{n(m)}) \\
\label{eq_newton_cvode}
$$

where 

$$
 P \approx I - \gamma J \text{ , } J = \partial f / \partial y \text{ , and } \gamma = h_n \beta_{n,0}
$$

Equation \ref{eq_newton_cvode} requires a division by the Jacobian $J$, i.e. a multiplication by its inverse. Due to the expensive computation of the inverse of the Jacobian, CVODE allows for a trade-off of accuracy and solution time of any equation, by requiring one to only supply a solver for $P y = b$ instead. For the linear case, although NEURON supports a resolution with the Hines solver (default) or the (a) or (c) methods mentioned previously, there has been little exploration of the non-Hines approaches.

##### Error Handling

One main difference between fixed and variable time stepping, is that from the user's perspective, one does not specify the time step but the relative local (*rtol*) and absolute errors (*atol*) instead. The solver then adjusts $\Delta t$ based on the two values. The local error is based on the extrapolation process within a time step. CVODE uses a weighted root-mean-square norm for all error-like quantities, denoted as:

$$
|| v || = \sqrt{ \frac{1}{N} \sum_{i=1}^N (v_i / W_i)^2}
$$

where $v_i$ represent the deviations and the weights $W_i$ are based on the current solution (with components $y^i$), with user-specified absolute tolerances $atol_i$ and relative tolerance $rtol$ as:

$$
W_i =  rtol \cdot | y^i | + atol_i
$$

If $y_n$ is the exact solution of equation \ref{eq_newton_method_derivation}, we want to ensure that the iteration error is small enough such that $y_{n,m} - y_{n,m-1} < 0.1 \epsilon$ (where $0.1$ is user-changeable). For this, we also estimate the linear convergence rate constant $R$, initialized at $1$ and reset to $1$ when $M$ or $P$ is updated. After computing a correction $\delta_m = y_{n(m)} - y_{n(m-1)}$, we update $R$ if $m > 1$ as:

$$
R \leftarrow \text{max} \left\{ 0.3 R, \frac{|| \delta_m ||}{||\delta_{m-1} ||} \right\}
$$

We use it to estimate

$$
||y_n - y_{n(m)}|| \approx || y_{n(m+1)} - y_{n(m)}|| \approx R || y_{n(m)} - y_{n(m-1)}|| = R || \delta_m ||
$$

Thus, the convergence test is 

$$
R || \delta_m || < 0.1 \epsilon
$$

##### Local and Global Time Stepping

[lytton2005independent]: https://www.mitpressjournals.org/doi/abs/10.1162/0899766053429453

Large networks of cells can not be integrated by a global adaptive time stepping, as a high frequency of events such as gating channels or synapses causes a discontinuity in a parameter or state variable, therefore requiring the reinitialization of computation and forcing the integrator to start again with a new initial condition problem.NEURON's workaround allows for a local variable time stepping scheme called *lvardt* [Lytton et al., Independent Variable Time-Step Integration of Individual Neurons for Network Simulations, MIT Press][lytton2005independent] that creates individual CVODE integrators per cell and aims at short time step integration for active neurons and long time step integration for inactive neurons, while maintaining consistent the integration accuracy. All cells are on a ordered queue with its position given by event delivery time or actual time of simulation. Because the system's integrator moves forward in time by multiple, independent integrators, the overall computation is reduced when compared to the global counterpart. Nevertheless, the main issue is the coordination of time advancement of each cell, i.e. all the state variables for a given cell should be correctly available to any incoming event. To solve that issue, every cell holds the information for all states between its current time instant and the previous time steps. This allows for backward interpolation when required. The integration coordinator also ensures that the interpolation intervals of the connected cells overlap, so that no event between two cells is missed. The critical point of this approach is that the time step is tentative, and an event received by the cell in the interim may reinitialize the cell to a time prior to the tentatively calculated threshold time.

In the following picture, we illustrate A time line (from A to D) of six cells running the *lvardt* algorithm. The intervals of possible interpolation are displayed as a black rectangle. The system's earliest event time is shown by the vertical line traversing all neurons. Triggers of discrete events on a particular neuron are displayed as a short thick vertical bar. (A) the smallest $t_b$ advances by the length of the hashed rectangle; (B) Cell 5 has the smallest $t_bb$ and integrates forward; (C) The next earliest event on the system is at neuron 3. This event trigger creates three new events to be delivered to cells 3,4,5; (D) The new early event forces cell 3 back-interpolates, the event is handled, and cell 3 reinitializes. Cell 3 will be the next cell to integrate forward:

<p align="center"><img width="30%" height="30%" src="/assets/2017-Neuron-Variable-Timestep/local_variable_timestep_six_cell_example.png"><br/><small>source: [The NEURON book][neuron-book]</small></p>

##### Single Neuron Performance

One can notice the difference in performance of the Backward Euler and Variable Timestep interpolators during a stiff trajectory of an Action Potential in the following picture.

<p align="center"><img width="50%" height="50%" src="/assets/2017-Neuron-Variable-Timestep/L5_neuron_soma_voltage_per_step_6ms.png"></p>

Note the difference in interpolation instants, placed at a equidistant interval on the Backward Euler use case, and adapting to solution gradient on the variable step counterpart. A current injection at the soma forces the neuron to spike at a rate that increases with the injected current. A simple consists in measuring the number of interpolations betweeen both methods, by trying to enforce different levels of stiffness and measure the number of steps:

<p align="center"><img width="50%" height="50%" src="/assets/2017-Neuron-Variable-Timestep/cvode_pulse_currents_table.png"></p>

The slow dynamics of the fixed step method --- that end up accounting for the events at discrete intervals instead of event delivery times of its variable step counterpart --- lead to a propagation of voltage trajectory (you can assume Euler with $$\Delta t=1 \mu s$$ and $$CVODE with atol=10^{-4}$$ as reference solutions): 

<p align="center"><img width="50%" height="50%" src="/assets/2017-Neuron-Variable-Timestep/L5_neuron_pulse1_3mA_100ms_cvode_vs_backward_euler.png"></p>

After a second of simulation, the spike time phase-shifting is substantial, demonstrating the gain in using variable step methods for this use case. 

<p align="center"><img width="50%" height="50%" src="/assets/2017-Neuron-Variable-Timestep/L5_neuron_pulse_1000ms_results.png"></p>

To study wether these advantages are feasible, we measure the simulation runtime of both methods. Our first test measures the impact of the stiffness in terms of simulation steps and time to solution on an intel i5 2.6Ghz. Thedifferent current values are injected as proportional to the *action potential threshold* current, i.e. the minimum current that is required to be continuously injected for the neuron to spike.

<p align="center"><img width="50%" height="50%" src="/assets/2017-Neuron-Variable-Timestep/cvode_dependency_on_variation_solution.png"></p>

Results look promising, even at an extremelly high current value we have a significant speed-up. A second test enforces discontinuities and reset of the IVP problems by artificially injecting current pulses of $1 \mu s$ at given frequencies. 

<p align="center"><img width="50%" height="50%" src="/assets/2017-Neuron-Variable-Timestep/cvode_dependency_on_events_arrival.png"></p>

In this scenario, we see a high dependency on the number of discontinuities, and some yet minor dependency on the discontinuity value (i.e. amount of current injected). The holy grail is then in: How does this apply to the real behaviour of neurons?

<div class="alert alert-warning" role="alert">
In the future I will complete this information with a digital reconstruction of laboratorial experiment. For the time being, I must hold that information until it's published.
</div>

##### Previous Experiments

The performance of *lvardt*'s integration model was tested [(Lytton et al. 2015)][lytton2005independent] on a connected homogeneous network of 100 inhibitory neurons on a single compute node. Due to their inhibitory features, the circuit presents rapid synchronization through mutual inhibition, where variable time steps integrators are likely to encounter problems. A fixed time step of $0.0025 ms$ yielded *closely comparable* results to those of *lvardt* with absolute error tolerance (atol) of $10^{-3}$ or $10^{-5}$. The event deliveries and the associated interpolations took about $12\%$ of total execution time. When using *lvardt*'s global time stepping, the author claims the large amount of events lead to an increase of execution time in the order of 60-fold. Another experiment featured a Thalamocortical simulation, featuring four cell types (cortical pyramidal neurons and interneurons, thalamocortical
cells, and thalamic reticular neurons). The two thalamic cell types produced bursts with prolonged interburst intervals, therefore very suitable for the use of the *lvardt* algorithm. A simulation of $150 ms$ of simulation time, aiming at high-precision with an error tolerance (atol) of $10^{-6}$ for *lvardt* and $\Delta t = 10^{-4} ms$ for the fixed time step, yielded an execution time of 6 minute 13 seconds and 10 hour 20 minute, respectively, a 100-fold speed-up. For a standard-precision run, with an atol of $10^{-3}$ and $\Delta t = 10^{-2}$, the simulations took 2 minutes for *lvardt* and 5 minutes 53 seconds for fixed time step, a 3-fold speed-up.

In the following picture we present the afforementioned results for the comparison of fixed and variable time step methods for the mutual-inhibition model. Execution time grows linearly with simulation time for the fixed-step integration (dashed lines). Variable-step methods have a reduction in CPU load at the onset of synchrony. The rationale behind the non-linearity of adaptive time stepping, is that *lvardt* produces time steps that jump around during the initial presynchrony phase of the simulation (fast uprising phase), and then slowly stabilizes to an alternation between large steps in the long intervals separating the population spikes and extremely short time steps during the population spike itself:

<p align="center"><img width="30%" height="30%" src="/assets/2017-Neuron-Variable-Timestep/lvardt_exec_time_comparison.png"><br/><small>source: Lytton et al. 2015 </small></p>

NEURON's default error setting for CVODE is $10 \mu M$ for membrane potential and $0.1 nM$ for internal free calcium concentration, allowing a HH-based action potential to have an accuracy equivalent of fixed time step with $\Delta t = 25 \mu s$. The default maximum order for BDF is $5$, although *it is recommended* to let CVODE decide the best order.

##### Distributed CVODE executions

For parallel computing environments, the synchronization of cells spread across different computing nodes presents a bottleneck, due to the insufficient network latency for real-time message passing, and high amount of information to be actively transmitted (i.e. one per event per node). To tackle the mentioned issues, simulation is split equidistant time intervals equal to the shortest delay of any global event --- typically the minimal synaptic delay across all synapses, or (when applicable) the update frequency for extra-cellular mechanisms or gap junctions. On every time interval, all the nodes on the network perform a collective communication call and share the events to be sent on the next time frame. The downsides of such approach are that (a) no compute node can ever perform a CVODE time step that exceeds the limits of the current time frame, (b) all compute nodes must wait for the slowest to reach the end of the time frame, and (c) all compute nodes restart the CVODE at the initial instant of every time frame.

<div class="alert alert-warning" role="alert">
  In the future I will detail a smarter approach for distributed variable time stepping. For the time being, I must hold that information until it's published.
</div>

