---
layout: post
title:  "Variable Timestep Simulation of the Electrical Activity of Neurons"
categories: ["Computational Neuroscience"]
tags: ["CVODE", "variable timestep", "numerical methods", "simulation", "HPC"]
---

[neuron-methods-post]: {{ site.baseurl }}{% post_url 2016-08-15-neuron-electrical-activity %}

The [previous post][neuron-methods-post] covered the mathematical background of the electrical activity of neurons, and its simulation with a fixed time step $$\Delta t$$, usually 25 μs. It showed that, when each state ODE is linear in its own state and independent of the other states, staggered time steps turn every step into a system of linear equations. Such a solution, however, is only possible for *simple* models. Models with nonlinear state ODEs, or with states that depend on each other, need a fully implicit method, which solves all the equations of a step together. An example is the synaptic plasticity model of [Graupner and Brunel (2012)][graupner2012], a cubic ODE of the synaptic efficacy driven by the calcium concentration. Mechanisms that couple the equations of several compartments, such as the extracellular field or linear circuits with capacitors between nodes, need an implicit solution of all their equations as well.

[graupner2012]: https://www.pnas.org/doi/10.1073/pnas.1109359109

A fully implicit method commonly used for this problem is the backward Euler method:

$$
y_{n+1} - y_n = h \, f(t_{n+1}, y_{n+1})
$$

where $$h$$ is the time step, $$y$$ is the vector of all states, and $$f$$ is the right-hand side of the ODE $$dy/dt = f(t, y)$$. Since the unknown $$y_{n+1}$$ appears on both sides, every step solves a nonlinear system, with Newton iterations. With fixed steps, NEURON offers this method as `derivimplicit`, at first order and one mechanism at a time, so the coupling between mechanisms stays explicit. CVODE, NEURON's variable time step method, is often a more efficient alternative: it solves the voltages and all the states together, implicitly, with a variable order and a variable time step.

The main argument for a variable time step is efficiency. A fixed step must resolve the fastest change of the solution, the rise of a spike, which lasts a fraction of a millisecond. Between spikes, however, the voltages and states change slowly, and most fixed steps add no accuracy. A [variable time step][adaptive-step] method sizes each step from an estimate of its error, so it takes short steps during a spike and long steps in the intervals between spikes. Its weakness is discontinuities: every synaptic event changes the input of a neuron at once, and restarts the method with short steps. Variable steps thus pay off on neurons that receive few events, and the tests below measure how few.

[adaptive-step]: https://en.wikipedia.org/wiki/Adaptive_step_size

## CVODE

[cvode]: https://aip.scitation.org/doi/pdf/10.1063/1.4822377
[sundials]: https://computation.llnl.gov/projects/sundials

CVODE ([Cohen and Hindmarsh, 1996][cvode]) solves the [initial value problem][ivp], an ODE together with the value of its unknown function at an initial time:

$$
\dot{y} = f(t,y) \text{, } y(t_0) = y_0 \text{, with } y \in \mathbb{R}^N
$$

where $$\dot{y} = dy/dt$$. For neurons, $$y$$ holds the voltages of all compartments and the states of all mechanisms, of one cell or of a whole network. NEURON's `CVode` class is an interface to two solvers of the [SUNDIALS][sundials] suite (SUite of Nonlinear and DIfferential/ALgebraic equation Solvers): CVODE for ODEs, and IDA, the Integrator for Differential-Algebraic problems, which the `extracellular` and `LinearMechanism` mechanisms require.

[ivp]: https://en.wikipedia.org/wiki/Initial_value_problem

Neuron models are [stiff][stiff]: their time scales differ by orders of magnitude, and explicit methods are stable on them only with steps of the order of the fastest time scale, i.e. extremely small steps. For stiff problems, CVODE uses the [backward differentiation formulas][bdf] (BDF), a family of implicit [linear multistep methods][lmm]. As the name says, a BDF approximates the derivative at the new time from past, or backward, values of the solution: $$\dot{y}_n$$ is a linear combination of $$y_n, y_{n-1}, \dots, y_{n-q}$$, and more past values give a higher order $$q$$ and a higher accuracy. Solved for $$y_n$$, the approximation of $$y(t_n)$$, the BDF of order $$q$$ reads:

$$
y_n = \sum_{i=1}^{q} \alpha_{n,i} \, y_{n-i} + h_n \beta_{n,0} \, \dot{y}_n
$$

where $$h_n = t_n - t_{n-1}$$ is the step size, and the coefficients $$\alpha_{n,i}$$ and $$\beta_{n,0}$$ depend on the order and on the recent step sizes. Since $$\dot{y}_n = f(t_n, y_n)$$ depends on the unknown $$y_n$$, the method is implicit, and every step solves nonlinear equations. With a constant step $$h$$, the BDFs of orders 1 to 6 are:

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

The list stops at order 6 for a reason. A method converges only if it is [zero-stable][zero-stability]: a perturbation $$\epsilon$$ of its initial values must change the numerical solution on a finite interval by at most $$K \epsilon$$, with $$K$$ independent of the step size. The BDFs above order 6 are not zero-stable, so they cannot be used. The usable orders also differ on stiff problems. Orders 1 and 2 are A-stable: on a decaying linear ODE, they are stable with any step. Orders 3 to 6 are stable only when $$h \lambda$$ lies in a region that shrinks as the order grows, where $$\lambda$$ is an eigenvalue of the [Jacobian][jacobian] $$J = \partial f / \partial y$$. CVODE uses the BDFs of orders 1 to 5.

[zero-stability]: https://en.wikipedia.org/wiki/Zero_stability
[jacobian]: https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant

### Newton Iterations

Like any implicit method, each step of CVODE must solve, at least approximately, the nonlinear system

$$
G(y_n) \equiv y_n - \gamma f(t_n, y_n) - a_n = 0 \text{, with } \gamma = h_n \beta_{n,0} \text{ and } a_n = \sum_{i=1}^{q} \alpha_{n,i} \, y_{n-i}
\label{eq_variable_time_step_bdf}
$$

The simplest way to solve it is the functional, or fixed-point, iteration $$y_{n(m+1)} = \gamma f(t_n, y_{n(m)}) + a_n$$, where $$y_{n(m)}$$ is the iterate $$m$$. It converges, however, only when all eigenvalues of $$\gamma J$$ have a magnitude below 1. A stiff problem has eigenvalues of large magnitude, so this condition limits the step to the fastest time scale, as for an explicit method. CVODE uses [Newton's method][newton] instead, which finds successively better approximations to the root of a function: for a function $$g$$ of one variable, it iterates $$x_{m+1} = x_m - g(x_m) / g'(x_m)$$ until the value is accurate enough. For the system $$G$$, the derivative becomes the Jacobian of $$G$$, $$I - \gamma J$$, and each iteration solves a linear system. Starting from a prediction $$y_{n(0)}$$, extrapolated from the past values, iteration $$m$$ solves

$$
M \, (y_{n(m+1)} - y_{n(m)}) = - G(y_{n(m)}) \text{, with } M \approx I - \gamma J
$$

[newton]: https://en.wikipedia.org/wiki/Newton%27s_method

The matrix $$M$$ only approximates the Jacobian of $$G$$, since computing and factorizing the exact Jacobian at every iteration is expensive. This trades accuracy for time: a poorer $$M$$ slows the convergence of the iterations, which then need more iterations or shorter steps, but each iteration is cheaper. The solves with $$M$$ are the main cost of a step, and NEURON offers three solvers for them, selected with `CVode.jacobian`:
- **dense** (`jacobian(1)`): $$J$$ from finite differences, and an [LU factorization][lu] of $$M$$ in $$O(N^3)$$ time and $$O(N^2)$$ memory. A detailed cell has $$10^4$$ or more states, so its matrix alone takes 800 MB or more;
- **diagonal** (`jacobian(2)`): the diagonal of $$J$$ only, in $$O(N)$$ time and memory. It ignores the axial coupling between compartments, the stiffest part of the system;
- **NEURON's own solver** (`jacobian(0)`, the default): $$J$$ keeps the axial coupling of the voltages and the derivatives of the state equations of each mechanism with respect to its own states. It drops the derivatives of the currents with respect to the states, and of the state equations with respect to the voltage. The voltage part is then the tree matrix of the backward Euler step of the previous post, with $$\gamma$$ in place of $$\Delta t$$, which the Hines algorithm solves in $$O(N)$$ time, and each mechanism then solves for its own states.

[lu]: https://en.wikipedia.org/wiki/LU_decomposition

In the tests of Magalhães et al. (<a href="{{ site.assets }}/neuron-variable-timestep/neurox-vardt-arxiv.pdf">preprint, 2019</a>), the diagonal solver ran faster but gave inaccurate results, and the dense solver was accurate but too slow and too large. CVODE also has a banded solver and a matrix-free [Krylov][gmres] solver (SPGMR), which NEURON does not use.

[gmres]: https://en.wikipedia.org/wiki/Generalized_minimal_residual_method

### Error Control

One main difference between fixed and variable time steps is that the user does not set the step size, but the local error tolerances instead: an absolute tolerance $$\text{atol}_i$$ per state and a relative tolerance $$\text{rtol}$$. CVODE then adjusts the steps to these two values. It measures all errors and corrections with the weighted [root mean square][rms] norm

$$
\Vert v \Vert = \sqrt{ \frac{1}{N} \sum_{i=1}^N \left( \frac{v_i}{W_i} \right)^2 } \text{, with } W_i = \text{rtol} \, \vert y_i \vert + \text{atol}_i
$$

where each weight $$W_i$$ combines the tolerance relative to the current size of state $$i$$ with its absolute tolerance, so that a vector of norm 1 is exactly at the tolerance. The local truncation error of a step is estimated from the difference $$\Delta_n = y_{n(m)} - y_{n(0)}$$ between the corrected and the predicted solution, as $$C' \Delta_n$$, where $$C'$$ depends on the order. The step passes the error test when $$\Vert \Delta_n \Vert \le \epsilon = 1 / \vert C' \vert$$. Otherwise, CVODE repeats it with a shorter step, and after three failures, at order 1. The next step size is $$\eta h_n$$, with

$$
\eta = \left( \frac{\epsilon}{6 \, \Vert \Delta_n \Vert} \right)^{1/(q+1)}
$$

capped at 10, so a step with a small error is followed by a longer one. After $$q+1$$ steps at order $$q$$, CVODE also computes $$\eta$$ for the orders $$q-1$$ and $$q+1$$, and moves to the order that allows the longest step.

[rms]: https://en.wikipedia.org/wiki/Root_mean_square

The Newton iterations stop when the iteration error is ten times below the error test: $$\Vert y_n - y_{n(m)} \Vert < 0.1 \epsilon$$, where $$y_n$$ is the exact solution of equation \eqref{eq_variable_time_step_bdf}, and the factor 0.1 can be changed. Since $$y_n$$ is unknown, CVODE estimates this error from the convergence rate $$R$$ of the iterations. $$R$$ starts at 1, returns to 1 when $$M$$ is updated, and after each correction $$\delta_m = y_{n(m)} - y_{n(m-1)}$$ with $$m > 1$$ becomes

$$
R \leftarrow \max \left\{ 0.3 R, \frac{\Vert \delta_m \Vert}{\Vert \delta_{m-1} \Vert} \right\}
$$

This rate gives the estimate $$\Vert y_n - y_{n(m)} \Vert \approx \Vert y_{n(m+1)} - y_{n(m)} \Vert \approx R \, \Vert \delta_m \Vert$$, and so the convergence test is

$$
R \, \Vert \delta_m \Vert < 0.1 \epsilon
$$

A step allows at most three iterations. When they do not converge, CVODE updates $$M$$ and retries the step, or, if $$M$$ was up to date, retries it with a step four times shorter.

[neuron-book]: https://www.cambridge.org/core/books/neuron-book/7C8D9BD861D288E658BEB652F593F273

In NEURON, $$\text{rtol}$$ is 0 by default, so the error control is absolute, in the units of each state, and `CVode.atolscale` scales $$\text{atol}$$ for each state. In The NEURON Book ([Carnevale and Hines, 2006][neuron-book]), the default tolerances are 10 μV for voltages in mV, i.e. $$\text{atol} = 10^{-2}$$, and 0.1 nM for the internal calcium concentration, and for a Hodgkin-Huxley action potential at 6.3 °C, they give an accuracy comparable to that of the second-order fixed-step method with $$\Delta t = 25$$ μs. The source code of NEURON sets a default of $$\text{atol} = 10^{-3}$$, the value that the tests below use, while its documentation gives $$10^{-2}$$.

## Events and the Local Variable Time Step

[lytton2005independent]: https://www.mitpressjournals.org/doi/abs/10.1162/0899766053429453

An event, such as a synaptic input, changes the right-hand side $$f$$ at a given time, i.e. it is a discontinuity in the parameters or the states. The BDF history before the event does not describe the solution after it, so the integrator stops at the event time, applies the event, and restarts as a new initial value problem, at order 1 and with a short step. Each event thus costs a series of short steps.

This is a problem for a network with a single, global integrator: every event restarts all cells, and all cells advance with the steps of the most active one. NEURON's solution is the **local variable time step** method (lvardt) of [Lytton and Hines (2005)][lytton2005independent], which gives each cell its own CVODE integrator instead. An event restarts the integrator of its target cell only, so an active cell takes short steps while a quiet cell takes long ones, with the same accuracy for both, and the total work drops. The difficulty is the coordination of the cells: when an event arrives, the states of its target cell must be available at the event time, and no event between two cells may be missed. A coordinator solves this as follows:
- each integrator can interpolate its solution to any time of its last step, $$[t_a, t_b]$$, from its BDF history, so it can go back to an event time inside that step;
- an integrator queue orders the cells by $$t_b$$, and an event queue orders the events by delivery time. Both are [priority queues][priority-queue], implemented as [splay trees][splay-tree], with an amortized time per operation that grows with the logarithm of the number of items they hold;
- at each iteration, the coordinator takes the earliest item of the two queues. A cell integrates one step forward. An event makes its target cell interpolate back to the event time, handle the event, and restart its integrator there.

[priority-queue]: https://en.wikipedia.org/wiki/Priority_queue
[splay-tree]: https://en.wikipedia.org/wiki/Splay_tree

Since the coordinator always takes the earliest item, no $$t_b$$ and no event time ever falls behind any $$t_a$$: every event lies within the interpolation interval of its target cell, or ahead of it. A spike found in a step, however, stays tentative until the earliest $$t_b$$ and event time reach it, since an earlier event to the spiking cell can restart the cell before the spike.

The picture shows lvardt on six cells, in four snapshots from A to D. The black rectangles are the intervals $$[t_a, t_b]$$, the vertical line marks the earliest $$t_b$$ or event time, and the short thick bars are events. (A) Cell 0 has the smallest $$t_b$$ and integrates one step (hatched). Its voltage crosses the spike threshold during the step, which creates a tentative spike. (B) Cell 5 now has the smallest $$t_b$$ and integrates one step. (C) The spike of cell 0 is now the earliest item. Handling it creates three events, for cells 3, 4 and 5, with different delays, and the event of cell 3 becomes the earliest. (D) Cell 3 interpolates back to the event time, handles the event and restarts its integrator, so its interval shrinks to a point. Cell 3 integrates next:

{: .picture}
<img width="40.4%" height="40.4%" src="{{ site.assets }}/neuron-variable-timestep/local_variable_timestep_six_cell_example.svg">

{: .caption}
Adapted from: [Lytton and Hines (2005), Independent variable time-step integration of individual neurons for network simulations, Fig. 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC2712447/)

## Single Neuron

How do the two methods compare in practice? The following tests of Magalhães et al. (<a href="{{ site.assets }}/neuron-variable-timestep/neurox-vardt-arxiv.pdf">preprint, 2019</a>) compare NEURON's backward Euler, with its default $$\Delta t = 25$$ μs, and CVODE, with $$\text{atol} = 10^{-3}$$, on a model of a layer 5 pyramidal cell of the Blue Brain Project (L5_TTPC2_cADpyr232_1), with a current injected at the soma. The first test follows the stiff trajectory of one spike. Backward Euler takes equidistant steps, while CVODE adapts its steps to the gradient of the solution: it takes long steps while the voltage changes slowly, before the spike, and short steps on the rise and around the peak:

{: .picture}
<img width="97%" height="97%" src="{{ site.assets }}/neuron-variable-timestep/l5_neuron_soma_voltage_per_step_6ms.svg">

A constant current from 100 ms to 1,100 ms makes the neuron fire at a rate that grows with the current, so a stronger current brings more of the fast changes of a spike. Backward Euler takes 48,001 steps for every current. CVODE takes 338 steps without spikes, and 9,750 at the highest current:

{: .picture}
<img width="60.3%" height="60.3%" src="{{ site.assets }}/neuron-variable-timestep/cvode_pulse_currents_table.svg">

The step size also sets the accuracy. Over 100 ms, backward Euler with $$\Delta t = 1$$ μs and CVODE with $$\text{atol} = 10^{-4}$$ give the same spike times, and serve as references. A fixed step sets the fastest reaction time of the system, so a longer step slows down its dynamics, and backward Euler spikes later. CVODE with $$\text{atol} = 10^{-2}$$ matches the accuracy of backward Euler with $$\Delta t = 25$$ μs in 7 times fewer steps (566 against 4,001), and CVODE with $$\text{atol} = 10^{-3}$$ matches that of $$\Delta t = 5$$ μs in 22 times fewer steps (910 against 20,001), and in 40% fewer steps than the reference CVODE, with 1,512 steps:

{: .picture}
<img width="93.6%" height="93.6%" src="{{ site.assets }}/neuron-variable-timestep/l5_neuron_pulse1_3ma_100ms_cvode_vs_backward_euler.svg">

The delay of backward Euler grows spike after spike, while CVODE accumulates no such shift. After 1 s, the delay is about 2 ms with $$\Delta t = 5$$ μs and 6.5 ms with $$\Delta t = 25$$ μs, a large error at the time scale of spikes, and CVODE with $$\text{atol} = 10^{-3}$$ takes 6,240 steps, against 40,001 and 200,001 for backward Euler:

{: .picture}
<img width="95.2%" height="95.2%" src="{{ site.assets }}/neuron-variable-timestep/l5_neuron_pulse_1000ms_results.svg">

Fewer steps do not guarantee a shorter runtime, since a CVODE step costs more than a backward Euler step, so the next tests measure both. The first one measures how the stiffness of the solution affects them, with a constant current injected for 1 s, from 0 to 5 times the threshold current, the smallest constant current that makes the neuron spike. The runs use an Intel Core i5 at 1.6 GHz. Below half the threshold current, the voltage settles to a steady state, and CVODE takes 434 times fewer steps than backward Euler and runs 98 times faster. At the threshold current, the gains are 62 and 11.6 times, and even at 5 times the threshold current, 9.4 and 2.3 times. The step counts and the runtimes thus show that a CVODE step costs 4 to 5 times as much as a backward Euler step: it runs Newton iterations, each with an evaluation of $$f$$ and a linear solve, and an error test.

{: .picture}
<img width="98%" height="98%" src="{{ site.assets }}/neuron-variable-timestep/cvode_dependency_on_variation_solution.svg">

The last single-cell test forces discontinuities, and the restarts of the IVP that they cause, by replacing the constant current with pulses of 1 μs at a fixed rate, as a model of synaptic events. The runtime of CVODE then depends strongly on the pulse rate, and much less on the pulse amplitude: a stronger pulse changes the trajectory more, and needs a few more steps. At 10 pulses per second, CVODE takes 153 to 322 times fewer steps than backward Euler and runs 51 to 100 times faster, from the strongest to the weakest pulse. Both methods run equally fast at about 1,000 pulses per second for the strongest pulse, and at about 1,600 for the weakest. Real synaptic events arrive at irregular times rather than at a fixed rate, so this test only estimates the event rate that CVODE can afford.

{: .picture}
<img width="97.9%" height="97.9%" src="{{ site.assets }}/neuron-variable-timestep/cvode_dependency_on_events_arrival.svg">

## Homogeneous Networks

The key question is then how these results carry over to networks of neurons. [Lytton and Hines (2005)][lytton2005independent] tested lvardt on one compute node. Their worst case is a fully connected network of 100 inhibitory neurons, which synchronize quickly through mutual inhibition: each population spike sends 9,900 events at almost the same time, a case where variable time steps are likely to struggle. A fixed step of 2.5 μs gave results close to those of lvardt with $$\text{atol} = 10^{-3}$$ or $$10^{-5}$$. Event delivery, with its interpolations, took about 12% of the CPU time of lvardt. A single integrator for the whole network was up to 60 times slower than lvardt, since every event restarted all cells.

The picture shows the CPU time of this model. With fixed steps (dashed lines), it grows linearly with the simulated time. lvardt is slow at the start, while the cells are not yet synchronized and its steps vary widely. Once the network synchronizes, lvardt settles into an alternation of long steps between the population spikes and short steps during them, and its CPU time grows slowly:

{: .picture}
<img width="31.1%" height="31.1%" src="{{ site.assets }}/neuron-variable-timestep/lvardt_exec_time_comparison.svg">

{: .caption}
Adapted from: [Lytton and Hines (2005), Fig. 3A](https://pmc.ncbi.nlm.nih.gov/articles/PMC2712447/)

Their best case, a synfire chain of 100 single-compartment Hodgkin-Huxley cells, ran 20 times faster with lvardt than with a fixed step of similar accuracy. A thalamocortical model has four cell types: cortical pyramidal cells and interneurons, thalamocortical cells and thalamic reticular cells. The two thalamic types fire bursts separated by long silent intervals, which suits lvardt well. For 150 ms of simulation at high precision, lvardt with $$\text{atol} = 10^{-6}$$ took 6 min 13 s, and a fixed step of 0.1 μs took 10 h 20 min, 100 times longer. At standard precision, with $$\text{atol} = 10^{-3}$$ and $$\Delta t = 10$$ μs, the runs took 2 min and 5 min 53 s, 3 times longer.

## A Heterogeneous Network

[markram2015cell]: https://www.cell.com/fulltext/S0092-8674(15)01191-5

Networks of similar neurons give only a rough estimate of the speed-up, since in practice the activity of the brain varies widely across neurons and over time. To measure how often real neurons receive events, Magalhães et al. reproduced the digital reconstruction of a laboratory experiment by the Blue Brain Project ([Markram et al., 2015][markram2015cell]): a fixed-step simulation of 7.5 s of the spontaneous electrical activity of 219,247 neurons of the rat neocortex under tonic depolarization. The network shows slow oscillations, with population bursts that start in layer 5, spread down to layer 6 and then up to layers 4 and 2/3, with secondary bursts that spread back to layer 6. The section *Simulating Spontaneous Activity* of the paper gives more details on the biology.

The pictures show two neurons of the top 1% by number of incoming events, i.e. of discontinuities: the events per 10 ms (left), and the histogram of the intervals between events (right). The neurons of the top 1% receive 3,040 to 6,146 events, 405 to 820 per second:

{: .picture}
<img width="59.3%" height="59.3%" src="{{ site.assets }}/neuron-variable-timestep/neuron_203076.svg">
<img width="59.5%" height="59.5%" src="{{ site.assets }}/neuron-variable-timestep/neuron_138083.svg">

The neurons of the median 1% receive 541 to 558 events, about 73 per second. The two neurons below receive 584 and 586 events:

{: .picture}
<img width="58.3%" height="58.3%" src="{{ site.assets }}/neuron-variable-timestep/neuron_208223.svg">
<img width="58.1%" height="58.1%" src="{{ site.assets }}/neuron-variable-timestep/neuron_87784.svg">

The neurons of the bottom 1% receive fewer than 100 events:

{: .picture}
<img width="57.6%" height="57.6%" src="{{ site.assets }}/neuron-variable-timestep/neuron_132412.svg">
<img width="58%" height="58%" src="{{ site.assets }}/neuron-variable-timestep/neuron_2101.svg">

In total, the neurons receive about 155 million events, 707 per neuron on average, or 94 per second: an order of magnitude below the break-even rate of the pulse test. The median and bottom neurons have long silent intervals between events, which suits variable steps, while the top neurons do not. Two issues remain before the speed-ups of the single-neuron tests can be expected on such a network. A network of this size runs on many compute nodes, which must synchronize the time steps of their neurons, and its spike rates change with the species, the brain region and the state of the animal. The rest of the post addresses both.

## Distributed Execution

Networks of this size need distributed memory and compute: as in the [previous post][neuron-methods-post], each rank integrates its own neurons, and the ranks exchange spikes. The synchronization of the neurons across ranks then becomes the bottleneck. When a neuron spikes, the synaptic current must reach each target neuron before the target advances past the delivery time, or the spike is missed, so the ranks must know how far the neurons of the other ranks have advanced.

### Bulk-Synchronous Parallel

NEURON splits the simulation into intervals of $$\Delta t_{\text{comm}}$$, the minimum synaptic delay between neurons on different ranks. A spike reaches the other ranks only after this delay, i.e. in a later interval, so within an interval each rank integrates its neurons independently. At the end of the interval, a collective call exchanges the spikes, and acts as a barrier: no rank starts the next interval before all ranks finish the current one. This is the [bulk synchronous parallel][bsp] (BSP) model. With variable steps, a neuron may take one step past the barrier. When an event from another rank arrives for a time inside that step, the neuron interpolates back and restarts, which discards part of the step. The barriers also cost more with variable steps than with fixed steps. The work of a rank then depends on the activity of its neurons, so the ranks finish each interval at different times, and all of them wait for the slowest one: a [load imbalance][load-balancing] at every interval.

[bsp]: https://en.wikipedia.org/wiki/Bulk_synchronous_parallel
[load-balancing]: https://en.wikipedia.org/wiki/Load_balancing_(computing)

### Fully-Asynchronous Parallel

The fully-asynchronous parallel (FAP) model of Magalhães et al. (<a href="{{ site.assets }}/neuron-variable-timestep/neurox-cache-efficient.pdf">ICCS 2019</a>) removes the barriers, so that computation, communication and synchronization are all asynchronous. It rests on three ideas:
- each neuron keeps a map of its pre-synaptic neurons, those that may spike and deliver current into it, which small stepping notifications from these neurons keep up to date;
- each neuron keeps a map of its post-synaptic neurons, those that receive current from it when it spikes, and besides its spikes, it regularly informs them that it stepped, i.e. advanced in time;
- instead of waiting at barriers, each neuron advances as far as its pre-synaptic neurons allow: up to the time that guarantees that no event is missed if one of them spikes.

Concretely, each neuron $$j$$ stores, for each pre-synaptic neuron $$i$$, the last time $$t_i$$ that $$i$$ reported, and the minimum delay $$d_{ij}$$ of the connections from $$i$$ to $$j$$. A spike of $$i$$ after $$t_i$$ reaches $$j$$ at $$t_i + d_{ij}$$ or later, so $$j$$ can integrate up to

$$
t_j^{\max} = \min_{i \in \text{pre}(j)} \left( t_i + d_{ij} \right)
$$

without missing an event. Each neuron reports its time to its post-synaptic neurons at intervals set per pair, from 0.1 ms to several milliseconds, with one message per target rank. A scheduler on each rank steps the neuron with the earliest time first. No step is ever undone, and no rank waits at a barrier. Most delays $$d_{ij}$$ exceed $$\Delta t_{\text{comm}}$$, so neurons step past the intervals of BSP and take more steps in a row, while their data stays in the faster levels of the cache. With fixed steps, FAP alone runs 26% to 65% faster than BSP on one compute node, depending on the architecture, and 16% to 40% faster on 32 nodes. The gain shrinks as the network grows.

A speculative variant, which lets neurons step ahead and undo the steps that a late event invalidates, does not scale: an undone step can invalidate spikes already sent to other ranks, which then undo their own steps.

### Benchmark

[LIF-model-post]: {{ site.baseurl }}{% post_url 2016-02-08-lif-brunel %}

The <a href="{{ site.assets }}/neuron-variable-timestep/neurox-vardt-arxiv.pdf">preprint</a> compares fixed and variable steps, with BSP and FAP, on 64 Cray XE6 nodes (AMD Opteron 6380, 16 cores at 2.5 GHz), for networks of 1,024 to 65,536 neurons and 1 s of activity, with $$\Delta t = 25$$ μs and $$\text{atol} = 10^{-3}$$:
- simple models, with state ODEs linear in their own state: (1a) BSP with `cnexp`, (1b) BSP with explicit Euler for the states, and (1c) FAP with explicit Euler;
- complex models: (2a) BSP with `derivimplicit`, (2b) BSP with CVODE, and (2c) FAP with CVODE, with each event delivered at its exact time, or with event grouping (EG): the events of each window of $$\Delta t / 2$$ or $$\Delta t$$ are delivered together, as with fixed steps.

A constant current drives all neurons to one of five spiking rates, which stand for different species, brain regions and states:
- quiet, 0.25 Hz: neurons almost at rest. The studies that the preprint cites place about 90% of the neurons of the brain at or below this rate in regular activity, so this rate bounds their runtime from above;
- slow, 1.5 Hz: the lowest rate of active neurons;
- moderate, 6.5 Hz: the rate of the SI slow state of the [Brunel network][LIF-model-post], above the 0.005 to 5 Hz of the rat frontal cortex, and a loose approximation of the visual cortex of cats;
- fast, 38 Hz: the rate of the AI state of the Brunel network, and of cortical and thalamic neurons during high vigilance, 30 to 40 Hz;
- burst, 55.8 Hz: the rate of the SI fast state of the Brunel network, typical of strong depolarizing currents.

The activity is quasi-homogeneous across neurons, on densely connected networks, and the preprint notes that this set-up favors the fixed-step and BSP methods, so its speed-ups are a lower bound.

{: .picture}
<img width="87.6%" height="87.6%" src="{{ site.assets }}/neuron-variable-timestep/benchmark_two_rows_2.svg">

The runtimes of the fixed-step methods differ by about 2% across the five rates: each step costs the same with or without spikes, and the synaptic events and the collective communication add little work. The variable-step methods, on the other hand, are penalized at high event rates: their runtimes approach those of the fixed-step methods as the spiking rate grows, and grow faster with the network size. There are two reasons. First, the events of a variable-step method arrive at any time on a continuous time line, not only at the ends of fixed steps, so they add steps. Second, each event restarts its target cell with short steps, while a fixed-step method keeps its step size. Against the fixed-step method for complex models (2a), FAP CVODE with grouping per $$\Delta t$$ runs 544 times faster on 1,024 neurons and 65 times faster on 65,536 neurons at the quiet rate, and 7.7 to 1.8 times faster at the moderate rate. At the fast rate, it runs 2 times faster on 1,024 neurons and as fast on 65,536. At the burst rate, which is unlikely in practice, it runs 1.5 times faster on 1,024 neurons and 1.5 times slower on 65,536. At the quiet rate, FAP CVODE, which solves the complex model with Newton iterations, even beats the fixed-step methods for simple models, which solve only one linear system per step: despite the cost of the Newton iterations, the rare events allow very long steps.

Event grouping matters on large networks: on 65,536 neurons at the quiet rate, the speed-up falls from 65 times to 47 with grouping per $$\Delta t / 2$$, and to 33 without grouping. The gap between BSP and FAP shrinks as the network grows, since more connections per neuron bring $$t_j^{\max}$$ closer to $$\Delta t_{\text{comm}}$$. On 65,536 neurons, both models run in similar times with fixed steps, and with CVODE at the moderate, fast and burst rates.

To estimate the speed-up on a realistic mix of rates, the preprint groups the 31,346 neurons of the central column of the 219,247-neuron network by firing rate, between 2 s and 4 s of activity: 31.43% quiet, 38.44% slow, 27.02% moderate, 3.10% fast and 0.01% burst. The mean of the speed-ups of FAP CVODE over the fixed-step method (2a) at the five rates, weighted by these shares, is 224.5 to 11.9 with exact event delivery, and 228.5 to 24.6 with grouping per $$\Delta t$$, from 1,024 to 65,536 neurons. This mean is higher than the speed-up of a network with this mix of rates. The runtime of such a network is the weighted sum of the runtimes of the five rates, so its speed-up is the weighted [harmonic mean](https://en.wikipedia.org/wiki/Harmonic_mean) of their speed-ups, which the smallest speed-ups dominate. With 27.02% of the neurons at the moderate rate, where CVODE is 7.7 to 1.8 times faster, the speed-up is below $$1/(0.2702/7.7) \approx 28.5$$ on 1,024 neurons and below $$1/(0.2702/1.8) \approx 6.7$$ on 65,536 neurons. The quiet rate holds 31% of the neurons here, against about 90% in the literature, so the speed-up on a brain-like mix is higher. At the three slowest rates, which hold over 95% of the neurons, the preprint finds that the runtimes of both methods grow at the same rate beyond 16,384 to 32,768 neurons, so the speed-up should hold on larger networks. The preprint gives more details on each of these comparisons: event grouping, the two execution models, and the dependency of the runtime on the network size and the spike rate.
