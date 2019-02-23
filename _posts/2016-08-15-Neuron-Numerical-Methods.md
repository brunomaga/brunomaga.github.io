---
layout: post
title:  "Numerical Resolution of the Electrical Activity in Branched Neuron Models"
date:   2016-08-15 
categories: [numerical methods, simulation, biological neuron models]
tags: [simulation]
---

[number-of-neurons]: https://en.wikipedia.org/wiki/List_of_animals_by_number_of_neurons

Neurons are a very complex structure of the human body that govern behaviour, pain, feelings, motion, and most factors that interfere with out human life. The human body has approximately $8.6^{10}$ neurons, connected by $1.5^{14}$ synapses. For the sake of comparison, larvae have 231 neurons, mice have approximately $71$ Million neurons, and the african elephant has almost four times the human scale, with $25.7^{10}$ neurons --- see the wikipedia entry [List of animals by number of neurons][number-of-neurons] for a larger list neurons per mammal. 

Due to very sophisticated anatomical structure and behaviour of neurons, we utilise numerical methods applied to approximation of neuron models in order to simulate the electrical activity on neural networks.

<p align="center"><img width="40%" height="40%" src="/assets/2016-Numerical-Methods-Neuron/Neuron.svg"><br/><small>source: wikipedia</small></p>

##### Hodgkin-Huxley Model and Cable Theory

[hodgkin1952quantitative]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1392413/
[cable-theory]: http://www.scholarpedia.org/article/Neuronal_cable_theory

The Hogkin-Huxley (HH) model([A. L. Hodgkin and A. F. Huxley, 1952][hodgkin1952quantitative]) is built on several assumptions that disregard various features of the living cell and reduces the multidimensional complexity of the propagation and interaction of electrical signals in spatially extended nerve cells to a set of one dimensional constraints that fits the [Cable Theory][cable-theory], a mathematical model to describe the changes in electric current and voltage along passive neurites.

A neuron's membrane is formed by a phospholipid bilayer, a dielectric material, acting as a leaky capacitor. The cytoplasm inside the neurite (a part of a dendrite or axon) is considered to be an excellent ohmic conductor and the voltage within its membrane is assumed to be constant. The neurites are considered to be long and thin in such way that the voltage is assumed to vary much more along the long axis of the neural process than perpendicular to it. Therefore, diffusion of ions along the membrane surface and the dynamics of current perpendicular to the membrane are deemed irrelevant. The neuron morphology is simplified to a branched tree of interconnected neurites a length small enough to accept that the voltage across each neurite is constant. External to the neuron's morphologies, certain mechanisms lead to electrical and chemical reactions that affect the potential of each individual capacitor. These mechanisms are typically gating channels, and allow the in/outflux of molecules (ions), that depolarize or hyperpolarise the membrane accordingly.

Another important mechanism are the synapses, which represent the interneural connections of the circuit. Synapses can drive the post-synaptic membrane potential upwards or downwards based on the excitatory or inhibitory properties of the pre-synaptic cell. Synapses are initiated when the potential of a membrane hits a particular Action Potential (AP) threshold, and is characterized by a stiff change of membrane potential. The lowest AP threshold on a neuron is typically at the beginning of the axon (the Axon Initial Segment), where synapses are first initiated. Other AP may happen throughout the axonal arborization as a current propagation mechanism. Synapses are driven by a fast influx of Sodium ($Na^+$) ions, and by the slow outflux of the Potassium ($K^+$) channels. The chlorine ($Cl^-$) conductance helps to bring the membrane potential back to the rest state following an activation. Several other mechanisms such as gap junctions (active electrical synapses), extracellular space mechanisms, or second messengers also exist. In the cable approximation model, the complexity of these mechanisms are reduced to the measurement of their ionic conductance. When the conductance of these molecules is constant, we call it a leak (passive) conductance. A compartment in ionic equilibrium, i.e. influx and outflux of neurons cause no variation of its potential, is said to be at its reversal potential.

##### Mathematical Basis

The application of cable theory on the calculation of electrical signaling throughout neurons, resumes the activity of a neuron compartment to a second order partial differential equation that governs the change of axial current depending of time and distance, and reduces the relationship between current and voltage to a one-dimensional cable described by:

$$
\frac{\partial V}{\partial T} + F(V) = \frac{\partial^2 V}{\partial X^2}.
\label{equation338_neuronbook}
$$

Because the cable equation is linear and homogeneous, it obeys the superposition principle. The superposition principle states that, for all linear systems, the net response at given place and time caused by several stimuli, equals the sum of the responses of each stimulus individually. More general solutions suitable for branched neurons can be obtained by summing equations of this form, with an appropriate boundary condition. Neurons are represented as interconnected neurites with individual properties, later discretized in time and space to allow for a conversion into algebraic difference equations, which can be solved iteratively using numerical methods. For any given neurite, the principle of conversation of charge requires that the sum of currents from all sources must equal 0. The charge conservation principle states that electric charge can neither be created nor destroyed, and the amount of net quantity of electric charged given by sum of positive change and negative charge is always conserved. I.e.

$$
\sum i_a - \int_A i_m dA = 0
$$

where $i_a$ is the axial current [mA], flowing through the compartment's region, and  $\int_A i_m dA$ is the transmembrane current density $i_m$ [mA/cm$^2$] over the membrane area $A$ [cm$^2$]. This concept is illustrated in the following picture:

<p align="center"><img width="20%" height="20%" src="/assets/2016-Numerical-Methods-Neuron/net_current_conservation.png"></p>

A common assumption is that a neuron is spatially divided into compartments small enough in such way that the spatially varying $i_m$ is any compartment $j$ is well approximated to the value on its center, therefore

$$
\int_A i_{m_j} dA_j = i_{m_j} A_j = \sum_k i_{a_{kj}}
$$

Ohm's law states that the current between two points of a conductor is directly proportional to the voltage across those two points, as described by the relationship $I = V / R$. In order to resolve the axial current between a compartment $j$ and its neighbors, we calculate the value based on Ohm's where each axial current $i_a$ is approximated  by the voltage *drop* between the centers of the compartments and divided by the resistance of the path between them (the axial resistance), therefore:

$$
i_{a_{kj}} = \frac{v_k - v_j}{r_{jk}}
$$

where $r_{jk}$ represents the axial resistance i.e. the resistance of the path between the compartments $k$ and $j$. Finally, the total membrane current given by the sum of the capacitive and ionic components must equal the sum of axial currents $i_{a_{kj}}$ that enter the compartment $j$ from each adjacent neighbor $k$, thus:

$$
c_j \frac{dv_j}{dt}+i_{ion_j} (v_j, t) = i_{m_j}A_j
\label{convervation_energy_eq}
$$

where $c_j$ is the membrane capacitance of the compartment, $c_j (dv_j / dt)$ the capacitive component, and $i_{ion} (v_j, t)$ represents the ion channels conductances. Applying spatial discretazation to the equation of conservation of energy leads to a set of ordinary differential equations  of the form:

$$
c_j \frac{dv_j}{dt}+i_{ion_j} (v_j, t) = \sum_k \frac{v_k - v_j}{r_{jk}}.
\label{equation345_neuronbook}
$$

We ommit from equation \ref{equation345_neuronbook} any injected source currents from the equation, which (if existent) is included on the right-hand side of the equation. If we consider the special case of an unbranched cable with constant parameter, the axial current is delimited by two points involving compartments $j-1$ and $j+1$ i.e.

$$
c_j \frac{dv_j}{dt}+i_{ion_j} (v_j, t) = \frac{v_{j-1} - v_j}{r_{j-1,k}} + \frac{v_{j+1} - v_j}{r_{j+1,k}}
\label{equation346_neuronbook}
$$

For a compartment with length $\Delta x$ and diameter $d$, the capacitance is given by $C_m \pi d \Delta x$ and the axial resistance by $R_a \frac{\Delta x}{\pi} (\frac{d}{2})^2$, where $C_m$ is the membrane capacitance and $R_a$ its cytoplasmic resistivity. Replacing those terms on equation \ref{equation346_neuronbook} leads to:

$$
C_m \frac{dv_j}{dt}+i_{ion_j} (v_j, t) = \frac{d}{4Ra} \frac{v_{j+1} - 2 v_j + v_{j-1}}{ \Delta x^2}
\label{equation347_neuronbook}
$$

If we assume the discretizaton variable to be small enough i.e. $\Delta x \leftarrow 0$, using finite difference we wave the first derivative at the point in $j$ given by $\frac{v_j - v_{j-1}}{\Delta x}$ and $\frac{v_{j+1} - v_j}{\Delta x}$, and the second-order correct approximation as 

$$
\frac{\partial^2v}{\partial x^2} = \left( \frac{v_{j+1} - v_j}{\Delta x} - \frac{v_j - v_{j-1}}{\Delta x} \right) \frac{1}{\Delta x} = \frac {v_{j+1} - 2v_j + v_{j-1}}{ \Delta x^2}
\label{eq_2nd_order_finite_difference}
$$

we can replace this term in equation \ref{equation347_neuronbook}, where we get

$$
C_m \frac{dv_j}{dt}+i_{ion_j} (v_j, t) = \frac{d}{4Ra} \frac{\partial^2v}{\partial x^2}
\label{equation348_neuronbook}
$$

Following Ohm's law we make the substitution $i = v / R_m$, and multiply $R_m$ on both sides, leading to the final formulation

$$
R_m C_m \frac{dv_j}{dt} + v = \frac{d R_m}{4Ra} \frac{\partial^2v}{\partial x^2}
\label{equation349_neuronbook}
$$

By substituting $t = T \tau$ and $x = X \lambda$ we scale equation \ref{equation349_neuronbook} to the time and space constants $\tau = RC$ and $\lambda = (1/2) \sqrt{d R_m / R_a}$, leading to a final formulation in the form of the Cable equation \ref{equation338_neuronbook}

##### Ionic Currents

All channels may be characterized by their resistance or by their conductance $g = 1/R$, and by gating variables that add the **open probability** of a given channel, that generalizes external events that may block an ion channel. Hodgkin and Huxley formulated the ionic component components of a membrane as:

$$
\sum_k I_k = g_{N_a} m^3 h (u - E_{Na}) + g_K n^4 (u - E_K) + g_L (u - E_L)
$$

where $E$ represents the reversal potentials and $m$, $n$ and $h$ the gating variables --- see the original HH paper for details of individual ODEs and reported values. The leak generalizes any other voltage-independent conductance and is represented by $L$. 

##### Reversal Potential

The reversal potential for a particular ion flux in a compartment is given by the Nernst equation

$$
E = \frac{RT}{zF} ln \frac{\text{[ion]}_{out}}{\text{[ion]}_{in}} \approx \frac{RT}{zF} 2.3026 \log _{10} \frac{\text{[ion]}_{out}}{\text{[ion]}_{in}}
$$

where $$ext{[ion]}_{out}$$ and $$\text{[ion]}_{ion}$$ denote the ionic concentration on the inside and outside of the cell, $T$ is the temperature (Kelvin) (assumed to be $307.15 K$ or $34 ^{\circ}\mathrm{C}$), $z$ is the charge of the ion, and $R$ and $F$ are the Gas ($8.3144621 J$ $K^{-1}$ $mol^{=1}$) and Faraday constants ($96485.33289(59)$ $C$ $mol-1$) respectively. The typical reversal potential for the most common ions are approximately: 
- $$E_{Cl^-}=-70 mV$$ with $$[Cl^-]_{in}=5 mM$$ and $$[Cl^-]_{out}=120 mM$;
- $$E_{K^+}=-90 mV$$ with $$[K^-]_{in}=150 mM$$ and $$[K^-]_{out}=4 mM$$; 
- $$E_{Na^+}=60 mV$$ with $$[Na^+]_{in}=12 mM$$ and $$[Na^+]_{out}=145 mM$$; and
- $$E_{Ca^{2+}}=130 mV$$ with $$[Ca^{2+}]_{in}=0.1 mM$$ and $$[Ca^{2+}]_{out}=1 mM$$.

The reversal potential of a typical HH compartment with $Na^+$, $K^+$ and $Cl^-$ ionic pumps is given by the Goldman–Hodgkin–Katz flux equation as:

$$
V = - \frac{RT}{F} ln \frac{P_{Na} [Na^+]_{out} + P_{K} [K^+]_{out} + P_{Cl} [Cl^-]_{in}} {P_{Na} [Na^+]_{in} + P_{K} [K^+]_{in} + P_{Cl} [Cl^-]_{out}} \approx -60 mV
$$

##### Boundary Conditions 

The value of the normal derivative of the boundary function is a Neumann boundary condition assuming a neurite with a *free* ending point follows the form $\frac{\partial v}{\partial x}=0$, i.e. no current flows on the terminal of a cable.

##### Branching Points

Branching points are modeled as *idealized* portions of the neurite that do not have membrane properties yet have axial current. Spatial discretization is equivalent to reducing the spatially distributed neuron to a set of connected compartments, and results on a set of ordinary differential equation that respect Kirchoff's current law. Kirchoff's current law, based on the principle of conservation of electric charge, states that the sum of currents entering any junction is equal to the sum of currents leaving that junction.  

### Numerical Resolution Methods in NEURON

[neuron]: https://www.neuron.yale.edu/neuron/

Most equations that govern the brain mechanisms do not have an analytic solution. The [NEURON simulator][neuron] --- henceforth referred to simple as NEURON --- addresses such problems by allowing for a biologically realistic --- not infinitely detailed --- models of brain mechanisms, by utilizing several methods for accurate discretization of the neuronal activity into a discrete space. 
##### The Hines Solver

[hines1984efficient]: https://www.sciencedirect.com/science/article/pii/0020710184900084

Due to the unique morphology of a cable model representation of a discretized neuron, NEURON implements an optimized linear solver for bifurcated trees, denominated as the Hines Algorithm ([Efficient computation of branched nerve equations, Hines et al. Elsevier][hines1984efficient]). Data is represented as a sparse tridiagonal matrix where the indices of the parent nodes (the ones immediately above on the tree structure) are provided by an extra vector $p$. The membrane potential of each compartment is represented by the main diagonal. The contribution from  a parent to its children compartments (and vice-versa) are represented by all cells on the same row (column) on the upper (lower) diagonal of the matrix. As each compartment can have only one parent, this enforces only one cell per row on the lower diagonal, and one cell per column on the upper diagonal. For completion, the followin pictures presents a neuron's dendritic arborization spatially discretized based on the cable model (left) and its sparse matrix representation according to the Hines algorithm (right).
<p align="center"><img width="35%" height="35%" src="/assets/2016-Numerical-Methods-Neuron/solver_compartment_neuron.png"> <img width="25%" height="25%" src="/assets/2016-Numerical-Methods-Neuron/solver_grid_neuron.png"></p>


The main rationale behind the Hines algorithm is that if the we can number sequentially the compartments in such way that they all have an index greater than any of its children, then we can reduce the computational complexity of the solver from the $O(n^3)$ found on a typical Gaussian elimination to $O(n)$. Given a tridiagonal matrix with vectors $d$ (main diagonal), $b$ (lower diagonal) and $a$ (upper diagonal), representing a single cable without branching, we can show that the Hines algorithm is an inverted Gaussian elimination adapted for branched structures:

- Starting with the forward triangulation process on each $cell_i$ on row $i$:
  - $$cell_i \leftarrow cell_i \frac{b_i}{d_{i-1}} row_{i-1}$$;
- Changing function variable from $i$ to $i+1$:
  - $$cell_{i+1} \leftarrow cell_{i+1} \frac{b_{i+1}}{d_i} row_i$$;
- Replacing forward by backward triangulation:
  - $$cell_{i-1} \leftarrow cell_{i-1} \frac{a_{i-1}}{d_i} row_i$$;
- Parent contributions for top node ($a_0$ and $b_0$) do not exist:
  - $$cell_{i-1} \leftarrow cell_{i-1} \frac{a_i}{d_i} cell_i$$;
- Changing parents' indices from single cable ($i-1$) to branched tree structure ($p_i$):
  - $$cell_{p_i} \leftarrow cell_{p_i} \frac{a_i}{d_i} cell_i$$;

which is the backwards triangulation applied. The substitution step in Hines is replaced by a forward substitution instead.

An important feature of the Hines algorithm is that it allows parallelization at the junction level, by starting the computation at the branches without children (*terminal* branches), the individual contributions from children branches can be computed in parallel, and included recursively at the parent compartment's node.

##### Mechanisms and Events

[carnevale2002efficient]: https://www.researchgate.net/publication/267700936_EFFICIENT_DISCRETE_EVENT_SIMULATION_OF_SPIKING_NEURONS_IN_NEURON
[hines2000expanding]: https://www.neuron.yale.edu/neuron/static/papers/nc2000/nmodl400.pdf

In NEURON, extra-cellular activity, recording patches, voltage clamps, interneuron kinetics (such as synapses and gap junctions) or any other contribution that affects the membrane potential itself are introduced as **mechanisms**. A mechanisms with a specific execution time is called an **event**. Events in NEURON are managed by the event delivery delivery system ([Efficient discrete event simulation of spiking neurons in NEURON, Carnevale et al., ResearchGate][carnevale2002efficient]). Mechanisms are encoded in NMODL language [Expanding NEURON’s Repertoire of Mechanisms with NMODL, Hines et al.][hines2000expanding], where the main user-provided blocks are *BREAKPOINT* (the main computation block of the model where solves are integrated by a *SOLVE* statement), *DERIVATIVE* (if states are governed by differential equations, assigns values to the derivative), *PROCEDURE* and *FUNCTION* (function declaration and calls), *PARAMETER* and *CONSTANT* (for read-write and read-only variables, respectively) and *NET_RECEIVE* (a protocol to inform the variable time step scheduler that an event has occurred within the previous time step).

### Fixed Time Stepping Interpolation

We showed previously that the principle of conversation of charge between two or more coupled compartments, including an external source of current injection $$I_{inj}$$, can be stated by a set of equations of the form:

$$
c_j \frac{dv_j}{dt}+i_{ion_j} (v_j, t) = \sum_k \frac{v_k - v_j}{r_{jk}} + I_{inj}
\label{equation42_neuronbook}
$$

[neuron-book]: https://www.cambridge.org/core/books/neuron-book/7C8D9BD861D288E658BEB652F593F273

The resolution of the aforementioned based on numerical methods raises several concerns on stability, accuracy and efficiency, caused by the errors on the spatial discretization on the cable theory model and by the algorithms for resolution of the numerical solutions. This has been analyzed in [The NEURON book][neuron-book], where the authors compare its numerical accuracy against the spatial discretization of the Fourier cosine terms $$cos ( \pi n x / L)$$.

NEURON performs a spatial discretization of a compartment by placing $m$  points on the center of equidistant intervals of size $ \Delta x = L /(m-1)$, thus each point is placed at $x = (i+0.5) L/m$, for $0 \geq i > m$. This is known (by numerical methods theory) to have higher accuracy when compared with the traditional method that places the $m$ points at positions $x = i \text{ } L/(m-1)$ for $0 \geq i \geq (m-1)$. In either case, $m$ points are placed at a given compartment and $m=1$ corresponds to spatial frequency of $0$, i.e. uniform membrane potential along the entire cable. Also, the highest number of half waves that can be represented by the discretized system is $n = m-1$, which is valid according to the Nyquist–Shannon sampling theorem. The Nyquist–Shannon sampling theorem establishes a condition for sufficient sampling rate of a discrete sequence of samples (digital information) to capture all the information from a continuous-time signal of finite bandwidth (analog information). In practice, it states that at least two samples must be captured per cycle in order to measure the frequency of a signal. The resolution of a spatially discretized compartment with respect to time follows a standard equation of a linear passive capacitor --- a capacitor's charge $V$ at a given instante $t$ is given by $V = V_0 e^{-t / \tau}$ where the time constant $\tau = RC$ and $V_0$, $R$ and $C$ denote the initial charge, the capacitors resistance and capacitance, respectively --- of the form:

$$
\frac{dV_{nm}}{dt} = -k_{nm} V_{nm}
\label{eq_310_neuronbook}
$$

where its analytic solution is 

$$
V_{nm} = V_{0_{nm}} e^{-k_{nm}t}
$$

and the rate constant $k_{nm}$ is the inverse of the membrane time constant $\tau _m = c/g$ for the point $n$ of $m$ points per compartment. The temporal discretization for a fixed time step approach follows the basic approximation principle for time stepping between two intervals:

$$
\frac{d V}{d t} \approx \frac{V(t + \Delta t) - V(t)}{ \Delta t}
$$

[euler]: https://en.wikipedia.org/wiki/Euler_method
[cn]: https://en.wikipedia.org/wiki/Crank%E2%80%93Nicolson_method

Fixed time step interpolation is typically performed with either [Euler][cn] or [Crank-Nicholson][cn] methods. The explicit Forward Euler provides first-order accuracy and that the computational time step must never be more than twice the smallest constant on the system. As a reminder, a numerical solution to a differential equation is said to be of n$^{th}$-order if the error $E$ is proportional to the $n^{th}$ power of the step-size $h$: $E(h)=Ch^n$. The gap between the analytic and numerically solved solution is particularly high when the system's voltage is changing rapidly, making it unstable. This has been described on detail in [The NEURON book][neuron-book]. For an extremely small time step $\Delta t=0.0001 ms$ the system almost achieves equilibrium with the added cost of very high computation. Therefore Forward Euler is not applied in NEURON The Backward Euler method computes the solution of a set of nonlinear simultaneous equations at each step, therefore requiring a step size as large as possible to compensate for the extra work. For *reasonable* values of $\Delta t$ it produces fast simulations that are very often generally correct, explained by the fact that tightly coupled compartments do not generate large error oscillations and approach the system's equilibrium quickly. As its numerical error is proportional to $\Delta t$ the Backward Euler does not deal with non-linearities. A function or map $f(x)$ is said to be linear it satisfies the properties of superposition ($f(x+y) = f(x) + f(y)$) and homogeneity ($ f (\alpha x) = \alpha f (x)$). The equation(s) of a non-linear system cannot be written as a linear combination of its unknown variables or functions.

The accuracy of the Backward Euler is improved by the [Crank-Nicholson][cn] method, that provides a local error proportional to the square of the step size. As $\Delta t$ becomes large, the solution oscillates with decreasing amplitude, making it numerically stable --- see Hines et al. (1997) for a numerical resolution of passive compartments and action potential. Computation-wise, the Crank-Nicholson method  is stable and more accurate than the Backward Euler, while requiring the same number of computation steps as Backward Euler. This can be shown by the following: if we take implicit (Backward) Euler to be defined as

$$
y^{n+1}= y^n + \Delta t \text{ } A y^{n+1} = (I - \Delta t \text{ } A)^{-1} y^n 
$$ 

and explicit (Forward) Euler as:

$$
y^{n+1} = y^n + \Delta t \text{ } A y^n = (I + \Delta t \text{ } A) y^n 
$$ 

then Crank-Nicholson can be described as:

$$
y^{n+1} = y^n + \frac{1}{2}\Delta t \space{ } A y^{n+1} + \frac{1}{2}\Delta t \space{ } A y^{n} = \left( I - \frac{1}{2} \Delta t \text{ } A \right)^{-1} \left(I + \frac{1}{2} \Delta t \text{ } A \right) y^n
$$

which is 2 steps of implicit Euler, therefore requiring no extra computation for the added accuracy:

$$
2 y^{n+1} - y^n = (2(I - \Delta t \text{ } A)^{-1} - I) y^n = (I - \Delta t \space{ } A)^{-1} (I + \Delta t \space{ } A) y^n
\label{eq_crank_nicholson_steps}
$$

<div class="alert alert-warning" role="alert">
 TODO: I need to confirm this very last step.
</div>

##### Computational Implementation

NEURON's workflow for the resolution based on interleaved time stepping is defined by a set of iterations that run sequentially for every computation and communication time step of duration $\Delta t$ and $\Delta t_{comm}$ respectively as:
1. at instant $t$:
   - delivery of all (queued) spikes delivery for the subsequent iteration;
   - channels run their mechanisms to deliver current;
   - synaptic currents are delivered to the compartments;
2. at instant $t + \Delta t /2$:
   - Linear algebra resolution via (Hines) solver;
3. at instant $t + \Delta t$:
   - Update channels states;
   - Update synapses states;
   - Action Potential threshold detection;
4. if $t$ is a multiple of $\Delta t_{comm}$:
   - Perform collective call to send/recv spikes;
   - Place all received spikes on spikes queue;
5. Jump to next time step: $t \leftarrow t + \Delta t$;

$\Delta t$ is defined by the fastest mechanisms requiring no data exchange between compute nodes (normally the fastest $Na^+$ channels).   $\Delta t_{comm}$ is fixed as the fastest mechanism requiring communication among compute nodes --- typically the minimal synaptic delay across all synapses, or (when applicable) the update frequency for extra-cellular mechanisms or gap junctions.

NEURON's implementation of fixed time step integrates the equation over the $\Delta t$ step by calling all the *BREAKPOINT* blocks of the models at $t + \Delta t /2$ twice with $v$ and $v+.001$ in order to compute the current and contribution of conductance from each mechanism, in order to form the matrix conductance*voltage=current. The matrix is then solved for $v(t+ \Delta t)$. Lastly, in line with the staggered time stepping scheme, the *SOLVE* statement of the models (within the *BREAKPOINT* block) is executed with $t+ \Delta t$ and the new $v$ in order to integrate those states, from the new $t-\Delta t/2$ to new $t+\Delta t/2$.

A strong argument supporting a fixed time step approach is the possibility of data vectorization. At every time step, several mechanisms run the same operations with different inputs, therefore allowing for a data layout in memory that takes advantage of hyperthreading or other vectorization modules on modern architectures.


##### Handling of non linearity

Although nonlinear equations generally require an iterative resolution in order to maintain second order correcteness, the HH membrane properties allow the cable equation to be described linearly and solved without iterations, while keeping second-order error accuracy. NEURON's variant of Crank-Nicholson with staggered time stepping applies the Strang splitting method to calculate the values of the compartment voltage and Hodgkin-Huxley gating states on interleaved time intervals. Therefore it converts a large non-linear system into two linear systems and the problem has now second-order accuracy, with a computation cost *almost* identical to the Backward Euler method. As a side note, a direct solution of voltage equation using a linearized membrane current $I(V,t)=g(V-E)$ at a time step $t \rightarrow t + \Delta t$ is possible if the conductance $g$ and reversal potential $E$ have second-order accuracy at time $t+\Delta t/2$. as detailed in [The NEURON book][neuron-book]. Since the conduction of HH-type channels is given by a function of state variables $n$ ($K^+$), $m$ and $h$ ($Na^+$), this second-order accuracy at $t+\Delta t/2$ is achieved by performing a calculation with a time step offset of $\Delta t/2$ from the current voltage time step. In brief, to calculate the second-order accurate integration between $t-\Delta t/2$ and $t+\Delta t/2$ we only need the second-order correct value of voltage-dependent rates at the instant $t$.

$\Delta t$ is defined by the fastest mechanisms requiring no data exchange between compute nodes (normally the fastest $Na^+$ channels).   $\Delta t_{comm}$ is fixed as the fastest mechanism requiring communication among compute nodes --- typically the minimal synaptic delay across all synapses, or (when applicable) the update frequency for extra-cellular mechanisms or gap junctions.

An important message here is that this resolution as a system is linear equations is only possible for *simple* models. Neuron models with non-linear state variable OPDEs and correlation between states require a fully-implicit resolution, and is possible with Backward Euler with Newton iterations. This is a common use case, and underlies the resolution of extra-cellular mechanisms, capacitors between nodes, and other mechanisms that connect neurons' evolution over time.

In the following section we will present a fully-implicit method based on variable-order variable timestepping.

### Adaptive Time Stepping Interpolation
 
[lytton2005independent]: https://aip.scitation.org/doi/pdf/10.1063/1.4822377
[sundials]: https://computation.llnl.gov/projects/sundials

The main principle supporting the adaptive time stepping is its particular efficiency due to a large step size upon large and low-varying states e.g. interspiking intervals. NEURON's adaptive algorithm, [CVODE][lytton2005independent] uses separate variable-step integrators for individual neurons in the network. CVODE is used by NEURON as an interface to a collection of methods built on top of existing CVODE and IDA (Integrator for Differential-Algebraic problems) integrators of the [SUNDIALS package][sundials], a SUite of Nonlinear and DIfferential/ALgebraic Equation Solvers. NEURON employs the Backward Differentiation Formula. The backward differentiation formula is a family of implicit multi-step methods for the numerical integration of (stiff) ordinary differential equations. For a given function and time, the BDF approximates its derivative using information from previously computed times, thus increasing the accuracy of the approximation. BDF methods are implicit thus require the computing the solution of nonlinear equations at every step. The BDF multistep method is suitable for stiff problems that define neuronal activity. The main idea behind a stiff equation is that the equation includes terms that may lead to rapid variation of the solution, therefore causing certain numerical methods for solving the equation numerically unstable, unless the step size is taken to be *extremely* small. CVODES is a solver for the Initial Value Problem --- the IVP is an ODE together with an initial condition of the unknown function at a given point in the domain of the solution --- that can be written abstractly as:

$$
\dot{y} = f(t,y) \text{, } y(t_0) = y_0 \text{ where } y \in \mathbb{R}^N
$$

where $\dot{y}$ denotes $\frac{dy}{dt}$. The solution $y$ is differentiated using past (backward) values, ie $$\dot{y}_n$$ is the linear combination of $$y_n$$, $$y_{n-1}$$, $$y_{n-2}, ...$$.  $$y$$ is the array of all state variables, in our case the $$dV_n/dt$$ equation of all compartments, and $$dx_i/dt$$ where $x_i$ is a given state variable of an ion $o$. The variable-step BDF with $q$-order can be written as:

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

<p align="center"><img width="30%" height="30%" src="/assets/2016-Numerical-Methods-Neuron/local_variable_timestep_six_cell_example.png"><br/><small>source: [The NEURON book][neuron-book]</small></p>

##### Previous Experiments

The performance of *lvardt*'s integration model was tested [(Lytton et al. 2015)][lytton2005independent] on a connected homogeneous network of 100 inhibitory neurons on a single compute node. Due to their inhibitory features, the circuit presents rapid synchronization through mutual inhibition, where variable time steps integrators are likely to encounter problems. A fixed time step of $0.0025 ms$ yielded *closely comparable* results to those of *lvardt* with absolute error tolerance (atol) of $10^{-3}$ or $10^{-5}$. The event deliveries and the associated interpolations took about $12\%$ of total execution time. When using *lvardt*'s global time stepping, the author claims the large amount of events lead to an increase of execution time in the order of 60-fold. Another experiment featured a Thalamocortical simulation, featuring four cell types (cortical pyramidal neurons and interneurons, thalamocortical
cells, and thalamic reticular neurons). The two thalamic cell types produced bursts with prolonged interburst intervals, therefore very suitable for the use of the *lvardt* algorithm. A simulation of $150 ms$ of simulation time, aiming at high-precision with an error tolerance (atol) of $10^{-6}$ for *lvardt* and $\Delta t = 10^{-4} ms$ for the fixed time step, yielded an execution time of 6 minute 13 seconds and 10 hour 20 minute, respectively, a 100-fold speed-up. For a standard-precision run, with an atol of $10^{-3}$ and $\Delta t = 10^{-2}$, the simulations took 2 minutes for *lvardt* and 5 minutes 53 seconds for fixed time step, a 3-fold speed-up.

In the following picture we present the afforementioned results for the comparison of fixed and variable time step methods for the mutual-inhibition model. Execution time grows linearly with simulation time for the fixed-step integration (dashed lines). Variable-step methods have a reduction in CPU load at the onset of synchrony. The rationale behind the non-linearity of adaptive time stepping, is that *lvardt* produces time steps that jump around during the initial presynchrony phase of the simulation (fast uprising phase), and then slowly stabilizes to an alternation between large steps in the long intervals separating the population spikes and extremely short time steps during the population spike itself:

<p align="center"><img width="30%" height="30%" src="/assets/2016-Numerical-Methods-Neuron/lvardt_exec_time_comparison.png"><br/><small>source:[(Lytton et al. 2015)][lytton2005independent]</small></p>

NEURON's default error setting for CVODE is $10 \mu M$ for membrane potential and $0.1 nM$ for internal free calcium concentration, allowing a HH-based action potential to have an accuracy equivalent of fixed time step with $\Delta t = 25 \mu s$. The default maximum order for BDF is $5$, although *it is recommended* to let CVODE decide the best order.

##### Distributed CVODE executions

For parallel computing environments, the synchronization of cells spread across different computing nodes presents a bottleneck, due to the insufficient network latency for real-time message passing, and high amount of information to be actively transmitted (i.e. one per event per node). To tackle the mentioned issues, simulation is split equidistant time intervals equal to the shortest delay of any global event --- typically the minimal synaptic delay across all synapses, or (when applicable) the update frequency for extra-cellular mechanisms or gap junctions. On every time interval, all the nodes on the network perform a collective communication call and share the events to be sent on the next time frame. The downsides of such approach are that (a) no compute node can ever perform a CVODE time step that exceeds the limits of the current time frame, (b) all compute nodes must wait for the slowest to reach the end of the time frame, and (c) all compute nodes restart the CVODE at the initial instant of every time frame.

A future post will focus on fully-asynchronous execution based on variable timestepping, where we will show that we can gain a significant speed-up in time to solution if we remove communication barriers and advance solution beyond the barriers dictated by the minimum synaptic delays.
