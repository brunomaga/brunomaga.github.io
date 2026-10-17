---
layout: post
title:  "Numerical Resolution of the Electrical Activity of Detailed Neuron Models"
categories: ["Computational Neuroscience"]
tags: ["Hodgkin-Huxley", "cable theory", "NEURON", "numerical methods", "simulation"]
---

[number-of-neurons]: https://en.wikipedia.org/wiki/List_of_animals_by_number_of_neurons

Neurons are complex structures of the human body that govern behaviour, pain, feelings, motion, and most other aspects of human life. The human brain has approximately $$8.6 \times 10^{10}$$ neurons, connected by $$1.5 \times 10^{14}$$ synapses. For comparison, the larva of the sea squirt *Ciona intestinalis* has 231 neurons, mice have approximately $$71$$ million neurons, and the African elephant has almost three times as many neurons as humans, with $$25.7 \times 10^{10}$$ neurons --- see the Wikipedia entry [List of animals by number of neurons][number-of-neurons] for a longer list of neuron counts per animal.

The brain can be studied at many scales, as shown in the picture below. At the largest scale, the brain is divided into regions, each with a different role (a). Each region is made of many neurons. A neuron is a cell with a body, many branched dendrites that receive signals, and a long axon that sends signals to other neurons (b). Neurons pass signals to each other through synapses, where chemical messengers cross a small gap between two cells (c). At the smallest scale, ion channels in the neuron's membrane let sodium and potassium ions flow in and out of the cell. This flow of ions changes the voltage across the membrane, and creates the neuron's electrical signal (d). This post focuses on the electrical activity at the level of the neuron: how the flow of ions changes the membrane voltage, and how we simulate it.

{: style="text-align:center; font-size: small;"}
<img width="50%" height="50%" src="{{ site.assets }}/neuron-electrical-activity/brain_scales.svg">

{: style="text-align:center; font-size: small;"}
The scales of the brain: (a) brain regions, (b) neuron, (c) synapse, and (d) ion channels in the neuron membrane. Adapted from: Mariana Villarreal (LadyofHats), Wikimedia Commons; and Blausen.com staff, Blausen Medical 2014, WikiJournal of Medicine

Because of the complex anatomical structure and behaviour of neurons, numerical methods are used to approximate neuron models and simulate the electrical activity of neural networks.

{: style="text-align:center; font-size: small;"}
<img width="40%" height="40%" src="{{ site.assets }}/neuron-electrical-activity/neuron.svg">

{: style="text-align:center; font-size: small;"}
Adapted from: Wikipedia

### Hodgkin-Huxley Model and Cable Theory

[hodgkin1952quantitative]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1392413/
[cable-theory]: http://www.scholarpedia.org/article/Neuronal_cable_theory

The Hodgkin-Huxley (HH) model ([A. L. Hodgkin and A. F. Huxley, 1952][hodgkin1952quantitative]) relies on several assumptions that disregard various features of the living cell. It reduces the multidimensional complexity of the propagation and interaction of electrical signals in spatially extended nerve cells to a set of one-dimensional constraints that fit the [Cable Theory][cable-theory]. Cable theory is a mathematical model that describes the changes in electric current and voltage along passive neurites.

A neuron's membrane is formed by a phospholipid bilayer, a dielectric material that acts as a leaky capacitor. The cytoplasm inside the neurite (a part of a dendrite or axon) is assumed to be an excellent ohmic conductor, and the voltage within its membrane is assumed to be constant. Neurites are assumed to be long and thin, so the voltage varies much more along the long axis of the neural process than perpendicular to it. Therefore, the diffusion of ions along the membrane surface and the dynamics of current perpendicular to the membrane are ignored. The neuron morphology is simplified to a branched tree of interconnected neurites, each short enough that the voltage across it can be taken as constant. External to the neuron's morphology, certain mechanisms cause electrical and chemical reactions that affect the potential of each individual capacitor. These mechanisms are typically gating channels, which allow the influx and outflux of molecules (ions) that depolarize or hyperpolarise the membrane.

Synapses are another important mechanism. They represent the interneural connections of the circuit. Synapses drive the post-synaptic membrane potential upwards or downwards, depending on the excitatory or inhibitory properties of the pre-synaptic cell. A spike is initiated when the membrane potential reaches a given Action Potential (AP) threshold, and is characterized by a stiff change of membrane potential. The lowest AP threshold of a neuron is typically at the beginning of the axon (the Axon Initial Segment), where spikes are first initiated. Other APs may occur throughout the axonal arborization as a current propagation mechanism. Spikes are driven by a fast influx of Sodium ($$Na^+$$) ions and by a slow outflux of Potassium ($$K^+$$) ions. The chloride ($$Cl^-$$) conductance helps to bring the membrane potential back to the rest state after an activation. Several other mechanisms also exist, such as gap junctions (active electrical synapses), extracellular space mechanisms, or second messengers. In the cable approximation model, the complexity of these mechanisms is reduced to their ionic conductance. A constant conductance is called a leak (passive) conductance. A compartment in ionic equilibrium, i.e. where the influx and outflux of ions cause no variation of its potential, is said to be at its reversal potential.

{: style="text-align:center; font-size: small;"}
<img width="40%" height="40%" src="{{ site.assets }}/neuron-electrical-activity/action_potential.svg">

{: style="text-align:center; font-size: small;"}
The voltage trajectory during an Action Potential (spike). Adapted from: unknown

### Mathematical Basis

Applying cable theory to electrical signaling in neurons reduces the activity of a neuron compartment to a second-order partial differential equation. This equation governs the change of axial current with time and distance, and models the relationship between current and voltage as a one-dimensional cable:

$$
\frac{\partial V}{\partial T} + F(V) = \frac{\partial^2 V}{\partial X^2}.
\label{equation338_neuronbook}
$$

The cable equation is linear and homogeneous, so it obeys the superposition principle. This principle states that, for all linear systems, the net response at a given place and time caused by several stimuli equals the sum of the responses to each stimulus individually. More general solutions for branched neurons are obtained by summing equations of this form, with an appropriate boundary condition. Neurons are represented as interconnected neurites with individual properties. These are then discretized in time and space, which converts the problem into algebraic difference equations that can be solved iteratively with numerical methods. For any given neurite, the principle of conservation of charge requires that the sum of currents from all sources equals 0. This principle states that electric charge can neither be created nor destroyed, and that the net quantity of electric charge, given by the sum of positive and negative charges, is always conserved, i.e.

$$
\sum i_a - \int_A i_m dA = 0
$$

where $$i_a$$ is the axial current [mA] flowing through the compartment's region, and $$\int_A i_m dA$$ is the integral of the transmembrane current density $$i_m$$ [mA/cm$$^2$$] over the membrane area $$A$$ [cm$$^2$$]. The following picture illustrates this concept:

{: style="text-align:center; font-size: small;"}
<img width="20%" height="20%" src="{{ site.assets }}/neuron-electrical-activity/net_current_conservation.svg">

A common assumption is that the neuron is divided into compartments small enough that the spatially varying $$i_m$$ in any compartment $$j$$ is well approximated by its value at the center. Therefore:

$$
\int_A i_{m_j} dA_j = i_{m_j} A_j = \sum_k i_{a_{kj}}
$$

Ohm's law states that the current between two points of a conductor is directly proportional to the voltage across those two points, i.e. $$I = V / R$$. The axial current between a compartment $$j$$ and its neighbors is computed with Ohm's law: each axial current $$i_a$$ is approximated by the voltage *drop* between the centers of the compartments divided by the resistance of the path between them (the axial resistance):

$$
i_{a_{kj}} = \frac{v_k - v_j}{r_{jk}}
$$

where $$r_{jk}$$ is the axial resistance, i.e. the resistance of the path between the compartments $$k$$ and $$j$$. Finally, the total membrane current, given by the sum of the capacitive and ionic components, must equal the sum of the axial currents $$i_{a_{kj}}$$ that enter the compartment $$j$$ from each adjacent neighbor $$k$$:

$$
c_j \frac{dv_j}{dt}+i_{ion_j} (v_j, t) = i_{m_j}A_j
\label{convervation_energy_eq}
$$

where $$c_j$$ is the membrane capacitance of the compartment, $$c_j (dv_j / dt)$$ is the capacitive component, and $$i_{ion_j} (v_j, t)$$ is the current through the ion channels' conductances. Applying spatial discretization to the equation of conservation of charge leads to a set of ordinary differential equations of the form:

$$
c_j \frac{dv_j}{dt}+i_{ion_j} (v_j, t) = \sum_k \frac{v_k - v_j}{r_{jk}}.
\label{equation345_neuronbook}
$$

Equation \ref{equation345_neuronbook} omits any injected source currents, which (if present) are added to its right-hand side. In the special case of an unbranched cable with constant parameters, the axial current is delimited by two points involving compartments $$j-1$$ and $$j+1$$, i.e.

$$
c_j \frac{dv_j}{dt}+i_{ion_j} (v_j, t) = \frac{v_{j-1} - v_j}{r_{j-1,j}} + \frac{v_{j+1} - v_j}{r_{j+1,j}}
\label{equation346_neuronbook}
$$

For a compartment with length $$\Delta x$$ and diameter $$d$$, the capacitance is $$C_m \pi d \Delta x$$ and the axial resistance is $$R_a \frac{\Delta x}{\pi (d/2)^2}$$, where $$C_m$$ is the specific membrane capacitance and $$R_a$$ the cytoplasmic resistivity. Replacing these terms in equation \ref{equation346_neuronbook} leads to:

$$
C_m \frac{dv_j}{dt}+i_{ion_j} (v_j, t) = \frac{d}{4R_a} \frac{v_{j+1} - 2 v_j + v_{j-1}}{ \Delta x^2}
\label{equation347_neuronbook}
$$

If the discretization variable is small enough, i.e. $$\Delta x \rightarrow 0$$, finite differences give the first derivative at the point $$j$$ as $$\frac{v_j - v_{j-1}}{\Delta x}$$ and $$\frac{v_{j+1} - v_j}{\Delta x}$$, and the second-order correct approximation of the second derivative as

$$
\frac{\partial^2v}{\partial x^2} = \left( \frac{v_{j+1} - v_j}{\Delta x} - \frac{v_j - v_{j-1}}{\Delta x} \right) \frac{1}{\Delta x} = \frac {v_{j+1} - 2v_j + v_{j-1}}{ \Delta x^2}
\label{eq_2nd_order_finite_difference}
$$

Replacing this term in equation \ref{equation347_neuronbook} gives

$$
C_m \frac{dv_j}{dt}+i_{ion_j} (v_j, t) = \frac{d}{4R_a} \frac{\partial^2v}{\partial x^2}
\label{equation348_neuronbook}
$$

Following Ohm's law, substituting $$i = v / R_m$$ and multiplying both sides by $$R_m$$ leads to the formulation

$$
R_m C_m \frac{dv_j}{dt} + v = \frac{d R_m}{4R_a} \frac{\partial^2v}{\partial x^2}
\label{equation349_neuronbook}
$$

Substituting $$t = T \tau$$ and $$x = X \lambda$$ scales equation \ref{equation349_neuronbook} to the time and space constants $$\tau = R_m C_m$$ and $$\lambda = (1/2) \sqrt{d R_m / R_a}$$. This leads to the final formulation in the form of the Cable equation \ref{equation338_neuronbook}.

### Ionic Currents

All channels can be characterized by their resistance or by their conductance $$g = 1/R$$, and by gating variables that describe the **open probability** of a given channel. These variables generalize external events that may block an ion channel. Hodgkin and Huxley formulated the ionic components of a membrane as:

$$
\sum_k I_k = g_{Na} m^3 h (u - E_{Na}) + g_K n^4 (u - E_K) + g_L (u - E_L)
$$

where $$u$$ is the membrane potential, $$E$$ are the reversal potentials and $$m$$, $$n$$ and $$h$$ are the gating variables --- see the original HH paper for the individual ODEs and reported values. The leak, represented by $$L$$, generalizes any other voltage-independent conductance.

### Reversal Potential

The reversal potential of a given ion flux in a compartment is given by the Nernst equation

$$
E = \frac{RT}{zF} \ln \frac{\text{[ion]}_{out}}{\text{[ion]}_{in}} \approx \frac{RT}{zF} 2.3026 \log _{10} \frac{\text{[ion]}_{out}}{\text{[ion]}_{in}}
$$

where $$\text{[ion]}_{out}$$ and $$\text{[ion]}_{in}$$ are the ionic concentrations outside and inside the cell, $$T$$ is the temperature in Kelvin (assumed to be $$307.15 K$$, or $$34 ^{\circ}\mathrm{C}$$), $$z$$ is the charge of the ion, and $$R$$ and $$F$$ are the gas ($$8.3144621 J$$ $$K^{-1}$$ $$mol^{-1}$$) and Faraday ($$96485.33289(59)$$ $$C$$ $$mol^{-1}$$) constants, respectively. The typical reversal potentials of the most common ions are approximately:
- $$E_{Cl^-}=-84 mV$$ with $$[Cl^-]_{in}=5 mM$$ and $$[Cl^-]_{out}=120 mM$$;
- $$E_{K^+}=-96 mV$$ with $$[K^+]_{in}=150 mM$$ and $$[K^+]_{out}=4 mM$$; 
- $$E_{Na^+}=66 mV$$ with $$[Na^+]_{in}=12 mM$$ and $$[Na^+]_{out}=145 mM$$; and
- $$E_{Ca^{2+}}=122 mV$$ with $$[Ca^{2+}]_{in}=0.1 \mu M$$ and $$[Ca^{2+}]_{out}=1 mM$$.

The Goldman–Hodgkin–Katz voltage equation gives the resting membrane potential of a typical HH compartment with $$Na^+$$, $$K^+$$ and $$Cl^-$$ ion channels:

$$
V = \frac{RT}{F} \ln \frac{P_{Na} [Na^+]_{out} + P_{K} [K^+]_{out} + P_{Cl} [Cl^-]_{in}} {P_{Na} [Na^+]_{in} + P_{K} [K^+]_{in} + P_{Cl} [Cl^-]_{out}} \approx -60 mV
$$

### Boundary Conditions 

A neurite with a *free* end follows a Neumann boundary condition (which specifies the value of the normal derivative of the function at the boundary) of the form $$\frac{\partial v}{\partial x}=0$$, i.e. no current flows through the terminal of a cable.

### Branching Points

Branching points are modeled as *idealized* portions of the neurite that have axial current but no membrane properties. Spatial discretization reduces the spatially distributed neuron to a set of connected compartments, and results in a set of ordinary differential equations that respect Kirchhoff's current law. Kirchhoff's current law, based on the principle of conservation of electric charge, states that the sum of currents entering any junction equals the sum of currents leaving that junction.

## Numerical Resolution Methods in NEURON

[neuron]: https://www.neuron.yale.edu/neuron/

Most equations that govern brain mechanisms do not have an analytic solution. The [NEURON simulator][neuron] --- henceforth simply NEURON --- addresses such problems with biologically realistic (not infinitely detailed) models of brain mechanisms, and uses several methods for an accurate discretization of the neuronal activity.
### The Hines Solver

[hines1984efficient]: https://www.sciencedirect.com/science/article/pii/0020710184900084

Given the tree-shaped morphology of a discretized neuron in the cable model, NEURON implements an optimized linear solver for branched trees, called the Hines Algorithm ([Efficient computation of branched nerve equations, Hines, Elsevier][hines1984efficient]). Data is represented as a sparse tridiagonal matrix, and an extra vector $$p$$ holds the indices of the parent nodes (the ones immediately above in the tree structure). The main diagonal holds the membrane potential of each compartment. The contributions from a parent to its children compartments (and vice versa) are stored in the cells on the same row (column) of the upper (lower) diagonal of the matrix. Since each compartment has only one parent, there is only one cell per row on the lower diagonal, and one cell per column on the upper diagonal. The following pictures show a neuron's dendritic arborization spatially discretized based on the cable model (left) and its sparse matrix representation in the Hines algorithm (right).

{: style="text-align:center; font-size: small;"}
<img width="35%" height="35%" src="{{ site.assets }}/neuron-electrical-activity/solver_compartment_neuron.svg"> <img width="25%" height="25%" src="{{ site.assets }}/neuron-electrical-activity/solver_grid_neuron.svg">


The main rationale of the Hines algorithm is that, if the compartments are numbered so that each one has an index greater than its parent's, the computational complexity of the solver drops from the $$O(n^3)$$ of a typical Gaussian elimination to $$O(n)$$. Consider a tridiagonal matrix with vectors $$d$$ (main diagonal), $$b$$ (lower diagonal) and $$a$$ (upper diagonal), representing a single cable without branching. The following steps show that the Hines algorithm is an inverted Gaussian elimination adapted for branched structures:

- Start with the forward triangulation process on each $$cell_i$$ on row $$i$$:
  - $$cell_i \leftarrow cell_i - \frac{b_i}{d_{i-1}} row_{i-1}$$;
- Change the function variable from $$i$$ to $$i+1$$:
  - $$cell_{i+1} \leftarrow cell_{i+1} - \frac{b_{i+1}}{d_i} row_i$$;
- Replace forward by backward triangulation:
  - $$cell_{i-1} \leftarrow cell_{i-1} - \frac{a_{i-1}}{d_i} row_i$$;
- Index each off-diagonal element by its child node, so that $$a_i$$ is the element on row $$i-1$$ and column $$i$$ (the top node has no parent, so $$a_0$$ and $$b_0$$ are not used):
  - $$cell_{i-1} \leftarrow cell_{i-1} - \frac{a_i}{d_i} cell_i$$;
- Change the parents' indices from a single cable ($$i-1$$) to a branched tree structure ($$p_i$$):
  - $$cell_{p_i} \leftarrow cell_{p_i} - \frac{a_i}{d_i} cell_i$$;

The result is the backward triangulation applied by Hines. The substitution step in Hines is, in turn, replaced by a forward substitution.

An important feature of the Hines algorithm is that it allows parallelization at the junction level. The computation starts at the branches without children (*terminal* branches), so the contributions of children branches can be computed in parallel and included recursively at the parent compartment's node.

### Mechanisms and Events

[carnevale2002efficient]: https://www.researchgate.net/publication/267700936_EFFICIENT_DISCRETE_EVENT_SIMULATION_OF_SPIKING_NEURONS_IN_NEURON
[hines2000expanding]: https://www.neuron.yale.edu/neuron/static/papers/nc2000/nmodl400.pdf

In NEURON, extra-cellular activity, recording patches, voltage clamps, interneuron kinetics (such as synapses and gap junctions) and any other contribution that affects the membrane potential are introduced as **mechanisms**. A mechanism with a specific execution time is called an **event**. Events are managed by NEURON's event delivery system ([Efficient discrete event simulation of spiking neurons in NEURON, Carnevale and Hines, ResearchGate][carnevale2002efficient]). Mechanisms are written in the NMODL language ([Expanding NEURON’s Repertoire of Mechanisms with NMODL, Hines and Carnevale][hines2000expanding]). The main user-provided blocks are *BREAKPOINT* (the main computation block of the model, where states are integrated by a *SOLVE* statement), *DERIVATIVE* (assigns values to the derivatives, if states are governed by differential equations), *PROCEDURE* and *FUNCTION* (function declarations and calls), *PARAMETER* and *CONSTANT* (read-write and read-only variables, respectively) and *NET_RECEIVE* (the actions taken when the mechanism receives an event, e.g. a synaptic input).

## Fixed Time Stepping Interpolation

As shown previously, the principle of conservation of charge between two or more coupled compartments, including an external source of current injection $$I_{inj}$$, is stated by a set of equations of the form:

$$
c_j \frac{dv_j}{dt}+i_{ion_j} (v_j, t) = \sum_k \frac{v_k - v_j}{r_{jk}} + I_{inj}
\label{equation42_neuronbook}
$$

[neuron-book]: https://www.cambridge.org/core/books/neuron-book/7C8D9BD861D288E658BEB652F593F273

The numerical resolution of these equations raises concerns about stability, accuracy and efficiency. These concerns come from the errors of the spatial discretization of the cable model and from the algorithms used to solve it numerically. [The NEURON book][neuron-book] analyzes them, comparing the numerical accuracy against the spatial discretization of the Fourier cosine terms $$\cos ( \pi n x / L)$$.

NEURON performs the spatial discretization of a compartment by placing $$m$$ points at the centers of equidistant intervals of size $$ \Delta x = L/m$$, so each point is placed at $$x = (i+0.5) L/m$$, for $$0 \leq i < m$$. Numerical methods theory shows that this has higher accuracy than the traditional method, which places the $$m$$ points at positions $$x = i \,\, L/(m-1)$$ for $$0 \leq i \leq (m-1)$$. In either case, $$m$$ points are placed in a compartment, and $$m=1$$ corresponds to a spatial frequency of $$0$$, i.e. a uniform membrane potential along the entire cable. Also, the highest number of half waves that the discretized system can represent is $$n = m-1$$, which is valid according to the Nyquist–Shannon sampling theorem. This theorem gives a sampling rate sufficient for a discrete sequence of samples (digital information) to capture all the information from a continuous-time signal of finite bandwidth (analog information). In practice, it states that at least two samples per cycle are needed to measure the frequency of a signal. Over time, a spatially discretized compartment follows the standard equation of a linear passive capacitor --- the voltage $$V$$ of a capacitor at a given instant $$t$$ is $$V = V_0 e^{-t / \tau}$$, where the time constant is $$\tau = RC$$, and $$V_0$$, $$R$$ and $$C$$ denote the initial voltage and the circuit's resistance and capacitance, respectively --- of the form:

$$
\frac{dV_{nm}}{dt} = -k_{nm} V_{nm}
\label{eq_310_neuronbook}
$$

whose analytic solution is 

$$
V_{nm} = V_{0_{nm}} e^{-k_{nm}t}
$$

where $$V_{nm}$$ is the amplitude of the spatial frequency $$n$$ (the number of half waves) for $$m$$ points per compartment. Discretizing equation \ref{equation349_neuronbook} at the $$m$$ points gives the rate constant

$$
k_{nm} = \frac{1}{\tau_m} \left( 1 + \left( \frac{2 m \lambda}{L} \right)^2 \sin^2 \frac{n \pi}{2m} \right)
$$

where $$\tau _m = c/g$$ is the membrane time constant and $$L$$ is the length of the cable. For $$n=0$$ (uniform potential), $$k_{nm} = 1/\tau_m$$. Temporal discretization with a fixed time step follows the basic approximation for stepping between two time instants:

$$
\frac{d V}{d t} \approx \frac{V(t + \Delta t) - V(t)}{ \Delta t}
$$

[euler]: https://en.wikipedia.org/wiki/Euler_method
[cn]: https://en.wikipedia.org/wiki/Crank%E2%80%93Nicolson_method

Fixed time step interpolation is typically performed with either the [Euler][euler] or the [Crank-Nicolson][cn] method. The explicit Forward Euler method is first-order accurate, and its time step must never be more than twice the smallest time constant in the system. As a reminder, a numerical solution to a differential equation is of n$$^{th}$$-order if the error $$E$$ is proportional to the $$n^{th}$$ power of the step size $$h$$: $$E(h)=Ch^n$$. The gap between the analytic and the numerical solutions is particularly large when the system's voltage changes rapidly, which makes the method unstable. [The NEURON book][neuron-book] describes this in detail. With an extremely small time step of $$\Delta t=0.0001 ms$$, the system almost reaches equilibrium, at the cost of a very high computational load. For this reason, NEURON does not use Forward Euler. The Backward Euler method computes the solution of a set of nonlinear simultaneous equations at each step, so the step size should be as large as possible to compensate for the extra work. For *reasonable* values of $$\Delta t$$, it produces fast simulations that are generally correct, because tightly coupled compartments do not generate large error oscillations and quickly approach the system's equilibrium. Backward Euler is first-order accurate, i.e. its numerical error is proportional to $$\Delta t$$. For a non-linear system, the implicit equations of each step must be solved iteratively, e.g. with Newton's method. A function or map $$f(x)$$ is linear if it satisfies the properties of superposition ($$f(x+y) = f(x) + f(y)$$) and homogeneity ($$ f (\alpha x) = \alpha f (x)$$). The equation(s) of a non-linear system cannot be written as a linear combination of its unknown variables or functions.

The [Crank-Nicolson][cn] method improves the accuracy of Backward Euler, with a local error proportional to the square of the step size. As $$\Delta t$$ becomes large, the solution oscillates with decreasing amplitude, so the method remains numerically stable --- see Hines and Carnevale (1997) for a numerical resolution of passive compartments and action potentials. Computationally, Crank-Nicolson is stable and more accurate than Backward Euler, and requires the same number of computation steps. To show this, define implicit (Backward) Euler as

$$
y^{n+1}= y^n + \Delta t \,\, A y^{n+1} = (I - \Delta t \,\, A)^{-1} y^n 
$$ 

and explicit (Forward) Euler as:

$$
y^{n+1} = y^n + \Delta t \,\, A y^n = (I + \Delta t \,\, A) y^n 
$$ 

Crank-Nicolson can then be described as:

$$
y^{n+1} = y^n + \frac{1}{2}\Delta t \space{ } A y^{n+1} + \frac{1}{2}\Delta t \space{ } A y^{n} = \left( I - \frac{1}{2} \Delta t \,\, A \right)^{-1} \left(I + \frac{1}{2} \Delta t \,\, A \right) y^n
$$

This is one step of implicit Euler with half the step size, $$y^{n+1/2} = (I - \frac{1}{2} \Delta t \,\, A)^{-1} y^n$$, followed by a linear extrapolation. The added accuracy therefore requires no extra computation:

$$
y^{n+1} = 2 y^{n+1/2} - y^n = \left(2 \left(I - \frac{1}{2} \Delta t \,\, A \right)^{-1} - I \right) y^n = \left(I - \frac{1}{2} \Delta t \,\, A \right)^{-1} \left(I + \frac{1}{2} \Delta t \,\, A \right) y^n
\label{eq_crank_nicholson_steps}
$$

### Computational Implementation

NEURON's workflow for interleaved time stepping is a sequence of iterations that run for every computation and communication time step, of duration $$\Delta t$$ and $$\Delta t_{comm}$$ respectively:
1. at instant $$t$$:
   - delivery of all (queued) spikes for the subsequent iteration;
   - channels run their mechanisms to deliver current;
   - synaptic currents are delivered to the compartments;
2. at instant $$t + \Delta t /2$$:
   - Linear algebra resolution via the (Hines) solver;
3. at instant $$t + \Delta t$$:
   - Update channel states;
   - Update synapse states;
   - Action Potential threshold detection;
4. if $$t$$ is a multiple of $$\Delta t_{comm}$$:
   - Perform a collective call to send/recv spikes;
   - Place all received spikes in the spike queue;
5. Jump to next time step: $$t \leftarrow t + \Delta t$$;

$$\Delta t$$ is set by the fastest mechanisms that require no data exchange between compute nodes (normally the fastest $$Na^+$$ channels). $$\Delta t_{comm}$$ is set by the fastest mechanism that requires communication among compute nodes --- typically the minimal synaptic delay across all synapses, or (when applicable) the update frequency of extra-cellular mechanisms or gap junctions.

NEURON's fixed time step implementation integrates the equation over the $$\Delta t$$ step as follows. It calls all the *BREAKPOINT* blocks of the models at $$t + \Delta t /2$$ twice, with $$v$$ and $$v+.001$$, to compute the current and conductance contributions of each mechanism and form the matrix conductance × voltage = current. The matrix is then solved for $$v(t+ \Delta t)$$. Finally, following the staggered time stepping scheme, the *SOLVE* statement of the models (within the *BREAKPOINT* block) is executed with $$t+ \Delta t$$ and the new $$v$$ to integrate those states from the new $$t-\Delta t/2$$ to the new $$t+\Delta t/2$$.

A strong argument for a fixed time step approach is data vectorization. At every time step, several mechanisms run the same operations on different inputs. This allows a memory data layout that takes advantage of the vectorization (SIMD) units of modern architectures.


### Handling of non-linearity

Nonlinear equations generally require an iterative resolution to maintain second-order correctness. However, the HH membrane properties allow the cable equation to be described linearly and solved without iterations, while keeping second-order accuracy. NEURON's variant of Crank-Nicolson with staggered time stepping applies the Strang splitting method to compute the compartment voltage and the Hodgkin-Huxley gating states on interleaved time intervals. It therefore converts a large non-linear system into two linear systems, which gives second-order accuracy at a computational cost *almost* identical to the Backward Euler method. A direct solution of the voltage equation using a linearized membrane current $$I(V,t)=g(V-E)$$ at a time step $$t \rightarrow t + \Delta t$$ is also possible if the conductance $$g$$ and reversal potential $$E$$ have second-order accuracy at time $$t+\Delta t/2$$, as detailed in [The NEURON book][neuron-book]. Since the conductance of HH-type channels is a function of the state variables $$n$$ ($$K^+$$), $$m$$ and $$h$$ ($$Na^+$$), this second-order accuracy at $$t+\Delta t/2$$ is achieved by computing the states with a time step offset of $$\Delta t/2$$ from the voltage time step. In brief, the second-order accurate integration between $$t-\Delta t/2$$ and $$t+\Delta t/2$$ only requires the second-order correct value of voltage-dependent rates at the instant $$t$$.

This resolution as a system of linear equations is only possible for *simple* models. Neuron models with non-linear state variable ODEs and correlation between states require a fully-implicit resolution, such as Backward Euler with Newton iterations. This is a common use case. It underlies the resolution of extra-cellular mechanisms, capacitors between nodes, and other mechanisms that couple the evolution of neurons over time. In such cases, the regular Euler and Crank-Nicolson methods described previously must be used, at the cost of a computationally heavier simulation.

