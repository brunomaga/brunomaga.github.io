---
layout: post
title:  "Numerical Resolution of the Electrical Activity of Detailed Neuron Models"
categories: ["Computational Neuroscience"]
tags: ["Hodgkin-Huxley", "cable theory", "NEURON", "numerical methods", "simulation"]
---

Neurons are the cells of the nervous system that process and transmit signals. The human brain has about $$8.6 \times 10^{10}$$ [neurons][number-of-neurons], connected by about $$1.5 \times 10^{14}$$ synapses. The picture shows four scales of the brain:
- (a) the regions, each with its own function;
- (b) a neuron: a cell body (soma), branched dendrites that receive signals, and an axon that sends them;
- (c) a [synapse][synapse], where chemical messengers (neurotransmitters) cross the gap between two neurons;
- (d) the [ion channels][ion-channel], proteins of the membrane that let ions such as sodium and potassium in or out of the cell. The flow of ions changes the voltage across the membrane, which is the electrical signal of the neuron.

[number-of-neurons]: https://en.wikipedia.org/wiki/List_of_animals_by_number_of_neurons
[synapse]: https://en.wikipedia.org/wiki/Synapse
[ion-channel]: https://en.wikipedia.org/wiki/Ion_channel

{: .picture}
<img width="50%" height="50%" src="{{ site.assets }}/neuron-electrical-activity/brain_scales.svg">

{: .caption}
The scales of the brain: (a) brain regions, (b) neuron, (c) synapse, and (d) ion channels in the neuron membrane. Adapted from: Mariana Villarreal (LadyofHats), Wikimedia Commons; and Blausen.com staff, Blausen Medical 2014, WikiJournal of Medicine

A detailed neuron model computes this voltage over the branched shape of the neuron. Its equations have no analytic solution, so simulators solve them numerically.

## Levels of Detail in Brain Simulation

Brain simulators model neurons at different levels of detail. In the picture, each model describes in more detail the part circled in the model on its left:

{: .picture}
<img width="100%" height="100%" src="{{ site.assets }}/neuron-electrical-activity/neuron_model_scales.svg">

{: .caption}
Levels of detail in brain simulation. Each dashed circle marks the part that the next model describes in more detail. This post covers the compartmental neuron model (in bold). Adapted from: J. M. Eppler, Introduction to NEST, First community workshop on NESTML, 2015

- **Population models** describe the mean activity of large groups of neurons, such as an excitatory population $$E$$ and an inhibitory population $$I$$ that drive each other, as in the [Wilson–Cowan model][wilson-cowan].
- **Point neuron networks** reduce each neuron to a point without shape, such as the [leaky integrate-and-fire][lif-post] (LIF) neuron: its membrane is an RC circuit, and the neuron spikes when its voltage reaches a threshold.
- **Compartmental neuron models** keep the branched shape of the neuron, cut into short compartments. Each compartment has its own voltage and ion channels, and exchanges current with its neighbors. The rest of the post covers this level.
- **Compartmental membrane models** describe small parts of the membrane in more detail, such as the [dendritic spines][spine] that receive synapses.
- **Reaction–diffusion models** describe the chemical reactions in the cell, and the diffusion of molecules such as calcium.

[wilson-cowan]: https://en.wikipedia.org/wiki/Wilson%E2%80%93Cowan_model
[lif-post]: {{ site.baseurl }}{% post_url 2016-02-08-lif-brunel %}
[spine]: https://en.wikipedia.org/wiki/Dendritic_spine

The cost grows with the detail. A point neuron has one to a few state variables, and all neurons of a type run the same equations, which suits [SIMD][simd] units. A compartmental neuron with $$N$$ compartments has $$O(N)$$ state variables: the voltage and the channel states of each compartment. The voltages of neighbor compartments are coupled, so every time step solves one linear system per neuron.

[simd]: https://en.wikipedia.org/wiki/Single_instruction,_multiple_data

## Cable Theory and the Hodgkin-Huxley Model

A compartmental model combines two models. [Cable theory][cable-theory] describes the current along a neurite, a branch of a dendrite or of the axon, and through its membrane. The [Hodgkin–Huxley model][hh] (HH) describes the current through the voltage-gated ion channels of the membrane, fitted to measurements on the squid giant axon ([Hodgkin and Huxley, 1952][hodgkin1952quantitative]).

[cable-theory]: http://www.scholarpedia.org/article/Neuronal_cable_theory
[hh]: https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model
[hodgkin1952quantitative]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1392413/

Cable theory models a neurite as a cylinder, with four assumptions:
- the membrane, a lipid bilayer, is a capacitor in parallel with the ion channels, which are resistors;
- the cytoplasm is an ohmic conductor with resistivity $$R_a$$, and the extracellular space is a perfect conductor at a uniform potential;
- the neurite is long and thin, so the voltage is uniform over a cross-section and varies only along the axis;
- the diffusion of ions along the neurite is ignored: the axial current is ohmic.

A compartmental model cuts the neurites into compartments short enough to hold one voltage each.

Synapses raise (excitatory) or lower (inhibitory) the membrane potential of the receiving neuron. When the potential at the [axon initial segment][ais], the most excitable part of the neuron, crosses a threshold, the neuron fires an [action potential][action-potential], or spike. The potential rises by about 100 mV and falls back in about 1 ms, then undershoots the resting potential before it returns to rest. The inflow of sodium ions drives the rise. The inactivation of the sodium channels and the outflow of potassium ions drive the fall. The spike then travels along the axon to the synapses.

[ais]: https://en.wikipedia.org/wiki/Axon#Axonal_initial_segment
[action-potential]: https://en.wikipedia.org/wiki/Action_potential

{: .picture}
<img width="40%" height="40%" src="{{ site.assets }}/neuron-electrical-activity/action_potential.svg">

{: .caption}
The voltage trajectory during an Action Potential (spike). Adapted from: unknown

Other mechanisms also change the membrane current: [gap junctions][gap-junction], electrical synapses that connect two cells directly; the extracellular field; and second messengers, chemical signals inside the cell. In the cable model, every mechanism enters as a current through the membrane, most often as a conductance times a driving force (below). A channel with a constant conductance is a **leak**, or passive, channel.

[gap-junction]: https://en.wikipedia.org/wiki/Electrical_synapse

## Mathematical Basis

By conservation of charge, the axial currents $$i_a$$ (mA) that flow into a region of a neurite leave it through the membrane, as a current density $$i_m$$ (mA/cm²) over the membrane area $$A$$ (cm²):

$$
\sum i_a = \int_A i_m \, dA
$$

{: .picture}
<img width="20%" height="20%" src="{{ site.assets }}/neuron-electrical-activity/net_current_conservation.svg">

A compartment is short enough for $$i_m$$ to be uniform over it, so the integral over compartment $$j$$ is $$i_{m,j} A_j$$. By Ohm's law, the axial current from a neighbor compartment $$k$$ is the voltage difference between the two compartment centers over the axial resistance $$r_{jk}$$ between them, $$(v_k - v_j)/r_{jk}$$. The membrane current is the sum of a capacitive current and the ionic current $$i_{\text{ion},j}$$ through the channels. With the membrane capacitance $$c_j$$ of the compartment:

$$
c_j \frac{dv_j}{dt} + i_{\text{ion},j}(v_j, t) = \sum_k \frac{v_k - v_j}{r_{jk}}
\label{eq_compartment}
$$

An injected current, from an electrode, adds a term to the right-hand side. A neuron with $$N$$ compartments thus has $$N$$ coupled ordinary differential equations (ODEs) for the voltages, plus the ODEs of its channels.

For an unbranched cylinder of diameter $$d$$, cut into compartments of length $$\Delta x$$, compartment $$j$$ has the neighbors $$j-1$$ and $$j+1$$, the capacitance $$c_j = C_m \pi d \Delta x$$, and the axial resistance $$r = 4 R_a \Delta x / (\pi d^2)$$ to each neighbor. $$C_m$$ is the specific membrane capacitance, about 1 μF/cm². Dividing equation \eqref{eq_compartment} by the membrane area $$\pi d \Delta x$$:

$$
C_m \frac{dv_j}{dt} + i_{\text{ion},j} = \frac{d}{4 R_a} \frac{v_{j+1} - 2 v_j + v_{j-1}}{\Delta x^2}
$$

where $$i_{\text{ion},j}$$ is now a current per unit of membrane area. The fraction on the right is the [finite-difference][finite-difference] approximation of $$\partial^2 v / \partial x^2$$, with an error of order $$\Delta x^2$$. As $$\Delta x \rightarrow 0$$, the compartments converge to the **cable equation**:

$$
C_m \frac{\partial v}{\partial t} + i_{\text{ion}} = \frac{d}{4 R_a} \frac{\partial^2 v}{\partial x^2}
$$

It is a [reaction–diffusion equation][reaction-diffusion]: the axial current spreads the voltage along the neurite, as heat spreads in a rod, and the ionic current is the reaction term.

[finite-difference]: https://en.wikipedia.org/wiki/Finite_difference
[reaction-diffusion]: https://en.wikipedia.org/wiki/Reaction%E2%80%93diffusion_system

For a passive membrane, the ionic current is a leak through the specific membrane resistance $$R_m$$, $$i_{\text{ion}} = v / R_m$$, with $$v$$ measured from the resting potential. Multiplying by $$R_m$$:

$$
\tau_m \frac{\partial v}{\partial t} + v = \lambda^2 \frac{\partial^2 v}{\partial x^2}
\quad \text{with} \quad
\tau_m = R_m C_m
\quad \text{and} \quad
\lambda = \frac{1}{2} \sqrt{\frac{d R_m}{R_a}}
$$

The **membrane time constant** $$\tau_m$$ sets how fast the voltage of an isolated compartment decays to rest. The **length constant** $$\lambda$$ sets how far a steady voltage spreads: in a long cable, it decays as $$e^{-x/\lambda}$$. A dendrite with $$d = 1$$ μm, $$R_m = 20$$ kΩ·cm², $$R_a = 100$$ Ω·cm and $$C_m = 1$$ μF/cm² has $$\tau_m = 20$$ ms and $$\lambda = 0.7$$ mm. The passive cable equation is linear: the response to several injected currents is the sum of the responses to each current. HH channels make $$i_{\text{ion}}$$ nonlinear in $$v$$.

### Branching Points and Boundary Conditions

In a tree, every compartment $$j$$ except the root has one parent $$p_j$$, and zero or more children: the compartments $$k$$ with $$p_k = j$$. The sum over the neighbors in equation \eqref{eq_compartment} splits into one term for the parent and one term per child:

$$
c_j \frac{dv_j}{dt} + i_{\text{ion},j} = \frac{v_{p_j} - v_j}{r_j} + \sum_{k \, : \, p_k = j} \frac{v_k - v_j}{r_k}
\label{eq_tree}
$$

where $$r_j$$ is the axial resistance between compartment $$j$$ and its parent. The root, usually the soma, has no parent term. Branch points are modeled as nodes without membrane, with $$c_j = 0$$ and $$i_{\text{ion},j} = 0$$, so the axial currents into a branch point sum to zero, as Kirchhoff's current law states. A free end of a neurite is sealed: no axial current leaves it, which is the [Neumann boundary condition][neumann] $$\partial v / \partial x = 0$$. In equation \eqref{eq_tree}, the last compartment of a branch has no child term.

[neumann]: https://en.wikipedia.org/wiki/Neumann_boundary_condition

## Ionic Currents

The HH model has three channel types, sodium, potassium and leak, with the ionic current:

$$
i_{\text{ion}} = \bar{g}_{\text{Na}} m^3 h \, (v - E_{\text{Na}}) + \bar{g}_{\text{K}} n^4 (v - E_{\text{K}}) + g_{\text{L}} (v - E_{\text{L}})
$$

where the $$\bar{g}$$ are the maximum conductances, the $$E$$ are the reversal potentials (below), and $$m$$, $$h$$ and $$n$$ are the gating variables: the fractions of open gates, between 0 and 1. A sodium channel is open when its three activation gates $$m$$ and its inactivation gate $$h$$ are open, and a potassium channel when its four gates $$n$$ are open. The leak conductance $$g_{\text{L}}$$ is constant, and lumps the other ions. Each gating variable $$x$$ follows an ODE that is linear in $$x$$, with the voltage-dependent rates $$\alpha_x(v)$$ and $$\beta_x(v)$$:

$$
\frac{dx}{dt} = \alpha_x(v) \, (1 - x) - \beta_x(v) \, x, \quad x \in \{m, h, n\}
$$

An HH compartment thus has four ODEs, one for $$v$$ and one per gating variable. Many more channel types are known today, and [Channelpedia][channelpedia] lists their models. A compartment with the channel types $$s$$, each with a maximum conductance $$\bar{g}_s$$, a gating term $$x_s$$ (such as $$m^3 h$$ for sodium) and a reversal potential $$E_s$$, has the ionic current:

$$
i_{\text{ion}} = \sum_s \bar{g}_s \, x_s \, (v - E_s)
$$

With the capacitive current, this is the circuit in the picture: the membrane is a capacitor, and each channel type is a resistor in series with a battery, its reversal potential.

[channelpedia]: https://channelpedia.epfl.ch/

{: .picture}
<img width="60%" height="60%" src="{{ site.assets }}/neuron-electrical-activity/compartment_rc_circuit.svg">

{: .caption}
The electrical circuit of one compartment. The membrane is the capacitor $$C$$. Each ion channel is a resistor $$R$$ in series with a battery $$E$$, its reversal potential. Arrows mark gated channels, whose resistance changes over time. The leak $$L$$ has a constant resistance. $$I$$ is the total current through the membrane

### Reversal Potential

The [reversal potential][reversal] of an ion is the membrane potential at which the net flow of that ion through the membrane is zero: the electric force on the ions balances their diffusion down the concentration gradient. The [Nernst equation][nernst] gives it from the concentrations of the ion outside and inside the cell:

$$
E = \frac{RT}{zF} \ln \frac{[\text{ion}]_{\text{out}}}{[\text{ion}]_{\text{in}}}
$$

where $$z$$ is the charge of the ion, $$T$$ the temperature, $$R = 8.314$$ J/(K·mol) the gas constant and $$F = 96485$$ C/mol the Faraday constant. At 34 °C ($$T = 307.15$$ K), $$RT/F = 26.5$$ mV, and typical concentrations give:
- $$E_{\text{Cl}} = -84$$ mV, with $$[\text{Cl}^-]_{\text{in}} = 5$$ mM and $$[\text{Cl}^-]_{\text{out}} = 120$$ mM;
- $$E_{\text{K}} = -96$$ mV, with $$[\text{K}^+]_{\text{in}} = 150$$ mM and $$[\text{K}^+]_{\text{out}} = 4$$ mM;
- $$E_{\text{Na}} = 66$$ mV, with $$[\text{Na}^+]_{\text{in}} = 12$$ mM and $$[\text{Na}^+]_{\text{out}} = 145$$ mM;
- $$E_{\text{Ca}} = 122$$ mV, with $$[\text{Ca}^{2+}]_{\text{in}} = 0.1$$ μM and $$[\text{Ca}^{2+}]_{\text{out}} = 1$$ mM.

[reversal]: https://en.wikipedia.org/wiki/Reversal_potential
[nernst]: https://en.wikipedia.org/wiki/Nernst_equation

The membrane potential at which the total current of all ions is zero is the [resting potential][resting]. With the membrane permeability $$P$$ of each ion, the [Goldman–Hodgkin–Katz equation][ghk] gives it:

$$
V_{\text{rest}} = \frac{RT}{F} \ln \frac{P_{\text{Na}} [\text{Na}^+]_{\text{out}} + P_{\text{K}} [\text{K}^+]_{\text{out}} + P_{\text{Cl}} [\text{Cl}^-]_{\text{in}}}{P_{\text{Na}} [\text{Na}^+]_{\text{in}} + P_{\text{K}} [\text{K}^+]_{\text{in}} + P_{\text{Cl}} [\text{Cl}^-]_{\text{out}}}
$$

At rest, the membrane is most permeable to potassium, so the resting potential lies close to $$E_{\text{K}}$$, at about $$-70$$ mV in most neurons. During a spike, the sodium permeability rises, and the potential moves toward $$E_{\text{Na}}$$.

[resting]: https://en.wikipedia.org/wiki/Resting_potential
[ghk]: https://en.wikipedia.org/wiki/Goldman_equation

## Numerical Resolution in NEURON

The [NEURON][neuron] simulator solves equation \eqref{eq_tree} for all compartments, together with the ODEs of the channels ([Hines and Carnevale, 1997][hines1997neuron]; [The NEURON Book][neuron-book]). Two errors limit its accuracy: the spatial error of the compartments, and the time error of the integration method.

[neuron]: https://www.neuron.yale.edu/neuron/
[hines1997neuron]: https://neuron.yale.edu/neuron/static/papers/nc97/nctoc.htm
[neuron-book]: https://www.cambridge.org/core/books/neuron-book/7C8D9BD861D288E658BEB652F593F273

### Spatial Discretization

The NEURON Book analyzes both errors on a passive cable of length $$L$$ with sealed ends. Its voltage is a sum of cosine modes. Mode $$q$$ has $$q$$ half waves along the cable, and decays with the time constant $$\tau_q$$:

$$
v(x, t) = \sum_{q \geq 0} a_q \cos \left( \frac{q \pi x}{L} \right) e^{-t / \tau_q}
\quad \text{with} \quad
\frac{1}{\tau_q} = \frac{1}{\tau_m} \left( 1 + \left( \frac{q \pi \lambda}{L} \right)^2 \right)
$$

NEURON places the $$N$$ nodes of the cable at the centers of $$N$$ segments of length $$\Delta x = L / N$$. The discrete cable holds only the modes $$q = 0, \dots, N-1$$, since a mode needs more than two nodes per wavelength, as the [Nyquist–Shannon sampling theorem][nyquist] states. The time constants of the discrete modes are:

$$
\frac{1}{\tau_{q,N}} = \frac{1}{\tau_m} \left( 1 + \left( \frac{2 N \lambda}{L} \right)^2 \sin^2 \frac{q \pi}{2N} \right)
$$

Since $$\sin y \approx y$$ for small $$y$$, $$\tau_{q,N}$$ converges to $$\tau_q$$ as $$N$$ grows. The relative error of the spatial term is about $$(q \pi / 2N)^2 / 3$$. It is of second order in $$\Delta x$$, small for the modes with few half waves, and large for the modes near $$q = N$$. For the same number of nodes, nodes at the segment centers give a lower error than nodes spread from one end of the cable to the other.

The modes with many half waves are also the fastest. The last mode, $$q = N-1$$, has $$1/\tau_{N-1,N} \approx (1/\tau_m)(1 + 4 \lambda^2 / \Delta x^2)$$. Finer compartments thus add faster modes: the time constants span from $$\tau_m$$ down to about $$\tau_m \Delta x^2 / (4 \lambda^2)$$, and the system is [stiff][stiff]. For the dendrite above with $$\Delta x = 10$$ μm, they span from 20 ms down to 1 μs.

[nyquist]: https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem
[stiff]: https://en.wikipedia.org/wiki/Stiff_equation

### Time Integration

A fixed time step $$\Delta t$$ replaces the derivative by $$dv/dt \approx (v(t + \Delta t) - v(t)) / \Delta t$$. The methods differ in the time at which they evaluate the right-hand side. On one mode, $$dy/dt = -y / \tau$$, each method multiplies $$y$$ by a constant factor per step:

| Method | Factor per step | Global error | Stability | Work per step |
|:--|:-:|:-:|:--|:--|
| [Forward Euler][euler] (explicit) | $$1 - \Delta t / \tau$$ | $$O(\Delta t)$$ | only for $$\Delta t \leq 2 \tau$$ | one product with the matrix |
| [Backward Euler][backward-euler] (implicit) | $$\dfrac{1}{1 + \Delta t / \tau}$$ | $$O(\Delta t)$$ | any $$\Delta t$$; fast modes decay | one linear solve |
| [Crank–Nicolson][cn] (implicit) | $$\dfrac{1 - \Delta t / (2 \tau)}{1 + \Delta t / (2 \tau)}$$ | $$O(\Delta t^2)$$ | any $$\Delta t$$; fast modes oscillate | one linear solve |

[euler]: https://en.wikipedia.org/wiki/Euler_method
[backward-euler]: https://en.wikipedia.org/wiki/Backward_Euler_method
[cn]: https://en.wikipedia.org/wiki/Crank%E2%80%93Nicolson_method

Forward Euler needs no linear solve, but it is stable only if $$\Delta t \leq 2 \tau$$ for every mode, so the fastest mode sets $$\Delta t$$. For the dendrite above, $$\Delta t \leq 2$$ μs, while the voltage changes on a scale of milliseconds. Backward Euler is stable for any $$\Delta t$$, and damps the fast modes without oscillations, but its error is of first order. Crank–Nicolson is stable for any $$\Delta t$$ and of second order. A mode with $$\Delta t > 2 \tau$$ changes sign at every step, and a mode with $$\Delta t \gg 2 \tau$$ decays slowly, so a large $$\Delta t$$ gives damped oscillations of large amplitude. NEURON uses backward Euler by default, and Crank–Nicolson as an option.

The two implicit methods solve a linear system at every step. On a tree, the Hines algorithm below solves it in $$O(N)$$ operations, the same order as the forward Euler update. An implicit step thus costs a small constant factor more than an explicit step, and accuracy alone sets its $$\Delta t$$: with $$\Delta t = 25$$ μs, backward Euler takes 12.5 times fewer steps than forward Euler on the dendrite above.

For a linear system $$d\mathbf{y}/dt = A \mathbf{y}$$, the three methods compute:

$$
\begin{aligned}
\text{forward Euler:} \quad & \mathbf{y}(t + \Delta t) = (I + \Delta t \, A) \, \mathbf{y}(t) \\
\text{backward Euler:} \quad & \mathbf{y}(t + \Delta t) = (I - \Delta t \, A)^{-1} \, \mathbf{y}(t) \\
\text{Crank-Nicolson:} \quad & \mathbf{y}(t + \Delta t) = \left( I - \tfrac{\Delta t}{2} A \right)^{-1} \left( I + \tfrac{\Delta t}{2} A \right) \mathbf{y}(t)
\end{aligned}
$$

Crank–Nicolson is a backward Euler step of $$\Delta t / 2$$, followed by a linear extrapolation to $$t + \Delta t$$:

$$
\mathbf{y}(t + \tfrac{\Delta t}{2}) = \left( I - \tfrac{\Delta t}{2} A \right)^{-1} \mathbf{y}(t),
\qquad
\mathbf{y}(t + \Delta t) = 2 \, \mathbf{y}(t + \tfrac{\Delta t}{2}) - \mathbf{y}(t)
$$

since $$2 \left( I - \tfrac{\Delta t}{2} A \right)^{-1} - I = \left( I - \tfrac{\Delta t}{2} A \right)^{-1} \left( I + \tfrac{\Delta t}{2} A \right)$$. The extrapolation costs one vector operation, so Crank–Nicolson costs the same as backward Euler.

### The Hines Solver

Each implicit step solves a linear system $$H \mathbf{v} = \mathbf{b}$$, with one row per compartment. By equation \eqref{eq_tree}, $$H_{jk} \neq 0$$ only on the diagonal and where $$j$$ and $$k$$ are parent and child. An unbranched cable gives a tridiagonal matrix, and each branch adds a pair of entries outside the three diagonals. The pictures show a dendritic tree cut into 13 compartments (left) and the nonzero entries of its matrix (right). Compartment 6 branches from compartment 2, and compartment 9 from compartment 1, so their entries lie outside the band.

{: .picture}
<img width="35%" height="35%" src="{{ site.assets }}/neuron-electrical-activity/solver_compartment_neuron.svg"> <img width="25%" height="25%" src="{{ site.assets }}/neuron-electrical-activity/solver_grid_neuron.svg">

[Hines (1984)][hines1984efficient] showed that [Gaussian elimination][gauss] on this matrix takes $$O(N)$$ operations if it eliminates the compartments from the leaves to the root. NEURON numbers each compartment after its parent, for example in depth-first order as in the picture, with the root at index 0. The elimination then runs in two loops:
1. **triangulation**, from the leaves to the root: for $$j = N-1, \dots, 1$$, row $$j$$ eliminates the entry $$H_{p_j j}$$ of the row of its parent:

   $$
   f = \frac{H_{p_j j}}{H_{jj}}, \qquad
   H_{p_j p_j} \leftarrow H_{p_j p_j} - f \, H_{j p_j}, \qquad
   b_{p_j} \leftarrow b_{p_j} - f \, b_j
   $$

2. **substitution**, from the root to the leaves: $$v_0 = b_0 / H_{00}$$, then for $$j = 1, \dots, N-1$$:

   $$
   v_j = \frac{b_j - H_{j p_j} v_{p_j}}{H_{jj}}
   $$

[hines1984efficient]: https://www.sciencedirect.com/science/article/pii/0020710184900084
[gauss]: https://en.wikipedia.org/wiki/Gaussian_elimination

When row $$j$$ eliminates its parent entry, its own children are already eliminated, so its only nonzeros are $$H_{jj}$$ and $$H_{j p_j}$$. Each elimination thus changes only the diagonal entry and the right-hand side of the parent, and creates no new nonzero entry ([fill-in][fill-in]). Both loops do a constant number of operations per compartment, so the solver takes $$O(N)$$ time. It stores $$O(N)$$ values: per compartment, the diagonal entry, the two off-diagonal entries, the parent index and the right-hand side. Dense Gaussian elimination takes $$O(N^3)$$ time and $$O(N^2)$$ memory. A general sparse solver takes about 3 times the memory and 2 times the time of the tree solver, even on a tree, and NEURON uses one only when the connections between compartments contain loops ([NEURON source][treeset]). For an unbranched cable, the Hines algorithm is the [Thomas algorithm][thomas] for tridiagonal matrices, with the rows eliminated from the last to the first.

[fill-in]: https://en.wikipedia.org/wiki/Sparse_matrix#Reducing_fill-in
[treeset]: https://www.neuronsimulator.org/en/9.0.0/doxygen/treeset_8cpp_source.html
[thomas]: https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm

The tree also exposes parallelism. In the triangulation, a compartment waits only for its children, and in the substitution only for its parent, so disjoint subtrees run in parallel. In a network, the matrices of different neurons are independent, and a simulator distributes whole neurons over ranks and threads. A neuron much larger than the others limits the load balance. The [multisplit method][hines2008multisplit] (Hines, Markram and Schürmann, 2008) splits its tree into subtrees on different ranks, and solves the system exactly, with a small exchange of values between the ranks at every step.

[hines2008multisplit]: https://neuron.yale.edu/neuron/publications/fully-implicit-parallel-simulation-single-neurons

### Mechanisms and Events

NEURON calls **mechanism** any model that adds a current or a state to the compartments: an ion channel, a synapse, a gap junction, a current or voltage clamp, or the extracellular field. An **event** is an action at a given time, such as the arrival of a spike at a synapse. A NetCon object watches a source neuron for spikes and, after a delay, delivers a weighted event to its target mechanism ([Carnevale and Hines][carnevale2002efficient]).

[carnevale2002efficient]: https://www.researchgate.net/publication/267700936_EFFICIENT_DISCRETE_EVENT_SIMULATION_OF_SPIKING_NEURONS_IN_NEURON

Mechanisms are written in NMODL, a [domain-specific language][dsl] that NEURON compiles to native code ([Hines and Carnevale, 2000][hines2000expanding]). An NMODL file is made of blocks:
- `PARAMETER`: the values set by the user, such as maximum conductances, and `CONSTANT`: the values fixed in the file;
- `STATE`: the variables of the ODEs, such as gating variables;
- `BREAKPOINT`: computes the currents at each step. Its `SOLVE` statement integrates the states with a given method;
- `DERIVATIVE`: the ODEs of the states;
- `NET_RECEIVE`: the action taken when an event arrives, such as a conductance increase at a synapse;
- `PROCEDURE` and `FUNCTION`: subroutines; a `FUNCTION` returns a value.

[dsl]: https://en.wikipedia.org/wiki/Domain-specific_language
[hines2000expanding]: https://www.neuron.yale.edu/neuron/static/papers/nc2000/nmodl400.pdf

### Nonlinear Currents and Staggered Time Steps

The HH channels make the system nonlinear: the conductances depend on the gating variables, which depend on $$v$$. A nonlinear system generally needs [Newton][newton] iterations at every step, one linear solve per iteration, to keep second-order accuracy. NEURON avoids the iterations with **staggered time steps** ([Hines and Carnevale, 1997][hines1997neuron]): it computes the voltages at the times $$t, t + \Delta t, \dots$$, and the gating variables half a step later, at $$t + \Delta t / 2, t + 3 \Delta t / 2, \dots$$. Each step then splits into two linear problems:
1. **voltage**: the gating variables at $$t + \Delta t / 2$$ give each channel a current $$i = g \, (v - E)$$ that is linear in $$v$$, with the conductance $$g$$ and the reversal potential $$E$$ correct to second order at the midpoint of the step. NEURON evaluates the `BREAKPOINT` block of every mechanism at $$v$$ and at $$v + 0.001$$ mV, which gives its current and its conductance $$g = di/dv$$, assembles the matrix, and solves it for $$v(t + \Delta t)$$ with the Hines algorithm;
2. **gating**: $$v(t + \Delta t)$$ lies at the midpoint of the gating step, from $$t + \Delta t / 2$$ to $$t + 3 \Delta t / 2$$. With $$v$$ fixed, each gating ODE is linear in its own variable and independent of the others, so the `SOLVE` statement integrates it exactly, with the NMODL method `cnexp`.

[newton]: https://en.wikipedia.org/wiki/Newton%27s_method

With Crank–Nicolson for the voltage, both halves are of second order, and each step costs one Hines solve, as backward Euler. Backward Euler needs no iterations either, since its error is of first order in any case. The scheme requires state ODEs that are linear in their own state and independent of the other states, such as the HH gates. Other models need first-order implicit methods: `derivimplicit`, backward Euler with Newton iterations, for any ODE, or `sparse`, backward Euler on the linear and coupled ODEs of kinetic schemes.

### Computational Implementation

Each fixed step of NEURON runs these stages:
1. deliver the events due in this step to the `NET_RECEIVE` blocks of their targets;
2. at $$t + \Delta t / 2$$, evaluate the currents and conductances of all mechanisms (`BREAKPOINT`), and assemble the matrix;
3. solve the matrix for $$v(t + \Delta t)$$ with the Hines algorithm;
4. integrate the states of all mechanisms (`SOLVE`) with the new $$v$$;
5. detect the spikes: each threshold crossing at a NetCon source creates the events of its targets;
6. advance the time: $$t \leftarrow t + \Delta t$$.

The fastest dynamics of the neurons, usually the sodium channels of the spike, set $$\Delta t$$. On a parallel computer, each rank integrates its own neurons. A spike affects its targets only after the synaptic delay, so the ranks integrate independently over an interval $$\Delta t_{\text{comm}}$$, the minimum delay between neurons on different ranks, and then exchange the spikes of the interval. NEURON exchanges them with `MPI_Allgather` ([Hines et al., 2011][hines2011spikes]). Delays are typically about 1 ms and $$\Delta t$$ is 0.1 ms or less, so the ranks communicate once every 10 steps or more. Gap junctions between neurons on different ranks need the voltage of the other neuron at every step. NEURON copies these voltages at every step, and computes the gap junction currents from the copies, which assumes a weak coupling.

[hines2011spikes]: https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2011.00049/full

A fixed step suits vector hardware: at every step, all instances of a mechanism run the same code on different data. [CoreNEURON][coreneuron], the optimized compute engine of NEURON, stores the variables of each mechanism as a structure of arrays. The mechanism loops then vectorize on CPUs, and neighbor GPU threads read neighbor memory addresses. CoreNEURON uses 4 to 7 times less memory and 2 to 7 times less time than NEURON ([Kumbhar et al., 2019][coreneuron]), and runs network simulations with fixed steps only: network simulations rarely use variable time steps, since every synaptic event is a discontinuity. [Variable time steps]({{ site.baseurl }}{% post_url 2017-02-17-neuron-variable-timestep %}) are the topic of the next post.

[coreneuron]: https://frontiersin.org/articles/10.3389/fninf.2019.00063/full
