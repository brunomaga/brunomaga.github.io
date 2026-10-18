---
layout: post
title:  "Numerical Simulation of Detailed Neuron Models"
categories: ["Computational Neuroscience"]
tags: ["Hodgkin-Huxley", "cable theory", "NEURON", "numerical methods", "simulation"]
---

Neurons are the cells of the nervous system that receive, process and transmit signals, and through these signals they govern behavior, pain, feelings, motion and most other aspects of human life. The human brain has about $$8.6 \times 10^{10}$$ neurons, connected by about $$1.5 \times 10^{14}$$ synapses. For comparison, the larva of the sea squirt *Ciona intestinalis* has 231 neurons, a mouse has about 71 million, and the African elephant has about $$2.57 \times 10^{11}$$, almost three times as many as a human. The Wikipedia [list of animals by number of neurons][number-of-neurons] gives the counts of many more animals.

The brain can be studied at many scales, as the picture below shows. At the largest scale, it is divided into regions, each with its own function (a). Each region is made of many neurons, and a neuron is a cell with a body, the soma, many branched dendrites that receive signals, and a long axon that sends signals to other neurons (b). Neurons pass their signals to each other through [synapses][synapse], where chemical messengers, the neurotransmitters, cross a small gap between two cells (c). At the smallest scale, the [ion channels][ion-channel], proteins in the membrane of the neuron, let ions such as sodium and potassium flow into and out of the cell (d). This flow of ions changes the voltage across the membrane, and this voltage is the electrical signal of the neuron.

[number-of-neurons]: https://en.wikipedia.org/wiki/List_of_animals_by_number_of_neurons
[synapse]: https://en.wikipedia.org/wiki/Synapse
[ion-channel]: https://en.wikipedia.org/wiki/Ion_channel

{: .picture}
<img width="50%" height="50%" src="{{ site.assets }}/neuron-electrical-activity/brain_scales.svg">

{: .caption}
The scales of the brain: (a) brain regions, (b) neuron, (c) synapse, and (d) ion channels in the neuron membrane. Adapted from: Mariana Villarreal (LadyofHats), Wikimedia Commons; and Blausen.com staff, [Medical gallery of Blausen Medical 2014, WikiJournal of Medicine](https://doi.org/10.15347/wjm/2014.010)

A detailed neuron model computes how the flow of ions changes this voltage over the branched shape of the neuron. Because of the complex anatomy and behavior of neurons, its equations have no analytic solution, and simulators approximate them with numerical methods.

## Levels of Detail in Brain Simulation

Brain simulators work at different levels of detail too, as the picture below shows. Each model zooms into the part of the model on its left that the dashed circle marks:

{: .picture}
<img width="99.2%" height="99.2%" src="{{ site.assets }}/neuron-electrical-activity/neuron_model_scales.svg">

{: .caption}
Levels of detail in brain simulation. Each dashed circle marks the part that the next model describes in more detail. This post covers the compartmental neuron model (in bold). In the reaction-diffusion model, the line with a flat head marks the breakdown of cAMP by PDE. Adapted from: J. M. Eppler, Introduction to NEST, First community workshop on NESTML, 2015

- **Population models** describe the average activity of large groups of neurons. A simple example is one excitatory population (E) and one inhibitory population (I) that drive each other, as in the [Wilson–Cowan model][wilson-cowan].
- **Point neuron networks** reduce each neuron to a single point, with no shape. The simplest example is the [leaky integrate-and-fire][lif-post] (LIF) neuron, first described by Lapicque in 1907, which has no ion channels: it treats the membrane as an [RC circuit][rc], and the neuron spikes when its voltage reaches a threshold.
- **Compartmental neuron models** keep the branched shape of the neuron. They cut the dendrites and the axon into short compartments, and each compartment has its own voltage and its own ion channels, and exchanges current with its neighbors. The rest of the post covers this level.
- **Compartmental membrane models** describe small parts of the membrane in more detail, such as the [dendritic spines][spine] that receive synapses.
- **Reaction–diffusion models** describe the chemical reactions inside the cell, and how molecules such as calcium spread through it. The picture shows a chain of such reactions: calcium ions bind to the protein calmodulin (CaM), which activates two enzymes, adenylyl cyclase (AC) and the kinase CaMKII. AC produces cyclic AMP (cAMP), a [second messenger][second-messenger] that activates protein kinase A (PKA), while the enzyme phosphodiesterase (PDE) breaks cAMP down.

[wilson-cowan]: https://en.wikipedia.org/wiki/Wilson%E2%80%93Cowan_model
[lif-post]: {{ site.baseurl }}{% post_url 2016-02-08-lif-brunel %}
[rc]: https://en.wikipedia.org/wiki/RC_circuit
[spine]: https://en.wikipedia.org/wiki/Dendritic_spine
[second-messenger]: https://en.wikipedia.org/wiki/Second_messenger_system

The cost of a simulation grows with the detail. A point neuron, with LIF or simple Hodgkin–Huxley dynamics, has one to a few state variables, and all neurons of a type run the same few equations, so point neurons suit vector ([SIMD][simd]) architectures. Compartmental models are harder to simulate. A neuron with $$N$$ compartments has $$O(N)$$ state variables, the voltage and the states of the many types of ion channels of each compartment, and each compartment depends on its neighbors: their voltages are coupled, so every time step solves one linear system per neuron.

[simd]: https://en.wikipedia.org/wiki/Single_instruction,_multiple_data

## Cable Theory and the Hodgkin-Huxley Model

A compartmental model rests on two models. [Cable theory][cable-theory] describes how current flows along a neurite, i.e. a branch of a dendrite or of the axon, and through its membrane. The [Hodgkin–Huxley model][hh] (HH) describes the current through the voltage-gated ion channels of the membrane, with equations fitted to measurements on the squid giant axon ([Hodgkin and Huxley, 1952][hodgkin1952quantitative]). Together, the two models disregard several features of the living cell, and so reduce the propagation and interaction of electrical signals in a nerve cell, a problem in three dimensions, to a problem in one dimension, along cables. Cable theory models each neurite as a cylinder, with four assumptions:
- the membrane, a thin lipid bilayer, separates the charges inside and outside the cell and acts as a capacitor, while the ion channels let a current leak through it and act as resistors in parallel with the capacitor: the membrane is a leaky capacitor;
- the cytoplasm inside the neurite is an ohmic conductor with resistivity $$R_a$$, and the extracellular space is a perfect conductor at a uniform potential;
- the neurite is long and thin, so the voltage varies much more along its axis than across it, and is taken as uniform over each cross-section;
- the diffusion of ions along the neurite is ignored, so the axial current is ohmic.

[cable-theory]: http://www.scholarpedia.org/article/Neuronal_cable_theory
[hh]: https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model
[hodgkin1952quantitative]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1392413/

With these assumptions, the shape of a neuron becomes a branched tree of connected cables, and a compartmental model cuts each cable into compartments short enough that the voltage over each of them can be taken as constant. Mechanisms in the membrane then change the potential of each compartment. Most of them are gated ion channels, which open and close to let ions flow into or out of the cell, and so depolarize (raise) or hyperpolarize (lower) the membrane potential.

Synapses are another important mechanism, since they connect the neurons of a circuit. A synapse drives the membrane potential of the receiving (postsynaptic) neuron up or down, depending on whether the sending (presynaptic) neuron is excitatory or inhibitory. When the potential crosses a threshold, the neuron fires an [action potential][action-potential], or spike, a fast and large change of the membrane potential. The threshold is lowest at the [axon initial segment][ais], the first part of the axon, so spikes usually start there. The picture below follows one spike. From rest at about $$-70$$ mV (1), a stimulus raises the potential (2) until it crosses the threshold at about $$-55$$ mV. The voltage-gated sodium channels then open, sodium ions flow into the cell, and the potential rises by about 100 mV, to $$+30$$ mV (3). Next, the sodium channels inactivate and the slower potassium channels open, so potassium ions flow out of the cell and the potential falls (4), about 1 ms after the start of the rise. Since the potassium channels close slowly, the potential undershoots the resting potential at the end of the action potential (5), before it returns to rest (6). The spike then travels along the axon to the synapses, and regenerates at every point of the axon whose potential it raises above the threshold.

[ais]: https://en.wikipedia.org/wiki/Axon#Axonal_initial_segment
[action-potential]: https://en.wikipedia.org/wiki/Action_potential

{: .picture}
<img width="41.9%" height="41.9%" src="{{ site.assets }}/neuron-electrical-activity/action_potential.svg">

{: .caption}
The voltage trajectory during an action potential (spike). Adapted from: [OpenStax, Anatomy and Physiology, Fig. 12.24](https://openstax.org/books/anatomy-and-physiology/pages/12-4-the-action-potential)

Several other mechanisms also change the membrane current: [gap junctions][gap-junction], electrical synapses that connect two cells directly; the extracellular field; and second messengers, chemical signals inside the cell. In the cable model, the complexity of all these mechanisms is reduced to a current through the membrane, most often a conductance times a driving force, as the section on ionic currents shows. A channel with a constant conductance is called a **leak**, or passive, channel.

[gap-junction]: https://en.wikipedia.org/wiki/Electrical_synapse

## Mathematical Basis

Cable theory reduces the electrical activity of a neurite to a second-order partial differential equation, the cable equation, which relates the current and the voltage along a one-dimensional cable over time and distance. The equation follows from the principle of conservation of charge, which states that electric charge can be neither created nor destroyed. So, in any region of a neurite, the axial currents $$i_a$$ (mA) that flow into the region must leave it through the membrane, as a current density $$i_m$$ (mA/cm²) over the membrane area $$A$$ (cm²):

$$
\sum i_a = \int_A i_m \, dA
$$

The picture shows this balance on a region around a branch point of a neurite: the axial currents enter the region through its four cross-sections, and the membrane current leaves it through the membrane.

{: .picture}
<img width="22.8%" height="22.8%" src="{{ site.assets }}/neuron-electrical-activity/net_current_conservation.svg">

{: .caption}
The net current entering a region is zero: the axial currents $$i_a$$ balance the membrane current $$i_m$$. Adapted from: [Hines and Carnevale (1997), The NEURON simulation environment, Fig. 3.1](https://neuron.yale.edu/neuron/static/papers/nc97/nc3p1.htm)

A compartment is short enough for $$i_m$$ to be taken as uniform over it, i.e. well approximated by its value at the center of the compartment. In the cell models of the Blue Brain Project, for example, compartments are at most 40 μm long. The integral over the membrane of compartment $$j$$, of area $$A_j$$, then becomes a product, and the balance of the compartment reads:

$$
\int_{A_j} i_m \, dA = i_{m,j} A_j = \sum_k i_{a,kj}
$$

where $$i_{a,kj}$$ is the axial current that flows into compartment $$j$$ from a neighbor compartment $$k$$. [Ohm's law][ohm] gives these currents: the current through a conductor is proportional to the voltage between its ends, and inversely proportional to its resistance. So the axial current from compartment $$k$$ is the voltage drop between the centers of the two compartments, divided by the axial resistance $$r_{jk}$$ of the path between them:

$$
i_{a,kj} = \frac{v_k - v_j}{r_{jk}}
$$

On the other side of the balance, the membrane current has two components: a capacitive current, which charges the membrane capacitance $$c_j$$ of the compartment, and the ionic current through the channels, whose density $$i_{\text{ion},j}(v_j, t)$$ depends on the voltage and on time:

$$
i_{m,j} A_j = c_j \frac{dv_j}{dt} + i_{\text{ion},j}(v_j, t) \, A_j
$$

Putting the three equations together gives one ordinary differential equation (ODE) per compartment:

$$
c_j \frac{dv_j}{dt} + i_{\text{ion},j}(v_j, t) \, A_j = \sum_k \frac{v_k - v_j}{r_{jk}}
\label{eq_compartment}
$$

An injected current, such as the current of an electrode, adds a term to the right-hand side. A neuron with $$N$$ compartments thus has $$N$$ coupled ODEs for the voltages, plus the ODEs of its channels.

[ohm]: https://en.wikipedia.org/wiki/Ohm%27s_law

In the special case of an unbranched cable of diameter $$d$$ with constant parameters, cut into compartments of length $$\Delta x$$, compartment $$j$$ has only two neighbors, $$j-1$$ and $$j+1$$. Its membrane area is $$A_j = \pi d \Delta x$$, so its capacitance is $$c_j = C_m \pi d \Delta x$$, where $$C_m$$ is the specific membrane capacitance, about 1 μF/cm². The axial resistance to each neighbor is the resistivity times the length of the path, $$\Delta x$$, divided by the area of the cross-section, $$\pi d^2 / 4$$, i.e. $$r = 4 R_a \Delta x / (\pi d^2)$$. Replacing these terms in equation \eqref{eq_compartment} and dividing both sides by the membrane area $$\pi d \Delta x$$ gives:

$$
C_m \frac{dv_j}{dt} + i_{\text{ion},j} = \frac{d}{4 R_a} \frac{v_{j+1} - 2 v_j + v_{j-1}}{\Delta x^2}
$$

The fraction on the right has a familiar form. When $$\Delta x$$ is small, the [finite differences][finite-difference] $$(v_{j+1} - v_j) / \Delta x$$ and $$(v_j - v_{j-1}) / \Delta x$$ approximate the first derivative $$\partial v / \partial x$$ halfway between the centers of the compartments, at $$x_j + \Delta x / 2$$ and $$x_j - \Delta x / 2$$, where $$x_j$$ is the center of compartment $$j$$. The difference of these two derivatives, divided by the distance $$\Delta x$$ between them, approximates the second derivative at $$x_j$$:

$$
\frac{\partial^2 v}{\partial x^2} \approx \left( \frac{v_{j+1} - v_j}{\Delta x} - \frac{v_j - v_{j-1}}{\Delta x} \right) \frac{1}{\Delta x} = \frac{v_{j+1} - 2 v_j + v_{j-1}}{\Delta x^2}
$$

with an error of order $$\Delta x^2$$. As $$\Delta x \rightarrow 0$$, the equations of the compartments thus converge to the **cable equation**:

$$
C_m \frac{\partial v}{\partial t} + i_{\text{ion}} = \frac{d}{4 R_a} \frac{\partial^2 v}{\partial x^2}
$$

It is a [reaction–diffusion equation][reaction-diffusion]: the axial current spreads the voltage along the neurite, as heat spreads in a rod, and the ionic current is the reaction term.

[finite-difference]: https://en.wikipedia.org/wiki/Finite_difference
[reaction-diffusion]: https://en.wikipedia.org/wiki/Reaction%E2%80%93diffusion_system

The equation becomes clearer for a passive membrane, where the ionic current is only a leak through the specific membrane resistance $$R_m$$. By Ohm's law, $$i_{\text{ion}} = v / R_m$$, with $$v$$ measured from the resting potential, and multiplying both sides by $$R_m$$ gives:

$$
\tau_m \frac{\partial v}{\partial t} + v = \lambda^2 \frac{\partial^2 v}{\partial x^2}
\quad \text{with} \quad
\tau_m = R_m C_m
\quad \text{and} \quad
\lambda = \frac{1}{2} \sqrt{\frac{d R_m}{R_a}}
$$

The two constants have a physical meaning. The **membrane time constant** $$\tau_m$$ sets how fast the voltage of an isolated compartment decays to rest, and the **length constant** $$\lambda$$ sets how far a steady voltage spreads along the neurite: in a long cable, it decays as $$e^{-x/\lambda}$$. A dendrite with $$d = 1$$ μm, $$R_m = 20$$ kΩ·cm², $$R_a = 100$$ Ω·cm and $$C_m = 1$$ μF/cm², for example, has $$\tau_m = 20$$ ms and $$\lambda = 0.7$$ mm. Measured in these units, with the dimensionless time $$\tilde{t} = t / \tau_m$$ and position $$\tilde{x} = x / \lambda$$, the equation loses all its parameters:

$$
\frac{\partial v}{\partial \tilde{t}} + v = \frac{\partial^2 v}{\partial \tilde{x}^2}
$$

So, for given boundary conditions, a passive cable of length $$L$$ is fully described by its length in units of the length constant, $$L / \lambda$$, called its electrotonic length. The passive cable equation is also linear, so it obeys the [superposition principle][superposition]: the response at a given place and time to several stimuli, such as currents injected at different points, is the sum of the responses to each stimulus alone. This is why the voltage of a passive cable can be written as a sum of independent modes, as the section on spatial discretization shows. The HH channels, however, make $$i_{\text{ion}}$$ a nonlinear function of $$v$$, and the superposition principle no longer holds.

[superposition]: https://en.wikipedia.org/wiki/Superposition_principle

### Branching Points and Boundary Conditions

Real neurons branch, so the equations above need two more pieces: the coupling at the branch points, and a condition at the ends of the branches. In a tree, every compartment $$j$$ except the root has one parent $$p_j$$, and zero or more children, the compartments $$k$$ with $$p_k = j$$. The sum over the neighbors in equation \eqref{eq_compartment} then splits into one term for the parent and one term per child:

$$
c_j \frac{dv_j}{dt} + i_{\text{ion},j} A_j = \frac{v_{p_j} - v_j}{r_j} + \sum_{k \, : \, p_k = j} \frac{v_k - v_j}{r_k}
\label{eq_tree}
$$

where $$r_j$$ is the axial resistance between compartment $$j$$ and its parent. The root, usually the soma, has no parent term. Branch points are modeled as idealized nodes with axial current but no membrane, i.e. with $$A_j = 0$$ and so $$c_j = 0$$, which leaves only the axial currents in their equation: these currents sum to zero, as [Kirchhoff's current law][kirchhoff], a consequence of the conservation of charge, states for any junction. A free end of a neurite, on the other hand, is sealed: no axial current leaves it. This is a [Neumann boundary condition][neumann], which fixes the derivative of the solution normal to the boundary, here $$\partial v / \partial x = 0$$ at the end of the cable. In equation \eqref{eq_tree}, the last compartment of a branch has no child term. The Hines solver, later in the post, relies on this parent–child structure.

[kirchhoff]: https://en.wikipedia.org/wiki/Kirchhoff%27s_circuit_laws
[neumann]: https://en.wikipedia.org/wiki/Neumann_boundary_condition

## Ionic Currents

The ionic current $$i_{\text{ion}}$$ is where the biology enters the equations. Each channel type is characterized by its conductance, the inverse of its resistance, and by gating variables that describe the probability that its gates are open. Hodgkin and Huxley described the membrane of the squid giant axon with three channel types, sodium, potassium and leak, and the ionic current:

$$
i_{\text{ion}} = \bar{g}_{\text{Na}} m^3 h \, (v - E_{\text{Na}}) + \bar{g}_{\text{K}} n^4 (v - E_{\text{K}}) + g_{\text{L}} (v - E_{\text{L}})
$$

where the $$\bar{g}$$ are the maximum conductances, the $$E$$ are the reversal potentials, described below, and $$m$$, $$h$$ and $$n$$ are the gating variables, between 0 and 1. A sodium channel is open when its three activation gates $$m$$ and its inactivation gate $$h$$ are open, and a potassium channel when its four gates $$n$$ are open. The leak conductance $$g_{\text{L}}$$ is constant, and lumps together all other voltage-independent conductances. Each gating variable $$w$$ follows an ODE that is linear in $$w$$, with rates $$\alpha_w(v)$$ and $$\beta_w(v)$$ that depend on the voltage:

$$
\frac{dw}{dt} = \alpha_w(v) \, (1 - w) - \beta_w(v) \, w, \quad w \in \{m, h, n\}
$$

In words, the closed gates, a fraction $$1 - w$$, open at the rate $$\alpha_w$$, and the open gates close at the rate $$\beta_w$$. The original paper gives the rate functions and the values of all parameters. An HH compartment thus needs four ODEs: one for the voltage, and one for each gating variable.

Since 1952, many more types of ion channels have been found, and [Channelpedia][channelpedia] collects models of them. In general, a compartment with the channel types $$s$$ has the ionic current:

$$
i_{\text{ion}} = \sum_s g_s \, (v - E_s)
$$

where $$E_s$$ is the reversal potential of type $$s$$, and its conductance $$g_s$$ is its maximum conductance $$\bar{g}_s$$ times a product of gating variables, such as $$g_{\text{Na}} = \bar{g}_{\text{Na}} m^3 h$$ for sodium. Together with the capacitive current, this equation describes the electrical circuit in the picture below. The membrane is a capacitor, and each channel type is a resistor in series with a battery, whose voltage is the reversal potential of that type. The resistors of the gated channels change over time, and the picture marks them with arrows, while the resistor of the leak is constant. All branches of the circuit connect the inside of the cell to the outside, so they are in parallel, and their currents add up to the total membrane current of the compartment, $$i_{m,j} A_j$$.

[channelpedia]: https://channelpedia.epfl.ch/

{: .picture}
<img width="49%" height="49%" src="{{ site.assets }}/neuron-electrical-activity/compartment_rc_circuit.svg">

{: .caption}
The electrical circuit of one compartment. The membrane is the capacitor $$C$$. Each ion channel is a resistor $$R$$ in series with a battery $$E$$, its reversal potential. Arrows mark gated channels, whose resistance changes over time. The leak $$L$$ has a constant resistance. $$I$$ is the total current through the membrane

### Reversal Potential

The [reversal potential][reversal] of an ion is the membrane potential at which the net flow of that ion through the membrane is zero. At this potential, the electric force on the ions balances their diffusion down the concentration gradient, and the current of the ion changes its direction when the potential crosses it, hence the name. The [Nernst equation][nernst] gives it from the concentrations of the ion outside and inside the cell:

$$
E = \frac{RT}{zF} \ln \frac{[\text{ion}]_{\text{out}}}{[\text{ion}]_{\text{in}}}
$$

where $$z$$ is the charge of the ion, $$T$$ the temperature, $$R = 8.314$$ J/(K·mol) the gas constant and $$F = 96485$$ C/mol the Faraday constant. At 34 °C, i.e. $$T = 307.15$$ K, $$RT/F = 26.5$$ mV. Since the natural logarithm is 2.3026 times the decimal logarithm, every tenfold ratio of concentrations then adds about 61 mV to the reversal potential of an ion with a single charge. Typical concentrations give:
- $$E_{\text{Cl}} = -84$$ mV, with $$[\text{Cl}^-]_{\text{in}} = 5$$ mM and $$[\text{Cl}^-]_{\text{out}} = 120$$ mM;
- $$E_{\text{K}} = -96$$ mV, with $$[\text{K}^+]_{\text{in}} = 150$$ mM and $$[\text{K}^+]_{\text{out}} = 4$$ mM;
- $$E_{\text{Na}} = 66$$ mV, with $$[\text{Na}^+]_{\text{in}} = 12$$ mM and $$[\text{Na}^+]_{\text{out}} = 145$$ mM;
- $$E_{\text{Ca}} = 122$$ mV, with $$[\text{Ca}^{2+}]_{\text{in}} = 0.1$$ μM and $$[\text{Ca}^{2+}]_{\text{out}} = 1$$ mM.

[reversal]: https://en.wikipedia.org/wiki/Reversal_potential
[nernst]: https://en.wikipedia.org/wiki/Nernst_equation

Since the reversal potentials of the ions differ, no single potential stops the flow of all of them. The membrane potential at which the total current of all ions is zero is the [resting potential][resting], and the [Goldman–Hodgkin–Katz equation][ghk] gives it from the permeability $$P$$ of the membrane to each ion:

$$
V_{\text{rest}} = \frac{RT}{F} \ln \frac{P_{\text{Na}} [\text{Na}^+]_{\text{out}} + P_{\text{K}} [\text{K}^+]_{\text{out}} + P_{\text{Cl}} [\text{Cl}^-]_{\text{in}}}{P_{\text{Na}} [\text{Na}^+]_{\text{in}} + P_{\text{K}} [\text{K}^+]_{\text{in}} + P_{\text{Cl}} [\text{Cl}^-]_{\text{out}}}
$$

Each concentration is weighted by the permeability to its ion, and the concentrations of chloride appear swapped, with the inside one in the numerator, because its charge is negative. At rest, the membrane is most permeable to potassium, so the resting potential lies close to $$E_{\text{K}}$$, at about $$-70$$ mV in most neurons. During a spike, the sodium permeability rises, and the potential moves toward $$E_{\text{Na}}$$.

[resting]: https://en.wikipedia.org/wiki/Resting_potential
[ghk]: https://en.wikipedia.org/wiki/Goldman_equation

## Numerical Solution in NEURON

The [NEURON][neuron] simulator, or NEURON for short, solves these equations with biologically realistic, though not infinitely detailed, models of brain mechanisms. It integrates equation \eqref{eq_tree} for all compartments, together with the ODEs of the channels ([Hines and Carnevale, 1997][hines1997neuron]; [The NEURON Book][neuron-book]): discretized in space and time, the equations become algebraic difference equations, which NEURON solves step by step. The numerical solution raises three concerns, stability, accuracy and efficiency, and two errors limit its accuracy: the spatial error of the compartments, and the time error of the integration method.

[neuron]: https://www.neuron.yale.edu/neuron/
[hines1997neuron]: https://neuron.yale.edu/neuron/static/papers/nc97/nctoc.htm
[neuron-book]: https://www.cambridge.org/core/books/neuron-book/7C8D9BD861D288E658BEB652F593F273

### Spatial Discretization

The NEURON Book analyzes both errors on a passive cable of length $$L$$ with sealed ends. By the superposition principle, its voltage is a sum of cosine modes, and each mode decays over time as the voltage of a capacitor that discharges through a resistor: exponentially, with its own time constant. Mode $$q$$ has $$q$$ half waves along the cable, and decays with the time constant $$\tau_q$$:

$$
v(x, t) = \sum_{q \geq 0} a_q \cos \left( \frac{q \pi x}{L} \right) e^{-t / \tau_q}
\quad \text{with} \quad
\frac{1}{\tau_q} = \frac{1}{\tau_m} \left( 1 + \left( \frac{q \pi \lambda}{L} \right)^2 \right)
$$

The mode $$q = 0$$ is a uniform potential along the whole cable, and decays with $$\tau_m$$, as an isolated compartment does. The modes with more half waves decay faster, because the axial current also evens out the differences of voltage along the cable.

NEURON places the $$N$$ nodes of a cable at the centers of $$N$$ segments of length $$\Delta x = L / N$$, i.e. at $$x_j = (j + 0.5) \, L / N$$ for $$0 \leq j < N$$. A single node holds only the uniform mode $$q = 0$$, and in general the discrete cable holds only the modes $$q = 0, \dots, N-1$$. This follows from the [Nyquist–Shannon sampling theorem][nyquist], which states that a signal can be captured from its samples only if it is sampled more than twice per period of its highest frequency. Mode $$q$$ has a wavelength of $$2L/q$$, and so $$2N/q$$ nodes per wavelength, more than two only for $$q < N$$. The time constants of the discrete modes are:

$$
\frac{1}{\tau_{q,N}} = \frac{1}{\tau_m} \left( 1 + \left( \frac{2 N \lambda}{L} \right)^2 \sin^2 \frac{q \pi}{2N} \right)
$$

Since $$\sin \theta \approx \theta$$ for small $$\theta$$, $$\tau_{q,N}$$ converges to $$\tau_q$$ as $$N$$ grows. The relative error of the spatial term is about $$(q \pi / 2N)^2 / 3$$: it is of second order in $$\Delta x$$, as the finite difference above, small for the modes with few half waves, and large for the modes near $$q = N$$. Numerical analysis also shows that, for the same number of nodes, nodes at the centers of the segments give a lower error than the traditional choice of nodes spread from one end of the cable to the other, at $$x_j = j \, L / (N-1)$$ for $$0 \leq j \leq N-1$$.

The modes with many half waves are also the fastest. The last mode, $$q = N-1$$, has $$1/\tau_{N-1,N} \approx (1/\tau_m)(1 + 4 \lambda^2 / \Delta x^2)$$, so finer compartments add faster modes: the time constants span from $$\tau_m$$ down to about $$\tau_m \Delta x^2 / (4 \lambda^2)$$. For the dendrite above with $$\Delta x = 10$$ μm, they span from 20 ms down to 1 μs. A system whose time constants span many orders of magnitude, as this one, is called [stiff][stiff], and its stiffness decides which integration methods are practical.

[nyquist]: https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem
[stiff]: https://en.wikipedia.org/wiki/Stiff_equation

### Time Integration

A fixed time step $$\Delta t$$ replaces the time derivative by the basic approximation for stepping between two time instants, $$dv/dt \approx (v(t + \Delta t) - v(t)) / \Delta t$$. The methods differ in the time at which they evaluate the right-hand side of the equations: at the start of the step, at its end, or as the average of both. A method is of $$n$$-th [order of accuracy][order] if its error is proportional to $$\Delta t^n$$, so halving the step divides the error of a first-order method by 2, and the error of a second-order method by 4. On a single mode, $$dy/dt = -y / \tau$$, each method multiplies $$y$$ by a constant factor per step:

| Method | Factor per step | Global error | Stability | Work per step |
|:--|:-:|:-:|:--|:--|
| [Forward Euler][euler] (explicit) | $$1 - \Delta t / \tau$$ | $$O(\Delta t)$$ | only for $$\Delta t \leq 2 \tau$$ | one product with the matrix |
| [Backward Euler][backward-euler] (implicit) | $$\dfrac{1}{1 + \Delta t / \tau}$$ | $$O(\Delta t)$$ | any $$\Delta t$$; fast modes decay | one linear solve |
| [Crank–Nicolson][cn] (implicit) | $$\dfrac{1 - \Delta t / (2 \tau)}{1 + \Delta t / (2 \tau)}$$ | $$O(\Delta t^2)$$ | any $$\Delta t$$; fast modes oscillate | one linear solve |

[order]: https://en.wikipedia.org/wiki/Order_of_accuracy
[euler]: https://en.wikipedia.org/wiki/Euler_method
[backward-euler]: https://en.wikipedia.org/wiki/Backward_Euler_method
[cn]: https://en.wikipedia.org/wiki/Crank%E2%80%93Nicolson_method

Forward Euler is explicit: it needs no linear solve, but it is stable only if $$\Delta t \leq 2 \tau$$ for every mode, so the fastest mode sets the time step. For the dendrite above, this means $$\Delta t \leq 2$$ μs, while the voltage changes on a scale of milliseconds, and for this reason NEURON does not use forward Euler for the voltage. Backward Euler is implicit: at every step, it solves a system of simultaneous equations for the new values, so its step should be as large as the accuracy allows, to pay for this extra work. It is stable for any $$\Delta t$$, and its factor tends to zero for the fast modes, so these modes decay at once and without oscillations, as they do in the real cable, where tightly coupled compartments quickly reach their equilibrium. With reasonable steps, backward Euler thus gives fast simulations that are generally correct, but its error is of first order. Crank–Nicolson is stable for any $$\Delta t$$ too, and of second order, so it is more accurate. Its factor, however, tends to $$-1$$ for the fast modes: a mode with $$\Delta t > 2 \tau$$ changes sign at every step, and a mode with $$\Delta t \gg 2 \tau$$ also decays slowly, so a large $$\Delta t$$ gives damped oscillations of large amplitude. NEURON uses backward Euler by default, and Crank–Nicolson as an option.

The two implicit methods solve a linear system at every step. On a tree, the Hines algorithm of the next section solves it in $$O(N)$$ operations, the same order as the forward Euler update. An implicit step thus costs only a small constant factor more than an explicit step, and the accuracy alone sets its $$\Delta t$$: with $$\Delta t = 25$$ μs, backward Euler takes 12.5 times fewer steps than forward Euler on the dendrite above.

For a linear system $$d\mathbf{y}/dt = J \mathbf{y}$$, with a matrix $$J$$, the three methods read:

$$
\begin{aligned}
\text{forward Euler:} \quad & \mathbf{y}(t + \Delta t) = \mathbf{y}(t) + \Delta t \, J \, \mathbf{y}(t) \\
\text{backward Euler:} \quad & \mathbf{y}(t + \Delta t) = \mathbf{y}(t) + \Delta t \, J \, \mathbf{y}(t + \Delta t) \\
\text{Crank–Nicolson:} \quad & \mathbf{y}(t + \Delta t) = \mathbf{y}(t) + \tfrac{\Delta t}{2} J \left( \mathbf{y}(t) + \mathbf{y}(t + \Delta t) \right)
\end{aligned}
$$

Solving each equation for $$\mathbf{y}(t + \Delta t)$$, with the identity matrix $$I$$, gives one step of each method:

$$
\begin{aligned}
\text{forward Euler:} \quad & \mathbf{y}(t + \Delta t) = (I + \Delta t \, J) \, \mathbf{y}(t) \\
\text{backward Euler:} \quad & \mathbf{y}(t + \Delta t) = (I - \Delta t \, J)^{-1} \, \mathbf{y}(t) \\
\text{Crank–Nicolson:} \quad & \mathbf{y}(t + \Delta t) = \left( I - \tfrac{\Delta t}{2} J \right)^{-1} \left( I + \tfrac{\Delta t}{2} J \right) \mathbf{y}(t)
\end{aligned}
$$

The implicit methods multiply by an inverse matrix, which in practice means solving a linear system. Crank–Nicolson seems to need more work than backward Euler, but it is a backward Euler step of $$\Delta t / 2$$, followed by a linear extrapolation to $$t + \Delta t$$:

$$
\mathbf{y}(t + \tfrac{\Delta t}{2}) = \left( I - \tfrac{\Delta t}{2} J \right)^{-1} \mathbf{y}(t),
\qquad
\mathbf{y}(t + \Delta t) = 2 \, \mathbf{y}(t + \tfrac{\Delta t}{2}) - \mathbf{y}(t)
$$

since $$2 \left( I - \tfrac{\Delta t}{2} J \right)^{-1} - I = \left( I - \tfrac{\Delta t}{2} J \right)^{-1} \left( I + \tfrac{\Delta t}{2} J \right)$$. The extrapolation costs one vector operation, so the added accuracy of Crank–Nicolson comes at the cost of backward Euler.

### The Hines Solver

Each implicit step solves a linear system $$H \mathbf{v} = \mathbf{b}$$, with one row per compartment, and NEURON solves it with an algorithm built for the tree shape of neurons, the Hines algorithm ([Hines, 1984][hines1984efficient]). By equation \eqref{eq_tree}, $$H_{jk} \neq 0$$ only on the diagonal and where $$j$$ and $$k$$ are parent and child. An unbranched cable gives a tridiagonal matrix, and each branch adds a pair of entries outside the three diagonals. The pictures show a dendritic tree cut into 13 compartments (left) and the nonzero entries of its matrix (right), with the diagonal in dark blue. Compartment 6 branches from compartment 2, and compartment 9 from compartment 1, so their entries lie outside the band.

{: .picture}
<img width="44%" height="44%" src="{{ site.assets }}/neuron-electrical-activity/solver_compartment_neuron.svg"> <img width="34.2%" height="34.2%" src="{{ site.assets }}/neuron-electrical-activity/solver_grid_neuron.svg">

[Gaussian elimination][gauss] takes $$O(N^3)$$ operations on a dense matrix, but Hines showed that it takes only $$O(N)$$ operations on this matrix if each compartment has a larger index than its parent, for example in depth-first order as in the picture, with the root at index 0. With this numbering, the entry $$H_{j p_j}$$ of the parent of compartment $$j$$ lies below the diagonal, and since each compartment has only one parent, every row except the first has exactly one entry below the diagonal, and every column except the first has exactly one entry above it.

The algorithm follows from the [Thomas algorithm][thomas], the Gaussian elimination for tridiagonal matrices, in three steps. On an unbranched cable numbered from the root, the parent of compartment $$j$$ is $$j - 1$$. The Thomas algorithm eliminates the entries below the diagonal from the first row to the last: for $$j = 1, \dots, N-1$$, it subtracts $$H_{j, j-1} / H_{j-1, j-1}$$ times row $$j - 1$$ from row $$j$$, and then finds the unknowns by back substitution, from the last row to the first. The elimination can run in the other direction too: for $$j = N-1, \dots, 1$$, subtracting $$H_{j-1, j} / H_{jj}$$ times row $$j$$ from row $$j - 1$$ removes the entries above the diagonal, and the substitution then runs forward, from the first row to the last. Finally, replacing the parent $$j - 1$$ by $$p_j$$ turns this backward elimination on a cable into an elimination on a tree, which runs in two loops:
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
[thomas]: https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm

When row $$j$$ eliminates its parent entry, its own children have larger indices and are already eliminated, so its only nonzero entries are $$H_{jj}$$ and $$H_{j p_j}$$. Each elimination thus changes only the diagonal entry and the right-hand side of the parent, and creates no new nonzero entry ([fill-in][fill-in]). Both loops do a constant number of operations per compartment, so the solver takes $$O(N)$$ time. It stores $$O(N)$$ values, per compartment the diagonal entry, the two off-diagonal entries, the parent index and the right-hand side, while a dense matrix would take $$O(N^2)$$ memory. A general sparse solver takes about 3 times the memory and 2 times the time of the tree solver, even on a tree, and NEURON uses one only when the connections between compartments contain loops ([NEURON source][treeset]).

[fill-in]: https://en.wikipedia.org/wiki/Sparse_matrix#Reducing_fill-in
[treeset]: https://www.neuronsimulator.org/en/9.0.0/doxygen/treeset_8cpp_source.html

The tree also allows parallelism at the level of the branches. The triangulation starts at the branches without children, the terminal branches, and in it a compartment waits only for its children, while in the substitution it waits only for its parent. So disjoint subtrees can be computed in parallel, and their contributions are added at the node of their parent compartment. In a network, the matrices of different neurons are independent, and a simulator distributes whole neurons over ranks and threads. A neuron much larger than the others then limits the load balance, and the [multisplit method][hines2008multisplit] (Hines, Markram and Schürmann, 2008) splits its tree into subtrees on different ranks, and solves the system exactly, with a small exchange of values between the ranks at every step.

[hines2008multisplit]: https://neuron.yale.edu/neuron/publications/fully-implicit-parallel-simulation-single-neurons

### Mechanisms and Events

In NEURON, any model that adds a current or a state to the compartments is a **mechanism**: an ion channel, a synapse, a gap junction, a current or voltage clamp, or the extracellular field. An **event** is an action at a given time, such as the arrival of a spike at a synapse, and NEURON's event delivery system handles the events ([Carnevale and Hines][carnevale2002efficient]). A NetCon object watches a source neuron for spikes and, after a delay, delivers a weighted event to its target mechanism.

[carnevale2002efficient]: https://www.researchgate.net/publication/267700936_EFFICIENT_DISCRETE_EVENT_SIMULATION_OF_SPIKING_NEURONS_IN_NEURON

Mechanisms are written in NMODL, a [domain-specific language][dsl] that NEURON compiles to native code ([Hines and Carnevale, 2000][hines2000expanding]). An NMODL file is made of blocks, of which the main ones are:
- `PARAMETER`: the values set by the user, such as maximum conductances, and `CONSTANT`: the values fixed in the file;
- `STATE`: the variables of the ODEs, such as gating variables;
- `BREAKPOINT`: the main computation block, which computes the currents at each step. Its `SOLVE` statement integrates the states with a given method;
- `DERIVATIVE`: the ODEs of the states, when differential equations govern them;
- `NET_RECEIVE`: the action taken when an event arrives, such as a conductance increase at a synapse;
- `PROCEDURE` and `FUNCTION`: subroutines; a `FUNCTION` returns a value.

[dsl]: https://en.wikipedia.org/wiki/Domain-specific_language
[hines2000expanding]: https://www.neuron.yale.edu/neuron/static/papers/nc2000/nmodl400.pdf

### Nonlinear Currents and Staggered Time Steps

The HH channels make the system nonlinear: the conductances depend on the gating variables, which depend on $$v$$. A nonlinear system generally needs an iterative solution at every step to keep second-order accuracy, such as [Newton][newton] iterations, with one linear solve per iteration. The HH membrane, however, allows NEURON to solve the cable equation as a linear system without iterations, and still keep second-order accuracy. The idea is that a second-order step from $$t$$ to $$t + \Delta t$$ only needs the conductance $$g_s$$ and the reversal potential $$E_s$$ of each channel type correct to second order at the midpoint of the step, $$t + \Delta t / 2$$. Since the conductances are functions of the gating variables, NEURON computes the gating variables half a step away from the voltages, with **staggered time steps** ([Hines and Carnevale, 1997][hines1997neuron]): the voltages at the times $$t, t + \Delta t, \dots$$, and the gating variables at $$t + \Delta t / 2, t + 3 \Delta t / 2, \dots$$. Each step then splits into two linear problems:
1. **voltage**: the gating variables at $$t + \Delta t / 2$$ give each channel type a current $$i_s = g_s \, (v - E_s)$$ that is linear in $$v$$. NEURON evaluates the `BREAKPOINT` block of every mechanism twice, at $$v$$ and at $$v + 0.001$$ mV, which gives its current and its conductance $$g_s = di_s/dv$$, assembles the matrix, and solves it for $$v(t + \Delta t)$$ with the Hines algorithm;
2. **gating**: $$v(t + \Delta t)$$ lies at the midpoint of the gating step, from $$t + \Delta t / 2$$ to $$t + 3 \Delta t / 2$$, so the voltage-dependent rates computed from it are correct to second order there, which is all that a second-order step needs. With $$v$$ fixed, each gating ODE is linear in its own variable and independent of the others, so the `SOLVE` statement integrates it exactly, with the NMODL method `cnexp`.

[newton]: https://en.wikipedia.org/wiki/Newton%27s_method

In this way, one large nonlinear system becomes two linear ones. With Crank–Nicolson for the voltage, both halves are of second order, and each step costs one Hines solve, almost the same as a backward Euler step. Backward Euler needs no iterations either, since its error is of first order in any case. The scheme only works for *simple* models, whose state ODEs are linear in their own state and independent of the other states, such as the HH gates. Models with nonlinear state ODEs, or with states that depend on each other, need a fully implicit method of first order, at a higher cost per step: `derivimplicit`, backward Euler with Newton iterations, for any ODE, or `sparse`, backward Euler on the linear and coupled ODEs of kinetic schemes.

### Computational Implementation

Putting the pieces together, each fixed step of NEURON, from $$t$$ to $$t + \Delta t$$, runs these stages:
1. deliver the events due in this step, such as the spikes that arrive at synapses, to the `NET_RECEIVE` blocks of their targets;
2. at $$t + \Delta t / 2$$, evaluate the currents and conductances of all mechanisms (`BREAKPOINT`), the channels and the synapses, and assemble the matrix;
3. solve the matrix for $$v(t + \Delta t)$$ with the Hines algorithm;
4. integrate the states of all mechanisms (`SOLVE`) with the new $$v$$;
5. detect the spikes: each threshold crossing at a NetCon source creates the events of its targets;
6. advance the time: $$t \leftarrow t + \Delta t$$;
7. on a parallel computer, at the end of each communication interval $$\Delta t_{\text{comm}}$$, exchange the spikes of the interval between the ranks, and place the received spikes in the event queue.

Two time scales set the two intervals. The fastest dynamics that need no data from other ranks, usually the sodium channels of the spike, set $$\Delta t$$. The fastest interaction between neurons on different ranks sets $$\Delta t_{\text{comm}}$$: a spike affects its targets only after the synaptic delay, so the ranks can integrate their own neurons independently over an interval as long as the minimum delay between neurons on different ranks, and then exchange the spikes of the interval. NEURON exchanges them with the collective call `MPI_Allgather` ([Hines et al., 2011][hines2011spikes]). The minimum delay is 0.1 ms in the Blue Brain Project model of the neocortex, so with NEURON's default $$\Delta t = 25$$ μs, an interval holds only 4 steps, while the delay of 1.5 ms of the Brunel network, in the [post on LIF neurons][lif-post], gives intervals of 60 steps. Gap junctions between neurons on different ranks are more demanding, since they need the voltage of the other neuron at every step: NEURON exchanges these voltages at every step, with `MPI_Alltoallv`, and computes the gap junction currents from them.

[hines2011spikes]: https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2011.00049/full

A strong argument for a fixed time step is data vectorization. At every step, all instances of a mechanism run the same operations on different inputs, which allows a memory layout that takes advantage of the vector units of modern processors. [CoreNEURON][coreneuron], the optimized compute engine of NEURON, stores the variables of each mechanism as a structure of arrays: the mechanism loops then vectorize on CPUs, and neighbor GPU threads read neighbor memory addresses. CoreNEURON uses 4 to 7 times less memory and 2 to 7 times less time than NEURON ([Kumbhar et al., 2019][coreneuron]), and it supports fixed steps only, since network simulations rarely use variable time steps: every synaptic event is a discontinuity. [Variable time steps]({{ site.baseurl }}{% post_url 2017-02-17-neuron-variable-timestep %}), and how they handle these discontinuities, are the topic of the next post.

[coreneuron]: https://frontiersin.org/articles/10.3389/fninf.2019.00063/full
