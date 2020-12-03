---
layout: default
title: "Bruno Magalhaes"
---

<h1 class="mt-5" itemprop="name headline">{{ page.title | escape }}</h1>

<p class="lead"><b>Researcher for High Performance Computing, Machine Learning Engineer</b></p>

I'm an AI resident at [Microsoft Research](https://www.microsoft.com/en-us/research/lab/microsoft-research-cambridge/) in Cambridge (UK), working on probabilistic machine learning, time-sequence data and graph neural networks. Prior to this, I completed a PhD in Computational [Neuroscience](https://www.epfl.ch/education/phd/edne-neuroscience/) at the [EPFL](https://www.epfl.ch/en/) in Switzerland --- researching large-scale reconstruction and simulation of brain-inspired spiking neural networks --- and worked as a research software engineer for distributed algorithms at the [Blue Brain Project](https://www.epfl.ch/research/domains/bluebrain/). My fields of interest are Parallel and Distributed Computing, Machine Learning and Simulation. To know more about me, read my [one-page resume]({{ site.resume}}), [publications]({{ site.google_scholar }}), or [email me](mailto:{{ site.email }}?subject=Hello).


<!--
<h4 class="mt-5 mb-3">Research Interests</h4>

I'm interested in applying Machine Learning to solve big engineering problems. When such ML systems don't exist, I like to create them, or at least face the challenge of trying. There are two big questions that I faced in life and would like to solve at some point, for the benefit of all:
- How to scale the processing of machine learning systems ([Moore's Law](https://en.wikipedia.org/wiki/Moore%27s_law)) to adapt to the exponential growth of new-generation data such as bots, generative ML, etc? For this I tend to read about decentralized and federated ML;
- How to make artificial neurons learn like spiking neurons, i.e. without back-propagation, loss function, and iterations?

<h4 class="mt-5 mb-3">Education</h4>

asdsa

-->

<h4 class="mt-5 mb-3">Publications</h4>

Here's a list of my most relevant publications. In the following, I was the first author and publications were peer-reviewed, unless mentioned otherwise. Conference journals/proceedings required a presentation at the venue as well. For a more exhaustive list, see my [Google Scholar]({{ site.google_scholar }}) profile.

||||
|--- ||--- |
|2020||[Fully-Asynchronous Fully-Implicit Variable-Order Variable-Timestep Simulation of Neural Networks](https://arxiv.org/abs/1907.00670), Proc. International Conference on Computational Science, Amsterdam, Holland (ICCS 2020)|
|2019||[Asynchronous SIMD-Enabled Branch-Parallelism of Morphologically-Detailed Neuron Models](https://www.frontiersin.org/articles/10.3389/fninf.2019.00054/full), Frontiers in Neuroinformatics|
|2019||(PhD thesis) [Asynchronous Simulation of Neuronal Activity](https://infoscience.epfl.ch/record/268035?ln=en), EPFL Scientific publications|
|2019||[Fully-Asynchronous Cache-Efficient Simulation of Detailed Neural Networks](https://www.researchgate.net/publication/333664427_Fully-Asynchronous_Cache-Efficient_Simulation_of_Detailed_Neural_Networks), Proc.  International Conference on Computational Science (ICCS 2019), Faro, Portugal|
|2019||[Exploiting Implicit Flow Graph of System of ODEs to Accelerate the Simulation of Neural Networks](https://ieeexplore.ieee.org/abstract/document/8821008), Proc. International Parallel and Distributed Processing Symposium (IPDPS 2019), Rio de Janeiro, Brazil|
|2016||[An efficient parallel load-balancing strategy for orthogonal decomposition of geometrical data](http://link.springer.com/chapter/10.1007/978-3-319-41321-1_5), Proc. International Super Computing (ISC 2016), Frankfurt, Germany|
|2015||(co-author) [Reconstruction and Simulation of Neocortical Microcircuitry](http://www.cell.com/abstract/S0092-8674(15)01191-5), Cell 163, 456–492.|
|2010||(MSc final project) [GPU-enabled steady-state solution of large Markov models](http://eprints.ma.man.ac.uk/1533/), Proc. International Workshop on the Numerical Solution of Markov Chains (NSMC 2010), Williamsburg, Virginia|
|ongoing||(arXiv) Distributed Async. Execution Speeds and Scales Up Over 100x The Detection Of Contacts Between Detailed Neuron Morphologies|
|ongoing||(arXiv) Efficient Distributed Transposition of Large-Scale Multigraphs And High-Cardinality Sparse Matrices|


<h4 class="mt-5 mb-3">Posts</h4>

When time allows, I post about HPC or ML projects I was involved in, or publications and discussions I find interesting.

  <table>
  {% for post in site.posts %}
      <tr>
      <td class="align-top">
        {%- assign date_format = site.minima.date_format | default: "%Y" -%}
        {{ post.date | date: date_format }}
      </td>
      <td><span style="display:inline-block; width:0.2cm;"></span></td>
      <td class="align-top">
      <a href="{{ post.url }}">{{ post.title }}</a>
      </td>
      </tr>
  {% endfor %}
  </table>

<h4 class="mt-5 mb-3">Misc</h4>

I have been playing waterpolo for most of my life, the last 10 years with [Lausanne Waterpolo](https://lausannenatation.ch/section/waterpolo/) and [Cambridge City Waterpolo](https://www.cambridgewaterpolo.co.uk/) clubs. I am also a big fan of winter sports, particularly skiing and snowboarding, and I feel lucky to live close to many great [ski resorts in Switzerland](https://www.myswitzerland.com/en-ch/experiences/winter/). I also enjoy cooking and hope one day that my cooking skills reach very high standards.

As a general rule, I prefer not to be addressed by my academic title or surname (e.g. *"Dear Dr Bruno"* or *"Dear Mr Magalhaes"*). I believe education is about knowledge and not academic titles. So addressing me simply by my first name (*"Hi Bruno"*) is totally fine :)
