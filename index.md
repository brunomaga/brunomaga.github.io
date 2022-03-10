---
layout: default
title: "Bruno Magalhaes"
---

<h1 class="mt-5" itemprop="name headline">{{ page.title | escape }}</h1>

<p class="lead mb-4"><b>Research Engineer for Machine Learning and High Performance Computing</b></p>

<div class="row">
  <div class="col-3">
    <img src="{{site.photo}}" class="img-fluid rounded float-left" alt="my photo"/>
  </div>
  <div class="col">
<p>
Hi! I'm a machine learning (ML) research engineer with a background on high performance computing (HPC). I'm interested in solving big engineering problems using either or both of these fields, with a sweet spot for neuromorphic computing and other brain-related technology.
</p>

<p>
I work at <a href="https://www.microsoft.com/en-us/research/lab/microsoft-research-cambridge/">Microsoft Research Cambridge</a> as an AI researcher for <a href="https://www.microsoft.com/en-us/research/project/project-silica/">Project Silica</a>, where I create large computer vision models and ML pipelines for the cloud. Prior to this, I completed a PhD in Computational Neuroscience at the <a href="https://www.epfl.ch/en/">EPFL</a> in Switzerland, researching large-scale reconstruction and simulation of brain-inspired spiking neural networks. Before that, I was an HPC research engineer at the <a href="https://www.epfl.ch/research/domains/bluebrain/">Blue Brain Project</a>, working on large-scale distributed computing, storage and multicore/GPU algorithms.
</p>
    
  </div>
</div>



<ul class="nav mt-3">
  <li class="nav-item">
    <a class="btn btn-link" href="mailto:{{ site.email }}?subject=Hello" class="btn btn-link"><i class="fas fa-envelope" title="Email"></i> {{site.email}}</a>
  </li>
  <li class="nav-item">
    <a class="nav-link" href="{{ site.url }}" class="btn btn-link"><i class="fas fa-mouse-pointer" title="homepage"></i> {{ site.url }} </a>
  </li>
  <li class="nav-item">
    <a class="btn btn-link" href="https://twitter.com/{{ site.twitter_username }}" class="btn btn-link"><i class="fab fa-fw fa-twitter-square" ></i> {{ site.twitter_username }} </a>
  </li>
  <li class="nav-item">
    <a class="btn btn-link" href="https://github.com/{{ site.github_username }}" class="btn btn-link"><i class="fab fa-fw fa-github" ></i>{{ site.github_username}}</a>
  </li>
  <li class="nav-item">
    <a class="btn btn-link" href="https://www.linkedin.com/in/{{ site.linkedin_username }}" class="btn btn-link"><i class="fab fa-linkedin" ></i> {{ site.linkedin_username }}</a>
  </li>
  <li class="nav-item">
    <a class="btn btn-link" href="skype:{{ site.skype_username }}" class="btn btn-link"><i class="fab fa-skype" aria-hidden="true"></i> {{ site.skype_username }} </a>
  </li>
  <li class="nav-item">
    <a class="nav-item btn btn-link" href="{{ site.google_scholar }}" class="btn btn-link"><i class="ai ai-google-scholar ai-1x"  title="Google Scholar"></i> google scholar</a>
  </li>
  <li class="nav-item">
    <a class="nav-link btn btn-link" href="https://en.wikipedia.org/wiki/Lausanne"><i class="fa fa-home"  title="Home"></i> Lausanne, Switzerland</a>
  </li>
  <li class="nav-item">
    <a class="nav-link btn btn-link" href="https://en.wikipedia.org/wiki/Portugal"><i class="fas fa-passport"  title="Nationality"></i> Portuguese</a>
  </li>
  <li class="nav-item">
    <a class="btn btn-link" href="{{ site.resume }}"><i class="far fa-user-circle"  title="resume"></i> resume</a>
  </li>
  <!--
  <div class="noprint">
    <li class="nav-item">
      <a class="btn btn-link" href="javascript:window.print()"><i class="fas fa-download" title="download this page"></i> Download this page</a>
    </li>
  </div>
  -->
</ul>


<h4 class="mt-5 mb-3">Professional Experience</h4>

<table class="mt-3">
      <tr>
        <td style="min-width:70px"> 2019-present</td>
        <td> <b> AI Researcher at Microsoft Research, Cambridge, UK </b> </td>
      </tr>
      <tr> <td/> <td>
       I design computer vision models for object recognition and classification of information written on glass for <a href="https://www.microsoft.com/en-us/research/project/project-silica/">Project Silica</a>.  I also perform full-stack development of large data pipelines and scalable Machine Learning models on the cloud (AzureML), in order to handle the large amount of super-high resolution input data.
      </td> </tr>
      <tr> <td/> <td>
Previously, as an AI resident, I improved the CPU load balancing of email servers, based on an ML system that learnt time series from email usage patterns, using DNNs, RNNs, Encoder-Decoders, Bayesian linear regression (closed-form solution) and Bayesian neural nets (Variational Inference); 
      </td> </tr>
      <tr> <td/> <td>
I also built a recommendation system using Graph Neural Nets to learn from related nodes on on very large (trillion edges) Meetings/Documents/Users/Emails graphs;
      </td> </tr>

      <tr>
        <td style="min-width:70px"> 2015-19 </td>
        <td> <b>PhD candidate at École polytechnique fédérale de Lausanne ‐ EPFL, Switzerland</b> </td>
      </tr>
      <tr> <td/> <td>
I researched, conceptualized, implementated and published new methods for the asynchronous variable-step simulation
of detailed spiking neural networks on large networks of highly-heterogeneous compute nodes. My main contribution was the first ever fully-asynchronous execution model (with async. computation, communication and IO), demonstrated on our use case and yielding a much faster execution and a higher numerical accuracy and stability;
      </td> </tr>
      <tr> <td/> <td>
I programmed in C and C++ on top of an asynchronous runtime system with global memory addressing (HPX) and implemented all the simulation logic (fixed and variable timestep interpolation) and the underlying HPC algorithms such as distributed task scheduling, multicore parallelism, concurrency, threading, dynamic load-balancing, vectorization and cache optimizations on four distinct Intel and compute AMD clusters. 
      </td> </tr>
      <tr> <td/> <td>
Most of my work was implemented and validated on the <a href="https://neuron.yale.edu/neuron/">NEURON</a> and <a href="https://github.com/BlueBrain/CoreNeuron">CoreNeuron</a> open-source simulators, and has been executed full steam on several supercomputers with thousands of compute nodes processing terabytes of data.
      </td> </tr>
      <tr>
        <td style="min-width:70px"> 2015-18 </td>
        <td> <b>Teaching Assistant at École polytechnique fédérale de Lausanne ‐ EPFL, Switzerland</b> </td>
      </tr>
      <tr> <td/> <td>
During my PhD, I performed 400h of teaching assistant duties for the courses of <a href="https://edu.epfl.ch/coursebook/en/unsupervised-reinforcement-learning-in-neural-networks-CS-434">unsupervised and reinforcement learning</a>, <a href="https://edu.epfl.ch/coursebook/en/project-in-informatics-CS-116">project in neuroinformatics</a> and <a href="https://edu.epfl.ch/coursebook/en/in-silico-neuroscience-BIOENG-450#:~:text=%22In%20silico%20Neuroscience%22%20introduces%20masters,management%2C%20modelling%20and%20computing%20technologies.">in silico neuroscience</a>, preparing exams, coursework, and tutorials;
      </td> </tr>
      <tr>
        <td style="min-width:70px"> 2011-15 </td>
        <td> <b> Research Engineer at Blue Brain Project, EPFL, Switzerland </b> </td>
      </tr>
      <tr> <td/> <td>
Aiming at scaling up the largest ever digital reconstruction of a detailed mammal neocortex, I designed and developed several algorithms for efficient computation and storage on BlueGene/P, Bluegene/Q and SGI supercomputers. To name a few, parallel/distributed volumetric spatial decomposition, load balancing, spatial indexing, sorting, I/O, sparse matrix transpose, and graph navigation;
      </td> </tr>
      <tr> <td/> <td>
My work led to the first ever digital reconstruction of detailed brain model at the scale of the mouse brain (80M neurons), and is the underlying technology supporting the lab's landmark <a href="http://www.cell.com/abstract/S0092-8674(15)01191-5">Cell paper</a>;
      </td> </tr>

      <tr>
        <td style="min-width:70px"> 2009-11 </td>
        <td> <b> Junior IT architect at Noble Group, London, UK </b> </td>
      </tr>
      <tr> <td/> <td>
As part of an international traineeship, I did three rotational placements on different headquarters where I performed the following duties:
      </td> </tr>
      <tr> <td/> <td>
(1) Network design and configuration for a backup data centre for EU Power & Gas trading infrastructure, London, UK;
      </td> </tr>
      <tr> <td/> <td>
(2) Network configuration and infrastructure design for a port and warehouse for coffee and soy beans, Santos, Brazil;
      </td> </tr>
      <tr> <td/> <td>
(3) Implementation of a web-based software for metals and coffee trading, New York, USA;
      </td> </tr>

      <tr>
        <td style="min-width:70px"> 2007-08 </td>
        <td> <b> Analyst Programmer at MSCI real estate, London, UK </b> </td>
      </tr>
      <tr> <td/> <td>
My first full-time job, where I consolidated my knowledge of algorithms, programming and end-to-end development of software systems. I developed a web app, a windows app, and a search engine on C++, C# and ASP (.NET) that would allow efficient storage and gathering of analytics on financial data;
      </td> </tr>

</table>


<h4 class="mt-5 mb-3">Education</h4>

<table class="mt-3">

      <tr>
        <td style="min-width:70px"> 2015-19 </td>
        <td> <b>PhD Computational Neuroscience at École polytechnique fédérale de Lausanne ‐ EPFL, Switzerland</b> </td>
      </tr>
      <tr> <td/> <td>
I was hired by the <a href="https://www.epfl.ch/research/domains/bluebrain/">Blue Brain Project</a> (BBP), directed by  <a href="https://en.wikipedia.org/wiki/Henry_Markram">Henry Markram </a> (the father of the <a href="https://en.wikipedia.org/wiki/Spike-timing-dependent_plasticity#:~:text=Spike%2Dtiming%2Ddependent%20plasticity%20(,action%20potentials%20(or%20spikes).">STDP</a> plasticity model), under the supervision of <a href="https://www.epfl.ch/research/domains/bluebrain/blue-brain/people/divisionleaders/felix-schurmann/">Felix Schuerman</a> (professor at EPFL), and <a href="https://en.wikipedia.org/wiki/Thomas_Sterling_(computing)">Thomas Sterling </a>(professor at Indiana University, winner of the <a href="https://en.wikipedia.org/wiki/Gordon_Bell_Prize">Gordon Bell prize</a> and inventor of the Beowulf cluster);
      </td> </tr>
      <tr> <td/> <td>
My research aimed at combining numerical methods, distributed computing, and neuroscience to discover new ways of performing <i>better</i> (ie faster, more accurate, more stable) simulations of detailed spiking neural networks. The numerical models followed the <a href="https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model">Hodgkin-Huxley</a> model (1963 Medicine Nobel Prize) with extensions for detailed branching and further ionic channels;
      </td> </tr>
      <tr> <td/> <td>
My thesis entitled <a href="https://infoscience.epfl.ch/record/268035?ln=en">Asynchronous Simulation of Neuronal Activity</a> was nominated for the EPFL doctoral program distinction award (best 8%) and for the IBM research award for the best thesis in computational sciences. The jury was composed by <a href="https://www.bsc.es/labarta-mancho-jesus">Jesus Labarta</a> (director of Barcelona Supercomputing Center), <a href="https://www.fz-juelich.de/SharedDocs/Personen/INM/INM-6/EN/staff/Diesmann_Markus.html?nn=724620">Markus Diesmann</a> (Director of Jullich Research Center) and <a href="https://people.epfl.ch/simone.deparis">Simone Deparis</a> (Professor at the dept. of Mathematics at EPFL);  
      </td> </tr>
      <tr> <td/> <td>
As part of the doctoral program I was trained on <a href="http://isa.epfl.ch/imoniteur_ISAP/!itffichecours.htm?ww_i_matiere=2555928173&ww_x_anneeAcad=2020-2021&ww_i_section=2140391&ww_i_niveau=&ww_c_langue=en">cellular and circuit mechanisms in neuroscience</a>, <a href="https://edu.epfl.ch/coursebook/en/neuroscience-behavior-and-cognition-BIO-483">behavior and cognition in neuroscience</a>, <a href="https://edu.epfl.ch/coursebook/en/biological-modeling-of-neural-networks-BIO-465">biological modeling of neural networks</a>, <a href="https://edu.epfl.ch/coursebook/en/machine-learning-CS-433">machine learning</a>, <a href="https://edu.epfl.ch/coursebook/en/introduction-to-natural-language-processing-CS-431">natural language processing</a> and <a href="https://edu.epfl.ch/coursebook/en/statistics-for-data-science-MATH-413">statistics for data science</a>;
      </td> </tr>
      <tr> <td/> <td>
During the Summer periods of 2015, 2016 and 2017, I was a visiting researcher at the <a href="https://pti.iu.edu/centers/crest.html">Center for Research in Extreme Scale Technologies (CREST)</a> at Indiana University, working with the developers of HPX on benchmarking, profiling and finetuning the HPX runtime to our use case;
      </td> </tr>

      <tr>
        <td style="min-width:70px"> 2009 </td>
        <td> <b> MSc Advanced Computing at Imperial College London, UK </b> </td>
      </tr>
      <tr> <td/> <td>
I was trained on the theoretical aspects of computer science such as compilers, logic, computer vision, type systems, etc. My final project aimed at developing a distributed, multi-core CPU and GPU (CUDA) computation of large Markov models on a distributed network, was awarded distinction and was published as <a href="http://eprints.maths.manchester.ac.uk/1533/">GPU-enabled steady-state solution of large Markov models</a> at NSMC’10;
      </td> </tr>

      <tr>
        <td style="min-width:70px"> 2007 </td>
        <td> <b> BSc Systems Engineering and Computer Science at University of Minho, Portugal </b> </td>
      </tr>
      <tr> <td/> <td>
Between 2005 and 2006 I was an ERASMUS exchange student at the University of Maribor in Slovenia. I finished the degree with A (best 10%);
      </td> </tr>

</table>


<h4 class="mt-5 mb-3">Publications</h4>

Here's a list of my most relevant publications. In the following, I was the first author and publications were peer-reviewed, unless mentioned otherwise. Conference journals/proceedings required a presentation at the venue as well. For a more exhaustive list, see my [Google Scholar]({{ site.google_scholar }}) profile.

|--- ||--- |
|2020||[Fully-Asynchronous Fully-Implicit Variable-Order Variable-Timestep Simulation of Neural Networks](https://arxiv.org/abs/1907.00670), Proc. International Conference on Computational Science, Amsterdam, Holland (ICCS 2020)|
|2020||[Efficient Distributed Transposition of Large-Scale Multigraphs And High-Cardinality Sparse Matrices](https://arxiv.org/abs/2012.06012), arXiv|
|2019||[Asynchronous SIMD-Enabled Branch-Parallelism of Morphologically-Detailed Neuron Models](https://www.frontiersin.org/articles/10.3389/fninf.2019.00054/full), Frontiers in Neuroinformatics|
|2019||[Asynchronous Simulation of Neuronal Activity](https://infoscience.epfl.ch/record/268035?ln=en), EPFL Scientific publications (PhD thesis)|
|2019||[Fully-Asynchronous Cache-Efficient Simulation of Detailed Neural Networks](https://www.researchgate.net/publication/333664427_Fully-Asynchronous_Cache-Efficient_Simulation_of_Detailed_Neural_Networks), Proc.  International Conference on Computational Science (ICCS 2019), Faro, Portugal|
|2019||[Exploiting Implicit Flow Graph of System of ODEs to Accelerate the Simulation of Neural Networks](https://ieeexplore.ieee.org/abstract/document/8821008), Proc. International Parallel and Distributed Processing Symposium (IPDPS 2019), Rio de Janeiro, Brazil|
|2016||[An efficient parallel load-balancing strategy for orthogonal decomposition of geometrical data](http://link.springer.com/chapter/10.1007/978-3-319-41321-1_5), Proc. International Super Computing (ISC 2016), Frankfurt, Germany|
|2015||(co-author) [Reconstruction and Simulation of Neocortical Microcircuitry](http://www.cell.com/abstract/S0092-8674(15)01191-5), Cell 163, 456–492.|
|2010||[GPU-enabled steady-state solution of large Markov models](http://eprints.ma.man.ac.uk/1533/), Proc. International Workshop on the Numerical Solution of Markov Chains (NSMC 2010), Williamsburg, Virginia (MSc final project)|
|ongoing||(arXiv) Distributed Async. Execution Speeds and Scales Up Over 100x The Detection Of Contacts Between Detailed Neuron Morphologies|


<div class="noprint">
<h4 class="mt-5 mb-3">Posts</h4>

When time allows, I post about HPC or ML projects I was involved in, or publications and discussions I find interesting.

<p>
  <table class="mt-3">
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
</p>

<p>
#I also maintain a <a href="{{ site.publications_permalink }}">summary of publications</a> and <a href="{{ site.resources_permalink }}">resources page</a> where I keep track of several free HPC and ML resources used as reference in my posts. 
I also maintain a <a href="{{ site.resources_permalink }}">resources page</a> where I keep track of several free HPC and ML resources used as reference in my posts. 
</p>

<h4 class="mt-5 mb-3">Misc</h4>

<p>
I've been playing waterpolo for most of my life, the last 12 years with <a href="https://lausannenatation.ch/section/waterpolo/">Lausanne Natation</a> and <a href="https://uk.teamunify.com/SubTabGeneric.jsp?_stabid_=153844/">Cambridge City</a> clubs. I enjoy cooking and winter sports - particularly skiing - and I am a cryptocurrency enthusiast. As a general rule, I prefer not to be addressed by my academic title or surname, so addressing me simply by my first name (<i>"Hi Bruno"</i>) is perfectly fine :)
</p>
</div> <!-- noprint -->
