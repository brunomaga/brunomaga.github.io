---
layout: default
title: "Bruno Magalhaes"
---

<h1 class="mt-5" itemprop="name headline">{{ page.title | escape }}</h1>

<p class="lead"><b>Research Engineer for Machine Learning and High Performance Computing</b></p>

<div class="row">
  <div class="col-3">
    <img src="{{site.photo}}" class="img-fluid rounded float-left" alt="my photo"/>
  </div>
  <div class="col">
<p>
I'm a Machine Learning research engineer with a large expertise on High Performance Computing, i.e. large-scale data science, distributed storage, data processing, multicore/GPU algorithms implemented on several supercomputers with over 10K nodes and exabyte scale datasets.
</p>

<p>
Previously, I was an AI resident at <a href="https://www.microsoft.com/en-us/research/lab/microsoft-research-cambridge">Microsoft Research</a> in Cambridge (UK), working on probabilistic machine learning, time-sequence data and graph neural networks. Prior to this, I completed a PhD in Computational Neuroscience at the <a href="https://www.epfl.ch/en/">EPFL</a> in Switzerland (researching large-scale reconstruction and simulation of brain-inspired spiking neural networks) and worked as a research software engineer for distributed algorithms at the <a href="https://www.epfl.ch/research/domains/bluebrain/">Blue Brain Project</a>. 
</p>

<p>
My research interests are Machine Learning applied to chaotic and massively-large datasets and compute infrastructure, and all related questions that can arise from the field, such as data confidentiality, poisoning attacks, learning from population clusters, learning from networks, federated ML, parallel ML processing, etc.
</p>
  </div>
</div>

<h4 class="mt-5 mb-3">Personal Details</h4>

<ul class="nav">
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
    <a class="nav-item btn btn-link" href="{{ site.google_scholar }}" class="btn btn-link"><i class="ai ai-google-scholar ai-1x"  title="Google Scholar"></i> Bruno Magalhaes</a>
  </li>
  <li class="nav-item">
    <a class="nav-link btn btn-link" href="https://en.wikipedia.org/wiki/Lausanne"><i class="fa fa-home"  title="Home"></i> Lausanne, Switzerland</a>
  </li>
  <li class="nav-item">
    <a class="nav-link btn btn-link" href="https://en.wikipedia.org/wiki/Portugal"><i class="fas fa-passport"  title="Nationality"></i> Portuguese</a>
  </li>
  <div class="noprint">
    <li class="nav-item">
      <a class="btn btn-link" href="{{ site.resume }}"><i class="far fa-user-circle"  title="resume"></i> resume</a>
    </li>
  </div>
  <!--
  <div class="noprint">
    <li class="nav-item">
      <a class="btn btn-link" href="javascript:window.print()"><i class="fas fa-download" title="download this page"></i> Download this page</a>
    </li>
  </div>
  -->
</ul>

<!--
<h4 class="mt-5 mb-3">Research Interests</h4>

I'm interested in applying Machine Learning to solve big engineering problems. When such ML systems don't exist, I like to create them, or at least face the challenge of trying. There are two big questions that I faced in life and would like to solve at some point, for the benefit of all:
- How to scale the processing of machine learning systems ([Moore's Law](https://en.wikipedia.org/wiki/Moore%27s_law)) to adapt to the exponential growth of new-generation data such as bots, generative ML, etc? For this I tend to read about decentralized and federated ML;
- How to make artificial neurons learn like spiking neurons, i.e. without back-propagation, loss function, and iterations?

<h4 class="mt-5 mb-3">Education</h4>

asdsa

-->

<h4 class="mt-5 mb-3">Biographical timeline</h4>


<!--
  <table class="mt-3">
      <tr>
        <td style="min-width:70px"> 2019-20 </td>
        <td> <b> AI resident at Microsoft Research, Cambridge, UK </b> </td>
      </tr>
      <tr> <td/> <td>
         We improved the load balancing of the CPU (and later storage) of thousands of Exchange email servers, dictated by the usage of individual email users across daytime, location, and logging information with over 2 exabytes of information. We provided a better mapping of user-to-CPU and a better answer to the question "which user should be allocated to each server at a given interval" by learning time series from user usage patterns. We implemented several algorithms to efficiently handle and do feature extraction on such distributed dataset. We later used Deep Neural Nets, Recursive Neural Nets, GRUs Encoder-Decoder with Attention Mechanism, and Bayesian Optimization (closed-form, Variational Inference, Markov Chain Monte Carlo) to learn from the data;
      </td> </tr>
      <tr> <td/> <td>
         We also built a recommendation system on a distributed graph with over a trillion edges, mapping all Meetings, Documents, Users and Emails. We developed several techniques to efficiently traverse the graph and filter the subgraph that would best relate to the query of similarity for two given nodes. This similarity metric would be learnt using a Graph Neural Net and Deep Neural Nets and the target would allow similar nodes to be recommended when creating meetings or emails; 
      </td> </tr>
  </table>
-->

|--- ||--- |
|2019-20|| AI resident at Microsoft Research, Cambridge, UK |
|2015-19|| PhD Computational Neuroscience at EPFL, Switzerland |
|2011-15|| Research Engineer at Blue Brain Project, EPFL, Switzerland |
|2009-11|| Junior IT architect at the Noble Group, London, UK |
|2008-09|| MSc Advanced Computing at Imperial College London, UK |
|2007-08|| Analyst Programmer at MSCI real estate, London, UK |
|2002-07|| BSc Systems Engineering and Computer Science at Univ. Minho, Portugal |

<h4 class="mt-5 mb-3">Publications</h4>

Here's a list of my most relevant publications. In the following, I was the first author and publications were peer-reviewed, unless mentioned otherwise. Conference journals/proceedings required a presentation at the venue as well. For a more exhaustive list, see my [Google Scholar]({{ site.google_scholar }}) profile.

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


<div class="noprint">
<h4 class="mt-5 mb-3">Posts</h4>

When time allows, I post about HPC or ML projects I was involved in, or publications and discussions I find interesting.

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
</div> <!-- noprint -->

I also keep note of a [resources page]({{ site.resources_permalink }}) where I list several HPC and ML ebooks available online that I use as reference on my posts. 

<h4 class="mt-5 mb-3">Misc</h4>

I have been playing waterpolo for most of my life, the last 10 years with [Lausanne Waterpolo](https://lausannenatation.ch/section/waterpolo/) and [Cambridge City Waterpolo](https://www.cambridgewaterpolo.co.uk/) clubs. I am also a big fan of winter sports, particularly skiing and snowboarding, and I feel lucky to live close to many great ski resorts in Switzerland. I also enjoy cooking and hope one day my cooking skills reach very high standards.

As a general rule, I prefer not to be addressed by my academic title or surname (e.g. *"Dear Dr Bruno"* or *"Dear Mr Magalhaes"*). I believe education is about knowledge and not diplomas. So addressing me simply by my first name (*"Hi Bruno"*) is perfectly fine :)
