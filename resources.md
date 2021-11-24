---
layout: default
title: Resources
permalink: /resources/
---

<h1 class="mt-5" itemprop="name headline">{{ page.title | escape }}</h1>

  <div class="mt-3"></div>
Some books freely available online, and related to the topics discussed: 
<ul>
          <li><a href="https://www.springer.com/gp/book/9780387310732">Pattern Classification and Machine Learning, Christopher M. Bishop</a> (<a href="https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf">pdf</a>)</li>
          <li><a href="http://mbmlbook.com/">Model-Based Machine Learning, John Winn et al.</a> (<a href="http://mbmlbook.com/MBMLbook.pdf">pdf</a>)</li>
          <li><a href="https://mml-book.github.io/">Mathematics for Machine Learning, Deisenroth, Aldo Faisal, Cheng S. Ong</a> (<a href="https://mml-book.github.io/book/mml-book.pdf">pdf</a>)</li>
          <li><a href="https://neuronaldynamics.epfl.ch/">Neuronal Dynamics, Wulfram Gerstner et al.</a> (<a href="https://neuronaldynamics.epfl.ch/online/index.html">online</a>)</li>
          <li><a href="https://www.pdfdrive.com/schaums-outline-of-linear-algebra-5th-ed-e179255854.html">Schaum's Outline of Linear Algebra (5th Ed.), Seymour Lipschutz and Marc Lipson</a> (<a href="https://www.pdfdrive.com/download.pdf?id=179255854&h=b429869e4e59b3160a950bd30678a00a&u=cache&ext=pdf">pdf</a>)</li>
          <li><a href="https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/copy.html">Understanding Machine Learning: From Theory to Algorithms,  Shai Shalev-Shwartz and Shai Ben-David</a> (<a href="https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf">pdf</a>)</li>
          <li><a href="http://www.inference.org.uk/mackay/itila/book.html">Information Theory, Inference, and Learning Algorithms, David MacKay</a> (<a href="http://www.inference.org.uk/itprnn/book.pdf">pdf</a>)</li>
</ul>

<br/>
Material for the course <a href="https://edu.epfl.ch/coursebook/en/statistics-for-data-science-MATH-413">Statistics for data Science</a> at EPFL:
<ul>
<li>lecture slides:
  {% for i in (1..23) %}
    <a href="{{ site.statistics_lectures | replace: 'XXX', i }}"> {{i}} </a>
  {% endfor %}
</li>
<li>lecture videos:
  {% for keyval in site.statistics_videos %}
      <a href="{{site.statistics_videos_preffix}}{{ keyval[1] }}">{{ keyval[0] }}</a>
  {% endfor %}
</li>
<li>exercises:
  {% for i in (1..12) %}
    <a href="{{ site.statistics_exercises | replace: 'XXX', i }}"> {{i}} </a>
  {% endfor %}
</li>
<li>solutions:
  {% for i in (1..12) %}
    <a href="{{ site.statistics_solutions | replace: 'XXX', i }}"> {{i}} </a>
  {% endfor %}
</li>
<li>probabilistic density, distribution and parameters for 
    <a href="{{ site.statistics_distributions | replace: 'XXX', 'CONTINUOUS' }}"> continuous </a>
    and
    <a href="{{ site.statistics_distributions | replace: 'XXX', 'DISCRETE' }}"> discrete </a> 
    distributions
</li>
</ul>

<br/>
Other useful resources:
<ul>
          <li><a href="{{ site.assets }}/latex_math_symbols.pdf">Latex Mathematical Symbols</a></li>
          <li><a href="{{ site.assets }}/the_matrix_cookbook.pdf">The Matrix Cookbook</a></li>
</ul>

