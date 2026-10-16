---
layout: default
---

<!-- name, field and links, with the photo on the right -->
<div class="home-header">
  <div class="home-header-text">
    <h1 class="post-title">{{ site.author.name }}</h1>
    <h2>Machine Learning and High-Performance Computing</h2>

    <div class="home-links">
      <a href="https://linkedin.com/in/brunomaga"><i class="fab fa-linkedin"></i> linkedin</a>
      <a href="https://github.com/brunomaga"><i class="fab fa-fw fa-github"></i> github</a>
      <a href="{{ site.google_scholar }}"><i class="ai ai-google-scholar ai-1x" title="Google Scholar"></i> scholar</a>
      <a href="mailto:{{ site.author.email }}?subject=Hello"><i class="far fa-envelope" title="Email"></i> email</a>
      <a href="{{ '/feed.xml' | relative_url }}"><i class="fas fa-fw fa-rss"></i> RSS</a>
    </div>
  </div>

  <img class="home-photo" src="{{ '/photo.png' | relative_url }}" alt="photo" />
</div>

<!-- introduction paragraph -->
<p class="home-intro">
Hi 👋🏽! I am Bruno, an ML Systems Researcher at <a href="https://www.huawei.com/ch-en/corporate-information/local-states">Huawei Research Switzerland</a>. Previously, I was an ML Researcher at <a href="https://www.microsoft.com/en-us/research/lab/microsoft-research-cambridge/">Microsoft Research Cambridge</a>, and an HPC Engineer and PhD in Computational Neuroscience at <a href="https://epfl.ch">EPFL</a>. I focus on ML efficiency — from embedded inference to distributed training — across Transformer, MoE, diffusion, and agentic models.
</p>

<p class="home-text">
In this space, I write about my projects and topics related to my fields of interest:
</p>

<!-- blog posts -->
<table class="label-table home-table">
{%- assign previous_year = "" -%}
{% for post in site.posts %}
{%- assign year = post.date | date: "%Y" -%}
<tr>
<td>{% if year != previous_year %}<span class="post-label">{{ year }}</span>{% endif %}</td>
<td>
<a href="{{ post.url }}">{{ post.categories.first }}: {{ post.title }}</a>
</td>
</tr>
{%- assign previous_year = year -%}
{% endfor %}
</table>

<!-- resources -->
<p class="home-text">
I also keep a curated list of related books and other resources available online:
</p>

<table id="resources-table" class="label-table home-table">
<tr>
<td><span class="post-label">book</span></td>
<td>
  <a href="https://www.bishopbook.com/">Deep Learning - Foundations and Concepts, Christopher M. Bishop, Hugh Bishop</a> (<a href="https://github.com/luzmontserrat/deep-learning/blob/main/Christopher%20M.%20Bishop%2C%20Hugh%20Bishop%20-%20Deep%20Learning_%20Foundations%20and%20Concepts-Springer%20(2024).pdf">pdf</a>)
</td>
</tr>
<tr>
<td></td>
<td>
  <a href="https://www.springer.com/gp/book/9780387310732">Pattern Recognition and Machine Learning, Christopher M. Bishop</a> (<a href="https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf">pdf</a>)
</td>
</tr>
<tr>
<td></td>
<td>
  <a href="http://mbmlbook.com/">Model-Based Machine Learning, John Winn et al.</a> (<a href="http://mbmlbook.com/MBMLbook.pdf">pdf</a>)
</td>
</tr>
<tr>
<td></td>
<td>
  <a href="https://mml-book.github.io/">Mathematics for Machine Learning, Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong</a> (<a href="https://mml-book.github.io/book/mml-book.pdf">pdf</a>)
</td>
</tr>
<tr>
<td></td>
<td>
  <a href="https://neuronaldynamics.epfl.ch/">Neuronal Dynamics, Wulfram Gerstner et al.</a> (<a href="https://neuronaldynamics.epfl.ch/online/index.html">online</a>)
</td>
</tr>
<tr>
<td></td>
<td>
  <a href="https://fleuret.org/public/lbdl.pdf">The Little Book of Deep Learning, François Fleuret</a> (<a href="{{ site.assets }}/resources/lbdl.pdf">pdf</a>)
</td>
</tr>
<tr>
<td></td>
<td>
  <a href="https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/copy.html">Understanding Machine Learning: From Theory to Algorithms, Shai Shalev-Shwartz and Shai Ben-David</a> (<a href="https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf">pdf</a>)
</td>
</tr>
<tr>
<td></td>
<td>
  <a href="http://www.inference.org.uk/mackay/itila/book.html">Information Theory, Inference, and Learning Algorithms, David MacKay</a> (<a href="http://www.inference.org.uk/itprnn/book.pdf">pdf</a>)
</td>
</tr>
<tr>
<td><span class="post-label">blog</span></td>
<td>
  <a href="https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html">The vLLM MoE Playbook: A Practical Guide to TP, DP, PP and Expert Parallelism (AMD)</a>
</td>
</tr>
<tr>
<td></td>
<td>
  <a href="https://jax-ml.github.io/scaling-book">How to scale your model (JAX ML)</a>
</td>
</tr>
<tr>
<td></td>
<td>
  <a href="https://huggingface.co/spaces/nanotron/ultrascale-playbook">The Ultra-Scale Playbook: Training LLMs on GPU Clusters (Hugging Face)</a>
</td>
</tr>
<tr>
<td><span class="post-label">univ</span></td>
<td>
  <a href="https://www.cs.princeton.edu/courses/archive/fall18/cos324/files/backprop.pdf">Computing Gradients with Backpropagation - Automatic Differentiation (Princeton course COS-324)</a> (<a href="{{ site.assets }}/resources/princeton_course_autodiff.pdf">pdf</a>)
</td>
</tr>
<tr>
<td></td>
<td>
  <a href="{{ site.epfl_statistics }}/README.md">Statistics for Data Science (EPFL course MATH-413)</a>
</td>
</tr>
<tr>
<td><span class="post-label">misc</span></td>
<td>
  <a href="{{ site.assets }}/resources/the_matrix_cookbook.pdf">The Matrix Cookbook</a>
</td>
</tr>
</table>

<p class="home-text">
And finally, I maintain a list of <a href="{{ site.publications_permalink }}">relevant publications</a>. Enjoy, and feel free to reach out with feedback, fixes, and questions 🚀!
</p>
