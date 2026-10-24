---
layout: default
---

<!-- name, field and links, with the photo on the right -->
<div class="home-header">
  <div class="home-header-text">
    <h1 class="post-title">{{ site.author.name }}</h1>
    <h2 class="home-field">Machine Learning<span class="home-field-sep" aria-hidden="true"></span><br class="home-field-break">High-Performance Computing</h2>
    <div class="home-links">
      <a href="https://linkedin.com/in/brunomaga" aria-label="LinkedIn" title="LinkedIn"><i class="fab fa-linkedin" aria-hidden="true"></i></a>
      <a href="https://github.com/brunomaga" aria-label="GitHub" title="GitHub"><i class="fab fa-github" aria-hidden="true"></i></a>
      <a href="{{ site.google_scholar }}" aria-label="Google Scholar" title="Google Scholar"><svg viewBox="0 0 24 24" width="1em" height="1em" fill="currentColor" aria-hidden="true"><path d="M5.242 13.769L0 9.5 12 0l12 9.5-5.242 4.269C17.548 11.249 14.978 9.5 12 9.5c-2.977 0-5.548 1.748-6.758 4.269zM12 10a7 7 0 1 0 0 14 7 7 0 0 0 0-14z"/></svg></a>
      <!-- <a href="{{ '/resume.pdf' | relative_url }}" aria-label="Resume" title="Resume"><i class="far fa-file-lines" aria-hidden="true"></i></a> -->
      <a href="mailto:{{ site.author.email }}?subject=Hello" aria-label="Email" title="Email"><i class="fas fa-envelope" aria-hidden="true"></i></a>
    </div>
  </div>

  <div class="home-photo"><img src="{{ '/photo.png' | relative_url }}" alt="photo" /></div>
</div>

<!-- introduction paragraph -->
<p class="home-intro">
Hi 👋🏽! I am Bruno, an ML Systems Researcher at <a href="https://www.huawei.com/ch-en/corporate-information/local-states">Huawei Research Switzerland</a>. Previously, I was an ML Researcher at <a href="https://www.microsoft.com/en-us/research/lab/microsoft-research-cambridge/">Microsoft Research Cambridge</a>, and an HPC Engineer and PhD Computational Neuroscience at <a href="https://epfl.ch">EPFL</a>. I focus on ML efficiency — from embedded inference to distributed training — across Transformer, MoE, diffusion, and agentic models.
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
I also keep a curated list of publications, books and other resources in the field:
</p>

<table id="resources-table" class="label-table home-table">
<tr>
<td><span class="post-label">pubs</span></td>
<td>
  <a href="{{ site.publications_permalink }}">Publication Summaries</a> <span class="pub-meta">(continuously updated)</span>
</td>
</tr>
<tr>
<td><span class="post-label">book</span></td>
<td>
  <a href="https://www.bishopbook.com/">Deep Learning - Foundations and Concepts, Christopher M. Bishop, Hugh Bishop</a><a class="pub-link" href="https://www.bishopbook.com/" target="_blank" rel="noopener" title="Open the page"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a><a class="pub-link" href="https://github.com/luzmontserrat/deep-learning/blob/main/Christopher%20M.%20Bishop%2C%20Hugh%20Bishop%20-%20Deep%20Learning_%20Foundations%20and%20Concepts-Springer%20(2024).pdf" target="_blank" rel="noopener" title="Download the PDF"><i class="fa-solid fa-download" aria-hidden="true"></i></a>
</td>
</tr>
<tr>
<td></td>
<td>
  <a href="https://www.springer.com/gp/book/9780387310732">Pattern Recognition and Machine Learning, Christopher M. Bishop</a><a class="pub-link" href="https://www.springer.com/gp/book/9780387310732" target="_blank" rel="noopener" title="Open the page"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a><a class="pub-link" href="https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf" target="_blank" rel="noopener" title="Download the PDF"><i class="fa-solid fa-download" aria-hidden="true"></i></a>
</td>
</tr>
<tr>
<td></td>
<td>
  <a href="http://mbmlbook.com/">Model-Based Machine Learning, John Winn et al.</a><a class="pub-link" href="http://mbmlbook.com/" target="_blank" rel="noopener" title="Open the page"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a><a class="pub-link" href="http://mbmlbook.com/MBMLbook.pdf" target="_blank" rel="noopener" title="Download the PDF"><i class="fa-solid fa-download" aria-hidden="true"></i></a>
</td>
</tr>
<tr>
<td></td>
<td>
  <a href="https://mml-book.github.io/">Mathematics for Machine Learning, Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong</a><a class="pub-link" href="https://mml-book.github.io/" target="_blank" rel="noopener" title="Open the page"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a><a class="pub-link" href="https://mml-book.github.io/book/mml-book.pdf" target="_blank" rel="noopener" title="Download the PDF"><i class="fa-solid fa-download" aria-hidden="true"></i></a>
</td>
</tr>
<tr>
<td></td>
<td>
  <a href="https://neuronaldynamics.epfl.ch/">Neuronal Dynamics, Wulfram Gerstner et al.</a><a class="pub-link" href="https://neuronaldynamics.epfl.ch/" target="_blank" rel="noopener" title="Open the page"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a>
</td>
</tr>
<tr>
<td></td>
<td>
  <a href="https://fleuret.org/public/lbdl.pdf">The Little Book of Deep Learning, François Fleuret</a><a class="pub-link" href="{{ site.assets }}/resources/lbdl.pdf" target="_blank" rel="noopener" title="Download the PDF"><i class="fa-solid fa-download" aria-hidden="true"></i></a>
</td>
</tr>
<tr>
<td></td>
<td>
  <a href="https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/copy.html">Understanding Machine Learning: From Theory to Algorithms, S. Shalev-Shwartz, S. Ben-David</a><a class="pub-link" href="https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/copy.html" target="_blank" rel="noopener" title="Open the page"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a><a class="pub-link" href="https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf" target="_blank" rel="noopener" title="Download the PDF"><i class="fa-solid fa-download" aria-hidden="true"></i></a>
</td>
</tr>
<tr>
<td></td>
<td>
  <a href="http://www.inference.org.uk/mackay/itila/book.html">Information Theory, Inference, and Learning Algorithms, David MacKay</a><a class="pub-link" href="http://www.inference.org.uk/mackay/itila/book.html" target="_blank" rel="noopener" title="Open the page"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a><a class="pub-link" href="http://www.inference.org.uk/itprnn/book.pdf" target="_blank" rel="noopener" title="Download the PDF"><i class="fa-solid fa-download" aria-hidden="true"></i></a>
</td>
</tr>
<tr>
<td><span class="post-label">blog</span></td>
<td>
  <a href="https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html">The vLLM MoE Playbook: A Practical Guide to TP, DP, PP and Expert Parallelism (AMD)</a><a class="pub-link" href="https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html" target="_blank" rel="noopener" title="Open the page"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a>
</td>
</tr>
<tr>
<td></td>
<td>
  <a href="https://jax-ml.github.io/scaling-book">How to scale your model (JAX ML)</a><a class="pub-link" href="https://jax-ml.github.io/scaling-book" target="_blank" rel="noopener" title="Open the page"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a>
</td>
</tr>
<tr>
<td></td>
<td>
  <a href="https://huggingface.co/spaces/nanotron/ultrascale-playbook">The Ultra-Scale Playbook: Training LLMs on GPU Clusters (Hugging Face)</a><a class="pub-link" href="https://huggingface.co/spaces/nanotron/ultrascale-playbook" target="_blank" rel="noopener" title="Open the page"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a>
</td>
</tr>
<tr>
<td><span class="post-label">univ</span></td>
<td>
  <a href="https://www.cs.princeton.edu/courses/archive/fall18/cos324/">Introduction to Machine Learning - Automatic Differentiation (Princeton COS 324)</a><a class="pub-link" href="https://www.cs.princeton.edu/courses/archive/fall18/cos324/" target="_blank" rel="noopener" title="Open the page"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a><a class="pub-link" href="{{ site.assets }}/resources/princeton_course_autodiff.pdf" target="_blank" rel="noopener" title="Download the PDF"><i class="fa-solid fa-download" aria-hidden="true"></i></a>
</td>
</tr>
<tr>
<td></td>
<td>
  <a href="{{ site.epfl_statistics }}/README.md">Statistics for Data Science (EPFL MATH-413)</a><a class="pub-link" href="{{ site.epfl_statistics }}/README.md" target="_blank" rel="noopener" title="Open the page"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a>
</td>
</tr>
<tr>
<td></td>
<td>
  <a href="{{ site.epfl_machine_learning }}/README.md">Machine Learning (EPFL CS-433)</a><a class="pub-link" href="{{ site.epfl_machine_learning }}/README.md" target="_blank" rel="noopener" title="Open the page"><i class="fa-solid fa-arrow-up-right-from-square" aria-hidden="true"></i></a>
</td>
</tr>
<tr>
<td><span class="post-label">misc</span></td>
<td>
  <a href="{{ site.assets }}/resources/the_matrix_cookbook.pdf">The Matrix Cookbook</a><a class="pub-link" href="{{ site.assets }}/resources/the_matrix_cookbook.pdf" target="_blank" rel="noopener" title="Download the PDF"><i class="fa-solid fa-download" aria-hidden="true"></i></a>
</td>
</tr>
</table>

<p class="home-text">
Enjoy, and feel free to reach out with feedback, fixes, and questions 🚀!
</p>
