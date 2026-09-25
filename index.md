---
layout: default
---

<!-- Top container with Flexbox responsiveness -->
<div style="width: 100%; display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.5em; gap: 20px;">
  
  <div style="flex: 1;">
    <h1 class="post-title p-name" itemprop="name headline" style="margin-top: 0; margin-bottom: 0.2em; line-height: 1.1;">{{ site.author.name }}</h1>
    <h2 style='margin-top: 0em; margin-bottom: 0.3em; font-weight: normal; color: #555; font-size: 1.5em;'>Machine Learning and High-Performance Computing</h2>
    
    <div style="display: flex; flex-wrap: wrap; gap: 12px 24px; color: #444; line-height: 1.5;">
      <a href="https://linkedin.com/in/brunomaga" style="text-decoration: none; white-space: nowrap;"><i class="fab fa-linkedin"></i> linkedin</a>
      <a href="https://github.com/brunomaga" style="text-decoration: none; white-space: nowrap;"><i class="fab fa-fw fa-github"></i> github</a>
      <a href="{{ site.google_scholar }}" style="text-decoration: none; white-space: nowrap;"><i class="ai ai-google-scholar ai-1x" title="Google Scholar"></i> scholar</a>
      <a href="{{ '/resume.pdf' | relative_url }}" style="text-decoration: none; white-space: nowrap;"><i class="far fa-file-alt" title="Resume"></i> resume</a>
      <a href="mailto:{{ site.author.email }}?subject=Hello" style="text-decoration: none; white-space: nowrap;"><i class="far fa-envelope" title="Email"></i> email</a>
      <a href="{{ '/feed.xml' | relative_url }}" style="text-decoration: none; white-space: nowrap;"><i class="fas fa-fw fa-rss"></i> RSS</a>
    </div>
  </div>

  <!-- profile photo -->
  <div style="flex-shrink: 0;">
    <img src="{{ '/photo.png' | relative_url }}" 
         alt="photo" 
         style="width: 132px; height: 132px; border-radius: 50%; object-fit: cover; display: block; border: 1px solid #ddd;" />
  </div>

</div>

<!-- introduction paragraph -->
<p style="line-height: 1.6; margin-bottom: 1.8em; clear: both;">
Hi👋🏽! I am Bruno, an ML Systems Researcher at <a href="https://www.huawei.com/ch-en/corporate-information/local-states">Huawei Research Switzerland</a>. Previously, I was an ML Researcher at <a href="https://www.microsoft.com/en-us/research/lab/microsoft-research-cambridge/">Microsoft Research Cambridge</a>, as well as an HPC Engineer, PhD, and Postdoc at <a href="https://epfl.ch">EPFL</a>. I specialize in edge-to-cloud ML system efficiency — from resource-constrained on-device inference to GPU distributed pre-training — for Large Language Models (LLMs), Mixture-of-Experts (MoEs), diffusion models, and AI agents.
</p>

<p style="line-height: 1.6; margin-bottom: 0; clear: both;">
In this space, I write about my projects and topics related to my fields of interest:
</p>

<!-- blog posts -->
<table style='width: 100%; border:none; border-collapse:collapse; margin-top: 0.5em;'>
{%- assign date_format = site.minima.date_format | default: "%Y" -%}
{% for post in site.posts %}
<tr style="border: none;">
<td class="align-top" style="border:none; width: 3.0em; padding-top: 0.4em; padding-bottom: 0.4em; color: #666; line-height: 1.4; vertical-align: top;">
{{ post.date | date: date_format }}
</td>
<td class="align-top" style="border:none; padding-top: 0.4em; padding-bottom: 0.4em; line-height: 1.4; vertical-align: top;">
<a href="{{ post.url }}">{{ post.title }}</a>
</td>
</tr>
{% endfor %}
</table>

<!-- resources -->
<p style="line-height: 1.6; margin-bottom: 0; clear: both;">
I also keep a curated list of related books, articles, and reference materials available online:
</p>

<table id="resources-table" style='width: 100%; border:none; border-collapse:collapse; margin-top: 0.5em;'>
<tr style="border: none;">
<td class="align-top" style="border:none; width: 3.0em; padding-top: 0.4em; padding-bottom: 0.4em; color: #666; line-height: 1.4; vertical-align: top;">book</td>
<td class="align-top" style="border:none; padding-top: 0.4em; padding-bottom: 0.4em; line-height: 1.4; vertical-align: top;">
  <a href="https://www.bishopbook.com/">Deep Learning - Foundations and Concepts, Christopher M. Bishop, Hugh Bishop</a> (<a href="https://github.com/luzmontserrat/deep-learning/blob/main/Christopher%20M.%20Bishop%2C%20Hugh%20Bishop%20-%20Deep%20Learning_%20Foundations%20and%20Concepts-Springer%20(2024).pdf">pdf</a>)
</td>
</tr>
<tr style="border: none;">
<td class="align-top" style="border:none; width: 3.0em; padding-top: 0.4em; padding-bottom: 0.4em; color: #666; line-height: 1.4; vertical-align: top;">book</td>
<td class="align-top" style="border:none; padding-top: 0.4em; padding-bottom: 0.4em; line-height: 1.4; vertical-align: top;">
  <a href="https://www.springer.com/gp/book/9780387310732">Pattern Recognition and Machine Learning, Christopher M. Bishop</a> (<a href="https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf">pdf</a>)
</td>
</tr>
<tr style="border: none;">
<td class="align-top" style="border:none; width: 3.0em; padding-top: 0.4em; padding-bottom: 0.4em; color: #666; line-height: 1.4; vertical-align: top;">book</td>
<td class="align-top" style="border:none; padding-top: 0.4em; padding-bottom: 0.4em; line-height: 1.4; vertical-align: top;">
  <a href="http://mbmlbook.com/">Model-Based Machine Learning, John Winn et al.</a> (<a href="http://mbmlbook.com/MBMLbook.pdf">pdf</a>)
</td>
</tr>
<tr style="border: none;">
<td class="align-top" style="border:none; width: 3.0em; padding-top: 0.4em; padding-bottom: 0.4em; color: #666; line-height: 1.4; vertical-align: top;">book</td>
<td class="align-top" style="border:none; padding-top: 0.4em; padding-bottom: 0.4em; line-height: 1.4; vertical-align: top;">
  <a href="https://mml-book.github.io/">Mathematics for Machine Learning, Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong</a> (<a href="https://mml-book.github.io/book/mml-book.pdf">pdf</a>)
</td>
</tr>
<tr style="border: none;">
<td class="align-top" style="border:none; width: 3.0em; padding-top: 0.4em; padding-bottom: 0.4em; color: #666; line-height: 1.4; vertical-align: top;">book</td>
<td class="align-top" style="border:none; padding-top: 0.4em; padding-bottom: 0.4em; line-height: 1.4; vertical-align: top;">
  <a href="https://neuronaldynamics.epfl.ch/">Neuronal Dynamics, Wulfram Gerstner et al.</a> (<a href="https://neuronaldynamics.epfl.ch/online/index.html">online</a>)
</td>
</tr>
<tr style="border: none;">
<td class="align-top" style="border:none; width: 3.0em; padding-top: 0.4em; padding-bottom: 0.4em; color: #666; line-height: 1.4; vertical-align: top;">book</td>
<td class="align-top" style="border:none; padding-top: 0.4em; padding-bottom: 0.4em; line-height: 1.4; vertical-align: top;">
  <a href="https://fleuret.org/public/lbdl.pdf">The Little Book of Deep Learning, François Fleuret</a> (<a href="{{ site.assets }}/resources/lbdl.pdf">pdf</a>)
</td>
</tr>
<tr style="border: none;">
<td class="align-top" style="border:none; width: 3.0em; padding-top: 0.4em; padding-bottom: 0.4em; color: #666; line-height: 1.4; vertical-align: top;">book</td>
<td class="align-top" style="border:none; padding-top: 0.4em; padding-bottom: 0.4em; line-height: 1.4; vertical-align: top;">
  <a href="https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/copy.html">Understanding Machine Learning: From Theory to Algorithms, Shai Shalev-Shwartz and Shai Ben-David</a> (<a href="https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf">pdf</a>)
</td>
</tr>
<tr style="border: none;">
<td class="align-top" style="border:none; width: 3.0em; padding-top: 0.4em; padding-bottom: 0.4em; color: #666; line-height: 1.4; vertical-align: top;">book</td>
<td class="align-top" style="border:none; padding-top: 0.4em; padding-bottom: 0.4em; line-height: 1.4; vertical-align: top;">
  <a href="http://www.inference.org.uk/mackay/itila/book.html">Information Theory, Inference, and Learning Algorithms, David MacKay</a> (<a href="http://www.inference.org.uk/itprnn/book.pdf">pdf</a>)
</td>
</tr>
<tr style="border: none;">
<td class="align-top" style="border:none; width: 3.0em; padding-top: 0.4em; padding-bottom: 0.4em; color: #666; line-height: 1.4; vertical-align: top;">article</td>
<td class="align-top" style="border:none; padding-top: 0.4em; padding-bottom: 0.4em; line-height: 1.4; vertical-align: top;">
  <a href="{{ site.assets }}/resources/In-Network_Collective_Operations_Game_Changer_or_Challenge_for_AI_Workloads.pdf">In-Network Collective Operations: Game Changer or Challenge for AI Workloads?</a>
</td>
</tr>
<tr style="border: none;">
<td class="align-top" style="border:none; width: 3.0em; padding-top: 0.4em; padding-bottom: 0.4em; color: #666; line-height: 1.4; vertical-align: top;">post</td>
<td class="align-top" style="border:none; padding-top: 0.4em; padding-bottom: 0.4em; line-height: 1.4; vertical-align: top;">
  <a href="https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html">The vLLM MoE Playbook: A Practical Guide to TP, DP, PP and Expert Parallelism (AMD)</a>
</td>
</tr>
<tr style="border: none;">
<td class="align-top" style="border:none; width: 3.0em; padding-top: 0.4em; padding-bottom: 0.4em; color: #666; line-height: 1.4; vertical-align: top;">post</td>
<td class="align-top" style="border:none; padding-top: 0.4em; padding-bottom: 0.4em; line-height: 1.4; vertical-align: top;">
  <a href="https://jax-ml.github.io/scaling-book">How to scale your model (JAX ML)</a>
</td>
</tr>
<tr style="border: none;">
<td class="align-top" style="border:none; width: 3.0em; padding-top: 0.4em; padding-bottom: 0.4em; color: #666; line-height: 1.4; vertical-align: top;">post</td>
<td class="align-top" style="border:none; padding-top: 0.4em; padding-bottom: 0.4em; line-height: 1.4; vertical-align: top;">
  <a href="https://huggingface.co/spaces/nanotron/ultrascale-playbook">The Ultra-Scale Playbook: Training LLMs on GPU Clusters (Hugging Face)</a>
</td>
</tr>
<tr style="border: none;">
<td class="align-top" style="border:none; width: 3.0em; padding-top: 0.4em; padding-bottom: 0.4em; color: #666; line-height: 1.4; vertical-align: top;">notes</td>
<td class="align-top" style="border:none; padding-top: 0.4em; padding-bottom: 0.4em; line-height: 1.4; vertical-align: top;">
  <a href="{{ site.assets }}/resources/the_matrix_cookbook.pdf">The Matrix Cookbook</a>
</td>
</tr>
<tr style="border: none;">
<td class="align-top" style="border:none; width: 3.0em; padding-top: 0.4em; padding-bottom: 0.4em; color: #666; line-height: 1.4; vertical-align: top;">lecture</td>
<td class="align-top" style="border:none; padding-top: 0.4em; padding-bottom: 0.4em; line-height: 1.4; vertical-align: top;">
  <a href="https://www.cs.princeton.edu/courses/archive/fall18/cos324/files/backprop.pdf">Computing Gradients with Backpropagation - Automatic Differentiation (Princeton COS-324)</a> (<a href="{{ site.assets }}/resources/princeton_course_autodiff.pdf">pdf</a>)
</td>
</tr>
<tr style="border: none;">
<td class="align-top" style="border:none; width: 3.0em; padding-top: 0.4em; padding-bottom: 0.4em; color: #666; line-height: 1.4; vertical-align: top;">course</td>
<td class="align-top" style="border:none; padding-top: 0.4em; padding-bottom: 0.4em; line-height: 1.4; vertical-align: top;">
  <a href="https://edu.epfl.ch/coursebook/en/statistics-for-data-science-MATH-413">Statistics for Data Science (EPFL MATH-413)</a>: <a href="{{ site.epfl_statistics | append: 'lecture-slides/' }}">lectures</a>, <a href="{{ site.epfl_statistics_videos }}">videos</a>, <a href="{{ site.epfl_statistics | append: 'exercises/' }}">exercises</a>, <a href="{{ site.epfl_statistics | append: 'formula-sheet-continuous-distributions.pdf' }}">continuous</a> and <a href="{{ site.epfl_statistics | append: 'formula-sheet-discrete-distributions.pdf'  }}">discrete</a> distributions
</td>
</tr>
</table>

<p style="line-height: 1.6; margin-bottom: 0; clear: both;">
And finally, I maintain a list of <a href="{{ site.publications_permalink }}">relevant publications</a>. Enjoy, and feel free to reach out with feedback, fixes, or questions 🚀!
</p>
