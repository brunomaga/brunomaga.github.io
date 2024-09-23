---
layout: default
---

<h1 class="post-title p-name" itemprop="name headline">Bruno Magalhaes</h1>

<h2 style='margin-top:0em; margin-bottom:1em'> Machine Learning and High Performance Computing</h2>

<table style='table-layout:fixed; border:none; border-collapse:collapse; cellspacing:0; cellpadding:0'>
<tr>
<td width="19%" style='border:none; vertical-align: top;'> <img src="{{site.photo}}" alt="my photo" /> </td>
<td style="border:none">
Hi! I'm a research engineer on the fields of Machine Learning (ML) and High Performance Computing (HPC), currently working on large-scale AI at <a href="https://www.synthesia.io">Synthesia</a>. Previously, I was a researcher at <a href="https://www.microsoft.com/en-us/research/lab/microsoft-research-cambridge/">Microsoft Research Cambridge</a> for <a href="https://www.microsoft.com/en-us/research/project/project-silica/">Project Silica</a>. Before that, I was an HPC engineer, PhD and postdoc researcher at <a href="https://www.epfl.ch/en/">EPFL</a>, researching variable-step simulation of brain-inspired spiking neural networks on large supercomputers.

<!-- CSS of table defined in _includes/head.html -->
<div class="Rtable Rtable--2cols Rtable--collapse">
  <div class="Rtable-cell"> <a href="mailto:{{ site.author.email }}?subject=Hello"><i class="far fa-envelope" title="Email">&nbsp;</i>{{site.author.email}}</a> </div>
  <div class="Rtable-cell"> <a href="https://www.linkedin.com/in/{{ site.linkedin_username }}"> <i class="fab fa-linkedin" >&nbsp;</i>{{ site.linkedin_username }}</a> </div>
  <div class="Rtable-cell"> <a href="https://github.com/{{ site.github_username }}"><i class="fab fa-fw fa-github" >&nbsp;</i>{{ site.github_username}}</a> </div>
  <div class="Rtable-cell"> <a href="{{ site.google_scholar }}"> <i class="ai ai-google-scholar ai-1x" title="Google Scholar">&nbsp;</i>google scholar</a> </div>
</div>

</td>
</tr>
</table> 

On the side, I maintain a [publications bookmark]({{ site.publications_permalink }}) and a [resources page]({{ site.resources_permalink }}) where I keep track of papers, books and other material of interest. My [scholar page]({{ site.google_scholar }}) indexes most of my publications. When time allows, I post about HPC and ML:

<table style='border:none; border-collapse:collapse; cellspacing:0; cellpadding:0'>
{%- assign date_format = site.minima.date_format | default: "%Y" -%}
{% for post in site.posts %}
<tr>
<td class="align-top" style="border:none">
{{ post.date | date: date_format }}
</td>
<td class="align-top" style="border:none">
<a href="{{ post.url }}">{{ post.title }}</a>
</td>
</tr>
{% endfor %}
</table>


