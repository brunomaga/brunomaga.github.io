---
layout: default
---

<h1 class="post-title p-name" itemprop="name headline">Bruno Magalhaes</h1>

<h2 style='margin-top:0em; margin-bottom:1em'> Machine Learning and High Performance Computing</h2>

<table style='table-layout:fixed; border:none; border-collapse:collapse; cellspacing:0; cellpadding:0'>
<tr>

<td width="19%" style='border:none; vertical-align: top;'>
    <img src="{{site.photo}}"/>
</td>

<td style="border:none">
Hi! I'm Bruno, a research engineer for large-scale AI at <a href="https://www.synthesia.io">Synthesia</a>. Previously, I was an ML researcher at <a href="https://www.microsoft.com/en-us/research/lab/microsoft-research-cambridge/">Microsoft Research Cambridge</a> on <a href="https://www.microsoft.com/en-us/research/project/project-silica/">Project Silica</a>. And before that, an HPC engineer, PhD and postdoc at <a href="https://www.epfl.ch/en/">EPFL</a>, researching variable-step simulation of spiking neural networks on large supercomputers.

On the side, I maintain a <a href="{{ site.publications_permalink }}">publications bookmark</a> and a <a href="{{ site.resources_permalink }}">resources page</a> where I track of material of interest. And I post about ML and HPC &#128640;.

<!-- CSS of table defined in _includes/head.html -->
<div class="Rtable Rtable--4cols Rtable--collapse">
  <div class="Rtable-cell"> <a href="mailto:{{ site.author.email }}?subject=Hello"><i class="far fa-envelope" title="Email">&nbsp;</i>email</a> </div>
  <div class="Rtable-cell"> <a href="https://www.linkedin.com/in/{{ site.linkedin_username }}"> <i class="fab fa-linkedin" >&nbsp;</i>linkedin</a> </div>
  <div class="Rtable-cell"> <a href="https://github.com/{{ site.github_username }}"><i class="fab fa-fw fa-github" >&nbsp;</i>github</a> </div>
  <div class="Rtable-cell"> <a href="{{ site.google_scholar }}"> <i class="ai ai-google-scholar ai-1x" title="Google Scholar">&nbsp;</i>google scholar</a> </div>
</div>

</td>
</tr>
</table> 

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


