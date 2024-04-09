---
layout: default
---

<h1 class="post-title p-name" itemprop="name headline">Bruno Magalhaes</h1>

## Machine Learning and High Performance Computing

<table style='table-layout:fixed; border:none; border-collapse:collapse; cellspacing:0; cellpadding:0'>
<tr>
<td width="20%" style='border:none; vertical-align: top;'> <img src="{{site.photo}}" alt="my photo" /> </td>
<td style="border:none">
Hi! I'm a research engineer on the fields of Machine Learning (ML) and High Performance Computing (HPC). I work at <a href="https://www.microsoft.com/en-us/research/lab/microsoft-research-cambridge/">Microsoft Research Cambridge</a> on <a href="https://www.microsoft.com/en-us/research/project/project-silica/">Project Silica</a>, where I create large parallel-distributed ML models and pipelines on the cloud. <br/><br/>Prior to this, I completed a PhD in Computational Neuroscience at <a href="https://www.epfl.ch/en/">EPFL</a>, researching large-scale variable-step simulation of brain-inspired spiking neural networks. Before that, I was an HPC research engineer at the <a href="https://www.epfl.ch/research/domains/bluebrain/">Blue Brain Project at EPFL</a>, focused on distributed computing, storage and multicore/GPU algorithms on supercomputers.
</td>
</tr></table> 

<!-- CSS of table defined in _includes/head.html -->
<div class="Rtable Rtable--3cols Rtable--collapse">
  <!-- <div class="Rtable-cell"> <a href="{{site.resume}}"><i class="far fa-file">&nbsp;</i>resume</a> </div> -->
  <div class="Rtable-cell"> <a href="mailto:{{ site.author.email }}?subject=Hello"><i class="far fa-envelope" title="Email">&nbsp;</i>{{site.author.email}}</a> </div>
  <div class="Rtable-cell"> <a href="https://www.linkedin.com/in/{{ site.linkedin_username }}"> <i class="fab fa-linkedin" >&nbsp;</i>{{ site.linkedin_username }}</a> </div>
  <div class="Rtable-cell"> <a href="https://github.com/{{ site.github_username }}"><i class="fab fa-fw fa-github" >&nbsp;</i>{{ site.github_username}}</a> </div>
  <!-- <div class="Rtable-cell"> <a href="https://twitter.com/{{ site.twitter_username }}"> <i class="fab fa-fw fa-twitter" ></i> {{ site.twitter_username }}</a> </div> -->
  <div class="Rtable-cell"> <a href="{{ site.google_scholar }}"> <i class="ai ai-google-scholar ai-1x" title="Google Scholar">&nbsp;</i>google scholar</a> </div>
  <div class="Rtable-cell"> <a href="{{ site.url }}"><i class="fas fa-mouse-pointer">&nbsp;</i>{{site.url | replace:'http://','' | replace:'https://','' }}</a> </div>
  <div class="Rtable-cell"> <i class="fas fa-passport" title="Nationality"></i>&nbsp;<a href="https://en.wikipedia.org/wiki/Lusitanians">Portuguese</a> and <a href="https://en.wikipedia.org/wiki/Helvetii">Swiss</a></div>
</div>


On the side, I maintain a [publications bookmark]({{ site.publications_permalink }}) where I summarize several papers of interest, and a [resources page]({{ site.resources_permalink }}) where I keep track of related books and material available online. My [google scholar page]({{ site.google_scholar }}) indexes most of my scientific publications. When time allows, I post about HPC and ML:

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


