---
layout: default
title: "Home"
---

<h1 class="post-title p-name" itemprop="name headline">Bruno Magalhaes</h1>

## Machine Learning and High Performance Computing

<table style='table-layout:fixed; border:none; border-collapse:collapse; cellspacing:0; cellpadding:0'>
<tr><td width="20%" style='border:none'>
<img src="{{site.photo}}" alt="my photo" width="100%" height="100%"/>
</td><td style="border:none">
Hi! I'm a research engineer on the fields of Machine Learning (ML) and High Performance Computing (HPC). I work at <a href="https://www.microsoft.com/en-us/research/lab/microsoft-research-cambridge/">Microsoft Research Cambridge</a> on <a href="https://www.microsoft.com/en-us/research/project/project-silica/">Project Silica</a>, where I create large parallel-distributed ML models and pipelines on the cloud. <br/><br/>Prior to this, I completed a PhD in Computational Neuroscience at <a href="https://www.epfl.ch/en/">EPFL</a>, researching large-scale reconstruction and simulation of brain-inspired spiking neural networks. Before that, I was an HPC research engineer at the <a href="https://www.epfl.ch/research/domains/bluebrain/">Blue Brain Project</a>, foccusing on distributed computing, storage and multicore/GPU algorithms.
</td></tr></table> 

|--- |--- |--- |
| <a href="{{site.resume}}"><i class="far fa-file"></i> one page resume</a> | <a href="{{site.cv}}"><i class="far fa-file"></i> full cv</a> | <a href="mailto:{{ site.author.email }}?subject=Hello"><i class="far fa-envelope" title="Email"></i> {{site.author.email}}</a> |
| <a href="https://github.com/{{ site.github_username }}"><i class="fab fa-fw fa-github" ></i> {{ site.github_username}}</a> | <a href="https://www.linkedin.com/in/{{ site.linkedin_username }}"> <i class="fab fa-linkedin" ></i> {{ site.linkedin_username }}</a> | <a href="https://twitter.com/{{ site.twitter_username }}"> <i class="fab fa-fw fa-twitter" ></i> {{ site.twitter_username }}</a> |
| <a href="{{ site.google_scholar }}"> <i class="ai ai-google-scholar ai-1x" title="Google Scholar"></i> google scholar</a> | <a href="{{ site.url }}"><i class="fas fa-mouse-pointer"></i> {{site.url | replace:'http://','' | replace:'https://','' }}</a> | <a href="#"> <i class="fas fa-passport" title="Nationality"></i> Portuguese and Swiss</a> |

When time allows, I post about HPC and ML.
I also maintain a <a href="{{ site.publications_permalink }}">publications bookmark</a> and a <a href="{{ site.resources_permalink }}">resources page</a> where I keep track of several resources used as reference in these posts. 
My <a href="{{ site.google_scholar }}">google scholar page</a> indexes most of my scientific publications.

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

