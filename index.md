---
layout: default
title: "Home"
---

# Bruno Magalhaes

## Machine Learning and High Performance Computing

<table style='table-layout:fixed; border:none; border-collapse:collapse; cellspacing:0; cellpadding:0'>
<tr><td width="25%" style='border:none'>
<img src="{{site.photo}}" alt="my photo" width="100%" height="100%"/>
</td><td style="border:none">
Hi! I'm a research engineer on the fields of Machine Learning (ML) and High Performance Computing (HPC). I work at <a href="https://www.microsoft.com/en-us/research/lab/microsoft-research-cambridge/">Microsoft Research Cambridge</a> on <a href="https://www.microsoft.com/en-us/research/project/project-silica/">Project Silica</a>, where I create large parallel-distributed machine learning models and pipelines on GPU clusters and cloud. <br/><br/>

Prior to this, I completed a PhD in Computational Neuroscience at <a href="https://www.epfl.ch/en/">EPFL</a> in Switzerland, researching large-scale reconstruction and simulation of brain-inspired spiking neural networks. Before that, I was an HPC research engineer at the <a href="https://www.epfl.ch/research/domains/bluebrain/">Blue Brain Project</a>, working on large-scale distributed computing, storage and multicore/GPU algorithms.
</td></tr>
</table>

<table style='table-layout:fixed; border:none; border-collapse:collapse; cellspacing:0; cellpadding:0'>
  <tr>
    <td style="border:none; text-align:left"><a href="{{site.resume}}"><i class="far fa-file" title="resume"></i> one page resume</a></td>
    <td style="border:none; text-align:left"><a href="{{site.cv}}"><i class="far fa-file" title="resume"></i> full cv</a></td>
    <td style="border:none; text-align:left"><a href="mailto:{{ site.author.email }}?subject=Hello"><i class="far fa-envelope" title="Email"></i> {{site.author.email}}</a></td>
  </tr>
  <tr>
    <td style="border:none; text-align:left"><a href="https://github.com/{{ site.github_username }}"><i class="fab fa-fw fa-github" ></i> {{ site.github_username}}</a></td>
    <td style="border:none; text-align:left"><a href="https://www.linkedin.com/in/{{ site.linkedin_username }}"> <i class="fab fa-linkedin" ></i> {{ site.linkedin_username }}</a></td>
    <td style="border:none; text-align:left"><a href="https://twitter.com/{{ site.twitter_username }}"> <i class="fab fa-fw fa-twitter" ></i> {{ site.twitter_username }}</a></td>
  </tr>
  <tr>
    <td style="border:none; text-align:left"><a href="{{ site.google_scholar }}"> <i class="ai ai-google-scholar ai-1x" title="Google Scholar"></i> google scholar</a></td>
    <!-- <td style="border:none; text-align:left"><a href="https://en.wikipedia.org/wiki/Lausanne"> <i class="fa fa-home" title="Home"></i> Lausanne, Switzerland</a></td> -->
    <td style="border:none; text-align:left"><a href="{{ site.url }}"><i class="fas fa-mouse-pointer"></i> {{site.url}}</a></td>
    <td style="border:none; text-align:left"><a href="#"> <i class="fas fa-passport" title="Nationality"></i> Portuguese and Swiss</a></td>
  </tr>
</table>


When time allows, I post about HPC and ML.
I also maintain a <a href="{{ site.publications_permalink }}">publications bookmark</a> and a <a href="{{ site.resources_permalink }}">resources page</a> where I keep track of several resources used as reference in these posts. 
My <a href="{{ site.google_scholar }}">google scholar page</a> indexes most of my scientific peer-reviewed publications.

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


<br/>
As hobbies, I've been playing waterpolo for most of my life, the last 14 years with <a href="https://lausannenatation.ch/section/waterpolo/">Lausanne Aquatique</a> and <a href="https://uk.teamunify.com/SubTabGeneric.jsp?team=cocsc&_stabid_=154244">Cambridge City</a> clubs. I enjoy cooking, winter sports and board games. And as a general rule, I prefer not to be addressed by my academic title or surname, so addressing me simply by my first name (<i>"Hi Bruno"</i>) is perfectly fine <i class="far fa-smile"></i>

