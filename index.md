---
layout: default
---

<h1 class="post-title p-name" itemprop="name headline">{{ site.author.name }} </h1>

<h2 style='margin-top:0em; margin-bottom:1em'> Machine Learning and High Performance Computing</h2>

<table style='table-layout:fixed; border:none; border-collapse:collapse; cellspacing:0; cellpadding:0'>
<tr>

<td width="16%" style='border:none; vertical-align: top;'>
    <img src="{{site.photo}}"/>
</td>

<td style="border:none">
Hi👋🏽! I am Bruno, an ML Systems researcher at <a href="https://www.huawei.com/ch-en/corporate-information/local-states/">Huawei Research</a>. Previously, I was an ML researcher at <a href="https://www.microsoft.com/en-us/research/lab/microsoft-research-cambridge/">Microsoft Research</a>, and an HPC engineer, PhD and postdoc at <a href="https://www.epfl.ch/en/">EPFL</a>. In this space, I keep track of <a href="{{ site.publications_permalink }}">publications</a> and <a href="{{ site.resources_permalink }}">resources</a> of interest, and post about ML and HPC. Enjoy🚀!

<!-- CSS of table defined in _includes/head.html -->
<div class="Rtable Rtable--4cols Rtable--collapse">
  <div class="Rtable-cell"> <a href="mailto:{{ site.author.email }}?subject=Hello"><i class="far fa-envelope" title="Email">&nbsp;</i>email</a> </div>
  <div class="Rtable-cell"> <a href="https://github.com/{{ site.github_username }}"><i class="fab fa-fw fa-github" >&nbsp;</i>github</a> </div>
  <div class="Rtable-cell"> <a href="https://www.linkedin.com/in/{{ site.linkedin_username }}"> <i class="fab fa-linkedin" >&nbsp;</i>linkedin</a> </div>
  <div class="Rtable-cell"> <a href="{{ site.google_scholar }}"> <i class="ai ai-google-scholar ai-1x" title="Google Scholar">&nbsp;</i>scholar</a> </div>
  <!-- <div class="Rtable-cell"> <a href="{{ "/feed.xml" | relative_url }}"><i class="fas fa-fw fa-rss" ></i>RSS</a></div> -->
</div>

<br/>
</td>
</tr>
</table> 

<center> <div class="info-warning"> <strong>Beware:</strong> scammers are using my name and face (with AI) to impersonate me in job interviews.</div> </center>

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


<center>
<div class="info-panel">
<strong>Support this blog!</strong> If you like this content and would like to show appreciation, please donate instead to the children's cancer hospital in Porto via this <a href="https://www.gofundme.com/f/support-the-childrens-cancer-hospital-in-porto">GoFundMe campaign</a>. Thank you for caring❤️‍🩹
</div>
</center>














