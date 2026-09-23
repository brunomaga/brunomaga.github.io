---
layout: default
---

<!-- Contentor do topo estruturado com Flexbox responsivo -->
<div style="width: 100%; display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.5em; gap: 20px;">
  
  <div style="flex: 1;">
    <h1 class="post-title p-name" itemprop="name headline" style="margin-top: 0; margin-bottom: 0.2em; line-height: 1.1;">{{ site.author.name }}</h1>
    <h2 style='margin-top: 0em; margin-bottom: 0.3em; font-weight: normal; color: #555; font-size: 1.5em;'>Machine Learning and High Performance Computing</h2>
    
    <!-- Bloco de contactos otimizado para telemóveis com Feed RSS incluído -->
    <div style="display: flex; flex-wrap: wrap; gap: 12px 24px; color: #444; line-height: 1.5;">
      <a href="mailto:{{ site.author.email }}?subject=Hello" style="text-decoration: none; white-space: nowrap;"><i class="far fa-envelope" title="Email"></i> email</a>
      <a href="https://linkedin.com{{ site.linkedin_username }}" style="text-decoration: none; white-space: nowrap;"><i class="fab fa-linkedin"></i> linkedin</a>
      <a href="https://github.com{{ site.github_username }}" style="text-decoration: none; white-space: nowrap;"><i class="fab fa-fw fa-github"></i> github</a>
      <a href="{{ site.google_scholar }}" style="text-decoration: none; white-space: nowrap;"><i class="ai ai-google-scholar ai-1x" title="Google Scholar"></i> scholar</a>
      <a href="{{ '/feed.xml' | relative_url }}" style="text-decoration: none; white-space: nowrap;"><i class="fas fa-fw fa-rss"></i> RSS</a>
    </div>
  </div>

  <!-- Fotografia de perfil -->
  <div style="flex-shrink: 0;">
    <img src="{{ '/photo.png' | relative_url }}" 
         alt="Bruno Magalhaes" 
         style="width: 132px; height: 132px; border-radius: 50%; object-fit: cover; display: block; border: 1px solid #ddd;">
  </div>

</div>

<!-- Parágrafo de introdução -->
<p style="line-height: 1.6; margin-bottom: 0; clear: both;">
Hi! I am Bruno, an ML Systems researcher at <a href="https://huawei.com">Huawei Research</a>. Previously, I was an ML researcher at <a href="https://microsoft.com">Microsoft Research</a>, and an HPC engineer, PhD and postdoc at <a href="https://epfl.ch">EPFL</a>. In this space, I keep track of <a href="{{ site.publications_permalink }}">publications</a> and <a href="{{ site.resources_permalink }}">resources</a> of interest, and post about ML and HPC. Enjoy🚀!
</p>

<!-- Lista de Posts -->
<table style='width: 100%; border:none; border-collapse:collapse; cellspacing:0; cellpadding:0; margin-top: 1.5em;'>
{%- assign date_format = site.minima.date_format | default: "%Y" -%}
{% for post in site.posts %}
<tr style="border: none;">
<td class="align-top" style="border:none; width: 2.2em; padding-bottom: 0.2em; color: #666;">
{{ post.date | date: date_format }}
</td>
<td class="align-top" style="border:none; padding-bottom: 0.2em;">
<a href="{{ post.url }}">{{ post.title }}</a>
</td>
</tr>
{% endfor %}
</table>