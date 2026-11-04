---
layout: page
title: projects
permalink: /projects/
description: Research I have done, from RL post-training and evaluation to retrieval and open datasets, plus the open-source repositories behind it. Teaching and coursework from Stanford at the bottom.
nav: true
nav_order: 2
display_categories: [Research, Teaching, Coursework]
horizontal: false
---

<!-- pages/projects.md -->
<div class="projects">
{% if site.enable_project_categories and page.display_categories %}
  <!-- Display categorized projects -->
  {% for category in page.display_categories %}
  <a id="{{ category }}" href=".#{{ category }}">
    <h2 class="category">{{ category }}</h2>
  </a>
  {% assign categorized_projects = site.projects | where: "category", category %}
  {% assign sorted_projects = categorized_projects | sort: "importance" %}
  <!-- Generate cards for each project -->
  {% if page.horizontal %}
  <div class="container">
    <div class="row row-cols-1 row-cols-md-2">
    {% for project in sorted_projects %}
      {% include projects_horizontal.liquid %}
    {% endfor %}
    </div>
  </div>
  {% else %}
  <div class="row row-cols-1 row-cols-md-3">
    {% for project in sorted_projects %}
      {% include projects.liquid %}
    {% endfor %}
  </div>
  {% endif %}
  {% if category == "Research" %}
  <a id="Open source" href=".#Open source">
    <h2 class="category">Open source</h2>
  </a>
<ul class="repo-list">
  <li class="repo-row">
    <div class="repo-name"><a href="https://github.com/thu-coai/CDial-GPT">CDial-GPT</a> <span class="repo-full">thu-coai/CDial-GPT</span></div>
    <div class="repo-desc">LCCC, a 12M-dialogue cleaned Chinese conversation corpus, with pre-trained dialogue models. NLPCC 2020 Best Student Paper.</div>
    <div class="repo-badges"><a href="https://github.com/thu-coai/CDial-GPT"><img class="inline-badge" src="https://img.shields.io/github/stars/thu-coai/CDial-GPT?style=social" alt="GitHub stars for thu-coai/CDial-GPT" loading="lazy"></a> <a href="https://huggingface.co/datasets/thu-coai/lccc"><img class="inline-badge" src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fthu-coai%2Flccc&query=%24.downloads&label=downloads%2Fmonth&logo=huggingface&logoColor=FFD21E&style=social" alt="Hugging Face monthly downloads for thu-coai/lccc" loading="lazy"></a></div>
  </li>
  <li class="repo-row">
    <div class="repo-name"><a href="https://github.com/thu-coai/CrossWOZ">CrossWOZ</a> <span class="repo-full">thu-coai/CrossWOZ</span></div>
    <div class="repo-desc">The first large-scale Chinese cross-domain task-oriented dialogue dataset, with a user simulator and benchmark models. TACL 2020.</div>
    <div class="repo-badges"><a href="https://github.com/thu-coai/CrossWOZ"><img class="inline-badge" src="https://img.shields.io/github/stars/thu-coai/CrossWOZ?style=social" alt="GitHub stars for thu-coai/CrossWOZ" loading="lazy"></a> <a href="https://huggingface.co/datasets/GEM/CrossWOZ"><img class="inline-badge" src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FGEM%2FCrossWOZ&query=%24.downloads&label=downloads%2Fmonth&logo=huggingface&logoColor=FFD21E&style=social" alt="Hugging Face monthly downloads for GEM/CrossWOZ" loading="lazy"></a></div>
  </li>
  <li class="repo-row">
    <div class="repo-name"><a href="https://github.com/thu-coai/KdConv">KdConv</a> <span class="repo-full">thu-coai/KdConv</span></div>
    <div class="repo-desc">A multi-domain knowledge-driven dialogue dataset grounded in knowledge graphs, with benchmark models. ACL 2020.</div>
    <div class="repo-badges"><a href="https://github.com/thu-coai/KdConv"><img class="inline-badge" src="https://img.shields.io/github/stars/thu-coai/KdConv?style=social" alt="GitHub stars for thu-coai/KdConv" loading="lazy"></a> <a href="https://huggingface.co/datasets/thu-coai/kdconv"><img class="inline-badge" src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fthu-coai%2Fkdconv&query=%24.downloads&label=downloads%2Fmonth&logo=huggingface&logoColor=FFD21E&style=social" alt="Hugging Face monthly downloads for thu-coai/kdconv" loading="lazy"></a></div>
  </li>
  <li class="repo-row">
    <div class="repo-name"><a href="https://github.com/stanford-futuredata/colbert-serve">ColBERT-Serve</a> <span class="repo-full">stanford-futuredata/colbert-serve</span></div>
    <div class="repo-desc">Memory-mapped, multi-stage serving for ColBERT with hybrid scoring; 90% less RAM. ECIR 2025.</div>
    <div class="repo-badges"><a href="https://github.com/stanford-futuredata/colbert-serve"><img class="inline-badge" src="https://img.shields.io/github/stars/stanford-futuredata/colbert-serve?style=social" alt="GitHub stars for stanford-futuredata/colbert-serve" loading="lazy"></a></div>
  </li>
</ul>
  {% endif %}
  {% endfor %}

{% else %}

<!-- Display projects without categories -->

{% assign sorted_projects = site.projects | sort: "importance" %}

  <!-- Generate cards for each project -->

{% if page.horizontal %}

  <div class="container">
    <div class="row row-cols-1 row-cols-md-2">
    {% for project in sorted_projects %}
      {% include projects_horizontal.liquid %}
    {% endfor %}
    </div>
  </div>
  {% else %}
  <div class="row row-cols-1 row-cols-md-3">
    {% for project in sorted_projects %}
      {% include projects.liquid %}
    {% endfor %}
  </div>
  {% endif %}
{% endif %}
</div>
