---
layout: default
permalink: /group/
title: group
description:
nav: true
nav_order: 4
display_categories: [Postdocs and PhD Students, MS and BS Students, Former Students]
horizontal: false
---

# **T**rustworthy and **Re**sponsible **L**earning and **O**ptimization (**TReLO**) Group

I lead the TReLO group where I am fortunate to work with an amazing set of students and collaborators.
We work on foundational topics relating to machine learning and optimization, privacy and fairness.
We often ground our research in applications at the intersection of physical sciences and energy, as
well as policy and decision making.



<!-- pages/projects.md -->
<div class="projects">
{%- if site.enable_project_categories and page.display_categories %}
  <!-- Display categorized projects -->
  {%- for category in page.display_categories %}
  <h2 class="category">{{ category }}</h2>
  {%- assign categorized_projects = site.group | where: "category", category -%}
  {%- assign sorted_projects = categorized_projects | sort: "importance" %}
  <!-- Generate cards for each project -->
  {% if page.horizontal -%}
  <div class="container">
    <div class="row row-cols-2">
    {%- for project in sorted_projects -%}
      {% include projects_horizontal.html %}
    {%- endfor %}
    </div>
  </div>
  {%- else -%}
  <div class="grid">
    {%- for project in sorted_projects -%}
      {% include group.html %}
    {%- endfor %}
  </div>
  {%- endif -%}
  {% endfor %}

{%- else -%}
<!-- Display projects without categories -->
  {%- assign sorted_projects = site.group | sort: "importance" -%}
  <!-- Generate cards for each project -->
  {% if page.horizontal -%}
  <div class="container">
    <div class="row row-cols-2">
    {%- for project in sorted_projects -%}
      {% include projects_horizontal.html %}
    {%- endfor %}
    </div>
  </div>
  {%- else -%}
  <div class="grid">
    {%- for project in sorted_projects -%}
      {% include group.html %}
    {%- endfor %}
  </div>
  {%- endif -%}
{%- endif -%}
</div>

| Student Name     | Visit period| Next known position                      |
|------------------|-------------|------------------------------------------|
| [**St John Grimbly**](https://stjohngrimbly.com), MS	Visitor | Winter 2023 | PhD Student at  University of Cape Town |
| **Jayanta Mandi**,      PhD Visitor | Summer 2023 | Postdoc at  Vrije Universiteit Brussel |
| **Michele Marchiani**, MS	Visitor | Winter 2022 | |
| **Rakshit Naidu**, MS	Visitor | Summer 2022 | PhD student at Georgia Tech |
| **Saswat Das**, BS	Visitor  | Summer 2022  | PhD student at UVA |
| **Kyle Beiter**, NSF REU Student |  Summer 2021	 | MS at Syracsue |
| **Zhiyan Yao**, BS Visitor	| Summer 2020 | Software Engineer at Microsoft|
| **Anudit Nagar**, BSVisitor	| Summer 2020 | Co-Founder of The Convo Space  |
