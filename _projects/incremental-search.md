---
layout: page
title: incremental search
description: (archived 2019)
img: assets/img/project-incremental-search.jpg
importance: -2019
category: inactive

years: [2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008]
sort_by: type, year
type_order: [article, inproceedings]
order: descending_
---

<img style="float: left; margin: 5px 10px 5px 0px; width: 240px; box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);" src="/assets/img/project-incremental-search.jpg">
Incremental search algorithms reuse information from previous searches to speed up the current search and solve search problems potentially much faster than solving them repeatedly from scratch. They are widely popular in solving dynamic path-planning problems such as navigation for unmanned ground vehicles and motion planning for articulated robots. For example, existing incremental search algorithms such as D* and D* Lite have been adapted for use with much success in various robotic applications including the Mars rovers and autonomous vehicles in the DARPA Urban Challenge.

Our current research focus is on investigating the applicability of such algorithms in conjunction with answer set programming to solve multi-agent pathfinding problems. Applications of this problem is numerous including the coordination of robots in automated warehouses.<br/>
<font size = "1"><br/></font>


<h3>selected publications</h3>

<p></p>

<div class="publications">

{%- for y in page.years %}
  {% bibliography -f {{ site.scholar.primary_bibliography }} -q @*[year={{y}},project-selected=true,project=incremental-search]* %}
{% endfor %}

</div>




