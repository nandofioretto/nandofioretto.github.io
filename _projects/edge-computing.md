---
layout: page
title: edge computing
description: (archived 2021)
img: assets/img/project-edge-computing.jpg
importance: -2021
category: inactive

years: [2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008]
sort_by: type, year
type_order: [article, inproceedings]
order: descending_
---

<img style="float: left; margin: 5px 10px 5px 0px; width: 240px; box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);" src="/assets/img/project-edge-computing.jpg">
Edge computing is a method of optimizing cloud computing systems by performing data processing at the edge of the network, closer to the users and sources of data. As data processing is traditionally done in large data centers, typically located at the center of the network, the edge computing paradigm will reduce the communication bottleneck to the data centers, thereby improving overall performance. This becomes more important as the number of Internet-of-Things (IoT) devices increases.

Our research focuses on how to coordinate the computational nodes at the edge of the network in a distributed way to collaboratively process incoming data. Our current plan is to use a [distributed constraint optimization](projects/dcop/) (DCOP) formulation and to develop DCOP algorithms that are tailored to this specific application.

<h2>sponsors</h2>

<p></p>

<img style="float: left; margin: 0px 10px 0px 0px;" src="/assets/img/darpa.png" width="50px">
[Mission-oriented Adaptive Placement of Task and Data (MAP)](https://beta.sam.gov/awards/70580675%2BAWARD).<br/>
Defense Advanced Research Projects Agency (2017 – 2021).<br/>
<font size = "1"><br/></font>


<h3>selected publications</h3>

<p></p>

<div class="publications">

{%- for y in page.years %}
  {% bibliography -f {{ site.scholar.primary_bibliography }} -q @*[year={{y}},project-selected=true,project=edge-computing]* %}
{% endfor %}

</div>


