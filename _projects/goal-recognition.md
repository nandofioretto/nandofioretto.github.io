---
layout: page
title: goal recognition and design
img: assets/img/project-grd.jpg
importance: 2
category: active

years: [2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008]
sort_by: type, year
type_order: [article, inproceedings]
order: descending_
---
<img style="float: left; margin: 5px 10px 5px 0px; width: 240px; box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);" src="/assets/img/project-grd.jpg">
In goal recognition problems, the goal is to identify the goal of an observed agent as quickly as possible before they reach their goal. This problem can be applied to a number of applications ranging from software personal assistants and robots that anticipate the needs of humans; intelligent tutoring systems that recognize sources of confusion or misunderstanding in students through their interactions with the system; and security applications that recognize terrorists plans.

In goal recognition design problems, the objective is to identify the best ways to modify the underlying environment of the agents in such a way that they are forced to reveal their goals as early as possible. This problem is typically applicable in the same goal recognition applications as long as the underlying environment can be modified. For example, in intelligent tutoring systems, it is possible to modify the sequence or type of questions asked so that students’ misunderstanding can be identified sooner. Similarly, in security applications, it may be possible to block some paths so that an attacker’s plan can be recognized earlier. To learn more, please see our [tutorial on goal recognition design](https://www.youtube.com/watch?v=_17faWBmF14){:target="_blank"}, which we gave at ICAPS 2019. Slides available [here](/assets/pdf/tutorial-icaps19-grd){:target="_blank"}.

Our current research is on both the algorithmic back-end, where the goal is to enrich the models so as to better capture more realistic problem characteristics and develop efficient algorithms to solve them, as well as the user interface front-end, where the goal is to develop visualization interfaces that allow our methods to be adapted to specific applications and evaluated with human users.



<h2>sponsors</h2>

<p></p>

<img style="float: left; margin: 0px 10px 0px 0px;" src="/assets/img/jpmc.jpg" width="50px">
[Improving Client Experience Through Goal Recognition and Explainable Assistance in Adaptive Systems](https://www.jpmorgan.com/technology/artificial-intelligence/research-awards/faculty-research-awards-2022).
J.P. Morgan Chase Bank (2022 – 2023).

<img style="float: left; margin: 0px 10px 0px 0px;" src="/assets/img/boeing.png" width="50px">
[Integrated Computational and Cognitive Workflows for Improved Security and Usability](#). <br/>
Boeing (2019).

<img style="float: left; margin: 0px 10px 0px 0px;" src="/assets/img/nsf.png" width="50px">
[CAREER: Decentralized Constraint-based Optimization for Multi-Agent Planning and Coordination](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1550662).
National Science Foundation (2016 – 2021).<br/>
<font size = "1"><br/></font>


<h3>selected publications</h3>

<p></p>

<div class="publications">

{%- for y in page.years %}
  {% bibliography -f {{ site.scholar.primary_bibliography }} -q @*[year={{y}},project-selected=true,project=grd]* %}
{% endfor %}

</div>
