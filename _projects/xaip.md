---
layout: page
title: explainable planning and scheduling
img: assets/img/project-xaip.jpg
importance: 1
category: active

years: [2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008]
sort_by: type, year
type_order: [article, inproceedings]
order: descending_
---
<img style="float: left; margin: 5px 10px 5px 0px; width: 240px; box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);" src="/assets/img/project-xaip.jpg">
In human-aware planning and scheduling systems, when the agent recommends a plan or schedule to a human user, it is often the case that the user might not understand why the recommendation is good, for example, compared to an alternative in the user’s mind. In such a scenario, there is a need for the agent to *explain* its recommendation to the user, providing them with the necessary information to understand properties of the recommendation (e.g., optimality, feasibility, etc.).

We are approaching this problem from a knowledge representation and reasoning (KR) perspective, where we represent the mental models of both the agent and the human user using logical facts and rules. Within this framework, we adapt and generalize KR notions (e.g., entailment, hitting sets, model counting) to solve this problem.

<h3>sponsors</h3>

<p></p>

<img style="float: left; margin: 0px 10px 0px 0px;" src="/assets/img/nsf.png" width="50px">
[Collaborative Research: RI: Small: End-to-end Learning of Fair and Explainable Schedules for Court Systems](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2232055).
National Science Foundation (2023 – 2026).

<img style="float: left; margin: 0px 10px 0px 0px;" src="/assets/img/jpmc.jpg" width="50px">
[Improving Client Experience Through Goal Recognition and Explainable Assistance in Adaptive Systems](https://www.jpmorgan.com/technology/artificial-intelligence/research-awards/faculty-research-awards-2022).
J.P. Morgan Chase Bank (2022 – 2023).

<img style="float: left; margin: 0px 10px 0px 0px;" src="/assets/img/nsf.png" width="50px">
[RI: Small: Collaborative Research: Preference Elicitation and Device Scheduling for Smart Homes](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1812619).
National Science Foundation (2018 – 2021).<br/>
<font size = "1"><br/></font>


<h3>selected publications</h3>

<p></p>

<div class="publications">

{%- for y in page.years %}
  {% bibliography -f {{ site.scholar.primary_bibliography }} -q @*[year={{y}},project-selected=true,project=xaip]* %}
{% endfor %}

</div>




