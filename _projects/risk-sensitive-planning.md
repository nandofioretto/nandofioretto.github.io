---
layout: page
title: risk-sensitive planning
description: (archived 2017)
img: assets/img/project-risk-sensitive-planning.jpg
importance: -2017
category: inactive

years: [2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008]
sort_by: type, year
type_order: [article, inproceedings]
order: descending_
---

<img style="float: left; margin: 5px 10px 5px 0px; width: 240px; box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);" src="/assets/img/project-risk-sensitive-planning.jpg">
Many real-world planning problems occur in environments where there may be incomplete information or where actions may not always lead to the same results. Examples include planning for retirement, where the state of the economy in the future is uncertain, and planning in logistics, where the duration of travel between two cities is uncertain due to potential congestion.

A Markov decision process (MDP) is a popular framework for modeling decision making in these kinds of problems, where an agent needs to plan a sequence of actions that maximizes its chances of reaching its goal. A partially observable MDP (POMDP) is an extension where the world that the agent is operating in is only partially observable, and a decentralized (PO)MDP is an extension where a team of agents needs to collectively plan their joint actions.

We proposed the risk-sensitive MDP and POMDP models, where the objective is to find a policy that maximizes the probability of reaching a goal state. Such a model and its accompanying algorithms are useful in high-risk applications, where a single failure can be catastrophic.

<h2>sponsors</h2>

<p></p>

<img style="float: left; margin: 0px 10px 0px 0px;" src="/assets/img/nsf.png" width="50px">
[CAREER: Decentralized Constraint-based Optimization for Multi-Agent Planning and Coordination](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1550662).
National Science Foundation (2016 – 2021).<br/>
<font size = "1"><br/></font>


<h3>selected publications</h3>

<p></p>

<div class="publications">

{%- for y in page.years %}
  {% bibliography -f {{ site.scholar.primary_bibliography }} -q @*[year={{y}},project-selected=true,project=risk-sensitive-planning]* %}
{% endfor %}

</div>


