---
layout: page
title: smart grid and homes
description: (archived 2021)
img: assets/img/project-smartgrid.jpg
importance: -2021
category: inactive

years: [2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008]
sort_by: type, year
type_order: [article, inproceedings]
order: descending_
---

<img style="float: left; margin: 5px 10px 5px 0px; width: 240px; box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);" src="/assets/img/project-smartgrid.jpg">
Transitioning our aging power grid to a smart grid brings a number of benefits including increased efficiency and sustainability as renewable resources (e.g., solar and wind) are incorporated into the grid. Additionally, it will also improve reliability and resiliency as smart sensors will be deployed throughout the grid to detect, respond, and adapt to events such as faults and failures. Finally, through smart meters and other Internet-of-Things devices in smart homes, the smart grid will also bring economical benefits to both power producers and consumers, as producers can introduce real-time pricing to reduce peak power consumption, and home automation systems of consumers can adapt accordingly while satisfying the needs and constraints of the homeowners.

We are currently pursuing a number of parallel subprojects within this area:
- We are investigating appropriate formalisms and algorithms for the elicitation and scheduling problem of home automation systems. Home automation systems need to intelligently elicit preferences and constraints from homeowners regarding their use of smart devices as exhaustive elicitation is not feasible. Such systems also need to schedule the devices in an efficient manner while optimizing certain objectives (e.g., minimizing the monetary cost to the homeowner, minimizing the peak power demand) while satisfying the preferences and constraints elicited.
- We are working towards transitioning off-the-shelf scheduling and coordination algorithms towards a working prototype of a home automation system. The goal is to deploy this prototype system in our microgrid at NMSU. Such a system can also be a testbed for the algorithms developed above. [We also have a dataset that we curated for this problem](https://github.com/nandofioretto/SHDS_dataset).
- We are also participating in [PowerTAC](https://powertac.org/) – a trading agent competition, where an autonomous agent plays the role of a broker in an energy marketplace. The goal of the agent is to buy energy from wholesalers and sell them to consumers in such a way that maximizes their profit.

<h2>sponsors</h2>

<p></p>

<img style="float: left; margin: 0px 10px 0px 0px;" src="/assets/img/nsf.png" width="50px">
[RI: Small: Collaborative Research: Preference Elicitation and Device Scheduling for Smart Homes](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1812619).
National Science Foundation (2018 – 2021).

<img style="float: left; margin: 0px 10px 0px 0px;" src="/assets/img/nsf.png" width="50px">
[iCREDITS: Interdisciplinary Center of Research Excellence in Design of Intelligent Technologies for Smartgrids](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1345232).
National Science Foundation (2014 – 2019).<br/>
<font size = "1"><br/></font>


<h3>selected publications</h3>

<p></p>

<div class="publications">

{%- for y in page.years %}
  {% bibliography -f {{ site.scholar.primary_bibliography }} -q @*[year={{y}},project-selected=true,project=smartgrid]* %}
{% endfor %}

</div>

