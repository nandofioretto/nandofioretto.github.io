---
layout: page
permalink: /publications/
title: publications
description: 
article_years: [2023,2022,2019,2018,2017,2016,2012,2010]
conference_years: [2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008]
ext_abstract_years: [2021,2020,2018,2015,2014,2012,2011,2008]
workshop_years: [2022,2021]
sort_by: type, year
type_order: [article, inproceedings]
order: descending_
nav: true
nav_order: 2
---
 
<!-- _pages/publications.md -->
<div class="publications">

<h3>journal and magazine articles</h3>
{%- for y in page.article_years %}
	<h2 class="year">{{y}}</h2>
	{% bibliography -f {{ site.scholar.primary_bibliography }} -q @article[year={{y}}]* %}
{% endfor %}

<br/>

<h3>conference publications (full papers)</h3>
{%- for y in page.conference_years %}
	<h2 class="year">{{y}}</h2>
	{% bibliography -f {{ site.scholar.primary_bibliography }} -q @inproceedings[year={{y}}]* %}
{% endfor %}

<br/>

<h3>conference publications (short papers, extended abstracts)</h3>
{%- for y in page.ext_abstract_years %}
	<h2 class="year">{{y}}</h2>
	{% bibliography -f {{ site.scholar.secondary_bibliography }} -q @inproceedings[year={{y}}]* %}
{% endfor %}

</div>
