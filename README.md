# covid-road-traffic-regression-analysis

## The Task
For this Machine Learning (CS7CS4) group project we are required to pick a problem, and develop a learning algorithm to solve it. We are required to gather raw data to be pre-processed, before running a feature engineering and model selection process. The performance of final models should be evaluated against a reasonable baseline predictor. Two significantly different types of models should be investigated (e.g. logistic regression and kNN classifier).

## The Chosen Problem
The question we wish to investigate is how strong is the correlation between peoples movement and the number of new infection cases. Coming up with a direct measurement of peoples movement is challenging, so we are using a proxy metric, road traffic volumes. An interesting follow on question is whether increasing road traffic (which represents peoples increasing movement) cause more Covid-19 cases, or is it the increasing Covid-19 cases which causes the road traffic conditions (people may move around dependant on their perception of threat from the virus). Since machine learning methods only find correlation and not causation, this is an important issue to bear in mind for this project.

We will be looking at this problem using data from the Republic of Ireland only. 
Our two sources are: 
- Covid-19 statistics from the Health Surveillance Protection Center (HSPC). https://covid-19.geohive.ie/datasets/d8eb52d56273413b84b0187a4e9117be
- Transport Infrastructure Ireland (TII) traffic counter data. https://www.nratrafficdata.ie/

TII maintains 370 traffic counters across Ireland's road network. The traffic counters are located on motorways and national primary roads and count the number of each type of vehicle (car, HGV, motorbike, etc.) passing over the road sensors. To collect the TII traffic data, we have written write a web scraper that will collect the data from the 370 different sites and aggregate it together into a total daily figure. Collecting the HSPC data is more straightforward as they have provided an easy way to download the data online.
