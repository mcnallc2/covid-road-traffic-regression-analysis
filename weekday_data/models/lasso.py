import pandas as pd
import matplotlib.pyplot as plt
from covid_traffic_modelling import CovidTrafficModelling

data = pd.read_csv("../../data/formatted_data_new.csv")
days = data.iloc[:, 0]
cases = data.iloc[:, 1]
traffic = data.iloc[:, 2]

covid_traffic = CovidTrafficModelling()

# convert data to weekday only
weekday_data = covid_traffic.remove_weekends(cases, traffic, days)
cases = weekday_data[0]
traffic = weekday_data[1]
days = weekday_data[2]

# format data
cases_df = cases
traffic_df = traffic
cases = pd.DataFrame(cases).to_numpy()
traffic = pd.DataFrame(traffic).to_numpy()
cases = cases.reshape(-1, 1)

# DATA PLOT
covid_traffic.plot_data(1, cases, traffic, days)

# TRAFFIC ==> CASES
covid_traffic.cases_predictor(2, cases, traffic, days, 'lasso', K='N/A', C=10)

# CASES ==> TRAFFIC
covid_traffic.traffic_predictor(3, cases, traffic, days, 'lasso', K='N/A', C=10)