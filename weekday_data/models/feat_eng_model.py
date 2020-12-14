import pandas as pd
import matplotlib.pyplot as plt
from covid_traffic_modelling import CovidTrafficModelling
from sklearn.preprocessing import PolynomialFeatures
import sys

data = pd.read_csv("../../data/formatted_data.csv")
days = data.iloc[:, 0]
cases = data.iloc[:, 1]
traffic = data.iloc[:, 2]

print(sys.argv)
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

# Try different amounts of features 
for p in range(1, 6):
    print("\n\n--- Degree: " + str(p) + " ---")
    poly = PolynomialFeatures(p)

    # TRAFFIC ==> CASES
    fe_traffic = poly.fit_transform(traffic)
    covid_traffic.cases_predictor(
        2, cases, fe_traffic, days, 'linreg', K='N/A', C='N/A')

    # CASES ==> TRAFFIC
    fe_cases = poly.fit_transform(cases)
    covid_traffic.traffic_predictor(
        3, fe_cases, traffic, days, 'linreg', K='N/A', C='N/A')
