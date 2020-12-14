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

# CASES PREDICTIOR
# K-FOLD CROSS VALIDATION
# covid_traffic.k_folds_cross_validation(
#     1, cases, traffic, days, pred_type='cases', model_type='lasso', Q=1, K='N/A', C=10)

# # POLY FEATURES CROSS VALIDATION
# covid_traffic.poly_feature_cross_validation(
#     2, cases, traffic, days, pred_type='cases', model_type='lasso', folds=2, K='N/A', C=10)

# # POLY FEATURES CROSS VALIDATION
# covid_traffic.c_penalty_cross_validation(
#     3, cases, traffic, days, pred_type='cases', model_type='lasso', folds=2, Q=5, K='N/A')

# PLOT PREDICTIONS
covid_traffic.plot_predictions(
    4, cases, traffic, days, pred_type='cases', model_type='lasso', folds=2, Q=5, K='N/A', C=10)