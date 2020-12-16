import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from covid_traffic_modelling import CovidTrafficModelling

data = pd.read_csv("../../data/formatted_data_new.csv")
days = data.iloc[:, 0]
cases = data.iloc[:, 1]
traffic = data.iloc[:, 2]

covid_traffic = CovidTrafficModelling()

## convert data to weekday only
[cases, traffic, days] = covid_traffic.remove_weekends(cases, traffic, days)

## format data
cases = pd.DataFrame(cases).to_numpy()
traffic = pd.DataFrame(traffic).to_numpy()
days = pd.DataFrame(days).to_numpy()

# ## DATA PLOT
# covid_traffic.plot_data(1, cases, traffic, days)

## concat days and trafffic data for train/test split
X = np.column_stack((days, traffic))
y = cases

## 80/20 (train/test) split of data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

## PLOT PREDICTIONS
covid_traffic.plot_baseline(1, X_train, X_test, y_train, y_test, model_type='baseline')