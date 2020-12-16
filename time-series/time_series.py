from covid_traffic_modelling import CovidTrafficModelling
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



covid_traffic = CovidTrafficModelling()


## First evaluating a baseline classifier
data = pd.read_csv("../data/formatted_data_new.csv")
days = data.iloc[:, 0]
cases = data.iloc[:, 1]
traffic = data.iloc[:, 2]
## concat days and trafffic data for train/test split
X = np.column_stack((days, traffic))
y = cases
## 80/20 (train/test) split of data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
## PLOT BASELINE
covid_traffic.plot_baseline(6, X_train, X_test, y_train, y_test, model_type='baseline')


## Now time series modelling
raw_data = pd.read_csv("../data/formatted_data_new.csv")
raw_data = raw_data.drop(columns=["Date"])
traffic = raw_data[["Total Traffic"]]

# create new df and rename column to "Cases"
data = raw_data[["Confirmed Cases-6"]]
data = data.rename(columns={"Confirmed Cases-6" : "Cases"})

# make new columns of previous data-N by shifting previous data by 1
data.loc[:, "Cases-1"] = data.loc[:,"Cases"].shift()
data.loc[:, "Cases-2"] = data.loc[:,"Cases-1"].shift()
data.loc[:, "Cases-3"] = data.loc[:,"Cases-2"].shift()
data.loc[:, "Cases-4"] = data.loc[:,"Cases-3"].shift()
data.loc[:, "Cases-5"] = data.loc[:,"Cases-4"].shift()
data.loc[:, "Cases-6"] = data.loc[:,"Cases-5"].shift()
data.loc[:, "Cases-7"] = data.loc[:,"Cases-6"].shift()

data.loc[:, "Traffic"] =  traffic.loc[:,"Total Traffic"].rolling(7).mean()
# data.loc[:, "Traffic-7"] =  traffic.loc[:,"Total Traffic"].shift(7).rolling(7).mean()

# drop and NA entries
data =  data.dropna()
print(data.iloc[:, 1:])

#  plot the training data
covid_traffic.plot_data(1, cases=data.iloc[:, 1], traffic=data.iloc[:, 8], days=data.index )

# input X will be previous data
X = data.drop(columns=["Cases"]).to_numpy()
# output y will be next days covid data
y = data.loc[:,"Cases"].to_numpy()
days = (data.index).to_numpy()

X = np.column_stack((days, X))
## 80/20 (train/test) split of data 
X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2, shuffle=False)

## K-FOLD CROSS VALIDATION
covid_traffic.k_folds_cross_validation(2, X_train, y_train, model_type="ridge", Q=1, K='N/A', C=1)

## POLY FEATURES CROSS VALIDATION
covid_traffic.poly_feature_cross_validation(
    3, X_train, y_train, model_type='ridge', folds=100, K='N/A', C=1)

## C PENALTY CROSS VALIDATION
covid_traffic.c_penalty_cross_validation(
    4, X_train, y_train, model_type='ridge', folds=100, Q=1, K='N/A')

## PLOT PREDICTIONS
covid_traffic.plot_predictions(5, X_train, X_test, y_train, y_test, model_type='ridge', folds=100, Q=1, K='N/A', C=1000)