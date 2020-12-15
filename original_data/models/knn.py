import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from evaluate_models import EvaluateModels
from cross_validation import CrossValidation

cross_validate = CrossValidation()
evaluate = EvaluateModels()

# Set up data
data = pd.read_csv("../../data/formatted_data_new.csv")
days = data.iloc[:, 0]
cases = data.iloc[:, 1]
traffic = data.iloc[:, 2]

cases_df = cases
traffic_df = traffic
cases = pd.DataFrame(cases).to_numpy()
traffic = pd.DataFrame(traffic).to_numpy()
cases = cases.reshape(-1, 1)

# Cases, Traffic Plot
fig, ax1 = plt.subplots()
ax1.set_title("Days vs. Traffic & Cases")
ax1.set_xlabel("days")
ax1.plot(days, cases, color="r")
ax1.set_ylabel("traffic")

ax2 = ax1.twinx()
ax2.set_ylabel("cases")
ax2.plot(days, traffic)
plt.show()
##########################

# Choose optimal KFold
print("Performing Cross Validation...")

cross_validate.k_folds_cross_validation(
    1, cases, traffic, days, pred_type='cases', model_type='knn', Q=1, K=18, C='N/A')

# Choose optimal polynomial features
cross_validate.poly_feature_cross_validation(
    2, cases, traffic, days, pred_type='cases', model_type='knn', folds=2, K=18, C='N/A')

# Choose optimal neighbours for KNN = 18
cross_validate.knn_cross_validation(
    3, cases, traffic, days, pred_type='cases', model_type='knn', folds=2, Q=1, C='N/A')

# TRAFFIC ==> CASES
kf = KFold(n_splits=2)
pred_array = []
y = []
plt.figure(5)
plt.plot(days, cases)
for train, test in kf.split(traffic):
    model = KNeighborsRegressor(n_neighbors=18).fit(
        traffic[train], cases[train])
    predictions = model.predict(traffic[test])

    plt.plot(days[test], predictions, c="lime")

    y = y + cases[test].tolist()
    pred_array = pred_array + predictions.tolist()
## 
evaluate.evaluate_model(
    pred_type='cases', model_type='knn', y=y, y_pred=pred_array)
plt.title("KNN Model using traffic to predict cases")
plt.xlabel("Days")
plt.ylabel("Cases")
plt.legend(["training cases", "predicted cases"])
plt.show()


# CASES ==> TRAFFIC
kf = KFold(n_splits=5)
pred_array = []
y = []
plt.figure(5)
plt.plot(days, traffic)
for train, test in kf.split(cases):
    model = KNeighborsRegressor(n_neighbors=16).fit(
        cases[train], traffic[train])
    predictions = model.predict(cases[test])

    plt.plot(days[test], predictions, c="lime")

    y = y + traffic[test].tolist()
    pred_array = pred_array + predictions.tolist()

evaluate.evaluate_model(
    pred_type='traffic', model_type='knn', y=y, y_pred=pred_array)
plt.title("KNN Model using cases to predict traffic")
plt.xlabel("Days")
plt.ylabel("Traffic")
plt.legend(["training traffic", "predicted traffic"])
plt.show()
