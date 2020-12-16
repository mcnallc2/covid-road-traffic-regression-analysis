from sklearn.dummy import DummyRegressor
import pandas as pd 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from evaluate_models import EvaluateModels

evaluate = EvaluateModels()

data = pd.read_csv("../../data/formatted_data_new.csv")
days = data.iloc[:, 0]
cases = data.iloc[:, 1]
traffic = data.iloc[:, 2]

cases_df = cases
traffic_df = traffic
cases = pd.DataFrame(cases).to_numpy()
traffic = pd.DataFrame(traffic).to_numpy()
cases = cases.reshape(-1, 1)


# Traffic ==> Cases 
kf = KFold(n_splits=5)
plt.figure(5)
plt.plot(days, traffic)
y = [] 
p = []
for train, test in kf.split(traffic):
    dummy_reg = DummyRegressor()
    dummy_reg.fit(traffic[train], cases[train])
    predictions = dummy_reg.predict(traffic[test])

    plt.plot(days[test], predictions, c="lime")

    y = y + cases[test].tolist()
    p = p + predictions.tolist()
evaluate.evaluate_model(
    pred_type='cases', model_type='linear regression', y=y, y_pred=p)
plt.title("Baseline Model using traffic to predict cases")
plt.xlabel("Days")
plt.ylabel("Cases")
plt.legend(["training cases", "predicted cases"])
plt.show()


# Traffic ==> Cases 
kf = KFold(n_splits=5)
plt.figure(5)
plt.plot(days, cases)
y = [] 
p = []
for train, test in kf.split(cases):
    dummy_reg = DummyRegressor()
    dummy_reg.fit(cases[train], traffic[train])
    predictions = dummy_reg.predict(cases[test])

    plt.plot(days[test], predictions, c="lime")

    y = y + traffic[test].tolist()
    p = p + predictions.tolist()
evaluate.evaluate_model(
    pred_type='cases', model_type='linear regression', y=y, y_pred=p)
plt.title("Baseline Model using traffic to predict cases")
plt.xlabel("Days")
plt.ylabel("Cases")
plt.legend(["training cases", "predicted cases"])
plt.show()