import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score


data = pd.read_csv("formatted_data.csv")
days = data.iloc[:, 0]
cases = data.iloc[:, 1]
traffic = data.iloc[:, 2]

cases_df = cases
traffic_df = traffic

## shift traffic_day list 3 days as first day is a wednesday
for i in range(len(days)):
  days[i] += 3

## creating a list of the day numbers
## (1st Jan = day 1)
## init a list for week-day traffic and covid cases
day_num = []
traffic_wd = []
cases_wd = []
day = 0

## for each traffic day
for i in range(len(days)):
    ## if the day is not a sat (6) or sun (0)
    if not ((days[i] % 7 == 6) or (days[i] % 7 == 0)):
        traffic_wd.append(traffic_df[i])
        cases_wd.append(cases_df[i])
        day_num.append(day)
        day += 1

days = pd.DataFrame(day_num).to_numpy()
cases = pd.DataFrame(cases_wd).to_numpy()
traffic = pd.DataFrame(traffic_wd).to_numpy()
cases = cases.reshape(-1, 1)

avg_cases = cases_df.rolling(window=5).mean()
avg_traffic = traffic_df.rolling(window=15).mean()

avg_cases = avg_cases.fillna(0)
avg_traffic = avg_traffic.fillna(0)

avg_traffic = pd.DataFrame(avg_traffic).to_numpy()
avg_cases = pd.DataFrame(avg_cases).to_numpy()

avg_traffic = avg_traffic.reshape(-1, 1)
avg_cases = avg_cases.reshape(-1, 1)

fig, ax1 = plt.subplots()
ax1.set_title("Days vs. Traffic & Cases")
ax1.set_xlabel("days")
ax1.plot(days, cases, color="r")
ax1.set_ylabel("traffic")

ax2 = ax1.twinx()
ax2.set_ylabel("cases")
ax2.plot(days, traffic)
plt.savefig('./weekday_data/plots/weekday_data_plot.png')
plt.show()


# TRAFFIC ==> CASES
print("-> Use traffic to predict case figures")
print("---------------------------------------")
kf = KFold(n_splits=5)
plt.figure(5)
plt.plot(days, cases)
for train, test in kf.split(traffic):
    model = LinearRegression().fit(traffic[train], cases[train])
    predictions = model.predict(traffic[test])
    predictions = [round(num[0]) for num in predictions]
    print("mse: ", mean_squared_error(cases[test], predictions))
    print("Accuracy: ", accuracy_score(cases[test], predictions))
    print("R2: ", r2_score(cases[test], predictions))
    plt.plot(days[test], predictions, c="lime")
plt.title("Model using traffic to predict cases")
plt.xlabel("Days")
plt.ylabel("Cases")
plt.legend(["training cases", "predicted cases"])
plt.savefig('./weekday_data/plots/linreg_cases_pred.png')
plt.show()


# CASES ==> TRAFFIC
print("-> Use cases to predict traffic figures")
print("---------------------------------------")
kf = KFold(n_splits=5)
pred_array = []
plt.figure(5)
plt.plot(days, traffic)
for train, test in kf.split(cases):
    model = LinearRegression().fit(cases[train], traffic[train])
    predictions = model.predict(cases[test])
    predictions = [round(num[0]) for num in predictions]
    print("mse: ", mean_squared_error(traffic[test], predictions))
    print("Accuracy: ", accuracy_score(traffic[test], predictions))
    print("R2: ", r2_score(traffic[test], predictions))
    plt.plot(days[test], predictions, c="lime")
plt.title("Model using cases to predict traffic")
plt.xlabel("Days")
plt.ylabel("Traffic")
plt.legend(["training traffic", "predicted traffic"])
plt.savefig('./weekday_data/plots/linreg_traffic_pred.png')
plt.show()
