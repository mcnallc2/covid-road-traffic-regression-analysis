import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv("../../data/formatted_data_new.csv")
data = data.drop(columns=["Date"])

# create new df and rename column to "Cases"
cases = data[["Confirmed Cases-6"]]
cases = cases.rename(columns={"Confirmed Cases-6" : "Cases"})

# make new columns of previous cases-N by shifting previous cases by 1
cases.loc[:, "Cases-1"] = cases.loc[:,"Cases"].shift()
cases.loc[:, "Cases-2"] = cases.loc[:,"Cases-1"].shift()
cases.loc[:, "Cases-3"] = cases.loc[:,"Cases-2"].shift()
cases.loc[:, "Cases-4"] = cases.loc[:,"Cases-3"].shift()
cases.loc[:, "Cases-5"] = cases.loc[:,"Cases-4"].shift()
cases.loc[:, "Cases-6"] = cases.loc[:,"Cases-5"].shift()
cases.loc[:, "Cases-7"] = cases.loc[:,"Cases-6"].shift()

# drop and NA entries
cases =  cases.dropna()
print(cases)

# input X will be previous cases
X = cases.drop(columns=["Cases"]).to_numpy()
# output y will be next days covid cases
y = cases.loc[:,"Cases"].to_numpy()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
t_train = range(len(y_train))
t_test = range(len(y_test))

# Train linear regression model
model = LinearRegression().fit(X_train, y_train)
print(model.coef_)

# Predict on traing data and plot 
y_pred = model.predict(X_train)
plt.figure()
plt.plot(t_train, y_pred, "k-", label="Train predictions")
plt.plot(t_train, y_train, "b--", label = "Train true")
plt.legend()
plt.xlabel("Time (days)")
plt.ylabel("Covid cases")
plt.savefig("Train.png")

# Predict on test data and plot 
y_pred = model.predict(X_test)
plt.figure()
plt.plot(t_test, y_test, "k-", label="Test true")
plt.plot(t_test, y_pred, "b--", label = "Test predictions")
plt.legend()
plt.xlabel("Time (days)")
plt.ylabel("Covid cases")
plt.savefig("Test.png")