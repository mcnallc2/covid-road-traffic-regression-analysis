import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

class CovidTrafficModelling:

    def remove_weekends(self, cases_df, traffic_df, days):
        # shift traffic_day list 3 days as first day is a wednesday
        for i in range(len(days)):
            days[i] += 3

        # creating a list of the day numbers
        # (1st Jan = day 1)
        # init a list for week-day traffic and covid cases
        day_num = []
        traffic_wd = []
        cases_wd = []
        day = 0

        # for each traffic day
        for i in range(len(days)):
            # if the day is not a sat (6) or sun (0)
            if not ((days[i] % 7 == 6) or (days[i] % 7 == 0)):
                traffic_wd.append(traffic_df[i])
                cases_wd.append(cases_df[i])
                day_num.append(day)
                day += 1

        days = pd.DataFrame(day_num).to_numpy()

        return [cases_wd, traffic_wd, days]


    def plot_data(self, fig, cases, traffic, days):
        fig, ax1 = plt.subplots(fig)
        ax1.set_title("Days vs. Traffic & Cases")
        ax1.set_xlabel("days")
        ax1.plot(days, cases, color="r")
        ax1.set_ylabel("traffic")

        ax2 = ax1.twinx()
        ax2.set_ylabel("cases")
        ax2.plot(days, traffic)
        plt.savefig('../plots/weekday_data_plot.png')
        plt.show()


    def cases_predictor(self, fig, cases, traffic, days, model_type, K, C):
        print("-> Use traffic to predict case figures")
        kf = KFold(n_splits=5)
        plt.figure(fig)
        plt.plot(days, cases)
        for train, test in kf.split(traffic):
            ##
            if model_type == 'knn':
                model = KNeighborsRegressor(n_neighbors=K).fit(traffic[train], cases[train])
            elif model_type == 'lasso':
                model = Lasso(alpha=1/2*C).fit(traffic[train], cases[train])
            elif model_type == 'ridge':
                model = Ridge(alpha=1/2*C).fit(traffic[train], cases[train])
            else:
                model = LinearRegression().fit(traffic[train], cases[train])
            ##
            predictions = model.predict(traffic[test])
            predictions = [round(num[0]) for num in predictions]
            print("mse: ", mean_squared_error(cases[test], predictions))
            # print(cases[test], "ASS\n", predictions)
            print("Accuracy: ", accuracy_score(cases[test], predictions))
            print("R2: ", r2_score(cases[test], predictions))
            plt.plot(days[test], predictions, c="lime")
        plt.title("Model using traffic to predict cases")
        plt.xlabel("Days")
        plt.ylabel("Cases")
        plt.legend(["training cases", "predicted cases"])
        plt.savefig('../plots/knn_cases_pred.png')
        plt.show()


    def traffic_predictor(self, fig, cases, traffic, days, model_type, K, C):
        print("-> Use cases to predict traffic figures")
        kf = KFold(n_splits=5)
        plt.figure(fig)
        plt.plot(days, traffic)
        for train, test in kf.split(cases):
            ##
            if model_type == 'knn':
                model = KNeighborsRegressor(n_neighbors=K).fit(cases[train], traffic[train])
            elif model_type == 'lasso':
                model = Lasso(alpha=1/2*C).fit(cases[train], traffic[train])
            elif model_type == 'ridge':
                model = Ridge(alpha=1/2*C).fit(cases[train], traffic[train])
            else:
                model = LinearRegression().fit(cases[train], traffic[train])
            ##
            # predictions = model.predict(traffic[test])
            predictions = model.predict(cases[test])
            predictions = [round(num[0]) for num in predictions]
            print("mse: ", mean_squared_error(traffic[test], predictions))
            print("Accuracy: ", accuracy_score(traffic[test], predictions))
            plt.plot(days[test], predictions, c="lime")
            print("R2: ", r2_score(traffic[test], predictions))
        plt.title("Model using cases to predict traffic")
        plt.xlabel("Days")
        plt.ylabel("Traffic")
        plt.legend(["training traffic", "predicted traffic"])
        plt.savefig(f'../plots/{model_type}_traffic_pred.png')
        plt.show()