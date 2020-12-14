import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
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


    def k_folds_cross_validation(self, fig, cases, traffic, days, pred_type, model_type, Q, K, C):
        ##
        k_folds = [2, 10, 25, 50, 100]

        ## init mean, var and std lists
        mean_mse = []
        var_mse = []
        std_mse = []

        ## for each k-fold value
        for folds in k_folds:
            ## run cross validation
            cross_val_results = self.cross_val_model(cases, traffic, days, pred_type, model_type, folds, Q, K, C)
            ## append append mean, var and std error values to lists
            mean_mse.append(cross_val_results[0])
            var_mse.append(cross_val_results[1])
            std_mse.append(cross_val_results[2])

        ## add results to errorbar plot
        plt.figure(fig)
        kf_vals = ['2', '10', '25', '50', '100']
        plt.errorbar(kf_vals, mean_mse, yerr=var_mse, capsize=5, ecolor='red', label='Mean prediction error with varience')
        plt.title(f'{pred_type} predictor - {model_type} - K-folds cross validation')
        plt.xlabel('K-folds')
        plt.ylabel('MSE')
        plt.legend()
        plt.savefig(f'../plots/cross_val/K-FOLDS_{pred_type}_{model_type}.png')
        plt.show()


    def poly_feature_cross_validation(self, fig, cases, traffic, days, pred_type, model_type, folds, K, C):
        ##
        q_range = [1, 2, 3, 4, 5]

        ## init mean, var and std lists
        mean_mse = []
        var_mse = []
        std_mse = []
        ##
        ## loop through Q values
        for Q in q_range:
            ##
            cross_val_results = self.cross_val_model(cases, traffic, days, pred_type, model_type, folds, Q, K, C)

            ## append append mean, var and std error values to lists
            mean_mse.append(cross_val_results[0])
            var_mse.append(cross_val_results[1])
            std_mse.append(cross_val_results[2])

        plt.figure(fig)
        q_vals = ['1', '2', '3', '4', '5']
        plt.errorbar(q_vals, mean_mse, yerr=var_mse, capsize=5, ecolor='red', label='Mean prediction error with varience')
        plt.title(f'{pred_type} predictor - {model_type} - Polynomial Features cross validation')
        plt.xlabel('Polynomial Features')
        plt.ylabel('MSE')
        plt.legend()
        plt.savefig(f'../plots/cross_val/POLY-FEATURES_{pred_type}_{model_type}.png')
        plt.show()


    def c_penalty_cross_validation(self, fig, cases, traffic, days, pred_type, model_type, folds, Q, K):
        ##
        C_range = [0.01, 0.1, 1, 10, 1000]

        ## init mean, var and std lists
        mean_mse = []
        var_mse = []
        std_mse = []

        ## loop through C values
        for C in C_range:
            ## run cross validation
            cross_val_results = self.cross_val_model(cases, traffic, days, pred_type, model_type, folds, Q, K, C)

            ## append append mean, var and std error values to lists
            mean_mse.append(cross_val_results[0])
            var_mse.append(cross_val_results[1])
            std_mse.append(cross_val_results[2])

        plt.figure(fig)
        c_vals = ['0.01', '0.1', '1', '10', '1000']
        plt.errorbar(c_vals, mean_mse, yerr=var_mse, capsize=5, ecolor='red', label='Mean prediction error with varience')
        plt.title(f'{pred_type} predictor - {model_type} - C penalty cross validation')
        plt.xlabel('C penalty')
        plt.ylabel('MSE')
        plt.legend()
        plt.savefig(f'../plots/cross_val/C-PEN_{pred_type}_{model_type}.png')
        plt.show()


    def knn_cross_validation(self, fig, cases, traffic, days, pred_type, model_type, folds, Q, C):
        ##
        knn_range = [2, 5, 10, 20, 35, 50, 75, 100]

        ## init mean, var and std lists
        mean_mse = []
        var_mse = []
        std_mse = []

        ## loop through KNN values
        for KNN in knn_range:
            ## run cross validation
            cross_val_results = self.cross_val_model(cases, traffic, days, pred_type, model_type, folds, Q, KNN, C)

            ## append append mean, var and std error values to lists
            mean_mse.append(cross_val_results[0])
            var_mse.append(cross_val_results[1])
            std_mse.append(cross_val_results[2])

        print(mean_mse)
        print(var_mse) 
        plt.figure(fig)
        knn_vals = ['2', '5', '10', '20', '35', '50', '75', '100']
        plt.errorbar(knn_vals, mean_mse, yerr=var_mse, capsize=5, ecolor='red', label='Mean prediction error with varience')
        plt.title(f'{pred_type} predictor - {model_type} - KNN cross validation')
        plt.xlabel('KNN')
        plt.ylabel('MSE')
        plt.legend()
        plt.savefig(f'../plots/cross_val/KNN_{pred_type}_{model_type}.png')
        plt.show()


    def cross_val_model(self, cases, traffic, days, pred_type, model_type, folds, Q, K, C):
        ##
        ## assign data based on specified predictor type
        if pred_type == 'cases':
            X = traffic
            y = cases
        elif pred_type == 'traffic':
            X = cases
            y = traffic
        else:
            print('ERROR: Incorrect predictor type')

        ## init mean-squared-error array
        mse=[]

        ## generate specifed degree of polynomial features for training
        X_poly = PolynomialFeatures(Q).fit_transform(X)

        ## make k-fold obj
        kf = KFold(n_splits=folds)

        ## loop through each k-fold split
        for train, test in kf.split(X):
            ## select specified model
            if model_type == 'knn':
                model = KNeighborsRegressor(n_neighbors=K).fit(X_poly[train], y[train])
            elif model_type == 'lasso':
                model = Lasso(alpha=1/2*C).fit(X_poly[train], y[train])
            elif model_type == 'ridge':
                model = Ridge(alpha=1/2*C).fit(X_poly[train], y[train])
            else:
                model = LinearRegression().fit(X_poly[train], y[train])

            ## get pridictions using test part of split
            ypred = model.predict(X_poly[test])

            ## get error for predictions and append to error list
            mse.append(mean_squared_error(y[test], ypred))

        ## return mean, varience and standard dev error values
        return [np.mean(mse), np.var(mse), np.std(mse)]


    def plot_predictions(self, fig, cases, traffic, days, pred_type, model_type, folds, Q, K, C):
        ##
        print(f"\n\n-> Plotting {pred_type} predictions")

        ## plot all data
        plt.figure(fig)
        plt.plot(days, cases)

        ## assign data based on specified predictor type
        if pred_type == 'cases':
            X = traffic
            y = cases
        elif pred_type == 'traffic':
            X = cases
            y = traffic
        else:
            print('ERROR: Incorrect predictor type')

        ## generate specifed degree of polynomial features for training
        X_poly = PolynomialFeatures(Q).fit_transform(X)

        ## make k-fold obj
        kf = KFold(n_splits=folds)

        ## loop through each k-fold split
        for train, test in kf.split(X):
            ## select specified model
            if model_type == 'knn':
                model = KNeighborsRegressor(n_neighbors=K).fit(X_poly[train], y[train])
            elif model_type == 'lasso':
                model = Lasso(alpha=1/2*C).fit(X_poly[train], y[train])
            elif model_type == 'ridge':
                model = Ridge(alpha=1/2*C).fit(X_poly[train], y[train])
            else:
                model = LinearRegression().fit(X_poly[train], y[train])
            ##
            predictions = model.predict(X_poly[test])
            
            ## print model evaluation results
            self.evaluate_model(pred_type, model_type, y[test], predictions)

        ## plot the predictions for the all data
        predictions = model.predict(X_poly)
        plt.plot(days, predictions, c="lime")

        plt.title("Model using traffic to predict cases")
        plt.xlabel("Days")
        plt.ylabel("Cases")
        plt.legend(["training cases", "predicted cases"])
        plt.savefig(f'../plots/Predictions_{pred_type}_{model_type}.png')
        plt.show()


    def evaluate_model(self, pred_type, model_type, y, y_pred):
        ##
        if model_type == 'knn':
            y_pred = [round(num[0]) for num in y_pred]
        else:
            y_pred = [round(num) for num in y_pred]

        ## print the model error and accuracy and R2 score
        print(f"\n~~~~~ {pred_type} Predictor ~~~~~")
        print(f"Mean-Squared-Error: {mean_squared_error(y, y_pred)}")
        print(f"Mean-Absolute-Error: {mean_absolute_error(y, y_pred)}")
        print(f"Accuracy: {accuracy_score(y, y_pred)}")
        print(f"R2-score: {r2_score(y, y_pred)}")
