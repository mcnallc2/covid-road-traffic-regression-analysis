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


class CrossValidation:
    def optimal_value(self, err, values):
        index_of_best = err.index(min(err))
        if min(err) == err[0]:  #  if they're all the same
            index_of_best = 0
        return values[index_of_best]

    def k_folds_cross_validation(self, fig, cases, traffic, days, pred_type, model_type, Q, K, C):
        ##
        k_folds = [2, 5, 10, 25, 50, 100]

        # init mean, var and std lists
        mean_mse = []
        var_mse = []
        std_mse = []

        # for each k-fold value
        for folds in k_folds:
            # run cross validation
            cross_val_results = self.cross_val_model(
                cases, traffic, days, pred_type, model_type, folds, Q, K, C)
            # append append mean, var and std error values to lists
            mean_mse.append(cross_val_results[0])
            var_mse.append(cross_val_results[1])
            std_mse.append(cross_val_results[2])

        print("-> KFold Cross Val. -> Recommended: Lowest variance @ KFolds =",
              self.optimal_value(var_mse, k_folds))

        # add results to errorbar plot
        plt.figure(fig)
        kf_vals = ['2', '5', '10', '25', '50', '100']
        plt.errorbar(kf_vals, mean_mse, yerr=var_mse, capsize=5,
                     ecolor='red', label='Mean prediction error with varience')
        plt.title(
            f'{pred_type} predictor - {model_type} - K-folds cross validation')
        plt.xlabel('K-folds')
        plt.ylabel('MSE')
        plt.legend()
        plt.show()

    def poly_feature_cross_validation(self, fig, cases, traffic, days, pred_type, model_type, folds, K, C):
        ##
        q_range = [1, 2, 3, 4, 5]

        # init mean, var and std lists
        mean_mse = []
        var_mse = []
        std_mse = []
        ##
        # loop through Q values
        for Q in q_range:
            ##
            cross_val_results = self.cross_val_model(
                cases, traffic, days, pred_type, model_type, folds, Q, K, C)

            # append append mean, var and std error values to lists
            mean_mse.append(cross_val_results[0])
            var_mse.append(cross_val_results[1])
            std_mse.append(cross_val_results[2])

        print("-> Polynomial Features Cross Val. -> Recommended: Lowest variance @ q =",
              self.optimal_value(var_mse, q_range))

        plt.figure(fig)
        q_vals = ['1', '2', '3', '4', '5']
        plt.errorbar(q_vals, mean_mse, yerr=var_mse, capsize=5,
                     ecolor='red', label='Mean prediction error with varience')
        plt.title(
            f'{pred_type} predictor - {model_type} - Polynomial Features cross validation')
        plt.xlabel('Polynomial Features')
        plt.ylabel('MSE')
        plt.legend()
        plt.show()

    def c_penalty_cross_validation(self, fig, cases, traffic, days, pred_type, model_type, folds, Q, K):
        ##
        C_range = [0.01, 0.1, 1, 10, 1000]

        # init mean, var and std lists
        mean_mse = []
        var_mse = []
        std_mse = []

        # loop through C values
        for C in C_range:
            # run cross validation
            cross_val_results = self.cross_val_model(
                cases, traffic, days, pred_type, model_type, folds, Q, K, C)

            # append append mean, var and std error values to lists
            mean_mse.append(cross_val_results[0])
            var_mse.append(cross_val_results[1])
            std_mse.append(cross_val_results[2])

        print("-> C Penalty Cross Val. -> Recommending: Lowest variance @ C =",
              self.optimal_value(var_mse, C_range))

        plt.figure(fig)
        c_vals = ['0.01', '0.1', '1', '10', '1000']
        plt.errorbar(c_vals, mean_mse, yerr=var_mse, capsize=5,
                     ecolor='red', label='Mean prediction error with varience')
        plt.title(
            f'{pred_type} predictor - {model_type} - C penalty cross validation')
        plt.xlabel('C penalty')
        plt.ylabel('MSE')
        plt.legend()
        plt.show()

    def knn_cross_validation(self, fig, cases, traffic, days, pred_type, model_type, folds, Q, C):
        ##
        knn_range = [2, 5, 10, 20, 35, 50, 75, 100]

        # init mean, var and std lists
        mean_mse = []
        var_mse = []
        std_mse = []

        # loop through KNN values
        for KNN in knn_range:
            # run cross validation
            cross_val_results = self.cross_val_model(
                cases, traffic, days, pred_type, model_type, folds, Q, KNN, C)

            # append append mean, var and std error values to lists
            mean_mse.append(cross_val_results[0])
            var_mse.append(cross_val_results[1])
            std_mse.append(cross_val_results[2])

        print("-> KNN Cross Val. -> Recommending: Lowest variance @ knn =",
              self.optimal_value(var_mse, knn_range))

        plt.figure(fig)
        knn_vals = ['2', '5', '10', '20', '35', '50', '75', '100']

        plt.errorbar(knn_vals, mean_mse, yerr=var_mse, capsize=5,
                     ecolor='red', label='Mean prediction error with varience')
        plt.title(
            f'{pred_type} predictor - {model_type} - KNN cross validation')
        plt.xlabel('KNN')
        plt.ylabel('MSE')
        plt.legend()
        plt.show()

    def cross_val_model(self, cases, traffic, days, pred_type, model_type, folds, Q, K, C):
        ##
        # assign data based on specified predictor type
        if pred_type == 'cases':
            X = traffic
            y = cases
        elif pred_type == 'traffic':
            X = cases
            y = traffic
        else:
            print('ERROR: Incorrect predictor type')

        # init mean-squared-error array
        mse = []

        # generate specifed degree of polynomial features for training
        X_poly = PolynomialFeatures(Q).fit_transform(X)

        # make k-fold obj
        kf = KFold(n_splits=folds)

        # loop through each k-fold split
        for train, test in kf.split(X):
            # select specified model
            if model_type == 'knn':
                model = KNeighborsRegressor(
                    n_neighbors=K).fit(X_poly[train], y[train])
            elif model_type == 'lasso':
                model = Lasso(
                    alpha=1/2*C, normalize=True).fit(X_poly[train], y[train])
            elif model_type == 'ridge':
                model = Ridge(
                    alpha=1/2*C, normalize=True).fit(X_poly[train], y[train])
            else:
                model = LinearRegression().fit(X_poly[train], y[train])

            # get pridictions using test part of split
            ypred = model.predict(X_poly[test])

            # get error for predictions and append to error list
            mse.append(mean_squared_error(y[test], ypred))

        # return mean, varience and standard dev error values
        return [np.mean(mse), np.var(mse), np.std(mse)]
