from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score


class EvaluateModels:
    def evaluate_model(self, pred_type, model_type, y, y_pred):
        ##
        if model_type == 'knn':
            y_pred = [round(num[0]) for num in y_pred]
        else:
            y_pred = [round(num) for num in y_pred]

        # print the model error and accuracy and R2 score
        print(f"\n~~~~~ {pred_type} Predictor ~~~~~")
        print(f"Mean-Squared-Error: {mean_squared_error(y, y_pred)}")
        print(f"Mean-Absolute-Error: {mean_absolute_error(y, y_pred)}")
        print(f"Accuracy: {accuracy_score(y, y_pred)}")
        print(f"R2-score: {r2_score(y, y_pred)}")
