import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

class ModelTrainer:
    def __init__(self, X, y, models_dict, cv):
        self.X = X
        self.y = y
        self.models_dict = models_dict
        self.cv = cv
        self.metrics = {name: {"acc": [], "prec": [], "rec": [], "f1": [], "pred": [], "proba": [], "time": []} for name in models_dict}
        self.all_y_test = []

    def train_and_evaluate(self):
        for train_idx, test_idx in self.cv.split(self.X, self.y):
            X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            self.all_y_test.extend(y_test)
            for name, model in self.models_dict.items():
                start_time = time.time()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
                elapsed_time = time.time() - start_time
                self.metrics[name]["time"].append(elapsed_time)
                self.metrics[name]["acc"].append(accuracy_score(y_test, y_pred) * 100)
                self.metrics[name]["prec"].append(precision_score(y_test, y_pred, average='macro', zero_division=0) * 100)
                self.metrics[name]["rec"].append(recall_score(y_test, y_pred, average='macro', zero_division=0) * 100)
                self.metrics[name]["f1"].append(f1_score(y_test, y_pred, average='macro', zero_division=0) * 100)
                self.metrics[name]["pred"].extend(y_pred)
                self.metrics[name]["proba"].extend(y_pred_proba)

    def get_mean_metrics(self):
        mean_metrics = {}
        for name in self.models_dict:
            mean_metrics[name] = {
                "accuracy": np.mean(self.metrics[name]["acc"]),
                "precision": np.mean(self.metrics[name]["prec"]),
                "recall": np.mean(self.metrics[name]["rec"]),
                "f1": np.mean(self.metrics[name]["f1"]),
                "time": np.mean(self.metrics[name]["time"])
            }
        return mean_metrics

    def get_predictions(self):
        return self.all_y_test, {name: self.metrics[name]["pred"] for name in self.models_dict}

    def get_probabilities(self):
        return self.all_y_test, {name: self.metrics[name]["proba"] for name in self.models_dict}