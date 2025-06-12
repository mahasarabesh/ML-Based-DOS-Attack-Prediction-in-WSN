from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

class Predictor:
    def __init__(self, selected_features, class_names):
        self.selected_features = selected_features
        self.class_names = class_names
        self.model = make_pipeline(StandardScaler(), XGBClassifier(n_estimators=10, eval_metric='mlogloss', n_jobs=-1))

    def train(self, X_train, y_train):
        X_train_selected = X_train[self.selected_features]
        self.model.fit(X_train_selected, y_train)

    def predict(self, X_new):
        X_new_selected = X_new[self.selected_features]
        y_pred = self.model.predict(X_new_selected)
        y_pred_labels = [self.class_names[pred] for pred in y_pred]
        return y_pred_labels