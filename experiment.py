from data_loader import DataLoader
from feature_selector import FeatureSelector
from model_trainer import ModelTrainer
from visualizer import Visualizer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

models_dict = {
    "Decision Tree": make_pipeline(StandardScaler(), DecisionTreeClassifier()),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Random Forest": make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=10, random_state=42)),
    "XGBoost": make_pipeline(StandardScaler(), XGBClassifier(n_estimators=10, eval_metric='mlogloss', n_jobs=-1))
}

data_loader = DataLoader(r"C:\Users\Manikandan Iyyappan\Downloads\MINI_PROJECT\WSN-DS.csv")
data_loader.load_data()
X, y_encoded, class_names = data_loader.get_data()

feature_selector = FeatureSelector(X, y_encoded)
selected_features = feature_selector.select_features(threshold=0.01)
X_filtered = X[selected_features]

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

model_trainer = ModelTrainer(X_filtered, y_encoded, models_dict, skf)
model_trainer.train_and_evaluate()

mean_metrics = model_trainer.get_mean_metrics()
print("\nFinal Mean Performance Metrics:")
for name, metrics in mean_metrics.items():
    print(f"{name:<15} Accuracy={metrics['accuracy']:.1f}% | "
          f"Precision={metrics['precision']:.1f}% | "
          f"Recall={metrics['recall']:.1f}% | "
          f"F1 Score={metrics['f1']:.1f}% | "
          f"Time={metrics['time']:.2f}s")

all_y_test, y_pred_dict = model_trainer.get_predictions()
all_y_test, y_proba_dict = model_trainer.get_probabilities()
visualizer = Visualizer()
visualizer.plot_roc_curves(y_proba_dict, all_y_test, class_names)
visualizer.plot_accuracy_per_fold(model_trainer.metrics)
visualizer.plot_attack_wise_accuracy(all_y_test, y_pred_dict, class_names)
visualizer.plot_classifier_errors(mean_metrics)