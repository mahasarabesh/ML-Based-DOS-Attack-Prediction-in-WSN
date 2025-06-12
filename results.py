import sys
from data_loader import DataLoader
from feature_selector import FeatureSelector
from predictor import Predictor
from visualizer import Visualizer

def main(train_csv_path,new_csv_path):
    data_loader_train = DataLoader(train_csv_path)
    data_loader_train.load_data()
    X_train, y_train_encoded, class_names = data_loader_train.get_data()

    feature_selector = FeatureSelector(X_train, y_train_encoded)
    selected_features = feature_selector.select_features(threshold=0.01)

    predictor = Predictor(selected_features, class_names)
    predictor.train(X_train, y_train_encoded)

    data_loader_new = DataLoader(new_csv_path, has_target=False)
    data_loader_new.load_data()
    X_new, _, _ = data_loader_new.get_data()

    predicted_labels = predictor.predict(X_new)

    print("\nPredicted Attack Types Distribution:")
    from collections import Counter
    distribution = Counter(predicted_labels)
    for label, count in distribution.items():
        print(f"{label}: {count}")

    print("\nSample Predictions:")
    for i in range(min(5, len(X_new))):
        print(f"Row {i}: Predicted: {predicted_labels[i]}")

    visualizer = Visualizer()
    visualizer.plot_prediction_distribution(predicted_labels, class_names)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python results.py <data_set_path> <new_csv_path>")
        sys.exit(1)
    main(sys.argv[1],sys.argv[2])