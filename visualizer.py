import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np

class Visualizer:
    def plot_roc_curves(self, y_proba_dict, y_true, class_names):
        plt.figure(figsize=(8, 6))
        for name, y_proba in y_proba_dict.items():
            n_classes = len(class_names)
            fpr, tpr, roc_auc = {}, {}, {}
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true, [p[i] for p in y_proba], pos_label=i)
                roc_auc[i] = auc(fpr[i], tpr[i])
            mean_auc = np.mean(list(roc_auc.values()))
            plt.plot(fpr[0], tpr[0], label=f"{name} (Mean AUC={mean_auc:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title("ROC Curve (One-vs-Rest)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_accuracy_per_fold(self, metrics):
        plt.figure(figsize=(10, 6))
        for name in metrics:
            plt.plot(range(1, 11), metrics[name]['acc'], marker='o', label=name)
        plt.xlabel("Fold")
        plt.ylabel("Accuracy (%)")
        plt.title("10-Fold Cross Validation Accuracy")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_attack_wise_accuracy(self, y_true, y_pred_dict, class_names):
        plt.figure(figsize=(14, 6))
        x = np.arange(len(class_names))
        width = 0.2
        colors = ['gold', 'green', 'orange', 'purple']
        for i, (name, y_pred) in enumerate(y_pred_dict.items()):
            cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
            acc = (cm.diagonal() / cm.sum(axis=1)) * 100
            bars = plt.bar(x + (i - 1.5) * width, acc, width=width, label=name, color=colors[i])
            for bar in bars:
                height = bar.get_height()
                plt.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height), 
                             xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        plt.xticks(x, class_names, rotation=45)
        plt.xlabel("Attack Type")
        plt.ylabel("Accuracy (%)")
        plt.title("Attack-wise Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_classifier_errors(self, mean_metrics):
        classifier_errors = {name: 100 - metrics['accuracy'] for name, metrics in mean_metrics.items()}
        plt.figure(figsize=(8, 5))
        bars = plt.bar(classifier_errors.keys(), classifier_errors.values(), color='crimson')
        plt.ylabel("Error Rate (%)")
        plt.title("Classifier Error for Each Model")
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height), 
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    def plot_prediction_distribution(self, predicted_labels, class_names):
        plt.figure(figsize=(10, 6))
        counts = np.unique(predicted_labels, return_counts=True)
        plt.bar(counts[0], counts[1], color='skyblue')
        plt.xlabel("Attack Type")
        plt.ylabel("Number of Predictions")
        plt.title("Distribution of Predicted Attack Types")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()