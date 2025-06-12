import numpy as np

class FeatureSelector:
    def __init__(self, X, y_encoded):
        self.X = X
        self.y_encoded = y_encoded
        self.selected_features = None

    def gini_impurity(self, labels):
        if len(labels) == 0:
            return 0
        _, counts = np.unique(labels, return_counts=True)
        p = counts / len(labels)
        return 1 - np.sum(p ** 2)

    def compute_gini_scores(self, num_thresholds=100):
        parent_impurity = self.gini_impurity(self.y_encoded)
        gini_scores = {}
        for feature in self.X.columns:
            X_feature = self.X[feature].values
            unique_vals = np.unique(X_feature)
            if len(unique_vals) == 1:
                gini_scores[feature] = 0
                continue
            percentiles = np.linspace(0, 100, num_thresholds)
            thresholds = np.percentile(X_feature, percentiles)
            thresholds = np.unique(thresholds)
            best_reduction = 0
            for thresh in thresholds:
                left_mask = X_feature <= thresh
                right_mask = X_feature > thresh
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                gini_left = self.gini_impurity(self.y_encoded[left_mask])
                gini_right = self.gini_impurity(self.y_encoded[right_mask])
                w_left = np.sum(left_mask) / len(X_feature)
                w_right = np.sum(right_mask) / len(X_feature)
                weighted_gini = w_left * gini_left + w_right * gini_right
                reduction = parent_impurity - weighted_gini
                if reduction > best_reduction:
                    best_reduction = reduction
            gini_scores[feature] = best_reduction
        return gini_scores

    def select_features(self, threshold=0.01):
        gini_scores = self.compute_gini_scores()
        self.selected_features = [feature for feature, score in gini_scores.items() if score > threshold]
        return self.selected_features