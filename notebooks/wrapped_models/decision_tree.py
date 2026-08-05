import numpy as np
import pandas as pd
from math import sqrt


class DecisionTree:
    def __init__(self, max_depth: int = 5, min_samples: int = 10, random_subspace: bool = True):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.tree = None
        self.is_part_of_random_forest = random_subspace

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.tree = self._build_tree(X, y, list(X.columns), set(), depth=0)

    def predict(self, X: pd.DataFrame) -> list[int]:
        predictions = list()

        for _, row in X.iterrows():
            prediction = self._predict_sample(row, node=self.tree)
            predictions.append(prediction)

        return predictions

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        true = np.sum(y == 1) / len(y)
        false = np.sum(y == 0) / len(y)
        return 1.0 - true ** 2 - false ** 2

    def _weighted_gini(self, y_left, y_right):
        n = len(y_left) + len(y_right)
        return (len(y_left) / n) * self._gini(y_left) + (len(y_right) / n) * self._gini(y_right)

    def _sample_random_features(self, features: list[str]) -> list[str]:
        return np.random.choice(a=features, size=int(sqrt(len(features))), replace=False)

    def _best_split(self, X, y, feature_names, used_features):
        best_impurity = self._gini(y)
        best_feature = None
        best_threshold = None

        features = self._sample_random_features(feature_names) if self.is_part_of_random_forest else feature_names

        for feature in features:
            if not pd.api.types.is_numeric_dtype(X[feature]):
                if feature in used_features:
                    continue
                for threshold in X[feature].unique():
                    mask = X[feature] == threshold
                    true, false = y[mask], y[~mask]
                    if len(true) == 0 or len(false) == 0:
                        continue
                    curr_gini = self._weighted_gini(true, false)
                    if curr_gini < best_impurity:
                        best_impurity, best_feature, best_threshold = curr_gini, feature, threshold
            else:
                sorted_thresholds = np.sort(X[feature].unique())
                midpoints = sorted_thresholds[:-1] + np.diff(sorted_thresholds) / 2
                for threshold in midpoints:
                    mask = X[feature] <= threshold
                    true, false = y[mask], y[~mask]
                    if len(true) == 0 or len(false) == 0:
                        continue
                    curr_gini = self._weighted_gini(true, false)
                    if curr_gini < best_impurity:
                        best_impurity, best_feature, best_threshold = curr_gini, feature, threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, feature_names, used_features, depth):
        if depth == self.max_depth or len(y) < self.min_samples:
            return {"leaf": True, "prediction": 1 if np.sum(y == 1) >= len(y) / 2 else 0}

        feature, threshold = self._best_split(X, y, feature_names, used_features)

        if feature is None:
            return {"leaf": True, "prediction": 1 if np.sum(y == 1) >= len(y) / 2 else 0}

        is_categorical = not pd.api.types.is_numeric_dtype(X[feature])
        if is_categorical:
            mask = X[feature] == threshold
            used_features = used_features.copy()
            used_features.add(feature)
        else:
            mask = X[feature] <= threshold

        left_X, right_X = X[mask], X[~mask]
        left_y, right_y = y[mask], y[~mask]

        if len(left_X) == 0 or len(right_X) == 0:
            return {"leaf": True, "prediction": 1 if np.sum(y == 1) >= len(y) / 2 else 0}

        return {
            "leaf": False,
            "feature": feature,
            "threshold": threshold,
            "left": self._build_tree(left_X, left_y, feature_names, used_features, depth + 1),
            "right": self._build_tree(right_X, right_y, feature_names, used_features, depth + 1),
        }

    def _predict_sample(self, sample, node):
        if node["leaf"]:
            return node["prediction"]
        feature, threshold = node["feature"], node["threshold"]
        if isinstance(threshold, str):
            child = node["left"] if sample[feature] == threshold else node["right"]
        else:
            child = node["left"] if sample[feature] <= threshold else node["right"]
        return self._predict_sample(sample, child)