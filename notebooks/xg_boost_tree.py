import numpy as np
import pandas as pd


class XGBoostTree:
    def __init__(self, max_depth: int = 5, min_samples: int = 10):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.tree = None

    def fit(self, X: pd.DataFrame, residuals: np.ndarray) -> None:
        self.tree = self._build_tree(X, residuals, list(X.columns), set())

    def predict(self, X: pd.DataFrame) -> list[float]:
        predictions = list()
        
        for _, row in X.iterrows():
            prediction = self._predict_sample(row, node=self.tree)
            predictions.append(prediction)

        return predictions

    def _calculate_similarity_score(self, residuals: np.ndarray) -> float:
        return np.sum(residuals) ** 2 / len(residuals)

    def _calculate_gain(self, parent_ss: float, left_child_ss: float, right_child_ss: float) -> float:
        return left_child_ss + right_child_ss - parent_ss

    def _best_split(
            self, 
            X: pd.DataFrame, 
            residuals: np.ndarray, 
            features: list[str], 
            used_features: set[str]
            ) -> tuple[str, str | int | float]:
        parent_ss = self._calculate_similarity_score(residuals)
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for feature in features:
            is_numeric = pd.api.types.is_numeric_dtype(X[feature])

            # there is no point at reusing categorical feature since the data was already splitted based on it 
            if not is_numeric and feature in used_features:
                continue

            thresholds = X[feature].unique()
            
            for threshold in thresholds:
                # apply mask based on the feature's type
                mask = X[feature] <= threshold if is_numeric else X[feature] == threshold
                left, right = residuals[mask], residuals[~mask]

                # prevent empty split
                if len(left) == 0 or len(right) == 0:
                    continue

                # gain = left_ss + right_ss - parent_ss
                gain = self._calculate_gain(
                    parent_ss,
                    left_child_ss=self._calculate_similarity_score(left),
                    right_child_ss=self._calculate_similarity_score(right)
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _leaf_value(self, residuals: np.ndarray) -> float:
        return np.mean(residuals)   # returns the avg across residuals 

    def _build_tree(
            self,
            X: pd.DataFrame,
            residuals: np.ndarray,
            features: list[str],
            used_features: set[str],
            depth: int = 0
    ):
        if depth == self.max_depth or len(residuals) < self.min_samples:
            return {"leaf": True, "prediction": self._leaf_value(residuals=residuals)}

        feature, threshold = self._best_split(X, residuals, features, used_features)
        
        if feature is None:
            return {"leaf": True, "prediction": self._leaf_value(residuals=residuals)}

        is_categorical = not pd.api.types.is_numeric_dtype(X[feature])

        if is_categorical:
            mask = X[feature] == threshold
            used_features = used_features.copy()
            used_features.add(feature)
        else:
            mask = X[feature] <= threshold

        left_X, right_X = X[mask], X[~mask]
        left_y, right_y = residuals[mask], residuals[~mask]

        if len(left_X) == 0 or len(right_X) == 0:
            return {"leaf": True, "prediction": self._leaf_value(residuals=residuals)}

        return {
            "leaf": False,
            "feature": feature,
            "threshold": threshold,
            "left": self._build_tree(left_X, left_y, features, used_features, depth + 1),
            "right": self._build_tree(right_X, right_y, features, used_features, depth + 1),
        }

    def _predict_sample(self, sample: pd.Series, node) -> float:
        if node["leaf"]:
            return node["prediction"]
        
        feature, threshold = node["feature"], node["threshold"]
        if isinstance(threshold, str):
            child = node["left"] if sample[feature] == threshold else node["right"]
        else:
            child = node["left"] if sample[feature] <= threshold else node["right"]
        return self._predict_sample(sample, child)