import numpy as np
import pandas as pd
from wrapped_models.xg_boost_tree import XGBoostTree

class XGBoostClassifier:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, reg_lambda=1.0, gamma=0.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.trees = []
        self.z0 = 0.0

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def _compute_gradients(self, p: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.subtract(p, y)

    def _compute_hessians(self, p: np.ndarray) -> np.ndarray:
        return np.multiply(p, 1 - p)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        log_odds = np.full(shape=y.shape, fill_value=self.z0)

        for _ in range(self.n_estimators):
            p = self._sigmoid(z=log_odds)
            gradients = self._compute_gradients(p, y)
            hessians = self._compute_hessians(p)

            new_tree = XGBoostTree(self.max_depth, self.min_samples_split, self.reg_lambda, self.gamma)
            new_tree.fit(X, gradients, hessians)
            self.trees.append(new_tree)

            new_odds = np.array(new_tree.predict(X))
            log_odds = log_odds + self.learning_rate * new_odds

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        log_odds = np.full(shape=X.shape[0], fill_value=self.z0)
        
        for tree in self.trees:
            log_odds += self.learning_rate * np.array(tree.predict(X))
        
        return self._sigmoid(z=log_odds)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)