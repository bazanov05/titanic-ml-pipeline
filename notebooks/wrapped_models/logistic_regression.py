import numpy as np
import pandas as pd

class LogisticRegression:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def _binary_cross_entropy_loss(self, p: np.ndarray, y: np.ndarray) -> float:
        return -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        X_array = X.to_numpy()
        n, f = X_array.shape
        self.weights = np.zeros(f, dtype=X_array.dtype)
        self.bias = 0.0

        for _ in range(self.epochs):
            z = X_array @ self.weights + self.bias
            p = self._sigmoid(z)
            
            dw = (1 / n) * (X_array.T @ (p - y))
            db = np.mean(p - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        z = X.to_numpy() @ self.weights + self.bias
        return self._sigmoid(z)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)