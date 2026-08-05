import numpy as np
import pandas as pd
from wrapped_models.decision_tree import DecisionTree

class RandomForest:
    def __init__(self, n_trees: int = 5):
        self.n_trees = n_trees
        self.trees = []
        self.oob_indices_list = []

    def _bootstrap_sampling(self, X: pd.DataFrame, y: np.ndarray):
        num_of_indices = len(X)
        random_indices = np.random.choice(a=num_of_indices, size=num_of_indices, replace=True)
        
        bootstrapped_X = X.iloc[random_indices]
        bootstrapped_y = y[random_indices]
        
        oob_indices = [i for i in range(num_of_indices) if i not in random_indices]
        return bootstrapped_X, bootstrapped_y, oob_indices

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.trees = []
        self.oob_indices_list = []

        for _ in range(self.n_trees):
            new_tree = DecisionTree(random_subspace=True)
            training_data, results, oob_indices = self._bootstrap_sampling(X, y)
            new_tree.fit(X=training_data, y=results)
            
            self.trees.append(new_tree)
            self.oob_indices_list.append(oob_indices)

    def predict(self, X: pd.DataFrame) -> list[int]:
        predictions = []
        
        for index in range(len(X)):
            curr_predictions = []
            for decision_tree in self.trees:
                curr_predictions.extend(decision_tree.predict(X.iloc[[index]]))
                
            curr_predictions = np.array(curr_predictions)
            final_prediction = 1 if np.sum(curr_predictions) >= (len(curr_predictions) / 2) else 0
            predictions.append(final_prediction)

        return predictions