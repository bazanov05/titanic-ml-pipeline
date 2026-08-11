from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class TitanicPreprocessor(BaseEstimator, TransformerMixin):
    """
    Custom sklearn-compatible transformer for Titanic dataset preprocessing.
    
    Learns statistics from training data in fit() and applies all feature
    engineering transformations in transform(). Designed to be used as a
    step inside a sklearn Pipeline.
    
    Args:
        _mode_embarked (str): Most frequent embarkation port learned from training data.
        _avg_age_per_class_and_title (pd.Series): Mean age per Title+Pclass combination, primary Age imputation source.
        _avg_age_per_class (pd.Series): Mean age per Pclass, first fallback for Age imputation.
        _avg_age_per_title (pd.Series): Mean age per Title, second fallback for Age imputation.
        _overall_avg_age (float): Global mean age, final fallback for Age imputation.
    """
    def __init__(self):
        super().__init__()
        self._mode_embarked: str = None
        self._avg_age_per_class_and_title: pd.Series = None

        # fallback if in the testing data appears new combination of class and title
        # class will be used first during the fallback since it has higher impact on survival rate
        self._avg_age_per_class: pd.Series = None
        self._avg_age_per_title: pd.Series = None 
        self._overall_avg_age: float = None     # in case we meet new class and new title in test data   
        
    def fit(self, X: pd.DataFrame, y: np.ndarray = None):
        """
        Learn statistics from training data needed for transformation.
        Nothing is applied to the data here — only learned and remembered on self.

        Args:
            X (pd.DataFrame): Training feature matrix.
            y (np.ndarray, optional): Target values, not used but accepted for Pipeline compatibility.

        Returns:
            self: Returns the instance itself for method chaining.
        """
        self._mode_embarked = X["Embarked"].mode()[0]

        # extract titles from Names: title sits between coma and period
        titles = X["Name"].str.extract(r",\s*([A-Za-z]+)\.", expand=False)

        # calculate the avg age for every existing combination of title and class 
        self._avg_age_per_class_and_title = (
            X.assign(Title=titles)
            .groupby(["Title", "Pclass"])["Age"]
            .mean()
        )

        self._avg_age_per_class = X.groupby("Pclass")["Age"].mean()
        self._avg_age_per_title = (
            X.assign(Title=titles)
            .groupby("Title")["Age"]
            .mean()
        )

        self._overall_avg_age = X["Age"].mean()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering transformations to the data.
        Uses statistics learned during fit() — never recomputes anything from X.

        Transformations applied in order:
            - Encode Sex as binary (male=1, female=0)
            - Create Has_Cabin binary feature
            - Extract Title from Name
            - Fill missing Embarked with mode learned in fit
            - Impute missing Age hierarchically:
                1. Mean age for same Title + Pclass combination
                2. Mean age for same Pclass
                3. Mean age for same Title
                4. Overall mean age
            - Create FamilySize (SibSp + Parch + 1)
            - Create IsSingle binary feature from FamilySize
            - Encode Embarked (C=0, Q=1, S=2)
            - Drop unused columns: Name, Ticket, PassengerId, Cabin

        Args:
            X (pd.DataFrame): Feature matrix to transform, train or test.

        Returns:
            pd.DataFrame: Transformed DataFrame ready for model input.
        """
        X = X.copy()

        # create a binary col Has_Cabin since this fact increases a chances of survival
        X["Has_Cabin"] = X["Cabin"].notna().astype(int)

        X['Title'] = X["Name"].str.extract(r",\s*([A-Za-z]+)\.", expand=False)
        X["Embarked"] = X["Embarked"].fillna(self._mode_embarked)

        # map indices created with group by with the calculated averages 
        # wrap the index.map result in a pd.Series so fillna() accepts it
        age_class_title = pd.Series(
            X.set_index(["Title", "Pclass"]).index.map(self._avg_age_per_class_and_title), 
            index=X.index
        )        
        age_class = X["Pclass"].map(self._avg_age_per_class)
        age_title = X["Title"].map(self._avg_age_per_title)

        # hierarchical imputation of missing values
        X["Age"] = (
            X["Age"]
            .fillna(age_class_title)
            .fillna(age_class)
            .fillna(age_title)
            .fillna(self._overall_avg_age)
        )

        X["FamilySize"] = X["Parch"] + X["SibSp"] + 1
        X["IsSingle"] = (X["FamilySize"] == 1).astype(int)

        X = X.drop(columns=["Name", "Ticket", "PassengerId", "Cabin"])
        
        return X
