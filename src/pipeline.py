from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.features import TitanicPreprocessor


def build_pipeline(model) -> Pipeline:
    """
    Build a sklearn Pipeline with Titanic preprocessing and a given classifier.

    Args:
        model: An instantiated sklearn-compatible classifier, e.g. LogisticRegression().

    Returns:
        Pipeline: A Pipeline with three steps — preprocessor, scaler, classifier.
    """
    # split columns into numerical and categorical
    # numerical ones require scaler
    # categotical ones require encoder to int since model understands only numbers 
    numeric_columns = ["Fare", "Pclass", "SibSp", "Parch", "FamilySize", "Age", "Has_Cabin"]
    categorical_columns = ["Sex", "Embarked", "Title"]

    # scale numeric columns and encode string ones to int 
    scaler_and_encoder = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
        ]
    )

    # feature engineering -> encode and scale -> classify
    pipeline = Pipeline(
        steps=[
            ("preprocessor", TitanicPreprocessor()),
            ("normalizer", scaler_and_encoder),
            ("classifier", model)
        ]
    )

    return pipeline
