from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.features import TitanicPreprocessor


def build_pipeline(model) -> Pipeline:
    """
    Build a sklearn Pipeline with Titanic preprocessing and a given classifier.

    Args:
        model: An instantiated sklearn-compatible classifier, e.g. LogisticRegression().

    Returns:
        Pipeline: A Pipeline with three steps — preprocessor, scaler, classifier.
    """
    # preprocess data -> scale -> apply model
    pipeline = Pipeline(
        steps=[
            ('preprocessor', TitanicPreprocessor()),
            ('scaler', StandardScaler()),
            ('classifier', model)
        ]
    )

    return pipeline
