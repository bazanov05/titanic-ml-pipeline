import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.pipeline import build_pipeline
from sklearn.model_selection import train_test_split


PATH_TO_CSV_FILE = "data/raw/Titanic-Dataset.csv"
PATH_TO_BEST_MODEL = "src/models/pipeline.joblib"


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    try:
        df = pd.read_csv(filepath_or_buffer=path)
        y = df["Survived"]
        X = df.drop(columns=["Survived"])

        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def save_model(model: Pipeline, filepath: str | Path) -> None:
    """Saves a fitted scikit-learn model/pipeline to disk using joblib."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)  # create a parent dir if it did not exist
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")


def get_models() -> dict:
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier()
    }

    return models


def train_and_evaluate(
        models: dict,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series
) -> Pipeline:
    best_score = 0.0
    best_pipeline = None

    for name, model in models.items():
        pipeline = build_pipeline(model=model)  # build pipeline for this exact model
        # train data only on train data to prevent data leakage
        # fit -> transform -> scale -> train: all in this fit()
        pipeline.fit(X=X_train, y=y_train)

        # get score for train and val data
        train_score = pipeline.score(X=X_train, y=y_train)
        val_score = pipeline.score(X=X_val, y=y_val)

        if val_score > best_score:
            best_score = val_score
            best_pipeline = pipeline

        print(f"Model: {name}\ttrain score: {train_score}\tvalidation score:{val_score}")

    return best_pipeline


if __name__ == "__main__":
    X, y = load_data(path=PATH_TO_CSV_FILE)

    # 80% of data goes to train, 20% of data goes to validate
    X_train, X_val, y_train, y_val = train_test_split(X, y ,test_size=0.2, random_state=42)

    models = get_models()

    best_pipeline = train_and_evaluate(
        models=models,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val
    )

    save_model(model=best_pipeline, filepath=PATH_TO_BEST_MODEL)
