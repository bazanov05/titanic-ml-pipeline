import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier
from src.pipeline import build_pipeline
from src.evaluate import plot_confusion_matrix, plot_roc_curves


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
        y_train: pd.Series,
) -> Pipeline:
    best_auc = 0.0
    best_pipeline = None

    for name, model in models.items():
        pipeline = build_pipeline(model=model)  # build pipeline for this exact model
        
        # use cross validation to get more accurate results
        # divide data into 5 folds and have 5 iterations
        # so every fold can be a validating set
        acc_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
        auc_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="roc_auc")
        print(f"{name} | acc: {acc_scores.mean():.4f} ± {acc_scores.std():.4f} | auc: {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")

        if auc_scores.mean() > best_auc:
            best_auc = auc_scores.mean()
            best_pipeline = pipeline
            
    return best_pipeline


if __name__ == "__main__":
    X, y = load_data(path=PATH_TO_CSV_FILE)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    models = get_models()
    best_pipeline = train_and_evaluate(models=models, X_train=X_train, y_train=y_train)

    # refit best model on full training data, then evaluate once on unseen test set
    best_pipeline.fit(X_train, y_train)
    final_score = best_pipeline.score(X_val, y_val)
    print(f"Final score: {final_score:.4f}")

    plot_roc_curves(models=models, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)
    plot_confusion_matrix(best_pipeline=best_pipeline, X_val=X_val, y_val=y_val)

    save_model(model=best_pipeline, filepath=PATH_TO_BEST_MODEL)
