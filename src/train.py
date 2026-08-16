import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
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
) -> tuple[Pipeline, float] :
    """
    Evaluates multiple baseline models using 5-fold cross-validation 
    with ROC-AUC scoring, and returns the best-performing pipeline.

    Args:
        models: Dictionary mapping model names to instantiated sklearn classifiers.
        X_train: Training features DataFrame.
        y_train: Training target labels Series.

    Returns:
        tuple[Pipeline, float]: Best-performing sklearn Pipeline and its mean CV AUC score.
    """
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
            
    return best_pipeline, best_auc


def find_best_xgboost_pipeline(xgboost: XGBClassifier, X_train: pd.DataFrame, y_train: pd.Series) -> tuple[Pipeline, float]:
    """
    Performs grid search with 5-fold cross-validation over XGBoost hyperparameters
    using ROC-AUC scoring, and returns the best-fitted pipeline.

    Args:
        xgboost: Instantiated XGBClassifier to be tuned.
        X_train: Training features DataFrame.
        y_train: Training target labels Series.

    Returns:
        tuple[Pipeline, float]: Best-fitted sklearn Pipeline refitted on the entire 
            training set, and its CV AUC score from GridSearchCV.
    """
    # we have 3*3*3 = 27 possible combinations
    # we need a prefix so GridSearchCV knows at which steps apply the parameters
    parameters = {
        "classifier__n_estimators": [100, 300, 500],
        "classifier__max_depth": [3, 4, 5],
        "classifier__learning_rate": [0.01, 0.05, 0.1]
    }

    pipeline = build_pipeline(model=xgboost)

    grid_cv = GridSearchCV(
        estimator=pipeline,
        param_grid=parameters,
        scoring="roc_auc",
        cv=5,       # 5 folds, 4 for training, 1 for validating
        n_jobs=-1   # use all available CPU cores to train models in parallel
    )

    # train all possible 27 models and return the best one
    grid_cv.fit(X=X_train, y=y_train)

    print(f"Best params: {grid_cv.best_params_}")
    print(f"Best AUC: {grid_cv.best_score_:.4f}")

    return grid_cv.best_estimator_, grid_cv.best_score_


if __name__ == "__main__":
    X, y = load_data(path=PATH_TO_CSV_FILE)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    best_pipeline = None
    best_auc = 0.0

    models = get_models()
    best_default, default_auc = train_and_evaluate(models=models, X_train=X_train, y_train=y_train)
    best_xgboost, xgboost_auc = find_best_xgboost_pipeline(xgboost=XGBClassifier(), X_train=X_train, y_train=y_train)

    if xgboost_auc > default_auc:
        best_pipeline = best_xgboost
        best_auc = xgboost_auc
        print("XGBoost classifier won!")
    else:
        best_pipeline = best_default
        best_auc = default_auc
        print(f"{best_pipeline.named_steps['classifier'].__class__.__name__} won!")

    print(f"Best AUC: {best_auc}")
    print(f"Final score: {best_pipeline.score(X=X_val, y=y_val)}")

    # add finetuned xgboost classifier to our models to have it on plot as well
    models["Tuned XGBoost"] = best_xgboost.named_steps["classifier"]

    plot_roc_curves(models=models, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)
    plot_confusion_matrix(best_pipeline=best_pipeline, X_val=X_val, y_val=y_val)

    save_model(model=best_pipeline, filepath=PATH_TO_BEST_MODEL)
