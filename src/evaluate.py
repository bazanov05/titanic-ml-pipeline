import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

from src.pipeline import build_pipeline


def plot_roc_curves(
        models: dict, 
        X_train: pd.DataFrame, 
        X_val: pd.DataFrame, 
        y_train: pd.Series, 
        y_val: pd.Series
    ) -> None:
    """
    Fits each model on training data, computes ROC curve on validation data,
    plots all curves on one graph and saves to outputs/roc_curve.png.

    Args:
        models: dict of model name to instantiated sklearn classifier
        X_train: training features
        X_val: validation features
        y_train: training labels
        y_val: validation labels

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        pipeline = build_pipeline(model=model)
        pipeline.fit(X=X_train, y=y_train)

        # gives the list of probabilites for every passenger to survive
        # predict_proba() returns predictions for both classes
        # we need only TRUE ones
        y_prob = pipeline.predict_proba(X=X_val)[:, 1]

        # list of tuples - [FPR, TPR, threshold] for every threshold between 0 and 1
        # fpr - false positive rate, how many of NO cases were flagged as YES
        # tpr - true positive rate(recall), how many of all YES cases did we catch
        fpr, tpr, _ = roc_curve(y_true=y_val, y_score=y_prob)
        
        # area under the curve, show the trade off between tpr and fpr 
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("outputs/roc_curve.png")
    plt.close()


def plot_confusion_matrix(best_pipeline: Pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> None:
    """
    Predicts on validation data using the fitted pipeline,
    plots confusion matrix heatmap saved to outputs/confusion_matrix.png,
    and prints precision, recall, F1 classification report.

    Args:
        best_pipeline: fitted sklearn Pipeline
        X_val: validation features
        y_val: validation labels

    Returns:
        None
    """
    plt.figure()
    y_pred = best_pipeline.predict(X=X_val)

    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=["Died", "Survived"], yticklabels=["Died", "Survived"])
    plt.title("Confusion Matrix")
    plt.savefig("outputs/confusion_matrix.png")

    print(classification_report(y_val, y_pred))
