import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, cohen_kappa_score


def classification_metrics(targets, predictions, labels=range(5)):
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, labels=list(labels), average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "quadratic_weighted_kappa": float(cohen_kappa_score(targets, predictions, labels=list(labels), weights="quadratic")),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=list(labels)).tolist(),
    }
