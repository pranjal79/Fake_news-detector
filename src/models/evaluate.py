import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

def evaluate_model(model, X_test, y_test) -> dict:
    logger.info("Evaluating model...")

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy"  : round(accuracy_score(y_test, y_pred), 4),
        "precision" : round(precision_score(y_test, y_pred), 4),
        "recall"    : round(recall_score(y_test, y_pred), 4),
        "f1_score"  : round(f1_score(y_test, y_pred), 4),
    }

    logger.info(f"Accuracy  : {metrics['accuracy']}")
    logger.info(f"Precision : {metrics['precision']}")
    logger.info(f"Recall    : {metrics['recall']}")
    logger.info(f"F1 Score  : {metrics['f1_score']}")

    # Full report
    report = classification_report(y_test, y_pred,
                                   target_names=["Fake", "Real"])
    logger.info(f"\nClassification Report:\n{report}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\nConfusion Matrix:\n{cm}")

    return metrics, y_pred