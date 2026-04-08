import os
import pickle
import yaml
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from src.features.tfidf import load_artifacts
from src.models.evaluate import evaluate_model
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ── Define all 3 models ──────────────────────────────────────────────
def get_models() -> dict:
    return {
        "LogisticRegression": LogisticRegression(
            max_iter    = 1000,
            C           = 1.0,
            solver      = "lbfgs",
            random_state= 42
        ),
        "NaiveBayes": MultinomialNB(
            alpha = 0.1
        ),
        "SVM": CalibratedClassifierCV(
            LinearSVC(
                C          = 1.0,
                max_iter   = 2000,
                random_state = 42
            )
        )
    }

# ── Save model to disk ───────────────────────────────────────────────
def save_model(model, model_name: str) -> str:
    os.makedirs("models", exist_ok=True)
    path = f"models/{model_name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved → {path}")
    return path

# ── Train single model with MLflow tracking ──────────────────────────
def train_single_model(
    model_name: str,
    model,
    X_train, X_test,
    y_train, y_test,
    params: dict
) -> dict:

    logger.info(f"\n{'='*50}")
    logger.info(f"Training: {model_name}")
    logger.info(f"{'='*50}")

    with mlflow.start_run(run_name=model_name):

        # ── Log model parameters ────────────────────────────
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("dataset", "fake-news-kaggle")

        if hasattr(model, "C"):
            mlflow.log_param("C", model.C)
        if hasattr(model, "alpha"):
            mlflow.log_param("alpha", model.alpha)
        if hasattr(model, "max_iter"):
            mlflow.log_param("max_iter", model.max_iter)

        mlflow.log_param("tfidf_max_features",
                         params["tfidf"]["max_features"])
        mlflow.log_param("tfidf_ngram_range",
                         str(params["tfidf"]["ngram_range"]))
        mlflow.log_param("test_size",
                         params["model"]["test_size"])

        # ── Train ───────────────────────────────────────────
        logger.info("Fitting model...")
        model.fit(X_train, y_train)

        # ── Evaluate ────────────────────────────────────────
        metrics, y_pred = evaluate_model(model, X_test, y_test)

        # ── Log metrics to MLflow ────────────────────────────
        mlflow.log_metric("accuracy",  metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall",    metrics["recall"])
        mlflow.log_metric("f1_score",  metrics["f1_score"])

        # ── Save and log model artifact ──────────────────────
        model_path = save_model(model, model_name)
        mlflow.sklearn.log_model(model, model_name)
        mlflow.log_artifact(model_path)

        logger.info(f"MLflow run logged for {model_name}")

    return metrics

# ── Train all models ─────────────────────────────────────────────────
def train_all_models():
    params = load_params()

    # Setup MLflow
    mlflow.set_experiment("fake-news-detection")

    # Load features
    logger.info("Loading TF-IDF artifacts...")
    vectorizer, X_train, X_test, y_train, y_test = load_artifacts()

    models  = get_models()
    results = {}

    for model_name, model in models.items():
        metrics = train_single_model(
            model_name,
            model,
            X_train, X_test,
            y_train, y_test,
            params
        )
        results[model_name] = metrics

    # ── Summary Table ────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    logger.info(f"{'Model':<25} {'Acc':>6} {'Prec':>6} "
                f"{'Rec':>6} {'F1':>6}")
    logger.info("-"*60)

    best_model = None
    best_f1    = 0.0

    for name, m in results.items():
        logger.info(f"{name:<25} {m['accuracy']:>6} "
                    f"{m['precision']:>6} {m['recall']:>6} "
                    f"{m['f1_score']:>6}")
        if m["f1_score"] > best_f1:
            best_f1    = m["f1_score"]
            best_model = name

    logger.info("="*60)
    logger.info(f"🏆 Best Model: {best_model} (F1={best_f1})")

    return results, best_model

if __name__ == "__main__":
    results, best = train_all_models()