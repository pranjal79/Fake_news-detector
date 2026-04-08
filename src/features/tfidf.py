import pandas as pd
import numpy as np
import yaml
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import save_npz, load_npz
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ── Build and fit TF-IDF vectorizer ─────────────────────────────────
def build_tfidf_vectorizer(params: dict) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(
        max_features = params["tfidf"]["max_features"],
        ngram_range  = tuple(params["tfidf"]["ngram_range"]),
        sublinear_tf = True,   # Apply log normalization to TF
        min_df       = 2,      # Ignore terms appearing in < 2 docs
        max_df       = 0.95    # Ignore terms in > 95% of docs
    )
    logger.info(f"TF-IDF config → max_features: {params['tfidf']['max_features']}, "
                f"ngram_range: {params['tfidf']['ngram_range']}")
    return vectorizer

# ── Split data into train/test ───────────────────────────────────────
def split_data(df: pd.DataFrame, params: dict):
    logger.info("Splitting data into train/test sets...")

    X = df["processed_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = params["model"]["test_size"],
        random_state = params["model"]["random_state"],
        stratify     = y   # Keep label distribution balanced
    )

    logger.info(f"Train size : {len(X_train)}")
    logger.info(f"Test  size : {len(X_test)}")
    logger.info(f"Train label dist:\n{y_train.value_counts().to_dict()}")
    logger.info(f"Test  label dist:\n{y_test.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test

# ── Apply TF-IDF transformation ──────────────────────────────────────
def apply_tfidf(X_train, X_test, vectorizer: TfidfVectorizer):
    logger.info("Fitting TF-IDF on training data...")
    X_train_tfidf = vectorizer.fit_transform(X_train)

    logger.info("Transforming test data with fitted TF-IDF...")
    X_test_tfidf  = vectorizer.transform(X_test)

    logger.info(f"TF-IDF matrix shape → Train: {X_train_tfidf.shape}, "
                f"Test: {X_test_tfidf.shape}")
    return X_train_tfidf, X_test_tfidf

# ── Save artifacts ───────────────────────────────────────────────────
def save_artifacts(
    vectorizer, 
    X_train_tfidf, X_test_tfidf,
    y_train, y_test
) -> None:

    os.makedirs("data/features", exist_ok=True)

    # Save vectorizer
    with open("data/features/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info("Saved → data/features/tfidf_vectorizer.pkl")

    # Save sparse matrices
    save_npz("data/features/X_train.npz", X_train_tfidf)
    save_npz("data/features/X_test.npz",  X_test_tfidf)
    logger.info("Saved → data/features/X_train.npz, X_test.npz")

    # Save labels
    np.save("data/features/y_train.npy", y_train.values)
    np.save("data/features/y_test.npy",  y_test.values)
    logger.info("Saved → data/features/y_train.npy, y_test.npy")

# ── Load artifacts (for model training step) ─────────────────────────
def load_artifacts():
    logger.info("Loading saved TF-IDF artifacts...")

    with open("data/features/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    X_train = load_npz("data/features/X_train.npz")
    X_test  = load_npz("data/features/X_test.npz")
    y_train = np.load("data/features/y_train.npy")
    y_test  = np.load("data/features/y_test.npy")

    logger.info("All artifacts loaded successfully")
    return vectorizer, X_train, X_test, y_train, y_test

# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.etl.load import fetch_processed_data

    # Fetch from MongoDB
    df = fetch_processed_data()

    params = load_params()

    # Split
    X_train, X_test, y_train, y_test = split_data(df, params)

    # Build + Apply TF-IDF
    vectorizer = build_tfidf_vectorizer(params)
    X_train_tfidf, X_test_tfidf = apply_tfidf(X_train, X_test, vectorizer)

    # Save everything
    save_artifacts(vectorizer, X_train_tfidf, X_test_tfidf, y_train, y_test)

    print("\n✅ Feature engineering complete!")
    print(f"   Vocab size    : {len(vectorizer.vocabulary_)}")
    print(f"   X_train shape : {X_train_tfidf.shape}")
    print(f"   X_test  shape : {X_test_tfidf.shape}")