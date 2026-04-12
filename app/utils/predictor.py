import os
import pickle
import numpy as np
from app.utils.preprocess import preprocess_input

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
VECTORIZER_PATH = os.path.join(BASE_DIR, "data", "features", "tfidf_vectorizer.pkl")
MODEL_DIR      = os.path.join(BASE_DIR, "models")

# ── Label mapping ────────────────────────────────────────────────────
LABELS = {
    0: "FAKE",
    1: "REAL"
}

# ── Load vectorizer ──────────────────────────────────────────────────
def load_vectorizer():
    """Load the saved TF-IDF vectorizer from disk."""
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            f"Vectorizer not found at: {VECTORIZER_PATH}\n"
            f"Please run the training pipeline first."
        )
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return vectorizer

# ── Load model ───────────────────────────────────────────────────────
def load_model(model_name: str = "SVM"):
    """
    Load a trained model from disk.

    Args:
        model_name: One of 'SVM', 'LogisticRegression', 'NaiveBayes'

    Returns:
        Loaded sklearn model
    """
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at: {model_path}\n"
            f"Please run the training pipeline first."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model

# ── Get available models ─────────────────────────────────────────────
def get_available_models() -> list:
    """Return list of trained model names available on disk."""
    available = []
    for name in ["SVM", "LogisticRegression", "NaiveBayes"]:
        path = os.path.join(MODEL_DIR, f"{name}.pkl")
        if os.path.exists(path):
            available.append(name)
    return available

# ── Core prediction function ─────────────────────────────────────────
def predict(text: str, model_name: str = "SVM") -> dict:
    """
    Full prediction pipeline:
    raw text → preprocess → TF-IDF → model → label

    Args:
        text       : Raw news text from user
        model_name : Which model to use for prediction

    Returns:
        dict with keys:
            - label      : 'FAKE' or 'REAL'
            - confidence : probability score (0.0 to 1.0)
            - processed  : cleaned text (for transparency)
            - model_used : which model was used
    """

    # ── Validate input ───────────────────────────────────────
    if not text or not text.strip():
        return {
            "label"      : None,
            "confidence" : None,
            "processed"  : None,
            "model_used" : model_name,
            "error"      : "Input text is empty."
        }

    if len(text.strip()) < 20:
        return {
            "label"      : None,
            "confidence" : None,
            "processed"  : None,
            "model_used" : model_name,
            "error"      : "Text too short. Please enter at least 20 characters."
        }

    # ── Step 1: Preprocess ───────────────────────────────────
    processed_text = preprocess_input(text)

    if not processed_text.strip():
        return {
            "label"      : None,
            "confidence" : None,
            "processed"  : processed_text,
            "model_used" : model_name,
            "error"      : "Text became empty after preprocessing."
        }

    # ── Step 2: Load artifacts ───────────────────────────────
    vectorizer = load_vectorizer()
    model      = load_model(model_name)

    # ── Step 3: Vectorize ────────────────────────────────────
    features = vectorizer.transform([processed_text])

    # ── Step 4: Predict ──────────────────────────────────────
    prediction = model.predict(features)[0]
    label      = LABELS[int(prediction)]

    # ── Step 5: Confidence score ─────────────────────────────
    confidence = None
    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(features)[0]
        confidence = round(float(np.max(proba)) * 100, 2)
    elif hasattr(model, "decision_function"):
        decision   = model.decision_function(features)[0]
        # Convert decision score to 0-100 scale
        confidence = round(
            float(1 / (1 + np.exp(-abs(decision)))) * 100, 2
        )

    return {
        "label"      : label,
        "confidence" : confidence,
        "processed"  : processed_text,
        "model_used" : model_name,
        "error"      : None
    }


# ── Quick test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        {
            "text"  : "BREAKING NEWS: Scientists confirm the earth is flat "
                      "and NASA has been lying for decades. Share before "
                      "they delete this! The government is hiding the truth.",
            "label" : "Expected: FAKE"
        },
        {
            "text"  : "Washington (Reuters) - The Federal Reserve raised its "
                      "benchmark interest rate by a quarter of a percentage "
                      "point on Wednesday, citing continued strength in the "
                      "labor market and rising inflation pressures.",
            "label" : "Expected: REAL"
        }
    ]

    print("=" * 60)
    print("PREDICTOR TEST")
    print("=" * 60)

    available = get_available_models()
    print(f"Available models: {available}\n")

    for case in test_cases:
        result = predict(case["text"], model_name="SVM")
        print(f"📥 Input (truncated):\n{case['text'][:80]}...")
        print(f"🏷️  {case['label']}")
        print(f"🤖 Prediction  : {result['label']}")
        print(f"📊 Confidence  : {result['confidence']}%")
        print(f"⚙️  Model used  : {result['model_used']}")
        print(f"🧹 Processed   : {result['processed'][:60]}...")
        if result["error"]:
            print(f"❌ Error: {result['error']}")
        print("-" * 60)