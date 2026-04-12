import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ── Initialize once at module load ──────────────────────────────────
lemmatizer = WordNetLemmatizer()
STOPWORDS  = set(stopwords.words("english"))

# ── Step 1: Clean raw text ───────────────────────────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove punctuation and special characters
    text = re.sub(r"[^a-z\s]", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

# ── Step 2: Tokenize ─────────────────────────────────────────────────
def tokenize(text: str) -> list:
    return word_tokenize(text)

# ── Step 3: Remove stopwords + Lemmatize ────────────────────────────
def remove_stopwords_and_lemmatize(tokens: list) -> list:
    return [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in STOPWORDS and len(token) > 2
    ]

# ── Step 4: Full pipeline (single entry point for app) ──────────────
def preprocess_input(text: str) -> str:
    """
    Full preprocessing pipeline for user input.
    Must match training pipeline exactly.

    Args:
        text: Raw news text from user input

    Returns:
        Cleaned, tokenized, lemmatized string
    """
    cleaned  = clean_text(text)
    tokens   = tokenize(cleaned)
    filtered = remove_stopwords_and_lemmatize(tokens)
    result   = " ".join(filtered)
    return result


# ── Quick test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_texts = [
        "BREAKING: Donald Trump says the election was RIGGED! Visit www.fakenews.com",
        "Washington (Reuters) - The Federal Reserve raised interest rates on Wednesday.",
        "Scientists discover new vaccine that cures ALL diseases!! Click here!!!"
    ]

    print("=" * 60)
    print("PREPROCESSING TEST")
    print("=" * 60)

    for text in sample_texts:
        result = preprocess_input(text)
        print(f"\n📥 Input:\n{text}")
        print(f"\n✅ Processed:\n{result}")
        print("-" * 60)