import pandas as pd
import re
import yaml
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
STOPWORDS = set(stopwords.words("english"))

def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ── Step 1: Clean raw text ──────────────────────────────────────────
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

# ── Step 2: Tokenize ────────────────────────────────────────────────
def tokenize(text: str) -> list:
    return word_tokenize(text)

# ── Step 3: Remove stopwords + Lemmatize ────────────────────────────
def remove_stopwords_and_lemmatize(tokens: list) -> list:
    return [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in STOPWORDS and len(token) > 2
    ]

# ── Step 4: Full pipeline per row ───────────────────────────────────
def preprocess_text(text: str) -> str:
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    filtered = remove_stopwords_and_lemmatize(tokens)
    return " ".join(filtered)

# ── Step 5: Transform entire DataFrame ──────────────────────────────
def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting text transformation...")

    # Combine title + text for richer features
    logger.info("Combining 'title' and 'text' columns...")
    df["combined_text"] = df["title"].fillna("") + " " + df["text"].fillna("")

    # Apply preprocessing
    logger.info("Applying text cleaning, tokenization, stopword removal...")
    from tqdm import tqdm
    tqdm.pandas()
    df["processed_text"] = df["combined_text"].progress_apply(preprocess_text)

    # Drop rows where processed text is empty
    before = len(df)
    df = df[df["processed_text"].str.strip() != ""].reset_index(drop=True)
    after = len(df)
    logger.info(f"Dropped {before - after} empty rows after preprocessing")

    # Keep only needed columns
    df = df[["title", "text", "combined_text", "processed_text", "label"]]

    logger.info(f"Transformation complete. Shape: {df.shape}")
    return df

# ── Step 6: Save processed data ─────────────────────────────────────
def save_processed_data(df: pd.DataFrame) -> None:
    params = load_params()
    output_path = params["data"]["processed"]

    df.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to: {output_path}")

if __name__ == "__main__":
    from src.etl.extract import extract_data

    # Extract
    raw_df = extract_data()

    # Transform
    processed_df = transform_data(raw_df)

    # Preview
    print(processed_df.head(3))
    print(f"\nShape: {processed_df.shape}")
    print(f"\nSample processed text:\n{processed_df['processed_text'][0]}")

    # Save
    save_processed_data(processed_df)