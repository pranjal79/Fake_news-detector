import pandas as pd
import yaml
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def extract_data() -> pd.DataFrame:
    params = load_params()

    fake_path = params["data"]["raw_fake"]
    true_path = params["data"]["raw_true"]

    logger.info(f"Loading Fake news from: {fake_path}")
    fake_df = pd.read_csv(fake_path)
    fake_df["label"] = 0  # Fake

    logger.info(f"Loading True news from: {true_path}")
    true_df = pd.read_csv(true_path)
    true_df["label"] = 1  # Real

    logger.info(f"Fake: {len(fake_df)} | True: {len(true_df)}")

    # Merge datasets
    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(f"Total records: {len(df)}")

    # ── Save interim output for DVC pipeline ────────────────
    os.makedirs("data/interim", exist_ok=True)
    interim_path = "data/interim/raw_combined.csv"

    df.to_csv(interim_path, index=False)
    logger.info(f"Saved interim data → {interim_path}")

    return df


if __name__ == "__main__":
    df = extract_data()

    logger.info("\nSample Data:")
    logger.info(df.head())

    logger.info("\nLabel Distribution:")
    logger.info(df["label"].value_counts())