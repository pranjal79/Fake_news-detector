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
    fake_df["label"] = 0  # 0 = Fake

    logger.info(f"Loading True news from: {true_path}")
    true_df = pd.read_csv(true_path)
    true_df["label"] = 1  # 1 = Real

    logger.info(f"Fake records: {len(fake_df)} | True records: {len(true_df)}")

    # Combine both
    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(f"Total records after merge: {len(df)}")
    logger.info(f"Columns: {df.columns.tolist()}")

    return df

if __name__ == "__main__":
    df = extract_data()
    print(df.head())
    print(df["label"].value_counts())