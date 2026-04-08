import pandas as pd
import yaml
from pymongo import MongoClient, errors
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ── Connect to MongoDB ───────────────────────────────────────────────
def get_mongo_client(uri: str) -> MongoClient:
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        # Force connection check
        client.server_info()
        logger.info("MongoDB connection successful")
        return client
    except errors.ServerSelectionTimeoutError as e:
        logger.error(f"MongoDB connection failed: {e}")
        raise

# ── Load Raw Data into MongoDB ───────────────────────────────────────
def load_raw_data(df: pd.DataFrame) -> None:
    params = load_params()
    uri     = params["mongodb"]["uri"]
    db_name = params["mongodb"]["db_name"]
    col     = params["mongodb"]["raw_collection"]

    client = get_mongo_client(uri)
    db = client[db_name]
    collection = db[col]

    # Drop existing to avoid duplicates on re-run
    collection.drop()
    logger.info(f"Dropped existing collection: {col}")

    # Convert DataFrame to list of dicts
    records = df.to_dict(orient="records")

    # Insert in batches of 1000
    batch_size = 1000
    total = len(records)

    for i in range(0, total, batch_size):
        batch = records[i : i + batch_size]
        collection.insert_many(batch)
        logger.info(f"Inserted raw batch {i // batch_size + 1} "
                    f"({min(i + batch_size, total)}/{total})")

    logger.info(f"✅ Raw data loaded: {total} records → '{db_name}.{col}'")
    client.close()

# ── Load Processed Data into MongoDB ────────────────────────────────
def load_processed_data(df: pd.DataFrame) -> None:
    params = load_params()
    uri     = params["mongodb"]["uri"]
    db_name = params["mongodb"]["db_name"]
    col     = params["mongodb"]["processed_collection"]

    client = get_mongo_client(uri)
    db = client[db_name]
    collection = db[col]

    # Drop existing to avoid duplicates on re-run
    collection.drop()
    logger.info(f"Dropped existing collection: {col}")

    records = df.to_dict(orient="records")

    batch_size = 1000
    total = len(records)

    for i in range(0, total, batch_size):
        batch = records[i : i + batch_size]
        collection.insert_many(batch)
        logger.info(f"Inserted processed batch {i // batch_size + 1} "
                    f"({min(i + batch_size, total)}/{total})")

    logger.info(f"✅ Processed data loaded: {total} records → '{db_name}.{col}'")
    client.close()

# ── Fetch Data back from MongoDB (for downstream use) ───────────────
def fetch_processed_data() -> pd.DataFrame:
    params = load_params()
    uri     = params["mongodb"]["uri"]
    db_name = params["mongodb"]["db_name"]
    col     = params["mongodb"]["processed_collection"]

    client = get_mongo_client(uri)
    db = client[db_name]
    collection = db[col]

    records = list(collection.find({}, {"_id": 0}))
    df = pd.DataFrame(records)

    logger.info(f"Fetched {len(df)} records from '{db_name}.{col}'")
    client.close()
    return df

if __name__ == "__main__":
    from src.etl.extract import extract_data
    from src.etl.transform import transform_data

    # Extract
    logger.info("=== EXTRACT ===")
    raw_df = extract_data()

    # Transform
    logger.info("=== TRANSFORM ===")
    processed_df = transform_data(raw_df)

    # Load both into MongoDB
    logger.info("=== LOAD ===")
    load_raw_data(raw_df)
    load_processed_data(processed_df)

    # Verify by fetching back
    logger.info("=== VERIFY ===")
    fetched = fetch_processed_data()
    print(f"\nFetched shape: {fetched.shape}")
    print(fetched[["processed_text", "label"]].head(3))