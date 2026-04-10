import json
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

def save_metrics(results: dict, best_model: str) -> None:
    os.makedirs("reports", exist_ok=True)

    report = {
        "best_model": best_model,
        "models": results
    }

    path = "reports/metrics.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Metrics report saved → {path}")
    print(f"\n📊 Metrics saved to {path}")