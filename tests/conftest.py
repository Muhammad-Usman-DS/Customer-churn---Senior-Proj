import os
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent

# Must be set before src.serving.inference is imported: it loads the model at import time
# and defaults to /app/model, which only exists inside the Docker image.
os.environ.setdefault("MODEL_DIR", str(ROOT / "model"))

from src.data.load_data import load_data  # noqa: E402
from src.data.preprocess import preprocess_data  # noqa: E402
from src.features.build_features import build_features  # noqa: E402

RAW_CSV = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET = "Churn"


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return load_data(str(RAW_CSV))


@pytest.fixture(scope="session")
def features_df(raw_df) -> pd.DataFrame:
    """Training-time features, built exactly as scripts/run_pipeline.py does."""
    df = build_features(preprocess_data(raw_df.copy(), target_col=TARGET), target_col=TARGET)
    for c in df.select_dtypes(include=["bool"]).columns:
        df[c] = df[c].astype(int)
    return df


@pytest.fixture(scope="session")
def feature_cols() -> list:
    # model/ is committed (artifacts/ is gitignored), and it's the copy the live API reads
    with open(ROOT / "model" / "feature_columns.txt") as f:
        return [ln.strip() for ln in f if ln.strip()]


@pytest.fixture(scope="session")
def raw_customers(raw_df) -> list:
    """Raw customer dicts as the API receives them (no ID, no target)."""
    return raw_df.drop(columns=["customerID", TARGET]).to_dict(orient="records")
