"""
INFERENCE PIPELINE - Production ML Model Serving with Feature Consistency
=========================================================================

This module provides the core inference functionality for the Telco Churn prediction model.
It ensures that serving-time feature transformations exactly match training-time transformations,
which is CRITICAL for model accuracy in production.

Key Responsibilities:
1. Load MLflow-logged model and feature metadata from training
2. Apply identical feature transformations as used during training
3. Ensure correct feature ordering for model input
4. Return prediction label + churn probability for the frontend

CRITICAL PATTERN: Training/Serving Consistency
- Uses fixed BINARY_MAP for deterministic binary encoding
- Applies one-hot encoding aligned to the training columns (training uses drop_first=True)
- Maintains exact feature column order from training
- Handles missing/new categorical values gracefully

Production Deployment:
- MODEL_DIR points to containerized model artifacts
- Feature schema loaded from training-time artifacts
- Optimized for single-row inference (real-time serving)
"""

import os
import pandas as pd
import mlflow
import mlflow.sklearn

# === MODEL LOADING CONFIGURATION ===
# IMPORTANT: This path is set during Docker container build
# In development: uses local MLflow artifacts
# In production: uses model copied to container at build time
# Override with the MODEL_DIR env var (e.g. tests point it at the repo's model/ folder)
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/model")

# === CLASSIFICATION THRESHOLD ===
# Lowered from default 0.5 to 0.35 to prioritize recall (catching churners)
# This matches the threshold used during model evaluation in run_pipeline.py
THRESHOLD = 0.35

try:
    # Load MLflow pyfunc model (used for standard prediction)
    model = mlflow.pyfunc.load_model(MODEL_DIR)
    print(f"✅ Model loaded successfully from {MODEL_DIR}")
except Exception as e:
    print(f"❌ Failed to load model from {MODEL_DIR}: {e}")
    try:
        import glob
        local_model_paths = glob.glob("./mlruns/*/*/artifacts/model")
        if local_model_paths:
            latest_model = max(local_model_paths, key=os.path.getmtime)
            model = mlflow.pyfunc.load_model(latest_model)
            MODEL_DIR = latest_model
            print(f"✅ Fallback: Loaded model from {latest_model}")
        else:
            raise Exception("No model found in local mlruns")
    except Exception as fallback_error:
        raise Exception(f"Failed to load model: {e}. Fallback failed: {fallback_error}")

# FOR FRONTEND: Load sklearn model separately to access predict_proba and feature importances
try:
    sklearn_model = mlflow.sklearn.load_model(MODEL_DIR)
    print(f"✅ Sklearn model loaded for probability scoring")
except Exception as e:
    print(f"⚠️  Sklearn model load failed, probabilities unavailable: {e}")
    sklearn_model = None

# === FEATURE SCHEMA LOADING ===
# CRITICAL: Load the exact feature column order used during training
try:
    feature_file = os.path.join(MODEL_DIR, "feature_columns.txt")
    with open(feature_file) as f:
        FEATURE_COLS = [ln.strip() for ln in f if ln.strip()]
    print(f"✅ Loaded {len(FEATURE_COLS)} feature columns from training")
except Exception as e:
    raise Exception(f"Failed to load feature columns: {e}")

# === FEATURE TRANSFORMATION CONSTANTS ===
# CRITICAL: These mappings must exactly match those used in training
# Any changes here will cause train/serve skew and degrade model performance

# Deterministic binary feature mappings (consistent with training)
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},           # Demographics
    "Partner": {"No": 0, "Yes": 1},               # Has partner
    "Dependents": {"No": 0, "Yes": 1},            # Has dependents
    "PhoneService": {"No": 0, "Yes": 1},          # Phone service
    "PaperlessBilling": {"No": 0, "Yes": 1},      # Billing preference
}

# Numeric columns that need type coercion
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# === INPUT VALIDATION ===
# Every category seen in training, spelled exactly as in the training data.
# _serve_transform silently maps an unknown or misspelled value to the baseline category
# (e.g. "Two Year" -> Month-to-month), so inputs are normalised and checked against this first.
CATEGORIES = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}

REQUIRED_FIELDS = list(CATEGORIES) + ["SeniorCitizen"] + NUMERIC_COLS


class InvalidInputError(ValueError):
    """Raised when a customer record can't be scored reliably; `errors` lists every problem."""

    def __init__(self, errors: list):
        super().__init__("; ".join(errors))
        self.errors = errors


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v)) or (isinstance(v, str) and not v.strip())


def _canonical(v) -> str:
    """Case- and whitespace-insensitive key: ' Two  YEAR ' -> 'two year'."""
    return " ".join(str(v).split()).lower()


_LOOKUP = {col: {_canonical(v): v for v in values} for col, values in CATEGORIES.items()}


def normalize_input(input_dict: dict) -> dict:
    """
    Clean a raw customer record and reject anything the model would silently misread.

    - Trims whitespace and ignores capitalisation for categories ("two year " -> "Two year")
    - Accepts SeniorCitizen as 0/1 or Yes/No
    - Treats a blank TotalCharges as 0, matching training (new customers have no charges yet)
    - Ignores extra fields such as customerID or Churn

    Raises InvalidInputError listing every missing field and invalid value.
    """
    errors = []
    clean = {}

    for col, lookup in _LOOKUP.items():
        v = input_dict.get(col)
        if _is_missing(v):
            errors.append(f"{col} is missing")
        elif _canonical(v) not in lookup:
            errors.append(f"{col} '{v}' is not valid (expected one of: {', '.join(CATEGORIES[col])})")
        else:
            clean[col] = lookup[_canonical(v)]

    senior = input_dict.get("SeniorCitizen")
    senior_map = {"0": 0, "1": 1, "0.0": 0, "1.0": 1, "no": 0, "yes": 1}
    if _is_missing(senior):
        errors.append("SeniorCitizen is missing")
    elif _canonical(senior) not in senior_map:
        errors.append(f"SeniorCitizen '{senior}' is not valid (expected 0 or 1)")
    else:
        clean["SeniorCitizen"] = senior_map[_canonical(senior)]

    for col in NUMERIC_COLS:
        v = input_dict.get(col)
        if _is_missing(v):
            if col == "TotalCharges":
                clean[col] = 0.0
            else:
                errors.append(f"{col} is missing")
            continue
        num = pd.to_numeric(str(v).strip(), errors="coerce")
        if pd.isna(num):
            errors.append(f"{col} '{v}' is not a number")
        elif num < 0:
            errors.append(f"{col} cannot be negative (got {v})")
        else:
            clean[col] = float(num)

    if errors:
        raise InvalidInputError(errors)
    return clean

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply identical feature transformations as used during model training.

    Transformation Pipeline:
    1. Clean column names and handle data types
    2. Apply deterministic binary encoding (using BINARY_MAP)
    3. One-hot encode remaining categorical features
    4. Convert boolean columns to integers
    5. Align features with training schema and order

    Args:
        df: Single-row DataFrame with raw customer data

    Returns:
        DataFrame with features transformed and ordered for model input
    """
    df = df.copy()

    # Clean column names (remove any whitespace)
    df.columns = df.columns.str.strip()

    # === STEP 1: Numeric Type Coercion ===
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(0)

    # === STEP 2: Binary Feature Encoding ===
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .map(mapping)
                .astype("Int64")
                .fillna(0)
                .astype(int)
            )

    # === STEP 3: One-Hot Encoding for Remaining Categorical Features ===
    # NOTE: drop_first must be False here. Inference sees one row at a time, so each column
    # has a single category and drop_first=True would drop it, zeroing out every one-hot
    # feature. Training's drop_first baseline is reproduced by the reindex in STEP 5, which
    # discards the baseline-category columns that aren't in FEATURE_COLS.
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=False)

    # === STEP 4: Boolean to Integer Conversion ===
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # === STEP 5: Feature Alignment with Training Schema ===
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)

    return df


def predict(input_dict: dict) -> dict:
    """
    Main prediction function — returns both label and churn probability.

    Args:
        input_dict: Dictionary containing raw customer data (19 features)

    Returns:
        Dict with:
        - prediction: "Likely to churn" or "Not likely to churn"
        - probability: churn probability as a percentage (0-100)
        - risk_level: "High", "Medium", or "Low"

    Raises:
        InvalidInputError: if a field is missing or has a value the model wasn't trained on
    """
    df = pd.DataFrame([normalize_input(input_dict)])
    df_enc = _serve_transform(df)

    # FOR FRONTEND: Use sklearn model for probability + apply custom threshold
    if sklearn_model is not None:
        proba = float(sklearn_model.predict_proba(df_enc)[0][1])
        result = 1 if proba >= THRESHOLD else 0
    else:
        # Fallback to pyfunc predict if sklearn model unavailable
        preds = model.predict(df_enc)
        if hasattr(preds, "tolist"):
            preds = preds.tolist()
        result = preds[0] if isinstance(preds, (list, tuple)) else preds
        proba = None

    # FOR FRONTEND: Compute risk level for colour-coded UI display
    prob_pct = round(proba * 100, 1) if proba is not None else None
    if prob_pct is not None:
        if prob_pct >= 65:
            risk_level = "High"
        elif prob_pct >= 35:
            risk_level = "Medium"
        else:
            risk_level = "Low"
    else:
        risk_level = "High" if result == 1 else "Low"

    return {
        "prediction": "Likely to churn" if result == 1 else "Not likely to churn",
        "probability": prob_pct,
        "risk_level": risk_level,
    }
