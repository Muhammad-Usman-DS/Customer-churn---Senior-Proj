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
- Applies same one-hot encoding with drop_first=True
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
MODEL_DIR = "/app/model"

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
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

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
    """
    df = pd.DataFrame([input_dict])
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
