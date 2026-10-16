"""
FASTAPI SERVING APPLICATION
============================
REST API for the Telco Customer Churn prediction model.
Gradio has been replaced by a custom React frontend.

Endpoints:
- GET  /                    Health check
- POST /predict             Single customer prediction
- POST /predict-batch       Bulk CSV prediction (FOR FRONTEND)
- GET  /feature-importance  Feature importance scores (FOR FRONTEND)
- GET  /model-stats         Model performance metrics (FOR FRONTEND)
"""

import io
import json
import os

import traceback

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File  # FOR FRONTEND: UploadFile for bulk CSV
from fastapi.middleware.cors import CORSMiddleware  # FOR FRONTEND: allow React app to call this API
from pydantic import BaseModel

from src.serving.inference import (  # FOR FRONTEND: expose model internals
    FEATURE_COLS,
    MODEL_DIR,
    REQUIRED_FIELDS,
    InvalidInputError,
    predict,
    sklearn_model,
)

# Initialize FastAPI application
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="ML API for predicting customer churn in telecom industry",
    version="2.0.0"
)

# FOR FRONTEND: CORS middleware — allows the React app (Vercel) to call this API
# In production replace "*" with your actual Vercel URL e.g. "https://telco-churn.vercel.app"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === HEALTH CHECK ===
# HEAD is included because Render's port scanner probes with HEAD requests
@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"status": "ok"}


# Which commit this server was built from (set by CI as a Docker build arg).
# CI polls this after triggering a Render deploy to know when the new version is live.
@app.get("/version")
def version():
    return {"git_sha": os.environ.get("GIT_SHA", "unknown")}


# === REQUEST SCHEMA ===
class CustomerData(BaseModel):
    """19-feature schema for a single customer churn prediction request."""
    # Demographics
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    # Phone
    PhoneService: str
    MultipleLines: str
    # Internet
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    # Account
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    # Numeric
    tenure: int
    MonthlyCharges: float
    TotalCharges: float


# === SINGLE PREDICTION ===
@app.post("/predict")
def get_prediction(data: CustomerData):
    """
    Predict churn for a single customer.
    Returns prediction label, probability %, and risk level.
    Responds 422 with a list of problems if a value isn't one the model knows.
    """
    try:
        return predict(data.model_dump())
    except InvalidInputError as e:
        raise HTTPException(status_code=422, detail=e.errors)
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Prediction failed due to a server error")


def _json_safe(row: dict) -> dict:
    """Blank CSV cells arrive as NaN, which JSON can't encode; send them as null."""
    return {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in row.items()}


# FOR FRONTEND: Bulk prediction — accepts a CSV file, returns predictions for every row
@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Predict churn for multiple customers from a CSV upload.
    The CSV must contain the same 19 columns as the single predict endpoint.
    Returns a list of predictions, one per row. Rows that can't be scored are kept,
    with an "error" field instead of a prediction, and counted in "failed".
    Responds 422 if the file can't be read or required columns are missing.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8-sig")))  # -sig strips Excel's BOM
    except Exception:
        raise HTTPException(status_code=422, detail=["Could not read the file as a CSV"])

    df.columns = df.columns.str.strip()
    missing = [c for c in REQUIRED_FIELDS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=[f"CSV is missing columns: {', '.join(missing)}"])
    if df.empty:
        raise HTTPException(status_code=422, detail=["CSV has no rows"])

    results = []
    for i, row_dict in enumerate(df.to_dict(orient="records"), start=1):
        row_dict = _json_safe(row_dict)
        try:
            results.append({**row_dict, **predict(row_dict)})
        except InvalidInputError as e:
            results.append({**row_dict, "error": f"Row {i}: " + "; ".join(e.errors)})
        except Exception:
            traceback.print_exc()
            results.append({**row_dict, "error": f"Row {i}: server error"})

    failed = sum("error" in r for r in results)
    return {"predictions": results, "total": len(results), "failed": failed}


# FOR FRONTEND: Feature importance — drives the bar chart on the Insights page
@app.get("/feature-importance")
def get_feature_importance():
    """
    Returns XGBoost feature importance scores for all 30 model features.
    Used to render the feature importance bar chart on the frontend.
    """
    if sklearn_model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    try:
        importances = sklearn_model.feature_importances_.tolist()
        data = [
            {"feature": feat, "importance": round(imp, 4)}
            for feat, imp in zip(FEATURE_COLS, importances)
        ]
        # Sort descending so frontend can render top-N easily
        data.sort(key=lambda x: x["importance"], reverse=True)
        return {"feature_importance": data}
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Could not compute feature importance")


# FOR FRONTEND: Model stats — drives the metrics cards on the Insights page
@app.get("/model-stats")
def get_model_stats():
    """
    Returns model performance metrics logged during training.
    Reads from model_metrics.json stored alongside the model artifacts.
    """
    try:
        metrics_path = os.path.join(MODEL_DIR, "model_metrics.json")
        with open(metrics_path) as f:
            stats = json.load(f)
        return stats
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Model metrics not available")
