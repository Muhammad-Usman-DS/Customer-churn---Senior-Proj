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

import pandas as pd
from fastapi import FastAPI, UploadFile, File  # FOR FRONTEND: UploadFile for bulk CSV
from fastapi.middleware.cors import CORSMiddleware  # FOR FRONTEND: allow React app to call this API
from pydantic import BaseModel

from src.serving.inference import predict, sklearn_model, FEATURE_COLS  # FOR FRONTEND: expose model internals

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
@app.get("/")
def root():
    return {"status": "ok"}


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
    """
    try:
        result = predict(data.dict())
        return result
    except Exception as e:
        return {"error": str(e)}


# FOR FRONTEND: Bulk prediction — accepts a CSV file, returns predictions for every row
@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Predict churn for multiple customers from a CSV upload.
    The CSV must contain the same 19 columns as the single predict endpoint.
    Returns a list of predictions, one per row.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        results = []
        for _, row in df.iterrows():
            try:
                row_dict = row.to_dict()
                pred = predict(row_dict)
                results.append({**row_dict, **pred})
            except Exception as row_err:
                results.append({**row.to_dict(), "error": str(row_err)})

        return {"predictions": results, "total": len(results)}
    except Exception as e:
        return {"error": str(e)}


# FOR FRONTEND: Feature importance — drives the bar chart on the Insights page
@app.get("/feature-importance")
def get_feature_importance():
    """
    Returns XGBoost feature importance scores for all 30 model features.
    Used to render the feature importance bar chart on the frontend.
    """
    try:
        if sklearn_model is None:
            return {"error": "Sklearn model not available"}

        importances = sklearn_model.feature_importances_.tolist()
        data = [
            {"feature": feat, "importance": round(imp, 4)}
            for feat, imp in zip(FEATURE_COLS, importances)
        ]
        # Sort descending so frontend can render top-N easily
        data.sort(key=lambda x: x["importance"], reverse=True)
        return {"feature_importance": data}
    except Exception as e:
        return {"error": str(e)}


# FOR FRONTEND: Model stats — drives the metrics cards on the Insights page
@app.get("/model-stats")
def get_model_stats():
    """
    Returns model performance metrics logged during training.
    Reads from model_metrics.json stored alongside the model artifacts.
    """
    try:
        metrics_path = os.path.join("/app/model", "model_metrics.json")
        with open(metrics_path) as f:
            stats = json.load(f)
        return stats
    except Exception as e:
        return {"error": str(e)}
