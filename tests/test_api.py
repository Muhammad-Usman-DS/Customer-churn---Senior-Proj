"""FastAPI endpoints (was scripts/test_fastapi.py).

Calls the route functions directly rather than through TestClient, which would need httpx
(not in requirements.txt). This still exercises the request schema, inference and response shape.
"""
import asyncio
import io

import pytest
from fastapi import UploadFile
from pydantic import ValidationError

from src.app.main import (
    CustomerData,
    get_feature_importance,
    get_model_stats,
    get_prediction,
    predict_batch,
    root,
)

SAMPLE = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 350.75,
}


def test_health():
    assert root() == {"status": "ok"}


def test_predict_returns_valid_response():
    result = get_prediction(CustomerData(**SAMPLE))
    assert "error" not in result, result
    assert result["prediction"] in {"Likely to churn", "Not likely to churn"}
    assert 0 <= result["probability"] <= 100
    assert result["risk_level"] in {"High", "Medium", "Low"}


def test_predict_rejects_missing_field():
    incomplete = {k: v for k, v in SAMPLE.items() if k != "Contract"}
    with pytest.raises(ValidationError):
        CustomerData(**incomplete)


def test_predict_batch(raw_df):
    csv = raw_df.head(5).drop(columns=["Churn"]).to_csv(index=False).encode()
    result = asyncio.run(predict_batch(UploadFile(file=io.BytesIO(csv), filename="customers.csv")))
    assert result["total"] == 5
    assert all("error" not in row and "probability" in row for row in result["predictions"])


def test_feature_importance(feature_cols):
    result = get_feature_importance()
    assert "error" not in result, result
    assert {row["feature"] for row in result["feature_importance"]} == set(feature_cols)


def test_model_stats():
    result = get_model_stats()
    assert "error" not in result, result
    assert result["threshold"] == 0.35
