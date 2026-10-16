"""FastAPI endpoints (was scripts/test_fastapi.py).

Calls the route functions directly rather than through TestClient, which would need httpx
(not in requirements.txt). This still exercises the request schema, inference and response shape.
"""
import asyncio
import io
import json

import pytest
from fastapi import HTTPException, UploadFile
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


def test_predict_invalid_value_returns_422():
    with pytest.raises(HTTPException) as exc:
        get_prediction(CustomerData(**dict(SAMPLE, Contract="Two Yr")))
    assert exc.value.status_code == 422
    assert any("Contract 'Two Yr' is not valid" in d for d in exc.value.detail)


def test_predict_accepts_messy_casing():
    assert get_prediction(CustomerData(**dict(SAMPLE, Contract=" month-to-MONTH "))) == get_prediction(
        CustomerData(**SAMPLE)
    )


def _upload(csv_text: str):
    return asyncio.run(predict_batch(UploadFile(file=io.BytesIO(csv_text.encode()), filename="customers.csv")))


def test_predict_batch(raw_df):
    result = _upload(raw_df.head(5).drop(columns=["Churn"]).to_csv(index=False))
    assert result["total"] == 5
    assert result["failed"] == 0
    assert all("error" not in row and "probability" in row for row in result["predictions"])


def test_predict_batch_flags_bad_rows_and_scores_the_rest(raw_df):
    df = raw_df.head(4).drop(columns=["Churn"])
    df.loc[1, "Contract"] = "Two Yr"  # invalid -> error
    df.loc[2, "Contract"] = "two year "  # messy but recognisable -> scored
    df.loc[3, "gender"] = None  # blank cell -> error
    result = _upload(df.to_csv(index=False))

    rows = result["predictions"]
    assert result["total"] == 4 and result["failed"] == 2
    assert "probability" in rows[0] and "probability" in rows[2]
    assert rows[1]["error"].startswith("Row 2:") and "Contract 'Two Yr'" in rows[1]["error"]
    assert "gender is missing" in rows[3]["error"]
    json.dumps(result, allow_nan=False)  # blank cells must not break the JSON response


def test_predict_batch_missing_column_returns_422(raw_df):
    with pytest.raises(HTTPException) as exc:
        _upload(raw_df.head(3).drop(columns=["Churn", "Contract"]).to_csv(index=False))
    assert exc.value.status_code == 422
    assert "missing columns: Contract" in exc.value.detail[0]


def test_predict_batch_handles_excel_bom_and_padded_headers(raw_df):
    csv = raw_df.head(2).drop(columns=["Churn"]).to_csv(index=False)
    header, rest = csv.split("\n", 1)
    padded = ",".join(f" {h} " for h in header.split(","))
    result = _upload("\ufeff" + padded + "\n" + rest)
    assert result["failed"] == 0


def test_predict_batch_empty_file_returns_422():
    with pytest.raises(HTTPException) as exc:
        _upload("")
    assert exc.value.status_code == 422


def test_feature_importance(feature_cols):
    result = get_feature_importance()
    assert "error" not in result, result
    assert {row["feature"] for row in result["feature_importance"]} == set(feature_cols)


def test_model_stats():
    result = get_model_stats()
    assert "error" not in result, result
    assert result["threshold"] == 0.35
