"""Train/serve consistency: the API must encode a customer exactly as training did."""
import pandas as pd
import pytest

from src.serving import inference
from tests.conftest import TARGET

N_CUSTOMERS = 200


def test_serving_features_match_training(raw_customers, features_df, feature_cols):
    train = features_df.drop(columns=[TARGET])[feature_cols].astype(float)
    mismatched = []
    for i, customer in enumerate(raw_customers[:N_CUSTOMERS]):
        served = inference._serve_transform(pd.DataFrame([customer])).iloc[0].astype(float)
        diff = list(served.index[served != train.iloc[i]])
        if diff:
            mismatched.append((i, diff))
    assert mismatched == [], f"{len(mismatched)} customers encoded differently, e.g. {mismatched[:3]}"


def test_serving_column_order_matches_model(raw_customers, feature_cols):
    served = inference._serve_transform(pd.DataFrame([raw_customers[0]]))
    assert list(served.columns) == feature_cols


def test_predict_probability_matches_training_encoding(raw_customers, features_df, feature_cols):
    X = features_df.drop(columns=[TARGET])[feature_cols]
    expected = inference.sklearn_model.predict_proba(X.iloc[:50])[:, 1]
    for customer, p in zip(raw_customers[:50], expected):
        assert inference.predict(customer)["probability"] == pytest.approx(round(p * 100, 1))


def test_contract_type_changes_prediction(raw_customers):
    customer = raw_customers[0]
    probs = {
        c: inference.predict(dict(customer, Contract=c))["probability"]
        for c in ["Month-to-month", "One year", "Two year"]
    }
    # Contract is one of the strongest churn drivers; longer contracts must lower risk
    assert probs["Month-to-month"] > probs["One year"] > probs["Two year"], probs


def test_unknown_category_does_not_crash(raw_customers):
    result = inference.predict(dict(raw_customers[0], PaymentMethod="Crypto"))
    assert 0 <= result["probability"] <= 100
