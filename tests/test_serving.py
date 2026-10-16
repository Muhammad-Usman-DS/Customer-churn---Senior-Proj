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


# --- Input validation: messy values must be normalised or rejected, never silently misread ---

@pytest.mark.parametrize(
    "field, messy, clean",
    [
        ("Contract", "Two Year", "Two year"),
        ("Contract", "  two year ", "Two year"),
        ("InternetService", "fiber optic", "Fiber optic"),
        ("InternetService", "Fiber  Optic", "Fiber optic"),
        ("Partner", "yes", "Yes"),
        ("OnlineSecurity", "YES", "Yes"),
        ("PaymentMethod", "electronic check", "Electronic check"),
        ("SeniorCitizen", "Yes", 1),
        ("SeniorCitizen", "0", 0),
    ],
)
def test_messy_values_give_same_prediction_as_clean(raw_customers, field, messy, clean):
    customer = raw_customers[0]
    assert inference.predict(dict(customer, **{field: messy})) == inference.predict(dict(customer, **{field: clean}))


@pytest.mark.parametrize(
    "change, expected_error",
    [
        ({"Contract": "Two Yr"}, "Contract 'Two Yr' is not valid"),
        ({"PaymentMethod": "Crypto"}, "PaymentMethod 'Crypto' is not valid"),
        ({"Contract": None}, "Contract is missing"),
        ({"Contract": "   "}, "Contract is missing"),
        ({"tenure": "abc"}, "tenure 'abc' is not a number"),
        ({"MonthlyCharges": -5}, "MonthlyCharges cannot be negative"),
        ({"SeniorCitizen": 2}, "SeniorCitizen '2' is not valid"),
    ],
)
def test_invalid_values_are_rejected(raw_customers, change, expected_error):
    with pytest.raises(inference.InvalidInputError) as exc:
        inference.predict(dict(raw_customers[0], **change))
    assert any(expected_error in e for e in exc.value.errors), exc.value.errors


NO_INTERNET = {"InternetService": "No", **{c: "No internet service" for c in inference.INTERNET_ADDONS}}


@pytest.mark.parametrize(
    "change, expected_error",
    [
        (dict(NO_INTERNET, OnlineSecurity="Yes"), "OnlineSecurity must be 'No internet service'"),
        ({"InternetService": "DSL", "TechSupport": "No internet service"}, "TechSupport can't be 'No internet service'"),
        ({"PhoneService": "No", "MultipleLines": "Yes"}, "MultipleLines must be 'No phone service'"),
        ({"PhoneService": "Yes", "MultipleLines": "No phone service"}, "MultipleLines can't be 'No phone service'"),
    ],
)
def test_impossible_combinations_are_rejected(raw_customers, change, expected_error):
    with pytest.raises(inference.InvalidInputError) as exc:
        inference.predict(dict(raw_customers[0], **change))
    assert any(expected_error in e for e in exc.value.errors), exc.value.errors


def test_consistent_no_service_customers_are_accepted(raw_customers):
    customer = dict(raw_customers[0], PhoneService="No", MultipleLines="No phone service", **NO_INTERNET)
    assert 0 <= inference.predict(customer)["probability"] <= 100


def test_every_real_customer_is_accepted(raw_customers):
    for customer in raw_customers:
        inference.normalize_input(customer)  # raises if validation rejects a genuine record


def test_missing_field_is_rejected(raw_customers):
    customer = {k: v for k, v in raw_customers[0].items() if k != "Contract"}
    with pytest.raises(inference.InvalidInputError, match="Contract is missing"):
        inference.predict(customer)


def test_all_problems_reported_at_once(raw_customers):
    with pytest.raises(inference.InvalidInputError) as exc:
        inference.predict(dict(raw_customers[0], Contract="x", gender="y", tenure="z"))
    assert len(exc.value.errors) == 3


def test_blank_total_charges_treated_as_zero_like_training(raw_customers):
    # New customers have a blank TotalCharges in the raw data; preprocess_data fills it with 0
    customer = raw_customers[0]
    assert inference.predict(dict(customer, TotalCharges=" ")) == inference.predict(dict(customer, TotalCharges=0))


def test_extra_fields_are_ignored(raw_customers):
    customer = raw_customers[0]
    assert inference.predict(dict(customer, customerID="1234-ABCD", Churn="Yes")) == inference.predict(customer)


def test_categories_match_trained_features(feature_cols):
    # Every non-baseline category must have a model column, or it would be silently dropped
    for col, values in inference.CATEGORIES.items():
        if col in inference.BINARY_MAP:
            assert set(values) == set(inference.BINARY_MAP[col]), col
        else:
            baseline, *others = sorted(values)
            for v in others:
                assert f"{col}_{v}" in feature_cols, f"{col}_{v}"
            assert f"{col}_{baseline}" not in feature_cols, f"{col}_{baseline} should be the dropped baseline"
