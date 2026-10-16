import os
from typing import List, Tuple

import pandas as pd

# GX sends usage analytics by default; keep training runs and CI offline and quiet
os.environ.setdefault("GX_ANALYTICS_ENABLED", "False")

import great_expectations as gx  # noqa: E402
import great_expectations.expectations as gxe  # noqa: E402

# This module is responsible for validating the raw data before it enters the ML pipeline.
# We use Great Expectations to define a comprehensive set of expectations that the data must meet.
# this function is for QUALITY CHECKS

# this step is after loading the data (which is done in src/data/load_data.py)
# and before preprocessing (which is done in src/data/preprocess.py)
# for example:
#
#      1.       in the "gender" column, we expect only "Male" and "Female" values, and no nulls.
#               If we find any other values or nulls, the validation will fail.
#
#      2.       in the "tenure" column, we expect numeric values between 0 and 120 (months).
#               If we find negative values, or values greater than 120, or nulls, the validation will fail.

# So here we set what we expect to receive in the raw data, and if the data doesn't meet these expectations,
# we can catch it early and prevent bad data from entering our ML pipeline.

# Written against the Great Expectations 1.x API (requirements.txt pins 1.5.8). The pre-1.0
# ge.dataset.PandasDataset API this used to call no longer exists.

YES_NO = ["Yes", "No"]
INTERNET_ADDON_VALUES = ["Yes", "No", "No internet service"]

# Allowed values for every categorical column (same spelling as the raw CSV)
CATEGORICAL_SETS = {
    "gender": ["Male", "Female"],
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": INTERNET_ADDON_VALUES,
    "OnlineBackup": INTERNET_ADDON_VALUES,
    "DeviceProtection": INTERNET_ADDON_VALUES,
    "TechSupport": INTERNET_ADDON_VALUES,
    "StreamingTV": INTERNET_ADDON_VALUES,
    "StreamingMovies": INTERNET_ADDON_VALUES,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YES_NO,
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
    "Churn": YES_NO,
}


def _build_expectations() -> List[gxe.Expectation]:
    expectations: List[gxe.Expectation] = []

    # === SCHEMA VALIDATION - ESSENTIAL COLUMNS ===
    # Customer identifier must exist (required for business operations)
    expectations += [
        gxe.ExpectColumnToExist(column="customerID"),
        gxe.ExpectColumnValuesToNotBeNull(column="customerID"),
        gxe.ExpectColumnValuesToBeUnique(column="customerID"),
    ]
    required = list(CATEGORICAL_SETS) + ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    expectations += [gxe.ExpectColumnToExist(column=c) for c in required]

    # === BUSINESS LOGIC VALIDATION ===
    # Every categorical column may only contain the values the model was trained on, and no nulls
    for col, values in CATEGORICAL_SETS.items():
        expectations += [
            gxe.ExpectColumnValuesToNotBeNull(column=col),
            gxe.ExpectColumnValuesToBeInSet(column=col, value_set=values),
        ]
    expectations.append(gxe.ExpectColumnValuesToBeInSet(column="SeniorCitizen", value_set=[0, 1]))

    # === NUMERIC RANGE VALIDATION ===
    # Tenure: non-negative, at most ~10 years (120 months) for telecom
    # Monthly charges: no negative bills, within a reasonable business range
    # Total charges: non-negative (blank for brand-new customers, coerced to NaN and allowed here)
    expectations += [
        gxe.ExpectColumnValuesToNotBeNull(column="tenure"),
        gxe.ExpectColumnValuesToNotBeNull(column="MonthlyCharges"),
        gxe.ExpectColumnValuesToBeBetween(column="tenure", min_value=0, max_value=120),
        gxe.ExpectColumnValuesToBeBetween(column="MonthlyCharges", min_value=0, max_value=200),
        gxe.ExpectColumnValuesToBeBetween(column="TotalCharges", min_value=0),
    ]

    # === DATA CONSISTENCY CHECKS ===
    # Total charges should generally be >= monthly charges (except for very new customers)
    expectations.append(
        gxe.ExpectColumnPairValuesAToBeGreaterThanB(
            column_A="TotalCharges",
            column_B="MonthlyCharges",
            or_equal=True,
            mostly=0.95,  # Allow 5% exceptions for edge cases
        )
    )
    return expectations


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset using Great Expectations.

    This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.

    Returns:
        (success, failed) where failed lists each failing check, e.g.
        "expect_column_values_to_be_between(tenure)".
    """
    print("🔍 Starting data validation with Great Expectations...")

    # TotalCharges is read as text because of blank cells; check it as a number
    df = df.copy()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # In-memory GX context: nothing is written to disk
    context = gx.get_context(mode="ephemeral")
    batch = (
        context.data_sources.add_pandas("telco")
        .add_dataframe_asset("raw_customers")
        .add_batch_definition_whole_dataframe("full_dataframe")
        .get_batch(batch_parameters={"dataframe": df})
    )
    suite = context.suites.add(gx.ExpectationSuite(name="telco_raw_data", expectations=_build_expectations()))

    print(f"   ⚙️  Running {len(suite.expectations)} checks (schema, categories, ranges, consistency)...")
    results = batch.validate(suite)

    # === PROCESS RESULTS ===
    failed_expectations = []
    for r in results.results:
        if not r.success:
            kwargs = r.expectation_config.kwargs
            target = kwargs.get("column") or f"{kwargs.get('column_A')}, {kwargs.get('column_B')}"
            failed_expectations.append(f"{r.expectation_config.type}({target})")

    total_checks = len(results.results)
    passed_checks = total_checks - len(failed_expectations)
    if results.success:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {len(failed_expectations)}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectations}")

    return bool(results.success), failed_expectations
