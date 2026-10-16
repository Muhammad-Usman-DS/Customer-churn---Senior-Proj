"""Phase 1: load → validate → preprocess → build features (was scripts/test_pipeline_phase1_data_features.py)."""
import pytest

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.utils.validate_data import validate_telco_data
from tests.conftest import TARGET


def test_raw_data_shape(raw_df):
    assert raw_df.shape == (7043, 21)


def test_load_data_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_data("does/not/exist.csv")


def test_raw_data_passes_validation(raw_df):
    is_valid, failed = validate_telco_data(raw_df.copy())
    assert is_valid, f"Validation failed: {failed}"


def _negative_tenure(df):
    df.loc[:9, "tenure"] = -5


def _unknown_contract(df):
    df.loc[df["Contract"] == "Two year", "Contract"] = "Two Yr"


def _duplicate_id(df):
    df.loc[5, "customerID"] = df.loc[6, "customerID"]


def _missing_gender(df):
    df.loc[0, "gender"] = None


@pytest.mark.parametrize(
    "corrupt, expected_failure",
    [
        (_negative_tenure, "expect_column_values_to_be_between(tenure)"),
        (_unknown_contract, "expect_column_values_to_be_in_set(Contract)"),
        (_duplicate_id, "expect_column_values_to_be_unique(customerID)"),
        (_missing_gender, "expect_column_values_to_not_be_null(gender)"),
    ],
)
def test_validation_rejects_bad_data(raw_df, corrupt, expected_failure):
    bad = raw_df.copy()
    corrupt(bad)
    is_valid, failed = validate_telco_data(bad)
    assert not is_valid
    assert expected_failure in failed


def test_preprocess_cleans_data(raw_df):
    df = preprocess_data(raw_df.copy(), target_col=TARGET)
    assert "customerID" not in df.columns
    assert set(df[TARGET].unique()) == {0, 1}
    # TotalCharges has blank strings in the raw CSV; they must become numeric with no NaNs
    assert df["TotalCharges"].dtype.kind == "f"
    assert df["TotalCharges"].isna().sum() == 0


def test_features_match_saved_schema(features_df, feature_cols):
    # The model was trained on exactly these columns; any drift breaks serving
    assert sorted(features_df.drop(columns=[TARGET]).columns) == sorted(feature_cols)


def test_features_are_numeric_and_complete(features_df):
    assert features_df.isna().sum().sum() == 0
    non_numeric = [c for c in features_df.columns if features_df[c].dtype.kind not in "iuf"]
    assert non_numeric == []


def test_binary_and_dummy_features_are_0_1(features_df):
    continuous = {"tenure", "MonthlyCharges", "TotalCharges"}
    for c in features_df.columns:
        if c not in continuous:
            assert set(features_df[c].unique()) <= {0, 1}, f"{c} is not 0/1"


def test_churn_rate_is_plausible(features_df):
    assert 0.25 < features_df[TARGET].mean() < 0.29
