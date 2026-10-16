"""Phase 1: load → validate → preprocess → build features (was scripts/test_pipeline_phase1_data_features.py)."""
import pytest

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.utils.validate_data import validate_telco_data
from tests.conftest import TARGET

# validate_data.py uses ge.dataset.PandasDataset, which was removed in Great Expectations 1.x
# (requirements.txt pins 1.5.8). strict=True makes these fail loudly once validation is ported.
ge_broken = pytest.mark.xfail(
    raises=AttributeError, strict=True, reason="validate_data.py uses the pre-1.0 Great Expectations API"
)


def test_raw_data_shape(raw_df):
    assert raw_df.shape == (7043, 21)


def test_load_data_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_data("does/not/exist.csv")


@ge_broken
def test_raw_data_passes_validation(raw_df):
    is_valid, failed = validate_telco_data(raw_df.copy())
    assert is_valid, f"Validation failed: {failed}"


@ge_broken
def test_validation_rejects_bad_data(raw_df):
    bad = raw_df.copy()
    bad.loc[:9, "tenure"] = -5
    is_valid, _ = validate_telco_data(bad)
    assert not is_valid


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
