"""Phase 2: model training and the deployed model's quality (was scripts/test_pipeline_phase2_modeling.py)."""
import json

import pytest
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from tests.conftest import ROOT, TARGET

THRESHOLD = 0.35


@pytest.fixture(scope="module")
def split(features_df):
    # Same split as scripts/run_pipeline.py, so the test set is the one the model never saw
    X = features_df.drop(columns=[TARGET])
    y = features_df[TARGET]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def test_target_is_binary_with_no_nans(features_df):
    assert features_df[TARGET].isna().sum() == 0
    assert set(features_df[TARGET].unique()) <= {0, 1}


def test_quick_xgboost_learns_signal(split):
    X_train, X_test, y_train, y_test = split
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    assert roc_auc_score(y_test, proba) > 0.80
    assert recall_score(y_test, (proba >= THRESHOLD).astype(int)) > 0.70


def test_deployed_model_matches_reported_metrics(split, feature_cols):
    """The model in model/ should still score what model_metrics.json (shown on the frontend) claims."""
    from src.serving.inference import sklearn_model

    with open(ROOT / "model" / "model_metrics.json") as f:
        reported = json.load(f)

    _, X_test, _, y_test = split
    proba = sklearn_model.predict_proba(X_test[feature_cols])[:, 1]
    recall = recall_score(y_test, (proba >= reported["threshold"]).astype(int))
    auc = roc_auc_score(y_test, proba)

    assert recall == pytest.approx(reported["recall"], abs=0.02)
    assert auc == pytest.approx(reported["roc_auc"], abs=0.02)
