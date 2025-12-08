"""Pytest to load and test joblib models with data from jsonl file."""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def test_data():
    """Load test data once per module."""
    jsonl_file = Path("tests/test_raw.jsonl")
    assert jsonl_file.exists(), f"{jsonl_file} not found"

    with open(jsonl_file) as f:
        first_row = json.loads(f.readline())

    df = pd.DataFrame([first_row])
    drop_cols = ["pkg_name", "is_spam"]
    X_raw = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    models_path = Path("models")
    assert models_path.exists(), f"{models_path} not found"

    fitted_preprocessor = models_path / "fitted_preprocessor.joblib"
    assert fitted_preprocessor.exists(), f"{fitted_preprocessor} not found"

    fitted_scaler = models_path / "fitted_scaler.joblib"
    assert fitted_scaler.exists(), f"{fitted_scaler} not found"

    preprocessor = joblib.load(fitted_preprocessor)
    X_preprocessed = preprocessor.transform(X_raw)
    if sparse.issparse(X_preprocessed):
        X_preprocessed = X_preprocessed.toarray()

    scaler = joblib.load(fitted_scaler)
    X_scaled = scaler.transform(X_preprocessed)

    # Add constant as LAST column (statsmodels convention)
    X_with_constant = np.column_stack([X_scaled, np.ones((X_scaled.shape[0], 1))])

    return X_with_constant


@pytest.fixture(scope="module")
def dtc_spam_classifier():
    """Load decision tree classifier path."""
    path = Path("models/dtc_spam_classifier.joblib")
    assert path.exists(), f"{path} not found"
    return path


@pytest.fixture(scope="module")
def log_reg_spam_classifier():
    """Load logistic regression classifier path."""
    path = Path("models/log_reg_spam_classifier.joblib")
    assert path.exists(), f"{path} not found"
    return path


@pytest.fixture(scope="module")
def rf_spam_classifier():
    """Load random forest classifier path."""
    path = Path("models/rf_spam_classifier.joblib")
    assert path.exists(), f"{path} not found"
    return path


@pytest.fixture(scope="module")
def svm_linear_spam_classifier():
    """Load SVM linear classifier path."""
    path = Path("models/svm_linear_spam_classifier.joblib")
    assert path.exists(), f"{path} not found"
    return path


def test_model_loading(
    test_data,
    dtc_spam_classifier,
    log_reg_spam_classifier,
    rf_spam_classifier,
    svm_linear_spam_classifier,
):
    """Test all classifier models load and predict correctly."""
    X = test_data

    for model_name in [
        dtc_spam_classifier,
        log_reg_spam_classifier,
        rf_spam_classifier,
        svm_linear_spam_classifier,
    ]:
        model_path = Path(model_name)
        if not model_path.exists():
            pytest.skip(f"{model_name} not found")

        model = joblib.load(model_path)

        # Verify feature count
        if hasattr(model, "n_features_in_"):
            assert model.n_features_in_ == X.shape[1], (
                f"{model_name}: expected {model.n_features_in_} features, got {X.shape[1]}"
            )

        # Test prediction
        prediction = model.predict(X)
        assert prediction.shape == (1,), (
            f"{model_name}: prediction shape {prediction.shape}"
        )

        # Test proba if available
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            assert proba.shape == (1, 2), f"{model_name}: proba shape {proba.shape}"
            assert proba.shape == (1, 2), f"{model_name}: proba shape {proba.shape}"
