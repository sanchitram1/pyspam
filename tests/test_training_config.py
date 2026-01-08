"""Tests for training.config module."""

import pytest
from pathlib import Path

from training.config import (
    PROJECT_ROOT,
    DATA_DIR,
    MODELS_DIR,
    FEATURE_ENGINEERED_DATA,
    MODEL_FILEPATH,
    MODEL_METADATA_FILEPATH,
    PRIMARY_METRIC,
    EVAL_METRICS,
    HIGHER_IS_BETTER,
    TEST_SIZE,
    RANDOM_STATE,
    STRATIFY,
    ENSEMBLE_CONFIG,
)


class TestConfigPaths:
    """Test configuration paths."""

    def test_project_root_exists(self):
        """Test that PROJECT_ROOT is properly defined."""
        assert PROJECT_ROOT is not None
        assert isinstance(PROJECT_ROOT, Path)

    def test_data_dir_exists(self):
        """Test that DATA_DIR exists or can be created."""
        assert isinstance(DATA_DIR, Path)
        assert DATA_DIR.parent.exists()

    def test_models_dir_created(self):
        """Test that MODELS_DIR is created."""
        assert isinstance(MODELS_DIR, Path)
        assert MODELS_DIR.exists()
        assert MODELS_DIR.is_dir()

    def test_feature_engineered_data_path(self):
        """Test FEATURE_ENGINEERED_DATA path."""
        assert isinstance(FEATURE_ENGINEERED_DATA, Path)
        assert str(FEATURE_ENGINEERED_DATA).endswith("features_engineered.jsonl")

    def test_model_filepath(self):
        """Test MODEL_FILEPATH path."""
        assert isinstance(MODEL_FILEPATH, Path)
        assert str(MODEL_FILEPATH).endswith("model_production.joblib")

    def test_metadata_filepath(self):
        """Test MODEL_METADATA_FILEPATH path."""
        assert isinstance(MODEL_METADATA_FILEPATH, Path)
        assert str(MODEL_METADATA_FILEPATH).endswith("model_metadata.json")


class TestModelMetrics:
    """Test model evaluation metrics configuration."""

    def test_primary_metric_defined(self):
        """Test that PRIMARY_METRIC is defined."""
        assert PRIMARY_METRIC == "roc_auc"
        assert isinstance(PRIMARY_METRIC, str)

    def test_eval_metrics_not_empty(self):
        """Test that EVAL_METRICS is not empty."""
        assert len(EVAL_METRICS) > 0
        assert isinstance(EVAL_METRICS, list)

    def test_eval_metrics_contains_primary(self):
        """Test that EVAL_METRICS contains PRIMARY_METRIC."""
        assert PRIMARY_METRIC in EVAL_METRICS

    def test_all_eval_metrics_are_strings(self):
        """Test that all metrics are strings."""
        for metric in EVAL_METRICS:
            assert isinstance(metric, str)

    def test_expected_metrics_present(self):
        """Test that expected metrics are present."""
        expected_metrics = ["roc_auc", "accuracy", "precision", "recall", "f1"]
        for metric in expected_metrics:
            assert metric in EVAL_METRICS

    def test_higher_is_better_mapping(self):
        """Test HIGHER_IS_BETTER mapping."""
        assert isinstance(HIGHER_IS_BETTER, dict)
        assert len(HIGHER_IS_BETTER) > 0

    def test_all_metrics_have_higher_is_better(self):
        """Test that all metrics have a HIGHER_IS_BETTER entry."""
        for metric in EVAL_METRICS:
            assert metric in HIGHER_IS_BETTER
            assert isinstance(HIGHER_IS_BETTER[metric], bool)

    def test_all_metrics_are_higher_is_better(self):
        """Test that all metrics are marked as higher-is-better."""
        for metric, is_higher in HIGHER_IS_BETTER.items():
            assert is_higher is True


class TestTrainTestSplit:
    """Test train-test split configuration."""

    def test_test_size_valid(self):
        """Test that TEST_SIZE is valid."""
        assert isinstance(TEST_SIZE, (int, float))
        assert 0 < TEST_SIZE < 1
        assert TEST_SIZE == 0.2

    def test_random_state_defined(self):
        """Test that RANDOM_STATE is defined."""
        assert isinstance(RANDOM_STATE, int)
        assert RANDOM_STATE == 42

    def test_stratify_enabled(self):
        """Test that STRATIFY is enabled."""
        assert isinstance(STRATIFY, bool)
        assert STRATIFY is True


class TestEnsembleConfig:
    """Test ensemble model configuration."""

    def test_ensemble_config_is_dict(self):
        """Test that ENSEMBLE_CONFIG is a dictionary."""
        assert isinstance(ENSEMBLE_CONFIG, dict)

    def test_ensemble_voting_strategy(self):
        """Test that voting strategy is soft."""
        assert "voting" in ENSEMBLE_CONFIG
        assert ENSEMBLE_CONFIG["voting"] == "soft"

    def test_ensemble_has_estimators(self):
        """Test that ensemble has estimators."""
        assert "estimators" in ENSEMBLE_CONFIG
        assert isinstance(ENSEMBLE_CONFIG["estimators"], list)
        assert len(ENSEMBLE_CONFIG["estimators"]) > 0

    def test_estimator_structure(self):
        """Test that estimators have required fields."""
        required_fields = ["name", "class", "params"]
        for estimator in ENSEMBLE_CONFIG["estimators"]:
            assert isinstance(estimator, dict)
            for field in required_fields:
                assert field in estimator

    def test_estimator_names_unique(self):
        """Test that estimator names are unique."""
        names = [e["name"] for e in ENSEMBLE_CONFIG["estimators"]]
        assert len(names) == len(set(names))

    def test_expected_estimators_present(self):
        """Test that expected estimators are present."""
        expected_names = ["decision_tree", "logistic_regression", "random_forest", "svm"]
        actual_names = [e["name"] for e in ENSEMBLE_CONFIG["estimators"]]
        for expected in expected_names:
            assert expected in actual_names

    def test_estimator_params_are_dicts(self):
        """Test that estimator params are dictionaries."""
        for estimator in ENSEMBLE_CONFIG["estimators"]:
            assert isinstance(estimator["params"], dict)

    def test_estimator_class_strings(self):
        """Test that estimator class paths are strings."""
        for estimator in ENSEMBLE_CONFIG["estimators"]:
            assert isinstance(estimator["class"], str)
            assert "sklearn" in estimator["class"]

    def test_random_forest_estimators(self):
        """Test random forest has correct number of estimators."""
        rf = next(e for e in ENSEMBLE_CONFIG["estimators"] if e["name"] == "random_forest")
        assert rf["params"]["n_estimators"] == 200

    def test_svm_kernel_type(self):
        """Test SVM uses linear kernel."""
        svm = next(e for e in ENSEMBLE_CONFIG["estimators"] if e["name"] == "svm")
        assert svm["params"]["kernel"] == "linear"
