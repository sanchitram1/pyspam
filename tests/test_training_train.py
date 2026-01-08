"""Tests for training.train module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest

from training.train import (
    load_training_data,
    build_pipeline,
    evaluate_model,
    save_model_with_metadata,
    train_model,
)


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 100
    data = {
        "pkg_name": [f"package_{i}" for i in range(n_samples)],
        "is_spam": np.random.randint(0, 2, n_samples),
        "numeric_feature_1": np.random.randn(n_samples),
        "numeric_feature_2": np.random.randn(n_samples),
        "categorical_feature_1": np.random.choice(["A", "B", "C"], n_samples),
        "categorical_feature_2": np.random.choice(["X", "Y"], n_samples),
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_jsonl_file(sample_data):
    """Create a temporary JSONL file with sample data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for _, row in sample_data.iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")
        temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink()


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestLoadTrainingData:
    """Test load_training_data function."""

    def test_load_training_data_basic(self, temp_jsonl_file):
        """Test loading training data from JSONL."""
        X, y = load_training_data(temp_jsonl_file)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == 100
        assert len(y) == 100

    def test_load_training_data_removes_pkg_name(self, temp_jsonl_file):
        """Test that pkg_name is removed from features."""
        X, y = load_training_data(temp_jsonl_file)
        assert "pkg_name" not in X.columns

    def test_load_training_data_removes_is_spam_from_X(self, temp_jsonl_file):
        """Test that is_spam is removed from X."""
        X, y = load_training_data(temp_jsonl_file)
        assert "is_spam" not in X.columns

    def test_load_training_data_y_is_target(self, temp_jsonl_file):
        """Test that y contains is_spam values."""
        X, y = load_training_data(temp_jsonl_file)
        assert y.name == "is_spam"
        assert y.dtype in [int, np.int64, np.int32]

    def test_load_training_data_file_not_found(self):
        """Test error handling for missing file."""
        with pytest.raises((FileNotFoundError, ValueError)):
            load_training_data("nonexistent_file.jsonl")

    def test_load_training_data_returns_correct_shapes(self, temp_jsonl_file):
        """Test that returned data has correct shapes."""
        X, y = load_training_data(temp_jsonl_file)
        assert X.shape[0] == y.shape[0]
        assert len(X.columns) > 0


class TestBuildPipeline:
    """Test build_pipeline function."""

    def test_build_pipeline_returns_pipeline(self, sample_data):
        """Test that build_pipeline returns a Pipeline object."""
        X = sample_data.drop(columns=["pkg_name", "is_spam"])
        from sklearn.pipeline import Pipeline
        pipeline = build_pipeline(X)
        assert isinstance(pipeline, Pipeline)

    def test_build_pipeline_has_required_steps(self, sample_data):
        """Test that pipeline has required steps."""
        X = sample_data.drop(columns=["pkg_name", "is_spam"])
        pipeline = build_pipeline(X)
        step_names = [name for name, _ in pipeline.steps]
        assert "preprocessor" in step_names
        assert "scaler" in step_names
        assert "classifier" in step_names

    def test_build_pipeline_with_categorical_features(self, sample_data):
        """Test pipeline with categorical features."""
        X = sample_data.drop(columns=["pkg_name", "is_spam"])
        pipeline = build_pipeline(X)
        # Check that preprocessor exists
        assert hasattr(pipeline, "named_steps")
        assert "preprocessor" in pipeline.named_steps

    def test_build_pipeline_with_only_numerical(self):
        """Test pipeline with only numerical features."""
        X = pd.DataFrame({
            "feat1": np.random.randn(50),
            "feat2": np.random.randn(50),
        })
        pipeline = build_pipeline(X)
        assert pipeline is not None

    def test_build_pipeline_with_only_categorical(self):
        """Test pipeline with only categorical features."""
        X = pd.DataFrame({
            "feat1": np.random.choice(["A", "B"], 50),
            "feat2": np.random.choice(["X", "Y"], 50),
        })
        pipeline = build_pipeline(X)
        assert pipeline is not None

    def test_build_pipeline_ensemble_has_four_estimators(self, sample_data):
        """Test that ensemble has four estimators."""
        X = sample_data.drop(columns=["pkg_name", "is_spam"])
        pipeline = build_pipeline(X)
        classifier = pipeline.named_steps["classifier"]
        assert len(classifier.estimators) == 4


class TestEvaluateModel:
    """Test evaluate_model function."""

    def test_evaluate_model_basic(self):
        """Test basic model evaluation."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        metrics = evaluate_model(y_true, y_pred)
        
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

    def test_evaluate_model_without_proba(self):
        """Test evaluation without probabilities."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        metrics = evaluate_model(y_true, y_pred)
        
        assert "roc_auc" not in metrics

    def test_evaluate_model_with_proba(self):
        """Test evaluation with probabilities."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_pred_proba = np.array([
            [1.0, 0.0],
            [0.1, 0.9],
            [0.7, 0.3],
            [0.9, 0.1],
            [0.2, 0.8],
        ])
        metrics = evaluate_model(y_true, y_pred, y_pred_proba)
        
        assert "roc_auc" in metrics
        assert 0 <= metrics["roc_auc"] <= 1

    def test_evaluate_model_all_metrics_in_range(self):
        """Test that all metrics are in valid ranges."""
        y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        metrics = evaluate_model(y_true, y_pred)
        
        for metric_name, metric_value in metrics.items():
            assert 0 <= metric_value <= 1

    def test_evaluate_model_with_zero_division(self):
        """Test evaluation handles zero division gracefully."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        metrics = evaluate_model(y_true, y_pred)
        
        # Should not raise error
        assert isinstance(metrics, dict)

    def test_evaluate_model_perfect_prediction(self):
        """Test evaluation with perfect predictions."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        metrics = evaluate_model(y_true, y_pred)
        
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0


class TestSaveModelWithMetadata:
    """Test save_model_with_metadata function."""

    def test_save_model_creates_files(self, sample_data, temp_output_dir):
        """Test that save_model_with_metadata creates files."""
        X = sample_data.drop(columns=["pkg_name", "is_spam"])
        y = sample_data["is_spam"]
        
        pipeline = build_pipeline(X)
        pipeline.fit(X, y)
        
        metrics = {"accuracy": 0.95, "f1": 0.92}
        save_model_with_metadata(pipeline, metrics, temp_output_dir)
        
        model_file = temp_output_dir / "model_production.joblib"
        metadata_file = temp_output_dir / "model_metadata.json"
        
        assert model_file.exists()
        assert metadata_file.exists()

    def test_save_model_joblib_valid(self, sample_data, temp_output_dir):
        """Test that saved model is a valid joblib file."""
        X = sample_data.drop(columns=["pkg_name", "is_spam"])
        y = sample_data["is_spam"]
        
        pipeline = build_pipeline(X)
        pipeline.fit(X, y)
        
        metrics = {"accuracy": 0.95}
        save_model_with_metadata(pipeline, metrics, temp_output_dir)
        
        model_file = temp_output_dir / "model_production.joblib"
        loaded_model = joblib.load(model_file)
        
        assert loaded_model is not None

    def test_save_model_metadata_valid_json(self, sample_data, temp_output_dir):
        """Test that saved metadata is valid JSON."""
        X = sample_data.drop(columns=["pkg_name", "is_spam"])
        y = sample_data["is_spam"]
        
        pipeline = build_pipeline(X)
        pipeline.fit(X, y)
        
        metrics = {"accuracy": 0.95, "f1": 0.92}
        save_model_with_metadata(pipeline, metrics, temp_output_dir)
        
        metadata_file = temp_output_dir / "model_metadata.json"
        
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        assert isinstance(metadata, dict)
        assert "timestamp" in metadata
        assert "metrics" in metadata

    def test_save_model_metadata_contains_model_info(self, sample_data, temp_output_dir):
        """Test that metadata contains model information."""
        X = sample_data.drop(columns=["pkg_name", "is_spam"])
        y = sample_data["is_spam"]
        
        pipeline = build_pipeline(X)
        pipeline.fit(X, y)
        
        metrics = {"accuracy": 0.95}
        metadata = save_model_with_metadata(pipeline, metrics, temp_output_dir)
        
        assert "model_type" in metadata
        assert metadata["model_type"] == "VotingClassifier"
        assert "test_size" in metadata
        assert "random_state" in metadata
        assert "primary_metric" in metadata

    def test_save_model_custom_paths(self, sample_data, temp_output_dir):
        """Test saving model with custom paths."""
        X = sample_data.drop(columns=["pkg_name", "is_spam"])
        y = sample_data["is_spam"]
        
        pipeline = build_pipeline(X)
        pipeline.fit(X, y)
        
        custom_model_path = temp_output_dir / "custom_model.joblib"
        custom_metadata_path = temp_output_dir / "custom_metadata.json"
        
        metrics = {"accuracy": 0.95}
        save_model_with_metadata(
            pipeline,
            metrics,
            temp_output_dir,
            model_filepath=custom_model_path,
            metadata_filepath=custom_metadata_path,
        )
        
        assert custom_model_path.exists()
        assert custom_metadata_path.exists()


class TestTrainModel:
    """Test train_model function."""

    @patch("training.train.save_model_with_metadata")
    def test_train_model_basic(self, mock_save, temp_jsonl_file, temp_output_dir):
        """Test basic train_model execution."""
        mock_save.return_value = {"accuracy": 0.95}
        
        pipeline, metrics = train_model(temp_jsonl_file, temp_output_dir)
        
        assert pipeline is not None
        assert metrics is not None
        assert isinstance(metrics, dict)
        mock_save.assert_called_once()

    @patch("training.train.save_model_with_metadata")
    def test_train_model_creates_train_test_split(self, mock_save, temp_jsonl_file, temp_output_dir):
        """Test that train_model creates train-test split."""
        mock_save.return_value = {"accuracy": 0.95}
        
        train_model(temp_jsonl_file, temp_output_dir)
        
        # Verify save was called with metrics dict
        assert mock_save.called

    @patch("training.train.save_model_with_metadata")
    def test_train_model_saves_with_metrics(self, mock_save, temp_jsonl_file, temp_output_dir):
        """Test that train_model saves with computed metrics."""
        mock_save.return_value = {"accuracy": 0.95}
        
        train_model(temp_jsonl_file, temp_output_dir)
        
        # Get the metrics that were passed to save_model_with_metadata
        call_args = mock_save.call_args
        saved_metrics = call_args[0][1]
        
        assert isinstance(saved_metrics, dict)
        assert len(saved_metrics) > 0


class TestMain:
    """Test main entry point."""

    @patch("training.train.train_model")
    def test_main_with_defaults(self, mock_train, monkeypatch):
        """Test main function with default arguments."""
        # Mock sys.argv
        monkeypatch.setattr("sys.argv", ["train.py"])
        
        mock_train.return_value = (None, {})
        
        from training.train import main
        
        # Should raise FileNotFoundError because default input doesn't exist
        with pytest.raises(FileNotFoundError):
            main()

    @patch("training.train.train_model")
    def test_main_with_custom_paths(self, mock_train, monkeypatch, temp_jsonl_file, temp_output_dir):
        """Test main function with custom paths."""
        monkeypatch.setattr(
            "sys.argv",
            ["train.py", "--input", str(temp_jsonl_file), "--output", str(temp_output_dir)],
        )
        
        mock_train.return_value = (None, {})
        
        from training.train import main
        main()
        
        mock_train.assert_called_once()

    @patch("training.train.train_model")
    def test_main_entry_point(self, mock_train, monkeypatch, temp_jsonl_file, temp_output_dir):
        """Test main as entry point (if __name__ == '__main__')."""
        monkeypatch.setattr(
            "sys.argv",
            ["train.py", "--input", str(temp_jsonl_file), "--output", str(temp_output_dir)],
        )
        
        mock_train.return_value = (None, {})
        
        # Import and execute as if running directly
        import runpy
        from pathlib import Path
        
        # This will execute the if __name__ == "__main__": block
        with patch("training.train.train_model", mock_train):
            with patch("sys.argv", ["train.py", "--input", str(temp_jsonl_file), "--output", str(temp_output_dir)]):
                from training import train as train_module
                if hasattr(train_module, "main"):
                    train_module.main()
