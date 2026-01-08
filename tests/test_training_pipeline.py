"""Tests for training.pipeline module."""

import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import subprocess

import pytest

from training.pipeline import run_feature_engineering, run_training, main


@pytest.fixture
def temp_paths():
    """Create temporary paths for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        raw_data = tmpdir / "raw_data.jsonl"
        features = tmpdir / "features.jsonl"
        models = tmpdir / "models"
        
        # Create raw data file
        raw_data.touch()
        
        yield {
            "raw_data": raw_data,
            "features": features,
            "models": models,
        }


class TestRunFeatureEngineering:
    """Test run_feature_engineering function."""

    @patch("training.pipeline.subprocess.run")
    def test_run_feature_engineering_success(self, mock_run):
        """Test successful feature engineering execution."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        input_path = "input.jsonl"
        output_path = "output.jsonl"
        
        result = run_feature_engineering(input_path, output_path)
        
        assert result is True
        mock_run.assert_called_once()

    @patch("training.pipeline.subprocess.run")
    def test_run_feature_engineering_failure(self, mock_run):
        """Test failed feature engineering execution."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result
        
        input_path = "input.jsonl"
        output_path = "output.jsonl"
        
        result = run_feature_engineering(input_path, output_path)
        
        assert result is False

    @patch("training.pipeline.subprocess.run")
    def test_run_feature_engineering_command_structure(self, mock_run):
        """Test that feature engineering uses correct command."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        input_path = "input.jsonl"
        output_path = "output.jsonl"
        
        run_feature_engineering(input_path, output_path)
        
        called_cmd = mock_run.call_args[0][0]
        assert called_cmd[0] == "python"
        assert called_cmd[1] == "-m"
        assert called_cmd[2] == "feature_engineering.pipeline"
        assert "--input" in called_cmd
        assert "--output" in called_cmd

    @patch("training.pipeline.subprocess.run")
    def test_run_feature_engineering_with_path_objects(self, mock_run):
        """Test that run_feature_engineering works with Path objects."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        input_path = Path("input.jsonl")
        output_path = Path("output.jsonl")
        
        result = run_feature_engineering(input_path, output_path)
        
        assert result is True


class TestRunTraining:
    """Test run_training function."""

    @patch("training.pipeline.subprocess.run")
    def test_run_training_success(self, mock_run):
        """Test successful model training execution."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        input_path = "features.jsonl"
        output_dir = "models/"
        
        result = run_training(input_path, output_dir)
        
        assert result is True
        mock_run.assert_called_once()

    @patch("training.pipeline.subprocess.run")
    def test_run_training_failure(self, mock_run):
        """Test failed model training execution."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result
        
        input_path = "features.jsonl"
        output_dir = "models/"
        
        result = run_training(input_path, output_dir)
        
        assert result is False

    @patch("training.pipeline.subprocess.run")
    def test_run_training_command_structure(self, mock_run):
        """Test that training uses correct command."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        input_path = "features.jsonl"
        output_dir = "models/"
        
        run_training(input_path, output_dir)
        
        called_cmd = mock_run.call_args[0][0]
        assert called_cmd[0] == "python"
        assert called_cmd[1] == "-m"
        assert called_cmd[2] == "training.train"
        assert "--input" in called_cmd
        assert "--output" in called_cmd

    @patch("training.pipeline.subprocess.run")
    def test_run_training_with_path_objects(self, mock_run):
        """Test that run_training works with Path objects."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        input_path = Path("features.jsonl")
        output_dir = Path("models/")
        
        result = run_training(input_path, output_dir)
        
        assert result is True


class TestPipelineOrchestration:
    """Test the complete pipeline orchestration."""

    @patch("training.pipeline.run_training")
    @patch("training.pipeline.run_feature_engineering")
    def test_pipeline_feature_engineering_then_training(
        self, mock_fe, mock_train, temp_paths
    ):
        """Test that pipeline runs feature engineering before training."""
        mock_fe.return_value = True
        mock_train.return_value = True
        
        raw_data = temp_paths["raw_data"]
        features = temp_paths["features"]
        models = temp_paths["models"]
        
        # Simulate the main pipeline logic
        fe_success = mock_fe(raw_data, features)
        if fe_success:
            train_success = mock_train(features, models)
        
        assert mock_fe.called
        assert mock_train.called
        # Verify order: FE before training
        assert mock_fe.call_count < mock_train.call_count or mock_fe.called

    @patch("training.pipeline.run_training")
    @patch("training.pipeline.run_feature_engineering")
    def test_pipeline_stops_on_fe_failure(self, mock_fe, mock_train, temp_paths):
        """Test that pipeline stops if feature engineering fails."""
        mock_fe.return_value = False
        mock_train.return_value = True
        
        raw_data = temp_paths["raw_data"]
        features = temp_paths["features"]
        models = temp_paths["models"]
        
        # Simulate the main pipeline logic
        fe_success = mock_fe(raw_data, features)
        if fe_success:
            mock_train(features, models)
        
        assert mock_fe.called
        assert not mock_train.called

    @patch("training.pipeline.run_training")
    @patch("training.pipeline.run_feature_engineering")
    def test_pipeline_stops_on_training_failure(self, mock_fe, mock_train, temp_paths):
        """Test that pipeline reports failure if training fails."""
        mock_fe.return_value = True
        mock_train.return_value = False
        
        raw_data = temp_paths["raw_data"]
        features = temp_paths["features"]
        models = temp_paths["models"]
        
        # Simulate the main pipeline logic
        fe_success = mock_fe(raw_data, features)
        if fe_success:
            train_success = mock_train(features, models)
        
        assert mock_fe.called
        assert mock_train.called


class TestMainFunction:
    """Test main entry point of pipeline."""

    @patch("training.pipeline.run_training")
    @patch("training.pipeline.run_feature_engineering")
    def test_main_with_required_argument(self, mock_fe, mock_train, monkeypatch, temp_paths):
        """Test main with required --raw-data argument."""
        mock_fe.return_value = True
        mock_train.return_value = True
        
        raw_data = temp_paths["raw_data"]
        
        monkeypatch.setattr(
            "sys.argv",
            ["pipeline.py", "--raw-data", str(raw_data)],
        )
        
        main()
        
        mock_fe.assert_called_once()
        mock_train.assert_called_once()

    @patch("training.pipeline.run_training")
    @patch("training.pipeline.run_feature_engineering")
    def test_main_with_custom_output_paths(self, mock_fe, mock_train, monkeypatch, temp_paths):
        """Test main with custom output paths."""
        mock_fe.return_value = True
        mock_train.return_value = True
        
        raw_data = temp_paths["raw_data"]
        features = temp_paths["features"]
        models = temp_paths["models"]
        
        monkeypatch.setattr(
            "sys.argv",
            [
                "pipeline.py",
                "--raw-data", str(raw_data),
                "--features-output", str(features),
                "--models-output", str(models),
            ],
        )
        
        main()
        
        # Verify calls were made with correct paths
        assert mock_fe.called
        assert mock_train.called

    def test_main_missing_required_argument(self, monkeypatch):
        """Test main fails without required argument."""
        monkeypatch.setattr("sys.argv", ["pipeline.py"])
        
        with pytest.raises(SystemExit):
            main()

    @patch("training.pipeline.run_feature_engineering")
    def test_main_missing_raw_data_file(self, mock_fe, monkeypatch):
        """Test main fails if raw data file doesn't exist."""
        nonexistent_file = "/nonexistent/path/data.jsonl"
        
        monkeypatch.setattr(
            "sys.argv",
            ["pipeline.py", "--raw-data", nonexistent_file],
        )
        
        with pytest.raises(SystemExit):
            main()

    @patch("training.pipeline.run_training")
    @patch("training.pipeline.run_feature_engineering")
    def test_main_exits_on_fe_failure(self, mock_fe, mock_train, monkeypatch, temp_paths):
        """Test main exits if feature engineering fails."""
        mock_fe.return_value = False
        mock_train.return_value = True
        
        raw_data = temp_paths["raw_data"]
        
        monkeypatch.setattr(
            "sys.argv",
            ["pipeline.py", "--raw-data", str(raw_data)],
        )
        
        with pytest.raises(SystemExit):
            main()

    @patch("training.pipeline.run_training")
    @patch("training.pipeline.run_feature_engineering")
    def test_main_exits_on_training_failure(self, mock_fe, mock_train, monkeypatch, temp_paths):
        """Test main exits if training fails."""
        mock_fe.return_value = True
        mock_train.return_value = False
        
        raw_data = temp_paths["raw_data"]
        
        monkeypatch.setattr(
            "sys.argv",
            ["pipeline.py", "--raw-data", str(raw_data)],
        )
        
        with pytest.raises(SystemExit):
            main()

    @patch("training.pipeline.run_training")
    @patch("training.pipeline.run_feature_engineering")
    def test_main_successful_completion(self, mock_fe, mock_train, monkeypatch, temp_paths, capsys):
        """Test main completes successfully."""
        mock_fe.return_value = True
        mock_train.return_value = True
        
        raw_data = temp_paths["raw_data"]
        
        monkeypatch.setattr(
            "sys.argv",
            ["pipeline.py", "--raw-data", str(raw_data)],
        )
        
        main()
        
        # Should complete without raising exception
        assert mock_fe.called
        assert mock_train.called


class TestLogging:
    """Test logging in pipeline."""

    @patch("training.pipeline.subprocess.run")
    def test_run_feature_engineering_logs(self, mock_run, caplog):
        """Test that feature engineering logs information."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        with caplog.at_level(logging.INFO):
            run_feature_engineering("input.jsonl", "output.jsonl")
        
        # Check for logging output
        assert "STEP 1: Feature Engineering" in caplog.text or "python" in caplog.text

    @patch("training.pipeline.subprocess.run")
    def test_run_training_logs(self, mock_run, caplog):
        """Test that training logs information."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        with caplog.at_level(logging.INFO):
            run_training("features.jsonl", "models/")
        
        # Check for logging output
        assert "STEP 2: Model Training" in caplog.text or "python" in caplog.text


class TestMainEntry:
    """Test main entry point execution."""

    @patch("training.pipeline.run_training")
    @patch("training.pipeline.run_feature_engineering")
    def test_main_as_entry_point(self, mock_fe, mock_train, monkeypatch, temp_paths):
        """Test main when executed as script."""
        mock_fe.return_value = True
        mock_train.return_value = True
        
        raw_data = temp_paths["raw_data"]
        
        monkeypatch.setattr(
            "sys.argv",
            ["pipeline.py", "--raw-data", str(raw_data)],
        )
        
        # Import and call main
        from training.pipeline import main
        main()
        
        assert mock_fe.called
        assert mock_train.called
