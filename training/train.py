#!/usr/bin/env python
"""
Executable training pipeline for spam detection model.

Usage:
    python -m training.train --input /path/to/features.jsonl --output /path/to/models/
    python -m training.train  # uses config defaults
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from training.config import (
    ENSEMBLE_CONFIG,
    EVAL_METRICS,
    FEATURE_ENGINEERED_DATA,
    MODEL_FILEPATH,
    MODEL_METADATA_FILEPATH,
    PRIMARY_METRIC,
    RANDOM_STATE,
    TEST_SIZE,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_training_data(input_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load feature-engineered data and prepare X, y.

    :param input_path: Path to JSONL file with features
    :return: Tuple of (X, y) DataFrames
    """
    logger.info(f"Loading training data from {input_path}")
    df = pd.read_json(input_path, lines=True)
    logger.info(f"Loaded {len(df)} samples with {df.shape[1]} columns")

    # Drop non-predictive columns
    drop_cols = ["pkg_name", "is_spam"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y = df["is_spam"].astype(int)

    logger.info(f"Features: {X.shape[1]}, Target distribution: {y.value_counts().to_dict()}")
    return X, y


def build_pipeline(X_train: pd.DataFrame) -> Pipeline:
    """
    Build sklearn Pipeline with preprocessing and ensemble classifier.

    :param X_train: Training features for identifying categorical columns
    :return: Fitted sklearn Pipeline
    """
    logger.info("Building preprocessing and model pipeline")

    # Identify feature types
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    logger.info(
        f"Categorical features: {len(categorical_cols)}, Numerical features: {len(numerical_cols)}"
    )

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                categorical_cols,
            ),
        ],
        remainder="passthrough",
    )

    # Ensemble voter
    voter = VotingClassifier(
        estimators=[
            ("decision_tree", DecisionTreeClassifier(random_state=RANDOM_STATE)),
            ("log_reg", LogisticRegression(max_iter=1000)),
            ("random_forest", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
            ("svm", SVC(kernel="linear", probability=True, random_state=RANDOM_STATE)),
        ],
        voting="soft",
    )

    # Full pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("scaler", StandardScaler()),
            ("classifier", voter),
        ]
    )

    return pipeline


def evaluate_model(
    y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray = None
) -> dict:
    """
    Compute evaluation metrics.

    :param y_true: Ground truth labels
    :param y_pred: Predicted labels
    :param y_pred_proba: Predicted probabilities (for ROC-AUC)
    :return: Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_pred_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])

    return metrics


def save_model_with_metadata(
    pipeline: Pipeline,
    metrics: dict,
    output_dir: str | Path,
    model_filepath: str | Path = None,
    metadata_filepath: str | Path = None,
):
    """
    Save trained model and metadata.

    :param pipeline: Trained sklearn Pipeline
    :param metrics: Evaluation metrics dictionary
    :param output_dir: Directory to save files
    :param model_filepath: Override default model filepath
    :param metadata_filepath: Override default metadata filepath
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(model_filepath) if model_filepath else output_dir / "model_production.joblib"
    metadata_path = (
        Path(metadata_filepath) if metadata_filepath else output_dir / "model_metadata.json"
    )

    # Save model
    joblib.dump(pipeline, model_path)
    logger.info(f"Model saved to {model_path}")

    # Build metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "model_type": "VotingClassifier",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "primary_metric": PRIMARY_METRIC,
    }

    # Save metadata
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")

    return metadata


def train_model(input_path: str | Path, output_dir: str | Path):
    """
    Main training pipeline.

    :param input_path: Path to feature-engineered JSONL data
    :param output_dir: Directory to save model and metadata
    """
    logger.info("=" * 60)
    logger.info("Starting training pipeline")
    logger.info("=" * 60)

    # Load data
    X, y = load_training_data(input_path)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # Build and train
    pipeline = build_pipeline(X_train)
    logger.info("Training pipeline...")
    pipeline.fit(X_train, y_train)
    logger.info("Training complete")

    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)

    metrics = evaluate_model(y_test, y_pred, y_pred_proba)

    logger.info("\n" + "=" * 60)
    logger.info("Test Set Metrics:")
    logger.info("=" * 60)
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")
    logger.info("=" * 60)

    # Print detailed classification report
    logger.info("\nDetailed Classification Report:")
    logger.info(classification_report(y_test, y_pred))

    # Save
    save_model_with_metadata(pipeline, metrics, output_dir)

    logger.info("=" * 60)
    logger.info("Training pipeline completed successfully")
    logger.info("=" * 60)

    return pipeline, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train spam detection model with feature-engineered data"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(FEATURE_ENGINEERED_DATA),
        help=f"Path to feature-engineered JSONL file (default: {FEATURE_ENGINEERED_DATA})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(FEATURE_ENGINEERED_DATA.parent),
        help=f"Directory to save model and metadata (default: {FEATURE_ENGINEERED_DATA.parent})",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(f"Input file not found: {input_path}")

    train_model(input_path, output_dir)


if __name__ == "__main__":
    main()
