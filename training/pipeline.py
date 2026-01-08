#!/usr/bin/env python
"""
Orchestration script for the complete training pipeline.

Coordinates:
1. Feature engineering (data → features)
2. Model training (features → model)

Usage:
    python -m training.pipeline --raw-data /path/to/raw.jsonl
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from training.config import FEATURE_ENGINEERED_DATA, MODELS_DIR

logger = logging.getLogger(__name__)


def run_feature_engineering(input_path: str | Path, output_path: str | Path) -> bool:
    """
    Execute feature engineering pipeline.

    :param input_path: Path to raw data JSONL
    :param output_path: Path to save feature-engineered data
    :return: True if successful, False otherwise
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Feature Engineering")
    logger.info("=" * 60)

    cmd = [
        "python",
        "-m",
        "feature_engineering.pipeline",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        logger.error("Feature engineering pipeline failed")
        return False

    logger.info(f"Feature engineering complete: {output_path}")
    return True


def run_training(input_path: str | Path, output_dir: str | Path) -> bool:
    """
    Execute model training pipeline.

    :param input_path: Path to feature-engineered data
    :param output_dir: Directory to save model and metadata
    :return: True if successful, False otherwise
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Model Training")
    logger.info("=" * 60)

    cmd = [
        "python",
        "-m",
        "training.train",
        "--input",
        str(input_path),
        "--output",
        str(output_dir),
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        logger.error("Model training pipeline failed")
        return False

    logger.info(f"Model training complete: {output_dir}")
    return True


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Full training pipeline: feature engineering + model training"
    )
    parser.add_argument(
        "--raw-data",
        type=str,
        required=True,
        help="Path to raw data JSONL file",
    )
    parser.add_argument(
        "--features-output",
        type=str,
        default=str(FEATURE_ENGINEERED_DATA),
        help=f"Path to save feature-engineered data (default: {FEATURE_ENGINEERED_DATA})",
    )
    parser.add_argument(
        "--models-output",
        type=str,
        default=str(MODELS_DIR),
        help=f"Directory to save models (default: {MODELS_DIR})",
    )

    args = parser.parse_args()

    raw_data_path = Path(args.raw_data)
    features_path = Path(args.features_output)
    models_path = Path(args.models_output)

    if not raw_data_path.exists():
        logger.error(f"Raw data file not found: {raw_data_path}")
        sys.exit(1)

    logger.info("Starting full training pipeline")
    logger.info(f"Raw data: {raw_data_path}")
    logger.info(f"Features output: {features_path}")
    logger.info(f"Models output: {models_path}")

    # Step 1: Feature engineering
    if not run_feature_engineering(raw_data_path, features_path):
        sys.exit(1)

    # Step 2: Training
    if not run_training(features_path, models_path):
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Full training pipeline completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
