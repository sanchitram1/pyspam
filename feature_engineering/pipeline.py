#!/usr/bin/env pkgx uv run
"""
Feature engineering pipeline for spam detection.

This pipeline processes raw package data through multiple stages:
1. Load and type normalization
2. Name-based features (typosquatting detection)
3. Description-based features
4. Maintainer-based features
5. Dependency-based features
6. Time-based features
7. Remove redundant columns
8. Fill null values
9. Save processed data
"""

import logging

from dependency_offline import handle_dependency
from description_offline import handle_description
from handle_null import fill_null
from load_data import load_json
from maintainer_offline import handle_maintainers
from name_based_offline import add_name_based
from remove_redundant import drop_redundant
from save_json import save_json
from settings import INPUT_PATH, OUTPUT_PATH
from time_offline import handle_time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    """
    Execute the complete feature engineering pipeline.

    Processes raw package data through all feature engineering steps
    and saves the enriched dataset.

    :param input_path: Path to input JSONL file
    :type input_path: str
    :param output_path: Path to output JSONL file
    :type output_path: str
    """
    logger.info("Starting feature engineering pipeline")
    logger.info(f"Input: {input_path}, Output: {output_path}")

    # Step 1: Load and normalize data types
    logger.info("Step 1: Loading and normalizing data")
    df, legit_mask = load_json(input_path)
    logger.info(f"Loaded {len(df)} packages")

    # Step 2: Add name-based features (typosquatting detection)
    logger.info("Step 2: Computing name-based features")
    df = add_name_based(df, legit_mask_np=legit_mask)

    # Step 3: Add description-based features
    logger.info("Step 3: Computing description-based features")
    df = handle_description(df, legit_mask)

    # Step 4: Add maintainer-based features
    logger.info("Step 4: Computing maintainer-based features")
    df = handle_maintainers(df)

    # Step 5: Add dependency-based features
    logger.info("Step 5: Computing dependency-based features")
    df = handle_dependency(df)

    # Step 6: Add time-based features
    logger.info("Step 6: Computing time-based features")
    df = handle_time(df)

    # Step 7: Remove redundant columns
    logger.info("Step 7: Removing redundant columns")
    df = drop_redundant(df)

    # Step 8: Fill null values
    logger.info("Step 8: Filling null values")
    df = fill_null(df)

    # Step 9: Save processed data
    logger.info("Step 9: Saving processed data")
    save_json(df, output_path)
    logger.info(f"Pipeline completed successfully. Output shape: {df.shape}")


if __name__ == "__main__":
    file_path = "data/bq-results-20251207-053959-1765086112714.json"
    main(input_path=file_path)
