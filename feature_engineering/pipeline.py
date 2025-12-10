#!/usr/bin/env pkgx uv run
"""
Feature engineering pipeline for spam detection.
"""

import argparse
import logging

import pandas as pd

# Import your existing modules
from feature_engineering.dependency_offline import handle_dependency
from feature_engineering.description_offline import handle_description
from feature_engineering.handle_null import fill_null
from feature_engineering.legit_mask import create_legit_mask
from feature_engineering.load_data import load_json
from feature_engineering.maintainer_offline import handle_maintainers
from feature_engineering.name_based_offline import add_name_based
from feature_engineering.remove_redundant import drop_redundant
from feature_engineering.save_json import save_json
from feature_engineering.settings import INPUT_PATH, OUTPUT_PATH
from feature_engineering.time_offline import handle_time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    The CORE logic. Accepts a DataFrame (1 row or 10k rows),
    runs all transformations, and returns the enriched DataFrame.
    """
    logger.debug(f"Processing DataFrame with shape: {df.shape}")

    # Step 1: Create legit mask
    if "is_spam" in df.columns:
        legit_mask = create_legit_mask(df)
    else:
        raise ValueError("is_spam column not found in input data")

    # Step 2: Name-based features
    # (We skip "Step 1" because we assume data is already loaded into df)
    logger.debug("Computing name-based features")
    df = add_name_based(df, legit_mask_np=legit_mask)

    # Step 3: Description-based features
    logger.debug("Computing description-based features")
    df = handle_description(df, legit_mask)

    # Step 4: Maintainer-based features
    logger.debug("Computing maintainer-based features")
    df = handle_maintainers(df)

    # Step 5: Dependency-based features
    logger.debug("Computing dependency-based features")
    df = handle_dependency(df)

    # Step 6: Time-based features
    logger.debug("Computing time-based features")
    df = handle_time(df)

    # Step 7: Remove redundant columns
    logger.debug("Removing redundant columns")
    df = drop_redundant(df)

    # Step 8: Fill null values
    logger.debug("Filling null values")
    df = fill_null(df)

    return df


def transform_single_package(package_data: dict) -> dict:
    """
    API ADAPTER: Takes a single JSON object (dict), processes it,
    and returns the transformed features as a dict.
    """
    logger.debug(f"Processing single package: {package_data.get('name', 'unknown')}")

    # 1. Convert Dict -> DataFrame (1 row)
    # We wrap it in a list [] so pandas understands it's a row, not columns
    df = pd.DataFrame([package_data])
    df["is_spam"] = 0  # Dummy value for processing

    # 2. Run the Core Logic
    processed_df = process_dataframe(df)

    # 3. Convert DataFrame -> Dict
    # 'records' gives us [{'col': val, ...}]
    result_dict = processed_df.to_dict(orient="records")[0]

    return result_dict


def run_pipeline_file(input_path: str, output_path: str):
    """
    The CLI/Batch wrapper. Handles loading from disk and saving to disk.
    """
    logger.debug("Starting batch file pipeline")
    logger.debug(f"Input: {input_path}, Output: {output_path}")

    # Step 1: Load (IO)
    df = load_json(input_path)

    # Run the core logic
    processed_df = process_dataframe(df)

    # Step 9: Save (IO)
    save_json(processed_df, output_path)
    logger.debug("Batch pipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pyspam Feature Engineering Pipeline")
    parser.add_argument("--input", default=INPUT_PATH, help="Path to input JSONL file")
    parser.add_argument(
        "--output", default=OUTPUT_PATH, help="Path to output JSONL file"
    )
    parser.add_argument(
        "--single", action="store_true", help="Process a single package"
    )

    args = parser.parse_args()

    if args.single:
        transform_single_package(args.input)
    else:
        run_pipeline_file(args.input, args.output)
