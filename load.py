import json
from pathlib import Path

import pandas as pd


def load_training_data(filepath=None):
    """
    Load JSONL training data into a pandas DataFrame.

    Args:
        filepath: Path to the JSONL file. Defaults to ~/Downloads/training.json

    Returns:
        pandas.DataFrame with properly assigned columns
    """
    if filepath is None:
        filepath = Path.home() / "Downloads" / "training.json"

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Read JSONL file
    records = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Create DataFrame
    df = pd.DataFrame(records)

    # Clean up text columns that may have embedded newlines
    def clean_value(x):
        # Skip cleaning for complex types (lists, dicts, etc.)
        if isinstance(x, (list, dict)):
            return x
        # Skip NaN/None values
        if pd.isna(x):
            return x
        # Clean strings
        return str(x).replace("\n", " ").replace("\r", "")

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(clean_value)

    return df
