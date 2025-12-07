import ast

import numpy as np
import pandas as pd


# -------------------------------------------------------
# Utility: safe parsing of list-like fields from CSV/JSON
# -------------------------------------------------------
def parse_list_column(val):
    """
    Handle cases where a column is:
    - already a list (when loading from JSON)
    - a string representation of a Python list
    - NULL/NaN
    """
    if isinstance(val, list):
        return val
    if pd.isna(val):
        return []
    text = str(val)
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    except Exception:
        # fallback: treat as a single string
        return [text]


# -------------------------------------------------------
# Step 1: Load data
# -------------------------------------------------------


def load_json(INPUT_PATH: str, lines=True) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load raw JSONL dataset and enforce type consistency.

    Processes the dataset to ensure:
    - List-like columns are properly parsed as lists
    - Boolean columns are converted to numeric (0/1)
    - is_spam column is numeric and used to create legit_mask

    :param INPUT_PATH: Path to the JSONL file to load
    :type INPUT_PATH: str
    :param lines: Whether the JSON file is in JSONL format (one JSON object per line)
    :type lines: bool
    :return: Tuple of (DataFrame with processed data, numpy array of boolean legit_mask)
    :rtype: tuple[pd.DataFrame, np.ndarray]
    """
    df = pd.read_json(INPUT_PATH, lines=lines)

    # If some of these columns are stored as strings, we may want to ensure types:
    list_like_cols = [
        "distinct_authors",
        "distinct_maintainers",
        "distinct_keywords",
        "distinct_classifiers",
        "latest_project_urls",
        "latest_dependencies",
    ]
    for col in list_like_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_list_column)

    # Ensure numeric booleans are numeric (BigQuery often exports as 0/1 integers or 'true'/'false')
    bool_like_cols = [
        "has_homepage",
        "has_repo_url",
        "has_issue_tracker",
        "has_docs_url",
        "has_license",
        "has_prog_lang_classifier",
        "has_typing_classifier",
        "has_extras",
        "has_missing_author",
        "has_disposable_email",
    ]
    for col in bool_like_cols:
        if col in df.columns:
            # convert 'true'/'false' strings to 1/0, etc.
            df[col] = (
                df[col]
                .astype(str)
                .str.lower()
                .map({"true": 1, "false": 0})
                .fillna(df[col])
                .astype(float)
            )

    # ---- Ensure is_spam is numeric and build legit mask ----
    if "is_spam" in df.columns:
        df["is_spam"] = pd.to_numeric(df["is_spam"], errors="coerce")
        legit_mask = df["is_spam"] == 0
    else:
        legit_mask = pd.Series([True] * len(df))

    return df, legit_mask.to_numpy()
