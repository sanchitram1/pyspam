# -----------------------------------------------------------
# Step 8: Fill Null
# -----------------------------------------------------------
import pandas as pd
from feature_engineering.settings import NUM_FILL_VALUES


def fill_null(df: pd.DataFrame):
    """
    Fill null values in the dataframe with appropriate defaults.

    For numeric columns specified in NUM_FILL_VALUES, fills nulls with
    the configured default values. For categorical columns like
    cat_license_family, fills with "unknown".

    :param df: DataFrame with potentially null values
    :type df: pd.DataFrame
    :return: DataFrame with null values filled
    :rtype: pd.DataFrame
    """
    # Fill numeric columns with configured default values
    for col, fill_value in NUM_FILL_VALUES.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fill_value)

    # Fill categorical license family column
    # Standardize "other_or_unknown" to "unknown" for consistency
    if "cat_license_family" in df.columns:
        df["cat_license_family"] = (
            df["cat_license_family"]
            .fillna("unknown")
            .replace({"other_or_unknown": "unknown"})
        )
    return df
