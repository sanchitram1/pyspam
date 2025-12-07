# -----------------------------------------------------------
# Step 8: Fill Null
# -----------------------------------------------------------
from settings import NUM_FILL_VALUES
import pandas as pd


def fill_null(df: pd.DataFrame):
    for col, fill_value in NUM_FILL_VALUES.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fill_value)

    # Categorical: cat_license_family
    if "cat_license_family" in df.columns:
        df["cat_license_family"] = (
            df["cat_license_family"]
            .fillna("unknown")
            .replace({"other_or_unknown": "unknown"})
        )
    return df
