# -------------------------------------------------------
# Step 9: Save enriched dataset
# -------------------------------------------------------
import pandas as pd


def save_json(df: pd.DataFrame, OUTPUT_PATH, lines=True):
    df.to_json(OUTPUT_PATH, orient="records", lines=lines)
    print(f"Saved enriched features to {OUTPUT_PATH}")
