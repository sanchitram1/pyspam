# ------------------------------------------------------
# Step 6: Time offline features
#   - t_time_of_day (morning, evening, afternoon, night)
# ------------------------------------------------------
import pandas as pd


def bucket_to_time_of_day(bucket):
    """
    Convert time bucket string to standardized time of day category.

    Maps time bucket strings (e.g., "evening_bucket") to standardized
    categories: "morning", "afternoon", "evening", "night", or "unknown".

    Note: Checks are done in order, so if a bucket contains multiple
    keywords, the first match wins (e.g., "evening" before "night").

    :param bucket: Time bucket string (may be NaN)
    :type bucket: str or float
    :return: Standardized time of day category
    :rtype: str
    """
    # Handle NaN/null values
    if pd.isna(bucket):
        return "unknown"

    bucket_str = str(bucket).lower()

    # Check in order: evening, morning, night, afternoon
    # This order ensures "evening" is matched before "night" if both appear
    if "evening" in bucket_str:
        return "evening"
    elif "morning" in bucket_str:
        return "morning"
    elif "night" in bucket_str:
        return "night"
    elif "afternoon" in bucket_str:
        return "afternoon"
    else:
        return "unknown"


def handle_time(df: pd.DataFrame):
    """
    Add time-based features to the dataframe.

    Converts time bucket strings to standardized time of day categories.

    :param df: DataFrame containing 't_time_of_day_bucket' column
    :type df: pd.DataFrame
    :return: DataFrame with added 't_time_of_day' column
    :rtype: pd.DataFrame
    """
    df["t_time_of_day"] = df["t_time_of_day_bucket"].apply(bucket_to_time_of_day)
    return df
