# -------------------------------------------------------
# Step 7: Remove redundant features
# -------------------------------------------------------
import pandas as pd


def drop_redundant(df: pd.DataFrame):
    """
    Remove redundant columns that are no longer needed after feature engineering.

    Drops raw text fields, intermediate processing columns, and other
    columns that have been replaced by engineered features.

    :param df: DataFrame with columns to drop
    :type df: pd.DataFrame
    :return: DataFrame with redundant columns removed
    :rtype: pd.DataFrame
    """
    columns_to_drop = [
        "latest_description",  # Replaced by dist_embed_to_legit_desc
        "latest_summary",  # Not used in features
        "licenses",  # Replaced by cat_license_family
        "latest_dependencies",  # Replaced by dependency-based features
        "distinct_classifiers",  # Not used in features
        "distinct_keywords",  # Not used in features
        "distinct_maintainers",  # Replaced by maintainer-based features
        "distinct_authors",  # Replaced by maintainer-based features
        "latest_project_urls",  # Replaced by n_latest_project_urls
        "t_last_release",  # Not used in features
        "t_first_release",  # Not used in features
        "t_time_of_day_bucket",  # Replaced by t_time_of_day
        "versions",  # Not used in features
    ]

    # Only drop columns that actually exist in the dataframe
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    df.drop(columns=existing_columns, inplace=True)
    return df
