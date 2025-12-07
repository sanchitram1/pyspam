# -------------------------------------------------------
# Step 7: Remove redundant features
# -------------------------------------------------------
import pandas as pd


def drop_redundant(df: pd.DataFrame):
    df.drop(
        columns=[
            "latest_description",
            "latest_summary",
            "licenses",
            "latest_dependencies",
            "distinct_classifiers",
            "distinct_keywords",
            "distinct_maintainers",
            "distinct_authors",
            "latest_project_urls",
            "t_last_release",
            "t_first_release",
            "t_time_of_day_bucket",
            "versions",
        ],
        inplace=True,
    )
    return df
