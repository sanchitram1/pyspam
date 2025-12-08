import pandas as pd
from feature_engineering.settings import LOW_DOWNLOAD_THRESHOLD_30D


# -------------------------------------------------------
# Step 4: D.* maintainer-based offline features
#   - n_pkgs_by_maintainers_30d
#   - n_low_download_pkgs_by_maintainers
# -------------------------------------------------------
def collect_maintainers(row):
    """
    Collect and deduplicate all maintainer emails from authors and maintainers.

    Combines distinct_authors and distinct_maintainers into a single list,
    removing duplicates.

    :param row: DataFrame row containing distinct_authors and distinct_maintainers
    :type row: pd.Series
    :return: List of unique maintainer emails
    :rtype: list
    """
    # Ensure both are lists (they should be from load_data parsing)
    authors = (
        row["distinct_authors"] if isinstance(row["distinct_authors"], list) else []
    )
    maintainers = (
        row["distinct_maintainers"]
        if isinstance(row["distinct_maintainers"], list)
        else []
    )
    # Combine and deduplicate
    return list(set(authors + maintainers))


def handle_maintainers(df: pd.DataFrame):
    """
    Add maintainer-based features to the dataframe.

    Computes features based on maintainer behavior:
    1. n_pkgs_by_maintainers_30d: Sum of packages by maintainers released in last 30 days
    2. n_low_download_pkgs_by_maintainers: Sum of low-download packages by maintainers
    3. n_latest_project_urls: Count of project URLs

    :param df: DataFrame containing maintainer and package metadata
    :type df: pd.DataFrame
    :return: DataFrame with added maintainer-based features
    :rtype: pd.DataFrame
    """
    # Extract relevant columns for maintainer analysis
    maintainers_df = df[
        [
            "pkg_name",
            "distinct_authors",
            "distinct_maintainers",
            "t_age_last_release_days",
            "n_downloads_30d",
        ]
    ].copy()

    # Combine authors and maintainers into unified list
    maintainers_df["all_maintainers"] = maintainers_df.apply(
        collect_maintainers, axis=1
    )

    # Explode maintainers: one row per package-maintainer pair
    maintains_exploded = maintainers_df.explode("all_maintainers")
    maintains_exploded = maintains_exploded.rename(
        columns={"all_maintainers": "maintainer_email"}
    )

    # Filter out empty maintainer emails
    maintains_exploded = maintains_exploded[
        maintains_exploded["maintainer_email"] != ""
    ]

    # Feature 1: Count packages by maintainers released in last 30 days
    recent_mask = maintains_exploded["t_age_last_release_days"] <= 30
    maintainer_recent_pkg_counts = (
        maintains_exploded[recent_mask]
        .groupby("maintainer_email")["pkg_name"]
        .nunique()
        .rename("maintainer_recent_pkg_count")
        .reset_index()
    )

    # Feature 2: Count low-download packages by maintainers
    low_dl_mask = (
        maintains_exploded["n_downloads_30d"].fillna(0) < LOW_DOWNLOAD_THRESHOLD_30D
    )
    maintainer_lowdl_pkg_counts = (
        maintains_exploded[low_dl_mask]
        .groupby("maintainer_email")["pkg_name"]
        .nunique()
        .rename("maintainer_low_download_pkg_count")
        .reset_index()
    )

    # Join maintainer statistics back to exploded table
    maintains_exploded = maintains_exploded.merge(
        maintainer_recent_pkg_counts, on="maintainer_email", how="left"
    ).merge(maintainer_lowdl_pkg_counts, on="maintainer_email", how="left")

    # Fill NaN with 0 for maintainers with no recent/low-download packages
    maintains_exploded["maintainer_recent_pkg_count"] = maintains_exploded[
        "maintainer_recent_pkg_count"
    ].fillna(0)
    maintains_exploded["maintainer_low_download_pkg_count"] = maintains_exploded[
        "maintainer_low_download_pkg_count"
    ].fillna(0)

    # Aggregate per package: sum counts across all maintainers
    # (A package can have multiple maintainers)
    maintainer_agg = (
        maintains_exploded.groupby("pkg_name")
        .agg(
            n_pkgs_by_maintainers_30d=("maintainer_recent_pkg_count", "sum"),
            n_low_download_pkgs_by_maintainers=(
                "maintainer_low_download_pkg_count",
                "sum",
            ),
        )
        .reset_index()
    )

    # Merge maintainer features back to main dataframe
    df = df.merge(maintainer_agg, on="pkg_name", how="left")

    # Fill NaN with 0 for packages with no maintainer data
    df["n_pkgs_by_maintainers_30d"] = df["n_pkgs_by_maintainers_30d"].fillna(0)
    df["n_low_download_pkgs_by_maintainers"] = df[
        "n_low_download_pkgs_by_maintainers"
    ].fillna(0)

    # Feature 3: Count project URLs
    df["n_latest_project_urls"] = df["latest_project_urls"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    return df
