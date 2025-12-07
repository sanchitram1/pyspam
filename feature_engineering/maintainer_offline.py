import pandas as pd
from settings import LOW_DOWNLOAD_THRESHOLD_30D


# -------------------------------------------------------
# Step 4: D.* maintainer-based offline features
#   - n_pkgs_by_maintainers_30d
#   - n_low_download_pkgs_by_maintainers
# -------------------------------------------------------
# Explode maintainers to build per-maintainer stats
# unify all maintainer emails for this stage
def collect_maintainers(row):
    return list(set(row["distinct_authors"] + row["distinct_maintainers"]))


def handle_maintainers(df: pd.DataFrame):
    maintainers_df = df[
        [
            "pkg_name",
            "distinct_authors",
            "distinct_maintainers",
            "t_age_last_release_days",
            "n_downloads_30d",
        ]
    ].copy()

    maintainers_df["all_maintainers"] = maintainers_df.apply(
        collect_maintainers, axis=1
    )
    maintains_exploded = maintainers_df.explode("all_maintainers")
    maintains_exploded = maintains_exploded.rename(
        columns={"all_maintainers": "maintainer_email"}
    )
    maintains_exploded = maintains_exploded[
        maintains_exploded["maintainer_email"] != ""
    ]

    # a) n_pkgs_by_maintainers_30d (per maintainer)
    recent_mask = maintains_exploded["t_age_last_release_days"] <= 30
    maintainer_recent_pkg_counts = (
        maintains_exploded[recent_mask]
        .groupby("maintainer_email")["pkg_name"]
        .nunique()
        .rename("maintainer_recent_pkg_count")
        .reset_index()
    )

    # b) n_low_download_pkgs_by_maintainers
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

    # join stats back onto exploded maintainer table
    maintains_exploded = maintains_exploded.merge(
        maintainer_recent_pkg_counts, on="maintainer_email", how="left"
    ).merge(maintainer_lowdl_pkg_counts, on="maintainer_email", how="left")

    maintains_exploded["maintainer_recent_pkg_count"] = maintains_exploded[
        "maintainer_recent_pkg_count"
    ].fillna(0)
    maintains_exploded["maintainer_low_download_pkg_count"] = maintains_exploded[
        "maintainer_low_download_pkg_count"
    ].fillna(0)

    # aggregate per package (e.g., sum or max over maintainers)
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

    # merge back to main df
    df = df.merge(maintainer_agg, on="pkg_name", how="left")
    df["n_pkgs_by_maintainers_30d"] = df["n_pkgs_by_maintainers_30d"].fillna(0)
    df["n_low_download_pkgs_by_maintainers"] = df[
        "n_low_download_pkgs_by_maintainers"
    ].fillna(0)
    df["n_latest_project_urls"] = df["latest_project_urls"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    return df
