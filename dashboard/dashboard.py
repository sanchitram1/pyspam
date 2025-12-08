import streamlit as st
import pandas as pd
import joblib

DATA_PATH = "/Users/reezhan/Desktop/UCB/242A/project/spam/pyspam_features_with_offline.jsonl"
MODEL_PATH = "models/ensemble.joblib"   # <- new ensemble pipeline

MODEL_FEATURE_COLUMNS = [
    'n_name_len', 'has_digit_in_name', 'has_dash_or_underscore',
    'cat_name_case', 'n_summary_len', 'n_desc_len', 'n_desc_lines',
    'has_code_block_in_desc', 'n_urls_in_desc', 'has_suspicious_kw',
    'pct_non_ascii_desc', 't_age_first_release_days',
    't_age_last_release_days', 'n_versions', 't_median_release_gap_days',
    'has_single_release', 'cat_weekday_of_last_release', 'n_maintainers',
    'pct_free_email_domains', 'has_disposable_email', 'has_missing_author',
    'has_homepage', 'has_repo_url', 'cat_repo_host', 'has_issue_tracker',
    'has_docs_url', 'has_license', 'cat_license_family', 'n_classifiers',
    'has_prog_lang_classifier', 'has_typing_classifier', 'n_distributions',
    'n_requires', 'has_extras', 'rule_no_repo_low_desc_len',
    'rule_suspicious_name_and_dep', 'n_downloads_7d',
    'n_downloads_30d', 'n_dependents_est', 'n_lev_dist_to_top1',
    'n_lev_dist_to_alias', 'sim_tfidf_to_legit_centroid',
    'dist_embed_to_legit_desc', 'n_pkgs_by_maintainers_30d',
    'n_low_download_pkgs_by_maintainers', 'n_latest_project_urls',
    'has_dependency_to_top_brand', 'min_dep_lev_to_brand',
    'has_dependency_lev_close_to_brand', 't_time_of_day'
]

# -------------------------------------------------------------------
# Loading helpers
# -------------------------------------------------------------------
@st.cache_resource
def load_model():
    # ensemble pipeline: (preprocessor -> scaler -> ensemble voter)
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    df = pd.read_json(DATA_PATH, lines=True)
    df.set_index("pkg_name", inplace=True)
    return df

# -------------------------------------------------------------------
# Risk helpers
# -------------------------------------------------------------------
def classify_risk(score: float) -> str:
    if score >= 0.8:
        return "HIGH"
    elif score >= 0.5:
        return "MEDIUM"
    else:
        return "LOW"


@st.cache_data
def compute_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Attach risk_score and risk_label columns to a copy of df."""
    model = load_model()
    X_all = df[MODEL_FEATURE_COLUMNS]
    probs = model.predict_proba(X_all)[:, 1]
    out = df.copy()
    out["risk_score"] = probs
    out["risk_label"] = out["risk_score"].apply(classify_risk)
    return out


# -------------------------------------------------------------------
# Main app
# -------------------------------------------------------------------
def page_check_package(df: pd.DataFrame):
    st.title("PyPI Package Risk Checker (Ensemble)")

    st.caption(
        "Model: ensemble of multiple classifiers (e.g. decision tree, "
        "logistic regression, random forest, SVM) wrapped in a single pipeline."
    )

    pkg_name = st.text_input("Package name (e.g. `requests`)")
    threshold = st.slider(
        "Risk level threshold for HIGH / MEDIUM / LOW",
        0.0, 1.0, 0.5, 0.01,
        help="This affects only the textual label, not the raw score."
    )

    model = load_model()

    if st.button("Analyze") and pkg_name:
        if pkg_name not in df.index:
            st.error("Package not found in feature dataset.")
            st.stop()

        row = df.loc[[pkg_name]]  # keep as DataFrame
        X = row[MODEL_FEATURE_COLUMNS]

        # ensemble pipeline handles preprocessing internally
        score = float(model.predict_proba(X)[0, 1])
        risk_label = classify_risk(score if threshold == 0.5 else score)

        # --- Top cards ---
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader(pkg_name)
            st.metric("Risk score", f"{score:.3f}", help="1.0 = highest risk")
            st.write(f"**Risk level:** :red[{risk_label}]"
                     if risk_label == "HIGH"
                     else f"**Risk level:** :orange[{risk_label}]"
                     if risk_label == "MEDIUM"
                     else f"**Risk level:** :green[{risk_label}]")

        with col2:
            st.subheader("Downloads")
            st.metric("7 days", f"{int(row['n_downloads_7d'].values[0]):,}")
            st.metric("30 days", f"{int(row['n_downloads_30d'].values[0]):,}")

        with col3:
            st.subheader("Release activity")
            st.metric("Versions", int(row["n_versions"].values[0]))
            st.write("Single release:", bool(row["has_single_release"].values[0]))
            st.write(
                "Age since first release (days):",
                int(row["t_age_first_release_days"].values[0]),
            )
            st.write(
                "Age since last release (days):",
                int(row["t_age_last_release_days"].values[0]),
            )

        # --- Key signals sections ---
        st.markdown("### Key signals")

        with st.expander("Activity & Downloads", expanded=True):
            st.write(
                row[
                    [
                        "n_downloads_7d",
                        "n_downloads_30d",
                        "n_versions",
                        "has_single_release",
                        "t_age_first_release_days",
                        "t_age_last_release_days",
                        "n_distributions",
                    ]
                ]
            )

        with st.expander("Repository & Ecosystem"):
            st.write(
                row[
                    [
                        "has_repo_url",
                        "cat_repo_host",
                        "n_dependents_est",
                        "min_dep_lev_to_brand",
                        "n_requires",
                        "has_dependency_to_top_brand",
                        "has_dependency_lev_close_to_brand",
                    ]
                ]
            )

        with st.expander("Maintainers"):
            st.write(
                row[
                    [
                        "n_maintainers",
                        "n_low_download_pkgs_by_maintainers",
                        "n_pkgs_by_maintainers_30d",
                    ]
                ]
            )

        with st.expander("Description & Metadata"):
            st.write(
                row[
                    [
                        "n_desc_len",
                        "n_desc_lines",
                        "n_summary_len",
                        "n_classifiers",
                        "has_homepage",
                        "has_docs_url",
                        "has_license",
                        "dist_embed_to_legit_desc",
                    ]
                ]
            )


def page_explore(df_with_risk: pd.DataFrame):
    st.title("Explore Packages by Ensemble Risk")

    st.sidebar.markdown("### Filters")

    min_score = st.sidebar.slider("Min risk score", 0.0, 1.0, 0.5, 0.01)
    min_downloads = st.sidebar.number_input(
        "Min 30d downloads", 0, value=0, step=100
    )
    label_filter = st.sidebar.multiselect(
        "Risk levels",
        options=["HIGH", "MEDIUM", "LOW"],
        default=["HIGH", "MEDIUM", "LOW"],
    )

    mask = (
        (df_with_risk["risk_score"] >= min_score)
        & (df_with_risk["n_downloads_30d"] >= min_downloads)
        & (df_with_risk["risk_label"].isin(label_filter))
    )

    df_filtered = df_with_risk.loc[
        mask,
        [
            "risk_score",
            "risk_label",
            "n_downloads_7d",
            "n_downloads_30d",
            "n_versions",
            "has_single_release",
            "cat_repo_host",
            "n_dependents_est",
            "n_maintainers",
        ],
    ].sort_values("risk_score", ascending=False)

    st.caption(
        "Risk scores and labels are computed by the ensemble model on all packages."
    )
    st.dataframe(df_filtered, use_container_width=True)


def main():
    st.set_page_config(page_title="PyPI Risk Dashboard", layout="wide")

    df = load_data()
    df_with_risk = compute_risk_scores(df)

    page = st.sidebar.radio(
        "Navigate",
        ["Check a Package", "Explore Packages"],
        index=0,
    )

    if page == "Check a Package":
        page_check_package(df)
    else:
        page_explore(df_with_risk)


if __name__ == "__main__":
    main()