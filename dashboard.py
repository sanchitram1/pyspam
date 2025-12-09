import joblib
import pandas as pd
import streamlit as st

MODEL_FEATURE_COLUMNS = [
    "n_name_len",
    "has_digit_in_name",
    "has_dash_or_underscore",
    "cat_name_case",
    "n_summary_len",
    "n_desc_len",
    "n_desc_lines",
    "has_code_block_in_desc",
    "n_urls_in_desc",
    "has_suspicious_kw",
    "pct_non_ascii_desc",
    "t_age_first_release_days",
    "t_age_last_release_days",
    "n_versions",
    "t_median_release_gap_days",
    "has_single_release",
    "cat_weekday_of_last_release",
    "n_maintainers",
    "pct_free_email_domains",
    "has_disposable_email",
    "has_missing_author",
    "has_homepage",
    "has_repo_url",
    "cat_repo_host",
    "has_issue_tracker",
    "has_docs_url",
    "has_license",
    "cat_license_family",
    "n_classifiers",
    "has_prog_lang_classifier",
    "has_typing_classifier",
    "n_distributions",
    "n_requires",
    "has_extras",
    "rule_no_repo_low_desc_len",
    "rule_suspicious_name_and_dep",
    "n_downloads_7d",
    "n_downloads_30d",
    "n_dependents_est",
    "n_lev_dist_to_top1",
    "n_lev_dist_to_alias",
    "sim_tfidf_to_legit_centroid",
    "dist_embed_to_legit_desc",
    "n_pkgs_by_maintainers_30d",
    "n_low_download_pkgs_by_maintainers",
    "n_latest_project_urls",
    "has_dependency_to_top_brand",
    "min_dep_lev_to_brand",
    "has_dependency_lev_close_to_brand",
    "t_time_of_day",
]


@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.joblib")


@st.cache_data
def load_data():
    df = pd.read_json("pyspam_features_with_offline.jsonl", lines=True)  # or CSV
    df.set_index("pkg_name", inplace=True)
    return df


def classify_risk(score: float) -> str:
    if score >= 0.8:
        return "HIGH"
    elif score >= 0.5:
        return "MEDIUM"
    else:
        return "LOW"


def main():
    model = load_model()
    df = load_data()

    page = st.sidebar.radio("Navigate", ["Check a Package", "Explore Packages"])

    if page == "Check a Package":
        st.title("PyPI Package Risk Checker")

        pkg_name = st.text_input("Package name (e.g. requests)")
        if st.button("Analyze") and pkg_name:
            if pkg_name not in df.index:
                st.error("Package not found in feature dataset.")
                return

            row = df.loc[[pkg_name]]
            X = row[MODEL_FEATURE_COLUMNS]
            score = float(model.predict_proba(X)[0, 1])
            risk_label = classify_risk(score)

            # --- Top cards (3 columns) ---
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader(pkg_name)
                st.metric("Risk score", f"{score:.2f}", help="1.0 = highest risk")
                st.write(f"**Risk level:** {risk_label}")

            with col2:
                st.subheader("Downloads")
                st.metric("7 days", f"{row['n_downloads_7d'].values[0]}")
                st.metric("30 days", f"{row['n_downloads_30d'].values[0]}")

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

            with st.expander("Activity & Downloads"):
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
                            "dist_embed_to_legit_desc",
                        ]
                    ]
                )

    elif page == "Explore Packages":
        st.title("Explore Packages by Risk")

        # assume df already has risk_score/risk_label columns, or compute here
        if "risk_score" not in df.columns:
            X_all = df[MODEL_FEATURE_COLUMNS]
            df["risk_score"] = model.predict_proba(X_all)[:, 1]
            df["risk_label"] = df["risk_score"].apply(classify_risk)

        min_score = st.sidebar.slider("Min risk score", 0.0, 1.0, 0.5, 0.01)
        min_downloads = st.sidebar.number_input("Min 30d downloads", 0, value=0)

        mask = (df["risk_score"] >= min_score) & (
            df["n_downloads_30d"] >= min_downloads
        )
        df_filtered = df.loc[
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

        st.dataframe(df_filtered)


if __name__ == "__main__":
    main()
