import streamlit as st
import pandas as pd
import joblib
import numpy as np
from typing import Literal

# --- 1. CONFIGURATION & PATHS
DATA_PATH = "../project/pyspam_features_with_offline.jsonl"
MODEL_PATH = "models/ensemble.joblib"  # <- new ensemble pipeline

# Define feature groups for cleaner display in the Deep Dive section
FEATURE_GROUPS = {
    "A. Identity": ['n_name_len', 'has_digit_in_name', 'has_dash_or_underscore', 'cat_name_case'],
    "B. Summary & Description": ['n_summary_len', 'n_desc_len', 'n_desc_lines', 'has_code_block_in_desc', 'n_urls_in_desc', 'has_suspicious_kw', 'pct_non_ascii_desc'],
    "C. Release & Activity": ['t_age_first_release_days', 't_age_last_release_days', 'n_versions', 't_median_release_gap_days', 'has_single_release', 'cat_weekday_of_last_release'],
    "D. Ownership & Maintainer": ['n_maintainers', 'pct_free_email_domains', 'has_disposable_email', 'has_missing_author'],
    "E. Repository & Link": ['has_homepage', 'has_repo_url', 'cat_repo_host', 'has_issue_tracker', 'has_docs_url'],
    "F. License & Classifier": ['has_license', 'cat_license_family', 'n_classifiers', 'has_prog_lang_classifier', 'has_typing_classifier'],
    "G. Latest Distribution": ['n_distributions'],
    "H. Dependencies": ['n_requires', 'has_extras'],
    "I. Popularity": ['n_downloads_7d', 'n_downloads_30d', 'n_dependents_est'],
    "J. Rule-based": ['rule_no_repo_low_desc_len', 'rule_suspicious_name_and_dep'],
    "K. Offline/ML Features": ['n_lev_dist_to_top1', 'n_lev_dist_to_alias', 'sim_tfidf_to_legit_centroid', 'dist_embed_to_legit_desc', 'n_pkgs_by_maintainers_30d', 'n_low_download_pkgs_by_maintainers', 'n_latest_project_urls', 'has_dependency_to_top_brand', 'min_dep_lev_to_brand', 'has_dependency_lev_close_to_brand', 't_time_of_day']
}

MODEL_FEATURE_COLUMNS = [col for group in FEATURE_GROUPS.values() for col in group]

# --- 2. CORE HELPERS (LOADING, RISK CLASSIFICATION, STYLING)
# -------------------------------------------------------------------

@st.cache_resource
def load_model():
    """Load the trained ensemble model pipeline."""
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"Model file not found at {MODEL_PATH}. Cannot proceed.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

@st.cache_data
def load_data():
    """
    Load the feature data from JSONL file. 
    FIX: Ensure the index (pkg_name) is converted to lowercase for case-insensitive lookup.
    """
    try:
        df = pd.read_json(DATA_PATH, lines=True)
        # Assuming the column containing the package name is 'pkg_name' before set_index
        # If 'pkg_name' is not a column but intended to be derived from 'package' before being the index:
        if 'pkg_name' not in df.columns and 'package' in df.columns:
            df['pkg_name'] = df['package'] 
            
        df['pkg_name'] = df['pkg_name'].astype(str).str.lower()
        df.set_index("pkg_name", inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"Data file not found at {DATA_PATH}. Cannot proceed.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

def get_risk_color(label: str) -> str:
    """Returns a hex color based on risk label."""
    if label == "HIGH":
        return "#FF4B4B" # Red
    elif label == "MEDIUM":
        return "#FF8C00" # Dark Orange
    return "#00CC99" # Green

def classify_risk(score: float, threshold_high: float = 0.8, threshold_medium: float = 0.5) -> Literal["HIGH", "MEDIUM", "LOW"]:
    """Classifies the risk score into a label using user-defined thresholds."""
    if score >= threshold_high:
        return "HIGH"
    elif score >= threshold_medium:
        return "MEDIUM"
    return "LOW"

def display_risk_badge(score: float, label: str):
    """
    Renders a colored, professional-looking badge for the risk score/label.
    """
    color = get_risk_color(label)
    
    # Inject custom CSS for stylish metrics/badges
    st.markdown(
        f"""
        <style>
        .risk-score-metric > div[data-testid="stMetricValue"] {{
            font-size: 3rem; 
            color: {color};
            font-weight: 700;
        }}
        .risk-badge {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 0.5rem;
            font-weight: bold;
            color: white;
            background-color: {color};
            font-size: 1.1rem;
            text-transform: uppercase;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="risk-score-metric">', unsafe_allow_html=True)
        st.metric("Risk Score", f"{score:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f"**Risk Level**")
        st.markdown(f'<div class="risk-badge">{label}</div>', unsafe_allow_html=True)


@st.cache_data
def compute_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Attach risk_score and risk_label columns to a copy of df (using default thresholds)."""
    model = load_model()
    missing_cols = set(MODEL_FEATURE_COLUMNS) - set(df.columns)
    if missing_cols:
         st.warning(f"Missing model feature columns: {missing_cols}. Cannot compute scores.")
         return df.copy()

    X_all = df[MODEL_FEATURE_COLUMNS]
    probs = model.predict_proba(X_all)[:, 1]
    
    # Use .assign() for safe, efficient assignment
    out = df.assign(
        risk_score=probs,
        risk_label=[classify_risk(p, 0.8, 0.5) for p in probs] 
    )
    return out


# --- 3. PAGE FUNCTIONS
# -------------------------------------------------------------------

def page_check_package(df: pd.DataFrame):
    
    st.header("🔎 Package Risk Checker", divider='blue')
    st.caption("Input a package name to run an ensemble model prediction and review core features.")

    # --- INPUTS AND THRESHOLD SLIDER ---
    
    col_input, col_threshold = st.columns([2, 1])
    with col_input:
        pkg_name_raw = st.text_input("Package Name (e.g. `requests`)", key="pkg_input")
        # IMPORTANT FIX: Convert input to lowercase here, matching the data index.
        pkg_name = pkg_name_raw.strip().lower() 
        analyze_button = st.button("Analyze Package Risk", type="primary")

    # Threshold setup
    with col_threshold:
        st.markdown("**Threshold Control**")
        
        # Slider for the HIGH threshold
        threshold_high = st.slider(
             "HIGH Risk Threshold ($\ge$ value)",
             0.5, 1.0, 0.8, 0.01,
             key="threshold_high_slider",
             help="Score above this threshold is classified as HIGH."
        )
        threshold_medium = 0.5
        st.write(f"Medium Risk Threshold: $\ge$ **{threshold_medium}**")

    
    # --- ANALYSIS LOGIC ---
    if analyze_button and pkg_name:
        
        # FIX: Lookup is now case-insensitive because the index was lowercased in load_data()
        if pkg_name not in df.index:
            st.error(f"Package '{pkg_name_raw}' not found in the loaded feature dataset.")
            return

        try:
            # 1. Run Prediction
            model = load_model()
            row = df.loc[[pkg_name]] # Use the lowercased name for lookup
            X = row[MODEL_FEATURE_COLUMNS]
            score = float(model.predict_proba(X)[0, 1])
            
            # Use the user-defined HIGH threshold and fixed MEDIUM threshold
            risk_label = classify_risk(score, threshold_high, threshold_medium)
            
            st.markdown("---")
            
            # 2. Display Core Metrics (Row 1)
            # Use the raw name for display (e.g., 'Pandas' instead of 'pandas')
            st.markdown(f"## 📦 **{pkg_name_raw}**") 
            
            col_risk, col_pop, col_activity = st.columns(3)

            with col_risk:
                display_risk_badge(score, risk_label)
            
            with col_pop:
                st.subheader("Downloads (I.*)")
                st.metric("7 days", f"{int(row['n_downloads_7d'].values[0]):,}")
                st.metric("30 days", f"{int(row['n_downloads_30d'].values[0]):,}")
                st.caption(f"Estimated Dependents: {int(row['n_dependents_est'].values[0]):,}")

            with col_activity:
                st.subheader("Release Activity (C.*)")
                st.metric("Versions", int(row["n_versions"].values[0]))
                st.write(f"First Release (Days): {int(row['t_age_first_release_days'].values[0])}")
                st.write(f"Last Release (Days): {int(row['t_age_last_release_days'].values[0])}")
                st.write(f"Single Release: {'✅ Yes' if bool(row['has_single_release'].values[0]) else '❌ No'}")


            st.markdown("---")
            st.subheader("Model-Critical Feature Breakdown")

            # 3. Feature Breakdown (Tabs)
            row_df = row.T.reset_index().rename(columns={'index': 'Feature', pkg_name: 'Value'})
            
            tabs = st.tabs([f"{group_name.split('. ')[0]} - {group_name.split('. ')[1]}" for group_name in FEATURE_GROUPS.keys()])
            
            for i, (group_name, features) in enumerate(FEATURE_GROUPS.items()):
                with tabs[i]:
                    st.markdown(f"**{group_name}**")
                    
                    group_data = row_df[row_df['Feature'].isin(features)]
                    
                    def format_value(val, feature_name):
                        if isinstance(val, (int, float)):
                            if val in (0, 1) and feature_name.startswith('has_'):
                                return "✅ Yes" if val == 1 else "❌ No"
                        if isinstance(val, float):
                             return f"{val:.4f}"
                        if isinstance(val, str) and not val:
                            return "N/A"
                        return val
                    
                    group_data = group_data.copy()
                    group_data['Value'] = [format_value(v, f) for v, f in zip(group_data['Value'], group_data['Feature'])]
                    
                    st.dataframe(
                        group_data,
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Feature": st.column_config.TextColumn("Feature", help="Model input feature name."),
                            "Value": st.column_config.TextColumn("Value", help="Value for the current package."),
                        }
                    )
                    
        except Exception as e:
            st.exception(e)
            st.error("An error occurred during package analysis.")


def page_explore(df_with_risk: pd.DataFrame):
    
    st.header("⚡ Global Risk Overview", divider='red')
    st.caption("A high-level view of risk distribution and top threats across the entire dataset.")

    # 1. Global KPIs
    total_packages = len(df_with_risk)
    high_risk_count = (df_with_risk["risk_label"] == "HIGH").sum()
    avg_risk = df_with_risk["risk_score"].mean()

    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.metric("Total Packages Analyzed", f"{total_packages:,}")
    with col_kpi2:
        st.metric("Avg Risk Score", f"{avg_risk:.3f}")
    with col_kpi3:
        st.markdown(f"**<span style='color:{get_risk_color('HIGH')}'>High Risk Packages (Count)</span>**", unsafe_allow_html=True)
        st.markdown(f"## {high_risk_count:,}")
    with col_kpi4:
        st.metric("Avg Maintainers", f"{df_with_risk['n_maintainers'].mean():.2f}")

    st.markdown("---")

    # 2. Top Threats List (Filtered Table)
    st.subheader("Top Threats & Filtered List")
    
    with st.container(border=True):
        st.markdown("### ⚙️ Filter Dashboard Results")
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            min_score = st.slider("Minimum Risk Score (Filter)", 0.0, 1.0, 0.9, 0.01, key='explore_min_score')
        with col_f2:
            min_downloads = st.number_input("Minimum 30d Downloads", 0, value=0, step=100, key='explore_min_downloads')
        with col_f3:
            label_filter = st.multiselect(
                "Risk Levels to Display",
                options=["HIGH", "MEDIUM", "LOW"],
                default=["HIGH"],
                key='explore_label_filter'
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
            "cat_repo_host",
            "n_maintainers",
            "t_age_last_release_days",
        ],
    ].sort_values("risk_score", ascending=False)
    
    st.info(f"Showing **{len(df_filtered):,}** packages matching the criteria (using default classification thresholds).")
    
    st.dataframe(
        df_filtered, 
        use_container_width=True,
        column_config={
            "risk_score": st.column_config.ProgressColumn("Risk Score", help="Ensemble model probability of scam (0-1).", format="%.3f", min_value=0, max_value=1,),
            "risk_label": st.column_config.TextColumn("Risk", help="Classifier label: HIGH, MEDIUM, or LOW (using default 0.8/0.5 thresholds)."),
            "n_downloads_7d": st.column_config.NumberColumn("Downloads (7d)", format="%,d"),
            "n_downloads_30d": st.column_config.NumberColumn("Downloads (30d)", format="%,d"),
            "n_versions": "Versions",
            "t_age_last_release_days": "Last Release Age (Days)",
            "cat_repo_host": "Repo Host",
            "n_maintainers": "Maintainers",
        }
    )


# --- 4. MAIN APP EXECUTION
# -------------------------------------------------------------------

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(
        page_title="PyPI Security Dashboard", 
        layout="wide", 
        initial_sidebar_state="expanded", 
        menu_items=None
    )

    st.title("🛡️ PyPI Package Scam Detection Dashboard")
    st.markdown("The project's primary goal is to build an ensemble spam classifier for PyPI packages by tackling the challenge of creating a supervised dataset from unstructured metadata, achieved by labeling known-bad (typosquats/malware) and known-good packages to train models like Logistic Regression, Random Forest, and SVMs. Evaluation focuses on metrics like Precision and Recall, critical for ensuring the effective security screening of the software supply chain.")

    data_load_state = st.info("Initializing system...")
    
    data_load_step = st.empty()
    data_load_step.text("Loading features from data file...")
    
    # Use st.spinner to show progress while loading data (user experience enhancement)
    with st.spinner('Loading features... this may take a moment.'):
        df = load_data()
    
    data_load_step.text("Computing ensemble risk scores for all packages...")
    if 'package' not in df.columns:
        df['package'] = df.index # Add 'package' column if it's missing (for the explorer view)
        
    df_with_risk = compute_risk_scores(df)
    
    data_load_state.success("System ready: Data loaded and risk scores computed successfully!")
    data_load_step.empty()
    
    # Use the sidebar for navigation for a classic dashboard feel
    page = st.sidebar.radio(
        "Navigate",
        ["🔎 Package Risk Checker", "🌎 Global Risk Overview"],
        index=0,
    )

    if page == "🔎 Package Risk Checker":
        page_check_package(df)
    else:
        page_explore(df_with_risk)


if __name__ == "__main__":
    main()