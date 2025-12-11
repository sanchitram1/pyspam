import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
API_URL = os.getenv(
    "PYSPAM_API_URL", "http://127.0.0.1:8000/scan/"
)  # adjust to your real endpoint

st.set_page_config(
    page_title="PyPI Risk Radar",
    page_icon="🛰️",
    layout="wide",
)

# ---------------------------------------------------------
# Custom CSS for Elon-style sleek UI (dark, minimal)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global */
    .stApp {
        background: radial-gradient(circle at top, #141e30 0, #0b0c10 40%, #000000 100%);
        color: #f5f5f5;
        font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
    }
    /* Main container */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1300px;
    }
    /* Cards */
    .risk-card {
        background: rgba(15, 23, 42, 0.9);
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        border: 1px solid rgba(148, 163, 184, 0.3);
        box-shadow: 0 16px 45px rgba(15, 23, 42, 0.65);
    }
    .mini-card {
        background: rgba(15, 23, 42, 0.85);
        border-radius: 16px;
        padding: 0.9rem 1.1rem;
        border: 1px solid rgba(148, 163, 184, 0.18);
    }
    .metric-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9ca3af;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: #f9fafb;
    }
    .pill {
        display: inline-flex;
        align-items: center;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        border: 1px solid rgba(148, 163, 184, 0.55);
        color: #e5e7eb;
        background: rgba(15, 23, 42, 0.9);
    }
    .pill-dot {
        width: 7px;
        height: 7px;
        border-radius: 999px;
        margin-right: 0.35rem;
    }
    .pill-dot.low { background: #22c55e; }
    .pill-dot.medium { background: #eab308; }
    .pill-dot.high { background: #f97316; }
    .pill-dot.critical { background: #ef4444; }

    /* Risk donut */
    .risk-donut-wrapper {
        position: relative;
        width: 210px;
        height: 210px;
        margin: auto;
    }
    .risk-donut {
        width: 210px;
        height: 210px;
        border-radius: 50%;
        background: conic-gradient(
            from 220deg,
            #22c55e 0,
            #84cc16 22%,
            #eab308 44%,
            #f97316 66%,
            #ef4444 88%,
            #4b5563 100%
        );
        padding: 12px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.7);
    }
    .risk-donut-inner {
        width: 100%;
        height: 100%;
        border-radius: 50%;
        background: radial-gradient(circle at top, #020617 0, #020617 50%, #020617 100%);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .risk-score {
        font-size: 2.4rem;
        font-weight: 700;
    }
    .risk-score-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: #9ca3af;
        margin-top: 0.2rem;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        padding: 0.4rem 0.4rem;
        background: rgba(15, 23, 42, 0.85);
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.3);
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.3rem 0.9rem !important;
        border-radius: 999px !important;
        font-size: 0.8rem;
    }

    /* DataFrame tweaks */
    .dataframe tbody tr th {
        color: #9ca3af;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

OFFLINE_FEATURE_KEYS = [
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
    "t_time_of_day",  # vs bucket
]


def risk_tier(p: float):
    if p is None:
        return "Unknown", "low"
    if p < 0.2:
        return "Low", "low"
    if p < 0.5:
        return "Moderate", "medium"
    if p < 0.8:
        return "High", "high"
    return "Critical", "critical"


def bool_to_icon(v):
    if v is None:
        return "–"
    return "✅" if bool(v) else "❌"


def safe_get(d, key, default=None):
    return d.get(key, default) if isinstance(d, dict) else default


@st.cache_data(show_spinner=False)
def fetch_package_data(api_url: str, pkg_name: str):
    url = f"{api_url.rstrip('/')}/{pkg_name}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------
# Layout: Header & Controls
# ---------------------------------------------------------
col_title, col_input = st.columns([2, 3])

with col_title:
    st.markdown(
        """
        <div class="risk-card">
            <div class="pill">
                <span class="pill-dot low"></span>
                PYPI RISK RADAR
            </div>
            <h1 style="margin-top:0.6rem;margin-bottom:0.25rem;font-size:1.6rem;">
                Package Threat Intelligence Console
            </h1>
            <p style="color:#9ca3af;font-size:0.9rem;margin-bottom:0;">
                Type a package name, ping your scoring API, and get an at-a-glance threat profile
                mixing metadata, maintainers, and ML signals.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_input:
    with st.container():
        st.write("")  # spacing
        pkg_name = st.text_input(
            "Package name",
            value="requests",
            placeholder="e.g. numpy, torch, permalint",
        )
        api_url_override = st.text_input(
            "API base URL",
            value=API_URL,
            help="Override if you're testing a different backend.",
        )
        go = st.button("Scan package 🚀", type="primary", use_container_width=True)

# ---------------------------------------------------------
# Fetch & show results
# ---------------------------------------------------------
if go and pkg_name.strip():
    try:
        with st.spinner("Contacting API and computing threat profile..."):
            results_json = fetch_package_data(api_url_override, pkg_name.strip())

        package = safe_get(results_json, "package", pkg_name)
        raw_data = safe_get(results_json, "raw_data", {})
        features = safe_get(results_json, "features", {})
        pred = safe_get(results_json, "prediction", None)

        try:
            risk_prob = float(pred) if pred is not None else None
        except Exception:
            risk_prob = None

        tier_label, tier_slug = risk_tier(risk_prob)

        # ---------------------------------------------
        # TOP STRIP: risk donut + key metrics
        # ---------------------------------------------
        top_col1, top_col2 = st.columns([1.1, 1.9])

        from textwrap import dedent

        with top_col1:
            html = f"""
                <div class="risk-card" style="text-align:center;">
                    <p class="metric-label" style="margin-bottom:0.25rem;">PACKAGE</p>
                    <p style="font-size:1.05rem;font-weight:600;margin-top:0;margin-bottom:0.8rem;">
                        {package}
                    </p>
                    <div class="risk-donut-wrapper">
                        <div class="risk-donut">
                            <div class="risk-donut-inner">
                                <div class="risk-score">
                                    {f"{risk_prob * 100:.0f}%" if risk_prob is not None else "–"}
                                </div>
                                <div class="risk-score-label">SCAM RISK</div>
                                <div style="margin-top:0.4rem;font-size:0.8rem;">
                                    <span class="pill">
                                        <span class="pill-dot {tier_slug}"></span>
                                        {tier_label.upper()}
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>
                    <p style="margin-top:0.9rem;font-size:0.75rem;color:#9ca3af;">
                        Risk profile
                    </p>
                </div>
            """
            st.markdown(dedent(html), unsafe_allow_html=True)

        with top_col2:
            # Quick stats row
            c1, c2, c3, c4 = st.columns(4)

            age_first = safe_get(raw_data, "t_age_first_release_days")
            age_last = safe_get(raw_data, "t_age_last_release_days")
            n_versions = safe_get(raw_data, "n_versions")
            n_maintainers = safe_get(raw_data, "n_maintainers")
            n_downloads_7d = safe_get(raw_data, "n_downloads_7d")
            n_downloads_30d = safe_get(raw_data, "n_downloads_30d")
            n_dependents = safe_get(raw_data, "n_dependents_est")

            with c1:
                st.markdown('<div class="mini-card">', unsafe_allow_html=True)
                st.markdown(
                    '<div class="metric-label">Downloads (7d)</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="metric-value">{int(n_downloads_7d):,}'
                    if n_downloads_7d is not None
                    else '<div class="metric-value">–',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div style="font-size:0.75rem;color:#9ca3af;">Recent velocity</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with c2:
                st.markdown('<div class="mini-card">', unsafe_allow_html=True)
                st.markdown(
                    '<div class="metric-label">Downloads (30d)</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="metric-value">{int(n_downloads_30d):,}'
                    if n_downloads_30d is not None
                    else '<div class="metric-value">–',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div style="font-size:0.75rem;color:#9ca3af;">Medium-term usage</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with c3:
                st.markdown('<div class="mini-card">', unsafe_allow_html=True)
                st.markdown(
                    '<div class="metric-label">Versions</div>', unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="metric-value">{int(n_versions)}</div>'
                    if n_versions is not None
                    else '<div class="metric-value">–</div>',
                    unsafe_allow_html=True,
                )
                single_flag = bool(safe_get(raw_data, "has_single_release"))
                st.markdown(
                    f'<div style="font-size:0.75rem;color:#9ca3af;">{"Single release only" if single_flag else "Release history"}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with c4:
                st.markdown('<div class="mini-card">', unsafe_allow_html=True)
                st.markdown(
                    '<div class="metric-label">Maintainers</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="metric-value">{int(n_maintainers)}</div>'
                    if n_maintainers is not None
                    else '<div class="metric-value">–</div>',
                    unsafe_allow_html=True,
                )
                pct_free = safe_get(raw_data, "pct_free_email_domains")
                pct_text = (
                    f"{pct_free * 100:.0f}% free domains"
                    if pct_free is not None
                    else "Email hygiene"
                )
                st.markdown(
                    f'<div style="font-size:0.75rem;color:#9ca3af;">{pct_text}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            # Downloads 7d vs 30d bar
            if (n_downloads_7d is not None) or (n_downloads_30d is not None):
                dl_data = {
                    "Window": [],
                    "Downloads": [],
                }
                if n_downloads_7d is not None:
                    dl_data["Window"].append("7d")
                    dl_data["Downloads"].append(n_downloads_7d)
                if n_downloads_30d is not None:
                    dl_data["Window"].append("30d")
                    dl_data["Downloads"].append(n_downloads_30d)

                df_dl = pd.DataFrame(dl_data)
                fig_dl = px.bar(
                    df_dl,
                    x="Window",
                    y="Downloads",
                )
                fig_dl.update_layout(
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=200,
                    xaxis_title=None,
                    yaxis_title=None,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                fig_dl.update_yaxes(
                    showgrid=True, gridwidth=0.3, gridcolor="rgba(148,163,184,0.35)"
                )
                st.markdown(" ", unsafe_allow_html=True)
                st.plotly_chart(
                    fig_dl, use_container_width=True, config={"displayModeBar": False}
                )

        # ---------------------------------------------
        # TABS: Identity, Release, Ownership & Links, License & Deps, ML Signals, Raw fields
        # ---------------------------------------------
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            [
                "🧬 Identity & Text",
                "📡 Releases",
                "👤 Ownership & Links",
                "📜 License & Deps",
                "🤖 ML Signals",
                "🧾 Raw Fields",
            ]
        )

        # ---------- Tab 1: Identity & Text ----------
        with tab1:
            st.markdown("### Package identity & description quality")

            col_id, col_txt = st.columns(2)

            with col_id:
                id_rows = [
                    ["Name length", safe_get(raw_data, "n_name_len")],
                    [
                        "Has digit in name",
                        bool_to_icon(safe_get(raw_data, "has_digit_in_name")),
                    ],
                    [
                        "Has dash/underscore",
                        bool_to_icon(safe_get(raw_data, "has_dash_or_underscore")),
                    ],
                    ["Name casing", safe_get(raw_data, "cat_name_case")],
                    [
                        "Suspicious name & dep rule",
                        bool_to_icon(
                            safe_get(raw_data, "rule_suspicious_name_and_dep")
                        ),
                    ],
                ]
                df_id = pd.DataFrame(id_rows, columns=["Feature", "Value"])
                st.dataframe(df_id, use_container_width=True, hide_index=True)

            with col_txt:
                txt_rows = [
                    ["Summary length (chars)", safe_get(raw_data, "n_summary_len")],
                    ["Description length (chars)", safe_get(raw_data, "n_desc_len")],
                    ["Description lines", safe_get(raw_data, "n_desc_lines")],
                    [
                        "Code block in description",
                        bool_to_icon(safe_get(raw_data, "has_code_block_in_desc")),
                    ],
                    ["URLs in description", safe_get(raw_data, "n_urls_in_desc")],
                    [
                        "Suspicious keywords in desc",
                        bool_to_icon(safe_get(raw_data, "has_suspicious_kw")),
                    ],
                    [
                        "Non-ASCII fraction in desc",
                        safe_get(raw_data, "pct_non_ascii_desc"),
                    ],
                    [
                        "No repo & short desc rule",
                        bool_to_icon(safe_get(raw_data, "rule_no_repo_low_desc_len")),
                    ],
                ]
                df_txt = pd.DataFrame(txt_rows, columns=["Feature", "Value"])
                st.dataframe(df_txt, use_container_width=True, hide_index=True)

        # ---------- Tab 2: Release & activity ----------
        with tab2:
            st.markdown("### Release cadence & activity")

            c1, c2 = st.columns(2)

            with c1:
                rel_rows = [
                    ["Age since first release (days)", age_first],
                    ["Age since last release (days)", age_last],
                    [
                        "Median gap between releases (days)",
                        safe_get(raw_data, "t_median_release_gap_days"),
                    ],
                    [
                        "Time-of-day bucket (last)",
                        safe_get(raw_data, "t_time_of_day_bucket"),
                    ],
                    [
                        "Weekday of last release",
                        safe_get(raw_data, "cat_weekday_of_last_release"),
                    ],
                ]
                df_rel = pd.DataFrame(rel_rows, columns=["Feature", "Value"])
                st.dataframe(df_rel, use_container_width=True, hide_index=True)

            with c2:
                n_dist = safe_get(raw_data, "n_distributions")
                st.markdown("#### Release footprint")
                foot_rows = [
                    ["Number of versions", n_versions],
                    ["Distributions for latest version", n_dist],
                ]
                df_foot = pd.DataFrame(foot_rows, columns=["Feature", "Value"])
                st.dataframe(df_foot, use_container_width=True, hide_index=True)

        # ---------- Tab 3: Ownership & Links ----------
        with tab3:
            st.markdown("### Maintainers & project links")

            col_own, col_link = st.columns(2)

            with col_own:
                own_rows = [
                    ["Maintainers (count)", n_maintainers],
                    [
                        "% Free email domains",
                        safe_get(raw_data, "pct_free_email_domains"),
                    ],
                    [
                        "Has disposable email",
                        bool_to_icon(safe_get(raw_data, "has_disposable_email")),
                    ],
                    [
                        "Missing author list",
                        bool_to_icon(safe_get(raw_data, "has_missing_author")),
                    ],
                ]
                df_own = pd.DataFrame(own_rows, columns=["Feature", "Value"])
                st.dataframe(df_own, use_container_width=True, hide_index=True)

            with col_link:
                link_rows = [
                    ["Has homepage", bool_to_icon(safe_get(raw_data, "has_homepage"))],
                    ["Has repo URL", bool_to_icon(safe_get(raw_data, "has_repo_url"))],
                    ["Repo host category", safe_get(raw_data, "cat_repo_host")],
                    [
                        "Has issue tracker",
                        bool_to_icon(safe_get(raw_data, "has_issue_tracker")),
                    ],
                    ["Has docs URL", bool_to_icon(safe_get(raw_data, "has_docs_url"))],
                ]
                df_link = pd.DataFrame(link_rows, columns=["Feature", "Value"])
                st.dataframe(df_link, use_container_width=True, hide_index=True)

        # ---------- Tab 4: License & Dependencies ----------
        with tab4:
            st.markdown("### License, classifiers & dependency footprint")

            col_lic, col_dep = st.columns(2)

            with col_lic:
                lic_rows = [
                    ["Has license", bool_to_icon(safe_get(raw_data, "has_license"))],
                    ["License family", safe_get(raw_data, "cat_license_family")],
                    ["Number of classifiers", safe_get(raw_data, "n_classifiers")],
                    [
                        "Has Python language classifier",
                        bool_to_icon(safe_get(raw_data, "has_prog_lang_classifier")),
                    ],
                    [
                        "Has typing classifier",
                        bool_to_icon(safe_get(raw_data, "has_typing_classifier")),
                    ],
                ]
                df_lic = pd.DataFrame(lic_rows, columns=["Feature", "Value"])
                st.dataframe(df_lic, use_container_width=True, hide_index=True)

            with col_dep:
                dep_rows = [
                    ["Number of dependencies", safe_get(raw_data, "n_requires")],
                    ["Has extras", bool_to_icon(safe_get(raw_data, "has_extras"))],
                    ["Estimated dependents", n_dependents],
                ]
                df_dep = pd.DataFrame(dep_rows, columns=["Feature", "Value"])
                st.dataframe(df_dep, use_container_width=True, hide_index=True)

        # ---------- Tab 5: ML Signals (offline features) ----------
        with tab5:
            st.markdown("### Offline ML signals driving the score")

            # Collect offline features if present in features dict
            ml_rows = []
            for key in OFFLINE_FEATURE_KEYS:
                if key in features:
                    ml_rows.append([key, features[key]])

            if not ml_rows:
                st.info("No offline feature fields found in the `features` payload.")
            else:
                df_ml = pd.DataFrame(ml_rows, columns=["Feature", "Value"])
                st.dataframe(df_ml, use_container_width=True, hide_index=True)

                # Simple bar chart for numeric-only subset
                numeric_df = df_ml[
                    pd.to_numeric(df_ml["Value"], errors="coerce").notna()
                ].copy()
                if not numeric_df.empty:
                    numeric_df["Value"] = numeric_df["Value"].astype(float)
                    fig_ml = px.bar(
                        numeric_df,
                        x="Feature",
                        y="Value",
                    )
                    fig_ml.update_layout(
                        height=280,
                        margin=dict(l=0, r=0, t=10, b=90),
                        xaxis_title=None,
                        yaxis_title=None,
                        xaxis_tickangle=-35,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    fig_ml.update_yaxes(
                        showgrid=True, gridwidth=0.3, gridcolor="rgba(148,163,184,0.35)"
                    )
                    st.plotly_chart(
                        fig_ml,
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )

        # ---------- Tab 6: Raw fields ----------
        with tab6:
            st.markdown("### Raw text & arrays from PyPI metadata")

            latest_summary = safe_get(raw_data, "latest_summary") or ""
            latest_description = safe_get(raw_data, "latest_description") or ""
            latest_urls = safe_get(raw_data, "latest_project_urls") or []
            distinct_authors = safe_get(raw_data, "distinct_authors") or []
            distinct_maintainers = safe_get(raw_data, "distinct_maintainers") or []
            distinct_keywords = safe_get(raw_data, "distinct_keywords") or []
            distinct_classifiers = safe_get(raw_data, "distinct_classifiers") or []
            latest_deps = safe_get(raw_data, "latest_dependencies") or []

            with st.expander("Latest summary"):
                if latest_summary:
                    st.write(latest_summary)
                else:
                    st.write("No summary available.")

            with st.expander("Latest description (raw)"):
                if latest_description:
                    st.write(latest_description)
                else:
                    st.write("No description available.")

            with st.expander("Project URLs"):
                if latest_urls:
                    for u in latest_urls:
                        st.markdown(f"- `{u}`")
                else:
                    st.write("No project URLs found.")

            with st.expander("Authors & maintainers"):
                if distinct_authors:
                    st.markdown("**Authors (emails)**")
                    for a in distinct_authors:
                        st.markdown(f"- `{a}`")
                if distinct_maintainers:
                    st.markdown("**Maintainers (emails)**")
                    for m in distinct_maintainers:
                        st.markdown(f"- `{m}`")
                if not distinct_authors and not distinct_maintainers:
                    st.write("No author or maintainer emails captured.")

            with st.expander("Keywords & classifiers"):
                if distinct_keywords:
                    st.markdown("**Keywords**")
                    st.write(", ".join(distinct_keywords))
                if distinct_classifiers:
                    st.markdown("**Classifiers**")
                    for c in distinct_classifiers:
                        st.markdown(f"- `{c}`")
                if not distinct_keywords and not distinct_classifiers:
                    st.write("No keywords or classifiers listed.")

            with st.expander("Latest dependencies (requires_dist)"):
                if latest_deps:
                    for d in latest_deps:
                        st.markdown(f"- `{d}`")
                else:
                    st.write("No dependency list available.")

    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e}")
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

elif go and not pkg_name.strip():
    st.warning("Please enter a package name to scan.")
