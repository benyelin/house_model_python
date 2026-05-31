from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

INPUTS = Path("inputs")
OUTPUTS = Path("outputs")

HOUSE_INPUTS = INPUTS / "house_race_inputs.csv"
NATIONAL_ENV_AUDIT = INPUTS / "house_national_environment_audit.csv"

HOUSE_RACE_STATS = OUTPUTS / "house_race_stats.csv"
HOUSE_SEAT_DISTRIBUTION = OUTPUTS / "house_seat_distribution.csv"
HOUSE_FORECAST_SUMMARY = OUTPUTS / "house_forecast_summary.csv"

st.set_page_config(
    page_title="2026 House Forecast Dashboard",
    layout="wide",
)


# -----------------------------
# Helpers
# -----------------------------
def read_csv_safe(path):
    try:
        if Path(path).exists():
            return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Could not read {path}: {e}")
    return pd.DataFrame()


def as_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def fmt_margin(x):
    x = as_float(x)
    if pd.isna(x):
        return "—"
    if x > 0:
        return f"D+{x:.1f}"
    if x < 0:
        return f"R+{abs(x):.1f}"
    return "Even"


def fmt_pct(x):
    x = as_float(x)
    if pd.isna(x):
        return "—"
    return f"{x:.1%}"


def fmt_num(x, digits=2):
    x = as_float(x)
    if pd.isna(x):
        return "—"
    return f"{x:.{digits}f}"


def race_rating_from_prob(p):
    p = as_float(p)
    if pd.isna(p):
        return "Unknown"

    if p >= 0.95:
        return "Safe D"
    if p >= 0.85:
        return "Likely D"
    if p >= 0.65:
        return "Lean D"
    if p >= 0.55:
        return "Tilt D"
    if p > 0.45:
        return "Toss-up"
    if p > 0.35:
        return "Tilt R"
    if p > 0.15:
        return "Lean R"
    if p > 0.05:
        return "Likely R"
    return "Safe R"


def normalize_bool_series(s):
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def compact_district_label(row):
    state = str(row.get("state", "")).strip().upper()
    district = str(row.get("district", "")).strip()
    return normalize_district_id(state, district)


AT_LARGE_STATES = {
    "AK", "DE", "ND", "SD", "VT", "WY"
}


def normalize_district_value(state, district):
    state = str(state).strip().upper()
    district = str(district).strip().upper()

    if district in ["", "NAN", "NONE"]:
        return ""

    if state in AT_LARGE_STATES and district in ["1", "01", "AL", "AT-LARGE", "AT LARGE", "AT_LARGE"]:
        return "AL"

    # Normalize numbered districts like 01 -> 1.
    try:
        if district.isdigit():
            return str(int(district))
    except Exception:
        pass

    return district


def normalize_district_id(state, district):
    state = str(state).strip().upper()
    district = normalize_district_value(state, district)

    if state == "" or district == "":
        return ""

    return f"{state}-{district}"


def normalize_existing_district_id(raw_district_id):
    raw = str(raw_district_id).strip().upper()

    if "-" not in raw:
        return raw

    state, district = raw.split("-", 1)
    return normalize_district_id(state, district)


# -----------------------------
# Load data
# -----------------------------
df = read_csv_safe(HOUSE_INPUTS)
env = read_csv_safe(NATIONAL_ENV_AUDIT)

race_stats_output = read_csv_safe(HOUSE_RACE_STATS)
seat_distribution_output = read_csv_safe(HOUSE_SEAT_DISTRIBUTION)
forecast_summary_output = read_csv_safe(HOUSE_FORECAST_SUMMARY)

# Prefer simulated race stats when available.
if not race_stats_output.empty:
    df = race_stats_output.copy()

st.title("2026 House Forecast Dashboard")
st.caption("First-pass House fundamentals model using district presidential margins, shared national environment, and incumbency flags.")

if df.empty:
    st.error("No House input file found. Run `python3 import_house_model_seed.py` and `python3 recalculate_house_fundamentals.py` first.")
    st.stop()

df = df.copy()

if "state" in df.columns:
    df["state"] = df["state"].astype(str).str.strip().str.upper()

if "district_id" not in df.columns:
    df["district_id"] = df.apply(compact_district_label, axis=1)
else:
    df["district_id"] = df.apply(
        lambda row: normalize_district_id(row.get("state", ""), row.get("district", "")),
        axis=1
    )

for col in [
    "pres_2024_margin_dem",
    "pres_2020_margin_dem",
    "genballot_adjusted_margin_dem",
    "district_partisan_baseline_dem",
    "district_elasticity",
    "national_environment_margin_dem",
    "district_environment_adjustment_dem",
    "incumbency_adjustment_dem",
    "candidate_quality_adjustment_dem",
    "special_adjustment_dem",
    "fundamentals_margin_dem",
    "model_margin_dem",
    "dem_win_probability",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if "dem_candidate_is_incumbent" in df.columns:
    df["dem_candidate_is_incumbent_bool"] = normalize_bool_series(df["dem_candidate_is_incumbent"])
else:
    df["dem_candidate_is_incumbent_bool"] = False

if "gop_candidate_is_incumbent" in df.columns:
    df["gop_candidate_is_incumbent_bool"] = normalize_bool_series(df["gop_candidate_is_incumbent"])
else:
    df["gop_candidate_is_incumbent_bool"] = False

df["rating"] = df["dem_win_probability"].apply(race_rating_from_prob)
df["distance_to_50"] = (df["dem_win_probability"] - 0.5).abs()

# Prefer full simulation summary when available.
if not forecast_summary_output.empty:
    summary_row = forecast_summary_output.iloc[-1]
    expected_dem_seats = as_float(summary_row.get("expected_dem_seats"))
    median_like_dem_seats = as_float(summary_row.get("median_dem_seats"))
    dem_majority_probability = as_float(summary_row.get("dem_majority_probability"))
else:
    expected_dem_seats = df["dem_win_probability"].sum()
    median_like_dem_seats = int((df["dem_win_probability"] >= 0.5).sum())
    dem_majority_probability = np.nan

# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_ratings, tab_drivers, tab_manual_polls, tab_diagnostics = st.tabs(
    [
        "Overview",
        "District Ratings",
        "Model Drivers",
        "Manual Poll Entry",
        "Diagnostics",
    ]
)


# -----------------------------
# Overview
# -----------------------------
with tab_overview:
    st.subheader("Topline House Snapshot")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Expected Dem Seats", fmt_num(expected_dem_seats, 2))
    c2.metric("Median Dem Seats", fmt_num(median_like_dem_seats, 0))
    c3.metric("Dem Majority Odds", fmt_pct(dem_majority_probability))
    c4.metric(
        "National Environment",
        fmt_margin(df["national_environment_margin_dem"].dropna().iloc[0])
        if df["national_environment_margin_dem"].notna().any()
        else "—",
    )
    c5.metric("Districts", fmt_num(len(df), 0))

    st.caption(
        "Seat totals now come from the House simulation engine when outputs are available."
    )

    st.divider()

    if not seat_distribution_output.empty:
        st.subheader("Simulated Seat Distribution")
        fig_seats = px.bar(
            seat_distribution_output,
            x="dem_seats",
            y="probability",
            labels={
                "dem_seats": "Democratic seats",
                "probability": "Probability",
            },
            title="House Democratic Seat Distribution",
        )
        fig_seats.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig_seats, use_container_width=True)

    st.divider()

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Rating Distribution")

        rating_order = [
            "Safe D",
            "Likely D",
            "Lean D",
            "Tilt D",
            "Toss-up",
            "Tilt R",
            "Lean R",
            "Likely R",
            "Safe R",
            "Unknown",
        ]

        rating_counts = (
            df["rating"]
            .value_counts()
            .reindex(rating_order)
            .fillna(0)
            .astype(int)
            .reset_index()
        )
        rating_counts.columns = ["Rating", "Districts"]

        fig = px.bar(
            rating_counts,
            x="Rating",
            y="Districts",
            title="Districts by Rating",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(rating_counts, use_container_width=True, hide_index=True)

    with right:
        st.subheader("Most Competitive Districts")

        comp = df.sort_values("distance_to_50").head(20).copy()

        comp_rows = []
        for _, row in comp.iterrows():
            comp_rows.append(
                {
                    "District": row.get("district_id", ""),
                    "Rating": row.get("rating", ""),
                    "Dem candidate": row.get("dem_candidate", ""),
                    "GOP candidate": row.get("gop_candidate", ""),
                    "Dem odds": fmt_pct(row.get("dem_win_probability")),
                    "Model margin": fmt_margin(row.get("model_margin_dem")),
                    "Baseline": fmt_margin(row.get("district_partisan_baseline_dem")),
                }
            )

        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("National Environment Source")

    if env.empty:
        st.info("No House national environment audit file found. Run `python3 recalculate_house_fundamentals.py`.")
    else:
        env_show_cols = [
            "as_of_date",
            "generic_ballot_margin_dem",
            "presidential_approval",
            "presidential_disapproval",
            "presidential_net_approval",
            "approval_adjustment_dem",
            "midterm_adjustment_dem",
            "national_environment_margin_dem",
            "national_environment_source_path",
            "source_notes",
        ]
        env_show_cols = [c for c in env_show_cols if c in env.columns]
        st.dataframe(env[env_show_cols].tail(1), use_container_width=True, hide_index=True)


# -----------------------------
# District Ratings
# -----------------------------
with tab_ratings:
    st.subheader("District Ratings")

    c_filter1, c_filter2, c_filter3 = st.columns(3)

    with c_filter1:
        states = sorted(df["state"].dropna().unique().tolist())
        selected_states = st.multiselect(
            "Filter by state",
            states,
            default=[],
        )

    with c_filter2:
        selected_ratings = st.multiselect(
            "Filter by rating",
            [
                "Safe D",
                "Likely D",
                "Lean D",
                "Tilt D",
                "Toss-up",
                "Tilt R",
                "Lean R",
                "Likely R",
                "Safe R",
                "Unknown",
            ],
            default=[],
        )

    with c_filter3:
        sort_mode = st.selectbox(
            "Sort by",
            [
                "Competitiveness",
                "Dem win probability",
                "State/District",
                "Model margin",
            ],
        )

    view = df.copy()

    if selected_states:
        view = view[view["state"].isin(selected_states)]

    if selected_ratings:
        view = view[view["rating"].isin(selected_ratings)]

    if sort_mode == "Competitiveness":
        view = view.sort_values("distance_to_50")
    elif sort_mode == "Dem win probability":
        view = view.sort_values("dem_win_probability", ascending=False)
    elif sort_mode == "Model margin":
        view = view.sort_values("model_margin_dem", ascending=False)
    else:
        view = view.sort_values(["state", "district"])

    chart_view = view.copy()
    chart_view["Dem win probability"] = chart_view["dem_win_probability"]
    chart_view["District"] = chart_view["district_id"]

    st.subheader("Democratic Win Probability")

    chart_limit = st.slider(
        "Number of districts to chart",
        min_value=25,
        max_value=435,
        value=min(75, len(chart_view)),
        step=25,
    )

    chart_df = chart_view.head(chart_limit)

    fig = px.bar(
        chart_df,
        x="Dem win probability",
        y="District",
        orientation="h",
        hover_data=[
            "rating",
            "dem_candidate",
            "gop_candidate",
            "model_margin_dem",
            "district_partisan_baseline_dem",
        ],
        title="Democratic Win Probability by District",
    )
    fig.update_layout(
        xaxis_tickformat=".0%",
        yaxis={"categoryorder": "total ascending"},
        height=max(600, chart_limit * 14),
    )
    st.plotly_chart(fig, use_container_width=True)

    table_rows = []
    for _, row in view.iterrows():
        table_rows.append(
            {
                "District": row.get("district_id", ""),
                "Rating": row.get("rating", ""),
                "Dem candidate": row.get("dem_candidate", ""),
                "GOP candidate": row.get("gop_candidate", ""),
                "Incumbent party": row.get("inferred_incumbent_party", ""),
                "Dem incumbent": bool(row.get("dem_candidate_is_incumbent_bool", False)),
                "GOP incumbent": bool(row.get("gop_candidate_is_incumbent_bool", False)),
                "Dem odds": fmt_pct(row.get("dem_win_probability")),
                "Model margin": fmt_margin(row.get("model_margin_dem")),
                "Region": row.get("region", ""),
                "District type": row.get("district_type", ""),
                "College Share Tier": row.get("college_share_tier", ""),
                "White Share Tier": row.get("white_share_tier", ""),
                "Black Share Tier": row.get("black_share_tier", ""),
                "Hispanic Share Tier": row.get("hispanic_share_tier", ""),
                "Median Income Tier": row.get("median_income_tier", ""),
                "Baseline": fmt_margin(row.get("district_partisan_baseline_dem")),
                "National effect": fmt_margin(row.get("district_environment_adjustment_dem")),
                "State env.": fmt_margin(row.get("state_environment_adjustment_dem")),
                "Incumbency": fmt_margin(row.get("incumbency_adjustment_dem")),
            }
        )

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


# -----------------------------
# Model Drivers
# -----------------------------
with tab_drivers:
    st.subheader("Fundamentals Breakdown")

    key_default = (
        df.sort_values("distance_to_50")
        .head(25)["district_id"]
        .tolist()
    )

    selected_districts = st.multiselect(
        "Districts to show",
        options=df["district_id"].tolist(),
        default=key_default,
    )

    audit = df[df["district_id"].isin(selected_districts)].copy()

    audit = audit.sort_values("distance_to_50")

    rows = []
    for _, row in audit.iterrows():
        rows.append(
            {
                "District": row.get("district_id", ""),
                "Dem candidate": row.get("dem_candidate", ""),
                "GOP candidate": row.get("gop_candidate", ""),
                "2024 pres. margin": fmt_margin(row.get("pres_2024_margin_dem")),
                "2020 pres. margin": fmt_margin(row.get("pres_2020_margin_dem")),
                "Region": row.get("region", ""),
                "District type": row.get("district_type", ""),
                "College Share Tier": row.get("college_share_tier", ""),
                "White Share Tier": row.get("white_share_tier", ""),
                "Black Share Tier": row.get("black_share_tier", ""),
                "Hispanic Share Tier": row.get("hispanic_share_tier", ""),
                "Median Income Tier": row.get("median_income_tier", ""),
                "Education/Race Group": row.get("education_race_error_group", ""),
                "Demographic Group": row.get("demographic_error_group", ""),
                "District baseline": fmt_margin(row.get("district_partisan_baseline_dem")),
                "Elasticity": fmt_num(row.get("district_elasticity"), 2),
                "Type base elastic.": fmt_num(row.get("district_type_elasticity_base"), 2),
                "Baseline elastic adj.": fmt_num(row.get("partisan_baseline_elasticity_adjustment"), 2),
                "Region elastic adj.": fmt_num(row.get("region_elasticity_adjustment"), 2),
                "Nat'l env. effect": fmt_margin(row.get("district_environment_adjustment_dem")),
                "State env.": fmt_margin(row.get("state_environment_adjustment_dem")),
                "Incumbency": fmt_margin(row.get("incumbency_adjustment_dem")),
                "Candidate quality": fmt_margin(row.get("candidate_quality_adjustment_dem")),
                "Special adj.": fmt_margin(row.get("special_adjustment_dem")),
                "Fundamentals": fmt_margin(row.get("fundamentals_margin_dem")),
                "Model margin": fmt_margin(row.get("model_margin_dem")),
                "Dem odds": fmt_pct(row.get("dem_win_probability")),
                "Notes": row.get("race_notes", ""),
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("Formula notes", expanded=False):
        st.markdown(
            """
            **Current first-pass House fundamentals formula**

            `district_partisan_baseline_dem = 0.70 × 2024 presidential margin + 0.30 × 2020 presidential margin`

            `fundamentals_margin_dem = district baseline + national environment × district elasticity + state environment adjustment + incumbency + candidate quality + special adjustment`

            The current win probability is a simple logistic conversion of model margin. A full correlated House simulation has not been added yet.
            """
        )

    with st.expander("Full House input table", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)


# -----------------------------
# Diagnostics
# -----------------------------
with tab_diagnostics:
    st.subheader("Model Diagnostics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", fmt_num(len(df), 0))
    c2.metric("Missing baselines", fmt_num(df["district_partisan_baseline_dem"].isna().sum(), 0))
    c3.metric("Dem incumbents detected", fmt_num(df["dem_candidate_is_incumbent_bool"].sum(), 0))
    c4.metric("GOP incumbents detected", fmt_num(df["gop_candidate_is_incumbent_bool"].sum(), 0))

    st.divider()

    st.subheader("Simulation Error Structure")

    if not forecast_summary_output.empty:
        srow = forecast_summary_output.iloc[-1]
        e1, e2, e3, e4, e5, e6 = st.columns(6)
        e1.metric("National Error SD", fmt_num(srow.get("national_error_sd"), 2))
        e2.metric("State Error SD", fmt_num(srow.get("state_error_sd"), 2))
        e3.metric("Region Error SD", fmt_num(srow.get("region_error_sd"), 2))
        e4.metric("District Type Error SD", fmt_num(srow.get("district_type_error_sd"), 2))
        e5.metric("Education/Race Error SD", fmt_num(srow.get("education_race_error_sd"), 2))
        e6.metric("District Floor SD", fmt_num(srow.get("district_specific_error_sd_floor"), 2))

        g1, g2, g3, g4, g5 = st.columns(5)
        g1.metric("State Groups", fmt_num(srow.get("state_error_groups"), 0))
        g2.metric("Region Groups", fmt_num(srow.get("region_error_groups"), 0))
        g3.metric("District Type Groups", fmt_num(srow.get("district_type_error_groups"), 0))
        g4.metric("Education/Race Groups", fmt_num(srow.get("education_race_error_groups"), 0))
        g5.metric("Demographic Groups", fmt_num(srow.get("demographic_error_groups"), 0))
    else:
        st.info("No House forecast summary found. Run the House pipeline.")

    st.divider()

    st.subheader("Group Counts")

    group_rows = []
    for col, label in [
        ("region_error_group", "Region"),
        ("district_type_error_group", "District Type"),
        ("education_race_error_group", "Education/Race"),
        ("demographic_error_group", "Full Demographic"),
        ("state_error_group", "State"),
    ]:
        if col in df.columns:
            counts = df[col].fillna("Unknown").astype(str).value_counts().reset_index()
            counts.columns = [label, "Districts"]
            group_rows.append((label, counts))

    for label, counts in group_rows:
        with st.expander(f"{label} group counts", expanded=False):
            st.dataframe(counts, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Elasticity Diagnostics")

    if "district_elasticity" in df.columns:
        e_left, e_right = st.columns(2)

        with e_left:
            elastic_by_type = (
                df.groupby("district_type")["district_elasticity"]
                .agg(["count", "mean", "min", "max"])
                .round(3)
                .reset_index()
            )
            st.markdown("**By District Type**")
            st.dataframe(elastic_by_type, use_container_width=True, hide_index=True)

        with e_right:
            elastic_by_region = (
                df.groupby("region")["district_elasticity"]
                .agg(["count", "mean", "min", "max"])
                .round(3)
                .reset_index()
            )
            st.markdown("**By Region**")
            st.dataframe(elastic_by_region, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("File Status")

    files = [
        HOUSE_INPUTS,
        NATIONAL_ENV_AUDIT,
        Path("House Model Data.xlsx"),
        INPUTS / "House Model Data.xlsx",
    ]

    status_rows = []
    for f in files:
        status_rows.append(
            {
                "File": str(f),
                "Exists": f.exists(),
                "Size KB": fmt_num(f.stat().st_size / 1024, 1) if f.exists() else "—",
            }
        )

    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Potential Data Issues")

    issues = []

    missing_baseline = df[df["district_partisan_baseline_dem"].isna()]
    if not missing_baseline.empty:
        issues.append(
            {
                "Issue": "Missing district baseline",
                "Count": len(missing_baseline),
                "Examples": ", ".join(missing_baseline["district_id"].head(10).tolist()),
            }
        )

    no_candidates = df[
        df["dem_candidate"].fillna("").astype(str).str.strip().eq("")
        & df["gop_candidate"].fillna("").astype(str).str.strip().eq("")
    ]

    if not no_candidates.empty:
        issues.append(
            {
                "Issue": "No candidates listed yet",
                "Count": len(no_candidates),
                "Examples": ", ".join(no_candidates["district_id"].head(10).tolist()),
            }
        )

    both_incumbent = df[
        df["dem_candidate_is_incumbent_bool"]
        & df["gop_candidate_is_incumbent_bool"]
    ]

    if not both_incumbent.empty:
        issues.append(
            {
                "Issue": "Both candidates marked incumbent",
                "Count": len(both_incumbent),
                "Examples": ", ".join(both_incumbent["district_id"].head(10).tolist()),
            }
        )

    if issues:
        st.dataframe(pd.DataFrame(issues), use_container_width=True, hide_index=True)
    else:
        st.success("No obvious data issues detected.")

    st.divider()

    with st.expander("Raw House race inputs", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)

# -----------------------------
# Manual Poll Entry
# -----------------------------
with tab_manual_polls:
    st.subheader("Manual House Poll Entry")

    st.caption(
        "Add, edit, or delete manually entered House district polls. "
        "Polls are saved to inputs/house_manual_polls.csv. "
        "Later these will feed the House Bayesian polling blend."
    )

    manual_poll_path = INPUTS / "house_manual_polls.csv"

    house_poll_columns = [
        "race",
        "state",
        "district",
        "district_id",
        "pollster",
        "pollster_grade",
        "house_effect_dem",
        "start_date",
        "end_date",
        "sample_size",
        "sample_type",
        "dem_candidate",
        "gop_candidate",
        "ind_candidate",
        "other_candidate",
        "dem_pct",
        "gop_pct",
        "ind_pct",
        "other_pct",
        "undecided_pct",
        "notes",
    ]

    numeric_poll_columns = [
        "house_effect_dem",
        "sample_size",
        "dem_pct",
        "gop_pct",
        "ind_pct",
        "other_pct",
        "undecided_pct",
    ]

    existing_house_polls = read_csv_safe(manual_poll_path)

    if existing_house_polls.empty:
        existing_house_polls = pd.DataFrame(columns=house_poll_columns)

    for col in house_poll_columns:
        if col not in existing_house_polls.columns:
            existing_house_polls[col] = ""

    existing_house_polls = existing_house_polls[house_poll_columns].copy()

    for col in numeric_poll_columns:
        if col in existing_house_polls.columns:
            existing_house_polls[col] = pd.to_numeric(
                existing_house_polls[col],
                errors="coerce"
            )

    st.markdown("### Edit or Delete Existing Polls")

    st.caption(
        "Edit cells directly in the table. To delete a poll, check the Delete box. "
        "Then click Save Edits / Delete Marked Polls."
    )

    editable = existing_house_polls.copy()
    editable.insert(0, "delete", False)
    editable.insert(1, "row_id", range(1, len(editable) + 1))

    edited = st.data_editor(
        editable,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key="house_manual_poll_editor",
        column_config={
            "delete": st.column_config.CheckboxColumn(
                "Delete",
                help="Check this box and click Save Edits / Delete Marked Polls to delete the row.",
                default=False,
            ),
            "row_id": st.column_config.NumberColumn(
                "Row",
                disabled=True,
            ),
            "pollster_grade": st.column_config.SelectboxColumn(
                "Pollster grade",
                options=["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "Unknown"],
            ),
            "sample_type": st.column_config.SelectboxColumn(
                "Sample type",
                options=["LV", "RV", "A", "Other"],
            ),
            "house_effect_dem": st.column_config.NumberColumn(
                "House effect Dem",
                step=0.5,
                format="%.1f",
            ),
            "sample_size": st.column_config.NumberColumn(
                "Sample size",
                step=1,
                min_value=1,
            ),
            "dem_pct": st.column_config.NumberColumn("Dem %", step=0.1, format="%.1f"),
            "gop_pct": st.column_config.NumberColumn("GOP %", step=0.1, format="%.1f"),
            "ind_pct": st.column_config.NumberColumn("Independent %", step=0.1, format="%.1f"),
            "other_pct": st.column_config.NumberColumn("Other %", step=0.1, format="%.1f"),
            "undecided_pct": st.column_config.NumberColumn("Undecided %", step=0.1, format="%.1f"),
        },
    )

    save_edits = st.button(
        "Save Edits / Delete Marked Polls",
        type="primary",
        key="save_house_manual_poll_edits",
    )

    if save_edits:
        updated = edited.copy()

        if "delete" in updated.columns:
            updated = updated[updated["delete"] != True]

        for col in ["delete", "row_id"]:
            if col in updated.columns:
                updated = updated.drop(columns=[col])

        for col in house_poll_columns:
            if col not in updated.columns:
                updated[col] = ""

        updated = updated[house_poll_columns].copy()

        updated["state"] = updated["state"].fillna("").astype(str).str.strip().str.upper()
        updated["district"] = updated.apply(
            lambda row: normalize_district_value(row.get("state", ""), row.get("district", "")),
            axis=1
        )
        updated["district_id"] = updated.apply(
            lambda row: normalize_district_id(row.get("state", ""), row.get("district", "")),
            axis=1
        )

        for col in numeric_poll_columns:
            updated[col] = pd.to_numeric(
                updated[col],
                errors="coerce"
            ).fillna(0.0)

        # Remove fully blank rows.
        nonblank_mask = updated[["race", "state", "district", "pollster", "dem_candidate", "gop_candidate"]].fillna("").astype(str).agg(
            lambda row: any(x.strip() for x in row),
            axis=1,
        )
        updated = updated[nonblank_mask].copy()

        manual_poll_path.parent.mkdir(parents=True, exist_ok=True)

        if manual_poll_path.exists():
            backup_path = manual_poll_path.with_suffix(".csv.bak")
            existing_house_polls.to_csv(backup_path, index=False)

        updated.to_csv(manual_poll_path, index=False)

        st.success(
            f"Saved {len(updated)} House manual polls to {manual_poll_path}. "
            "Run the House pipeline after poll ingestion is connected."
        )

        st.dataframe(updated, use_container_width=True, hide_index=True)

    st.divider()

    st.markdown("### Add New District Poll")

    available_districts = []
    district_candidate_lookup = {}

    if "district_id" in df.columns:
        available_districts = sorted(df["district_id"].dropna().astype(str).unique().tolist())

        for _, row in df.iterrows():
            did = str(row.get("district_id", "")).strip()
            if did:
                district_candidate_lookup[did] = {
                    "state": str(row.get("state", "")).strip().upper(),
                    "district": str(row.get("district", "")).strip(),
                    "dem_candidate": str(row.get("dem_candidate", "") if pd.notna(row.get("dem_candidate", "")) else "").strip(),
                    "gop_candidate": str(row.get("gop_candidate", "") if pd.notna(row.get("gop_candidate", "")) else "").strip(),
                }

    with st.form("house_manual_poll_entry_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            if available_districts:
                selected_district = st.selectbox(
                    "District",
                    available_districts,
                    index=0,
                    help="District ID from house_race_inputs.csv."
                )
                selected_info = district_candidate_lookup.get(selected_district, {})
                default_state = selected_info.get("state", "")
                default_district_number = selected_info.get("district", "")
                default_dem_candidate = selected_info.get("dem_candidate", "")
                default_gop_candidate = selected_info.get("gop_candidate", "")
            else:
                selected_district = ""
                default_state = ""
                default_district_number = ""
                default_dem_candidate = ""
                default_gop_candidate = ""

            state = st.text_input("State abbreviation", value=default_state)
            district_number = st.text_input("District number", value=default_district_number)
            district_id = st.text_input(
                "District ID",
                value=selected_district or (state.strip().upper() + "-" + district_number.strip()),
            )
            race = st.text_input(
                "Race",
                value=(district_id + " House").strip() if district_id else "",
            )
            pollster = st.text_input("Pollster", value="")
            pollster_grade = st.selectbox(
                "Pollster grade",
                ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "Unknown"],
                index=4,
            )
            house_effect_dem = st.number_input(
                "House effect Dem",
                value=0.0,
                step=0.5,
                help="Positive means pollster is Dem-leaning; this amount is subtracted from the poll margin.",
            )

        with c2:
            start_date = st.date_input("Start date", key="house_poll_start_date")
            end_date = st.date_input("End date", key="house_poll_end_date")
            sample_size = st.number_input("Sample size", min_value=1, value=500, step=1)
            sample_type = st.selectbox("Sample type", ["LV", "RV", "A", "Other"], index=0)
            dem_candidate = st.text_input("Dem candidate", value=default_dem_candidate)
            gop_candidate = st.text_input("GOP candidate", value=default_gop_candidate)

        with c3:
            ind_candidate = st.text_input("Independent candidate", value="")
            other_candidate = st.text_input("Other candidate", value="")
            dem_pct = st.number_input("Dem %", value=0.0, step=0.1)
            gop_pct = st.number_input("GOP %", value=0.0, step=0.1)
            ind_pct = st.number_input("Independent %", value=0.0, step=0.1)
            other_pct = st.number_input("Other %", value=0.0, step=0.1)
            undecided_pct = st.number_input("Undecided %", value=0.0, step=0.1)

        notes = st.text_area("Notes", value="")

        submitted = st.form_submit_button("Save New House Poll")

        if submitted:
            clean_state = state.strip().upper()
            clean_district = normalize_district_value(clean_state, district_number)
            clean_district_id = normalize_district_id(clean_state, clean_district)

            new_row = {
                "race": race,
                "state": clean_state,
                "district": clean_district,
                "district_id": clean_district_id,
                "pollster": pollster,
                "pollster_grade": pollster_grade,
                "house_effect_dem": house_effect_dem,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "sample_size": sample_size,
                "sample_type": sample_type,
                "dem_candidate": dem_candidate,
                "gop_candidate": gop_candidate,
                "ind_candidate": ind_candidate,
                "other_candidate": other_candidate,
                "dem_pct": dem_pct,
                "gop_pct": gop_pct,
                "ind_pct": ind_pct,
                "other_pct": other_pct,
                "undecided_pct": undecided_pct,
                "notes": notes,
            }

            updated = pd.concat(
                [
                    existing_house_polls,
                    pd.DataFrame([new_row])
                ],
                ignore_index=True
            )

            manual_poll_path.parent.mkdir(parents=True, exist_ok=True)
            updated.to_csv(manual_poll_path, index=False)

            st.success(f"Saved new House poll to {manual_poll_path}.")
            st.dataframe(updated.tail(10), use_container_width=True, hide_index=True)

    st.divider()

    st.markdown("### Next Step")

    st.code("python3 run_house_full_pipeline.py", language="bash")

    st.caption(
        "Poll ingestion is the next infrastructure step. For now, this tab creates and manages the poll CSV."
    )
