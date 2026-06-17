from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

DEM_COLOR = "#1f77b4"
GOP_COLOR = "#d62728"
HOUSE_CONTROL_THRESHOLD = 218


INPUTS = Path("inputs")
OUTPUTS = Path("outputs")

HOUSE_INPUTS = INPUTS / "house_race_inputs.csv"
NATIONAL_ENV_AUDIT = INPUTS / "house_national_environment_audit.csv"

HOUSE_RACE_STATS = OUTPUTS / "house_race_stats.csv"
HOUSE_SEAT_DISTRIBUTION = OUTPUTS / "house_seat_distribution.csv"
HOUSE_FORECAST_SUMMARY = OUTPUTS / "house_forecast_summary.csv"
HOUSE_FORECAST_HISTORY = OUTPUTS / "house_forecast_history.csv"
HOUSE_CALIBRATION_AUDIT = OUTPUTS / "house_calibration_audit.csv"
HOUSE_LOCAL_CONTEXT_AUDIT = OUTPUTS / "house_local_context_audit.csv"

st.set_page_config(
    page_title="2026 House Forecast Dashboard",
    layout="wide",
)


# -----------------------------
# Helpers
# -----------------------------


def get_first_available(row, names, default=None):
    for name in names:
        try:
            value = row.get(name)
        except Exception:
            value = None

        if value is not None and not pd.isna(value):
            return value

    return default

def rating_party_bucket(rating):
    s = str(rating).strip().lower()

    if "safe d" in s or "likely d" in s or "lean d" in s or "tilt d" in s:
        return "Democratic"

    if "safe r" in s or "likely r" in s or "lean r" in s or "tilt r" in s:
        return "Republican"

    if "toss" in s:
        return "Toss-Up"

    return "Other"

def add_party_bar_color_columns(df):
    """
    Adds display columns used by Plotly/Altair race bars.
    Red means Republican is favored; blue means Democrat is favored.
    """
    out = df.copy()

    if "dem_win_probability" in out.columns:
        out["favored_party"] = out["dem_win_probability"].apply(
            lambda p: "Democrat" if float(p) >= 0.5 else "Republican"
        )
    elif "simulated_dem_win_probability" in out.columns:
        out["favored_party"] = out["simulated_dem_win_probability"].apply(
            lambda p: "Democrat" if float(p) >= 0.5 else "Republican"
        )
    else:
        out["favored_party"] = "Unknown"

    return out

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
forecast_history_output = read_csv_safe(HOUSE_FORECAST_HISTORY)

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
    dem_majority_probability = as_float(
        summary_row.get(
            "dem_majority_probability",
            summary_row.get("dem_control_probability"),
        )
    )
else:
    expected_dem_seats = df["dem_win_probability"].sum()
    median_like_dem_seats = int((df["dem_win_probability"] >= 0.5).sum())
    dem_majority_probability = np.nan



def margin_label_from_dem_margin(margin):
    try:
        margin = float(margin)
    except Exception:
        return "—"

    if abs(margin) < 0.05:
        return "Even"

    if margin > 0:
        return f"D+{margin:.1f}"

    return f"R+{abs(margin):.1f}"


def party_leader_from_margin(margin):
    try:
        margin = float(margin)
    except Exception:
        return "Unknown"

    if margin > 0.05:
        return "Democrat"

    if margin < -0.05:
        return "Republican"

    return "Even"


def style_margin_table(df):
    def row_style(row):
        margin = row.get("model_margin_dem", 0)

        try:
            margin = float(margin)
        except Exception:
            margin = 0

        if margin > 0.05:
            return ["background-color: rgba(30, 100, 220, 0.18)"] * len(row)

        if margin < -0.05:
            return ["background-color: rgba(220, 60, 60, 0.18)"] * len(row)

        return ["background-color: rgba(160, 160, 160, 0.14)"] * len(row)

    return df.style.apply(row_style, axis=1)


def render_candidate_war_visibility():
    st.header("Candidate WAR")

    st.caption(
        "Candidate WAR is an empirical candidate-quality layer based on how candidates performed "
        "relative to partisan baseline in prior elections. Both-candidate matches receive the full "
        "shrunk/capped adjustment; one-sided matches receive the configured one-sided discount."
    )

    war_path = OUTPUTS / "house_candidate_war_audit.csv"
    unmatched_path = OUTPUTS / "house_candidate_war_unmatched_competitive_review.csv"

    if not war_path.exists():
        st.info("No candidate WAR audit found yet. Run `python3 build_house_candidate_war.py`.")
        return

    war = pd.read_csv(war_path)

    if war.empty:
        st.info("Candidate WAR audit exists but is empty.")
        return

    settings_path = INPUTS / "house_calibration_settings.csv"
    war_active = "Unknown"
    shrinkage = "—"
    cap = "—"
    one_sided = "—"

    if settings_path.exists():
        try:
            settings = pd.read_csv(settings_path)

            def get_setting(name, default="—"):
                mask = settings["setting"].astype(str).str.strip().eq(name)
                if mask.any():
                    return settings.loc[mask, "value"].iloc[0]
                return default

            war_active = get_setting("use_candidate_war_adjustments", "0")
            shrinkage = get_setting("house_candidate_war_shrinkage", "—")
            cap = get_setting("house_candidate_war_cap", "—")
            one_sided = get_setting("house_candidate_war_one_sided_multiplier", "—")
        except Exception:
            pass

    for col in [
        "candidate_war_adjustment_dem",
        "candidate_war_adjustment_dem_before_match_quality",
        "candidate_war_match_quality_multiplier",
        "dem_candidate_war",
        "gop_candidate_war",
    ]:
        if col in war.columns:
            war[col] = pd.to_numeric(war[col], errors="coerce").fillna(0.0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("WAR active", str(war_active))
    m2.metric("Shrinkage", str(shrinkage))
    m3.metric("Cap", str(cap))
    m4.metric("One-sided multiplier", str(one_sided))

    st.divider()

    if "war_match_status" in war.columns:
        counts = war["war_match_status"].fillna("Unknown").value_counts().reset_index()
        counts.columns = ["Match status", "Districts"]

        c1, c2 = st.columns([1, 2])

        with c1:
            st.subheader("Match counts")
            st.dataframe(counts, use_container_width=True, hide_index=True)

        with c2:
            try:
                import plotly.express as px

                fig = px.bar(
                    counts,
                    x="Districts",
                    y="Match status",
                    orientation="h",
                    title="Candidate WAR match coverage",
                )
                fig.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

    st.divider()
    st.subheader("Largest WAR Adjustments")

    display_cols = [
        "district_id",
        "dem_candidate",
        "gop_candidate",
        "war_match_status",
        "candidate_war_adjustment_dem_before_match_quality",
        "candidate_war_match_quality_multiplier",
        "candidate_war_adjustment_dem",
        "dem_candidate_war",
        "gop_candidate_war",
        "dem_war_name",
        "gop_war_name",
    ]
    display_cols = [c for c in display_cols if c in war.columns]

    largest = war.reindex(
        war["candidate_war_adjustment_dem"].abs().sort_values(ascending=False).index
    )

    st.dataframe(
        largest[display_cols].head(75),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("WAR-Influenced Competitive Districts")

    race_path = OUTPUTS / "house_race_stats.csv"

    if race_path.exists():
        race = pd.read_csv(race_path)

        if "district_id" in race.columns:
            race["district_id"] = race["district_id"].astype(str).str.strip().str.upper()
            war["district_id"] = war["district_id"].astype(str).str.strip().str.upper()

            merged = race.merge(
                war[
                    [
                        c for c in [
                            "district_id",
                            "candidate_war_adjustment_dem",
                            "war_match_status",
                            "dem_candidate_war",
                            "gop_candidate_war",
                        ]
                        if c in war.columns
                    ]
                ],
                on="district_id",
                how="left",
                suffixes=("", "_war_audit"),
            )

            for col in ["dem_win_probability", "model_margin_dem", "candidate_war_adjustment_dem"]:
                if col in merged.columns:
                    merged[col] = pd.to_numeric(merged[col], errors="coerce")

            if "dem_win_probability" in merged.columns:
                merged["distance_from_50"] = (merged["dem_win_probability"] - 0.5).abs()
            else:
                merged["distance_from_50"] = 1

            war_competitive = merged[
                merged.get("candidate_war_adjustment_dem", pd.Series([0] * len(merged))).abs().gt(0)
                & merged["distance_from_50"].le(0.30)
            ].copy()

            war_competitive = war_competitive.sort_values(
                ["distance_from_50", "candidate_war_adjustment_dem"],
                ascending=[True, False],
            )

            comp_cols = [
                "district_id",
                "dem_candidate",
                "gop_candidate",
                "rating",
                "model_margin_dem",
                "dem_win_probability",
                "candidate_war_adjustment_dem",
                "war_match_status",
            ]
            comp_cols = [c for c in comp_cols if c in war_competitive.columns]

            st.dataframe(
                war_competitive[comp_cols].head(75),
                use_container_width=True,
                hide_index=True,
            )

    st.divider()
    st.subheader("Competitive Unmatched WAR Review")

    if unmatched_path.exists():
        unmatched = pd.read_csv(unmatched_path)

        if unmatched.empty:
            st.success("No unmatched competitive candidates found.")
        else:
            st.dataframe(
                unmatched.head(150),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("No competitive unmatched WAR review file found yet.")




def safe_float_for_display(value, default=None):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def probability_label(value):
    value = safe_float_for_display(value)

    if value is None:
        return "—"

    return f"{value:.1%}"


def signed_points_label(value):
    value = safe_float_for_display(value)

    if value is None:
        return "—"

    if abs(value) < 0.05:
        return "Even"

    if value > 0:
        return f"D+{value:.1f}"

    return f"R+{abs(value):.1f}"


def get_selected_value(row, col, default="—"):
    try:
        if col not in row.index:
            return default
        val = row.get(col)
        if pd.isna(val):
            return default
        if str(val).strip() == "":
            return default
        return val
    except Exception:
        return default


def build_district_component_breakdown(selected):
    component_specs = [
        ("Partisan baseline", "district_partisan_baseline_dem"),
        ("National environment / elasticity", "district_environment_adjustment_dem"),
        ("State adjustment", "state_environment_adjustment_dem"),
        ("Incumbency", "incumbency_adjustment_dem"),
        ("Candidate quality", "candidate_quality_adjustment_dem"),
        ("Candidate WAR", "candidate_war_adjustment_dem"),
        ("Special adjustment", "special_adjustment_dem"),
    ]

    rows = []

    for label, col in component_specs:
        value = safe_float_for_display(get_selected_value(selected, col, None), 0.0)

        rows.append(
            {
                "Component": label,
                "Column": col,
                "Dem margin contribution": value,
                "Display": signed_points_label(value),
            }
        )

    return pd.DataFrame(rows)


def render_district_detail_card(selected):
    district_id = get_selected_value(selected, "district_id")
    dem_candidate = get_selected_value(selected, "dem_candidate")
    gop_candidate = get_selected_value(selected, "gop_candidate")
    rating = get_selected_value(selected, "rating")
    model_margin = safe_float_for_display(get_selected_value(selected, "model_margin_dem", None))
    dem_prob = safe_float_for_display(get_selected_value(selected, "dem_win_probability", None))
    war_adj = safe_float_for_display(get_selected_value(selected, "candidate_war_adjustment_dem", None), 0.0)

    st.markdown(f"### {district_id} Detail Card")

    subtitle_parts = []

    if dem_candidate != "—":
        subtitle_parts.append(f"**D:** {dem_candidate}")

    if gop_candidate != "—":
        subtitle_parts.append(f"**R:** {gop_candidate}")

    if subtitle_parts:
        st.markdown(" &nbsp; | &nbsp; ".join(subtitle_parts))

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Model margin", signed_points_label(model_margin))
    c2.metric("Dem win probability", probability_label(dem_prob))
    c3.metric("Rating", str(rating))
    c4.metric("Candidate WAR", signed_points_label(war_adj))

    st.markdown("#### Margin Components")

    components = build_district_component_breakdown(selected)

    # Remove empty all-zero rows only if the column exists but has no meaningful impact.
    components_display = components.copy()

    try:
        import plotly.express as px

        chart_df = components.copy()
        chart_df["Leader"] = chart_df["Dem margin contribution"].apply(
            lambda x: "D benefit" if x > 0.05 else "R benefit" if x < -0.05 else "Neutral"
        )

        fig = px.bar(
            chart_df,
            x="Dem margin contribution",
            y="Component",
            orientation="h",
            color="Leader",
            color_discrete_map={
                "D benefit": "#2b6cb0",
                "R benefit": "#c53030",
                "Neutral": "#888888",
            },
            custom_data=["Component", "Display"],
            title=f"{district_id} model margin components",
        )

        fig.update_traces(
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Contribution: %{customdata[1]}"
                "<extra></extra>"
            )
        )

        fig.add_vline(x=0, line_width=1)
        fig.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=45, b=10),
            xaxis_title="Democratic margin contribution",
            yaxis_title="",
            legend_title_text="Effect",
        )

        fig.update_xaxes(tickformat="+.1f")

        st.plotly_chart(fig, use_container_width=True)

    except Exception:
        pass

    st.dataframe(
        components_display[["Component", "Display", "Dem margin contribution"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Polling / Context")

    context_cols = [
        "polling_margin_dem",
        "poll_count",
        "avg_poll_age_days",
        "total_poll_weight",
        "candidate_war_match_status",
        "district_elasticity",
        "region",
        "district_type",
        "general_election_party_structure",
    ]

    context_rows = []

    friendly = {
        "polling_margin_dem": "Polling margin",
        "poll_count": "Poll count",
        "avg_poll_age_days": "Average poll age",
        "total_poll_weight": "Total poll weight",
        "candidate_war_match_status": "WAR match status",
        "district_elasticity": "District elasticity",
        "region": "Region",
        "district_type": "District type",
        "general_election_party_structure": "Election structure",
    }

    for col in context_cols:
        if col not in selected.index:
            continue

        val = get_selected_value(selected, col)

        if col == "polling_margin_dem":
            val = signed_points_label(val)
        elif col in ["avg_poll_age_days", "total_poll_weight", "district_elasticity"]:
            num = safe_float_for_display(val)
            val = "—" if num is None else f"{num:.2f}"

        context_rows.append(
            {
                "Field": friendly.get(col, col),
                "Value": val,
            }
        )

    if context_rows:
        st.dataframe(
            pd.DataFrame(context_rows),
            use_container_width=True,
            hide_index=True,
        )



def render_district_margin_explorer():
    st.header("District Margin Explorer")

    st.caption(
        "Search, filter, and select any House district. Blue rows are Democratic-leading districts; "
        "red rows are Republican-leading districts. Margins are Democratic margin by convention."
    )

    race_path = OUTPUTS / "house_race_stats.csv"

    if not race_path.exists():
        st.info("No House race stats found. Run the House pipeline first.")
        return

    race = pd.read_csv(race_path)

    if race.empty or "district_id" not in race.columns:
        st.info("House race stats file exists but does not contain district_id.")
        return

    race["district_id"] = race["district_id"].astype(str).str.strip().str.upper()

    for col in [
        "model_margin_dem",
        "fundamentals_margin_dem",
        "dem_win_probability",
        "candidate_war_adjustment_dem",
        "poll_count",
    ]:
        if col in race.columns:
            race[col] = pd.to_numeric(race[col], errors="coerce")

    if "model_margin_dem" not in race.columns:
        st.info("house_race_stats.csv does not contain model_margin_dem.")
        return

    race["margin_label"] = race["model_margin_dem"].apply(margin_label_from_dem_margin)
    race["party_leader"] = race["model_margin_dem"].apply(party_leader_from_margin)

    if "state" not in race.columns:
        race["state"] = race["district_id"].str.extract(r"^([A-Z]{2})", expand=False)

    states = ["All"] + sorted(race["state"].dropna().astype(str).unique().tolist())
    ratings = ["All"]
    if "rating" in race.columns:
        ratings += sorted(race["rating"].dropna().astype(str).unique().tolist())

    f1, f2, f3, f4 = st.columns(4)

    selected_state = f1.selectbox("State", states, index=0, key="margin_state_filter")
    selected_rating = f2.selectbox("Rating", ratings, index=0, key="margin_rating_filter")
    selected_party = f3.selectbox(
        "Leader",
        ["All", "Democrat", "Republican", "Even"],
        index=0,
        key="margin_party_filter",
    )
    sort_mode = f4.selectbox(
        "Sort by",
        ["Closest races", "Most Democratic", "Most Republican", "District ID"],
        index=0,
        key="margin_sort_mode",
    )

    filtered = race.copy()

    if selected_state != "All":
        filtered = filtered[filtered["state"].astype(str).eq(selected_state)]

    if selected_rating != "All" and "rating" in filtered.columns:
        filtered = filtered[filtered["rating"].astype(str).eq(selected_rating)]

    if selected_party != "All":
        filtered = filtered[filtered["party_leader"].eq(selected_party)]

    if sort_mode == "Closest races":
        filtered = filtered.assign(_sort=filtered["model_margin_dem"].abs()).sort_values("_sort")
    elif sort_mode == "Most Democratic":
        filtered = filtered.sort_values("model_margin_dem", ascending=False)
    elif sort_mode == "Most Republican":
        filtered = filtered.sort_values("model_margin_dem", ascending=True)
    else:
        filtered = filtered.sort_values("district_id")

    st.divider()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Districts shown", len(filtered))
    m2.metric("D-leading", int((filtered["model_margin_dem"] > 0.05).sum()))
    m3.metric("R-leading", int((filtered["model_margin_dem"] < -0.05).sum()))
    m4.metric("Toss-up range", int((filtered["model_margin_dem"].abs() <= 1.0).sum()))

    st.subheader("Visual Margin Chart")

    chart_df = filtered.copy()

    # Keep chart readable by showing all when filtered, or closest 120 by default.
    if selected_state == "All" and selected_rating == "All" and selected_party == "All" and len(chart_df) > 120:
        chart_df = chart_df.assign(_abs=chart_df["model_margin_dem"].abs()).sort_values("_abs").head(120)
        st.caption("Showing the 120 closest districts in the chart. Use filters to inspect all districts in a state/category.")

    try:
        import plotly.express as px

        chart_df["leader_color"] = chart_df["party_leader"].map(
            {
                "Democrat": "Democrat-leading",
                "Republican": "Republican-leading",
                "Even": "Even",
            }
        )

        # Friendly display fields for hover labels.
        chart_df["District"] = chart_df["district_id"]
        chart_df["Model margin"] = chart_df["model_margin_dem"].apply(margin_label_from_dem_margin)
        chart_df["Leading party"] = chart_df["party_leader"]
        chart_df["Rating"] = chart_df["rating"] if "rating" in chart_df.columns else "—"

        if "dem_candidate" in chart_df.columns:
            chart_df["Democratic candidate"] = chart_df["dem_candidate"].fillna("—").astype(str)
        else:
            chart_df["Democratic candidate"] = "—"

        if "gop_candidate" in chart_df.columns:
            chart_df["Republican candidate"] = chart_df["gop_candidate"].fillna("—").astype(str)
        else:
            chart_df["Republican candidate"] = "—"

        if "dem_win_probability" in chart_df.columns:
            chart_df["Dem win probability"] = chart_df["dem_win_probability"].apply(
                lambda x: f"{float(x):.1%}" if pd.notna(x) else "—"
            )
        else:
            chart_df["Dem win probability"] = "—"

        if "candidate_war_adjustment_dem" in chart_df.columns:
            chart_df["Candidate WAR adjustment"] = chart_df["candidate_war_adjustment_dem"].apply(
                margin_label_from_dem_margin
            )
        else:
            chart_df["Candidate WAR adjustment"] = "—"

        custom_data_cols = [
            "District",
            "Model margin",
            "Leading party",
            "Rating",
            "Democratic candidate",
            "Republican candidate",
            "Dem win probability",
            "Candidate WAR adjustment",
        ]

        fig = px.bar(
            chart_df.sort_values("model_margin_dem"),
            x="model_margin_dem",
            y="district_id",
            orientation="h",
            color="leader_color",
            color_discrete_map={
                "Democrat-leading": "#2b6cb0",
                "Republican-leading": "#c53030",
                "Even": "#888888",
            },
            custom_data=custom_data_cols,
            title="House district model margins",
        )

        fig.update_traces(
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Model margin: %{customdata[1]}<br>"
                "Leading party: %{customdata[2]}<br>"
                "Rating: %{customdata[3]}<br>"
                "Democratic candidate: %{customdata[4]}<br>"
                "Republican candidate: %{customdata[5]}<br>"
                "Dem win probability: %{customdata[6]}<br>"
                "Candidate WAR adjustment: %{customdata[7]}"
                "<extra></extra>"
            )
        )

        fig.add_vline(x=0, line_width=1)
        fig.update_layout(
            height=max(500, min(2400, 22 * len(chart_df))),
            margin=dict(l=10, r=10, t=45, b=10),
            xaxis_title="Democratic margin",
            yaxis_title="District",
            legend_title_text="Leader",
            showlegend=True,
        )

        fig.update_xaxes(tickformat="+.1f")

        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Plotly chart unavailable. Showing table below.")

    st.divider()
    st.subheader("Select a District")

    district_options = filtered["district_id"].tolist()

    if district_options:
        selected_district = st.selectbox(
            "District",
            district_options,
            index=0,
            key="selected_margin_district",
        )

        selected = race[race["district_id"].eq(selected_district)].iloc[0]

        render_district_detail_card(selected)

    st.divider()
    st.subheader("Full District Margin Table")

    table_cols = [
        "district_id",
        "state",
        "dem_candidate",
        "gop_candidate",
        "margin_label",
        "model_margin_dem",
        "dem_win_probability",
        "rating",
        "candidate_war_adjustment_dem",
        "candidate_war_match_status",
        "poll_count",
    ]
    table_cols = [c for c in table_cols if c in filtered.columns]

    table = filtered[table_cols].copy()

    st.dataframe(
        style_margin_table(table),
        use_container_width=True,
        hide_index=True,
    )

    csv = table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered district margins",
        csv,
        file_name="house_district_margin_explorer_filtered.csv",
        mime="text/csv",
        key="download_district_margin_explorer",
    )



# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_ratings, tab_drivers, tab_margins, tab_war, tab_manual_polls, tab_local_context, tab_diagnostics = st.tabs(
    [
        "Overview",
        "Race Ratings",
        "Model Drivers",
        "District Margins",
        "Candidate WAR",
        "Manual Polls",
        "Local Context Audit",
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

    st.divider()

    if not seat_distribution_output.empty:
        st.subheader("Simulated Seat Distribution")
        seat_distribution_output = seat_distribution_output.copy()
        seat_distribution_output["Control"] = seat_distribution_output["dem_seats"].apply(
            lambda x: "Democratic House" if float(x) >= HOUSE_CONTROL_THRESHOLD else "Republican House"
        )

        fig_seats = px.bar(
            seat_distribution_output,
            x="dem_seats",
            y="probability",
            color="Control",
            color_discrete_map={
                "Democratic House": DEM_COLOR,
                "Republican House": GOP_COLOR,
            },
            labels={
                "dem_seats": "Democratic seats",
                "probability": "Probability",
            },
            title="House Democratic Seat Distribution",
        )
        fig_seats.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig_seats, use_container_width=True)

    st.divider()

    st.subheader("Model Odds Over Time")

    if forecast_history_output.empty:
        st.info("No forecast history yet. Run the House full pipeline to start building the time series.")
    else:
        history = forecast_history_output.copy()

        if "timestamp" in history.columns:
            history["timestamp"] = pd.to_datetime(history["timestamp"], errors="coerce")
            history = history.dropna(subset=["timestamp"]).sort_values("timestamp")
            history["Run"] = history["timestamp"].dt.strftime("%b %-d, %I:%M %p")
        elif "run_date" in history.columns:
            history["Run"] = history["run_date"].astype(str)
        else:
            history["Run"] = range(1, len(history) + 1)

        chart_cols = []

        if "dem_control_probability" in history.columns:
            history["Dem majority odds"] = pd.to_numeric(
                history["dem_control_probability"],
                errors="coerce",
            ) * 100
            chart_cols.append("Dem majority odds")

        if "expected_dem_seats" in history.columns:
            history["Expected Dem seats"] = pd.to_numeric(
                history["expected_dem_seats"],
                errors="coerce",
            )

        if "median_dem_seats" in history.columns:
            history["Median Dem seats"] = pd.to_numeric(
                history["median_dem_seats"],
                errors="coerce",
            )

        if chart_cols:
            fig_history = px.line(
                history,
                x="Run",
                y="Dem majority odds",
                markers=True,
                labels={
                    "Run": "Run",
                    "Dem majority odds": "Dem majority odds (%)",
                },
                title="House Democratic Majority Odds Over Time",
            )
            fig_history.update_layout(yaxis_ticksuffix="%", yaxis_range=[0, 100])
            st.plotly_chart(fig_history, use_container_width=True)
        else:
            st.info("Forecast history exists, but no dem_control_probability column was found.")

        with st.expander("Forecast history table"):
            display_cols = [
                "timestamp",
                "days_out",
                "expected_dem_seats",
                "median_dem_seats",
                "dem_control_probability",
                "national_environment",
                "total_error_sd",
                "uncertainty_engine",
            ]
            display_cols = [c for c in display_cols if c in history.columns]
            st.dataframe(history[display_cols].tail(25), use_container_width=True, hide_index=True)


    st.divider()

    if not forecast_summary_output.empty:
        srow = forecast_summary_output.iloc[-1]
        engine = str(srow.get("uncertainty_engine", "")).strip()

        if engine:
            st.subheader("Dynamic Uncertainty")
            st.caption(f"Uncertainty engine: {engine}")

            u1, u2, u3, u4, u5 = st.columns(5)
            u1.metric("Days Out", fmt_num(srow.get("days_out"), 0))
            u2.metric("National SD", fmt_num(srow.get("national_error_sd"), 2))
            u3.metric("Region SD", fmt_num(srow.get("region_error_sd"), 2))
            u4.metric("Demographic SD", fmt_num(srow.get("demographic_error_sd"), 2))
            u5.metric("District SD", fmt_num(srow.get("district_error_sd"), 2))

            v1, v2, v3, v4 = st.columns(4)
            v1.metric("Total Error SD", fmt_num(srow.get("total_error_sd"), 2))
            v2.metric("Implied Correlation", fmt_pct(srow.get("implied_correlation")))
            v3.metric("Region Groups", fmt_num(srow.get("region_groups"), 0))
            v4.metric("Demographic Groups", fmt_num(srow.get("demographic_groups"), 0))


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

        rating_counts["Rating Party"] = rating_counts["Rating"].apply(rating_party_bucket)

        fig = px.bar(
            rating_counts,
            x="Rating",
            y="Districts",
            color="Rating Party",
            color_discrete_map={
                "Democratic": DEM_COLOR,
                "Republican": GOP_COLOR,
                "Toss-Up": "#808080",
                "Other": "#808080",
            },
            title="Districts by Rating",
        )
        fig.update_layout(showlegend=True)
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

    chart_df = chart_view.head(chart_limit).copy()
    chart_df["Favored Party"] = chart_df["Dem win probability"].apply(
        lambda p: "Democrat" if float(p) >= 0.5 else "Republican"
    )

    fig = px.bar(
        chart_df,
        x="Dem win probability",
        y="District",
        orientation="h",
        color="Favored Party",
        color_discrete_map={
            "Democrat": DEM_COLOR,
            "Republican": GOP_COLOR,
        },
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
                "Election System": row.get("election_system", ""),
                "Party Structure": row.get("general_election_party_structure", ""),
                "Party Override": row.get("party_control_override", ""),
                "Fixed Control": row.get("party_control_fixed", ""),
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
                "Other": row.get("other_candidate", ""),
                "Election System": row.get("election_system", ""),
                "Party Structure": row.get("general_election_party_structure", ""),
                "Party Override": row.get("party_control_override", ""),
                "Fixed Control": row.get("party_control_fixed", ""),
                "Election Notes": row.get("election_system_notes", ""),
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


with tab_local_context:
    st.header("Local Context Audit")

    audit_path = OUTPUTS / "house_local_context_audit.csv"

    if not audit_path.exists():
        st.info("No local context audit found yet. Run `python3 build_house_local_context_audit.py`.")
    else:
        audit = pd.read_csv(audit_path)

        if audit.empty:
            st.info("Local context audit file exists but is empty.")
        else:
            st.caption(
                "Flags competitive districts where the forecast is relying mostly on fundamentals "
                "because polling, named candidates, incumbency, or special local adjustments are limited."
            )

            if "audit_priority" not in audit.columns:
                audit["audit_priority"] = "Unknown"
            if "mostly_fundamentals_only" not in audit.columns:
                audit["mostly_fundamentals_only"] = False
            if "has_polling" not in audit.columns:
                audit["has_polling"] = False

            for col in ["dem_win_probability", "model_margin_dem", "poll_count", "local_context_score"]:
                if col in audit.columns:
                    audit[col] = pd.to_numeric(audit[col], errors="coerce")

            priority_counts = audit["audit_priority"].fillna("Unknown").astype(str).value_counts()

            high_count = int(priority_counts.get("High", 0))
            medium_count = int(priority_counts.get("Medium", 0))
            low_count = int(priority_counts.get("Low", 0))

            mostly_fundamentals = audit["mostly_fundamentals_only"].astype(str).str.lower().isin(
                ["true", "1", "yes"]
            ).sum()

            with_polling = audit["has_polling"].astype(str).str.lower().isin(
                ["true", "1", "yes"]
            ).sum()

            competitive_count = 0
            if "competitiveness_band" in audit.columns:
                competitive_count = audit["competitiveness_band"].isin(
                    ["Toss-up range", "Competitive"]
                ).sum()

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("High Priority", high_count)
            m2.metric("Medium Priority", medium_count)
            m3.metric("Low Priority", low_count)
            m4.metric("Mostly Fundamentals", int(mostly_fundamentals))
            m5.metric("With Polling", int(with_polling))
            m6.metric("Competitive", int(competitive_count))

            st.divider()

            priority_order = {"High": 0, "Medium": 1, "Low": 2}
            audit["_priority_order"] = audit["audit_priority"].map(priority_order).fillna(9)

            if "dem_win_probability" in audit.columns:
                audit["_distance_from_50"] = (audit["dem_win_probability"] - 0.5).abs()
            else:
                audit["_distance_from_50"] = 9

            audit = audit.sort_values(["_priority_order", "_distance_from_50"])

            display_cols = [
                "district_id",
                "rating",
                "dem_win_probability",
                "model_margin_dem",
                "poll_count",
                "competitiveness_band",
                "audit_priority",
                "local_context_score",
                "mostly_fundamentals_only",
                "candidate_field_status",
                "dem_candidate",
                "gop_candidate",
                "incumbent",
                "incumbent_party",
                "general_election_party_structure",
                "recommended_review",
            ]

            display_cols = [c for c in display_cols if c in audit.columns]

            st.subheader("Highest-Priority Review Targets")

            high_medium = audit[audit["audit_priority"].isin(["High", "Medium"])]

            if high_medium.empty:
                st.success("No high- or medium-priority local context gaps found.")
            else:
                st.dataframe(
                    high_medium[display_cols].head(75),
                    use_container_width=True,
                    hide_index=True,
                )

            st.divider()
            st.subheader("Full Local Context Audit")

            f1, f2, f3 = st.columns(3)

            priorities = ["All"] + sorted(
                audit["audit_priority"].fillna("Unknown").astype(str).unique().tolist()
            )
            selected_priority = f1.selectbox(
                "Priority",
                priorities,
                index=0,
                key="local_context_priority",
            )

            if "competitiveness_band" in audit.columns:
                bands = ["All"] + sorted(
                    audit["competitiveness_band"].fillna("Unknown").astype(str).unique().tolist()
                )
            else:
                bands = ["All"]

            selected_band = f2.selectbox(
                "Competitiveness",
                bands,
                index=0,
                key="local_context_band",
            )

            fundamentals_filter = f3.selectbox(
                "Fundamentals-only status",
                ["All", "Mostly fundamentals only", "Has local context"],
                index=0,
                key="local_context_fundamentals_filter",
            )

            filtered = audit.copy()

            if selected_priority != "All":
                filtered = filtered[filtered["audit_priority"].astype(str) == selected_priority]

            if selected_band != "All" and "competitiveness_band" in filtered.columns:
                filtered = filtered[filtered["competitiveness_band"].astype(str) == selected_band]

            if fundamentals_filter == "Mostly fundamentals only":
                filtered = filtered[
                    filtered["mostly_fundamentals_only"]
                    .astype(str)
                    .str.lower()
                    .isin(["true", "1", "yes"])
                ]
            elif fundamentals_filter == "Has local context":
                filtered = filtered[
                    ~filtered["mostly_fundamentals_only"]
                    .astype(str)
                    .str.lower()
                    .isin(["true", "1", "yes"])
                ]

            st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

            csv = filtered[display_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download filtered local context audit",
                csv,
                file_name="house_local_context_audit_filtered.csv",
                mime="text/csv",
                key="download_local_context_audit",
            )

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
        e1.metric("National Error SD", fmt_num(get_first_available(srow, ["national_error_sd", "house_national_error_sd"]), 2))
        e2.metric("State Error SD", fmt_num(srow.get("state_error_sd"), 2))
        e3.metric("Region Error SD", fmt_num(get_first_available(srow, ["region_error_sd", "regional_error_sd", "house_region_error_sd"]), 2))
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

    if "general_election_party_structure" in df.columns and "party_control_override" in df.columns:
        same_party_missing_override = df[
            df["general_election_party_structure"].isin(["D_vs_D", "R_vs_R"])
            & df["party_control_override"].fillna("").astype(str).str.strip().eq("")
        ]

        if not same_party_missing_override.empty:
            issues.append(
                {
                    "Issue": "Same-party general election missing party-control override",
                    "Count": len(same_party_missing_override),
                    "Examples": ", ".join(same_party_missing_override["district_id"].head(10).tolist()),
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
with tab_margins:
    render_district_margin_explorer()


with tab_war:
    render_candidate_war_visibility()


if False:
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


def render_dynamic_uncertainty_audit():
    st.header("Dynamic Uncertainty")

    summary_path = OUTPUTS / "house_forecast_summary.csv"
    audit_path = OUTPUTS / "house_uncertainty_audit.csv"

    if not summary_path.exists():
        st.info("No House forecast summary found yet.")
        return

    summary = pd.read_csv(summary_path)

    if summary.empty:
        st.info("House forecast summary is empty.")
        return

    srow = summary.iloc[0]

    if str(srow.get("uncertainty_engine", "")).strip() == "":
        st.info("Dynamic uncertainty engine has not been run yet.")
        return

    u1, u2, u3 = st.columns(3)
    u1.metric("Uncertainty Engine", str(srow.get("uncertainty_engine", "NA")))
    u2.metric("Days Out", fmt_num(srow.get("days_out"), 0))
    u3.metric("Total Error SD", fmt_num(get_first_available(srow, ["total_error_sd", "house_total_error_sd"]), 2))

    u4, u5, u6, u7 = st.columns(4)
    u4.metric("National SD", fmt_num(get_first_available(srow, ["national_error_sd", "house_national_error_sd"]), 2))
    u5.metric("Region SD", fmt_num(get_first_available(srow, ["region_error_sd", "regional_error_sd", "house_region_error_sd"]), 2))
    u6.metric("Demographic SD", fmt_num(get_first_available(srow, ["demographic_error_sd", "education_race_error_sd", "demographic_group_error_sd", "house_demographic_error_sd"]), 2))
    u7.metric("District SD", fmt_num(get_first_available(srow, ["district_error_sd", "race_error_sd", "house_district_error_sd"]), 2))

    u8, u9, u10 = st.columns(3)
    u8.metric("Implied Correlation", fmt_pct(get_first_available(srow, ["implied_correlation", "house_implied_correlation"])))
    u9.metric("Region Groups", fmt_num(get_first_available(srow, ["region_groups", "region_error_groups"]), 0))
    u10.metric("Demographic Groups", fmt_num(get_first_available(srow, ["demographic_groups", "education_race_error_groups", "demographic_error_groups"]), 0))

    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        if not audit.empty:
            st.subheader("Uncertainty Settings Audit")
            st.dataframe(audit, use_container_width=True, hide_index=True)


def render_local_context_audit():
    st.header("Local Context Audit")

    if local_context_audit_output.empty:
        st.info("No local context audit found yet. Run `python3 build_house_local_context_audit.py` or the full House pipeline.")
        return

    audit = local_context_audit_output.copy()

    st.caption(
        "This audit flags districts where the model is relying mostly on fundamentals "
        "because polling, named candidates, candidate quality, incumbency, or local context are limited."
    )

    # Basic cleanup
    if "audit_priority" not in audit.columns:
        audit["audit_priority"] = "Unknown"

    if "mostly_fundamentals_only" not in audit.columns:
        audit["mostly_fundamentals_only"] = False

    if "has_polling" not in audit.columns:
        audit["has_polling"] = False

    for col in ["dem_win_probability", "model_margin_dem", "poll_count", "local_context_score"]:
        if col in audit.columns:
            audit[col] = pd.to_numeric(audit[col], errors="coerce")

    # Summary metrics
    priority_counts = audit["audit_priority"].fillna("Unknown").astype(str).value_counts()
    high_count = int(priority_counts.get("High", 0))
    medium_count = int(priority_counts.get("Medium", 0))
    low_count = int(priority_counts.get("Low", 0))

    mostly_fundamentals = audit["mostly_fundamentals_only"].astype(str).str.lower().isin(["true", "1", "yes"]).sum()
    with_polling = audit["has_polling"].astype(str).str.lower().isin(["true", "1", "yes"]).sum()

    competitive_count = 0
    if "competitiveness_band" in audit.columns:
        competitive_count = audit["competitiveness_band"].isin(["Toss-up range", "Competitive"]).sum()

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("High Priority", high_count)
    m2.metric("Medium Priority", medium_count)
    m3.metric("Low Priority", low_count)
    m4.metric("Mostly Fundamentals", int(mostly_fundamentals))
    m5.metric("With Polling", int(with_polling))
    m6.metric("Competitive", int(competitive_count))

    st.divider()

    # High-priority table
    st.subheader("Highest-Priority Review Targets")

    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    audit["_priority_order"] = audit["audit_priority"].map(priority_order).fillna(9)

    if "dem_win_probability" in audit.columns:
        audit["_distance_from_50"] = (audit["dem_win_probability"] - 0.5).abs()
    else:
        audit["_distance_from_50"] = 9

    review = audit.sort_values(["_priority_order", "_distance_from_50"]).copy()

    display_cols = [
        "district_id",
        "rating",
        "dem_win_probability",
        "model_margin_dem",
        "poll_count",
        "competitiveness_band",
        "audit_priority",
        "local_context_score",
        "mostly_fundamentals_only",
        "candidate_field_status",
        "dem_candidate",
        "gop_candidate",
        "incumbent",
        "incumbent_party",
        "general_election_party_structure",
        "recommended_review",
    ]

    display_cols = [c for c in display_cols if c in review.columns]

    high_medium = review[review["audit_priority"].isin(["High", "Medium"])]

    if high_medium.empty:
        st.success("No high- or medium-priority local context gaps found.")
    else:
        st.dataframe(
            high_medium[display_cols].head(50),
            use_container_width=True,
            hide_index=True,
        )

    # Filters for full audit
    st.divider()
    st.subheader("Full Local Context Audit")

    f1, f2, f3 = st.columns(3)

    priorities = ["All"] + sorted(audit["audit_priority"].fillna("Unknown").astype(str).unique().tolist())
    selected_priority = f1.selectbox("Priority", priorities, index=0)

    if "competitiveness_band" in audit.columns:
        bands = ["All"] + sorted(audit["competitiveness_band"].fillna("Unknown").astype(str).unique().tolist())
    else:
        bands = ["All"]
    selected_band = f2.selectbox("Competitiveness", bands, index=0)

    fundamentals_filter = f3.selectbox(
        "Fundamentals-only status",
        ["All", "Mostly fundamentals only", "Has local context"],
        index=0,
    )

    filtered = review.copy()

    if selected_priority != "All":
        filtered = filtered[filtered["audit_priority"].astype(str) == selected_priority]

    if selected_band != "All" and "competitiveness_band" in filtered.columns:
        filtered = filtered[filtered["competitiveness_band"].astype(str) == selected_band]

    if fundamentals_filter == "Mostly fundamentals only":
        filtered = filtered[filtered["mostly_fundamentals_only"].astype(str).str.lower().isin(["true", "1", "yes"])]
    elif fundamentals_filter == "Has local context":
        filtered = filtered[~filtered["mostly_fundamentals_only"].astype(str).str.lower().isin(["true", "1", "yes"])]

    st.dataframe(
        filtered[display_cols],
        use_container_width=True,
        hide_index=True,
    )

    csv = filtered[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered local context audit",
        csv,
        file_name="house_local_context_audit_filtered.csv",
        mime="text/csv",
    )

def render_calibration_audit():
    st.header("Calibration Audit")

    audit_path = OUTPUTS / "house_calibration_audit.csv"

    if not audit_path.exists():
        st.info("No calibration audit found yet. Run the House full pipeline or Build House Calibration Audit task.")
        return

    audit = pd.read_csv(audit_path)

    if audit.empty:
        st.info("Calibration audit file is empty.")
        return

    st.caption(
        "This table decomposes each district into baseline, environment, incumbency, candidate quality, polling, final margin, and rating."
    )

    flagged = audit[audit.get("audit_flags", "").fillna("").astype(str).str.strip().ne("")] if "audit_flags" in audit.columns else pd.DataFrame()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Districts", len(audit))
    c2.metric("Flagged Rows", len(flagged))
    c3.metric("Median Dem Win Prob", f"{audit['dem_win_probability'].median():.1%}" if "dem_win_probability" in audit.columns else "NA")
    c4.metric("Polled Districts", int((audit["poll_count"].fillna(0) > 0).sum()) if "poll_count" in audit.columns else 0)

    st.subheader("Filters")

    f1, f2, f3, f4 = st.columns(4)

    region_options = ["All"]
    if "region" in audit.columns:
        region_options += sorted(audit["region"].fillna("Unknown").astype(str).unique().tolist())

    district_type_options = ["All"]
    if "district_type" in audit.columns:
        district_type_options += sorted(audit["district_type"].fillna("Unknown").astype(str).unique().tolist())

    rating_options = ["All"]
    if "rating" in audit.columns:
        rating_options += sorted(audit["rating"].fillna("Unknown").astype(str).unique().tolist())

    flag_filter = f1.selectbox("Audit flags", ["All", "Flagged only", "Unflagged only"])
    region_filter = f2.selectbox("Region", region_options)
    type_filter = f3.selectbox("District type", district_type_options)
    rating_filter = f4.selectbox("Rating", rating_options)

    view = audit.copy()

    if flag_filter == "Flagged only" and "audit_flags" in view.columns:
        view = view[view["audit_flags"].fillna("").astype(str).str.strip().ne("")]
    elif flag_filter == "Unflagged only" and "audit_flags" in view.columns:
        view = view[view["audit_flags"].fillna("").astype(str).str.strip().eq("")]

    if region_filter != "All" and "region" in view.columns:
        view = view[view["region"].fillna("Unknown").astype(str) == region_filter]

    if type_filter != "All" and "district_type" in view.columns:
        view = view[view["district_type"].fillna("Unknown").astype(str) == type_filter]

    if rating_filter != "All" and "rating" in view.columns:
        view = view[view["rating"].fillna("Unknown").astype(str) == rating_filter]

    st.subheader("Calibration Table")

    display_cols = [
        "district_id",
        "rating",
        "model_margin_label",
        "dem_win_probability",
        "baseline_label",
        "state_environment_adjustment_dem",
        "audit_elasticity",
        "incumbency_adjustment_dem",
        "candidate_quality_adjustment_dem",
        "fundamentals_margin_label",
        "polling_margin_label",
        "poll_count",
        "bayesian_polling_weight",
        "region",
        "district_type",
        "college_share_tier",
        "white_share_tier",
        "black_share_tier",
        "hispanic_share_tier",
        "median_income_tier",
        "general_election_party_structure",
        "party_control_fixed",
        "audit_flags",
    ]

    display_cols = [c for c in display_cols if c in view.columns]

    st.dataframe(
        view[display_cols],
        use_container_width=True,
        hide_index=True,
    )

    if not flagged.empty:
        st.subheader("Flagged Rows")
        flagged_cols = [
            "district_id",
            "rating",
            "model_margin_label",
            "dem_win_probability",
            "audit_flags",
        ]
        flagged_cols = [c for c in flagged_cols if c in flagged.columns]
        st.dataframe(flagged[flagged_cols], use_container_width=True, hide_index=True)

    st.subheader("Largest Model Movement from Fundamentals")

    if "audit_final_vs_fundamentals_gap" in audit.columns:
        move = audit.copy()
        move["abs_gap"] = move["audit_final_vs_fundamentals_gap"].abs()
        move = move.sort_values("abs_gap", ascending=False)

        move_cols = [
            "district_id",
            "rating",
            "fundamentals_margin_label",
            "polling_margin_label",
            "model_margin_label",
            "poll_count",
            "bayesian_polling_weight",
            "audit_final_vs_fundamentals_gap",
        ]
        move_cols = [c for c in move_cols if c in move.columns]

        st.dataframe(move[move_cols].head(25), use_container_width=True, hide_index=True)


st.divider()
render_calibration_audit()

# --- Unified House manual poll editor with partisan metadata ---
import pandas as _house_poll_pd
from pathlib import Path as _HousePollPath

st.divider()
st.header("Manual House Poll Entry / Partisan Poll Metadata")

st.caption(
    "Add, edit, or delete manually entered House polls. Partisan/sponsor metadata "
    "feeds the House partisan pollster adjustment script. The pipeline generates "
    "pollster/house-effect adjustments; this dashboard only stores the poll metadata."
)

_house_poll_path = _HousePollPath("inputs/house_manual_polls.csv")


def _house_poll_clean_text_series(s):
    return (
        s.fillna("")
        .astype(str)
        .replace({"nan": "", "None": "", "NaN": ""})
    )


if not _house_poll_path.exists():
    st.warning(f"Could not find {_house_poll_path}. Creating an empty manual poll table.")
    _house_poll_df = _house_poll_pd.DataFrame()
else:
    _house_poll_df = _house_poll_pd.read_csv(_house_poll_path)

# Remove generated/audit columns from the editable dashboard view.
_house_generated_or_audit_cols = {
    "polling_margin_dem_original",
    "partisan_pollster_adjustment_dem",
    "partisan_pollster_weight_multiplier",
    "partisan_pollster_adjusted",
    "partisan_pollster_notes",
    "polling_margin_dem_adjusted",
    "margin_dem",
    "final_poll_margin_dem",
    "house_effect_adjusted_dem_margin",
    "house_effect_dem",
    "house_effect_adjustment_dem",
    "manual_house_effect",
    "manual_house_effect_adjustment_dem",
}

_house_poll_df = _house_poll_df.drop(
    columns=[c for c in _house_generated_or_audit_cols if c in _house_poll_df.columns],
    errors="ignore",
)

_house_metadata_defaults = {
    "poll_sponsor_type": "",
    "partisan_sponsor_party": "",
    "is_internal_poll": False,
    "pollster_partisan_affiliation": "",
    "partisan_pollster_review_notes": "",
}

for _col, _default in _house_metadata_defaults.items():
    if _col not in _house_poll_df.columns:
        _house_poll_df[_col] = _default

_house_preferred_cols = [
    "district_id",
    "district",
    "race",
    "state",
    "pollster",
    "pollster_grade",
    "sponsor",
    "poll_sponsor_type",
    "partisan_sponsor_party",
    "is_internal_poll",
    "pollster_partisan_affiliation",
    "partisan_pollster_review_notes",
    "start_date",
    "end_date",
    "sample_size",
    "population",
    "sample_type",
    "mode",
    "dem_candidate",
    "gop_candidate",
    "rep_candidate",
    "dem_pct",
    "gop_pct",
    "rep_pct",
    "ind_pct",
    "other_pct",
    "undecided_pct",
    "notes",
]

for _col in _house_preferred_cols:
    if _col not in _house_poll_df.columns:
        _house_poll_df[_col] = ""

_house_ordered_cols = [_c for _c in _house_preferred_cols if _c in _house_poll_df.columns]
_house_remaining_cols = [_c for _c in _house_poll_df.columns if _c not in _house_ordered_cols]
_house_poll_df = _house_poll_df.loc[:, _house_ordered_cols + _house_remaining_cols].copy()

_house_text_cols = [
    c for c in _house_poll_df.columns
    if c not in [
        "sample_size",
        "dem_pct",
        "gop_pct",
        "rep_pct",
        "ind_pct",
        "other_pct",
        "undecided_pct",
        "is_internal_poll",
    ]
]

for _col in _house_text_cols:
    _house_poll_df[_col] = _house_poll_clean_text_series(_house_poll_df[_col])

_house_poll_df["is_internal_poll"] = (
    _house_poll_df["is_internal_poll"]
    .fillna(False)
    .astype(str)
    .str.lower()
    .isin(["true", "1", "yes", "y"])
)

for _col in [
    "sample_size",
    "dem_pct",
    "gop_pct",
    "rep_pct",
    "ind_pct",
    "other_pct",
    "undecided_pct",
]:
    if _col in _house_poll_df.columns:
        _house_poll_df[_col] = _house_poll_pd.to_numeric(_house_poll_df[_col], errors="coerce")

st.subheader("Edit Existing Manual House Polls")

_house_editable = _house_poll_df.copy()
_house_editable.insert(0, "delete", False)
_house_editable.insert(1, "row_id", range(1, len(_house_editable) + 1))

_house_edited = st.data_editor(
    _house_editable,
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    key="dashboard_house_manual_poll_editor_unified_v1",
    column_config={
        "delete": st.column_config.CheckboxColumn("Delete", default=False),
        "row_id": st.column_config.NumberColumn("Row", disabled=True),
        "pollster_grade": st.column_config.SelectboxColumn(
            "Pollster grade",
            options=["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "Unknown", ""],
        ),
        "population": st.column_config.SelectboxColumn(
            "Population",
            options=["LV", "RV", "A", "Other", ""],
        ),
        "sample_type": st.column_config.SelectboxColumn(
            "Sample type",
            options=["LV", "RV", "A", "Other", ""],
        ),
        "poll_sponsor_type": st.column_config.SelectboxColumn(
            "Sponsor type",
            options=["", "independent", "media", "university", "party", "campaign", "super PAC", "other"],
        ),
        "partisan_sponsor_party": st.column_config.SelectboxColumn(
            "Sponsor party",
            options=["", "D", "R", "none", "unknown"],
        ),
        "is_internal_poll": st.column_config.CheckboxColumn(
            "Internal/campaign poll",
            default=False,
        ),
        "pollster_partisan_affiliation": st.column_config.SelectboxColumn(
            "Pollster partisan affiliation",
            options=["", "D", "R", "none", "unknown"],
        ),
        "partisan_pollster_review_notes": st.column_config.TextColumn(
            "Partisan poll notes",
        ),
        "sample_size": st.column_config.NumberColumn(
            "Sample size",
            min_value=0,
            step=1,
            format="%d",
        ),
        "dem_pct": st.column_config.NumberColumn("Dem %", step=0.1, format="%.1f"),
        "gop_pct": st.column_config.NumberColumn("GOP %", step=0.1, format="%.1f"),
        "rep_pct": st.column_config.NumberColumn("Rep %", step=0.1, format="%.1f"),
        "ind_pct": st.column_config.NumberColumn("Ind %", step=0.1, format="%.1f"),
        "other_pct": st.column_config.NumberColumn("Other %", step=0.1, format="%.1f"),
        "undecided_pct": st.column_config.NumberColumn("Undecided %", step=0.1, format="%.1f"),
    },
)

_house_save_col, _house_info_col = st.columns([1, 3])

with _house_save_col:
    _house_save_edits = st.button(
        "Save House Poll Edits / Delete Marked Polls",
        type="primary",
        key="dashboard_save_house_manual_poll_edits_unified_v1",
    )

with _house_info_col:
    st.caption("Saving overwrites inputs/house_manual_polls.csv and creates a .bak backup.")

if _house_save_edits:
    _house_updated = _house_edited.copy()

    if "delete" in _house_updated.columns:
        _house_updated = _house_updated[~_house_updated["delete"].fillna(False)].copy()

    for _col in ["delete", "row_id"]:
        if _col in _house_updated.columns:
            _house_updated = _house_updated.drop(columns=[_col])

    _house_updated = _house_updated.loc[
        :,
        [_c for _c in _house_poll_df.columns if _c in _house_updated.columns],
    ].copy()

    for _col in _house_text_cols:
        if _col in _house_updated.columns:
            _house_updated[_col] = _house_poll_clean_text_series(_house_updated[_col])

    if "state" in _house_updated.columns:
        _house_updated["state"] = _house_updated["state"].fillna("").astype(str).str.strip().str.upper()

    if "is_internal_poll" in _house_updated.columns:
        _house_updated["is_internal_poll"] = (
            _house_updated["is_internal_poll"]
            .fillna(False)
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes", "y"])
        )

    for _col in [
        "sample_size",
        "dem_pct",
        "gop_pct",
        "rep_pct",
        "ind_pct",
        "other_pct",
        "undecided_pct",
    ]:
        if _col in _house_updated.columns:
            _house_updated[_col] = _house_poll_pd.to_numeric(_house_updated[_col], errors="coerce")

    _key_cols = [
        _c for _c in [
            "district_id",
            "district",
            "race",
            "state",
            "pollster",
            "dem_pct",
            "gop_pct",
            "rep_pct",
        ]
        if _c in _house_updated.columns
    ]

    if _key_cols:
        _house_nonblank_mask = _house_updated[_key_cols].notna().any(axis=1)
        _house_updated = _house_updated[_house_nonblank_mask].copy()

    _house_poll_path.parent.mkdir(parents=True, exist_ok=True)

    if _house_poll_path.exists():
        _house_backup_path = _house_poll_path.with_suffix(".csv.bak")
        _house_poll_df.to_csv(_house_backup_path, index=False)

    _house_updated.to_csv(_house_poll_path, index=False)

    st.success(
        f"Saved {len(_house_updated)} manual House polls to {_house_poll_path}. "
        "Run the House full pipeline to ingest the changes."
    )

st.subheader("Add New House Poll")

_house_district_field = (
    "district_id"
    if "district_id" in _house_poll_df.columns
    else "district"
    if "district" in _house_poll_df.columns
    else "race"
)

_house_gop_pct_field = "gop_pct" if "gop_pct" in _house_poll_df.columns else "rep_pct"

with st.form("dashboard_house_manual_poll_entry_form_unified_v1"):
    _c1, _c2, _c3 = st.columns(3)

    with _c1:
        _district_value = st.text_input("District / Race", value="")
        _state = st.text_input("State", value="")
        _pollster = st.text_input("Pollster", value="")
        _pollster_grade = st.selectbox(
            "Pollster grade",
            ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D", "Unknown", ""],
            index=4,
        )
        _sponsor = st.text_input("Sponsor", value="")

    with _c2:
        _start_date = st.date_input("Start date")
        _end_date = st.date_input("End date")
        _sample_size = st.number_input("Sample size", min_value=0, value=800, step=1)
        _sample_type = st.selectbox("Sample type", ["LV", "RV", "A", "Other", ""], index=0)
        _dem_candidate = st.text_input("Dem candidate", value="")
        _gop_candidate = st.text_input("GOP/Rep candidate", value="")

    with _c3:
        _dem_pct = st.number_input("Dem %", value=0.0, step=0.1, format="%.1f")
        _gop_pct = st.number_input("GOP/Rep %", value=0.0, step=0.1, format="%.1f")
        _ind_pct = st.number_input("Ind %", value=0.0, step=0.1, format="%.1f")
        _other_pct = st.number_input("Other %", value=0.0, step=0.1, format="%.1f")
        _undecided_pct = st.number_input("Undecided %", value=0.0, step=0.1, format="%.1f")
        _notes = st.text_area("General notes", value="")

    st.markdown("#### Partisan / Sponsor Metadata")

    _p1, _p2, _p3 = st.columns(3)

    with _p1:
        _poll_sponsor_type = st.selectbox(
            "Sponsor type",
            ["", "independent", "media", "university", "party", "campaign", "super PAC", "other"],
            index=0,
        )

    with _p2:
        _partisan_sponsor_party = st.selectbox(
            "Sponsor party",
            ["", "D", "R", "none", "unknown"],
            index=0,
        )

    with _p3:
        _pollster_partisan_affiliation = st.selectbox(
            "Pollster partisan affiliation",
            ["", "D", "R", "none", "unknown"],
            index=0,
        )

    _is_internal_poll = st.checkbox("Internal/campaign poll", value=False)
    _partisan_pollster_review_notes = st.text_input("Partisan poll notes", value="")

    _submitted = st.form_submit_button("Add House Poll")

    if _submitted:
        _new_row = {_c: "" for _c in _house_poll_df.columns}

        _new_row[_house_district_field] = _district_value

        if "state" in _new_row:
            _new_row["state"] = _state.strip().upper()
        if "pollster" in _new_row:
            _new_row["pollster"] = _pollster
        if "pollster_grade" in _new_row:
            _new_row["pollster_grade"] = _pollster_grade
        if "sponsor" in _new_row:
            _new_row["sponsor"] = _sponsor
        if "start_date" in _new_row:
            _new_row["start_date"] = _start_date.isoformat()
        if "end_date" in _new_row:
            _new_row["end_date"] = _end_date.isoformat()
        if "sample_size" in _new_row:
            _new_row["sample_size"] = _sample_size
        if "sample_type" in _new_row:
            _new_row["sample_type"] = _sample_type
        if "population" in _new_row:
            _new_row["population"] = _sample_type
        if "dem_candidate" in _new_row:
            _new_row["dem_candidate"] = _dem_candidate
        if "gop_candidate" in _new_row:
            _new_row["gop_candidate"] = _gop_candidate
        if "rep_candidate" in _new_row:
            _new_row["rep_candidate"] = _gop_candidate

        _new_row["dem_pct"] = _dem_pct
        _new_row[_house_gop_pct_field] = _gop_pct

        if "ind_pct" in _new_row:
            _new_row["ind_pct"] = _ind_pct
        if "other_pct" in _new_row:
            _new_row["other_pct"] = _other_pct
        if "undecided_pct" in _new_row:
            _new_row["undecided_pct"] = _undecided_pct
        if "notes" in _new_row:
            _new_row["notes"] = _notes

        _new_row["poll_sponsor_type"] = _poll_sponsor_type
        _new_row["partisan_sponsor_party"] = _partisan_sponsor_party
        _new_row["is_internal_poll"] = _is_internal_poll
        _new_row["pollster_partisan_affiliation"] = _pollster_partisan_affiliation
        _new_row["partisan_pollster_review_notes"] = _partisan_pollster_review_notes

        _house_updated = _house_poll_pd.concat(
            [_house_poll_df, _house_poll_pd.DataFrame([_new_row])],
            ignore_index=True,
        )

        _house_updated.to_csv(_house_poll_path, index=False)

        st.success(
            f"Saved new House poll to {_house_poll_path}. "
            "Run the House full pipeline to ingest it."
        )
        st.dataframe(_house_updated.tail(10), use_container_width=True, hide_index=True)

st.caption(
    "Public/media poll: leave sponsor party blank or use none. "
    "Internal poll: use sponsor type campaign, sponsor party D/R, and check internal/campaign poll."
)

