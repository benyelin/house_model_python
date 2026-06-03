from pathlib import Path
import pandas as pd
import numpy as np

INPUTS = Path("inputs")
OUTPUTS = Path("outputs")

RACE_INPUTS = INPUTS / "house_race_inputs.csv"
RACE_STATS = OUTPUTS / "house_race_stats.csv"
SUMMARY = OUTPUTS / "house_forecast_summary.csv"
AUDIT_OUTPUT = OUTPUTS / "house_calibration_audit.csv"


def read_csv(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def to_num(s, default=np.nan):
    return pd.to_numeric(s, errors="coerce").fillna(default)


def first_existing(df, cols, default=np.nan):
    for col in cols:
        if col in df.columns:
            return df[col]
    return pd.Series([default] * len(df), index=df.index)


def fmt_margin(x):
    try:
        x = float(x)
    except Exception:
        return ""
    if pd.isna(x):
        return ""
    if x > 0:
        return f"D+{x:.1f}"
    if x < 0:
        return f"R+{abs(x):.1f}"
    return "Even"


def infer_rating(prob):
    try:
        p = float(prob)
    except Exception:
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
        return "Toss-Up"
    if p > 0.35:
        return "Tilt R"
    if p > 0.15:
        return "Lean R"
    if p > 0.05:
        return "Likely R"
    return "Safe R"


def main():
    if not RACE_INPUTS.exists():
        raise FileNotFoundError("inputs/house_race_inputs.csv not found. Run the House pipeline first.")

    inputs = pd.read_csv(RACE_INPUTS)
    stats = read_csv(RACE_STATS)

    if inputs.empty:
        raise ValueError("inputs/house_race_inputs.csv is empty.")

    if "district_id" not in inputs.columns:
        raise ValueError("house_race_inputs.csv must include district_id.")

    inputs["district_id"] = inputs["district_id"].astype(str).str.strip()

    # Prefer model outputs for final margin/prob/rating when available.
    if not stats.empty and "district_id" in stats.columns:
        stats["district_id"] = stats["district_id"].astype(str).str.strip()

        keep = [
            "district_id",
            "model_margin_dem",
            "dem_win_probability",
            "rating",
            "party_control_fixed",
        ]
        keep = [c for c in keep if c in stats.columns]

        df = inputs.merge(
            stats[keep],
            on="district_id",
            how="left",
            suffixes=("", "_output"),
        )

        for col in ["model_margin_dem", "dem_win_probability", "rating", "party_control_fixed"]:
            out_col = f"{col}_output"
            if out_col in df.columns:
                if col in df.columns:
                    df[col] = df[out_col].combine_first(df[col])
                else:
                    df[col] = df[out_col]
                df = df.drop(columns=[out_col])
    else:
        df = inputs.copy()

    # Normalize useful numeric fields.
    numeric_defaults = {
        "pres_2024_margin_dem": np.nan,
        "pres_2020_margin_dem": np.nan,
        "genballot_adjusted_margin_dem": np.nan,
        "district_partisan_baseline_dem": np.nan,
        "state_environment_adjustment_dem": 0.0,
        "state_elasticity": np.nan,
        "district_elasticity": np.nan,
        "incumbency_adjustment_dem": 0.0,
        "candidate_quality_adjustment_dem": 0.0,
        "special_adjustment_dem": 0.0,
        "fundamentals_margin_dem": np.nan,
        "polling_margin_dem": np.nan,
        "poll_count": 0,
        "bayesian_polling_weight": np.nan,
        "bayesian_model_margin_dem": np.nan,
        "model_margin_dem": np.nan,
        "dem_win_probability": np.nan,
    }

    for col, default in numeric_defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    # Use whichever elasticity field exists.
    if "district_elasticity" in df.columns and df["district_elasticity"].notna().any():
        df["audit_elasticity"] = df["district_elasticity"]
    elif "state_elasticity" in df.columns:
        df["audit_elasticity"] = df["state_elasticity"]
    else:
        df["audit_elasticity"] = np.nan

    # Infer the actual environment adjustment used in fundamentals.
    # Some versions of the House model store the environment effect directly
    # in fundamentals rather than in state_environment_adjustment_dem.
    df["environment_adjustment_used_dem"] = (
        df["fundamentals_margin_dem"]
        - df["district_partisan_baseline_dem"].fillna(0)
        - df["incumbency_adjustment_dem"].fillna(0)
        - df["candidate_quality_adjustment_dem"].fillna(0)
        - df["special_adjustment_dem"].fillna(0)
    )

    # Component sum check using the inferred environment effect.
    df["audit_component_sum_dem"] = (
        df["district_partisan_baseline_dem"].fillna(0)
        + df["environment_adjustment_used_dem"].fillna(0)
        + df["incumbency_adjustment_dem"].fillna(0)
        + df["candidate_quality_adjustment_dem"].fillna(0)
        + df["special_adjustment_dem"].fillna(0)
    )

    df["audit_fundamentals_gap"] = (
        df["fundamentals_margin_dem"] - df["audit_component_sum_dem"]
    )

    # Polling gap/final blend diagnostics.
    df["audit_poll_vs_fundamentals_gap"] = (
        df["polling_margin_dem"] - df["fundamentals_margin_dem"]
    )

    df["audit_final_vs_fundamentals_gap"] = (
        df["model_margin_dem"] - df["fundamentals_margin_dem"]
    )

    df["audit_final_vs_polling_gap"] = (
        df["model_margin_dem"] - df["polling_margin_dem"]
    )

    # Helpful labels.
    df["model_margin_label"] = df["model_margin_dem"].apply(fmt_margin)
    df["fundamentals_margin_label"] = df["fundamentals_margin_dem"].apply(fmt_margin)
    df["polling_margin_label"] = df["polling_margin_dem"].apply(fmt_margin)
    df["baseline_label"] = df["district_partisan_baseline_dem"].apply(fmt_margin)

    if "rating" not in df.columns:
        df["rating"] = df["dem_win_probability"].apply(infer_rating)
    else:
        df["rating"] = df["rating"].fillna(
            df["dem_win_probability"].apply(infer_rating)
        )

    # Add issue flags for quick dashboard filtering.
    flags = []

    for _, row in df.iterrows():
        row_flags = []

        if pd.isna(row.get("district_partisan_baseline_dem")):
            row_flags.append("missing baseline")

        if abs(row.get("audit_fundamentals_gap", 0.0)) > 0.05:
            row_flags.append("fundamentals component mismatch")

        if row.get("poll_count", 0) > 0 and pd.isna(row.get("polling_margin_dem")):
            row_flags.append("poll count but no polling margin")

        if row.get("party_control_fixed", "") == "D" and row.get("dem_win_probability", 0) < 0.999:
            row_flags.append("fixed D control but probability not 1")

        if row.get("party_control_fixed", "") == "R" and row.get("dem_win_probability", 1) > 0.001:
            row_flags.append("fixed R control but probability not 0")

        structure = str(row.get("general_election_party_structure", "")).strip()
        override = str(row.get("party_control_override", "")).strip()

        if structure in ["D_vs_D", "D_unopposed"] and str(row.get("party_control_fixed", "")) != "D":
            row_flags.append("structure implies fixed D but not fixed")

        if structure in ["R_vs_R", "R_unopposed"] and str(row.get("party_control_fixed", "")) != "R":
            row_flags.append("structure implies fixed R but not fixed")

        if structure == "D_vs_R_vs_Other" and str(row.get("other_candidate", "")).strip() == "":
            row_flags.append("party structure has Other but Other candidate blank")

        flags.append("; ".join(row_flags))

    df["audit_flags"] = flags

    output_cols = [
        "district_id",
        "state",
        "district",
        "region",
        "district_type",
        "incumbent_party",
        "inferred_incumbent_party",
        "dem_candidate",
        "gop_candidate",
        "other_candidate",
        "election_system",
        "general_election_party_structure",
        "party_control_override",
        "party_control_fixed",
        "pres_2024_margin_dem",
        "pres_2020_margin_dem",
        "district_partisan_baseline_dem",
        "baseline_label",
        "audit_elasticity",
        "state_environment_adjustment_dem",
        "environment_adjustment_used_dem",
        "incumbency_adjustment_dem",
        "candidate_quality_adjustment_dem",
        "special_adjustment_dem",
        "audit_component_sum_dem",
        "fundamentals_margin_dem",
        "fundamentals_margin_label",
        "audit_fundamentals_gap",
        "polling_margin_dem",
        "polling_margin_label",
        "poll_count",
        "bayesian_polling_weight",
        "bayesian_model_margin_dem",
        "model_margin_dem",
        "model_margin_label",
        "dem_win_probability",
        "rating",
        "audit_poll_vs_fundamentals_gap",
        "audit_final_vs_fundamentals_gap",
        "audit_final_vs_polling_gap",
        "college_share_tier",
        "white_share_tier",
        "black_share_tier",
        "hispanic_share_tier",
        "median_income_tier",
        "education_race_error_group",
        "demographic_error_group",
        "audit_flags",
    ]

    output_cols = [c for c in output_cols if c in df.columns]

    audit = df[output_cols].copy()

    audit = audit.sort_values(
        by=["audit_flags", "dem_win_probability", "district_id"],
        ascending=[False, True, True],
    )

    OUTPUTS.mkdir(exist_ok=True)
    audit.to_csv(AUDIT_OUTPUT, index=False)

    print(f"Wrote {AUDIT_OUTPUT}")
    print(f"Rows: {len(audit)}")

    flagged = audit[audit["audit_flags"].fillna("").astype(str).str.strip().ne("")]
    print(f"Flagged rows: {len(flagged)}")

    if not flagged.empty:
        print()
        print("Flagged sample:")
        print(
            flagged[
                [
                    "district_id",
                    "rating",
                    "model_margin_label",
                    "dem_win_probability",
                    "audit_flags",
                ]
            ].head(20).to_string(index=False)
        )

    print()
    print("Rating counts:")
    print(audit["rating"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
