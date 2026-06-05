from pathlib import Path
import pandas as pd
import numpy as np

INPUTS = Path("inputs")
OUTPUTS = Path("outputs")

RACE_STATS_PATH = OUTPUTS / "house_race_stats.csv"
RACE_INPUTS_PATH = INPUTS / "house_race_inputs.csv"
OUTPUT_PATH = OUTPUTS / "house_local_context_audit.csv"


def as_num(series, default=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def clean_text(series):
    return series.fillna("").astype(str).str.strip()


def main():
    if not RACE_STATS_PATH.exists():
        raise FileNotFoundError("outputs/house_race_stats.csv not found. Run the House pipeline first.")

    stats = pd.read_csv(RACE_STATS_PATH)

    if RACE_INPUTS_PATH.exists():
        inputs = pd.read_csv(RACE_INPUTS_PATH)
    else:
        inputs = pd.DataFrame()

    if "district_id" not in stats.columns:
        raise ValueError("house_race_stats.csv must include district_id.")

    stats["district_id"] = clean_text(stats["district_id"])

    if not inputs.empty and "district_id" in inputs.columns:
        inputs["district_id"] = clean_text(inputs["district_id"])

        keep_cols = [
            "district_id",
            "dem_candidate",
            "gop_candidate",
            "other_candidate",
            "other",
            "incumbent",
            "incumbent_party",
            "general_election_party_structure",
            "party_control_override",
            "election_system",
            "district_type",
            "region",
            "college_share_tier",
            "white_share_tier",
            "black_share_tier",
            "hispanic_share_tier",
            "median_income_tier",
        ]

        keep_cols = [c for c in keep_cols if c in inputs.columns]

        df = stats.merge(
            inputs[keep_cols],
            on="district_id",
            how="left",
            suffixes=("", "_input"),
        )
    else:
        df = stats.copy()

    # Core numeric fields.
    for col in [
        "dem_win_probability",
        "model_margin_dem",
        "fundamentals_margin_dem",
        "polling_margin_dem",
        "poll_count",
        "incumbency_adjustment_dem",
        "candidate_quality_adjustment_dem",
        "special_adjustment_dem",
        "effective_district_error_sd",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    df["dem_win_probability_num"] = as_num(df["dem_win_probability"], np.nan)
    df["model_margin_dem_num"] = as_num(df["model_margin_dem"], np.nan)
    df["fundamentals_margin_dem_num"] = as_num(df["fundamentals_margin_dem"], np.nan)
    df["poll_count_num"] = as_num(df["poll_count"], 0)

    # Feature flags.
    df["has_polling"] = df["poll_count_num"] > 0

    df["has_incumbency_adjustment"] = as_num(df["incumbency_adjustment_dem"], 0).abs() > 0.01
    df["has_candidate_quality_adjustment"] = as_num(df["candidate_quality_adjustment_dem"], 0).abs() > 0.01
    df["has_special_adjustment"] = as_num(df["special_adjustment_dem"], 0).abs() > 0.01

    for col in ["dem_candidate", "gop_candidate", "other_candidate", "other", "general_election_party_structure", "party_control_override"]:
        if col not in df.columns:
            df[col] = ""

    df["dem_candidate_clean"] = clean_text(df["dem_candidate"])
    df["gop_candidate_clean"] = clean_text(df["gop_candidate"])

    other_field = clean_text(df["other_candidate"]) if "other_candidate" in df.columns else ""
    if "other" in df.columns:
        other_field = other_field.astype(str).where(other_field.astype(str).str.len() > 0, clean_text(df["other"]))

    df["has_dem_candidate"] = df["dem_candidate_clean"].str.len() > 0
    df["has_gop_candidate"] = df["gop_candidate_clean"].str.len() > 0
    df["has_other_candidate"] = pd.Series(other_field).fillna("").astype(str).str.strip().str.len() > 0

    df["candidate_field_status"] = np.select(
        [
            df["has_dem_candidate"] & df["has_gop_candidate"],
            df["has_dem_candidate"] & ~df["has_gop_candidate"],
            ~df["has_dem_candidate"] & df["has_gop_candidate"],
        ],
        [
            "D and R named",
            "Only D named",
            "Only R named",
        ],
        default="No major candidates named",
    )

    df["party_structure_clean"] = clean_text(df["general_election_party_structure"])
    df["party_control_override_clean"] = clean_text(df["party_control_override"]).str.upper()

    df["has_party_structure"] = df["party_structure_clean"].str.len() > 0
    df["has_party_control_override"] = df["party_control_override_clean"].isin(["D", "R"])

    # Competitive bands.
    df["abs_model_margin"] = df["model_margin_dem_num"].abs()
    df["distance_from_50"] = (df["dem_win_probability_num"] - 0.5).abs()

    df["competitiveness_band"] = np.select(
        [
            df["distance_from_50"] <= 0.05,
            df["distance_from_50"] <= 0.15,
            df["distance_from_50"] <= 0.35,
        ],
        [
            "Toss-up range",
            "Competitive",
            "Potentially competitive",
        ],
        default="Likely/Safe",
    )

    # Context score: higher means richer race-specific information.
    df["local_context_score"] = (
        df["has_polling"].astype(int) * 3
        + df["has_incumbency_adjustment"].astype(int) * 2
        + df["has_candidate_quality_adjustment"].astype(int) * 2
        + df["has_special_adjustment"].astype(int) * 2
        + df["has_dem_candidate"].astype(int)
        + df["has_gop_candidate"].astype(int)
        + df["has_party_structure"].astype(int)
        + df["has_party_control_override"].astype(int)
    )

    df["mostly_fundamentals_only"] = df["local_context_score"] <= 2

    df["audit_priority"] = np.select(
        [
            (df["competitiveness_band"].isin(["Toss-up range", "Competitive"])) & df["mostly_fundamentals_only"],
            (df["competitiveness_band"].isin(["Toss-up range", "Competitive"])) & (df["local_context_score"] <= 4),
            (df["competitiveness_band"].eq("Potentially competitive")) & df["mostly_fundamentals_only"],
        ],
        [
            "High",
            "Medium",
            "Medium",
        ],
        default="Low",
    )

    df["recommended_review"] = np.select(
        [
            df["audit_priority"].eq("High"),
            df["audit_priority"].eq("Medium"),
            df["has_polling"],
        ],
        [
            "Add/check candidates, incumbency, candidate quality, special local factors, and polling if available.",
            "Review candidate/local context before trusting rating.",
            "Check poll freshness and whether polling is reflected in model margin.",
        ],
        default="No immediate review needed.",
    )

    output_cols = [
        "district_id",
        "rating",
        "dem_win_probability",
        "model_margin_dem",
        "fundamentals_margin_dem",
        "polling_margin_dem",
        "poll_count",
        "competitiveness_band",
        "audit_priority",
        "local_context_score",
        "mostly_fundamentals_only",
        "candidate_field_status",
        "has_polling",
        "has_incumbency_adjustment",
        "has_candidate_quality_adjustment",
        "has_special_adjustment",
        "has_party_structure",
        "has_party_control_override",
        "dem_candidate",
        "gop_candidate",
        "other_candidate",
        "other",
        "incumbent",
        "incumbent_party",
        "general_election_party_structure",
        "party_control_override",
        "district_type",
        "region",
        "recommended_review",
    ]

    output_cols = [c for c in output_cols if c in df.columns]

    df = df.sort_values(
        by=["audit_priority", "distance_from_50"],
        ascending=[True, True],
    )

    # Put High before Medium before Low.
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    df["_priority_order"] = df["audit_priority"].map(priority_order).fillna(9)
    df = df.sort_values(["_priority_order", "distance_from_50"]).drop(columns=["_priority_order"])

    OUTPUTS.mkdir(exist_ok=True)
    df[output_cols].to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {OUTPUT_PATH}")
    print()
    print("Audit priority counts")
    print("---------------------")
    print(df["audit_priority"].value_counts().to_string())

    print()
    print("Mostly fundamentals-only counts")
    print("-------------------------------")
    print(df["mostly_fundamentals_only"].value_counts().to_string())

    print()
    print("Top review targets")
    print("------------------")
    show_cols = [
        "district_id",
        "rating",
        "dem_win_probability",
        "model_margin_dem",
        "poll_count",
        "competitiveness_band",
        "audit_priority",
        "local_context_score",
        "candidate_field_status",
        "recommended_review",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    print(df[df["audit_priority"].isin(["High", "Medium"])][show_cols].head(40).to_string(index=False))


if __name__ == "__main__":
    main()
