from pathlib import Path
import pandas as pd
import numpy as np

INPUTS = Path("inputs")
OUTPUTS = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)

OUT = OUTPUTS / "house_candidate_war_rankings.csv"

INPUT_CANDIDATES = [
    INPUTS / "house_race_inputs.csv",
    INPUTS / "district_inputs.csv",
    INPUTS / "house_district_inputs.csv",
    INPUTS / "race_inputs.csv",
]

RESULT_CANDIDATES = [
    OUTPUTS / "house_model_results.csv",
    OUTPUTS / "house_forecast_results.csv",
    OUTPUTS / "district_results.csv",
    OUTPUTS / "race_results.csv",
]


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def read_csv(path):
    if path and path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def district_key(df):
    for col in ["district_id", "district", "race", "seat", "district_name"]:
        if col in df.columns:
            return col
    return None


def safe_num(s, default=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(default)


def logistic_prob(margin, sigma=6.0):
    """
    Approximate win probability from margin.
    This is only for WAR diagnostics, not the core model simulation.
    """
    return 1 / (1 + np.exp(-margin / sigma))


def main():
    input_path = first_existing(INPUT_CANDIDATES)
    result_path = first_existing(RESULT_CANDIDATES)

    inputs = read_csv(input_path)
    results = read_csv(result_path)

    if inputs.empty:
        raise FileNotFoundError(
            "Could not find House input file. Tried: "
            + ", ".join(str(p) for p in INPUT_CANDIDATES)
        )

    key = district_key(inputs)
    if key is None:
        raise ValueError("Could not identify district key column in House inputs.")

    inputs[key] = inputs[key].astype(str).str.strip()

    df = inputs.copy()

    # Merge model results if available.
    if not results.empty:
        rkey = district_key(results)
        if rkey:
            results[rkey] = results[rkey].astype(str).str.strip()

            result_cols = [rkey] + [
                c for c in results.columns
                if c in [
                    "dem_win_prob",
                    "prob_dem_win",
                    "gop_win_prob",
                    "rep_win_prob",
                    "prob_rep_win",
                    "margin_dem",
                    "mean_margin_dem",
                    "median_margin_dem",
                    "model_margin_dem",
                    "expected_margin_dem",
                    "fundamentals_margin_dem",
                ]
            ]

            df = df.merge(
                results[result_cols].drop_duplicates(rkey),
                left_on=key,
                right_on=rkey,
                how="left",
            )

            if rkey != key and rkey in df.columns:
                df = df.drop(columns=[rkey])

    # Candidate names, if available.
    for col in ["dem_candidate", "gop_candidate", "rep_candidate"]:
        if col not in df.columns:
            df[col] = ""

    if "gop_candidate" not in df.columns and "rep_candidate" in df.columns:
        df["gop_candidate"] = df["rep_candidate"]

    # Candidate adjustment detection.
    possible_net_adjustment_cols = [
        "candidate_quality_adjustment_dem",
        "candidate_adjustment_dem",
        "candidate_war_adjustment_dem",
        "candidate_strength_adjustment_dem",
        "dem_candidate_quality_adjustment_net",
        "candidate_margin_adjustment_dem",
    ]

    net_col = next((c for c in possible_net_adjustment_cols if c in df.columns), None)

    if net_col:
        df["candidate_net_margin_adjustment_dem"] = safe_num(df[net_col], 0.0)
        adjustment_source = net_col
    else:
        # Try to build net adjustment from separate D/GOP fields.
        dem_cols = [
            c for c in [
                "dem_candidate_quality_adjustment",
                "dem_candidate_strength",
                "dem_candidate_score",
                "dem_war",
            ]
            if c in df.columns
        ]

        gop_cols = [
            c for c in [
                "gop_candidate_quality_adjustment",
                "rep_candidate_quality_adjustment",
                "gop_candidate_strength",
                "rep_candidate_strength",
                "gop_candidate_score",
                "rep_candidate_score",
                "gop_war",
                "rep_war",
            ]
            if c in df.columns
        ]

        if dem_cols and gop_cols:
            dcol = dem_cols[0]
            rcol = gop_cols[0]
            df["candidate_net_margin_adjustment_dem"] = safe_num(df[dcol], 0.0) - safe_num(df[rcol], 0.0)
            adjustment_source = f"{dcol} minus {rcol}"
        else:
            df["candidate_net_margin_adjustment_dem"] = 0.0
            adjustment_source = "none_found"

    # Baseline margin detection.
    margin_cols = [
        c for c in [
            "model_margin_dem",
            "mean_margin_dem",
            "median_margin_dem",
            "margin_dem",
            "expected_margin_dem",
            "fundamentals_margin_dem",
        ]
        if c in df.columns
    ]

    if margin_cols:
        margin_col = margin_cols[0]
        df["current_margin_dem"] = safe_num(df[margin_col], 0.0)
    else:
        margin_col = "none_found"
        df["current_margin_dem"] = 0.0

    # Estimate probability impact of candidate adjustment.
    # If candidate adjustment is +D, remove it to estimate replacement-level baseline.
    df["replacement_level_margin_dem"] = (
        df["current_margin_dem"] - df["candidate_net_margin_adjustment_dem"]
    )

    df["prob_dem_with_candidate"] = logistic_prob(df["current_margin_dem"])
    df["prob_dem_replacement_level"] = logistic_prob(df["replacement_level_margin_dem"])

    df["candidate_win_probability_added_dem"] = (
        df["prob_dem_with_candidate"] - df["prob_dem_replacement_level"]
    )

    # Seat WAR is the absolute probability impact; party-benefit labels show who benefits.
    df["candidate_war_seats"] = df["candidate_win_probability_added_dem"].abs()

    df["candidate_benefiting_party"] = np.where(
        df["candidate_win_probability_added_dem"] > 0,
        "D",
        np.where(df["candidate_win_probability_added_dem"] < 0, "R", "None"),
    )

    df["candidate_war_notes"] = np.where(
        df["candidate_net_margin_adjustment_dem"].eq(0),
        "No candidate adjustment detected or adjustment is zero.",
        "WAR estimate is an approximate diagnostic based on model margin and candidate margin adjustment.",
    )

    df["candidate_adjustment_source"] = adjustment_source
    df["margin_source"] = margin_col

    output_cols = [
        key,
        "state",
        "dem_candidate",
        "gop_candidate",
        "candidate_benefiting_party",
        "candidate_war_seats",
        "candidate_win_probability_added_dem",
        "candidate_net_margin_adjustment_dem",
        "current_margin_dem",
        "replacement_level_margin_dem",
        "prob_dem_with_candidate",
        "prob_dem_replacement_level",
        "candidate_adjustment_source",
        "margin_source",
        "candidate_war_notes",
    ]

    output_cols = [c for c in output_cols if c in df.columns]

    out = df[output_cols].copy()

    out = out.sort_values(
        ["candidate_war_seats", "candidate_net_margin_adjustment_dem"],
        ascending=[False, False],
    )

    out.to_csv(OUT, index=False)

    print(f"Built House candidate WAR rankings: {OUT}")
    print(f"Rows: {len(out)}")
    print(f"Candidate adjustment source: {adjustment_source}")
    print(f"Margin source: {margin_col}")
    print()
    print(out.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
