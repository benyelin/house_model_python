from pathlib import Path
import pandas as pd
import numpy as np

INPUTS = Path("inputs")
OUTPUTS = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)

DISTRICT_INPUTS_CANDIDATES = [
    INPUTS / "house_race_inputs.csv",
    INPUTS / "district_inputs.csv",
    INPUTS / "house_district_inputs.csv",
    INPUTS / "race_inputs.csv",
]

MODEL_RESULTS_CANDIDATES = [
    OUTPUTS / "house_model_results.csv",
    OUTPUTS / "house_forecast_results.csv",
    OUTPUTS / "district_results.csv",
    OUTPUTS / "race_results.csv",
]

MANUAL_POLLS = INPUTS / "house_manual_polls.csv"
ADJUSTED_POLLS = INPUTS / "house_manual_polls_adjusted.csv"

OUT = OUTPUTS / "house_race_review_queue.csv"


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def read_csv(path):
    if path and path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def safe_num(s, default=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(default)


def district_key(df):
    for col in ["district_id", "district", "race", "seat", "district_name"]:
        if col in df.columns:
            return col
    return None


def add_reason(row, condition, points, reason):
    if condition:
        row["review_score"] += points
        row["review_reasons"].append(reason)


def main():
    district_path = first_existing(DISTRICT_INPUTS_CANDIDATES)
    results_path = first_existing(MODEL_RESULTS_CANDIDATES)

    districts = read_csv(district_path)
    results = read_csv(results_path)
    manual_polls = read_csv(MANUAL_POLLS)
    adjusted_polls = read_csv(ADJUSTED_POLLS)

    if districts.empty:
        raise FileNotFoundError(
            "Could not find House district/race inputs. Tried: "
            + ", ".join(str(p) for p in DISTRICT_INPUTS_CANDIDATES)
        )

    dkey = district_key(districts)
    if dkey is None:
        raise ValueError("Could not identify district key column in House inputs.")

    districts[dkey] = districts[dkey].astype(str).str.strip()

    queue = districts.copy()

    # Merge model results.
    if not results.empty:
        rkey = district_key(results)
        if rkey:
            results[rkey] = results[rkey].astype(str).str.strip()

            result_cols = [rkey] + [
                c for c in results.columns
                if c in [
                    "dem_win_prob",
                    "gop_win_prob",
                    "rep_win_prob",
                    "prob_dem_win",
                    "prob_rep_win",
                    "margin_dem",
                    "mean_margin_dem",
                    "median_margin_dem",
                    "model_margin_dem",
                    "expected_margin_dem",
                ]
            ]

            queue = queue.merge(
                results[result_cols].drop_duplicates(rkey),
                left_on=dkey,
                right_on=rkey,
                how="left",
            )

            if rkey != dkey and rkey in queue.columns:
                queue = queue.drop(columns=[rkey])

    # Poll counts.
    if not manual_polls.empty:
        pkey = district_key(manual_polls)
        if pkey:
            manual_polls[pkey] = manual_polls[pkey].astype(str).str.strip()

            poll_counts = manual_polls.groupby(pkey).size().reset_index(name="manual_poll_count")
            queue = queue.merge(poll_counts, left_on=dkey, right_on=pkey, how="left")

            if pkey != dkey and pkey in queue.columns:
                queue = queue.drop(columns=[pkey])
        else:
            queue["manual_poll_count"] = 0
    else:
        queue["manual_poll_count"] = 0

    queue["manual_poll_count"] = safe_num(queue["manual_poll_count"], 0)

    # Partisan/internal poll counts.
    if not manual_polls.empty:
        pkey = district_key(manual_polls)

        if pkey:
            mp = manual_polls.copy()

            for col in [
                "poll_sponsor_type",
                "partisan_sponsor_party",
                "pollster_partisan_affiliation",
                "is_internal_poll",
            ]:
                if col not in mp.columns:
                    mp[col] = ""

            mp["is_internal_poll_bool"] = (
                mp["is_internal_poll"]
                .fillna(False)
                .astype(str)
                .str.lower()
                .isin(["true", "1", "yes", "y"])
            )

            mp["has_partisan_metadata"] = (
                mp["partisan_sponsor_party"].fillna("").astype(str).str.upper().isin(["D", "R"])
                | mp["pollster_partisan_affiliation"].fillna("").astype(str).str.upper().isin(["D", "R"])
                | mp["is_internal_poll_bool"]
                | mp["poll_sponsor_type"].fillna("").astype(str).str.lower().isin(["party", "campaign", "super pac"])
            )

            partisan_counts = (
                mp.groupby(pkey)["has_partisan_metadata"]
                .sum()
                .reset_index(name="partisan_or_internal_poll_count")
            )

            queue = queue.merge(partisan_counts, left_on=dkey, right_on=pkey, how="left")

            if pkey != dkey and pkey in queue.columns:
                queue = queue.drop(columns=[pkey])
        else:
            queue["partisan_or_internal_poll_count"] = 0
    else:
        queue["partisan_or_internal_poll_count"] = 0

    queue["partisan_or_internal_poll_count"] = safe_num(queue["partisan_or_internal_poll_count"], 0)

    # Independent/third party poll signal.
    if not manual_polls.empty:
        pkey = district_key(manual_polls)

        if pkey:
            mp = manual_polls.copy()

            for col in ["ind_pct", "other_pct"]:
                if col not in mp.columns:
                    mp[col] = 0.0
                mp[col] = safe_num(mp[col], 0.0)

            mp["third_party_poll_share"] = mp["ind_pct"] + mp["other_pct"]

            third_party = (
                mp.groupby(pkey)
                .agg(
                    max_third_party_poll_share=("third_party_poll_share", "max"),
                    avg_third_party_poll_share=("third_party_poll_share", "mean"),
                )
                .reset_index()
            )

            queue = queue.merge(third_party, left_on=dkey, right_on=pkey, how="left")

            if pkey != dkey and pkey in queue.columns:
                queue = queue.drop(columns=[pkey])
        else:
            queue["max_third_party_poll_share"] = 0.0
            queue["avg_third_party_poll_share"] = 0.0
    else:
        queue["max_third_party_poll_share"] = 0.0
        queue["avg_third_party_poll_share"] = 0.0

    queue["max_third_party_poll_share"] = safe_num(queue["max_third_party_poll_share"], 0)
    queue["avg_third_party_poll_share"] = safe_num(queue["avg_third_party_poll_share"], 0)

    # Competitiveness.
    prob_cols = [c for c in ["dem_win_prob", "prob_dem_win"] if c in queue.columns]

    if prob_cols:
        pcol = prob_cols[0]
        queue[pcol] = safe_num(queue[pcol], np.nan)
        queue["competitiveness"] = 1 - (queue[pcol] - 0.5).abs() * 2
    else:
        margin_cols = [
            c for c in [
                "margin_dem",
                "mean_margin_dem",
                "median_margin_dem",
                "model_margin_dem",
                "expected_margin_dem",
            ]
            if c in queue.columns
        ]

        if margin_cols:
            mcol = margin_cols[0]
            queue[mcol] = safe_num(queue[mcol], 0)
            queue["competitiveness"] = (1 - (queue[mcol].abs() / 20)).clip(lower=0, upper=1)
        else:
            queue["competitiveness"] = 0

    # Optional data gap / review fields.
    rationale_cols = [
        c for c in [
            "incumbency_rationale",
            "candidate_quality_rationale",
            "overperformance_rationale",
            "liability_rationale",
            "special_adjustment_rationale",
            "independent_adjustment_rationale",
            "human_review_status",
            "last_human_review_date",
        ]
        if c in queue.columns
    ]

    review_rows = []

    for _, source_row in queue.iterrows():
        row = source_row.copy()
        row["review_score"] = 0
        row["review_reasons"] = []

        competitiveness = float(row.get("competitiveness", 0) or 0)
        poll_count = float(row.get("manual_poll_count", 0) or 0)
        partisan_poll_count = float(row.get("partisan_or_internal_poll_count", 0) or 0)
        max_third_party_poll_share = float(row.get("max_third_party_poll_share", 0) or 0)

        add_reason(
            row,
            competitiveness >= 0.75,
            4,
            "Highly competitive district",
        )

        add_reason(
            row,
            competitiveness >= 0.50 and poll_count == 0,
            3,
            "Competitive district with no manual polls",
        )

        add_reason(
            row,
            poll_count > 0,
            1,
            "Manual polling is present",
        )

        add_reason(
            row,
            partisan_poll_count > 0,
            3,
            "Partisan/internal poll metadata present",
        )

        add_reason(
            row,
            max_third_party_poll_share >= 5,
            3,
            "Polls show notable independent/third-party vote share",
        )

        if "human_review_status" in row.index:
            add_reason(
                row,
                str(row.get("human_review_status", "")).strip().lower() in ["needs review", "review", "pending"],
                3,
                "Human review status indicates review needed",
            )

        if rationale_cols:
            missing_review_info = any(
                str(row.get(col, "")).strip() in ["", "nan", "None"]
                for col in rationale_cols
                if col not in ["last_human_review_date"]
            )

            add_reason(
                row,
                competitiveness >= 0.75 and missing_review_info,
                2,
                "Highly competitive district with incomplete review/rationale fields",
            )

        if row["review_score"] > 0:
            row["review_reasons"] = "; ".join(row["review_reasons"])
            review_rows.append(row)

    if review_rows:
        review_df = pd.DataFrame(review_rows)
    else:
        review_df = queue.head(0).copy()
        review_df["review_score"] = []
        review_df["review_reasons"] = []

    output_cols = [
        dkey,
        "state",
        "review_score",
        "review_reasons",
        "competitiveness",
        "manual_poll_count",
        "partisan_or_internal_poll_count",
        "max_third_party_poll_share",
        "avg_third_party_poll_share",
        "human_review_status",
        "last_human_review_date",
    ]

    output_cols = [c for c in output_cols if c in review_df.columns]

    review_df = review_df.sort_values(
        ["review_score", "competitiveness"],
        ascending=[False, False],
    )

    review_df[output_cols].to_csv(OUT, index=False)

    print(f"Built House race review queue: {OUT}")
    print(f"Rows: {len(review_df)}")
    if len(review_df):
        print(review_df[output_cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
