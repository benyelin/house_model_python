from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "full_production_replay"
    / "house_production_replay_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "diagnostics"
    / "outputs"
    / "shared_log_loss_audit"
)

SHARED_SPEC = "production_shared_uncertainty_v2"
COMPARISON_SPEC = "production_election_day_v1"

EPSILON = 1e-15


class AuditError(RuntimeError):
    pass


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    numeric_values = pd.to_numeric(
        series,
        errors="coerce",
    )

    text_values = (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return (
        numeric_values.eq(1)
        | text_values.isin(
            {
                "true",
                "t",
                "yes",
                "y",
                "1",
                "dem",
                "d",
            }
        )
    )


def find_first_column(
    frame: pd.DataFrame,
    candidates: list[str],
    *,
    required: bool = True,
) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    if required:
        raise AuditError(
            "None of the expected columns were found: "
            + ", ".join(candidates)
        )

    return None


def select_probability_column(
    frame: pd.DataFrame,
    spec: str,
) -> str:
    preferred = []

    if spec == SHARED_SPEC:
        preferred.extend(
            [
                "simulated_dem_win_probability",
                "dem_win_probability",
                "replay_dem_win_probability",
            ]
        )
    else:
        preferred.extend(
            [
                "dem_win_probability",
                "replay_dem_win_probability",
                "simulated_dem_win_probability",
            ]
        )

    for column in preferred:
        if column not in frame.columns:
            continue

        spec_values = numeric(
            frame.loc[
                frame["replay_spec"].eq(spec),
                column,
            ]
        )

        if spec_values.notna().any():
            return column

    raise AuditError(
        f"Could not find a populated probability column for {spec}."
    )


def row_log_loss(
    actual: pd.Series,
    probability: pd.Series,
) -> pd.Series:
    y = numeric(actual)
    p = numeric(probability).clip(
        EPSILON,
        1.0 - EPSILON,
    )

    return -(
        y * np.log(p)
        + (1.0 - y) * np.log(1.0 - p)
    )


def row_brier(
    actual: pd.Series,
    probability: pd.Series,
) -> pd.Series:
    y = numeric(actual)
    p = numeric(probability)

    return (p - y) ** 2


def calibration_label(probability: float) -> str:
    bins = [
        (0.00, 0.01, "0–1%"),
        (0.01, 0.05, "1–5%"),
        (0.05, 0.10, "5–10%"),
        (0.10, 0.20, "10–20%"),
        (0.20, 0.35, "20–35%"),
        (0.35, 0.50, "35–50%"),
        (0.50, 0.65, "50–65%"),
        (0.65, 0.80, "65–80%"),
        (0.80, 0.90, "80–90%"),
        (0.90, 0.95, "90–95%"),
        (0.95, 0.99, "95–99%"),
        (0.99, 1.01, "99–100%"),
    ]

    for lower, upper, label in bins:
        if lower <= probability < upper:
            return label

    return "Unknown"


def confidence_bucket(probability: float) -> str:
    certainty = max(probability, 1.0 - probability)

    if certainty >= 0.99:
        return "99%+"
    if certainty >= 0.95:
        return "95–99%"
    if certainty >= 0.90:
        return "90–95%"
    if certainty >= 0.80:
        return "80–90%"
    if certainty >= 0.65:
        return "65–80%"

    return "50–65%"


def safe_weighted_mean(
    values: pd.Series,
    weights: pd.Series,
) -> float:
    valid = (
        numeric(values).notna()
        & numeric(weights).notna()
        & numeric(weights).gt(0)
    )

    if not valid.any():
        return math.nan

    return float(
        np.average(
            numeric(values).loc[valid],
            weights=numeric(weights).loc[valid],
        )
    )


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Replay predictions not found: {INPUT_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    required = {
        "replay_spec",
        "cycle",
    }

    missing = required - set(predictions.columns)

    if missing:
        raise AuditError(
            "Replay predictions are missing columns: "
            + ", ".join(sorted(missing))
        )

    race_id_column = find_first_column(
        predictions,
        [
            "district_id",
            "race_id",
        ],
    )

    actual_column = find_first_column(
        predictions,
        [
            "actual_dem_win",
            "actual_winner_dem",
            "dem_won",
            "actual_winner",
        ],
    )

    scoring_column = find_first_column(
        predictions,
        [
            "include_in_scoring",
            "scorable",
        ],
        required=False,
    )

    shared_probability_column = select_probability_column(
        predictions,
        SHARED_SPEC,
    )

    comparison_probability_column = select_probability_column(
        predictions,
        COMPARISON_SPEC,
    )

    selected_specs = predictions.loc[
        predictions["replay_spec"].isin(
            [
                SHARED_SPEC,
                COMPARISON_SPEC,
            ]
        )
    ].copy()

    if selected_specs.empty:
        raise AuditError(
            "Neither required replay specification was found."
        )

    if scoring_column is not None:
        selected_specs = selected_specs.loc[
            parse_bool(
                selected_specs[scoring_column]
            )
        ].copy()

    if selected_specs.empty:
        raise AuditError(
            "No scored rows remain after applying the scoring mask."
        )

    if actual_column == "actual_winner":
        selected_specs["_actual_dem_win"] = (
            selected_specs[actual_column]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
            .eq("D")
            .astype(float)
        )
    else:
        selected_specs["_actual_dem_win"] = (
            parse_bool(
                selected_specs[actual_column]
            ).astype(float)
        )

    metadata_columns = [
        column
        for column in [
            "cycle",
            race_id_column,
            "state",
            "district",
            "actual_dem_margin",
            "model_margin_dem",
            "fundamentals_margin_dem",
            "district_partisan_baseline_dem",
            "general_election_party_structure",
            "party_control_fixed",
            "_actual_dem_win",
        ]
        if column in selected_specs.columns
    ]

    def spec_frame(
        spec: str,
        probability_column: str,
        suffix: str,
    ) -> pd.DataFrame:
        frame = selected_specs.loc[
            selected_specs["replay_spec"].eq(spec)
        ].copy()

        frame[f"probability_{suffix}"] = numeric(
            frame[probability_column]
        )

        frame[f"log_loss_{suffix}"] = row_log_loss(
            frame["_actual_dem_win"],
            frame[f"probability_{suffix}"],
        )

        frame[f"brier_{suffix}"] = row_brier(
            frame["_actual_dem_win"],
            frame[f"probability_{suffix}"],
        )

        frame[
            f"predicted_dem_win_{suffix}"
        ] = frame[f"probability_{suffix}"].ge(0.5)

        frame[
            f"correct_winner_{suffix}"
        ] = (
            frame[
                f"predicted_dem_win_{suffix}"
            ]
            == frame["_actual_dem_win"].astype(bool)
        )

        keep = [
            *metadata_columns,
            f"probability_{suffix}",
            f"log_loss_{suffix}",
            f"brier_{suffix}",
            f"predicted_dem_win_{suffix}",
            f"correct_winner_{suffix}",
        ]

        return frame[keep].copy()

    shared = spec_frame(
        SHARED_SPEC,
        shared_probability_column,
        "shared",
    )

    comparison = spec_frame(
        COMPARISON_SPEC,
        comparison_probability_column,
        "comparison",
    )

    join_keys = [
        "cycle",
        race_id_column,
    ]

    if not shared.duplicated(join_keys).any():
        pass
    else:
        raise AuditError(
            "Shared replay contains duplicate cycle/race keys."
        )

    if comparison.duplicated(join_keys).any():
        raise AuditError(
            "Comparison replay contains duplicate cycle/race keys."
        )

    shared_only_columns = [
        column
        for column in shared.columns
        if column in join_keys
        or column.endswith("_shared")
    ]

    comparison_only_columns = [
        column
        for column in comparison.columns
        if column in join_keys
        or column.endswith("_comparison")
    ]

    metadata = shared[
        metadata_columns
    ].copy()

    merged = metadata.merge(
        shared[shared_only_columns],
        on=join_keys,
        how="inner",
        validate="one_to_one",
    ).merge(
        comparison[comparison_only_columns],
        on=join_keys,
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(shared):
        raise AuditError(
            "Shared and comparison specifications do not cover "
            "the same scored race universe."
        )

    merged[
        "shared_minus_comparison_log_loss"
    ] = (
        merged["log_loss_shared"]
        - merged["log_loss_comparison"]
    )

    merged[
        "shared_minus_comparison_brier"
    ] = (
        merged["brier_shared"]
        - merged["brier_comparison"]
    )

    merged[
        "shared_probability_extremity"
    ] = (
        merged["probability_shared"] - 0.5
    ).abs()

    merged[
        "comparison_probability_extremity"
    ] = (
        merged["probability_comparison"] - 0.5
    ).abs()

    merged[
        "shared_more_extreme_by"
    ] = (
        merged["shared_probability_extremity"]
        - merged["comparison_probability_extremity"]
    )

    merged[
        "shared_confidence_bucket"
    ] = merged[
        "probability_shared"
    ].map(confidence_bucket)

    merged[
        "shared_calibration_bucket"
    ] = merged[
        "probability_shared"
    ].map(calibration_label)

    merged[
        "shared_confident_wrong"
    ] = (
        ~merged["correct_winner_shared"]
        & (
            merged["probability_shared"].ge(0.90)
            | merged["probability_shared"].le(0.10)
        )
    )

    merged[
        "comparison_confident_wrong"
    ] = (
        ~merged["correct_winner_comparison"]
        & (
            merged["probability_comparison"].ge(0.90)
            | merged["probability_comparison"].le(0.10)
        )
    )

    merged[
        "shared_catastrophic_miss"
    ] = (
        ~merged["correct_winner_shared"]
        & (
            merged["probability_shared"].ge(0.95)
            | merged["probability_shared"].le(0.05)
        )
    )

    margin_column = find_first_column(
        merged,
        [
            "actual_dem_margin",
        ],
        required=False,
    )

    if margin_column is not None:
        merged["actual_margin_abs"] = numeric(
            merged[margin_column]
        ).abs()

        merged["actual_close_race"] = (
            merged["actual_margin_abs"] <= 5.0
        )
    else:
        merged["actual_margin_abs"] = np.nan
        merged["actual_close_race"] = False

    # ----------------------------------------------------------
    # Overall and cycle summaries
    # ----------------------------------------------------------

    cycle_rows = []

    for cycle, frame in merged.groupby(
        "cycle",
        sort=True,
    ):
        excess = numeric(
            frame[
                "shared_minus_comparison_log_loss"
            ]
        )

        positive_excess = excess.clip(lower=0.0)

        top_five_excess = (
            positive_excess
            .sort_values(ascending=False)
            .head(5)
            .sum()
        )

        total_positive_excess = positive_excess.sum()

        top_five_share = (
            float(
                top_five_excess
                / total_positive_excess
            )
            if total_positive_excess > 0
            else math.nan
        )

        cycle_rows.append(
            {
                "cycle": int(cycle),
                "scored_races": int(len(frame)),
                "shared_log_loss": float(
                    frame["log_loss_shared"].mean()
                ),
                "comparison_log_loss": float(
                    frame["log_loss_comparison"].mean()
                ),
                "shared_minus_comparison_log_loss": float(
                    excess.mean()
                ),
                "shared_brier": float(
                    frame["brier_shared"].mean()
                ),
                "comparison_brier": float(
                    frame["brier_comparison"].mean()
                ),
                "shared_minus_comparison_brier": float(
                    frame[
                        "shared_minus_comparison_brier"
                    ].mean()
                ),
                "shared_winner_accuracy": float(
                    frame[
                        "correct_winner_shared"
                    ].mean()
                ),
                "comparison_winner_accuracy": float(
                    frame[
                        "correct_winner_comparison"
                    ].mean()
                ),
                "shared_confident_wrong_90_count": int(
                    frame[
                        "shared_confident_wrong"
                    ].sum()
                ),
                "comparison_confident_wrong_90_count": int(
                    frame[
                        "comparison_confident_wrong"
                    ].sum()
                ),
                "shared_catastrophic_miss_95_count": int(
                    frame[
                        "shared_catastrophic_miss"
                    ].sum()
                ),
                "shared_more_extreme_mean": float(
                    frame[
                        "shared_more_extreme_by"
                    ].mean()
                ),
                "positive_excess_log_loss_total": float(
                    total_positive_excess
                ),
                "top_5_positive_excess_log_loss": float(
                    top_five_excess
                ),
                "top_5_share_of_positive_excess": (
                    top_five_share
                ),
            }
        )

    cycle_summary = pd.DataFrame(
        cycle_rows
    )

    overall = pd.DataFrame(
        [
            {
                "cycles": int(
                    merged["cycle"].nunique()
                ),
                "scored_races": int(len(merged)),
                "shared_log_loss": float(
                    merged["log_loss_shared"].mean()
                ),
                "comparison_log_loss": float(
                    merged[
                        "log_loss_comparison"
                    ].mean()
                ),
                "shared_minus_comparison_log_loss": float(
                    merged[
                        "shared_minus_comparison_log_loss"
                    ].mean()
                ),
                "shared_brier": float(
                    merged["brier_shared"].mean()
                ),
                "comparison_brier": float(
                    merged["brier_comparison"].mean()
                ),
                "shared_minus_comparison_brier": float(
                    merged[
                        "shared_minus_comparison_brier"
                    ].mean()
                ),
                "shared_confident_wrong_90_count": int(
                    merged[
                        "shared_confident_wrong"
                    ].sum()
                ),
                "comparison_confident_wrong_90_count": int(
                    merged[
                        "comparison_confident_wrong"
                    ].sum()
                ),
                "shared_catastrophic_miss_95_count": int(
                    merged[
                        "shared_catastrophic_miss"
                    ].sum()
                ),
            }
        ]
    )

    # ----------------------------------------------------------
    # Calibration summary
    # ----------------------------------------------------------

    calibration = (
        merged.groupby(
            [
                "cycle",
                "shared_calibration_bucket",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            races=(
                "_actual_dem_win",
                "size",
            ),
            mean_predicted_dem_probability=(
                "probability_shared",
                "mean",
            ),
            actual_dem_win_rate=(
                "_actual_dem_win",
                "mean",
            ),
            mean_log_loss=(
                "log_loss_shared",
                "mean",
            ),
            mean_brier=(
                "brier_shared",
                "mean",
            ),
        )
    )

    calibration[
        "calibration_error_dem"
    ] = (
        calibration[
            "mean_predicted_dem_probability"
        ]
        - calibration["actual_dem_win_rate"]
    )

    # ----------------------------------------------------------
    # Concentration analysis
    # ----------------------------------------------------------

    ranked = merged.sort_values(
        "shared_minus_comparison_log_loss",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)

    positive = ranked.loc[
        ranked[
            "shared_minus_comparison_log_loss"
        ].gt(0)
    ].copy()

    positive[
        "cumulative_positive_excess_log_loss"
    ] = positive[
        "shared_minus_comparison_log_loss"
    ].cumsum()

    positive_total = positive[
        "shared_minus_comparison_log_loss"
    ].sum()

    if positive_total > 0:
        positive[
            "cumulative_share_of_positive_excess"
        ] = (
            positive[
                "cumulative_positive_excess_log_loss"
            ]
            / positive_total
        )
    else:
        positive[
            "cumulative_share_of_positive_excess"
        ] = np.nan

    concentration_rows = []

    for top_n in [
        1,
        3,
        5,
        10,
        20,
        50,
    ]:
        selected = positive.head(top_n)

        concentration_rows.append(
            {
                "top_n_races": top_n,
                "available_races": int(
                    len(selected)
                ),
                "positive_excess_log_loss": float(
                    selected[
                        "shared_minus_comparison_log_loss"
                    ].sum()
                ),
                "share_of_all_positive_excess": (
                    float(
                        selected[
                            "shared_minus_comparison_log_loss"
                        ].sum()
                        / positive_total
                    )
                    if positive_total > 0
                    else math.nan
                ),
            }
        )

    concentration = pd.DataFrame(
        concentration_rows
    )

    # ----------------------------------------------------------
    # Export
    # ----------------------------------------------------------

    merged.to_csv(
        OUTPUT_DIR
        / "house_shared_log_loss_race_audit.csv",
        index=False,
    )

    ranked.head(100).to_csv(
        OUTPUT_DIR
        / "house_shared_log_loss_top_100_excess.csv",
        index=False,
    )

    merged.loc[
        merged["shared_confident_wrong"]
    ].sort_values(
        "log_loss_shared",
        ascending=False,
    ).to_csv(
        OUTPUT_DIR
        / "house_shared_confident_wrong.csv",
        index=False,
    )

    merged.loc[
        merged["cycle"].eq(2018)
    ].sort_values(
        "shared_minus_comparison_log_loss",
        ascending=False,
    ).to_csv(
        OUTPUT_DIR
        / "house_shared_2018_race_audit.csv",
        index=False,
    )

    cycle_summary.to_csv(
        OUTPUT_DIR
        / "house_shared_log_loss_by_cycle.csv",
        index=False,
    )

    overall.to_csv(
        OUTPUT_DIR
        / "house_shared_log_loss_overall.csv",
        index=False,
    )

    calibration.to_csv(
        OUTPUT_DIR
        / "house_shared_probability_calibration.csv",
        index=False,
    )

    concentration.to_csv(
        OUTPUT_DIR
        / "house_shared_excess_log_loss_concentration.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Console report
    # ----------------------------------------------------------

    display_columns = [
        column
        for column in [
            "cycle",
            race_id_column,
            "state",
            "actual_dem_margin",
            "model_margin_dem",
            "_actual_dem_win",
            "probability_shared",
            "probability_comparison",
            "log_loss_shared",
            "log_loss_comparison",
            "shared_minus_comparison_log_loss",
            "shared_more_extreme_by",
            "shared_confident_wrong",
        ]
        if column in ranked.columns
    ]

    print("HOUSE SHARED-SIMULATOR LOG-LOSS AUDIT")
    print("=" * 100)
    print(f"Input: {INPUT_PATH}")
    print(
        f"Shared probability column: "
        f"{shared_probability_column}"
    )
    print(
        f"Comparison probability column: "
        f"{comparison_probability_column}"
    )
    print()

    print("OVERALL")
    print("-" * 100)
    print(
        overall.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("BY CYCLE")
    print("-" * 100)
    print(
        cycle_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("EXCESS LOG-LOSS CONCENTRATION")
    print("-" * 100)
    print(
        concentration.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("TOP 25 RACES WHERE SHARED LOG LOSS IS WORSE")
    print("-" * 100)
    print(
        ranked[
            display_columns
        ]
        .head(25)
        .to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("2018 TOP 20 EXCESS LOG-LOSS RACES")
    print("-" * 100)

    ranked_2018 = ranked.loc[
        ranked["cycle"].eq(2018)
    ]

    print(
        ranked_2018[
            display_columns
        ]
        .head(20)
        .to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("OUTPUTS")
    print("-" * 100)

    for output in sorted(
        OUTPUT_DIR.glob("*.csv")
    ):
        print(
            f"  {output.relative_to(PROJECT_ROOT)}"
        )

    print()
    print("VALIDATION PASSED")
    print(
        "Shared and comparison specifications were evaluated "
        "on an identical scored race universe."
    )


if __name__ == "__main__":
    main()
