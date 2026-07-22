#!/usr/bin/env python3
"""
House majority-translation analysis.

Purpose
-------
Evaluate how the national environment translated into district winners and
House seat totals, with special attention to the contrast between 2018 and
2020.

This script does not select or implement a production coefficient.

It answers:

1. How many Democratic seats did each multiplier predict?
2. How many Democratic seats actually won?
3. Which baseline partisan categories produced the seat error?
4. Which districts contributed most to the 2020 environment overshoot?
5. Did the national environment move the correct districts across zero?
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

PREDICTIONS_PATH = (
    ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "production_environment_bakeoff"
    / "house_production_environment_bakeoff_predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "majority_translation_analysis"
)

BASELINE_TYPE = "normalized_partisan_baseline_dem"

FOCUS_CYCLES = [2018, 2020]

REPRESENTATIVE_MULTIPLIERS = [
    0.00,
    0.50,
    0.70,
    0.73,
    0.80,
    0.90,
    1.00,
]

PROBABILITY_SCALE = 6.0


def require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}"
        )


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_PATH}"
        )

    frame = pd.read_csv(PREDICTIONS_PATH)

    require_columns(
        frame,
        [
            "baseline_type",
            "cycle",
            "district_id",
            "selected_baseline_margin_dem",
            "district_elasticity",
            "composite_environment_margin_dem",
            "house_environment_multiplier",
            "district_environment_adjustment_dem",
            "forecast_margin_dem",
            "actual_dem_margin",
            "dem_win_probability",
        ],
        "Production environment bakeoff predictions",
    )

    frame["cycle"] = pd.to_numeric(
        frame["cycle"],
        errors="raise",
    ).astype(int)

    numeric_columns = [
        "selected_baseline_margin_dem",
        "district_elasticity",
        "composite_environment_margin_dem",
        "house_environment_multiplier",
        "district_environment_adjustment_dem",
        "forecast_margin_dem",
        "actual_dem_margin",
        "dem_win_probability",
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame[
        (frame["baseline_type"] == BASELINE_TYPE)
        & frame["cycle"].isin(FOCUS_CYCLES)
    ].copy()

    if frame.empty:
        raise ValueError(
            "No normalized-baseline predictions found for "
            f"cycles {FOCUS_CYCLES}."
        )

    frame["actual_dem_win"] = (
        frame["actual_dem_margin"] > 0
    ).astype(int)

    frame["baseline_dem_win"] = (
        frame["selected_baseline_margin_dem"] > 0
    ).astype(int)

    frame["forecast_dem_win"] = (
        frame["forecast_margin_dem"] > 0
    ).astype(int)

    frame["actual_residual_from_baseline"] = (
        frame["actual_dem_margin"]
        - frame["selected_baseline_margin_dem"]
    )

    frame["environment_residual_error_dem"] = (
        frame["district_environment_adjustment_dem"]
        - frame["actual_residual_from_baseline"]
    )

    frame["forecast_error_dem"] = (
        frame["forecast_margin_dem"]
        - frame["actual_dem_margin"]
    )

    return frame


def baseline_partisan_bucket(
    margin: pd.Series,
) -> pd.Categorical:
    bins = [
        -np.inf,
        -20,
        -10,
        -5,
        0,
        5,
        10,
        20,
        np.inf,
    ]

    labels = [
        "R+20 or more",
        "R+10 to R+20",
        "R+5 to R+10",
        "R+0 to R+5",
        "D+0 to D+5",
        "D+5 to D+10",
        "D+10 to D+20",
        "D+20 or more",
    ]

    return pd.cut(
        margin,
        bins=bins,
        labels=labels,
        right=False,
        ordered=True,
    )


def competitiveness_bucket(
    margin: pd.Series,
) -> pd.Categorical:
    absolute = margin.abs()

    bins = [
        -np.inf,
        2.5,
        5,
        10,
        20,
        np.inf,
    ]

    labels = [
        "Tossup: within 2.5",
        "Highly competitive: 2.5-5",
        "Competitive: 5-10",
        "Likely: 10-20",
        "Safe: 20+",
    ]

    return pd.cut(
        absolute,
        bins=bins,
        labels=labels,
        right=False,
        ordered=True,
    )


def nearest_available_multiplier(
    available: np.ndarray,
    target: float,
) -> float:
    index = int(
        np.argmin(
            np.abs(available - target)
        )
    )

    selected = float(available[index])

    if abs(selected - target) > 0.0051:
        raise ValueError(
            f"Requested multiplier {target:.2f} is unavailable. "
            f"Nearest available value is {selected:.2f}."
        )

    return selected


def build_representative_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    available = np.sort(
        predictions[
            "house_environment_multiplier"
        ].dropna().unique()
    )

    selected_multipliers = []

    for target in REPRESENTATIVE_MULTIPLIERS:
        if target == 0.0:
            continue

        selected_multipliers.append(
            nearest_available_multiplier(
                available,
                target,
            )
        )

    selected_multipliers = sorted(
        set(selected_multipliers)
    )

    selected = predictions[
        predictions[
            "house_environment_multiplier"
        ].isin(selected_multipliers)
    ].copy()

    # Construct a baseline-only specification. This is descriptive and
    # uses the same probability mapping as the existing bakeoff.
    baseline = (
        predictions
        .sort_values(
            "house_environment_multiplier"
        )
        .drop_duplicates(
            subset=["cycle", "district_id"],
            keep="first",
        )
        .copy()
    )

    baseline["house_environment_multiplier"] = 0.0
    baseline[
        "district_environment_adjustment_dem"
    ] = 0.0

    baseline["forecast_margin_dem"] = baseline[
        "selected_baseline_margin_dem"
    ]

    baseline["dem_win_probability"] = (
        1.0
        / (
            1.0
            + np.exp(
                -baseline["forecast_margin_dem"]
                / PROBABILITY_SCALE
            )
        )
    )

    baseline["forecast_dem_win"] = (
        baseline["forecast_margin_dem"] > 0
    ).astype(int)

    baseline["environment_residual_error_dem"] = (
        -baseline[
            "actual_residual_from_baseline"
        ]
    )

    baseline["forecast_error_dem"] = (
        baseline["forecast_margin_dem"]
        - baseline["actual_dem_margin"]
    )

    combined = pd.concat(
        [baseline, selected],
        ignore_index=True,
    )

    combined["baseline_partisan_bucket"] = (
        baseline_partisan_bucket(
            combined[
                "selected_baseline_margin_dem"
            ]
        )
    )

    combined["competitiveness_bucket"] = (
        competitiveness_bucket(
            combined[
                "selected_baseline_margin_dem"
            ]
        )
    )

    return combined


def overall_seat_translation(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    for (
        cycle,
        multiplier,
    ), group in frame.groupby(
        [
            "cycle",
            "house_environment_multiplier",
        ],
        sort=True,
    ):
        actual_dem_seats = int(
            group["actual_dem_win"].sum()
        )

        baseline_dem_seats = int(
            group["baseline_dem_win"].sum()
        )

        deterministic_dem_seats = int(
            group["forecast_dem_win"].sum()
        )

        expected_dem_seats = float(
            group["dem_win_probability"].sum()
        )

        baseline_correct = (
            group["baseline_dem_win"]
            == group["actual_dem_win"]
        )

        forecast_correct = (
            group["forecast_dem_win"]
            == group["actual_dem_win"]
        )

        environment_helped = (
            (~baseline_correct)
            & forecast_correct
        )

        environment_hurt = (
            baseline_correct
            & (~forecast_correct)
        )

        records.append(
            {
                "cycle": cycle,
                "house_environment_multiplier": (
                    multiplier
                ),
                "districts": len(group),
                "actual_dem_seats": (
                    actual_dem_seats
                ),
                "baseline_dem_seats": (
                    baseline_dem_seats
                ),
                "deterministic_forecast_dem_seats": (
                    deterministic_dem_seats
                ),
                "expected_dem_seats": (
                    expected_dem_seats
                ),
                "baseline_seat_error": (
                    baseline_dem_seats
                    - actual_dem_seats
                ),
                "deterministic_seat_error": (
                    deterministic_dem_seats
                    - actual_dem_seats
                ),
                "expected_seat_error": (
                    expected_dem_seats
                    - actual_dem_seats
                ),
                "baseline_winner_accuracy": (
                    baseline_correct.mean()
                ),
                "forecast_winner_accuracy": (
                    forecast_correct.mean()
                ),
                "districts_environment_helped": (
                    int(environment_helped.sum())
                ),
                "districts_environment_hurt": (
                    int(environment_hurt.sum())
                ),
                "net_correct_winner_gain": (
                    int(environment_helped.sum())
                    - int(environment_hurt.sum())
                ),
                "margin_mae": (
                    group[
                        "forecast_error_dem"
                    ].abs().mean()
                ),
                "margin_rmse": np.sqrt(
                    np.mean(
                        group[
                            "forecast_error_dem"
                        ] ** 2
                    )
                ),
                "mean_margin_bias_dem": (
                    group[
                        "forecast_error_dem"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(records)


def grouped_translation(
    frame: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    records = []

    grouped = frame.groupby(
        [
            "cycle",
            "house_environment_multiplier",
            group_column,
        ],
        observed=True,
        sort=True,
    )

    for keys, group in grouped:
        cycle, multiplier, category = keys

        actual_dem_seats = int(
            group["actual_dem_win"].sum()
        )

        baseline_dem_seats = int(
            group["baseline_dem_win"].sum()
        )

        forecast_dem_seats = int(
            group["forecast_dem_win"].sum()
        )

        baseline_correct = (
            group["baseline_dem_win"]
            == group["actual_dem_win"]
        )

        forecast_correct = (
            group["forecast_dem_win"]
            == group["actual_dem_win"]
        )

        records.append(
            {
                "cycle": cycle,
                "house_environment_multiplier": (
                    multiplier
                ),
                group_column: str(category),
                "districts": len(group),
                "mean_baseline_margin_dem": (
                    group[
                        "selected_baseline_margin_dem"
                    ].mean()
                ),
                "mean_actual_margin_dem": (
                    group[
                        "actual_dem_margin"
                    ].mean()
                ),
                "mean_actual_residual_from_baseline": (
                    group[
                        "actual_residual_from_baseline"
                    ].mean()
                ),
                "mean_environment_adjustment_dem": (
                    group[
                        "district_environment_adjustment_dem"
                    ].mean()
                ),
                "mean_environment_residual_error_dem": (
                    group[
                        "environment_residual_error_dem"
                    ].mean()
                ),
                "actual_dem_seats": actual_dem_seats,
                "baseline_dem_seats": (
                    baseline_dem_seats
                ),
                "forecast_dem_seats": (
                    forecast_dem_seats
                ),
                "baseline_seat_error": (
                    baseline_dem_seats
                    - actual_dem_seats
                ),
                "forecast_seat_error": (
                    forecast_dem_seats
                    - actual_dem_seats
                ),
                "baseline_winner_accuracy": (
                    baseline_correct.mean()
                ),
                "forecast_winner_accuracy": (
                    forecast_correct.mean()
                ),
                "margin_mae": (
                    group[
                        "forecast_error_dem"
                    ].abs().mean()
                ),
            }
        )

    return pd.DataFrame(records)


def district_outlier_table(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    target_multiplier = nearest_available_multiplier(
        np.sort(
            frame[
                "house_environment_multiplier"
            ].dropna().unique()
        ),
        1.00,
    )

    outliers = frame[
        np.isclose(
            frame[
                "house_environment_multiplier"
            ],
            target_multiplier,
        )
    ].copy()

    outliers["absolute_environment_overshoot"] = (
        outliers[
            "environment_residual_error_dem"
        ].abs()
    )

    outliers["environment_changed_winner"] = (
        outliers["baseline_dem_win"]
        != outliers["forecast_dem_win"]
    )

    outliers["environment_change_correct"] = (
        outliers[
            "environment_changed_winner"
        ]
        & (
            outliers[
                "forecast_dem_win"
            ]
            == outliers[
                "actual_dem_win"
            ]
        )
    )

    outliers["environment_change_incorrect"] = (
        outliers[
            "environment_changed_winner"
        ]
        & (
            outliers[
                "forecast_dem_win"
            ]
            != outliers[
                "actual_dem_win"
            ]
        )
    )

    columns = [
        "cycle",
        "district_id",
        "selected_baseline_margin_dem",
        "composite_environment_margin_dem",
        "district_elasticity",
        "district_environment_adjustment_dem",
        "actual_residual_from_baseline",
        "environment_residual_error_dem",
        "forecast_margin_dem",
        "actual_dem_margin",
        "baseline_dem_win",
        "forecast_dem_win",
        "actual_dem_win",
        "environment_changed_winner",
        "environment_change_correct",
        "environment_change_incorrect",
        "absolute_environment_overshoot",
    ]

    return (
        outliers[columns]
        .sort_values(
            [
                "cycle",
                "absolute_environment_overshoot",
            ],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )


def winner_change_table(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    changed = frame[
        frame["baseline_dem_win"]
        != frame["forecast_dem_win"]
    ].copy()

    changed["change_direction"] = np.where(
        (
            changed["baseline_dem_win"] == 0
        )
        & (
            changed["forecast_dem_win"] == 1
        ),
        "R baseline to D forecast",
        "D baseline to R forecast",
    )

    changed["change_correct"] = (
        changed["forecast_dem_win"]
        == changed["actual_dem_win"]
    )

    columns = [
        "cycle",
        "house_environment_multiplier",
        "district_id",
        "change_direction",
        "change_correct",
        "selected_baseline_margin_dem",
        "district_environment_adjustment_dem",
        "forecast_margin_dem",
        "actual_dem_margin",
        "actual_dem_win",
    ]

    return (
        changed[columns]
        .sort_values(
            [
                "cycle",
                "house_environment_multiplier",
                "change_correct",
                "district_id",
            ],
            ascending=[
                True,
                True,
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def cycle_residual_distribution(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    # Baseline and actual residual are identical across multipliers,
    # so retain only one row per cycle/district.
    unique = (
        frame
        .sort_values(
            "house_environment_multiplier"
        )
        .drop_duplicates(
            subset=["cycle", "district_id"],
            keep="first",
        )
    )

    records = []

    for cycle, group in unique.groupby(
        "cycle",
        sort=True,
    ):
        residual = group[
            "actual_residual_from_baseline"
        ]

        records.append(
            {
                "cycle": cycle,
                "districts": len(group),
                "mean_residual": residual.mean(),
                "median_residual": residual.median(),
                "residual_std": residual.std(),
                "residual_p10": residual.quantile(0.10),
                "residual_p25": residual.quantile(0.25),
                "residual_p50": residual.quantile(0.50),
                "residual_p75": residual.quantile(0.75),
                "residual_p90": residual.quantile(0.90),
                "share_positive_residual": (
                    residual > 0
                ).mean(),
                "share_residual_above_environment": (
                    residual
                    > group[
                        "composite_environment_margin_dem"
                    ]
                ).mean(),
            }
        )

    return pd.DataFrame(records)


def print_table(
    title: str,
    frame: pd.DataFrame,
) -> None:
    print()
    print(title)
    print("=" * 125)
    print(frame.to_string(index=False))


def main() -> int:
    predictions = load_predictions()

    representative = (
        build_representative_predictions(
            predictions
        )
    )

    overall = overall_seat_translation(
        representative
    )

    by_partisanship = grouped_translation(
        representative,
        "baseline_partisan_bucket",
    )

    by_competitiveness = grouped_translation(
        representative,
        "competitiveness_bucket",
    )

    outliers = district_outlier_table(
        representative
    )

    winner_changes = winner_change_table(
        representative
    )

    residual_distribution = (
        cycle_residual_distribution(
            representative
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "house_majority_translation_overall.csv": (
            overall
        ),
        "house_majority_translation_by_partisanship.csv": (
            by_partisanship
        ),
        "house_majority_translation_by_competitiveness.csv": (
            by_competitiveness
        ),
        "house_majority_translation_district_outliers.csv": (
            outliers
        ),
        "house_majority_translation_winner_changes.csv": (
            winner_changes
        ),
        "house_majority_translation_residual_distribution.csv": (
            residual_distribution
        ),
    }

    for filename, frame in outputs.items():
        frame.to_csv(
            OUTPUT_DIR / filename,
            index=False,
        )

    print()
    print("House majority-translation analysis")
    print("=" * 125)
    print(
        f"Focus cycles: {FOCUS_CYCLES}"
    )
    print(
        "Representative prediction rows: "
        f"{len(representative):,}"
    )
    print(
        f"Outputs: {OUTPUT_DIR}"
    )

    print_table(
        "A. OVERALL SEAT AND MAJORITY TRANSLATION",
        overall.round(4),
    )

    print_table(
        "B. ACTUAL RESIDUAL DISTRIBUTION",
        residual_distribution.round(4),
    )

    print_table(
        "C. TRANSLATION BY BASELINE PARTISAN CATEGORY",
        by_partisanship.round(4),
    )

    print_table(
        "D. TRANSLATION BY BASELINE COMPETITIVENESS",
        by_competitiveness.round(4),
    )

    print_table(
        "E. LARGEST DISTRICT-LEVEL ENVIRONMENT MISSES "
        "AT MULTIPLIER 1.00 — TOP 30 PER CYCLE",
        (
            outliers
            .groupby(
                "cycle",
                group_keys=False,
            )
            .head(30)
            .round(4)
        ),
    )

    print_table(
        "F. DISTRICTS WHOSE PREDICTED WINNER CHANGED "
        "FROM THE BASELINE",
        winner_changes.round(4),
    )

    checks = {
        "Both focus cycles present": (
            set(
                representative[
                    "cycle"
                ].unique()
            )
            == set(FOCUS_CYCLES)
        ),
        "No duplicate cycle/district/multiplier rows": (
            not representative.duplicated(
                [
                    "cycle",
                    "district_id",
                    "house_environment_multiplier",
                ]
            ).any()
        ),
        "Overall table covers all representative multipliers": (
            overall[
                "house_environment_multiplier"
            ].nunique()
            == len(
                representative[
                    "house_environment_multiplier"
                ].unique()
            )
        ),
        "All actual outcomes are binary": (
            representative[
                "actual_dem_win"
            ].isin([0, 1]).all()
        ),
        "All forecast outcomes are binary": (
            representative[
                "forecast_dem_win"
            ].isin([0, 1]).all()
        ),
        "Expected seat totals are finite": (
            np.isfinite(
                overall[
                    "expected_dem_seats"
                ]
            ).all()
        ),
    }

    print()
    print("Validation checks")
    print("-" * 125)

    failed = []

    for label, passed in checks.items():
        print(
            f"{'PASS' if passed else 'FAIL'}: "
            f"{label}"
        )

        if not passed:
            failed.append(label)

    if failed:
        print()
        print(
            "House majority-translation analysis: "
            "FAILED"
        )
        return 1

    print()
    print(
        "House majority-translation analysis: "
        "PASSED"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
