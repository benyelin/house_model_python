#!/usr/bin/env python3
"""
Majority-focused House national-environment coefficient bakeoff.

Purpose
-------
Select the House environment multiplier using objectives aligned with
forecasting district winners and seat totals rather than primarily minimizing
district-margin error.

The historical population contains the common scorable districts available in
the production environment bakeoff. Therefore:

- Seat errors are measured within that common scorable population.
- The script does not claim to reconstruct literal 218-seat chamber control.
- Comparisons across multipliers remain valid because every multiplier is
  evaluated on the same districts within each cycle.

Primary objectives
------------------
Lower is better for the composite score.

    35% winner error rate
    25% Brier score
    20% expected seat error rate
    15% deterministic seat error rate
     5% margin MAE

Each raw metric is min-max normalized across candidate multipliers before the
weights are applied. Cycles receive equal weight in the pooled recommendation.

The script also performs leave-one-cycle-out selection and applies a
simplicity/shared-architecture preference toward multiplier 1.00 when it lies
within the near-best tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    / "majority_environment_bakeoff"
)

BASELINE_TYPE = "normalized_partisan_baseline_dem"

EXPECTED_CYCLES = [2016, 2018, 2020, 2022]

# A composite-score difference this small is treated as practically near-best.
# Since the component scores are normalized to [0, 1], 0.05 is a modest but
# meaningful tolerance.
NEAR_BEST_TOLERANCE = 0.05

SHARED_MULTIPLIER = 1.00

METRIC_WEIGHTS = {
    "winner_error_rate": 0.35,
    "brier_score": 0.25,
    "expected_seat_error_rate": 0.20,
    "deterministic_seat_error_rate": 0.15,
    "margin_mae": 0.05,
}


@dataclass(frozen=True)
class Recommendation:
    best_multiplier: float
    best_score: float
    near_best_multipliers: tuple[float, ...]
    recommended_multiplier: float
    recommendation_reason: str


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
            "house_environment_multiplier",
            "forecast_margin_dem",
            "actual_dem_margin",
            "dem_win_probability",
        ],
        "Production environment bakeoff predictions",
    )

    frame = frame[
        frame["baseline_type"] == BASELINE_TYPE
    ].copy()

    if frame.empty:
        raise ValueError(
            "No rows found for baseline type "
            f"{BASELINE_TYPE!r}."
        )

    frame["cycle"] = pd.to_numeric(
        frame["cycle"],
        errors="raise",
    ).astype(int)

    numeric_columns = [
        "house_environment_multiplier",
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
        frame["cycle"].isin(EXPECTED_CYCLES)
    ].copy()

    frame = frame.dropna(
        subset=[
            "house_environment_multiplier",
            "forecast_margin_dem",
            "actual_dem_margin",
            "dem_win_probability",
        ]
    )

    frame["actual_dem_win"] = (
        frame["actual_dem_margin"] > 0
    ).astype(int)

    frame["forecast_dem_win"] = (
        frame["forecast_margin_dem"] > 0
    ).astype(int)

    frame["forecast_error_dem"] = (
        frame["forecast_margin_dem"]
        - frame["actual_dem_margin"]
    )

    frame["squared_probability_error"] = (
        frame["dem_win_probability"]
        - frame["actual_dem_win"]
    ) ** 2

    duplicate_keys = [
        "cycle",
        "district_id",
        "house_environment_multiplier",
    ]

    if frame.duplicated(duplicate_keys).any():
        duplicates = frame[
            frame.duplicated(
                duplicate_keys,
                keep=False,
            )
        ].sort_values(duplicate_keys)

        optional_dimensions = [
            column
            for column in [
                "model_family",
                "environment_model",
                "formula_name",
                "scenario",
            ]
            if column in frame.columns
        ]

        detail = (
            duplicates[
                duplicate_keys
                + optional_dimensions
            ]
            .head(20)
            .to_string(index=False)
        )

        raise ValueError(
            "Duplicate cycle/district/multiplier prediction rows "
            "remain after filtering. An additional model-family filter "
            "may be required.\n\n"
            + detail
        )

    return frame


def calculate_cycle_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, float | int]] = []

    grouped = predictions.groupby(
        [
            "cycle",
            "house_environment_multiplier",
        ],
        sort=True,
    )

    for (
        cycle,
        multiplier,
    ), group in grouped:
        districts = len(group)

        actual_dem_seats = int(
            group["actual_dem_win"].sum()
        )

        deterministic_dem_seats = int(
            group["forecast_dem_win"].sum()
        )

        expected_dem_seats = float(
            group["dem_win_probability"].sum()
        )

        deterministic_seat_error = (
            deterministic_dem_seats
            - actual_dem_seats
        )

        expected_seat_error = (
            expected_dem_seats
            - actual_dem_seats
        )

        winner_accuracy = float(
            (
                group["forecast_dem_win"]
                == group["actual_dem_win"]
            ).mean()
        )

        records.append(
            {
                "cycle": int(cycle),
                "house_environment_multiplier": float(
                    multiplier
                ),
                "districts": districts,
                "actual_dem_seats": actual_dem_seats,
                "deterministic_dem_seats": (
                    deterministic_dem_seats
                ),
                "expected_dem_seats": (
                    expected_dem_seats
                ),
                "deterministic_seat_error": (
                    deterministic_seat_error
                ),
                "absolute_deterministic_seat_error": abs(
                    deterministic_seat_error
                ),
                "deterministic_seat_error_rate": (
                    abs(deterministic_seat_error)
                    / districts
                ),
                "expected_seat_error": (
                    expected_seat_error
                ),
                "absolute_expected_seat_error": abs(
                    expected_seat_error
                ),
                "expected_seat_error_rate": (
                    abs(expected_seat_error)
                    / districts
                ),
                "winner_accuracy": winner_accuracy,
                "winner_error_rate": (
                    1.0 - winner_accuracy
                ),
                "brier_score": float(
                    group[
                        "squared_probability_error"
                    ].mean()
                ),
                "margin_mae": float(
                    group[
                        "forecast_error_dem"
                    ].abs().mean()
                ),
                "margin_rmse": float(
                    np.sqrt(
                        np.mean(
                            group[
                                "forecast_error_dem"
                            ] ** 2
                        )
                    )
                ),
                "margin_bias_dem": float(
                    group[
                        "forecast_error_dem"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(records)


def aggregate_metrics(
    cycle_metrics: pd.DataFrame,
    cycles: list[int] | None = None,
) -> pd.DataFrame:
    frame = cycle_metrics.copy()

    if cycles is not None:
        frame = frame[
            frame["cycle"].isin(cycles)
        ].copy()

    if frame.empty:
        raise ValueError(
            "No cycle metrics remain for aggregation."
        )

    metric_columns = [
        "winner_accuracy",
        "winner_error_rate",
        "brier_score",
        "absolute_expected_seat_error",
        "expected_seat_error_rate",
        "absolute_deterministic_seat_error",
        "deterministic_seat_error_rate",
        "margin_mae",
        "margin_rmse",
        "margin_bias_dem",
    ]

    pooled = (
        frame
        .groupby(
            "house_environment_multiplier",
            as_index=False,
            sort=True,
        )[metric_columns]
        .mean()
    )

    pooled = pooled.rename(
        columns={
            column: f"mean_cycle_{column}"
            for column in metric_columns
        }
    )

    pooled["cycles"] = frame["cycle"].nunique()

    return pooled


def minmax_penalty(
    values: pd.Series,
) -> pd.Series:
    minimum = float(values.min())
    maximum = float(values.max())

    if np.isclose(
        minimum,
        maximum,
    ):
        return pd.Series(
            np.zeros(len(values)),
            index=values.index,
            dtype=float,
        )

    return (
        values - minimum
    ) / (
        maximum - minimum
    )


def add_composite_score(
    pooled: pd.DataFrame,
) -> pd.DataFrame:
    scored = pooled.copy()

    source_columns = {
        "winner_error_rate": (
            "mean_cycle_winner_error_rate"
        ),
        "brier_score": (
            "mean_cycle_brier_score"
        ),
        "expected_seat_error_rate": (
            "mean_cycle_expected_seat_error_rate"
        ),
        "deterministic_seat_error_rate": (
            "mean_cycle_deterministic_seat_error_rate"
        ),
        "margin_mae": (
            "mean_cycle_margin_mae"
        ),
    }

    score = pd.Series(
        np.zeros(len(scored)),
        index=scored.index,
        dtype=float,
    )

    for metric, weight in METRIC_WEIGHTS.items():
        source_column = source_columns[metric]
        penalty_column = f"{metric}_normalized_penalty"

        scored[penalty_column] = minmax_penalty(
            scored[source_column]
        )

        score = (
            score
            + weight
            * scored[penalty_column]
        )

    scored["majority_composite_score"] = score

    scored["composite_rank"] = (
        scored["majority_composite_score"]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )

    scored["winner_accuracy_rank"] = (
        scored["mean_cycle_winner_accuracy"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    scored["brier_rank"] = (
        scored["mean_cycle_brier_score"]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )

    scored["expected_seat_error_rank"] = (
        scored[
            "mean_cycle_expected_seat_error_rate"
        ]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )

    scored["deterministic_seat_error_rank"] = (
        scored[
            "mean_cycle_deterministic_seat_error_rate"
        ]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )

    scored["margin_mae_rank"] = (
        scored["mean_cycle_margin_mae"]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )

    return scored.sort_values(
        [
            "majority_composite_score",
            "house_environment_multiplier",
        ]
    ).reset_index(drop=True)


def build_recommendation(
    scored: pd.DataFrame,
) -> Recommendation:
    ordered = scored.sort_values(
        [
            "majority_composite_score",
            "house_environment_multiplier",
        ]
    ).reset_index(drop=True)

    best_row = ordered.iloc[0]

    best_multiplier = float(
        best_row[
            "house_environment_multiplier"
        ]
    )

    best_score = float(
        best_row["majority_composite_score"]
    )

    near_best = ordered[
        ordered["majority_composite_score"]
        <= best_score + NEAR_BEST_TOLERANCE
    ].copy()

    near_best_multipliers = tuple(
        sorted(
            float(value)
            for value in near_best[
                "house_environment_multiplier"
            ].tolist()
        )
    )

    shared_rows = near_best[
        np.isclose(
            near_best[
                "house_environment_multiplier"
            ],
            SHARED_MULTIPLIER,
            atol=0.0051,
        )
    ]

    if not shared_rows.empty:
        recommended_multiplier = float(
            shared_rows.iloc[0][
                "house_environment_multiplier"
            ]
        )

        reason = (
            "The shared 1.00 multiplier is within the "
            f"{NEAR_BEST_TOLERANCE:.2f} near-best composite-score "
            "tolerance, so it is preferred to avoid an additional "
            "House-specific attenuation layer."
        )
    else:
        near_best = near_best.copy()

        near_best["distance_from_shared_multiplier"] = (
            near_best[
                "house_environment_multiplier"
            ]
            - SHARED_MULTIPLIER
        ).abs()

        selected = near_best.sort_values(
            [
                "distance_from_shared_multiplier",
                "majority_composite_score",
            ]
        ).iloc[0]

        recommended_multiplier = float(
            selected[
                "house_environment_multiplier"
            ]
        )

        reason = (
            "The shared 1.00 multiplier is outside the near-best "
            "tolerance. The recommendation is therefore the near-best "
            "multiplier closest to 1.00."
        )

    return Recommendation(
        best_multiplier=best_multiplier,
        best_score=best_score,
        near_best_multipliers=(
            near_best_multipliers
        ),
        recommended_multiplier=(
            recommended_multiplier
        ),
        recommendation_reason=reason,
    )


def select_training_multiplier(
    cycle_metrics: pd.DataFrame,
    training_cycles: list[int],
) -> tuple[float, pd.DataFrame]:
    training_pooled = aggregate_metrics(
        cycle_metrics,
        cycles=training_cycles,
    )

    training_scored = add_composite_score(
        training_pooled
    )

    recommendation = build_recommendation(
        training_scored
    )

    return (
        recommendation.recommended_multiplier,
        training_scored,
    )


def build_leave_one_cycle_out(
    cycle_metrics: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []

    all_cycles = sorted(
        int(value)
        for value in cycle_metrics[
            "cycle"
        ].unique()
    )

    for holdout_cycle in all_cycles:
        training_cycles = [
            cycle
            for cycle in all_cycles
            if cycle != holdout_cycle
        ]

        (
            selected_multiplier,
            training_scored,
        ) = select_training_multiplier(
            cycle_metrics,
            training_cycles,
        )

        best_training_row = (
            training_scored
            .sort_values(
                "majority_composite_score"
            )
            .iloc[0]
        )

        selected_training_row = training_scored[
            np.isclose(
                training_scored[
                    "house_environment_multiplier"
                ],
                selected_multiplier,
                atol=0.0051,
            )
        ].iloc[0]

        heldout = cycle_metrics[
            (
                cycle_metrics["cycle"]
                == holdout_cycle
            )
            & np.isclose(
                cycle_metrics[
                    "house_environment_multiplier"
                ],
                selected_multiplier,
                atol=0.0051,
            )
        ]

        if len(heldout) != 1:
            raise ValueError(
                "Expected exactly one held-out row for "
                f"cycle={holdout_cycle}, "
                f"multiplier={selected_multiplier:.2f}; "
                f"found {len(heldout)}."
            )

        row = heldout.iloc[0]

        records.append(
            {
                "holdout_cycle": holdout_cycle,
                "training_cycles": ",".join(
                    str(cycle)
                    for cycle in training_cycles
                ),
                "selected_multiplier": (
                    selected_multiplier
                ),
                "training_best_numeric_multiplier": float(
                    best_training_row[
                        "house_environment_multiplier"
                    ]
                ),
                "training_best_composite_score": float(
                    best_training_row[
                        "majority_composite_score"
                    ]
                ),
                "selected_training_composite_score": float(
                    selected_training_row[
                        "majority_composite_score"
                    ]
                ),
                "heldout_districts": int(
                    row["districts"]
                ),
                "heldout_actual_dem_seats": int(
                    row["actual_dem_seats"]
                ),
                "heldout_deterministic_dem_seats": int(
                    row["deterministic_dem_seats"]
                ),
                "heldout_expected_dem_seats": float(
                    row["expected_dem_seats"]
                ),
                "heldout_deterministic_seat_error": float(
                    row[
                        "deterministic_seat_error"
                    ]
                ),
                "heldout_expected_seat_error": float(
                    row["expected_seat_error"]
                ),
                "heldout_winner_accuracy": float(
                    row["winner_accuracy"]
                ),
                "heldout_brier_score": float(
                    row["brier_score"]
                ),
                "heldout_margin_mae": float(
                    row["margin_mae"]
                ),
                "heldout_margin_bias_dem": float(
                    row["margin_bias_dem"]
                ),
            }
        )

    return pd.DataFrame(records)


def build_multiplier_selection_frequency(
    loo: pd.DataFrame,
) -> pd.DataFrame:
    frequency = (
        loo
        .groupby(
            "selected_multiplier",
            as_index=False,
        )
        .agg(
            loo_selections=(
                "holdout_cycle",
                "size",
            ),
            mean_heldout_winner_accuracy=(
                "heldout_winner_accuracy",
                "mean",
            ),
            mean_heldout_brier_score=(
                "heldout_brier_score",
                "mean",
            ),
            mean_abs_heldout_expected_seat_error=(
                "heldout_expected_seat_error",
                lambda values: np.mean(
                    np.abs(values)
                ),
            ),
            mean_abs_heldout_deterministic_seat_error=(
                "heldout_deterministic_seat_error",
                lambda values: np.mean(
                    np.abs(values)
                ),
            ),
            mean_heldout_margin_mae=(
                "heldout_margin_mae",
                "mean",
            ),
        )
        .sort_values(
            [
                "loo_selections",
                "selected_multiplier",
            ],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )

    return frequency


def best_multiplier_for_metric(
    scored: pd.DataFrame,
    column: str,
    ascending: bool,
) -> tuple[float, float]:
    row = (
        scored
        .sort_values(
            [
                column,
                "house_environment_multiplier",
            ],
            ascending=[
                ascending,
                False,
            ],
        )
        .iloc[0]
    )

    return (
        float(
            row[
                "house_environment_multiplier"
            ]
        ),
        float(row[column]),
    )


def print_table(
    title: str,
    frame: pd.DataFrame,
) -> None:
    print()
    print(title)
    print("=" * 132)
    print(frame.to_string(index=False))


def main() -> int:
    predictions = load_predictions()

    cycle_metrics = calculate_cycle_metrics(
        predictions
    )

    pooled = aggregate_metrics(
        cycle_metrics
    )

    scored = add_composite_score(
        pooled
    )

    recommendation = build_recommendation(
        scored
    )

    loo = build_leave_one_cycle_out(
        cycle_metrics
    )

    loo_frequency = (
        build_multiplier_selection_frequency(
            loo
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_frames = {
        "house_majority_environment_cycle_metrics.csv": (
            cycle_metrics
        ),
        "house_majority_environment_pooled_scores.csv": (
            scored
        ),
        "house_majority_environment_leave_one_cycle_out.csv": (
            loo
        ),
        "house_majority_environment_loo_selection_frequency.csv": (
            loo_frequency
        ),
    }

    for filename, frame in output_frames.items():
        frame.to_csv(
            OUTPUT_DIR / filename,
            index=False,
        )

    display_columns = [
        "house_environment_multiplier",
        "mean_cycle_winner_accuracy",
        "mean_cycle_brier_score",
        "mean_cycle_absolute_expected_seat_error",
        "mean_cycle_absolute_deterministic_seat_error",
        "mean_cycle_margin_mae",
        "majority_composite_score",
        "composite_rank",
        "winner_accuracy_rank",
        "brier_rank",
        "expected_seat_error_rank",
        "deterministic_seat_error_rank",
        "margin_mae_rank",
    ]

    print()
    print(
        "House majority-focused environment bakeoff"
    )
    print("=" * 132)
    print(
        f"Prediction rows: {len(predictions):,}"
    )
    print(
        "Cycles: "
        f"{sorted(predictions['cycle'].unique())}"
    )
    print(
        "Multipliers: "
        f"{predictions['house_environment_multiplier'].nunique()}"
    )
    print(
        f"Outputs: {OUTPUT_DIR}"
    )

    print()
    print("Composite weights")
    print("-" * 132)

    for metric, weight in METRIC_WEIGHTS.items():
        print(
            f"{metric}: {weight:.0%}"
        )

    print_table(
        "A. CYCLE-LEVEL RAW METRICS",
        cycle_metrics.round(5),
    )

    print_table(
        "B. POOLED, CYCLE-BALANCED MAJORITY SCORES",
        scored[
            display_columns
        ].round(5),
    )

    print_table(
        "C. LEAVE-ONE-CYCLE-OUT RESULTS",
        loo.round(5),
    )

    print_table(
        "D. LEAVE-ONE-CYCLE-OUT SELECTION FREQUENCY",
        loo_frequency.round(5),
    )

    (
        margin_best_multiplier,
        margin_best_value,
    ) = best_multiplier_for_metric(
        scored,
        "mean_cycle_margin_mae",
        ascending=True,
    )

    (
        winner_best_multiplier,
        winner_best_value,
    ) = best_multiplier_for_metric(
        scored,
        "mean_cycle_winner_accuracy",
        ascending=False,
    )

    (
        brier_best_multiplier,
        brier_best_value,
    ) = best_multiplier_for_metric(
        scored,
        "mean_cycle_brier_score",
        ascending=True,
    )

    (
        expected_seat_best_multiplier,
        expected_seat_best_value,
    ) = best_multiplier_for_metric(
        scored,
        "mean_cycle_absolute_expected_seat_error",
        ascending=True,
    )

    (
        deterministic_seat_best_multiplier,
        deterministic_seat_best_value,
    ) = best_multiplier_for_metric(
        scored,
        "mean_cycle_absolute_deterministic_seat_error",
        ascending=True,
    )

    print()
    print("Metric-specific recommendations")
    print("=" * 132)
    print(
        "Margin-MAE optimum: "
        f"{margin_best_multiplier:.2f} "
        f"(mean cycle MAE={margin_best_value:.5f})"
    )
    print(
        "Winner-accuracy optimum: "
        f"{winner_best_multiplier:.2f} "
        f"(mean accuracy={winner_best_value:.5f})"
    )
    print(
        "Brier optimum: "
        f"{brier_best_multiplier:.2f} "
        f"(mean Brier={brier_best_value:.5f})"
    )
    print(
        "Expected-seat optimum: "
        f"{expected_seat_best_multiplier:.2f} "
        f"(mean absolute cycle seat error="
        f"{expected_seat_best_value:.5f})"
    )
    print(
        "Deterministic-seat optimum: "
        f"{deterministic_seat_best_multiplier:.2f} "
        f"(mean absolute cycle seat error="
        f"{deterministic_seat_best_value:.5f})"
    )

    print()
    print("Majority-focused recommendation")
    print("=" * 132)
    print(
        "Numerically best composite multiplier: "
        f"{recommendation.best_multiplier:.2f}"
    )
    print(
        "Best composite score: "
        f"{recommendation.best_score:.5f}"
    )
    print(
        "Near-best tolerance: "
        f"{NEAR_BEST_TOLERANCE:.2f}"
    )
    print(
        "Near-best multipliers: "
        + ", ".join(
            f"{multiplier:.2f}"
            for multiplier
            in recommendation.near_best_multipliers
        )
    )
    print(
        "Simplicity/shared-architecture recommendation: "
        f"{recommendation.recommended_multiplier:.2f}"
    )
    print(
        "Reason: "
        f"{recommendation.recommendation_reason}"
    )

    expected_cycle_set = set(
        EXPECTED_CYCLES
    )

    actual_cycle_set = set(
        predictions["cycle"].unique()
    )

    multipliers_by_cycle = (
        predictions
        .groupby("cycle")[
            "house_environment_multiplier"
        ]
        .apply(
            lambda values: tuple(
                sorted(values.unique())
            )
        )
    )

    common_multiplier_grid = (
        multipliers_by_cycle.nunique()
        == 1
    )

    weight_sum = sum(
        METRIC_WEIGHTS.values()
    )

    checks = {
        "Expected four cycles present": (
            actual_cycle_set
            == expected_cycle_set
        ),
        "Common multiplier grid across cycles": (
            common_multiplier_grid
        ),
        "No duplicate cycle/district/multiplier rows": (
            not predictions.duplicated(
                [
                    "cycle",
                    "district_id",
                    "house_environment_multiplier",
                ]
            ).any()
        ),
        "Composite weights sum to one": (
            np.isclose(
                weight_sum,
                1.0,
            )
        ),
        "Composite scores are finite": (
            np.isfinite(
                scored[
                    "majority_composite_score"
                ]
            ).all()
        ),
        "Composite scores lie in [0, 1]": (
            scored[
                "majority_composite_score"
            ].between(
                0.0,
                1.0,
            ).all()
        ),
        "LOO covers every cycle": (
            set(
                loo["holdout_cycle"]
            )
            == expected_cycle_set
        ),
        "Recommended multiplier exists in grid": (
            np.isclose(
                scored[
                    "house_environment_multiplier"
                ],
                recommendation.recommended_multiplier,
                atol=0.0051,
            ).any()
        ),
        "All probabilities lie in [0, 1]": (
            predictions[
                "dem_win_probability"
            ].between(
                0.0,
                1.0,
            ).all()
        ),
    }

    print()
    print("Validation checks")
    print("-" * 132)

    failed: list[str] = []

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
            "House majority-focused environment bakeoff: "
            "FAILED"
        )
        return 1

    print()
    print(
        "House majority-focused environment bakeoff: "
        "PASSED"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
