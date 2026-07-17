#!/usr/bin/env python3
"""
Nested leave-one-cycle-out validation for House Layer 5.

Outer loop
----------
Hold out one election cycle completely.

Inner loop
----------
Using only the remaining cycles:
    1. Evaluate every Layer 5 configuration with leave-one-cycle-out
       district residual estimation.
    2. Select a configuration without examining the outer cycle.

Outer evaluation
----------------
Fit district residual multipliers on all inner-training cycles and apply
the selected configuration once to the untouched outer cycle.

Layer 5 changes probability dispersion only. It does not change:
    - point margins
    - margin MAE
    - margin RMSE
    - margin-based winner calls
"""

from __future__ import annotations

from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "canonical"
    / "house_multicycle_backtest_results.csv"
)

OUTPUT_DIR = (
    ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "layer5_nested_validation"
)

OUTER_RESULTS_PATH = (
    OUTPUT_DIR
    / "house_layer5_nested_outer_results.csv"
)

INNER_SELECTIONS_PATH = (
    OUTPUT_DIR
    / "house_layer5_nested_inner_selections.csv"
)

INNER_GRID_PATH = (
    OUTPUT_DIR
    / "house_layer5_nested_inner_grid.csv"
)

DETAIL_PATH = (
    OUTPUT_DIR
    / "house_layer5_nested_outer_detail.csv"
)

POOLED_PATH = (
    OUTPUT_DIR
    / "house_layer5_nested_pooled_results.csv"
)

DECISION_PATH = (
    OUTPUT_DIR
    / "house_layer5_nested_decision.csv"
)


BASELINE_TOTAL_SD = 6.5

DISTRICT_COMPONENT_SDS = [
    3.5,
    4.0,
    4.5,
    5.0,
    5.5,
    6.0,
]

SHRINKAGE_STRENGTHS = [
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
]

MULTIPLIER_BOUNDS = [
    (0.70, 1.30),
    (0.75, 1.25),
    (0.80, 1.20),
]

EPSILON = 1e-12
CALIBRATION_EDGES = np.linspace(0.0, 1.0, 11)


def find_column(
    frame: pd.DataFrame,
    candidates: list[str],
    label: str,
) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    raise ValueError(
        f"Could not identify {label}. Tried: {candidates}"
    )


def parse_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(
            {
                "true",
                "1",
                "yes",
                "y",
                "t",
            }
        )
    )


def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    cycle_column = find_column(
        raw,
        [
            "cycle",
            "election_cycle",
            "year",
            "held_out_cycle",
        ],
        "election-cycle column",
    )

    district_column = find_column(
        raw,
        [
            "district_id",
            "geography",
            "district",
        ],
        "district identifier",
    )

    model_margin_column = find_column(
        raw,
        [
            "model_margin_dem",
            "predicted_margin_dem",
            "forecast_margin_dem",
        ],
        "model Democratic margin",
    )

    actual_margin_column = find_column(
        raw,
        [
            "actual_dem_margin",
            "actual_margin_dem",
            "dem_margin",
        ],
        "actual Democratic margin",
    )

    actual_win_candidates = [
        "actual_dem_win",
        "dem_win",
        "dem_won",
    ]

    actual_win_column = next(
        (
            column
            for column in actual_win_candidates
            if column in raw.columns
        ),
        None,
    )

    include_column = next(
        (
            column
            for column in [
                "include_in_scoring",
                "scored",
                "is_scorable",
            ]
            if column in raw.columns
        ),
        None,
    )

    data = pd.DataFrame(
        {
            "cycle": pd.to_numeric(
                raw[cycle_column],
                errors="coerce",
            ),
            "district_id": (
                raw[district_column]
                .astype(str)
                .str.strip()
                .str.upper()
            ),
            "model_margin_dem": pd.to_numeric(
                raw[model_margin_column],
                errors="coerce",
            ),
            "actual_dem_margin": pd.to_numeric(
                raw[actual_margin_column],
                errors="coerce",
            ),
        }
    )

    if actual_win_column is not None:
        actual_win_numeric = pd.to_numeric(
            raw[actual_win_column],
            errors="coerce",
        )

        parsed_actual_win = np.where(
            actual_win_numeric.notna(),
            actual_win_numeric,
            parse_bool(raw[actual_win_column]).astype(float),
        )

        data["actual_dem_win"] = parsed_actual_win
    else:
        data["actual_dem_win"] = (
            data["actual_dem_margin"] > 0
        ).astype(float)

    if include_column is not None:
        data["include_in_scoring"] = parse_bool(
            raw[include_column]
        )
    else:
        data["include_in_scoring"] = True

    data = data.loc[
        data["include_in_scoring"]
    ].copy()

    data = data.dropna(
        subset=[
            "cycle",
            "district_id",
            "model_margin_dem",
            "actual_dem_margin",
            "actual_dem_win",
        ]
    ).copy()

    data["cycle"] = data["cycle"].astype(int)

    data["actual_dem_win"] = (
        data["actual_dem_win"]
        .astype(float)
        .clip(0.0, 1.0)
    )

    data["margin_error"] = (
        data["model_margin_dem"]
        - data["actual_dem_margin"]
    )

    data["baseline_total_sd"] = BASELINE_TOTAL_SD

    data["baseline_dem_win_probability"] = norm.cdf(
        data["model_margin_dem"]
        / BASELINE_TOTAL_SD
    )

    return data.reset_index(drop=True)


def calculate_brier(
    actual: np.ndarray,
    probability: np.ndarray,
) -> float:
    return float(
        np.mean(
            np.square(probability - actual)
        )
    )


def calculate_log_loss(
    actual: np.ndarray,
    probability: np.ndarray,
) -> float:
    probability = np.clip(
        probability,
        EPSILON,
        1.0 - EPSILON,
    )

    return float(
        -np.mean(
            actual * np.log(probability)
            + (1.0 - actual)
            * np.log(1.0 - probability)
        )
    )


def calculate_ece(
    actual: np.ndarray,
    probability: np.ndarray,
) -> float:
    bin_ids = np.digitize(
        probability,
        CALIBRATION_EDGES[1:-1],
        right=False,
    )

    total = len(actual)
    ece = 0.0

    for bin_id in range(10):
        mask = bin_ids == bin_id
        count = int(mask.sum())

        if count == 0:
            continue

        mean_probability = float(
            probability[mask].mean()
        )

        observed_rate = float(
            actual[mask].mean()
        )

        ece += (
            count / total
        ) * abs(
            mean_probability - observed_rate
        )

    return float(ece)


def calculate_coverage(
    actual_margin: np.ndarray,
    predicted_margin: np.ndarray,
    uncertainty_sd: np.ndarray,
    z_value: float,
) -> float:
    lower = (
        predicted_margin
        - z_value * uncertainty_sd
    )

    upper = (
        predicted_margin
        + z_value * uncertainty_sd
    )

    return float(
        np.mean(
            (actual_margin >= lower)
            & (actual_margin <= upper)
        )
    )


def build_district_multipliers(
    training: pd.DataFrame,
    shrinkage_strength: float,
    multiplier_floor: float,
    multiplier_ceiling: float,
) -> tuple[dict[str, float], float]:
    errors = training[
        "margin_error"
    ].to_numpy(dtype=float)

    pooled_variance = float(
        np.mean(
            np.square(errors)
        )
    )

    pooled_rmse = sqrt(
        max(pooled_variance, EPSILON)
    )

    grouped = (
        training.groupby(
            "district_id",
            as_index=False,
        )
        .agg(
            residual_observations=(
                "margin_error",
                "size",
            ),
            district_mse=(
                "margin_error",
                lambda values: float(
                    np.mean(
                        np.square(
                            values.to_numpy(
                                dtype=float
                            )
                        )
                    )
                ),
            ),
        )
    )

    grouped["reliability"] = (
        grouped["residual_observations"]
        / (
            grouped["residual_observations"]
            + shrinkage_strength
        )
    )

    grouped["shrunk_variance"] = (
        grouped["reliability"]
        * grouped["district_mse"]
        + (
            1.0 - grouped["reliability"]
        )
        * pooled_variance
    )

    grouped["multiplier"] = (
        np.sqrt(
            grouped["shrunk_variance"]
        )
        / pooled_rmse
    ).clip(
        lower=multiplier_floor,
        upper=multiplier_ceiling,
    )

    multipliers = dict(
        zip(
            grouped["district_id"],
            grouped["multiplier"],
        )
    )

    return multipliers, pooled_rmse


def apply_configuration(
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    district_component_sd: float,
    shrinkage_strength: float,
    multiplier_floor: float,
    multiplier_ceiling: float,
) -> pd.DataFrame:
    multipliers, pooled_rmse = (
        build_district_multipliers(
            training=training,
            shrinkage_strength=(
                shrinkage_strength
            ),
            multiplier_floor=(
                multiplier_floor
            ),
            multiplier_ceiling=(
                multiplier_ceiling
            ),
        )
    )

    result = evaluation.copy()

    result[
        "district_uncertainty_multiplier"
    ] = (
        result["district_id"]
        .map(multipliers)
        .fillna(1.0)
        .astype(float)
    )

    nondistrict_variance = max(
        BASELINE_TOTAL_SD ** 2
        - district_component_sd ** 2,
        0.0,
    )

    result[
        "adjusted_district_component_sd"
    ] = (
        district_component_sd
        * result[
            "district_uncertainty_multiplier"
        ]
    )

    result["adjusted_total_error_sd"] = np.sqrt(
        nondistrict_variance
        + np.square(
            result[
                "adjusted_district_component_sd"
            ]
        )
    )

    result[
        "adjusted_dem_win_probability"
    ] = norm.cdf(
        result["model_margin_dem"]
        / result[
            "adjusted_total_error_sd"
        ]
    )

    result["training_pooled_rmse"] = (
        pooled_rmse
    )

    result["district_component_sd"] = (
        district_component_sd
    )

    result["shrinkage_strength"] = (
        shrinkage_strength
    )

    result["multiplier_floor"] = (
        multiplier_floor
    )

    result["multiplier_ceiling"] = (
        multiplier_ceiling
    )

    return result


def evaluate_probabilities(
    frame: pd.DataFrame,
) -> dict[str, float]:
    actual = frame[
        "actual_dem_win"
    ].to_numpy(dtype=float)

    baseline_probability = frame[
        "baseline_dem_win_probability"
    ].to_numpy(dtype=float)

    adjusted_probability = frame[
        "adjusted_dem_win_probability"
    ].to_numpy(dtype=float)

    baseline_brier = calculate_brier(
        actual,
        baseline_probability,
    )

    adjusted_brier = calculate_brier(
        actual,
        adjusted_probability,
    )

    baseline_log_loss = calculate_log_loss(
        actual,
        baseline_probability,
    )

    adjusted_log_loss = calculate_log_loss(
        actual,
        adjusted_probability,
    )

    baseline_ece = calculate_ece(
        actual,
        baseline_probability,
    )

    adjusted_ece = calculate_ece(
        actual,
        adjusted_probability,
    )

    actual_dem_seats = float(
        actual.sum()
    )

    baseline_expected_dem_seats = float(
        baseline_probability.sum()
    )

    adjusted_expected_dem_seats = float(
        adjusted_probability.sum()
    )

    return {
        "scored_races": int(len(frame)),
        "actual_dem_seats": (
            actual_dem_seats
        ),
        "baseline_expected_dem_seats": (
            baseline_expected_dem_seats
        ),
        "layer5_expected_dem_seats": (
            adjusted_expected_dem_seats
        ),
        "baseline_abs_seat_error": abs(
            baseline_expected_dem_seats
            - actual_dem_seats
        ),
        "layer5_abs_seat_error": abs(
            adjusted_expected_dem_seats
            - actual_dem_seats
        ),
        "abs_seat_error_change": (
            abs(
                adjusted_expected_dem_seats
                - actual_dem_seats
            )
            - abs(
                baseline_expected_dem_seats
                - actual_dem_seats
            )
        ),
        "baseline_brier": baseline_brier,
        "layer5_brier": adjusted_brier,
        "brier_change": (
            adjusted_brier
            - baseline_brier
        ),
        "baseline_log_loss": (
            baseline_log_loss
        ),
        "layer5_log_loss": (
            adjusted_log_loss
        ),
        "log_loss_change": (
            adjusted_log_loss
            - baseline_log_loss
        ),
        "baseline_ece": baseline_ece,
        "layer5_ece": adjusted_ece,
        "ece_change": (
            adjusted_ece
            - baseline_ece
        ),
    }


def inner_validate_configuration(
    outer_training: pd.DataFrame,
    district_component_sd: float,
    shrinkage_strength: float,
    multiplier_floor: float,
    multiplier_ceiling: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    inner_detail_parts = []

    inner_cycles = sorted(
        outer_training[
            "cycle"
        ].unique()
    )

    for inner_holdout_cycle in inner_cycles:
        inner_training = outer_training.loc[
            outer_training["cycle"]
            != inner_holdout_cycle
        ].copy()

        inner_holdout = outer_training.loc[
            outer_training["cycle"]
            == inner_holdout_cycle
        ].copy()

        scored = apply_configuration(
            training=inner_training,
            evaluation=inner_holdout,
            district_component_sd=(
                district_component_sd
            ),
            shrinkage_strength=(
                shrinkage_strength
            ),
            multiplier_floor=(
                multiplier_floor
            ),
            multiplier_ceiling=(
                multiplier_ceiling
            ),
        )

        scored["inner_holdout_cycle"] = (
            inner_holdout_cycle
        )

        inner_detail_parts.append(scored)

    inner_detail = pd.concat(
        inner_detail_parts,
        ignore_index=True,
    )

    metrics = evaluate_probabilities(
        inner_detail
    )

    metrics.update(
        {
            "district_component_sd": (
                district_component_sd
            ),
            "shrinkage_strength": (
                shrinkage_strength
            ),
            "multiplier_floor": (
                multiplier_floor
            ),
            "multiplier_ceiling": (
                multiplier_ceiling
            ),
            "mean_multiplier": float(
                inner_detail[
                    "district_uncertainty_multiplier"
                ].mean()
            ),
            "minimum_multiplier": float(
                inner_detail[
                    "district_uncertainty_multiplier"
                ].min()
            ),
            "maximum_multiplier": float(
                inner_detail[
                    "district_uncertainty_multiplier"
                ].max()
            ),
        }
    )

    metrics["passes_primary_metrics"] = bool(
        metrics["brier_change"] <= 0
        and metrics["log_loss_change"] < 0
    )

    return inner_detail, metrics


def choose_configuration(
    grid: pd.DataFrame,
) -> pd.Series:
    passing = grid.loc[
        grid["passes_primary_metrics"]
    ].copy()

    if not passing.empty:
        # Primary selection:
        # lowest log loss among configurations that
        # do not worsen Brier score.
        return (
            passing.sort_values(
                [
                    "layer5_log_loss",
                    "layer5_brier",
                    "ece_change",
                    "district_component_sd",
                    "shrinkage_strength",
                    "multiplier_floor",
                    "multiplier_ceiling",
                ],
                ascending=[
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    True,
                ],
            )
            .iloc[0]
        )

    # Conservative fallback when no configuration
    # improves both primary metrics:
    #
    # Choose the lowest Brier score, then log loss.
    # The outer result will reveal whether this fallback
    # generalizes.
    return (
        grid.sort_values(
            [
                "layer5_brier",
                "layer5_log_loss",
                "ece_change",
                "district_component_sd",
                "shrinkage_strength",
            ]
        )
        .iloc[0]
    )


def add_outer_coverage(
    metrics: dict[str, float],
    frame: pd.DataFrame,
) -> None:
    actual_margin = frame[
        "actual_dem_margin"
    ].to_numpy(dtype=float)

    model_margin = frame[
        "model_margin_dem"
    ].to_numpy(dtype=float)

    baseline_sd = np.full(
        len(frame),
        BASELINE_TOTAL_SD,
        dtype=float,
    )

    adjusted_sd = frame[
        "adjusted_total_error_sd"
    ].to_numpy(dtype=float)

    metrics["baseline_50_coverage"] = (
        calculate_coverage(
            actual_margin,
            model_margin,
            baseline_sd,
            0.67448975,
        )
    )

    metrics["layer5_50_coverage"] = (
        calculate_coverage(
            actual_margin,
            model_margin,
            adjusted_sd,
            0.67448975,
        )
    )

    metrics["baseline_80_coverage"] = (
        calculate_coverage(
            actual_margin,
            model_margin,
            baseline_sd,
            1.28155157,
        )
    )

    metrics["layer5_80_coverage"] = (
        calculate_coverage(
            actual_margin,
            model_margin,
            adjusted_sd,
            1.28155157,
        )
    )

    metrics["baseline_95_coverage"] = (
        calculate_coverage(
            actual_margin,
            model_margin,
            baseline_sd,
            1.95996398,
        )
    )

    metrics["layer5_95_coverage"] = (
        calculate_coverage(
            actual_margin,
            model_margin,
            adjusted_sd,
            1.95996398,
        )
    )


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing canonical backtest file: "
            f"{INPUT_PATH}"
        )

    raw = pd.read_csv(INPUT_PATH)
    data = prepare_data(raw)

    cycles = sorted(
        data["cycle"].unique()
    )

    if len(cycles) < 4:
        raise ValueError(
            "Nested validation requires at least "
            "four election cycles."
        )

    print(
        "House Layer 5 nested validation"
    )
    print(
        "--------------------------------"
    )
    print(
        f"Scored races: {len(data)}"
    )
    print(
        f"Cycles: {cycles}"
    )
    print(
        "Configurations per outer fold: "
        f"{len(DISTRICT_COMPONENT_SDS) * len(SHRINKAGE_STRENGTHS) * len(MULTIPLIER_BOUNDS)}"
    )

    outer_rows = []
    inner_selection_rows = []
    inner_grid_rows = []
    outer_detail_parts = []

    for outer_holdout_cycle in cycles:
        print(
            f"\nOuter holdout: "
            f"{outer_holdout_cycle}"
        )

        outer_training = data.loc[
            data["cycle"]
            != outer_holdout_cycle
        ].copy()

        outer_holdout = data.loc[
            data["cycle"]
            == outer_holdout_cycle
        ].copy()

        grid_rows = []

        for district_component_sd in (
            DISTRICT_COMPONENT_SDS
        ):
            for shrinkage_strength in (
                SHRINKAGE_STRENGTHS
            ):
                for (
                    multiplier_floor,
                    multiplier_ceiling,
                ) in MULTIPLIER_BOUNDS:
                    _, metrics = (
                        inner_validate_configuration(
                            outer_training=(
                                outer_training
                            ),
                            district_component_sd=(
                                district_component_sd
                            ),
                            shrinkage_strength=(
                                shrinkage_strength
                            ),
                            multiplier_floor=(
                                multiplier_floor
                            ),
                            multiplier_ceiling=(
                                multiplier_ceiling
                            ),
                        )
                    )

                    metrics[
                        "outer_holdout_cycle"
                    ] = outer_holdout_cycle

                    grid_rows.append(metrics)

        grid = pd.DataFrame(grid_rows)

        selected = choose_configuration(
            grid
        )

        grid["selected_for_outer_fold"] = False

        selected_mask = (
            (
                grid["district_component_sd"]
                == selected[
                    "district_component_sd"
                ]
            )
            & (
                grid["shrinkage_strength"]
                == selected[
                    "shrinkage_strength"
                ]
            )
            & (
                grid["multiplier_floor"]
                == selected[
                    "multiplier_floor"
                ]
            )
            & (
                grid["multiplier_ceiling"]
                == selected[
                    "multiplier_ceiling"
                ]
            )
        )

        grid.loc[
            selected_mask,
            "selected_for_outer_fold",
        ] = True

        inner_grid_rows.append(grid)

        selected_configuration = {
            "outer_holdout_cycle": (
                outer_holdout_cycle
            ),
            "selected_district_component_sd": float(
                selected[
                    "district_component_sd"
                ]
            ),
            "selected_shrinkage_strength": float(
                selected[
                    "shrinkage_strength"
                ]
            ),
            "selected_multiplier_floor": float(
                selected[
                    "multiplier_floor"
                ]
            ),
            "selected_multiplier_ceiling": float(
                selected[
                    "multiplier_ceiling"
                ]
            ),
            "inner_passes_primary_metrics": bool(
                selected[
                    "passes_primary_metrics"
                ]
            ),
            "inner_brier_change": float(
                selected["brier_change"]
            ),
            "inner_log_loss_change": float(
                selected[
                    "log_loss_change"
                ]
            ),
            "inner_ece_change": float(
                selected["ece_change"]
            ),
        }

        inner_selection_rows.append(
            selected_configuration
        )

        outer_scored = apply_configuration(
            training=outer_training,
            evaluation=outer_holdout,
            district_component_sd=float(
                selected[
                    "district_component_sd"
                ]
            ),
            shrinkage_strength=float(
                selected[
                    "shrinkage_strength"
                ]
            ),
            multiplier_floor=float(
                selected[
                    "multiplier_floor"
                ]
            ),
            multiplier_ceiling=float(
                selected[
                    "multiplier_ceiling"
                ]
            ),
        )

        outer_scored[
            "outer_holdout_cycle"
        ] = outer_holdout_cycle

        outer_detail_parts.append(
            outer_scored
        )

        outer_metrics = evaluate_probabilities(
            outer_scored
        )

        add_outer_coverage(
            outer_metrics,
            outer_scored,
        )

        outer_metrics.update(
            selected_configuration
        )

        outer_metrics[
            "mean_outer_multiplier"
        ] = float(
            outer_scored[
                "district_uncertainty_multiplier"
            ].mean()
        )

        outer_metrics[
            "minimum_outer_multiplier"
        ] = float(
            outer_scored[
                "district_uncertainty_multiplier"
            ].min()
        )

        outer_metrics[
            "maximum_outer_multiplier"
        ] = float(
            outer_scored[
                "district_uncertainty_multiplier"
            ].max()
        )

        outer_rows.append(
            outer_metrics
        )

        print(
            "  Selected: "
            f"district={selected['district_component_sd']:.1f}, "
            f"shrinkage={selected['shrinkage_strength']:.1f}, "
            f"bounds={selected['multiplier_floor']:.2f}-"
            f"{selected['multiplier_ceiling']:.2f}"
        )

        print(
            "  Outer changes: "
            f"Brier={outer_metrics['brier_change']:+.8f}, "
            f"log loss={outer_metrics['log_loss_change']:+.6f}, "
            f"ECE={outer_metrics['ece_change']:+.6f}, "
            f"seat error={outer_metrics['abs_seat_error_change']:+.3f}"
        )

    outer_results = pd.DataFrame(
        outer_rows
    ).sort_values(
        "outer_holdout_cycle"
    )

    inner_selections = pd.DataFrame(
        inner_selection_rows
    ).sort_values(
        "outer_holdout_cycle"
    )

    inner_grid = pd.concat(
        inner_grid_rows,
        ignore_index=True,
    )

    outer_detail = pd.concat(
        outer_detail_parts,
        ignore_index=True,
    )

    pooled_metrics = evaluate_probabilities(
        outer_detail
    )

    add_outer_coverage(
        pooled_metrics,
        outer_detail,
    )

    pooled_metrics[
        "margin_mae"
    ] = float(
        np.mean(
            np.abs(
                outer_detail[
                    "margin_error"
                ]
            )
        )
    )

    pooled_metrics[
        "margin_rmse"
    ] = float(
        np.sqrt(
            np.mean(
                np.square(
                    outer_detail[
                        "margin_error"
                    ]
                )
            )
        )
    )

    pooled_metrics[
        "margin_bias_dem"
    ] = float(
        outer_detail[
            "margin_error"
        ].mean()
    )

    pooled_metrics[
        "winner_accuracy"
    ] = float(
        np.mean(
            (
                outer_detail[
                    "model_margin_dem"
                ] > 0
            ).astype(float)
            == outer_detail[
                "actual_dem_win"
            ]
        )
    )

    pooled = pd.DataFrame(
        [pooled_metrics]
    )

    cycles_brier_improved = int(
        (
            outer_results[
                "brier_change"
            ] < 0
        ).sum()
    )

    cycles_log_loss_improved = int(
        (
            outer_results[
                "log_loss_change"
            ] < 0
        ).sum()
    )

    cycles_ece_improved = int(
        (
            outer_results[
                "ece_change"
            ] < 0
        ).sum()
    )

    cycles_seat_error_improved = int(
        (
            outer_results[
                "abs_seat_error_change"
            ] < 0
        ).sum()
    )

    configuration_count = int(
        inner_selections[
            [
                "selected_district_component_sd",
                "selected_shrinkage_strength",
                "selected_multiplier_floor",
                "selected_multiplier_ceiling",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    # Strong-retention rule:
    # - Pooled Brier improves.
    # - Pooled log loss improves.
    # - Both improve in at least half the cycles.
    #
    # ECE, coverage, and seat error are reported as
    # secondary diagnostics rather than hard gates.
    strong_pass = bool(
        pooled_metrics[
            "brier_change"
        ] < 0
        and pooled_metrics[
            "log_loss_change"
        ] < 0
        and cycles_brier_improved >= 2
        and cycles_log_loss_improved >= 2
    )

    # Weak-retention rule:
    # - Neither pooled primary metric materially worsens.
    # - At least one pooled primary metric improves.
    #
    # Tolerance prevents floating-point noise from
    # driving the decision.
    tolerance = 1e-7

    weak_pass = bool(
        pooled_metrics[
            "brier_change"
        ] <= tolerance
        and pooled_metrics[
            "log_loss_change"
        ] <= tolerance
        and (
            pooled_metrics[
                "brier_change"
            ] < -tolerance
            or pooled_metrics[
                "log_loss_change"
            ] < -tolerance
        )
    )

    if strong_pass:
        decision_text = (
            "KEEP LAYER 5 — NESTED VALIDATION PASSED"
        )
        status = "validated"
    elif weak_pass:
        decision_text = (
            "KEEP PROVISIONALLY — WEAK NESTED EVIDENCE"
        )
        status = "provisional"
    else:
        decision_text = (
            "REVERT OR RETUNE LAYER 5"
        )
        status = "failed"

    decision = pd.DataFrame(
        [
            {
                "nested_validation_status": (
                    status
                ),
                "promotion_decision": (
                    decision_text
                ),
                "outer_cycles": int(
                    len(cycles)
                ),
                "scored_races": int(
                    len(outer_detail)
                ),
                "pooled_brier_change": (
                    pooled_metrics[
                        "brier_change"
                    ]
                ),
                "pooled_log_loss_change": (
                    pooled_metrics[
                        "log_loss_change"
                    ]
                ),
                "pooled_ece_change": (
                    pooled_metrics[
                        "ece_change"
                    ]
                ),
                "pooled_abs_seat_error_change": (
                    pooled_metrics[
                        "abs_seat_error_change"
                    ]
                ),
                "cycles_brier_improved": (
                    cycles_brier_improved
                ),
                "cycles_log_loss_improved": (
                    cycles_log_loss_improved
                ),
                "cycles_ece_improved": (
                    cycles_ece_improved
                ),
                "cycles_seat_error_improved": (
                    cycles_seat_error_improved
                ),
                "unique_selected_configurations": (
                    configuration_count
                ),
                "strong_retention_rule_passes": (
                    strong_pass
                ),
                "weak_retention_rule_passes": (
                    weak_pass
                ),
                "notes": (
                    "Each outer cycle was excluded "
                    "from both district multiplier "
                    "estimation and hyperparameter "
                    "selection."
                ),
            }
        ]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    outer_results.to_csv(
        OUTER_RESULTS_PATH,
        index=False,
    )

    inner_selections.to_csv(
        INNER_SELECTIONS_PATH,
        index=False,
    )

    inner_grid.to_csv(
        INNER_GRID_PATH,
        index=False,
    )

    outer_detail.to_csv(
        DETAIL_PATH,
        index=False,
    )

    pooled.to_csv(
        POOLED_PATH,
        index=False,
    )

    decision.to_csv(
        DECISION_PATH,
        index=False,
    )

    print(
        "\nNested outer-fold results:"
    )

    outer_display_columns = [
        "outer_holdout_cycle",
        "selected_district_component_sd",
        "selected_shrinkage_strength",
        "selected_multiplier_floor",
        "selected_multiplier_ceiling",
        "brier_change",
        "log_loss_change",
        "ece_change",
        "abs_seat_error_change",
    ]

    print(
        outer_results[
            outer_display_columns
        ].to_string(index=False)
    )

    print(
        "\nNested pooled results:"
    )

    pooled_display_columns = [
        "scored_races",
        "margin_mae",
        "margin_rmse",
        "margin_bias_dem",
        "winner_accuracy",
        "baseline_brier",
        "layer5_brier",
        "brier_change",
        "baseline_log_loss",
        "layer5_log_loss",
        "log_loss_change",
        "baseline_ece",
        "layer5_ece",
        "ece_change",
        "baseline_abs_seat_error",
        "layer5_abs_seat_error",
        "abs_seat_error_change",
    ]

    print(
        pooled[
            pooled_display_columns
        ].to_string(index=False)
    )

    print(
        "\nNested pooled coverage:"
    )

    coverage_columns = [
        "baseline_50_coverage",
        "layer5_50_coverage",
        "baseline_80_coverage",
        "layer5_80_coverage",
        "baseline_95_coverage",
        "layer5_95_coverage",
    ]

    print(
        pooled[
            coverage_columns
        ].to_string(index=False)
    )

    print(
        "\nFinal decision:"
    )

    print(
        decision.to_string(index=False)
    )

    print(
        f"\nWrote: {OUTER_RESULTS_PATH}"
    )
    print(
        f"Wrote: {INNER_SELECTIONS_PATH}"
    )
    print(
        f"Wrote: {INNER_GRID_PATH}"
    )
    print(
        f"Wrote: {DETAIL_PATH}"
    )
    print(
        f"Wrote: {POOLED_PATH}"
    )
    print(
        f"Wrote: {DECISION_PATH}"
    )


if __name__ == "__main__":
    main()
