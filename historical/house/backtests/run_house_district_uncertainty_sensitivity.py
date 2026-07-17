#!/usr/bin/env python3
"""
Leakage-safe Layer 5 sensitivity test for district-specific residual uncertainty.

For each held-out election cycle:
1. Estimate district residual variance using only the other cycles.
2. Shrink each district estimate toward the pooled residual variance.
3. Convert the shrunk estimate into a multiplier centered at 1.0.
4. Apply that multiplier only to the district-specific component of the
   baseline 6.5-point total uncertainty.
5. Evaluate held-out Brier score, log loss, calibration, and coverage.

This script does not alter point-margin predictions.
"""

from __future__ import annotations

from itertools import product
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
    / "district_uncertainty_sensitivity"
)

SUMMARY_PATH = OUTPUT_DIR / "house_district_uncertainty_sensitivity.csv"
DETAIL_PATH = OUTPUT_DIR / "house_district_uncertainty_best_detail.csv"
SELECTION_PATH = OUTPUT_DIR / "house_district_uncertainty_selection.csv"

BASELINE_TOTAL_SD = 6.5

# We do not yet know the historical decomposition of the 6.5-point total SD.
# Sweep plausible district-specific components and preserve the remainder as
# shared/correlated variance.
DISTRICT_COMPONENT_GRID = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

# Larger values imply stronger shrinkage toward the pooled residual variance.
SHRINKAGE_GRID = [1.0, 2.0, 4.0, 8.0, 16.0]

# Conservative caps prevent tiny samples from creating extreme uncertainty.
MULTIPLIER_BOUNDS_GRID = [
    (0.70, 1.30),
    (0.75, 1.25),
    (0.80, 1.20),
]

EPSILON = 1e-9


def parse_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def binary_log_loss(
    actual: np.ndarray,
    probability: np.ndarray,
) -> float:
    probability = np.clip(probability, EPSILON, 1.0 - EPSILON)
    return float(
        -np.mean(
            actual * np.log(probability)
            + (1.0 - actual) * np.log(1.0 - probability)
        )
    )


def calibration_error(
    actual: np.ndarray,
    probability: np.ndarray,
    bins: int = 10,
) -> float:
    """
    Expected calibration error using equal-width probability bins.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.digitize(probability, edges[1:-1], right=False)

    total = len(actual)
    error = 0.0

    for bin_id in range(bins):
        mask = bin_ids == bin_id
        count = int(mask.sum())

        if count == 0:
            continue

        observed = float(actual[mask].mean())
        predicted = float(probability[mask].mean())

        error += (count / total) * abs(observed - predicted)

    return float(error)


def interval_coverage(
    actual_margin: np.ndarray,
    predicted_margin: np.ndarray,
    total_sd: np.ndarray,
    z_value: float,
) -> float:
    lower = predicted_margin - z_value * total_sd
    upper = predicted_margin + z_value * total_sd

    return float(
        np.mean(
            (actual_margin >= lower)
            & (actual_margin <= upper)
        )
    )


def load_scored_results() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing canonical multicycle results: {INPUT_PATH}"
        )

    frame = pd.read_csv(INPUT_PATH)

    required = [
        "cycle",
        "district_id",
        "model_margin_dem",
        "actual_dem_margin",
        "margin_error",
        "include_in_scoring",
    ]

    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    frame = frame.loc[parse_bool(frame["include_in_scoring"])].copy()

    for column in [
        "cycle",
        "model_margin_dem",
        "actual_dem_margin",
        "margin_error",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["district_id"] = (
        frame["district_id"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    frame = frame.dropna(
        subset=[
            "cycle",
            "district_id",
            "model_margin_dem",
            "actual_dem_margin",
            "margin_error",
        ]
    ).copy()

    frame["cycle"] = frame["cycle"].astype(int)
    frame["actual_dem_win"] = (
        frame["actual_dem_margin"] > 0
    ).astype(float)

    return frame


def estimate_training_multipliers(
    training: pd.DataFrame,
    shrinkage_strength: float,
    multiplier_floor: float,
    multiplier_ceiling: float,
) -> tuple[pd.DataFrame, float]:
    pooled_variance = float(
        np.mean(np.square(training["margin_error"].to_numpy()))
    )
    pooled_rmse = float(np.sqrt(pooled_variance))

    grouped = (
        training.groupby("district_id", as_index=False)
        .agg(
            residual_observations=("margin_error", "size"),
            raw_residual_mse=(
                "margin_error",
                lambda values: float(
                    np.mean(np.square(values.to_numpy()))
                ),
            ),
        )
    )

    grouped["raw_residual_rmse"] = np.sqrt(
        grouped["raw_residual_mse"]
    )

    grouped["reliability"] = (
        grouped["residual_observations"]
        / (
            grouped["residual_observations"]
            + float(shrinkage_strength)
        )
    )

    # Shrink variance, not standard deviation.
    grouped["shrunk_residual_variance"] = (
        grouped["reliability"] * grouped["raw_residual_mse"]
        + (1.0 - grouped["reliability"]) * pooled_variance
    )

    grouped["shrunk_residual_rmse"] = np.sqrt(
        grouped["shrunk_residual_variance"]
    )

    grouped["raw_uncertainty_multiplier"] = (
        grouped["shrunk_residual_rmse"]
        / max(pooled_rmse, EPSILON)
    )

    grouped["district_uncertainty_multiplier"] = (
        grouped["raw_uncertainty_multiplier"]
        .clip(
            lower=multiplier_floor,
            upper=multiplier_ceiling,
        )
    )

    return grouped, pooled_rmse


def evaluate_configuration(
    scored: pd.DataFrame,
    district_component_sd: float,
    shrinkage_strength: float,
    multiplier_floor: float,
    multiplier_ceiling: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    if district_component_sd >= BASELINE_TOTAL_SD:
        raise ValueError(
            "District component must be below baseline total SD."
        )

    shared_variance = (
        BASELINE_TOTAL_SD ** 2
        - district_component_sd ** 2
    )

    cycle_details: list[pd.DataFrame] = []

    for held_out_cycle in sorted(scored["cycle"].unique()):
        training = scored.loc[
            scored["cycle"] != held_out_cycle
        ].copy()

        validation = scored.loc[
            scored["cycle"] == held_out_cycle
        ].copy()

        multipliers, pooled_rmse = estimate_training_multipliers(
            training=training,
            shrinkage_strength=shrinkage_strength,
            multiplier_floor=multiplier_floor,
            multiplier_ceiling=multiplier_ceiling,
        )

        validation = validation.merge(
            multipliers[
                [
                    "district_id",
                    "residual_observations",
                    "raw_residual_rmse",
                    "reliability",
                    "shrunk_residual_rmse",
                    "raw_uncertainty_multiplier",
                    "district_uncertainty_multiplier",
                ]
            ],
            on="district_id",
            how="left",
        )

        validation["residual_observations"] = (
            validation["residual_observations"]
            .fillna(0)
            .astype(int)
        )

        validation["district_uncertainty_multiplier"] = (
            validation["district_uncertainty_multiplier"]
            .fillna(1.0)
        )

        validation["raw_uncertainty_multiplier"] = (
            validation["raw_uncertainty_multiplier"]
            .fillna(1.0)
        )

        validation["training_pooled_residual_rmse"] = pooled_rmse

        validation["baseline_total_sd"] = BASELINE_TOTAL_SD
        validation["baseline_district_component_sd"] = (
            district_component_sd
        )
        validation["shared_variance"] = shared_variance

        validation["adjusted_district_component_sd"] = (
            district_component_sd
            * validation["district_uncertainty_multiplier"]
        )

        validation["adjusted_total_error_sd"] = np.sqrt(
            shared_variance
            + np.square(
                validation["adjusted_district_component_sd"]
            )
        )

        validation["baseline_dem_win_probability"] = norm.cdf(
            validation["model_margin_dem"] / BASELINE_TOTAL_SD
        )

        validation["adjusted_dem_win_probability"] = norm.cdf(
            validation["model_margin_dem"]
            / validation["adjusted_total_error_sd"]
        )

        validation["held_out_cycle"] = held_out_cycle
        validation["shrinkage_strength"] = shrinkage_strength
        validation["multiplier_floor"] = multiplier_floor
        validation["multiplier_ceiling"] = multiplier_ceiling

        cycle_details.append(validation)

    detail = pd.concat(cycle_details, ignore_index=True)

    actual = detail["actual_dem_win"].to_numpy(dtype=float)
    baseline_probability = detail[
        "baseline_dem_win_probability"
    ].to_numpy(dtype=float)
    adjusted_probability = detail[
        "adjusted_dem_win_probability"
    ].to_numpy(dtype=float)

    actual_margin = detail[
        "actual_dem_margin"
    ].to_numpy(dtype=float)
    predicted_margin = detail[
        "model_margin_dem"
    ].to_numpy(dtype=float)
    adjusted_sd = detail[
        "adjusted_total_error_sd"
    ].to_numpy(dtype=float)

    baseline_brier = float(
        np.mean(np.square(baseline_probability - actual))
    )
    adjusted_brier = float(
        np.mean(np.square(adjusted_probability - actual))
    )

    baseline_log_loss = binary_log_loss(
        actual,
        baseline_probability,
    )
    adjusted_log_loss = binary_log_loss(
        actual,
        adjusted_probability,
    )

    baseline_ece = calibration_error(
        actual,
        baseline_probability,
    )
    adjusted_ece = calibration_error(
        actual,
        adjusted_probability,
    )

    summary = {
        "district_component_sd": district_component_sd,
        "shrinkage_strength": shrinkage_strength,
        "multiplier_floor": multiplier_floor,
        "multiplier_ceiling": multiplier_ceiling,
        "scored_races": len(detail),
        "baseline_brier": baseline_brier,
        "adjusted_brier": adjusted_brier,
        "brier_change": adjusted_brier - baseline_brier,
        "baseline_log_loss": baseline_log_loss,
        "adjusted_log_loss": adjusted_log_loss,
        "log_loss_change": adjusted_log_loss - baseline_log_loss,
        "baseline_ece": baseline_ece,
        "adjusted_ece": adjusted_ece,
        "ece_change": adjusted_ece - baseline_ece,
        "baseline_50_interval_coverage": interval_coverage(
            actual_margin,
            predicted_margin,
            np.full(len(detail), BASELINE_TOTAL_SD),
            0.67448975,
        ),
        "adjusted_50_interval_coverage": interval_coverage(
            actual_margin,
            predicted_margin,
            adjusted_sd,
            0.67448975,
        ),
        "baseline_80_interval_coverage": interval_coverage(
            actual_margin,
            predicted_margin,
            np.full(len(detail), BASELINE_TOTAL_SD),
            1.28155157,
        ),
        "adjusted_80_interval_coverage": interval_coverage(
            actual_margin,
            predicted_margin,
            adjusted_sd,
            1.28155157,
        ),
        "baseline_95_interval_coverage": interval_coverage(
            actual_margin,
            predicted_margin,
            np.full(len(detail), BASELINE_TOTAL_SD),
            1.95996398,
        ),
        "adjusted_95_interval_coverage": interval_coverage(
            actual_margin,
            predicted_margin,
            adjusted_sd,
            1.95996398,
        ),
        "mean_multiplier": float(
            detail["district_uncertainty_multiplier"].mean()
        ),
        "multiplier_sd": float(
            detail["district_uncertainty_multiplier"].std(ddof=0)
        ),
        "minimum_multiplier": float(
            detail["district_uncertainty_multiplier"].min()
        ),
        "maximum_multiplier": float(
            detail["district_uncertainty_multiplier"].max()
        ),
        "mean_adjusted_total_sd": float(
            detail["adjusted_total_error_sd"].mean()
        ),
        "minimum_adjusted_total_sd": float(
            detail["adjusted_total_error_sd"].min()
        ),
        "maximum_adjusted_total_sd": float(
            detail["adjusted_total_error_sd"].max()
        ),
    }

    return summary, detail


def main() -> None:
    scored = load_scored_results()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, float]] = []
    details: dict[
        tuple[float, float, float, float],
        pd.DataFrame,
    ] = {}

    for (
        district_component_sd,
        shrinkage_strength,
        bounds,
    ) in product(
        DISTRICT_COMPONENT_GRID,
        SHRINKAGE_GRID,
        MULTIPLIER_BOUNDS_GRID,
    ):
        multiplier_floor, multiplier_ceiling = bounds

        summary, detail = evaluate_configuration(
            scored=scored,
            district_component_sd=district_component_sd,
            shrinkage_strength=shrinkage_strength,
            multiplier_floor=multiplier_floor,
            multiplier_ceiling=multiplier_ceiling,
        )

        key = (
            district_component_sd,
            shrinkage_strength,
            multiplier_floor,
            multiplier_ceiling,
        )

        summaries.append(summary)
        details[key] = detail

    summary_frame = pd.DataFrame(summaries)

    # Require no deterioration in either primary probabilistic metric.
    summary_frame["passes_primary_metrics"] = (
        (summary_frame["brier_change"] < 0.0)
        & (summary_frame["log_loss_change"] < 0.0)
    )

    # Rank primarily by log loss, then Brier score, then calibration.
    summary_frame = summary_frame.sort_values(
        [
            "passes_primary_metrics",
            "adjusted_log_loss",
            "adjusted_brier",
            "adjusted_ece",
        ],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    best = summary_frame.iloc[0]

    best_key = (
        float(best["district_component_sd"]),
        float(best["shrinkage_strength"]),
        float(best["multiplier_floor"]),
        float(best["multiplier_ceiling"]),
    )

    best_detail = details[best_key].copy()

    selection = pd.DataFrame(
        [
            {
                "selected_district_component_sd": best_key[0],
                "selected_shrinkage_strength": best_key[1],
                "selected_multiplier_floor": best_key[2],
                "selected_multiplier_ceiling": best_key[3],
                "baseline_brier": best["baseline_brier"],
                "selected_brier": best["adjusted_brier"],
                "brier_change": best["brier_change"],
                "baseline_log_loss": best["baseline_log_loss"],
                "selected_log_loss": best["adjusted_log_loss"],
                "log_loss_change": best["log_loss_change"],
                "baseline_ece": best["baseline_ece"],
                "selected_ece": best["adjusted_ece"],
                "ece_change": best["ece_change"],
                "passes_primary_metrics": bool(
                    best["passes_primary_metrics"]
                ),
                "promotion_recommendation": (
                    "PROMOTE"
                    if bool(best["passes_primary_metrics"])
                    else "DO NOT PROMOTE"
                ),
            }
        ]
    )

    summary_frame.to_csv(SUMMARY_PATH, index=False)
    best_detail.to_csv(DETAIL_PATH, index=False)
    selection.to_csv(SELECTION_PATH, index=False)

    print("Layer 5 district-uncertainty sensitivity complete")
    print("------------------------------------------------")
    print(f"Scored races: {len(scored)}")
    print(f"Configurations tested: {len(summary_frame)}")
    print()
    print("Best configuration:")
    print(
        selection.to_string(index=False)
    )

    print("\nTop 15 configurations:")
    display_columns = [
        "district_component_sd",
        "shrinkage_strength",
        "multiplier_floor",
        "multiplier_ceiling",
        "adjusted_brier",
        "brier_change",
        "adjusted_log_loss",
        "log_loss_change",
        "adjusted_ece",
        "ece_change",
        "mean_multiplier",
        "minimum_multiplier",
        "maximum_multiplier",
        "passes_primary_metrics",
    ]

    print(
        summary_frame[display_columns]
        .head(15)
        .to_string(index=False)
    )

    print("\nCoverage for selected configuration:")
    coverage_columns = [
        "baseline_50_interval_coverage",
        "adjusted_50_interval_coverage",
        "baseline_80_interval_coverage",
        "adjusted_80_interval_coverage",
        "baseline_95_interval_coverage",
        "adjusted_95_interval_coverage",
    ]

    print(
        best[coverage_columns]
        .to_frame()
        .T
        .to_string(index=False)
    )

    print(f"\nWrote: {SUMMARY_PATH}")
    print(f"Wrote: {DETAIL_PATH}")
    print(f"Wrote: {SELECTION_PATH}")


if __name__ == "__main__":
    main()
