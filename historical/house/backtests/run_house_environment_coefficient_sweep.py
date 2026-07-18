#!/usr/bin/env python3
"""
Run a validation-first House national-environment coefficient sweep.

The sweep evaluates three national-environment model families across nine
generic-ballot coefficients:

    1. Generic ballot only
    2. Generic ballot + presidential approval
    3. Generic ballot + presidential approval + midterm adjustment

The approval and midterm coefficients remain fixed at their current production
values of 0.50. The generic-ballot coefficient ranges from 0.00 through 1.10.

Each proposed national environment is added to the district presidential
baseline and evaluated against actual House margins using the canonical House
backtest scoring definitions.

Outputs include:
    - specification-level summary metrics
    - cycle-level metrics
    - race-level predictions
    - Pareto-optimal specifications
    - simplicity-aware recommendation
    - configuration metadata
    - validation report

This script does not modify production inputs or model configuration.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKTEST_DIR = Path(__file__).resolve().parent

if str(BACKTEST_DIR) not in sys.path:
    sys.path.insert(0, str(BACKTEST_DIR))

from house_backtest_metrics import (  # noqa: E402
    DEFAULT_PROBABILITY_SCALE,
    logistic_probability,
    score_forecasts,
)


INPUT_PATH = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "warehouse"
    / "house_historical_backtest_inputs_2016_2022.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "environment_coefficient_sweep"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "house_environment_coefficient_sweep_summary.csv"
)

BY_CYCLE_PATH = (
    OUTPUT_DIR
    / "house_environment_coefficient_sweep_by_cycle.csv"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "house_environment_coefficient_sweep_predictions.csv"
)

PARETO_PATH = (
    OUTPUT_DIR
    / "house_environment_coefficient_sweep_pareto.csv"
)

RECOMMENDATION_PATH = (
    OUTPUT_DIR
    / "house_environment_coefficient_sweep_recommendation.csv"
)

CONFIG_PATH = (
    OUTPUT_DIR
    / "house_environment_coefficient_sweep_config.json"
)

VALIDATION_PATH = (
    OUTPUT_DIR
    / "house_environment_coefficient_sweep_validation.txt"
)


GENERIC_BALLOT_COEFFICIENTS = tuple(
    round(value, 2)
    for value in np.arange(0.00, 1.1001, 0.05)
)

APPROVAL_COEFFICIENT = 0.50
MIDTERM_COEFFICIENT = 0.50
PROBABILITY_SCALE = DEFAULT_PROBABILITY_SCALE

EXPECTED_CYCLES = (2016, 2018, 2020, 2022)

MODEL_FAMILIES = (
    "generic_only",
    "generic_plus_approval",
    "generic_plus_approval_plus_midterm",
)

MODEL_LABELS = {
    "generic_only": "Generic ballot only",
    "generic_plus_approval": (
        "Generic ballot + approval"
    ),
    "generic_plus_approval_plus_midterm": (
        "Generic ballot + approval + midterm"
    ),
}

COMPLEXITY_SCORES = {
    "generic_only": 1,
    "generic_plus_approval": 2,
    "generic_plus_approval_plus_midterm": 3,
}


class ValidationError(RuntimeError):
    """Raised when sweep inputs or outputs violate their contract."""


@dataclass(frozen=True)
class Specification:
    model_family: str
    generic_ballot_coefficient: float

    @property
    def model_label(self) -> str:
        return MODEL_LABELS[self.model_family]

    @property
    def specification_id(self) -> str:
        return (
            f"{self.model_family}"
            f"__gb_{self.generic_ballot_coefficient:.2f}"
        )


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def require_columns(
    frame: pd.DataFrame,
    required_columns: set[str],
    *,
    label: str,
) -> None:
    missing = sorted(
        required_columns - set(frame.columns)
    )

    if missing:
        raise ValidationError(
            f"{label} is missing required columns: {missing}"
        )


def load_inputs() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Canonical House backtest input not found: {INPUT_PATH}"
        )

    frame = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    required_columns = {
        "cycle",
        "district",
        "district_id",
        "actual_dem_margin",
        "district_pres_margin_dem",
        "generic_ballot_margin_dem",
        "approval_adjustment_dem",
        "midterm_adjustment_dem",
        "include_in_canonical_margin_backtest",
    }

    require_columns(
        frame,
        required_columns,
        label="Canonical House backtest input",
    )

    numeric_columns = [
        "cycle",
        "actual_dem_margin",
        "district_pres_margin_dem",
        "generic_ballot_margin_dem",
        "approval_adjustment_dem",
        "midterm_adjustment_dem",
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame["include_in_canonical_margin_backtest"] = (
        parse_bool_series(
            frame["include_in_canonical_margin_backtest"]
        )
    )

    eligible = frame.loc[
        frame["include_in_canonical_margin_backtest"]
    ].copy()

    required_nonmissing = [
        "cycle",
        "actual_dem_margin",
        "district_pres_margin_dem",
        "generic_ballot_margin_dem",
        "approval_adjustment_dem",
        "midterm_adjustment_dem",
    ]

    missing_mask = eligible[
        required_nonmissing
    ].isna().any(axis=1)

    if missing_mask.any():
        examples = eligible.loc[
            missing_mask,
            [
                "cycle",
                "district",
                *required_nonmissing[1:],
            ],
        ].head(10)

        raise ValidationError(
            "Canonical margin-backtest rows contain missing required "
            "values:\n"
            + examples.to_string(index=False)
        )

    eligible["cycle"] = eligible[
        "cycle"
    ].astype(int)

    observed_cycles = tuple(
        sorted(eligible["cycle"].unique())
    )

    if observed_cycles != EXPECTED_CYCLES:
        raise ValidationError(
            f"Expected cycles {EXPECTED_CYCLES}; "
            f"found {observed_cycles}."
        )

    duplicate_count = int(
        eligible.duplicated(
            subset=["cycle", "district_id"]
        ).sum()
    )

    if duplicate_count:
        raise ValidationError(
            f"Found {duplicate_count} duplicate "
            "cycle-district rows."
        )

    return eligible.reset_index(drop=True)


def build_specifications() -> list[Specification]:
    return [
        Specification(
            model_family=model_family,
            generic_ballot_coefficient=coefficient,
        )
        for model_family in MODEL_FAMILIES
        for coefficient in GENERIC_BALLOT_COEFFICIENTS
    ]


def calculate_environment(
    frame: pd.DataFrame,
    specification: Specification,
) -> pd.DataFrame:
    output = frame.copy()

    output["generic_ballot_contribution_dem"] = (
        output["generic_ballot_margin_dem"]
        * specification.generic_ballot_coefficient
    )

    if specification.model_family in {
        "generic_plus_approval",
        "generic_plus_approval_plus_midterm",
    }:
        output["approval_contribution_dem"] = (
            output["approval_adjustment_dem"]
            * APPROVAL_COEFFICIENT
        )
    else:
        output["approval_contribution_dem"] = 0.0

    if (
        specification.model_family
        == "generic_plus_approval_plus_midterm"
    ):
        output["midterm_contribution_dem"] = (
            output["midterm_adjustment_dem"]
            * MIDTERM_COEFFICIENT
        )
    else:
        output["midterm_contribution_dem"] = 0.0

    output["national_environment_margin_dem"] = (
        output["generic_ballot_contribution_dem"]
        + output["approval_contribution_dem"]
        + output["midterm_contribution_dem"]
    )

    output["forecast_margin_dem"] = (
        output["district_pres_margin_dem"]
        + output["national_environment_margin_dem"]
    )

    output["dem_win_probability"] = (
        logistic_probability(
            output["forecast_margin_dem"],
            error_sd=PROBABILITY_SCALE,
        )
    )

    output["specification_id"] = (
        specification.specification_id
    )

    output["model_family"] = (
        specification.model_family
    )

    output["model_label"] = (
        specification.model_label
    )

    output["generic_ballot_coefficient"] = (
        specification.generic_ballot_coefficient
    )

    output["approval_coefficient"] = (
        APPROVAL_COEFFICIENT
        if specification.model_family
        in {
            "generic_plus_approval",
            "generic_plus_approval_plus_midterm",
        }
        else 0.0
    )

    output["midterm_coefficient"] = (
        MIDTERM_COEFFICIENT
        if specification.model_family
        == "generic_plus_approval_plus_midterm"
        else 0.0
    )

    return output


def metrics_row(
    frame: pd.DataFrame,
    specification: Specification,
    *,
    cycle: int | None,
) -> dict[str, object]:
    metrics = score_forecasts(
        actual_margin_dem=frame["actual_dem_margin"],
        forecast_margin_dem=frame["forecast_margin_dem"],
        probability_dem=frame["dem_win_probability"],
    )

    return {
        "specification_id": specification.specification_id,
        "model_family": specification.model_family,
        "model_label": specification.model_label,
        "generic_ballot_coefficient": (
            specification.generic_ballot_coefficient
        ),
        "approval_coefficient": (
            APPROVAL_COEFFICIENT
            if specification.model_family
            in {
                "generic_plus_approval",
                "generic_plus_approval_plus_midterm",
            }
            else 0.0
        ),
        "midterm_coefficient": (
            MIDTERM_COEFFICIENT
            if specification.model_family
            == "generic_plus_approval_plus_midterm"
            else 0.0
        ),
        "complexity_score": (
            COMPLEXITY_SCORES[
                specification.model_family
            ]
        ),
        "cycle": cycle,
        **metrics.as_dict(),
    }


def add_ranking_columns(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    output = summary.copy()

    output["absolute_mean_error"] = (
        output["mean_margin_error_dem_bias"].abs()
    )

    ascending_metrics = [
        "mean_absolute_error",
        "rmse",
        "brier_score",
        "log_loss",
        "absolute_mean_error",
        "absolute_expected_seat_error",
    ]

    output["absolute_expected_seat_error"] = (
        output["expected_win_count_error"].abs()
    )

    for metric in ascending_metrics:
        output[f"{metric}_rank"] = output[
            metric
        ].rank(
            method="average",
            ascending=True,
        )

    output["winner_accuracy_rank"] = output[
        "winner_accuracy"
    ].rank(
        method="average",
        ascending=False,
    )

    performance_rank_columns = [
        "mean_absolute_error_rank",
        "rmse_rank",
        "brier_score_rank",
        "winner_accuracy_rank",
        "absolute_mean_error_rank",
        "absolute_expected_seat_error_rank",
    ]

    output["performance_score"] = (
        output[performance_rank_columns]
        .mean(axis=1)
        / len(output)
    )

    output["complexity_penalty"] = (
        (output["complexity_score"] - 1)
        * 0.05
    )

    output["simplicity_adjusted_score"] = (
        output["performance_score"]
        + output["complexity_penalty"]
    )

    return output.sort_values(
        [
            "simplicity_adjusted_score",
            "performance_score",
            "mean_absolute_error",
            "rmse",
            "brier_score",
            "generic_ballot_coefficient",
        ]
    ).reset_index(drop=True)


def dominates(
    candidate: pd.Series,
    comparison: pd.Series,
) -> bool:
    minimize_metrics = [
        "mean_absolute_error",
        "rmse",
        "brier_score",
        "absolute_mean_error",
        "absolute_expected_seat_error",
    ]

    candidate_values = np.array(
        [
            float(candidate[metric])
            for metric in minimize_metrics
        ],
        dtype=float,
    )

    comparison_values = np.array(
        [
            float(comparison[metric])
            for metric in minimize_metrics
        ],
        dtype=float,
    )

    candidate_accuracy = float(
        candidate["winner_accuracy"]
    )

    comparison_accuracy = float(
        comparison["winner_accuracy"]
    )

    no_worse = (
        np.all(candidate_values <= comparison_values)
        and candidate_accuracy >= comparison_accuracy
    )

    strictly_better = (
        np.any(candidate_values < comparison_values)
        or candidate_accuracy > comparison_accuracy
    )

    return bool(no_worse and strictly_better)


def build_pareto_table(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    pareto_rows = []

    for index, candidate in summary.iterrows():
        dominated = False

        for other_index, comparison in summary.iterrows():
            if index == other_index:
                continue

            if dominates(comparison, candidate):
                dominated = True
                break

        if not dominated:
            pareto_rows.append(candidate)

    if not pareto_rows:
        raise ValidationError(
            "Pareto analysis produced no specifications."
        )

    return (
        pd.DataFrame(pareto_rows)
        .sort_values(
            [
                "simplicity_adjusted_score",
                "performance_score",
                "mean_absolute_error",
            ]
        )
        .reset_index(drop=True)
    )


def validate_outputs(
    inputs: pd.DataFrame,
    summary: pd.DataFrame,
    by_cycle: pd.DataFrame,
    predictions: pd.DataFrame,
) -> list[str]:
    messages: list[str] = []

    expected_specifications = (
        len(MODEL_FAMILIES)
        * len(GENERIC_BALLOT_COEFFICIENTS)
    )

    if len(summary) != expected_specifications:
        raise ValidationError(
            f"Expected {expected_specifications} summary rows; "
            f"found {len(summary)}."
        )

    expected_cycle_rows = (
        expected_specifications
        * len(EXPECTED_CYCLES)
    )

    if len(by_cycle) != expected_cycle_rows:
        raise ValidationError(
            f"Expected {expected_cycle_rows} cycle rows; "
            f"found {len(by_cycle)}."
        )

    expected_prediction_rows = (
        expected_specifications
        * len(inputs)
    )

    if len(predictions) != expected_prediction_rows:
        raise ValidationError(
            f"Expected {expected_prediction_rows} prediction rows; "
            f"found {len(predictions)}."
        )

    duplicate_predictions = int(
        predictions.duplicated(
            subset=[
                "specification_id",
                "cycle",
                "district_id",
            ]
        ).sum()
    )

    if duplicate_predictions:
        raise ValidationError(
            f"Found {duplicate_predictions} duplicate "
            "prediction keys."
        )

    numeric_columns = [
        "national_environment_margin_dem",
        "forecast_margin_dem",
        "dem_win_probability",
        "actual_dem_margin",
    ]

    numeric_values = predictions[
        numeric_columns
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValidationError(
            "Predictions contain non-finite numeric values."
        )

    invalid_probabilities = (
        (predictions["dem_win_probability"] < 0.0)
        | (predictions["dem_win_probability"] > 1.0)
    )

    if invalid_probabilities.any():
        raise ValidationError(
            "Predictions contain probabilities outside [0, 1]."
        )

    expected_scored_races = len(inputs)

    if not (
        summary["scored_races"]
        == expected_scored_races
    ).all():
        raise ValidationError(
            "Summary scored-race counts do not match inputs."
        )

    cycle_counts = inputs.groupby("cycle").size()

    for cycle, expected_count in cycle_counts.items():
        observed = by_cycle.loc[
            by_cycle["cycle"].eq(int(cycle)),
            "scored_races",
        ]

        if not (observed == int(expected_count)).all():
            raise ValidationError(
                f"Cycle {cycle} scored-race counts "
                "do not match inputs."
            )

    current_specification = summary.loc[
        summary["model_family"].eq(
            "generic_plus_approval_plus_midterm"
        )
        & np.isclose(
            summary["generic_ballot_coefficient"],
            0.85,
        )
    ]

    if len(current_specification) != 1:
        raise ValidationError(
            "Could not identify the current production "
            "0.85/0.50/0.50 specification."
        )

    messages.extend(
        [
            "Input validation: PASSED",
            (
                "Expected specification count: "
                f"{expected_specifications}"
            ),
            (
                "Summary specification count: "
                f"{len(summary)}"
            ),
            (
                "Cycle-level row count: "
                f"{len(by_cycle)}"
            ),
            (
                "Race-level prediction count: "
                f"{len(predictions)}"
            ),
            (
                "Scored input races: "
                f"{len(inputs)}"
            ),
            (
                "Cycles: "
                + ", ".join(
                    str(cycle)
                    for cycle in EXPECTED_CYCLES
                )
            ),
            (
                "Current production reference located: "
                "GB=0.85, approval=0.50, midterm=0.50"
            ),
            "Duplicate-key validation: PASSED",
            "Finite-value validation: PASSED",
            "Probability-range validation: PASSED",
            "Scoring-count validation: PASSED",
            "Validation: PASSED",
        ]
    )

    return messages


def main() -> None:
    inputs = load_inputs()
    specifications = build_specifications()

    summary_rows: list[dict[str, object]] = []
    cycle_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for specification in specifications:
        predictions = calculate_environment(
            inputs,
            specification,
        )

        summary_rows.append(
            metrics_row(
                predictions,
                specification,
                cycle=None,
            )
        )

        for cycle, cycle_frame in predictions.groupby(
            "cycle",
            sort=True,
        ):
            cycle_rows.append(
                metrics_row(
                    cycle_frame,
                    specification,
                    cycle=int(cycle),
                )
            )

        keep_columns = [
            "specification_id",
            "model_family",
            "model_label",
            "generic_ballot_coefficient",
            "approval_coefficient",
            "midterm_coefficient",
            "cycle",
            "district_id",
            "district",
            "actual_dem_margin",
            "district_pres_margin_dem",
            "generic_ballot_margin_dem",
            "approval_adjustment_dem",
            "midterm_adjustment_dem",
            "generic_ballot_contribution_dem",
            "approval_contribution_dem",
            "midterm_contribution_dem",
            "national_environment_margin_dem",
            "forecast_margin_dem",
            "dem_win_probability",
        ]

        prediction_frames.append(
            predictions[keep_columns].copy()
        )

    summary = pd.DataFrame(summary_rows)

    summary["absolute_expected_seat_error"] = (
        summary["expected_win_count_error"].abs()
    )

    summary = add_ranking_columns(summary)

    by_cycle = (
        pd.DataFrame(cycle_rows)
        .sort_values(
            [
                "cycle",
                "model_family",
                "generic_ballot_coefficient",
            ]
        )
        .reset_index(drop=True)
    )

    predictions = (
        pd.concat(
            prediction_frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "specification_id",
                "cycle",
                "district_id",
            ]
        )
        .reset_index(drop=True)
    )

    pareto = build_pareto_table(summary)

    recommendation = (
        summary.head(1)
        .copy()
        .reset_index(drop=True)
    )

    validation_messages = validate_outputs(
        inputs,
        summary,
        by_cycle,
        predictions,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    by_cycle.to_csv(
        BY_CYCLE_PATH,
        index=False,
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    pareto.to_csv(
        PARETO_PATH,
        index=False,
    )

    recommendation.to_csv(
        RECOMMENDATION_PATH,
        index=False,
    )

    config = {
        "input_path": str(
            INPUT_PATH.relative_to(PROJECT_ROOT)
        ),
        "output_directory": str(
            OUTPUT_DIR.relative_to(PROJECT_ROOT)
        ),
        "cycles": list(EXPECTED_CYCLES),
        "model_families": list(MODEL_FAMILIES),
        "generic_ballot_coefficients": list(
            GENERIC_BALLOT_COEFFICIENTS
        ),
        "approval_coefficient": APPROVAL_COEFFICIENT,
        "midterm_coefficient": MIDTERM_COEFFICIENT,
        "probability_scale": PROBABILITY_SCALE,
        "scoring_population": (
            "include_in_canonical_margin_backtest == True"
        ),
        "forecast_formula": (
            "district presidential margin + "
            "candidate national environment"
        ),
        "complexity_penalty_per_added_component": 0.05,
        "production_inputs_modified": False,
    }

    CONFIG_PATH.write_text(
        json.dumps(
            config,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    VALIDATION_PATH.write_text(
        "\n".join(validation_messages) + "\n",
        encoding="utf-8",
    )

    best = recommendation.iloc[0]

    current = summary.loc[
        summary["model_family"].eq(
            "generic_plus_approval_plus_midterm"
        )
        & np.isclose(
            summary["generic_ballot_coefficient"],
            0.85,
        )
    ].iloc[0]

    print("House Environment Coefficient Sweep")
    print("=" * 45)
    print(f"Input observations: {len(inputs)}")
    print(
        "Cycles: "
        + ", ".join(
            str(cycle)
            for cycle in EXPECTED_CYCLES
        )
    )
    print(
        f"Specifications evaluated: {len(summary)}"
    )
    print(
        f"Pareto-optimal specifications: {len(pareto)}"
    )

    print()
    print("Simplicity-aware recommendation")
    print("-" * 35)
    print(best["model_label"])
    print(
        "GB coefficient = "
        f"{best['generic_ballot_coefficient']:.2f}"
    )
    print(
        "Approval coefficient = "
        f"{best['approval_coefficient']:.2f}"
    )
    print(
        "Midterm coefficient = "
        f"{best['midterm_coefficient']:.2f}"
    )
    print()
    print(
        f"MAE = {best['mean_absolute_error']:.6f}"
    )
    print(f"RMSE = {best['rmse']:.6f}")
    print(
        f"Brier = {best['brier_score']:.6f}"
    )
    print(
        "Winner accuracy = "
        f"{best['winner_accuracy']:.4%}"
    )
    print(
        "Mean error = "
        f"{best['mean_margin_error_dem_bias']:+.6f}"
    )
    print(
        "Expected seat error = "
        f"{best['expected_win_count_error']:+.4f}"
    )

    print()
    print("Current production reference")
    print("-" * 28)
    print(
        "Generic ballot + approval + midterm"
    )
    print(
        "GB=0.85, approval=0.50, midterm=0.50"
    )
    print(
        f"MAE = {current['mean_absolute_error']:.6f}"
    )
    print(f"RMSE = {current['rmse']:.6f}")
    print(
        f"Brier = {current['brier_score']:.6f}"
    )
    print(
        "Winner accuracy = "
        f"{current['winner_accuracy']:.4%}"
    )
    print(
        "Mean error = "
        f"{current['mean_margin_error_dem_bias']:+.6f}"
    )

    print()
    print("Validation: PASSED")
    print(f"Wrote: {SUMMARY_PATH}")
    print(f"Wrote: {BY_CYCLE_PATH}")
    print(f"Wrote: {PREDICTIONS_PATH}")
    print(f"Wrote: {PARETO_PATH}")
    print(f"Wrote: {RECOMMENDATION_PATH}")
    print(f"Wrote: {CONFIG_PATH}")
    print(f"Wrote: {VALIDATION_PATH}")


if __name__ == "__main__":
    main()
