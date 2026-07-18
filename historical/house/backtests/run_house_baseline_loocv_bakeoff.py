#!/usr/bin/env python3
"""
Leave-one-cycle-out validation of House baseline architectures.

For each held-out election cycle, this script:

1. Uses the other cycles as training data.
2. Selects the best environment specification separately for:
   - raw presidential margin
   - normalized partisan baseline
3. Freezes the selected specification.
4. Scores it on the held-out cycle.

No production files are modified.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKTEST_DIR = Path(__file__).resolve().parent

if str(BACKTEST_DIR) not in sys.path:
    sys.path.insert(0, str(BACKTEST_DIR))

import run_house_baseline_environment_bakeoff as bakeoff  # noqa: E402
import run_house_environment_coefficient_sweep as sweep  # noqa: E402


OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "baseline_loocv_bakeoff"
)

TRAINING_SELECTION_PATH = (
    OUTPUT_DIR
    / "house_baseline_loocv_training_selection.csv"
)

FOLD_RESULTS_PATH = (
    OUTPUT_DIR
    / "house_baseline_loocv_fold_results.csv"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "house_baseline_loocv_predictions.csv"
)

OVERALL_PATH = (
    OUTPUT_DIR
    / "house_baseline_loocv_overall.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "house_baseline_loocv_summary.csv"
)

COEFFICIENT_STABILITY_PATH = (
    OUTPUT_DIR
    / "house_baseline_loocv_coefficient_stability.csv"
)

VALIDATION_PATH = (
    OUTPUT_DIR
    / "house_baseline_loocv_validation.txt"
)

CONFIG_PATH = (
    OUTPUT_DIR
    / "house_baseline_loocv_config.json"
)


class ValidationError(RuntimeError):
    """Raised when LOOCV inputs or outputs violate their contract."""


def select_training_specification(
    training_frame: pd.DataFrame,
    baseline_type: str,
    specifications: list[bakeoff.Specification],
) -> tuple[bakeoff.Specification, pd.DataFrame]:
    """
    Select the best specification using training cycles only.

    Selection uses the same ranking logic as the existing coefficient sweep.
    """

    rows: list[dict[str, object]] = []

    eligible_specs = [
        specification
        for specification in specifications
        if specification.baseline_type == baseline_type
    ]

    for specification in eligible_specs:
        calculated = bakeoff.calculate_forecast(
            training_frame,
            specification,
        )

        rows.append(
            bakeoff.metrics_row(
                calculated,
                specification,
                cycle=None,
            )
        )

    training_results = pd.DataFrame(rows)
    training_results = sweep.add_ranking_columns(
        training_results
    )

    training_results = training_results.sort_values(
        [
            "simplicity_adjusted_score",
            "performance_score",
            "mean_absolute_error",
            "rmse",
            "brier_score",
            "log_loss",
            "complexity_score",
            "generic_ballot_coefficient",
        ]
    ).reset_index(drop=True)

    winning_row = training_results.iloc[0]

    selected_specification = next(
        specification
        for specification in eligible_specs
        if specification.specification_id
        == winning_row["specification_id"]
    )

    return selected_specification, training_results


def build_fold_result(
    calculated: pd.DataFrame,
    specification: bakeoff.Specification,
    holdout_cycle: int,
    training_cycles: tuple[int, ...],
) -> dict[str, object]:
    metrics = sweep.score_forecasts(
        actual_margin_dem=calculated["actual_dem_margin"],
        forecast_margin_dem=calculated["forecast_margin_dem"],
        probability_dem=calculated["dem_win_probability"],
    )

    return {
        "holdout_cycle": holdout_cycle,
        "training_cycles": ",".join(
            str(cycle)
            for cycle in training_cycles
        ),
        "baseline_type": specification.baseline_type,
        "baseline_label": bakeoff.BASELINE_LABELS[
            specification.baseline_type
        ],
        "selected_specification_id": (
            specification.specification_id
        ),
        "selected_model_family": (
            specification.model_family
        ),
        "selected_model_label": sweep.MODEL_LABELS[
            specification.model_family
        ],
        "selected_generic_ballot_coefficient": (
            specification.generic_ballot_coefficient
        ),
        "observation_count": int(len(calculated)),
        **metrics.as_dict(),
    }


def build_overall_results(
    fold_results: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = [
        "mean_absolute_error",
        "rmse",
        "brier_score",
        "log_loss",
        "winner_accuracy",
        "mean_margin_error_dem_bias",
        "expected_win_count_error",
    ]

    rows: list[dict[str, object]] = []

    for baseline_type in bakeoff.BASELINE_TYPES:
        subset = fold_results.loc[
            fold_results["baseline_type"].eq(
                baseline_type
            )
        ].copy()

        row: dict[str, object] = {
            "baseline_type": baseline_type,
            "baseline_label": bakeoff.BASELINE_LABELS[
                baseline_type
            ],
            "fold_count": int(len(subset)),
            "total_observation_count": int(
                subset["observation_count"].sum()
            ),
        }

        for metric in metric_columns:
            row[f"mean_fold_{metric}"] = float(
                subset[metric].mean()
            )

        row["mean_absolute_dem_bias"] = float(
            subset[
                "mean_margin_error_dem_bias"
            ].abs().mean()
        )

        row["mean_absolute_expected_win_count_error"] = (
            float(
                subset[
                    "expected_win_count_error"
                ].abs().mean()
            )
        )

        row["weighted_mean_absolute_error"] = float(
            np.average(
                subset["mean_absolute_error"],
                weights=subset["observation_count"],
            )
        )

        row["weighted_rmse"] = float(
            np.sqrt(
                np.average(
                    subset["rmse"] ** 2,
                    weights=subset[
                        "observation_count"
                    ],
                )
            )
        )

        row["weighted_brier_score"] = float(
            np.average(
                subset["brier_score"],
                weights=subset["observation_count"],
            )
        )

        row["weighted_log_loss"] = float(
            np.average(
                subset["log_loss"],
                weights=subset["observation_count"],
            )
        )

        row["weighted_winner_accuracy"] = float(
            np.average(
                subset["winner_accuracy"],
                weights=subset["observation_count"],
            )
        )

        rows.append(row)

    overall = pd.DataFrame(rows)

    overall["mae_rank"] = overall[
        "weighted_mean_absolute_error"
    ].rank(
        method="min",
        ascending=True,
    )

    overall["rmse_rank"] = overall[
        "weighted_rmse"
    ].rank(
        method="min",
        ascending=True,
    )

    overall["brier_rank"] = overall[
        "weighted_brier_score"
    ].rank(
        method="min",
        ascending=True,
    )

    overall["accuracy_rank"] = overall[
        "weighted_winner_accuracy"
    ].rank(
        method="min",
        ascending=False,
    )

    overall["overall_validation_rank"] = (
        overall[
            [
                "mae_rank",
                "rmse_rank",
                "brier_rank",
                "accuracy_rank",
            ]
        ]
        .mean(axis=1)
        .rank(
            method="min",
            ascending=True,
        )
    )

    return overall.sort_values(
        [
            "overall_validation_rank",
            "weighted_mean_absolute_error",
            "weighted_rmse",
            "weighted_brier_score",
        ]
    ).reset_index(drop=True)


def build_summary(
    fold_results: pd.DataFrame,
) -> pd.DataFrame:
    raw = fold_results.loc[
        fold_results["baseline_type"].eq(
            "raw_presidential_margin"
        )
    ].copy()

    normalized = fold_results.loc[
        fold_results["baseline_type"].eq(
            "normalized_partisan_baseline"
        )
    ].copy()

    raw = raw.rename(
        columns={
            column: f"raw_{column}"
            for column in raw.columns
            if column != "holdout_cycle"
        }
    )

    normalized = normalized.rename(
        columns={
            column: f"normalized_{column}"
            for column in normalized.columns
            if column != "holdout_cycle"
        }
    )

    summary = raw.merge(
        normalized,
        on="holdout_cycle",
        how="inner",
        validate="one_to_one",
    )

    summary["mae_difference_normalized_minus_raw"] = (
        summary["normalized_mean_absolute_error"]
        - summary["raw_mean_absolute_error"]
    )

    summary["rmse_difference_normalized_minus_raw"] = (
        summary["normalized_rmse"]
        - summary["raw_rmse"]
    )

    summary["brier_difference_normalized_minus_raw"] = (
        summary["normalized_brier_score"]
        - summary["raw_brier_score"]
    )

    summary["accuracy_difference_normalized_minus_raw"] = (
        summary["normalized_winner_accuracy"]
        - summary["raw_winner_accuracy"]
    )

    summary["absolute_bias_difference_normalized_minus_raw"] = (
        summary[
            "normalized_mean_margin_error_dem_bias"
        ].abs()
        - summary[
            "raw_mean_margin_error_dem_bias"
        ].abs()
    )

    summary[
        "absolute_seat_error_difference_normalized_minus_raw"
    ] = (
        summary[
            "normalized_expected_win_count_error"
        ].abs()
        - summary[
            "raw_expected_win_count_error"
        ].abs()
    )

    summary["mae_winner"] = np.where(
        summary[
            "normalized_mean_absolute_error"
        ]
        < summary["raw_mean_absolute_error"],
        "normalized_partisan_baseline",
        np.where(
            summary[
                "normalized_mean_absolute_error"
            ]
            > summary["raw_mean_absolute_error"],
            "raw_presidential_margin",
            "tie",
        ),
    )

    return summary.sort_values(
        "holdout_cycle"
    ).reset_index(drop=True)


def validate_outputs(
    inputs: pd.DataFrame,
    training_selection: pd.DataFrame,
    fold_results: pd.DataFrame,
    predictions: pd.DataFrame,
    overall: pd.DataFrame,
) -> list[str]:
    expected_folds = len(sweep.EXPECTED_CYCLES)
    expected_fold_rows = (
        expected_folds
        * len(bakeoff.BASELINE_TYPES)
    )

    if len(fold_results) != expected_fold_rows:
        raise ValidationError(
            f"Expected {expected_fold_rows} fold-result rows; "
            f"found {len(fold_results)}."
        )

    if len(training_selection) != expected_fold_rows:
        raise ValidationError(
            "Expected one training selection per baseline "
            f"and fold: {expected_fold_rows}; "
            f"found {len(training_selection)}."
        )

    if len(overall) != len(bakeoff.BASELINE_TYPES):
        raise ValidationError(
            "Overall results do not contain exactly one "
            "row per baseline architecture."
        )

    expected_prediction_rows = (
        len(inputs)
        * len(bakeoff.BASELINE_TYPES)
    )

    if len(predictions) != expected_prediction_rows:
        raise ValidationError(
            f"Expected {expected_prediction_rows} held-out "
            f"predictions; found {len(predictions)}."
        )

    prediction_duplicates = predictions.duplicated(
        subset=[
            "holdout_cycle",
            "baseline_type",
            "district_id",
        ]
    )

    if prediction_duplicates.any():
        raise ValidationError(
            "Held-out predictions contain duplicate "
            "fold-baseline-district rows."
        )

    if not predictions[
        "dem_win_probability"
    ].between(0.0, 1.0).all():
        raise ValidationError(
            "Held-out probabilities fall outside [0, 1]."
        )

    if not predictions[
        "cycle"
    ].eq(
        predictions["holdout_cycle"]
    ).all():
        raise ValidationError(
            "At least one prediction was generated for a "
            "training cycle rather than its holdout cycle."
        )

    observed_cycles = tuple(
        sorted(
            fold_results["holdout_cycle"].unique()
        )
    )

    if observed_cycles != sweep.EXPECTED_CYCLES:
        raise ValidationError(
            f"Expected holdout cycles "
            f"{sweep.EXPECTED_CYCLES}; "
            f"found {observed_cycles}."
        )

    selection_counts = (
        training_selection.groupby(
            [
                "holdout_cycle",
                "baseline_type",
            ]
        )
        .size()
    )

    if not selection_counts.eq(1).all():
        raise ValidationError(
            "Each fold and baseline must have exactly one "
            "selected training specification."
        )

    messages = [
        "House baseline LOOCV validation: PASSED",
        f"Holdout cycles: {observed_cycles}",
        (
            f"Baseline architectures: "
            f"{len(bakeoff.BASELINE_TYPES)}"
        ),
        (
            f"Training selections: "
            f"{len(training_selection):,}"
        ),
        (
            f"Held-out fold results: "
            f"{len(fold_results):,}"
        ),
        (
            f"Held-out prediction rows: "
            f"{len(predictions):,}"
        ),
        (
            "No held-out observations were used during "
            "fold-specific specification selection."
        ),
        (
            "Production files modified: False"
        ),
    ]

    return messages


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    inputs = bakeoff.load_inputs()
    specifications = bakeoff.build_specifications()

    training_selection_rows: list[
        dict[str, object]
    ] = []

    fold_result_rows: list[
        dict[str, object]
    ] = []

    prediction_frames: list[pd.DataFrame] = []

    for holdout_cycle in sweep.EXPECTED_CYCLES:
        training_cycles = tuple(
            cycle
            for cycle in sweep.EXPECTED_CYCLES
            if cycle != holdout_cycle
        )

        training_frame = inputs.loc[
            inputs["cycle"].isin(training_cycles)
        ].copy()

        holdout_frame = inputs.loc[
            inputs["cycle"].eq(holdout_cycle)
        ].copy()

        if training_frame.empty:
            raise ValidationError(
                f"Training frame is empty for holdout "
                f"{holdout_cycle}."
            )

        if holdout_frame.empty:
            raise ValidationError(
                f"Holdout frame is empty for cycle "
                f"{holdout_cycle}."
            )

        for baseline_type in bakeoff.BASELINE_TYPES:
            (
                selected_specification,
                training_results,
            ) = select_training_specification(
                training_frame,
                baseline_type,
                specifications,
            )

            selected_training_row = (
                training_results.iloc[0]
            )

            training_selection_rows.append(
                {
                    "holdout_cycle": holdout_cycle,
                    "training_cycles": ",".join(
                        str(cycle)
                        for cycle in training_cycles
                    ),
                    "baseline_type": baseline_type,
                    "baseline_label": (
                        bakeoff.BASELINE_LABELS[
                            baseline_type
                        ]
                    ),
                    "selected_specification_id": (
                        selected_specification
                        .specification_id
                    ),
                    "selected_model_family": (
                        selected_specification
                        .model_family
                    ),
                    "selected_model_label": (
                        sweep.MODEL_LABELS[
                            selected_specification
                            .model_family
                        ]
                    ),
                    "selected_generic_ballot_coefficient": (
                        selected_specification
                        .generic_ballot_coefficient
                    ),
                    "training_observation_count": int(
                        len(training_frame)
                    ),
                    "training_mean_absolute_error": (
                        selected_training_row[
                            "mean_absolute_error"
                        ]
                    ),
                    "training_rmse": (
                        selected_training_row["rmse"]
                    ),
                    "training_brier_score": (
                        selected_training_row[
                            "brier_score"
                        ]
                    ),
                    "training_log_loss": (
                        selected_training_row[
                            "log_loss"
                        ]
                    ),
                    "training_winner_accuracy": (
                        selected_training_row[
                            "winner_accuracy"
                        ]
                    ),
                    "training_mean_margin_error_dem_bias": (
                        selected_training_row[
                            "mean_margin_error_dem_bias"
                        ]
                    ),
                    "training_expected_win_count_error": (
                        selected_training_row[
                            "expected_win_count_error"
                        ]
                    ),
                    "training_performance_score": (
                        selected_training_row[
                            "performance_score"
                        ]
                    ),
                    "training_simplicity_adjusted_score": (
                        selected_training_row[
                            "simplicity_adjusted_score"
                        ]
                    ),
                }
            )

            calculated = bakeoff.calculate_forecast(
                holdout_frame,
                selected_specification,
            )

            fold_result_rows.append(
                build_fold_result(
                    calculated,
                    selected_specification,
                    holdout_cycle,
                    training_cycles,
                )
            )

            calculated = calculated.copy()

            calculated["holdout_cycle"] = (
                holdout_cycle
            )

            calculated["training_cycles"] = (
                ",".join(
                    str(cycle)
                    for cycle in training_cycles
                )
            )

            calculated[
                "selected_model_family"
            ] = selected_specification.model_family

            calculated[
                "selected_model_label"
            ] = sweep.MODEL_LABELS[
                selected_specification.model_family
            ]

            calculated[
                "selected_generic_ballot_coefficient"
            ] = (
                selected_specification
                .generic_ballot_coefficient
            )

            prediction_frames.append(
                calculated[
                    [
                        "holdout_cycle",
                        "training_cycles",
                        "baseline_type",
                        "baseline_label",
                        "selected_model_family",
                        "selected_model_label",
                        "selected_generic_ballot_coefficient",
                        "cycle",
                        "district",
                        "district_id",
                        "presidential_result_year",
                        "actual_dem_margin",
                        "district_pres_margin_dem",
                        "national_pres_margin_dem",
                        "district_partisan_baseline_dem",
                        "selected_baseline_margin_dem",
                        "national_environment_margin_dem",
                        "forecast_margin_dem",
                        "dem_win_probability",
                    ]
                ].copy()
            )

    training_selection = (
        pd.DataFrame(training_selection_rows)
        .sort_values(
            [
                "holdout_cycle",
                "baseline_type",
            ]
        )
        .reset_index(drop=True)
    )

    fold_results = (
        pd.DataFrame(fold_result_rows)
        .sort_values(
            [
                "holdout_cycle",
                "baseline_type",
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
                "holdout_cycle",
                "baseline_type",
                "district_id",
            ]
        )
        .reset_index(drop=True)
    )

    overall = build_overall_results(
        fold_results
    )

    summary = build_summary(
        fold_results
    )

    coefficient_stability = (
        training_selection.groupby(
            [
                "baseline_type",
                "baseline_label",
            ],
            as_index=False,
        )
        .agg(
            fold_count=(
                "holdout_cycle",
                "count",
            ),
            minimum_selected_coefficient=(
                "selected_generic_ballot_coefficient",
                "min",
            ),
            maximum_selected_coefficient=(
                "selected_generic_ballot_coefficient",
                "max",
            ),
            mean_selected_coefficient=(
                "selected_generic_ballot_coefficient",
                "mean",
            ),
            median_selected_coefficient=(
                "selected_generic_ballot_coefficient",
                "median",
            ),
            coefficient_standard_deviation=(
                "selected_generic_ballot_coefficient",
                "std",
            ),
            unique_selected_coefficients=(
                "selected_generic_ballot_coefficient",
                "nunique",
            ),
        )
        .sort_values("baseline_type")
        .reset_index(drop=True)
    )

    validation_messages = validate_outputs(
        inputs,
        training_selection,
        fold_results,
        predictions,
        overall,
    )

    training_selection.to_csv(
        TRAINING_SELECTION_PATH,
        index=False,
    )

    fold_results.to_csv(
        FOLD_RESULTS_PATH,
        index=False,
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    overall.to_csv(
        OVERALL_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    coefficient_stability.to_csv(
        COEFFICIENT_STABILITY_PATH,
        index=False,
    )

    config = {
        "cycles": list(sweep.EXPECTED_CYCLES),
        "baseline_types": list(
            bakeoff.BASELINE_TYPES
        ),
        "model_families": list(
            sweep.MODEL_FAMILIES
        ),
        "generic_ballot_coefficients": [
            float(value)
            for value in (
                sweep.GENERIC_BALLOT_COEFFICIENTS
            )
        ],
        "selection_method": (
            "minimum simplicity_adjusted_score "
            "on training cycles only"
        ),
        "production_files_modified": False,
    }

    CONFIG_PATH.write_text(
        json.dumps(
            config,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    VALIDATION_PATH.write_text(
        "\n".join(validation_messages)
        + "\n"
    )

    print("\n".join(validation_messages))

    print()
    print("=" * 120)
    print("LOOCV FOLD RESULTS")
    print("=" * 120)

    fold_display_columns = [
        "holdout_cycle",
        "baseline_label",
        "selected_model_label",
        "selected_generic_ballot_coefficient",
        "observation_count",
        "mean_absolute_error",
        "rmse",
        "brier_score",
        "winner_accuracy",
        "mean_margin_error_dem_bias",
        "expected_win_count_error",
    ]

    print(
        fold_results[fold_display_columns]
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print("=" * 120)
    print("NORMALIZED MINUS RAW BY HOLDOUT CYCLE")
    print("=" * 120)

    summary_display_columns = [
        "holdout_cycle",
        "raw_selected_generic_ballot_coefficient",
        "normalized_selected_generic_ballot_coefficient",
        "raw_mean_absolute_error",
        "normalized_mean_absolute_error",
        "mae_difference_normalized_minus_raw",
        "rmse_difference_normalized_minus_raw",
        "brier_difference_normalized_minus_raw",
        "accuracy_difference_normalized_minus_raw",
        "mae_winner",
    ]

    print(
        summary[summary_display_columns]
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print("=" * 120)
    print("OVERALL HELD-OUT PERFORMANCE")
    print("=" * 120)

    overall_display_columns = [
        "baseline_label",
        "weighted_mean_absolute_error",
        "weighted_rmse",
        "weighted_brier_score",
        "weighted_log_loss",
        "weighted_winner_accuracy",
        "mean_absolute_dem_bias",
        "mean_absolute_expected_win_count_error",
        "overall_validation_rank",
    ]

    print(
        overall[overall_display_columns]
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print("=" * 120)
    print("COEFFICIENT STABILITY")
    print("=" * 120)

    print(
        coefficient_stability.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
