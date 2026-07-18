#!/usr/bin/env python3
"""
Robustness sweep for the normalized House baseline coefficient.

The architecture is fixed as:

    district_partisan_baseline_dem
        = district_pres_margin_dem
        - national_pres_margin_dem

    forecast_margin_dem
        = district_partisan_baseline_dem
        + coefficient * generic_ballot_margin_dem

Approval and midterm adjustments are excluded.

The script evaluates a narrow coefficient range, reports aggregate and
cycle-level performance, and identifies coefficients lying on a broad
near-optimal performance plateau.

No production files are modified.
"""

from __future__ import annotations

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
    / "normalized_baseline_coefficient_robustness"
)

OVERALL_PATH = (
    OUTPUT_DIR
    / "house_normalized_baseline_coefficient_robustness_overall.csv"
)

BY_CYCLE_PATH = (
    OUTPUT_DIR
    / "house_normalized_baseline_coefficient_robustness_by_cycle.csv"
)

PLATEAU_PATH = (
    OUTPUT_DIR
    / "house_normalized_baseline_coefficient_robustness_plateau.csv"
)

RECOMMENDATION_PATH = (
    OUTPUT_DIR
    / "house_normalized_baseline_coefficient_recommendation.csv"
)

VALIDATION_PATH = (
    OUTPUT_DIR
    / "house_normalized_baseline_coefficient_robustness_validation.txt"
)


COEFFICIENTS = tuple(
    round(value, 2)
    for value in np.arange(
        0.50,
        1.001,
        0.05,
    )
)

BASELINE_TYPE = "normalized_partisan_baseline"

MODEL_FAMILY = next(
    model_family
    for model_family, model_label in sweep.MODEL_LABELS.items()
    if model_label == "Generic ballot only"
)

# A coefficient is considered to be on the near-optimal plateau when:
#
# - MAE is within 0.10 points of the minimum
# - RMSE is within 0.10 points of the minimum
# - Brier score is within 0.0005 of the minimum
#
# These are deliberately narrow but not exact-minimum thresholds.
MAE_PLATEAU_TOLERANCE = 0.10
RMSE_PLATEAU_TOLERANCE = 0.10
BRIER_PLATEAU_TOLERANCE = 0.0005


class ValidationError(RuntimeError):
    """Raised when the robustness sweep violates its contract."""


def build_specification(
    coefficient: float,
) -> bakeoff.Specification:
    return bakeoff.Specification(
        baseline_type=BASELINE_TYPE,
        model_family=MODEL_FAMILY,
        generic_ballot_coefficient=coefficient,
    )


def score_frame(
    frame: pd.DataFrame,
    coefficient: float,
    *,
    cycle: int | None,
) -> dict[str, object]:
    specification = build_specification(coefficient)

    calculated = bakeoff.calculate_forecast(
        frame,
        specification,
    )

    metrics = sweep.score_forecasts(
        actual_margin_dem=calculated["actual_dem_margin"],
        forecast_margin_dem=calculated["forecast_margin_dem"],
        probability_dem=calculated["dem_win_probability"],
    )

    return {
        "generic_ballot_coefficient": coefficient,
        "cycle": cycle,
        "observation_count": int(len(calculated)),
        **metrics.as_dict(),
    }


def add_diagnostics(
    overall: pd.DataFrame,
) -> pd.DataFrame:
    output = overall.copy()

    metric_directions = {
        "mean_absolute_error": True,
        "rmse": True,
        "brier_score": True,
        "log_loss": True,
        "winner_accuracy": False,
        "absolute_dem_bias": True,
        "absolute_expected_win_count_error": True,
    }

    output["absolute_dem_bias"] = (
        output["mean_margin_error_dem_bias"].abs()
    )

    output["absolute_expected_win_count_error"] = (
        output["expected_win_count_error"].abs()
    )

    for metric, ascending in metric_directions.items():
        output[f"{metric}_rank"] = (
            output[metric]
            .rank(
                method="min",
                ascending=ascending,
            )
            .astype(int)
        )

    minimum_mae = float(
        output["mean_absolute_error"].min()
    )

    minimum_rmse = float(
        output["rmse"].min()
    )

    minimum_brier = float(
        output["brier_score"].min()
    )

    output["mae_degradation_from_best"] = (
        output["mean_absolute_error"]
        - minimum_mae
    )

    output["rmse_degradation_from_best"] = (
        output["rmse"]
        - minimum_rmse
    )

    output["brier_degradation_from_best"] = (
        output["brier_score"]
        - minimum_brier
    )

    output["on_mae_plateau"] = (
        output["mae_degradation_from_best"]
        <= MAE_PLATEAU_TOLERANCE
    )

    output["on_rmse_plateau"] = (
        output["rmse_degradation_from_best"]
        <= RMSE_PLATEAU_TOLERANCE
    )

    output["on_brier_plateau"] = (
        output["brier_degradation_from_best"]
        <= BRIER_PLATEAU_TOLERANCE
    )

    output["on_joint_performance_plateau"] = (
        output["on_mae_plateau"]
        & output["on_rmse_plateau"]
        & output["on_brier_plateau"]
    )

    output["primary_performance_score"] = (
        output[
            [
                "mean_absolute_error_rank",
                "rmse_rank",
                "brier_score_rank",
            ]
        ]
        .mean(axis=1)
    )

    return output.sort_values(
        [
            "primary_performance_score",
            "mean_absolute_error",
            "rmse",
            "brier_score",
            "generic_ballot_coefficient",
        ]
    ).reset_index(drop=True)


def build_cycle_stability(
    by_cycle: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for coefficient in COEFFICIENTS:
        subset = by_cycle.loc[
            by_cycle["generic_ballot_coefficient"].eq(
                coefficient
            )
        ].copy()

        rows.append(
            {
                "generic_ballot_coefficient": coefficient,
                "cycles_evaluated": int(
                    subset["cycle"].nunique()
                ),
                "mean_cycle_mae": float(
                    subset["mean_absolute_error"].mean()
                ),
                "maximum_cycle_mae": float(
                    subset["mean_absolute_error"].max()
                ),
                "mae_standard_deviation_across_cycles": float(
                    subset["mean_absolute_error"].std()
                ),
                "mean_cycle_rmse": float(
                    subset["rmse"].mean()
                ),
                "maximum_cycle_rmse": float(
                    subset["rmse"].max()
                ),
                "mean_cycle_brier": float(
                    subset["brier_score"].mean()
                ),
                "minimum_cycle_winner_accuracy": float(
                    subset["winner_accuracy"].min()
                ),
                "mean_absolute_cycle_bias": float(
                    subset[
                        "mean_margin_error_dem_bias"
                    ].abs().mean()
                ),
                "maximum_absolute_cycle_bias": float(
                    subset[
                        "mean_margin_error_dem_bias"
                    ].abs().max()
                ),
                "mean_absolute_cycle_seat_error": float(
                    subset[
                        "expected_win_count_error"
                    ].abs().mean()
                ),
                "maximum_absolute_cycle_seat_error": float(
                    subset[
                        "expected_win_count_error"
                    ].abs().max()
                ),
            }
        )

    return pd.DataFrame(rows)


def select_recommendation(
    overall: pd.DataFrame,
    stability: pd.DataFrame,
) -> pd.DataFrame:
    combined = overall.merge(
        stability,
        on="generic_ballot_coefficient",
        how="left",
        validate="one_to_one",
    )

    plateau = combined.loc[
        combined["on_joint_performance_plateau"]
    ].copy()

    if plateau.empty:
        plateau = combined.copy()

    # Within the near-optimal plateau, prefer:
    #
    # 1. Lower average absolute cycle bias
    # 2. Lower average absolute cycle seat error
    # 3. Lower maximum cycle MAE
    # 4. Better aggregate MAE
    # 5. Coefficient closer to the LOOCV median of 0.70
    plateau["distance_from_loocv_median"] = (
        plateau["generic_ballot_coefficient"]
        - 0.70
    ).abs()

    recommendation = (
        plateau.sort_values(
            [
                "mean_absolute_cycle_bias",
                "mean_absolute_cycle_seat_error",
                "maximum_cycle_mae",
                "mean_absolute_error",
                "distance_from_loocv_median",
                "generic_ballot_coefficient",
            ]
        )
        .head(1)
        .reset_index(drop=True)
    )

    return recommendation


def validate_outputs(
    inputs: pd.DataFrame,
    overall: pd.DataFrame,
    by_cycle: pd.DataFrame,
) -> list[str]:
    expected_overall_rows = len(COEFFICIENTS)

    expected_cycle_rows = (
        len(COEFFICIENTS)
        * len(sweep.EXPECTED_CYCLES)
    )

    if len(overall) != expected_overall_rows:
        raise ValidationError(
            f"Expected {expected_overall_rows} overall rows; "
            f"found {len(overall)}."
        )

    if len(by_cycle) != expected_cycle_rows:
        raise ValidationError(
            f"Expected {expected_cycle_rows} cycle rows; "
            f"found {len(by_cycle)}."
        )

    observed_coefficients = tuple(
        sorted(
            overall[
                "generic_ballot_coefficient"
            ].round(2)
        )
    )

    if observed_coefficients != COEFFICIENTS:
        raise ValidationError(
            "Observed coefficient grid does not match "
            "the configured coefficient grid."
        )

    observed_cycles = tuple(
        sorted(by_cycle["cycle"].unique())
    )

    if observed_cycles != sweep.EXPECTED_CYCLES:
        raise ValidationError(
            f"Expected cycles {sweep.EXPECTED_CYCLES}; "
            f"found {observed_cycles}."
        )

    expected_observations = int(len(inputs))

    if not overall[
        "observation_count"
    ].eq(expected_observations).all():
        raise ValidationError(
            "At least one coefficient was not evaluated "
            "on the complete eligible dataset."
        )

    return [
        "Normalized baseline coefficient robustness validation: PASSED",
        f"Eligible observations: {expected_observations:,}",
        f"Coefficients evaluated: {len(COEFFICIENTS)}",
        (
            "Coefficient range: "
            f"{min(COEFFICIENTS):.2f} to "
            f"{max(COEFFICIENTS):.2f}"
        ),
        (
            f"Coefficient increment: "
            f"{COEFFICIENTS[1] - COEFFICIENTS[0]:.2f}"
        ),
        f"Cycle-level rows: {len(by_cycle):,}",
        "Architecture fixed to normalized baseline.",
        "Environment fixed to generic ballot only.",
        "Production files modified: False",
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    inputs = bakeoff.load_inputs()

    overall_rows: list[dict[str, object]] = []
    by_cycle_rows: list[dict[str, object]] = []

    for coefficient in COEFFICIENTS:
        overall_rows.append(
            score_frame(
                inputs,
                coefficient,
                cycle=None,
            )
        )

        for cycle in sweep.EXPECTED_CYCLES:
            cycle_frame = inputs.loc[
                inputs["cycle"].eq(cycle)
            ].copy()

            by_cycle_rows.append(
                score_frame(
                    cycle_frame,
                    coefficient,
                    cycle=cycle,
                )
            )

    overall = add_diagnostics(
        pd.DataFrame(overall_rows)
    )

    by_cycle = (
        pd.DataFrame(by_cycle_rows)
        .sort_values(
            [
                "generic_ballot_coefficient",
                "cycle",
            ]
        )
        .reset_index(drop=True)
    )

    stability = build_cycle_stability(by_cycle)

    plateau = (
        overall.loc[
            overall["on_joint_performance_plateau"]
        ]
        .sort_values(
            "generic_ballot_coefficient"
        )
        .reset_index(drop=True)
    )

    recommendation = select_recommendation(
        overall,
        stability,
    )

    validation_messages = validate_outputs(
        inputs,
        overall,
        by_cycle,
    )

    overall.to_csv(
        OVERALL_PATH,
        index=False,
    )

    by_cycle.to_csv(
        BY_CYCLE_PATH,
        index=False,
    )

    plateau.to_csv(
        PLATEAU_PATH,
        index=False,
    )

    recommendation.to_csv(
        RECOMMENDATION_PATH,
        index=False,
    )

    VALIDATION_PATH.write_text(
        "\n".join(validation_messages)
        + "\n"
    )

    print("\n".join(validation_messages))

    print()
    print("=" * 125)
    print("OVERALL COEFFICIENT ROBUSTNESS")
    print("=" * 125)

    overall_columns = [
        "generic_ballot_coefficient",
        "mean_absolute_error",
        "mae_degradation_from_best",
        "rmse",
        "rmse_degradation_from_best",
        "brier_score",
        "brier_degradation_from_best",
        "winner_accuracy",
        "mean_margin_error_dem_bias",
        "expected_win_count_error",
        "on_joint_performance_plateau",
    ]

    print(
        overall[
            overall_columns
        ]
        .sort_values(
            "generic_ballot_coefficient"
        )
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print("=" * 125)
    print("JOINT NEAR-OPTIMAL PLATEAU")
    print("=" * 125)

    if plateau.empty:
        print(
            "No coefficient satisfied all configured "
            "plateau thresholds."
        )
    else:
        print(
            plateau[
                overall_columns
            ]
            .to_string(
                index=False,
                float_format=lambda value: f"{value:.6f}",
            )
        )

    print()
    print("=" * 125)
    print("RECOMMENDED PRODUCTION COEFFICIENT")
    print("=" * 125)

    recommendation_columns = [
        "generic_ballot_coefficient",
        "mean_absolute_error",
        "rmse",
        "brier_score",
        "winner_accuracy",
        "mean_margin_error_dem_bias",
        "expected_win_count_error",
        "mean_absolute_cycle_bias",
        "mean_absolute_cycle_seat_error",
        "maximum_cycle_mae",
        "distance_from_loocv_median",
    ]

    print(
        recommendation[
            recommendation_columns
        ]
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
