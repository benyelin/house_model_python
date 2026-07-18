#!/usr/bin/env python3
"""Compare raw and nationally normalized House presidential baselines."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKTEST_DIR = Path(__file__).resolve().parent

if str(BACKTEST_DIR) not in sys.path:
    sys.path.insert(0, str(BACKTEST_DIR))

import run_house_environment_coefficient_sweep as sweep  # noqa: E402


LOOKUP_PATH = (
    PROJECT_ROOT
    / "historical"
    / "warehouse"
    / "raw"
    / "national_environment"
    / "house_presidential_baseline_national_margins.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "baseline_environment_bakeoff"
)

BASELINE_TYPES = (
    "raw_presidential_margin",
    "normalized_partisan_baseline",
)

BASELINE_LABELS = {
    "raw_presidential_margin": "Raw presidential margin",
    "normalized_partisan_baseline": "Normalized partisan baseline",
}

BASELINE_COLUMNS = {
    "raw_presidential_margin": "district_pres_margin_dem",
    "normalized_partisan_baseline": "district_partisan_baseline_dem",
}


@dataclass(frozen=True)
class Specification:
    baseline_type: str
    model_family: str
    generic_ballot_coefficient: float

    @property
    def specification_id(self) -> str:
        return (
            f"{self.baseline_type}"
            f"__{self.model_family}"
            f"__gb_{self.generic_ballot_coefficient:.2f}"
        )


def load_inputs() -> pd.DataFrame:
    inputs = sweep.load_inputs().copy()

    if not LOOKUP_PATH.exists():
        raise FileNotFoundError(
            f"Baseline-margin lookup not found: {LOOKUP_PATH}"
        )

    lookup = pd.read_csv(LOOKUP_PATH)

    required_lookup_columns = {
        "presidential_result_year",
        "national_pres_margin_dem",
        "source_status",
    }

    missing_lookup_columns = (
        required_lookup_columns - set(lookup.columns)
    )

    if missing_lookup_columns:
        raise ValueError(
            "Lookup is missing columns: "
            f"{sorted(missing_lookup_columns)}"
        )

    if not lookup["source_status"].eq("verified").all():
        raise ValueError(
            "All lookup rows must be marked verified."
        )

    inputs["presidential_result_year"] = pd.to_numeric(
        inputs["presidential_result_year"],
        errors="raise",
    ).astype(int)

    lookup["presidential_result_year"] = pd.to_numeric(
        lookup["presidential_result_year"],
        errors="raise",
    ).astype(int)

    lookup["national_pres_margin_dem"] = pd.to_numeric(
        lookup["national_pres_margin_dem"],
        errors="raise",
    )

    inputs = inputs.merge(
        lookup[
            [
                "presidential_result_year",
                "national_pres_margin_dem",
            ]
        ],
        on="presidential_result_year",
        how="left",
        validate="many_to_one",
    )

    if inputs["national_pres_margin_dem"].isna().any():
        missing_years = sorted(
            inputs.loc[
                inputs["national_pres_margin_dem"].isna(),
                "presidential_result_year",
            ].unique()
        )

        raise ValueError(
            "Missing national margins for years: "
            f"{missing_years}"
        )

    inputs["district_partisan_baseline_dem"] = (
        inputs["district_pres_margin_dem"]
        - inputs["national_pres_margin_dem"]
    )

    return inputs


def build_specifications() -> list[Specification]:
    return [
        Specification(
            baseline_type=baseline_type,
            model_family=model_family,
            generic_ballot_coefficient=coefficient,
        )
        for baseline_type in BASELINE_TYPES
        for model_family in sweep.MODEL_FAMILIES
        for coefficient in sweep.GENERIC_BALLOT_COEFFICIENTS
    ]


def calculate_forecast(
    inputs: pd.DataFrame,
    specification: Specification,
) -> pd.DataFrame:
    sweep_specification = sweep.Specification(
        model_family=specification.model_family,
        generic_ballot_coefficient=(
            specification.generic_ballot_coefficient
        ),
    )

    output = sweep.calculate_environment(
        inputs,
        sweep_specification,
    )

    baseline_column = BASELINE_COLUMNS[
        specification.baseline_type
    ]

    output["selected_baseline_margin_dem"] = (
        output[baseline_column]
    )

    output["forecast_margin_dem"] = (
        output["selected_baseline_margin_dem"]
        + output["national_environment_margin_dem"]
    )

    output["dem_win_probability"] = (
        sweep.logistic_probability(
            output["forecast_margin_dem"],
            error_sd=sweep.PROBABILITY_SCALE,
        )
    )

    output["baseline_type"] = specification.baseline_type
    output["baseline_label"] = BASELINE_LABELS[
        specification.baseline_type
    ]
    output["specification_id"] = (
        specification.specification_id
    )

    return output


def metrics_row(
    frame: pd.DataFrame,
    specification: Specification,
    cycle: int | None,
) -> dict[str, object]:
    metrics = sweep.score_forecasts(
        actual_margin_dem=frame["actual_dem_margin"],
        forecast_margin_dem=frame["forecast_margin_dem"],
        probability_dem=frame["dem_win_probability"],
    )

    return {
        "specification_id": specification.specification_id,
        "baseline_type": specification.baseline_type,
        "baseline_label": BASELINE_LABELS[
            specification.baseline_type
        ],
        "model_family": specification.model_family,
        "model_label": sweep.MODEL_LABELS[
            specification.model_family
        ],
        "generic_ballot_coefficient": (
            specification.generic_ballot_coefficient
        ),
        "approval_coefficient": (
            sweep.APPROVAL_COEFFICIENT
            if specification.model_family
            in {
                "generic_plus_approval",
                "generic_plus_approval_plus_midterm",
            }
            else 0.0
        ),
        "midterm_coefficient": (
            sweep.MIDTERM_COEFFICIENT
            if specification.model_family
            == "generic_plus_approval_plus_midterm"
            else 0.0
        ),
        "complexity_score": sweep.COMPLEXITY_SCORES[
            specification.model_family
        ],
        "cycle": cycle,
        **metrics.as_dict(),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs()
    specifications = build_specifications()

    summary_rows = []
    by_cycle_rows = []
    prediction_frames = []

    for specification in specifications:
        calculated = calculate_forecast(
            inputs,
            specification,
        )

        summary_rows.append(
            metrics_row(
                calculated,
                specification,
                cycle=None,
            )
        )

        for cycle in sweep.EXPECTED_CYCLES:
            cycle_frame = calculated.loc[
                calculated["cycle"].eq(cycle)
            ]

            by_cycle_rows.append(
                metrics_row(
                    cycle_frame,
                    specification,
                    cycle=cycle,
                )
            )

        prediction_frames.append(
            calculated[
                [
                    "specification_id",
                    "baseline_type",
                    "baseline_label",
                    "model_family",
                    "model_label",
                    "generic_ballot_coefficient",
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

    summary = sweep.add_ranking_columns(
        pd.DataFrame(summary_rows)
    )

    by_cycle = pd.DataFrame(by_cycle_rows)

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    expected_specifications = (
        len(BASELINE_TYPES)
        * len(sweep.MODEL_FAMILIES)
        * len(sweep.GENERIC_BALLOT_COEFFICIENTS)
    )

    if len(summary) != expected_specifications:
        raise ValueError(
            f"Expected {expected_specifications} summary rows; "
            f"found {len(summary)}."
        )

    expected_predictions = (
        len(inputs) * expected_specifications
    )

    if len(predictions) != expected_predictions:
        raise ValueError(
            f"Expected {expected_predictions} predictions; "
            f"found {len(predictions)}."
        )

    pareto = sweep.build_pareto_table(summary)

    best_by_baseline = (
        summary.sort_values(
            [
                "simplicity_adjusted_score",
                "performance_score",
                "mean_absolute_error",
                "rmse",
                "brier_score",
            ]
        )
        .groupby(
            "baseline_type",
            as_index=False,
            sort=False,
        )
        .head(1)
        .reset_index(drop=True)
    )

    summary.to_csv(
        OUTPUT_DIR
        / "house_baseline_environment_bakeoff_summary.csv",
        index=False,
    )

    by_cycle.to_csv(
        OUTPUT_DIR
        / "house_baseline_environment_bakeoff_by_cycle.csv",
        index=False,
    )

    predictions.to_csv(
        OUTPUT_DIR
        / "house_baseline_environment_bakeoff_predictions.csv",
        index=False,
    )

    pareto.to_csv(
        OUTPUT_DIR
        / "house_baseline_environment_bakeoff_pareto.csv",
        index=False,
    )

    best_by_baseline.to_csv(
        OUTPUT_DIR
        / "house_baseline_environment_bakeoff_best_by_baseline.csv",
        index=False,
    )

    print("House baseline-environment bakeoff")
    print("=" * 90)
    print(f"Eligible observations: {len(inputs):,}")
    print(f"Specifications: {len(summary):,}")
    print(f"Prediction rows: {len(predictions):,}")
    print()
    print("Best specification by baseline")
    print("=" * 90)

    display_columns = [
        "baseline_label",
        "model_label",
        "generic_ballot_coefficient",
        "approval_coefficient",
        "midterm_coefficient",
        "mean_absolute_error",
        "rmse",
        "brier_score",
        "winner_accuracy",
        "mean_margin_error_dem_bias",
        "expected_win_count_error",
        "simplicity_adjusted_score",
    ]

    print(
        best_by_baseline[display_columns]
        .to_string(index=False)
    )

    print()
    print("Validation: PASSED")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
