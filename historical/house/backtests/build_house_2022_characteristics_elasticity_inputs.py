from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_BACKTEST_INPUT_PATH = (
    PROJECT_ROOT
    / "historical/house/backtests/outputs/"
    "house_2022_backtest_inputs_provisional.csv"
)

DEFAULT_PREDICTION_PATH = (
    PROJECT_ROOT
    / "historical/house/elasticity/"
    "house_characteristics_predicted_elasticity.csv"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "historical/house/backtests/outputs/"
    "house_2022_backtest_inputs_characteristics_elasticity.csv"
)

DEFAULT_VALIDATION_PATH = (
    PROJECT_ROOT
    / "historical/house/backtests/outputs/"
    "house_2022_backtest_inputs_characteristics_elasticity_validation.txt"
)


def build_inputs(
    races: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    required_race_columns = {
        "race_id",
        "cycle",
        "district_pres_margin_dem",
        "actual_dem_margin",
        "general_election_party_structure",
    }

    missing_race_columns = sorted(
        required_race_columns - set(races.columns)
    )

    if missing_race_columns:
        raise ValueError(
            "Backtest input is missing required columns: "
            + ", ".join(missing_race_columns)
        )

    required_prediction_columns = {
        "race_id",
        "characteristics_elasticity",
        "characteristics_elasticity_raw_prediction",
        "characteristics_elasticity_model_alpha",
        "characteristics_model_training_match",
        "characteristics_elasticity_method",
        "characteristics_elasticity_limitations",
    }

    missing_prediction_columns = sorted(
        required_prediction_columns - set(predictions.columns)
    )

    if missing_prediction_columns:
        raise ValueError(
            "Prediction table is missing required columns: "
            + ", ".join(missing_prediction_columns)
        )

    if len(races) != 435:
        raise ValueError(
            f"Expected 435 backtest races; found {len(races)}."
        )

    if len(predictions) != 435:
        raise ValueError(
            f"Expected 435 elasticity predictions; "
            f"found {len(predictions)}."
        )

    if races["race_id"].duplicated().any():
        raise ValueError(
            "Duplicate race IDs found in backtest input."
        )

    if predictions["race_id"].duplicated().any():
        raise ValueError(
            "Duplicate race IDs found in elasticity predictions."
        )

    prediction_keep = predictions[
        [
            "race_id",
            "characteristics_elasticity",
            "characteristics_elasticity_raw_prediction",
            "characteristics_elasticity_model_alpha",
            "characteristics_model_training_match",
            "characteristics_elasticity_method",
            "characteristics_elasticity_limitations",
        ]
    ].copy()

    combined = races.merge(
        prediction_keep,
        on="race_id",
        how="left",
        validate="one_to_one",
    )

    numeric_columns = [
        "characteristics_elasticity",
        "characteristics_elasticity_raw_prediction",
        "characteristics_elasticity_model_alpha",
    ]

    for column in numeric_columns:
        combined[column] = pd.to_numeric(
            combined[column],
            errors="coerce",
        )

    combined["district_elasticity"] = (
        combined["characteristics_elasticity"]
    )

    combined["elasticity_source"] = (
        "district characteristics ridge model"
    )

    combined["elasticity_match_status"] = np.where(
        combined["characteristics_model_training_match"].eq(True),
        "characteristics_prediction_with_historical_label_target",
        "characteristics_prediction_without_historical_label_target",
    )

    missing_predictions = int(
        combined["characteristics_elasticity"].isna().sum()
    )

    nonfinite_predictions = int(
        (
            combined["characteristics_elasticity"].notna()
            & ~np.isfinite(
                combined["characteristics_elasticity"]
            )
        ).sum()
    )

    duplicate_races = int(
        combined["race_id"].duplicated().sum()
    )

    failures: list[str] = []

    if len(combined) != 435:
        failures.append(
            f"Expected 435 merged rows; found {len(combined)}."
        )

    if duplicate_races:
        failures.append(
            f"Found {duplicate_races} duplicate race IDs."
        )

    if missing_predictions:
        failures.append(
            f"Found {missing_predictions} missing elasticity predictions."
        )

    if nonfinite_predictions:
        failures.append(
            f"Found {nonfinite_predictions} nonfinite predictions."
        )

    prediction_summary = (
        combined["characteristics_elasticity"]
        .describe()
    )

    match_counts = (
        combined["elasticity_match_status"]
        .value_counts(dropna=False)
    )

    report_lines = [
        "2022 House Characteristics Elasticity Input Validation",
        "=" * 54,
        "",
        f"Rows: {len(combined)}",
        f"Unique race IDs: {combined['race_id'].nunique()}",
        f"Duplicate race IDs: {duplicate_races}",
        f"Missing predictions: {missing_predictions}",
        f"Nonfinite predictions: {nonfinite_predictions}",
        "",
        "Match-status counts:",
        match_counts.to_string(),
        "",
        "Characteristics elasticity summary:",
        prediction_summary.to_string(
            float_format=lambda value: f"{value:.6f}"
        ),
        "",
        "Model alpha values:",
        combined[
            "characteristics_elasticity_model_alpha"
        ].value_counts(dropna=False).to_string(),
        "",
        "Important limitation:",
        (
            "The characteristics model was trained against historical "
            "elasticity targets based on 2012-2020 district-label "
            "histories. This input is for exploratory 2022 backtesting."
        ),
        "",
        "Validation status:",
    ]

    if failures:
        report_lines.append("FAILED")
        report_lines.extend(
            f"- {failure}"
            for failure in failures
        )
    else:
        report_lines.append("PASSED")

    report = "\n".join(report_lines)

    if failures:
        raise RuntimeError(report)

    return combined, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge characteristics-based elasticity predictions into "
            "the 2022 House backtest input."
        )
    )

    parser.add_argument(
        "--backtest-input-path",
        type=Path,
        default=DEFAULT_BACKTEST_INPUT_PATH,
    )

    parser.add_argument(
        "--prediction-path",
        type=Path,
        default=DEFAULT_PREDICTION_PATH,
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )

    parser.add_argument(
        "--validation-path",
        type=Path,
        default=DEFAULT_VALIDATION_PATH,
    )

    args = parser.parse_args()

    if not args.backtest_input_path.exists():
        raise FileNotFoundError(
            f"Missing backtest input: {args.backtest_input_path}"
        )

    if not args.prediction_path.exists():
        raise FileNotFoundError(
            f"Missing elasticity predictions: {args.prediction_path}"
        )

    races = pd.read_csv(
        args.backtest_input_path,
        dtype={"race_id": str},
    )

    predictions = pd.read_csv(
        args.prediction_path,
        dtype={"race_id": str},
    )

    combined, report = build_inputs(
        races=races,
        predictions=predictions,
    )

    args.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.validation_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        args.output_path,
        index=False,
    )

    args.validation_path.write_text(report)

    print(report)
    print()
    print(f"Wrote: {args.output_path}")
    print(f"Wrote: {args.validation_path}")


if __name__ == "__main__":
    main()
