from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from house_backtest_components import (
    BacktestParameters,
    calculate_forecast,
)
from run_house_layered_backtest import (
    build_calibration_table,
    logistic_probability,
    score_layer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "historical/house/backtests/outputs/"
    "house_2022_backtest_inputs_characteristics_elasticity.csv"
)

DEFAULT_ENVIRONMENT_PATH = (
    PROJECT_ROOT
    / "historical/warehouse/processed/national_environment/"
    "house_2022_election_day_national_environment.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical/house/backtests/outputs/layered"
)

DEFAULT_RETENTION_LEVELS = (
    0.00,
    0.20,
    0.30,
    0.40,
    1.00,
)

COMPETITIVE_THRESHOLDS = (
    5.0,
    10.0,
    15.0,
)


def parse_retention_levels(value: str) -> tuple[float, ...]:
    levels: list[float] = []

    for item in value.split(","):
        item = item.strip()

        if not item:
            continue

        try:
            level = float(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid retention level: {item!r}"
            ) from exc

        if not 0.0 <= level <= 1.0:
            raise argparse.ArgumentTypeError(
                "Retention levels must be between 0.0 and 1.0."
            )

        levels.append(round(level, 6))

    if not levels:
        raise argparse.ArgumentTypeError(
            "At least one retention level is required."
        )

    return tuple(sorted(set(levels)))


def build_scoring_mask(races: pd.DataFrame) -> pd.Series:
    return (
        races["general_election_party_structure"]
        .fillna("")
        .eq("D_vs_R")
        & races["district_pres_margin_dem"].notna()
        & races["actual_dem_margin"].notna()
    )


def construct_elasticity(
    races: pd.DataFrame,
    retention_level: float,
    scoring_mask: pd.Series,
) -> pd.Series:
    predicted = pd.to_numeric(
        races["characteristics_elasticity"],
        errors="coerce",
    )

    if predicted.isna().any():
        raise ValueError(
            "Missing characteristics elasticity predictions."
        )

    retained = (
        1.0
        + retention_level
        * (predicted - 1.0)
    )

    scored_mean = float(
        retained.loc[scoring_mask].mean()
    )

    if not np.isfinite(scored_mean) or scored_mean == 0:
        raise RuntimeError(
            "Scored-sample elasticity mean is invalid."
        )

    return retained / scored_mean


def score_subset(
    scored: pd.DataFrame,
    retention_level: float,
    subset_name: str,
) -> dict[str, object]:
    result = score_layer(
        scored_races=scored,
        model_name=(
            f"characteristics_retention_{retention_level:.2f}"
        ),
        forecast_margin_column="layer_3_margin_dem",
        probability_column="layer_3_dem_probability",
    )

    result["retention_level"] = retention_level
    result["subset_name"] = subset_name

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose characteristics-based House elasticity "
            "performance in competitive races and calibration buckets."
        )
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
    )

    parser.add_argument(
        "--environment-path",
        type=Path,
        default=DEFAULT_ENVIRONMENT_PATH,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--retention-levels",
        type=parse_retention_levels,
        default=DEFAULT_RETENTION_LEVELS,
    )

    parser.add_argument(
        "--incumbency-bonus",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--error-sd",
        type=float,
        default=6.5,
    )

    args = parser.parse_args()

    if not args.input_path.exists():
        raise FileNotFoundError(
            f"Missing input: {args.input_path}"
        )

    if not args.environment_path.exists():
        raise FileNotFoundError(
            f"Missing environment input: {args.environment_path}"
        )

    races = pd.read_csv(
        args.input_path,
        dtype={"race_id": str},
    )

    environment = pd.read_csv(
        args.environment_path,
    )

    if len(races) != 435:
        raise ValueError(
            f"Expected 435 races; found {len(races)}."
        )

    if races["race_id"].duplicated().any():
        raise ValueError(
            "Duplicate race IDs found."
        )

    if len(environment) != 1:
        raise ValueError(
            "Environment input must contain exactly one row."
        )

    required_columns = {
        "race_id",
        "district_pres_margin_dem",
        "actual_dem_margin",
        "actual_winner",
        "general_election_party_structure",
        "dem_is_incumbent",
        "gop_is_incumbent",
        "characteristics_elasticity",
    }

    missing = sorted(
        required_columns - set(races.columns)
    )

    if missing:
        raise ValueError(
            "Input is missing required columns: "
            + ", ".join(missing)
        )

    national_environment = pd.to_numeric(
        environment.iloc[0][
            "national_environment_margin_dem"
        ],
        errors="coerce",
    )

    if pd.isna(national_environment):
        raise ValueError(
            "National environment is blank or nonnumeric."
        )

    races["district_pres_margin_dem"] = pd.to_numeric(
        races["district_pres_margin_dem"],
        errors="coerce",
    )

    races["actual_dem_margin"] = pd.to_numeric(
        races["actual_dem_margin"],
        errors="coerce",
    )

    races["national_environment_margin_dem"] = float(
        national_environment
    )

    scoring_mask = build_scoring_mask(races)

    if not scoring_mask.any():
        raise RuntimeError(
            "No races are eligible for scoring."
        )

    parameters = BacktestParameters(
        national_environment_margin_dem=float(
            national_environment
        ),
        incumbency_bonus=args.incumbency_bonus,
        elasticity_default=1.0,
    )

    races[
        "layer_2_margin_dem"
    ], _ = calculate_forecast(
        df=races,
        component_names=[
            "presidential_baseline",
            "national_environment",
            "incumbency",
        ],
        parameters=parameters,
    )

    races["layer_2_dem_probability"] = (
        logistic_probability(
            races["layer_2_margin_dem"],
            args.error_sd,
        )
    )

    races["layer_2_predicted_dem_win"] = (
        races["layer_2_dem_probability"] >= 0.5
    )

    summary_rows: list[dict[str, object]] = []
    calibration_frames: list[pd.DataFrame] = []
    changed_call_frames: list[pd.DataFrame] = []
    district_frames: list[pd.DataFrame] = []

    for retention in args.retention_levels:
        work = races.copy()

        work["district_elasticity"] = construct_elasticity(
            races=work,
            retention_level=retention,
            scoring_mask=scoring_mask,
        )

        work[
            "layer_3_margin_dem"
        ], _ = calculate_forecast(
            df=work,
            component_names=[
                "presidential_baseline",
                "elasticity_environment",
                "incumbency",
            ],
            parameters=parameters,
        )

        work["layer_3_dem_probability"] = (
            logistic_probability(
                work["layer_3_margin_dem"],
                args.error_sd,
            )
        )

        work["layer_3_predicted_dem_win"] = (
            work["layer_3_dem_probability"] >= 0.5
        )

        work["actual_dem_win"] = (
            work["actual_dem_margin"] > 0
        )

        work["winner_call_changed"] = (
            work["layer_3_predicted_dem_win"]
            != work["layer_2_predicted_dem_win"]
        )

        work["layer_2_correct_winner"] = (
            work["layer_2_predicted_dem_win"]
            == work["actual_dem_win"]
        )

        work["layer_3_correct_winner"] = (
            work["layer_3_predicted_dem_win"]
            == work["actual_dem_win"]
        )

        work["probability_change_dem"] = (
            work["layer_3_dem_probability"]
            - work["layer_2_dem_probability"]
        )

        work["margin_change_dem"] = (
            work["layer_3_margin_dem"]
            - work["layer_2_margin_dem"]
        )

        scored = work.loc[
            scoring_mask
        ].copy()

        summary_rows.append(
            score_subset(
                scored=scored,
                retention_level=retention,
                subset_name="all_scored",
            )
        )

        for threshold in COMPETITIVE_THRESHOLDS:
            subset = scored.loc[
                scored["layer_2_margin_dem"]
                .abs()
                .le(threshold)
            ].copy()

            if subset.empty:
                continue

            summary_rows.append(
                score_subset(
                    scored=subset,
                    retention_level=retention,
                    subset_name=(
                        f"layer_2_margin_within_{int(threshold)}"
                    ),
                )
            )

        calibration = build_calibration_table(
            scored_races=scored,
            model_name=(
                f"characteristics_retention_{retention:.2f}"
            ),
            probability_column="layer_3_dem_probability",
        )

        calibration["retention_level"] = retention

        calibration_frames.append(calibration)

        changed = scored.loc[
            scored["winner_call_changed"]
        ].copy()

        if not changed.empty:
            changed["retention_level"] = retention

            changed_call_frames.append(
                changed[
                    [
                        "retention_level",
                        "race_id",
                        "district_pres_margin_dem",
                        "actual_dem_margin",
                        "actual_winner",
                        "layer_2_margin_dem",
                        "layer_3_margin_dem",
                        "layer_2_dem_probability",
                        "layer_3_dem_probability",
                        "layer_2_predicted_dem_win",
                        "layer_3_predicted_dem_win",
                        "layer_2_correct_winner",
                        "layer_3_correct_winner",
                        "characteristics_elasticity",
                        "district_elasticity",
                    ]
                ]
            )

        district_detail = scored[
            [
                "race_id",
                "actual_dem_margin",
                "actual_winner",
                "layer_2_margin_dem",
                "layer_3_margin_dem",
                "layer_2_dem_probability",
                "layer_3_dem_probability",
                "probability_change_dem",
                "margin_change_dem",
                "layer_2_correct_winner",
                "layer_3_correct_winner",
                "winner_call_changed",
                "characteristics_elasticity",
                "district_elasticity",
            ]
        ].copy()

        district_detail.insert(
            0,
            "retention_level",
            retention,
        )

        district_frames.append(district_detail)

    summary = pd.DataFrame(summary_rows)

    calibration = pd.concat(
        calibration_frames,
        ignore_index=True,
    )

    if changed_call_frames:
        changed_calls = pd.concat(
            changed_call_frames,
            ignore_index=True,
        )
    else:
        changed_calls = pd.DataFrame(
            columns=[
                "retention_level",
                "race_id",
            ]
        )

    district_detail = pd.concat(
        district_frames,
        ignore_index=True,
    )

    control = summary.loc[
        np.isclose(
            summary["retention_level"],
            0.0,
        )
        & summary["subset_name"].eq("all_scored")
    ]

    failures: list[str] = []

    if control.empty:
        failures.append(
            "Missing zero-retention all-scored control."
        )

    all_scored = summary.loc[
        summary["subset_name"].eq("all_scored")
    ].copy()

    all_scored = all_scored.sort_values(
        "retention_level"
    )

    comparison_columns = [
        "retention_level",
        "scored_races",
        "mean_absolute_error",
        "rmse",
        "winner_accuracy",
        "brier_score",
        "log_loss",
        "predicted_dem_wins_in_scored_sample",
        "expected_dem_wins_in_scored_sample",
        "predicted_win_count_error",
        "expected_win_count_error",
    ]

    competitive_pivot = summary.pivot(
        index="retention_level",
        columns="subset_name",
        values="mean_absolute_error",
    )

    changed_summary = (
        changed_calls.groupby("retention_level")
        .agg(
            changed_calls=("race_id", "size"),
            changed_from_correct_to_wrong=(
                "layer_2_correct_winner",
                lambda values: int(values.sum()),
            ),
            changed_to_correct=(
                "layer_3_correct_winner",
                lambda values: int(values.sum()),
            ),
        )
        if not changed_calls.empty
        else pd.DataFrame()
    )

    report_lines = [
        "2022 House Characteristics Elasticity Diagnostics",
        "=" * 49,
        "",
        f"Scored races: {int(scoring_mask.sum())}",
        (
            "Retention levels: "
            + ", ".join(
                f"{value:.2f}"
                for value in args.retention_levels
            )
        ),
        "",
        "All-scored performance:",
        all_scored[
            comparison_columns
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        ),
        "",
        "MAE by competitive subset:",
        competitive_pivot.to_string(
            float_format=lambda value: f"{value:.6f}",
        ),
        "",
        "Changed winner-call summary:",
        (
            changed_summary.to_string()
            if not changed_summary.empty
            else "No winner calls changed."
        ),
        "",
        "Interpretation note:",
        (
            "Competitive subsets are defined using the fixed Layer 2 "
            "forecast margin, preventing the tested Layer 3 setting "
            "from changing which races enter each subset."
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

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        args.output_dir
        / "house_2022_characteristics_elasticity_diagnostic_summary.csv"
    )

    calibration_path = (
        args.output_dir
        / "house_2022_characteristics_elasticity_diagnostic_calibration.csv"
    )

    changed_calls_path = (
        args.output_dir
        / "house_2022_characteristics_elasticity_changed_calls.csv"
    )

    district_detail_path = (
        args.output_dir
        / "house_2022_characteristics_elasticity_diagnostic_district_detail.csv"
    )

    validation_path = (
        args.output_dir
        / "house_2022_characteristics_elasticity_diagnostics_validation.txt"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    calibration.to_csv(
        calibration_path,
        index=False,
    )

    changed_calls.to_csv(
        changed_calls_path,
        index=False,
    )

    district_detail.to_csv(
        district_detail_path,
        index=False,
    )

    validation_path.write_text(report)

    print(report)
    print()
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {calibration_path}")
    print(f"Wrote: {changed_calls_path}")
    print(f"Wrote: {district_detail_path}")
    print(f"Wrote: {validation_path}")


if __name__ == "__main__":
    main()
