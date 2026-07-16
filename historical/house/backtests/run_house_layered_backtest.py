from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from house_backtest_components import (
    BacktestParameters,
    calculate_forecast,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "historical/house/backtests/outputs/"
    "house_2022_backtest_inputs_provisional.csv"
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


def logistic_probability(
    margin_dem: pd.Series,
    error_sd: float,
) -> pd.Series:
    """
    Convert Democratic forecast margin to Democratic win probability.

    This is only an initial benchmark transform. The error scale will later
    be estimated from historical residuals rather than assumed.
    """
    if error_sd <= 0:
        raise ValueError("error_sd must be positive.")

    margin = pd.to_numeric(
        margin_dem,
        errors="coerce",
    )

    z = (margin / error_sd).clip(
        lower=-40,
        upper=40,
    )

    return 1.0 / (1.0 + np.exp(-z))


def safe_log_loss(
    actual_dem_win: pd.Series,
    probability_dem: pd.Series,
) -> float:
    y = pd.to_numeric(
        actual_dem_win,
        errors="coerce",
    )

    p = pd.to_numeric(
        probability_dem,
        errors="coerce",
    )

    valid = y.notna() & p.notna()

    if not valid.any():
        return math.nan

    y = y.loc[valid].astype(float)

    p = (
        p.loc[valid]
        .clip(lower=1e-9, upper=1.0 - 1e-9)
    )

    return float(
        -(
            y * np.log(p)
            + (1.0 - y) * np.log(1.0 - p)
        ).mean()
    )


def score_layer(
    scored_races: pd.DataFrame,
    model_name: str,
    forecast_margin_column: str,
    probability_column: str,
) -> dict[str, object]:
    work = scored_races.copy()

    work["margin_error"] = (
        work[forecast_margin_column]
        - work["actual_dem_margin"]
    )

    work["absolute_margin_error"] = (
        work["margin_error"].abs()
    )

    work["squared_margin_error"] = (
        work["margin_error"] ** 2
    )

    work["predicted_dem_win"] = (
        work[probability_column] >= 0.5
    )

    work["actual_dem_win"] = (
        work["actual_dem_margin"] > 0
    )

    work["correct_winner"] = (
        work["predicted_dem_win"]
        == work["actual_dem_win"]
    )

    work["brier_score"] = (
        work[probability_column]
        - work["actual_dem_win"].astype(float)
    ) ** 2

    actual_dem_wins = int(
        work["actual_dem_win"].sum()
    )

    predicted_dem_wins = int(
        work["predicted_dem_win"].sum()
    )

    expected_dem_wins = float(
        work[probability_column].sum()
    )

    return {
        "model_name": model_name,
        "scored_races": len(work),
        "mean_margin_error_dem_bias": float(
            work["margin_error"].mean()
        ),
        "mean_absolute_error": float(
            work["absolute_margin_error"].mean()
        ),
        "median_absolute_error": float(
            work["absolute_margin_error"].median()
        ),
        "rmse": float(
            np.sqrt(
                work["squared_margin_error"].mean()
            )
        ),
        "winner_accuracy": float(
            work["correct_winner"].mean()
        ),
        "brier_score": float(
            work["brier_score"].mean()
        ),
        "log_loss": safe_log_loss(
            work["actual_dem_win"],
            work[probability_column],
        ),
        "actual_dem_wins_in_scored_sample": (
            actual_dem_wins
        ),
        "predicted_dem_wins_in_scored_sample": (
            predicted_dem_wins
        ),
        "expected_dem_wins_in_scored_sample": (
            expected_dem_wins
        ),
        "predicted_win_count_error": (
            predicted_dem_wins
            - actual_dem_wins
        ),
        "expected_win_count_error": (
            expected_dem_wins
            - actual_dem_wins
        ),
    }


def build_calibration_table(
    scored_races: pd.DataFrame,
    model_name: str,
    probability_column: str,
) -> pd.DataFrame:
    work = scored_races.copy()

    work["actual_dem_win"] = (
        work["actual_dem_margin"] > 0
    ).astype(float)

    bins = np.linspace(0.0, 1.0, 11)

    labels = [
        f"{int(bins[index] * 100)}–"
        f"{int(bins[index + 1] * 100)}%"
        for index in range(len(bins) - 1)
    ]

    work["probability_bucket"] = pd.cut(
        work[probability_column],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    rows: list[dict[str, object]] = []

    for bucket, group in work.groupby(
        "probability_bucket",
        observed=False,
    ):
        if group.empty:
            continue

        average_probability = float(
            group[probability_column].mean()
        )

        actual_win_rate = float(
            group["actual_dem_win"].mean()
        )

        rows.append(
            {
                "model_name": model_name,
                "probability_bucket": str(bucket),
                "races": len(group),
                "average_dem_probability": (
                    average_probability
                ),
                "actual_dem_win_rate": (
                    actual_win_rate
                ),
                "calibration_error": (
                    average_probability
                    - actual_win_rate
                ),
                "bucket_brier_score": float(
                    (
                        group[probability_column]
                        - group["actual_dem_win"]
                    ).pow(2).mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def add_layer_diagnostics(
    scored_races: pd.DataFrame,
    model_name: str,
    forecast_margin_column: str,
    probability_column: str,
) -> pd.DataFrame:
    output = scored_races.copy()

    output["model_name"] = model_name

    output["forecast_margin_dem"] = pd.to_numeric(
        output[forecast_margin_column],
        errors="coerce",
    )

    output["dem_win_probability"] = pd.to_numeric(
        output[probability_column],
        errors="coerce",
    )

    output["actual_dem_win"] = (
        output["actual_dem_margin"] > 0
    )

    output["predicted_dem_win"] = (
        output["dem_win_probability"] >= 0.5
    )

    output["correct_winner"] = (
        output["predicted_dem_win"]
        == output["actual_dem_win"]
    )

    output["margin_error"] = (
        output["forecast_margin_dem"]
        - output["actual_dem_margin"]
    )

    output["absolute_margin_error"] = (
        output["margin_error"].abs()
    )

    output["squared_margin_error"] = (
        output["margin_error"] ** 2
    )

    output["brier_score"] = (
        output["dem_win_probability"]
        - output["actual_dem_win"].astype(float)
    ) ** 2

    keep_columns = [
        "model_name",
        "cycle",
        "race_id",
        "dem_candidate",
        "gop_candidate",
        "district_pres_margin_dem",
        "national_environment_margin_dem",
        "forecast_margin_dem",
        "dem_win_probability",
        "actual_dem_margin",
        "actual_winner",
        "actual_dem_win",
        "predicted_dem_win",
        "correct_winner",
        "margin_error",
        "absolute_margin_error",
        "squared_margin_error",
        "brier_score",
        "boundary_compatibility",
        "general_election_party_structure",
    ]

    keep_columns = [
        column
        for column in keep_columns
        if column in output.columns
    ]

    return output[keep_columns]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run transparent layered House historical "
            "backtests."
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
        "--error-sd",
        type=float,
        default=6.5,
        help=(
            "Initial logistic probability scale. "
            "This will later be empirically calibrated."
        ),
    )

    parser.add_argument(
        "--incumbency-bonus",
        type=float,
        default=1.5,
        help=(
            "Margin adjustment for a sole-party incumbent. "
            "Positive for a Democratic incumbent and negative "
            "for a Republican incumbent."
        ),
    )

    args = parser.parse_args()

    if not args.input_path.exists():
        raise FileNotFoundError(
            f"Missing backtest input: {args.input_path}"
        )

    if not args.environment_path.exists():
        raise FileNotFoundError(
            "Missing historical national environment: "
            f"{args.environment_path}"
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
            "Duplicate race IDs found in backtest inputs."
        )

    if len(environment) != 1:
        raise ValueError(
            "Historical national-environment file "
            "must contain exactly one row."
        )

    required_race_columns = [
        "cycle",
        "race_id",
        "district_pres_margin_dem",
        "actual_dem_margin",
        "actual_winner",
        "general_election_party_structure",
        "dem_is_incumbent",
        "gop_is_incumbent",
    ]

    missing_race_columns = [
        column
        for column in required_race_columns
        if column not in races.columns
    ]

    if missing_race_columns:
        raise ValueError(
            "Backtest input is missing columns: "
            + ", ".join(missing_race_columns)
        )

    if (
        "national_environment_margin_dem"
        not in environment.columns
    ):
        raise ValueError(
            "Environment file is missing "
            "national_environment_margin_dem."
        )

    national_environment = pd.to_numeric(
        environment.iloc[0][
            "national_environment_margin_dem"
        ],
        errors="coerce",
    )

    if pd.isna(national_environment):
        raise ValueError(
            "Historical national environment is blank "
            "or nonnumeric."
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

    parameters = BacktestParameters(
        national_environment_margin_dem=float(
            national_environment
        ),
        incumbency_bonus=args.incumbency_bonus,
    )

    races[
        "baseline_only_margin_dem"
    ], baseline_components = calculate_forecast(
        df=races,
        component_names=[
            "presidential_baseline",
        ],
        parameters=parameters,
    )

    races[
        "layer_1_margin_dem"
    ], layer_1_components = calculate_forecast(
        df=races,
        component_names=[
            "presidential_baseline",
            "national_environment",
        ],
        parameters=parameters,
    )

    races[
        "layer_2_margin_dem"
    ], layer_2_components = calculate_forecast(
        df=races,
        component_names=[
            "presidential_baseline",
            "national_environment",
            "incumbency",
        ],
        parameters=parameters,
    )

    races["incumbency_adjustment_dem"] = (
        layer_2_components["incumbency"]
    )

    races["baseline_only_dem_probability"] = (
        logistic_probability(
            races["baseline_only_margin_dem"],
            args.error_sd,
        )
    )

    races["layer_1_dem_probability"] = (
        logistic_probability(
            races["layer_1_margin_dem"],
            args.error_sd,
        )
    )

    races["layer_2_dem_probability"] = (
        logistic_probability(
            races["layer_2_margin_dem"],
            args.error_sd,
        )
    )

    scoring_mask = (
        races["general_election_party_structure"]
        .fillna("")
        .eq("D_vs_R")
        & races["district_pres_margin_dem"].notna()
        & races["actual_dem_margin"].notna()
    )

    scored = races.loc[
        scoring_mask
    ].copy()

    if scored.empty:
        raise RuntimeError(
            "No races are eligible for layered scoring."
        )

    summaries = [
        score_layer(
            scored_races=scored,
            model_name="baseline_only",
            forecast_margin_column=(
                "baseline_only_margin_dem"
            ),
            probability_column=(
                "baseline_only_dem_probability"
            ),
        ),
        score_layer(
            scored_races=scored,
            model_name=(
                "layer_1_baseline_plus_environment"
            ),
            forecast_margin_column="layer_1_margin_dem",
            probability_column=(
                "layer_1_dem_probability"
            ),
        ),
        score_layer(
            scored_races=scored,
            model_name=(
                "layer_2_plus_incumbency"
            ),
            forecast_margin_column="layer_2_margin_dem",
            probability_column=(
                "layer_2_dem_probability"
            ),
        ),
    ]

    summary = pd.DataFrame(summaries)

    baseline_calibration = build_calibration_table(
        scored_races=scored,
        model_name="baseline_only",
        probability_column=(
            "baseline_only_dem_probability"
        ),
    )

    layer_1_calibration = build_calibration_table(
        scored_races=scored,
        model_name=(
            "layer_1_baseline_plus_environment"
        ),
        probability_column=(
            "layer_1_dem_probability"
        ),
    )

    layer_2_calibration = build_calibration_table(
        scored_races=scored,
        model_name="layer_2_plus_incumbency",
        probability_column=(
            "layer_2_dem_probability"
        ),
    )

    calibration = pd.concat(
        [
            baseline_calibration,
            layer_1_calibration,
            layer_2_calibration,
        ],
        ignore_index=True,
    )

    baseline_results = add_layer_diagnostics(
        scored_races=scored,
        model_name="baseline_only",
        forecast_margin_column=(
            "baseline_only_margin_dem"
        ),
        probability_column=(
            "baseline_only_dem_probability"
        ),
    )

    layer_1_results = add_layer_diagnostics(
        scored_races=scored,
        model_name=(
            "layer_1_baseline_plus_environment"
        ),
        forecast_margin_column="layer_1_margin_dem",
        probability_column=(
            "layer_1_dem_probability"
        ),
    )

    layer_2_results = add_layer_diagnostics(
        scored_races=scored,
        model_name="layer_2_plus_incumbency",
        forecast_margin_column="layer_2_margin_dem",
        probability_column=(
            "layer_2_dem_probability"
        ),
    )

    district_results = pd.concat(
        [
            baseline_results,
            layer_1_results,
            layer_2_results,
        ],
        ignore_index=True,
    )

    metric_columns = [
        "mean_absolute_error",
        "rmse",
        "winner_accuracy",
        "brier_score",
        "log_loss",
        "mean_margin_error_dem_bias",
    ]

    baseline_summary = summary.loc[
        summary["model_name"].eq("baseline_only")
    ].iloc[0]

    layer_1_summary = summary.loc[
        summary["model_name"].eq(
            "layer_1_baseline_plus_environment"
        )
    ].iloc[0]

    layer_2_summary = summary.loc[
        summary["model_name"].eq(
            "layer_2_plus_incumbency"
        )
    ].iloc[0]

    comparison_rows = []

    for metric in metric_columns:
        baseline_value = float(
            baseline_summary[metric]
        )

        layer_1_value = float(
            layer_1_summary[metric]
        )

        layer_2_value = float(
            layer_2_summary[metric]
        )

        comparison_rows.append(
            {
                "metric": metric,
                "baseline_only": baseline_value,
                "layer_1": layer_1_value,
                "layer_2": layer_2_value,
                "layer_1_minus_baseline": (
                    layer_1_value - baseline_value
                ),
                "layer_2_minus_layer_1": (
                    layer_2_value - layer_1_value
                ),
            }
        )

    comparison = pd.DataFrame(
        comparison_rows
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        args.output_dir
        / "house_2022_layered_backtest_summary.csv"
    )

    comparison_path = (
        args.output_dir
        / "house_2022_layer_1_comparison.csv"
    )

    calibration_path = (
        args.output_dir
        / "house_2022_layered_calibration.csv"
    )

    district_results_path = (
        args.output_dir
        / "house_2022_layered_district_results.csv"
    )

    validation_path = (
        args.output_dir
        / "house_2022_layered_backtest_validation.txt"
    )

    component_detail_path = (
        args.output_dir
        / "house_2022_layered_component_detail.csv"
    )

    component_detail = pd.DataFrame(
        {
            "race_id": races["race_id"],
            "baseline_only_presidential_baseline": (
                baseline_components[
                    "presidential_baseline"
                ]
            ),
            "layer_1_presidential_baseline": (
                layer_1_components[
                    "presidential_baseline"
                ]
            ),
            "layer_1_national_environment": (
                layer_1_components[
                    "national_environment"
                ]
            ),
            "layer_2_presidential_baseline": (
                layer_2_components[
                    "presidential_baseline"
                ]
            ),
            "layer_2_national_environment": (
                layer_2_components[
                    "national_environment"
                ]
            ),
            "layer_2_incumbency": (
                layer_2_components[
                    "incumbency"
                ]
            ),
            "baseline_only_margin_dem": (
                races["baseline_only_margin_dem"]
            ),
            "layer_1_margin_dem": (
                races["layer_1_margin_dem"]
            ),
            "layer_2_margin_dem": (
                races["layer_2_margin_dem"]
            ),
        }
    )

    component_detail.to_csv(
        component_detail_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    calibration.to_csv(
        calibration_path,
        index=False,
    )

    district_results.to_csv(
        district_results_path,
        index=False,
    )

    excluded_missing_baseline = int(
        (
            races[
                "general_election_party_structure"
            ].eq("D_vs_R")
            & races[
                "district_pres_margin_dem"
            ].isna()
        ).sum()
    )

    excluded_nonstandard = int(
        (
            ~races[
                "general_election_party_structure"
            ].eq("D_vs_R")
        ).sum()
    )

    validation_lines = [
        "2022 House Layered Backtest Validation",
        "=" * 38,
        "",
        f"All House races: {len(races)}",
        f"Scored ordinary D-vs-R races: {len(scored)}",
        (
            "Excluded D-vs-R races missing baseline: "
            f"{excluded_missing_baseline}"
        ),
        (
            "Excluded nonstandard races: "
            f"{excluded_nonstandard}"
        ),
        (
            "National environment Dem: "
            f"{float(national_environment):+.4f}"
        ),
        (
            "Probability error scale: "
            f"{args.error_sd:.4f}"
        ),
        "",
        "Model definitions:",
        (
            "baseline_only = 2020 presidential "
            "district margin"
        ),
        (
            "layer_1 = 2020 presidential district "
            "margin + national environment"
        ),
        (
            "layer_2 = layer_1 + symmetric incumbency "
            "adjustment"
        ),
        (
            "Incumbency bonus used: "
            f"{args.incumbency_bonus:.2f} points"
        ),
        "",
        "Important limitations:",
        (
            "- All 435 presidential district baselines use "
            "the congressional boundaries used in 2022."
        ),
        (
            "- Layer 2 includes only a symmetric incumbency "
            "adjustment. Elasticity, candidate quality, state "
            "adjustment, special adjustment, and polling remain "
            "excluded."
        ),
        (
            "- Brier score and log loss depend on an assumed "
            "6.5-point probability scale that has not yet "
            "been historically calibrated."
        ),
        (
            "- Win-count totals refer only to the scored sample, "
            "not all 435 House seats."
        ),
    ]

    validation_text = "\n".join(
        validation_lines
    )

    validation_path.write_text(
        validation_text
    )

    print(validation_text)

    print()
    print("Layer summaries")
    print("---------------")
    print(summary.to_string(index=False))

    print()
    print("Layer comparison")
    print("----------------")
    print(comparison.to_string(index=False))

    print()
    print("Largest Layer 1 margin misses")
    print("-----------------------------")

    layer_1_worst = (
        layer_1_results
        .sort_values(
            "absolute_margin_error",
            ascending=False,
        )
        .head(25)
    )

    display_columns = [
        "race_id",
        "district_pres_margin_dem",
        "national_environment_margin_dem",
        "forecast_margin_dem",
        "actual_dem_margin",
        "margin_error",
        "dem_win_probability",
        "actual_winner",
        "correct_winner",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in layer_1_worst.columns
    ]

    print(
        layer_1_worst[
            display_columns
        ].to_string(index=False)
    )

    print()
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {comparison_path}")
    print(f"Wrote: {calibration_path}")
    print(f"Wrote: {district_results_path}")
    print(f"Wrote: {validation_path}")
    print(f"Wrote: {component_detail_path}")


if __name__ == "__main__":
    main()
