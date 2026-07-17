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
    logistic_probability,
    score_layer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "historical/house/backtests/outputs/"
    "house_2022_backtest_inputs_layer3.csv"
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

DEFAULT_STRENGTHS = tuple(
    round(value, 2)
    for value in np.arange(0.0, 1.0001, 0.10)
)


def parse_strengths(value: str) -> tuple[float, ...]:
    """Parse comma-separated shrinkage strengths from 0 through 1."""
    strengths: list[float] = []

    for item in value.split(","):
        item = item.strip()

        if not item:
            continue

        try:
            strength = float(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid shrinkage strength: {item!r}"
            ) from exc

        if not 0.0 <= strength <= 1.0:
            raise argparse.ArgumentTypeError(
                "Shrinkage strengths must be between 0.0 and 1.0."
            )

        strengths.append(round(strength, 6))

    if not strengths:
        raise argparse.ArgumentTypeError(
            "At least one shrinkage strength is required."
        )

    return tuple(sorted(set(strengths)))


def build_scoring_mask(races: pd.DataFrame) -> pd.Series:
    """Return the established ordinary D-vs-R scoring sample."""
    return (
        races["general_election_party_structure"]
        .fillna("")
        .eq("D_vs_R")
        & races["district_pres_margin_dem"].notna()
        & races["actual_dem_margin"].notna()
    )


def construct_elasticity_vector(
    races: pd.DataFrame,
    shrinkage_strength: float,
    scoring_mask: pd.Series,
) -> tuple[pd.Series, float]:
    """
    Construct a partially pooled elasticity vector.

    Raw district deviations are shrunk toward the empirical target, then
    the vector is re-centered so its mean among scored races equals 1.0.
    This preserves the calibrated national-environment level and tests
    only whether district-level heterogeneity improves forecasting.
    """
    raw = pd.to_numeric(
        races["district_elasticity_raw"],
        errors="coerce",
    )

    target = pd.to_numeric(
        races["elasticity_shrink_target"],
        errors="coerce",
    )

    finite_targets = target[
        target.notna() & np.isfinite(target)
    ].unique()

    if len(finite_targets) != 1:
        raise ValueError(
            "Layer 3 input must contain exactly one finite "
            "elasticity shrink target."
        )

    shrink_target = float(finite_targets[0])

    raw_filled = raw.fillna(shrink_target)

    elasticity = (
        shrink_target
        + (1.0 - shrinkage_strength)
        * (raw_filled - shrink_target)
    )

    scored_mean = float(
        elasticity.loc[scoring_mask].mean()
    )

    if not np.isfinite(scored_mean) or scored_mean == 0:
        raise RuntimeError(
            "Could not calculate a finite nonzero scored-sample "
            "elasticity mean."
        )

    normalized = elasticity / scored_mean

    return normalized, scored_mean


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate House district-elasticity shrinkage against "
            "the established 2022 Layer 2 benchmark."
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
        "--strengths",
        type=parse_strengths,
        default=DEFAULT_STRENGTHS,
        help=(
            "Comma-separated shrinkage strengths. "
            "Default: 0.0 through 1.0 in 0.10 increments."
        ),
    )
    parser.add_argument(
        "--incumbency-bonus",
        type=float,
        default=2.0,
        help=(
            "Calibrated symmetric incumbency bonus used in Layer 2 "
            "and Layer 3. Default: 2.0."
        ),
    )
    parser.add_argument(
        "--error-sd",
        type=float,
        default=6.5,
        help=(
            "Logistic probability scale retained from the current "
            "layered benchmark."
        ),
    )
    args = parser.parse_args()

    if not args.input_path.exists():
        raise FileNotFoundError(
            f"Missing Layer 3 input: {args.input_path}"
        )

    if not args.environment_path.exists():
        raise FileNotFoundError(
            f"Missing national-environment input: "
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
            "Duplicate race IDs found in Layer 3 input."
        )

    required_columns = {
        "cycle",
        "race_id",
        "district_pres_margin_dem",
        "actual_dem_margin",
        "actual_winner",
        "general_election_party_structure",
        "dem_is_incumbent",
        "gop_is_incumbent",
        "district_elasticity_raw",
        "elasticity_shrink_target",
    }

    missing = sorted(required_columns - set(races.columns))

    if missing:
        raise ValueError(
            "Layer 3 input is missing required columns: "
            + ", ".join(missing)
        )

    if len(environment) != 1:
        raise ValueError(
            "National-environment input must contain one row."
        )

    if (
        "national_environment_margin_dem"
        not in environment.columns
    ):
        raise ValueError(
            "National-environment input is missing "
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
    scored_count = int(scoring_mask.sum())

    if scored_count == 0:
        raise RuntimeError(
            "No races are eligible for elasticity sensitivity scoring."
        )

    parameters = BacktestParameters(
        national_environment_margin_dem=float(
            national_environment
        ),
        incumbency_bonus=args.incumbency_bonus,
        elasticity_default=1.0,
    )

    # Establish Layer 2 once. It must remain fixed across the grid.
    races["layer_2_margin_dem"], layer_2_components = (
        calculate_forecast(
            df=races,
            component_names=[
                "presidential_baseline",
                "national_environment",
                "incumbency",
            ],
            parameters=parameters,
        )
    )

    races["layer_2_dem_probability"] = (
        logistic_probability(
            races["layer_2_margin_dem"],
            args.error_sd,
        )
    )

    scored_layer_2 = races.loc[scoring_mask].copy()

    layer_2_summary = score_layer(
        scored_races=scored_layer_2,
        model_name="layer_2_plus_incumbency",
        forecast_margin_column="layer_2_margin_dem",
        probability_column="layer_2_dem_probability",
    )

    sensitivity_rows: list[dict[str, object]] = []
    district_rows: list[pd.DataFrame] = []

    for strength in args.strengths:
        elasticity, pre_normalization_mean = (
            construct_elasticity_vector(
                races=races,
                shrinkage_strength=strength,
                scoring_mask=scoring_mask,
            )
        )

        work = races.copy()
        work["district_elasticity"] = elasticity

        work[
            "layer_3_margin_dem"
        ], layer_3_components = calculate_forecast(
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

        scored = work.loc[scoring_mask].copy()

        layer_3_summary = score_layer(
            scored_races=scored,
            model_name=(
                f"layer_3_elasticity_shrinkage_{strength:.2f}"
            ),
            forecast_margin_column="layer_3_margin_dem",
            probability_column="layer_3_dem_probability",
        )

        row = {
            "shrinkage_strength": strength,
            "elasticity_variation_retained": 1.0 - strength,
            "pre_normalization_scored_mean_elasticity": (
                pre_normalization_mean
            ),
            "normalized_scored_mean_elasticity": float(
                scored["district_elasticity"].mean()
            ),
            "normalized_scored_sd_elasticity": float(
                scored["district_elasticity"].std(ddof=0)
            ),
            "normalized_scored_min_elasticity": float(
                scored["district_elasticity"].min()
            ),
            "normalized_scored_max_elasticity": float(
                scored["district_elasticity"].max()
            ),
        }

        metric_columns = [
            "mean_absolute_error",
            "median_absolute_error",
            "rmse",
            "winner_accuracy",
            "brier_score",
            "log_loss",
            "mean_margin_error_dem_bias",
            "predicted_dem_wins_in_scored_sample",
            "expected_dem_wins_in_scored_sample",
            "predicted_win_count_error",
            "expected_win_count_error",
        ]

        for metric in metric_columns:
            layer_2_value = float(layer_2_summary[metric])
            layer_3_value = float(layer_3_summary[metric])

            row[f"layer_2_{metric}"] = layer_2_value
            row[f"layer_3_{metric}"] = layer_3_value
            row[f"layer_3_minus_layer_2_{metric}"] = (
                layer_3_value - layer_2_value
            )

        sensitivity_rows.append(row)

        detail = pd.DataFrame(
            {
                "shrinkage_strength": strength,
                "race_id": work["race_id"],
                "district_elasticity": (
                    work["district_elasticity"]
                ),
                "layer_2_environment_component": (
                    layer_2_components["national_environment"]
                ),
                "layer_3_environment_component": (
                    layer_3_components["elasticity_environment"]
                ),
                "layer_2_margin_dem": (
                    work["layer_2_margin_dem"]
                ),
                "layer_3_margin_dem": (
                    work["layer_3_margin_dem"]
                ),
                "layer_3_minus_layer_2_margin": (
                    work["layer_3_margin_dem"]
                    - work["layer_2_margin_dem"]
                ),
                "scored": scoring_mask,
            }
        )

        district_rows.append(detail)

    sensitivity = pd.DataFrame(sensitivity_rows)
    district_detail = pd.concat(
        district_rows,
        ignore_index=True,
    )

    sensitivity["mae_rank"] = sensitivity[
        "layer_3_mean_absolute_error"
    ].rank(method="min")

    sensitivity["rmse_rank"] = sensitivity[
        "layer_3_rmse"
    ].rank(method="min")

    sensitivity["brier_rank"] = sensitivity[
        "layer_3_brier_score"
    ].rank(method="min")

    sensitivity["log_loss_rank"] = sensitivity[
        "layer_3_log_loss"
    ].rank(method="min")

    sensitivity["combined_rank"] = (
        sensitivity["mae_rank"]
        + sensitivity["rmse_rank"]
        + sensitivity["brier_rank"]
        + sensitivity["log_loss_rank"]
    )

    sensitivity = sensitivity.sort_values(
        ["combined_rank", "shrinkage_strength"]
    ).reset_index(drop=True)

    control = sensitivity.loc[
        np.isclose(
            sensitivity["shrinkage_strength"],
            1.0,
        )
    ]

    failures: list[str] = []

    if control.empty:
        failures.append(
            "The sensitivity grid must include shrinkage strength 1.0 "
            "to verify the Layer 2 equivalence control."
        )
        control_margin_difference = np.nan
        control_mae_difference = np.nan
    else:
        control_row = control.iloc[0]
        control_mae_difference = float(
            control_row[
                "layer_3_minus_layer_2_mean_absolute_error"
            ]
        )

        control_detail = district_detail.loc[
            np.isclose(
                district_detail["shrinkage_strength"],
                1.0,
            )
        ]

        control_margin_difference = float(
            control_detail[
                "layer_3_minus_layer_2_margin"
            ].abs().max()
        )

        if control_margin_difference > 1e-9:
            failures.append(
                "100% shrinkage is not numerically identical to "
                f"Layer 2; maximum margin difference is "
                f"{control_margin_difference:.12f}."
            )

    best = sensitivity.iloc[0]

    report_lines = [
        "2022 House Elasticity Sensitivity Validation",
        "=" * 44,
        "",
        f"Scored races: {scored_count}",
        (
            "National environment Dem: "
            f"{float(national_environment):+.6f}"
        ),
        f"Incumbency bonus: {args.incumbency_bonus:.4f}",
        f"Probability error scale: {args.error_sd:.4f}",
        (
            "Sensitivity strengths: "
            + ", ".join(
                f"{value:.2f}"
                for value in args.strengths
            )
        ),
        "",
        "Normalization:",
        (
            "Each elasticity vector is re-centered to mean 1.0 "
            "within the scored sample. This isolates district-level "
            "variation from the overall national-environment coefficient."
        ),
        "",
        "Layer 2 equivalence control:",
        (
            "Maximum absolute Layer 3 minus Layer 2 margin at "
            f"100% shrinkage: {control_margin_difference:.12f}"
        ),
        (
            "MAE difference at 100% shrinkage: "
            f"{control_mae_difference:.12f}"
        ),
        "",
        "Best combined-rank setting:",
        f"Shrinkage strength: {best['shrinkage_strength']:.2f}",
        (
            "Elasticity variation retained: "
            f"{best['elasticity_variation_retained']:.2f}"
        ),
        (
            "Layer 3 MAE: "
            f"{best['layer_3_mean_absolute_error']:.6f}"
        ),
        (
            "Layer 3 minus Layer 2 MAE: "
            f"{best['layer_3_minus_layer_2_mean_absolute_error']:+.6f}"
        ),
        (
            "Layer 3 RMSE: "
            f"{best['layer_3_rmse']:.6f}"
        ),
        (
            "Layer 3 minus Layer 2 RMSE: "
            f"{best['layer_3_minus_layer_2_rmse']:+.6f}"
        ),
        (
            "Layer 3 Brier score: "
            f"{best['layer_3_brier_score']:.6f}"
        ),
        (
            "Layer 3 minus Layer 2 Brier: "
            f"{best['layer_3_minus_layer_2_brier_score']:+.6f}"
        ),
        (
            "Layer 3 log loss: "
            f"{best['layer_3_log_loss']:.6f}"
        ),
        (
            "Layer 3 minus Layer 2 log loss: "
            f"{best['layer_3_minus_layer_2_log_loss']:+.6f}"
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

    sensitivity_path = (
        args.output_dir
        / "house_2022_elasticity_sensitivity.csv"
    )

    district_detail_path = (
        args.output_dir
        / "house_2022_elasticity_sensitivity_district_detail.csv"
    )

    validation_path = (
        args.output_dir
        / "house_2022_elasticity_sensitivity_validation.txt"
    )

    sensitivity.to_csv(
        sensitivity_path,
        index=False,
    )

    district_detail.to_csv(
        district_detail_path,
        index=False,
    )

    validation_path.write_text(report)

    print(report)
    print()
    print("Elasticity sensitivity results")
    print("------------------------------")

    display_columns = [
        "shrinkage_strength",
        "elasticity_variation_retained",
        "normalized_scored_sd_elasticity",
        "layer_3_mean_absolute_error",
        "layer_3_minus_layer_2_mean_absolute_error",
        "layer_3_rmse",
        "layer_3_minus_layer_2_rmse",
        "layer_3_winner_accuracy",
        "layer_3_brier_score",
        "layer_3_minus_layer_2_brier_score",
        "layer_3_log_loss",
        "layer_3_minus_layer_2_log_loss",
        "layer_3_predicted_win_count_error",
        "layer_3_expected_win_count_error",
        "combined_rank",
    ]

    print(
        sensitivity[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print()
    print(f"Wrote: {sensitivity_path}")
    print(f"Wrote: {district_detail_path}")
    print(f"Wrote: {validation_path}")


if __name__ == "__main__":
    main()
