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
    "house_2022_backtest_inputs_characteristics_elasticity.csv"
)

DEFAULT_DNA_PATH = (
    PROJECT_ROOT
    / "historical/house/elasticity/"
    "house_district_dna_predicted_elasticity.csv"
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

DEFAULT_RETENTION_LEVELS = tuple(
    round(value, 2)
    for value in np.arange(0.0, 1.0001, 0.10)
)

DNA_COLUMN = (
    "district_dna_elasticity_bounded_normalized"
)


def parse_retention_levels(
    value: str,
) -> tuple[float, ...]:
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

        levels.append(
            round(level, 6)
        )

    if not levels:
        raise argparse.ArgumentTypeError(
            "At least one retention level is required."
        )

    return tuple(
        sorted(set(levels))
    )


def build_scoring_mask(
    races: pd.DataFrame,
) -> pd.Series:
    return (
        races[
            "general_election_party_structure"
        ]
        .fillna("")
        .eq("D_vs_R")
        & races[
            "district_pres_margin_dem"
        ].notna()
        & races[
            "actual_dem_margin"
        ].notna()
    )


def construct_retained_elasticity(
    races: pd.DataFrame,
    retention_level: float,
    scoring_mask: pd.Series,
) -> tuple[pd.Series, float]:
    """
    Retain a configurable share of District DNA variation around 1.0.

    retention_level = 0.0:
        every district receives elasticity 1.0

    retention_level = 1.0:
        full bounded and normalized District DNA elasticity is retained
    """
    predicted = pd.to_numeric(
        races[DNA_COLUMN],
        errors="coerce",
    )

    if predicted.isna().any():
        raise ValueError(
            "Missing District DNA elasticity predictions."
        )

    retained = (
        1.0
        + retention_level
        * (predicted - 1.0)
    )

    scored_mean_before_normalization = float(
        retained.loc[
            scoring_mask
        ].mean()
    )

    if (
        not np.isfinite(
            scored_mean_before_normalization
        )
        or scored_mean_before_normalization == 0
    ):
        raise RuntimeError(
            "Scored-sample elasticity mean is invalid."
        )

    # Re-center in the scored sample so the experiment isolates
    # heterogeneous district responsiveness rather than changing
    # the overall national-environment effect.
    normalized = (
        retained
        / scored_mean_before_normalization
    )

    return (
        normalized,
        scored_mean_before_normalization,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test District DNA House elasticity against the "
            "established 2022 Layer 2 benchmark."
        )
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
    )

    parser.add_argument(
        "--dna-path",
        type=Path,
        default=DEFAULT_DNA_PATH,
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
        help=(
            "Comma-separated shares of District DNA elasticity "
            "variation to retain."
        ),
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

    for path in [
        args.input_path,
        args.dna_path,
        args.environment_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required input: {path}"
            )

    races = pd.read_csv(
        args.input_path,
        dtype={"race_id": str},
    )

    dna = pd.read_csv(
        args.dna_path,
        dtype={"race_id": str},
    )

    environment = pd.read_csv(
        args.environment_path,
    )

    if len(races) != 435:
        raise ValueError(
            f"Expected 435 races; found {len(races)}."
        )

    if len(dna) != 435:
        raise ValueError(
            f"Expected 435 DNA predictions; found {len(dna)}."
        )

    if races["race_id"].duplicated().any():
        raise ValueError(
            "Duplicate race IDs found in backtest inputs."
        )

    if dna["race_id"].duplicated().any():
        raise ValueError(
            "Duplicate race IDs found in DNA predictions."
        )

    required_race_columns = {
        "race_id",
        "district_pres_margin_dem",
        "actual_dem_margin",
        "actual_winner",
        "general_election_party_structure",
        "dem_is_incumbent",
        "gop_is_incumbent",
    }

    missing_race_columns = sorted(
        required_race_columns
        - set(races.columns)
    )

    if missing_race_columns:
        raise ValueError(
            "Backtest input is missing columns: "
            + ", ".join(missing_race_columns)
        )

    required_dna_columns = {
        "race_id",
        DNA_COLUMN,
        "elasticity_source",
        "selected_regime_reliability",
        "selected_regime_scorable_elections",
    }

    missing_dna_columns = sorted(
        required_dna_columns
        - set(dna.columns)
    )

    if missing_dna_columns:
        raise ValueError(
            "District DNA file is missing columns: "
            + ", ".join(missing_dna_columns)
        )

    races = races.merge(
        dna[
            [
                "race_id",
                DNA_COLUMN,
                "district_dna_raw_prediction",
                "district_dna_elasticity_normalized",
                "elasticity_source",
                "selected_regime_reliability",
                "selected_regime_scorable_elections",
                "behavior_reliability_score",
            ]
        ],
        on="race_id",
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_district_dna",
        ),
    )

    if races[DNA_COLUMN].isna().any():
        missing_ids = races.loc[
            races[DNA_COLUMN].isna(),
            "race_id",
        ].tolist()

        raise ValueError(
            "Missing District DNA predictions after merge: "
            + ", ".join(missing_ids)
        )

    if len(environment) != 1:
        raise ValueError(
            "National-environment file must contain one row."
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

    races[
        "district_pres_margin_dem"
    ] = pd.to_numeric(
        races[
            "district_pres_margin_dem"
        ],
        errors="coerce",
    )

    races["actual_dem_margin"] = pd.to_numeric(
        races["actual_dem_margin"],
        errors="coerce",
    )

    races[
        "national_environment_margin_dem"
    ] = float(
        national_environment
    )

    scoring_mask = build_scoring_mask(
        races
    )

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
    ], layer_2_components = calculate_forecast(
        df=races,
        component_names=[
            "presidential_baseline",
            "national_environment",
            "incumbency",
        ],
        parameters=parameters,
    )

    races[
        "layer_2_dem_probability"
    ] = logistic_probability(
        races["layer_2_margin_dem"],
        args.error_sd,
    )

    scored_layer_2 = races.loc[
        scoring_mask
    ].copy()

    layer_2_summary = score_layer(
        scored_races=scored_layer_2,
        model_name="layer_2_plus_incumbency",
        forecast_margin_column=(
            "layer_2_margin_dem"
        ),
        probability_column=(
            "layer_2_dem_probability"
        ),
    )

    sensitivity_rows: list[
        dict[str, object]
    ] = []

    district_rows: list[
        pd.DataFrame
    ] = []

    for retention in args.retention_levels:
        (
            elasticity,
            pre_normalization_mean,
        ) = construct_retained_elasticity(
            races=races,
            retention_level=retention,
            scoring_mask=scoring_mask,
        )

        work = races.copy()

        work[
            "district_elasticity"
        ] = elasticity

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

        work[
            "layer_3_dem_probability"
        ] = logistic_probability(
            work["layer_3_margin_dem"],
            args.error_sd,
        )

        scored = work.loc[
            scoring_mask
        ].copy()

        layer_3_summary = score_layer(
            scored_races=scored,
            model_name=(
                "layer_3_district_dna_elasticity_"
                f"retention_{retention:.2f}"
            ),
            forecast_margin_column=(
                "layer_3_margin_dem"
            ),
            probability_column=(
                "layer_3_dem_probability"
            ),
        )

        row: dict[str, object] = {
            "retention_level": retention,
            "shrinkage_toward_one": (
                1.0 - retention
            ),
            (
                "pre_normalization_scored_"
                "mean_elasticity"
            ): pre_normalization_mean,
            (
                "normalized_scored_"
                "mean_elasticity"
            ): float(
                scored[
                    "district_elasticity"
                ].mean()
            ),
            (
                "normalized_scored_"
                "sd_elasticity"
            ): float(
                scored[
                    "district_elasticity"
                ].std(ddof=0)
            ),
            (
                "normalized_scored_"
                "min_elasticity"
            ): float(
                scored[
                    "district_elasticity"
                ].min()
            ),
            (
                "normalized_scored_"
                "max_elasticity"
            ): float(
                scored[
                    "district_elasticity"
                ].max()
            ),
        }

        metrics = [
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

        for metric in metrics:
            layer_2_value = float(
                layer_2_summary[metric]
            )

            layer_3_value = float(
                layer_3_summary[metric]
            )

            row[
                f"layer_2_{metric}"
            ] = layer_2_value

            row[
                f"layer_3_{metric}"
            ] = layer_3_value

            row[
                f"layer_3_minus_layer_2_{metric}"
            ] = (
                layer_3_value
                - layer_2_value
            )

        sensitivity_rows.append(
            row
        )

        detail = pd.DataFrame(
            {
                "retention_level": retention,
                "race_id": work["race_id"],
                "district_dna_elasticity_original": (
                    work[DNA_COLUMN]
                ),
                "district_elasticity_used": (
                    work[
                        "district_elasticity"
                    ]
                ),
                "elasticity_source": (
                    work[
                        "elasticity_source_district_dna"
                    ]
                    if (
                        "elasticity_source_district_dna"
                        in work.columns
                    )
                    else work[
                        "elasticity_source"
                    ]
                ),
                "selected_regime_reliability": (
                    work[
                        "selected_regime_reliability"
                    ]
                ),
                "selected_regime_scorable_elections": (
                    work[
                        "selected_regime_scorable_elections"
                    ]
                ),
                "layer_2_environment_component": (
                    layer_2_components[
                        "national_environment"
                    ]
                ),
                "layer_3_environment_component": (
                    layer_3_components[
                        "elasticity_environment"
                    ]
                ),
                "layer_2_margin_dem": (
                    work[
                        "layer_2_margin_dem"
                    ]
                ),
                "layer_3_margin_dem": (
                    work[
                        "layer_3_margin_dem"
                    ]
                ),
                "layer_3_minus_layer_2_margin": (
                    work[
                        "layer_3_margin_dem"
                    ]
                    - work[
                        "layer_2_margin_dem"
                    ]
                ),
                "actual_dem_margin": (
                    work[
                        "actual_dem_margin"
                    ]
                ),
                "scored": scoring_mask,
            }
        )

        district_rows.append(
            detail
        )

    sensitivity = pd.DataFrame(
        sensitivity_rows
    )

    district_detail = pd.concat(
        district_rows,
        ignore_index=True,
    )

    sensitivity["mae_rank"] = sensitivity[
        "layer_3_mean_absolute_error"
    ].rank(
        method="min"
    )

    sensitivity["rmse_rank"] = sensitivity[
        "layer_3_rmse"
    ].rank(
        method="min"
    )

    sensitivity["brier_rank"] = sensitivity[
        "layer_3_brier_score"
    ].rank(
        method="min"
    )

    sensitivity["log_loss_rank"] = sensitivity[
        "layer_3_log_loss"
    ].rank(
        method="min"
    )

    sensitivity["combined_rank"] = (
        sensitivity["mae_rank"]
        + sensitivity["rmse_rank"]
        + sensitivity["brier_rank"]
        + sensitivity["log_loss_rank"]
    )

    sensitivity = sensitivity.sort_values(
        [
            "combined_rank",
            "retention_level",
        ]
    ).reset_index(drop=True)

    control = sensitivity.loc[
        np.isclose(
            sensitivity[
                "retention_level"
            ],
            0.0,
        )
    ]

    failures: list[str] = []

    if control.empty:
        failures.append(
            "Retention level 0.0 is required "
            "as the Layer 2 control."
        )

        control_margin_difference = np.nan
        control_mae_difference = np.nan
    else:
        control_detail = district_detail.loc[
            np.isclose(
                district_detail[
                    "retention_level"
                ],
                0.0,
            )
        ]

        control_margin_difference = float(
            control_detail[
                "layer_3_minus_layer_2_margin"
            ]
            .abs()
            .max()
        )

        control_mae_difference = float(
            control.iloc[0][
                "layer_3_minus_layer_2_"
                "mean_absolute_error"
            ]
        )

        if control_margin_difference > 1e-9:
            failures.append(
                "Zero-retention control is not "
                "identical to Layer 2."
            )

    best = sensitivity.iloc[0]

    nonzero = sensitivity.loc[
        sensitivity[
            "retention_level"
        ].gt(0)
    ].copy()

    best_nonzero = (
        nonzero.iloc[0]
        if not nonzero.empty
        else None
    )

    report_lines = [
        "2022 House District DNA Elasticity Sensitivity",
        "=" * 46,
        "",
        f"Scored races: {int(scoring_mask.sum())}",
        (
            "National environment Dem: "
            f"{float(national_environment):+.6f}"
        ),
        (
            "Incumbency bonus: "
            f"{args.incumbency_bonus:.4f}"
        ),
        (
            "Probability error scale: "
            f"{args.error_sd:.4f}"
        ),
        "",
        "District DNA sources:",
        dna[
            "elasticity_source"
        ]
        .value_counts()
        .to_string(),
        "",
        "Control validation:",
        (
            "Maximum absolute margin difference "
            "at zero retention: "
            f"{control_margin_difference:.12f}"
        ),
        (
            "MAE difference at zero retention: "
            f"{control_mae_difference:.12f}"
        ),
        "",
        "Best combined-rank setting:",
        (
            "Retention level: "
            f"{best['retention_level']:.2f}"
        ),
        (
            "Shrinkage toward elasticity 1.0: "
            f"{best['shrinkage_toward_one']:.2f}"
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
            "Layer 3 winner accuracy: "
            f"{best['layer_3_winner_accuracy']:.6f}"
        ),
        (
            "Layer 3 Brier: "
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
    ]

    if best_nonzero is not None:
        report_lines.extend(
            [
                "",
                "Best nonzero-retention setting:",
                (
                    "Retention level: "
                    f"{best_nonzero['retention_level']:.2f}"
                ),
                (
                    "Layer 3 minus Layer 2 MAE: "
                    f"{best_nonzero['layer_3_minus_layer_2_mean_absolute_error']:+.6f}"
                ),
                (
                    "Layer 3 minus Layer 2 RMSE: "
                    f"{best_nonzero['layer_3_minus_layer_2_rmse']:+.6f}"
                ),
                (
                    "Layer 3 minus Layer 2 Brier: "
                    f"{best_nonzero['layer_3_minus_layer_2_brier_score']:+.6f}"
                ),
                (
                    "Layer 3 minus Layer 2 log loss: "
                    f"{best_nonzero['layer_3_minus_layer_2_log_loss']:+.6f}"
                ),
            ]
        )

    report_lines.extend(
        [
            "",
            "Validation status:",
        ]
    )

    if failures:
        report_lines.append(
            "FAILED"
        )

        report_lines.extend(
            f"- {failure}"
            for failure in failures
        )
    else:
        report_lines.append(
            "PASSED"
        )

    report = "\n".join(
        report_lines
    )

    if failures:
        raise RuntimeError(
            report
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sensitivity_path = (
        args.output_dir
        / "house_2022_district_dna_elasticity_sensitivity.csv"
    )

    district_detail_path = (
        args.output_dir
        / "house_2022_district_dna_elasticity_sensitivity_district_detail.csv"
    )

    validation_path = (
        args.output_dir
        / "house_2022_district_dna_elasticity_sensitivity_validation.txt"
    )

    sensitivity.to_csv(
        sensitivity_path,
        index=False,
    )

    district_detail.to_csv(
        district_detail_path,
        index=False,
    )

    validation_path.write_text(
        report
    )

    print(report)

    print()
    print("District DNA elasticity sensitivity")
    print("-" * 36)

    display_columns = [
        "retention_level",
        "shrinkage_toward_one",
        "normalized_scored_sd_elasticity",
        "normalized_scored_min_elasticity",
        "normalized_scored_max_elasticity",
        "layer_3_mean_absolute_error",
        "layer_3_minus_layer_2_mean_absolute_error",
        "layer_3_rmse",
        "layer_3_minus_layer_2_rmse",
        "layer_3_winner_accuracy",
        "layer_3_minus_layer_2_winner_accuracy",
        "layer_3_brier_score",
        "layer_3_minus_layer_2_brier_score",
        "layer_3_log_loss",
        "layer_3_minus_layer_2_log_loss",
        "layer_3_predicted_win_count_error",
        "layer_3_expected_win_count_error",
        "combined_rank",
    ]

    print(
        sensitivity[
            display_columns
        ].to_string(
            index=False,
            float_format=(
                lambda value: f"{value:.6f}"
            ),
        )
    )

    print()
    print(f"Wrote: {sensitivity_path}")
    print(f"Wrote: {district_detail_path}")
    print(f"Wrote: {validation_path}")


if __name__ == "__main__":
    main()
