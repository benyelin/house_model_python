from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_BACKTEST_PATH = (
    PROJECT_ROOT
    / "historical/house/backtests/outputs/"
    "house_2022_backtest_inputs_provisional.csv"
)

DEFAULT_WAR_PATH = (
    PROJECT_ROOT
    / "historical/house/backtests/outputs/candidate_war/"
    "house_2022_candidate_war_audit.csv"
)

DEFAULT_ENVIRONMENT_PATH = (
    PROJECT_ROOT
    / "historical/warehouse/processed/national_environment/"
    "house_2022_election_day_national_environment.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical/house/backtests/outputs/candidate_war/"
    "sensitivity"
)

SHRINKAGE_LEVELS = (
    0.00,
    0.10,
    0.20,
    0.30,
    0.40,
    0.45,
    0.50,
    0.60,
    0.70,
)

INCUMBENT_DISCOUNTS = (
    0.25,
    0.50,
    0.75,
    1.00,
)

ONE_SIDED_MULTIPLIERS = (
    0.25,
    0.50,
    0.75,
    1.00,
)

OBSERVATION_PRIOR_STRENGTHS = (
    0.0,
    0.5,
    1.0,
    2.0,
    3.0,
)

CAP_LEVELS = (
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
)


def parse_bool_series(
    series: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def logistic_probability(
    margin_dem: np.ndarray,
    error_sd: float,
) -> np.ndarray:
    scale = error_sd * np.sqrt(3.0) / np.pi

    values = np.clip(
        margin_dem / scale,
        -50.0,
        50.0,
    )

    return 1.0 / (
        1.0 + np.exp(-values)
    )


def binary_log_loss(
    actual: np.ndarray,
    probability: np.ndarray,
) -> float:
    probability = np.clip(
        probability,
        1e-12,
        1.0 - 1e-12,
    )

    return float(
        -np.mean(
            actual * np.log(probability)
            + (1 - actual)
            * np.log(1 - probability)
        )
    )


def observation_multiplier(
    observations: pd.Series,
    prior_strength: float,
) -> pd.Series:
    observations = pd.to_numeric(
        observations,
        errors="coerce",
    ).fillna(0.0).clip(lower=0.0)

    if prior_strength <= 0:
        return pd.Series(
            1.0,
            index=observations.index,
            dtype=float,
        )

    return (
        observations
        / (
            observations
            + prior_strength
        )
    )


def build_layer_2_margin(
    races: pd.DataFrame,
    national_environment: float,
    incumbency_bonus: float,
) -> pd.Series:
    baseline = pd.to_numeric(
        races["district_pres_margin_dem"],
        errors="coerce",
    )

    dem_incumbent = parse_bool_series(
        races["dem_is_incumbent"]
    )

    gop_incumbent = parse_bool_series(
        races["gop_is_incumbent"]
    )

    incumbency = pd.Series(
        0.0,
        index=races.index,
        dtype=float,
    )

    incumbency.loc[
        dem_incumbent & ~gop_incumbent
    ] = incumbency_bonus

    incumbency.loc[
        gop_incumbent & ~dem_incumbent
    ] = -incumbency_bonus

    return (
        baseline
        + national_environment
        + incumbency
    )


def calculate_war_adjustment(
    frame: pd.DataFrame,
    shrinkage: float,
    incumbent_discount: float,
    one_sided_multiplier: float,
    observation_prior_strength: float,
    cap: float,
) -> pd.Series:
    dem_war = pd.to_numeric(
        frame["dem_candidate_war"],
        errors="coerce",
    ).fillna(0.0)

    gop_war = pd.to_numeric(
        frame["gop_candidate_war"],
        errors="coerce",
    ).fillna(0.0)

    dem_observation_multiplier = (
        observation_multiplier(
            frame["dem_war_observations"],
            observation_prior_strength,
        )
    )

    gop_observation_multiplier = (
        observation_multiplier(
            frame["gop_war_observations"],
            observation_prior_strength,
        )
    )

    dem_effective = (
        dem_war
        * dem_observation_multiplier
    )

    gop_effective = (
        gop_war
        * gop_observation_multiplier
    )

    incumbent_party = (
        frame["incumbent_party"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    dem_effective = dem_effective.where(
        ~incumbent_party.eq("D"),
        dem_effective * incumbent_discount,
    )

    gop_effective = gop_effective.where(
        ~incumbent_party.isin({"R", "GOP"}),
        gop_effective * incumbent_discount,
    )

    raw_net = (
        dem_effective
        - gop_effective
    )

    adjustment = (
        raw_net
        * shrinkage
    ).clip(
        lower=-cap,
        upper=cap,
    )

    status = (
        frame["war_match_status"]
        .fillna("Neither matched")
        .astype(str)
    )

    both_matched = status.eq(
        "Both matched"
    )

    one_matched = status.isin(
        {
            "Only D matched",
            "Only R matched",
        }
    )

    match_multiplier = pd.Series(
        0.0,
        index=frame.index,
        dtype=float,
    )

    match_multiplier.loc[
        both_matched
    ] = 1.0

    match_multiplier.loc[
        one_matched
    ] = one_sided_multiplier

    return (
        adjustment
        * match_multiplier
    )


def score_predictions(
    actual_margin: np.ndarray,
    predicted_margin: np.ndarray,
    error_sd: float,
) -> dict[str, float]:
    error = (
        predicted_margin
        - actual_margin
    )

    actual_dem_win = (
        actual_margin > 0
    ).astype(int)

    predicted_dem_win = (
        predicted_margin > 0
    ).astype(int)

    probability = logistic_probability(
        predicted_margin,
        error_sd,
    )

    return {
        "mean_absolute_error": float(
            np.mean(
                np.abs(error)
            )
        ),
        "rmse": float(
            np.sqrt(
                np.mean(
                    np.square(error)
                )
            )
        ),
        "winner_accuracy": float(
            np.mean(
                predicted_dem_win
                == actual_dem_win
            )
        ),
        "brier_score": float(
            np.mean(
                np.square(
                    probability
                    - actual_dem_win
                )
            )
        ),
        "log_loss": binary_log_loss(
            actual_dem_win,
            probability,
        ),
        "actual_dem_wins": float(
            actual_dem_win.sum()
        ),
        "predicted_dem_wins": float(
            predicted_dem_win.sum()
        ),
        "expected_dem_wins": float(
            probability.sum()
        ),
        "mean_margin_bias_dem": float(
            np.mean(error)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a leakage-safe 2022 House candidate WAR "
            "parameter sensitivity test."
        )
    )

    parser.add_argument(
        "--backtest-path",
        type=Path,
        default=DEFAULT_BACKTEST_PATH,
    )

    parser.add_argument(
        "--war-path",
        type=Path,
        default=DEFAULT_WAR_PATH,
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
        args.backtest_path,
        args.war_path,
        args.environment_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required input: {path}"
            )

    races = pd.read_csv(
        args.backtest_path,
        dtype={"race_id": str},
    )

    war = pd.read_csv(
        args.war_path,
        dtype={
            "race_id": str,
            "district_id": str,
        },
    )

    environment = pd.read_csv(
        args.environment_path,
    )

    if len(races) != 435:
        raise ValueError(
            f"Expected 435 race rows; found {len(races)}."
        )

    if len(war) != 435:
        raise ValueError(
            f"Expected 435 WAR rows; found {len(war)}."
        )

    if races["race_id"].duplicated().any():
        raise ValueError(
            "Duplicate race IDs in backtest input."
        )

    if war["race_id"].duplicated().any():
        raise ValueError(
            "Duplicate race IDs in WAR input."
        )

    if len(environment) != 1:
        raise ValueError(
            "National-environment input must contain one row."
        )

    national_environment = float(
        pd.to_numeric(
            environment.iloc[0][
                "national_environment_margin_dem"
            ],
            errors="raise",
        )
    )

    frame = races.merge(
        war,
        on="race_id",
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_war",
        ),
    )

    scoring_mask = (
        frame[
            "general_election_party_structure"
        ]
        .fillna("")
        .eq("D_vs_R")
        & pd.to_numeric(
            frame["district_pres_margin_dem"],
            errors="coerce",
        ).notna()
        & pd.to_numeric(
            frame["actual_dem_margin"],
            errors="coerce",
        ).notna()
    )

    frame = frame.loc[
        scoring_mask
    ].copy()

    if len(frame) != 400:
        raise ValueError(
            f"Expected 400 scored races; found {len(frame)}."
        )

    frame[
        "layer_2_margin_dem"
    ] = build_layer_2_margin(
        races=frame,
        national_environment=national_environment,
        incumbency_bonus=args.incumbency_bonus,
    )

    actual_margin = pd.to_numeric(
        frame["actual_dem_margin"],
        errors="raise",
    ).to_numpy(dtype=float)

    layer_2_margin = frame[
        "layer_2_margin_dem"
    ].to_numpy(dtype=float)

    baseline_scores = score_predictions(
        actual_margin=actual_margin,
        predicted_margin=layer_2_margin,
        error_sd=args.error_sd,
    )

    rows: list[
        dict[str, float]
    ] = []

    detail_frames: list[
        pd.DataFrame
    ] = []

    parameter_grid = product(
        SHRINKAGE_LEVELS,
        INCUMBENT_DISCOUNTS,
        ONE_SIDED_MULTIPLIERS,
        OBSERVATION_PRIOR_STRENGTHS,
        CAP_LEVELS,
    )

    for (
        shrinkage,
        incumbent_discount,
        one_sided_multiplier,
        observation_prior_strength,
        cap,
    ) in parameter_grid:
        adjustment = calculate_war_adjustment(
            frame=frame,
            shrinkage=shrinkage,
            incumbent_discount=incumbent_discount,
            one_sided_multiplier=one_sided_multiplier,
            observation_prior_strength=observation_prior_strength,
            cap=cap,
        )

        predicted_margin = (
            layer_2_margin
            + adjustment.to_numpy(dtype=float)
        )

        scores = score_predictions(
            actual_margin=actual_margin,
            predicted_margin=predicted_margin,
            error_sd=args.error_sd,
        )

        row = {
            "shrinkage": shrinkage,
            "incumbent_discount": (
                incumbent_discount
            ),
            "one_sided_multiplier": (
                one_sided_multiplier
            ),
            "observation_prior_strength": (
                observation_prior_strength
            ),
            "cap": cap,
            "nonzero_adjustments": int(
                adjustment.abs().gt(0).sum()
            ),
            "mean_absolute_adjustment": float(
                adjustment.abs().mean()
            ),
            "maximum_absolute_adjustment": float(
                adjustment.abs().max()
            ),
            **scores,
        }

        for metric in [
            "mean_absolute_error",
            "rmse",
            "winner_accuracy",
            "brier_score",
            "log_loss",
            "predicted_dem_wins",
            "expected_dem_wins",
        ]:
            row[
                f"layer_2_{metric}"
            ] = baseline_scores[metric]

            row[
                f"change_vs_layer_2_{metric}"
            ] = (
                scores[metric]
                - baseline_scores[metric]
            )

        row[
            "predicted_win_count_error"
        ] = abs(
            scores["predicted_dem_wins"]
            - scores["actual_dem_wins"]
        )

        row[
            "expected_win_count_error"
        ] = abs(
            scores["expected_dem_wins"]
            - scores["actual_dem_wins"]
        )

        rows.append(row)

        detail = pd.DataFrame(
            {
                "race_id": frame["race_id"],
                "shrinkage": shrinkage,
                "incumbent_discount": (
                    incumbent_discount
                ),
                "one_sided_multiplier": (
                    one_sided_multiplier
                ),
                "observation_prior_strength": (
                    observation_prior_strength
                ),
                "cap": cap,
                "layer_2_margin_dem": (
                    layer_2_margin
                ),
                "candidate_war_adjustment_dem": (
                    adjustment.to_numpy(dtype=float)
                ),
                "war_adjusted_margin_dem": (
                    predicted_margin
                ),
                "actual_dem_margin": (
                    actual_margin
                ),
            }
        )

        detail[
            "layer_2_absolute_error"
        ] = np.abs(
            layer_2_margin
            - actual_margin
        )

        detail[
            "war_adjusted_absolute_error"
        ] = np.abs(
            predicted_margin
            - actual_margin
        )

        detail[
            "absolute_error_change"
        ] = (
            detail[
                "war_adjusted_absolute_error"
            ]
            - detail[
                "layer_2_absolute_error"
            ]
        )

        detail_frames.append(detail)

    grid = pd.DataFrame(rows)

    for metric in [
        "mean_absolute_error",
        "rmse",
        "brier_score",
        "log_loss",
        "expected_win_count_error",
    ]:
        grid[
            f"{metric}_rank"
        ] = grid[metric].rank(
            method="min",
            ascending=True,
        )

    grid[
        "combined_rank"
    ] = grid[
        [
            "mean_absolute_error_rank",
            "rmse_rank",
            "brier_score_rank",
            "log_loss_rank",
            "expected_win_count_error_rank",
        ]
    ].sum(axis=1)

    grid = grid.sort_values(
        [
            "combined_rank",
            "mean_absolute_error",
            "rmse",
            "brier_score",
            "log_loss",
        ]
    ).reset_index(drop=True)

    best = grid.iloc[0]

    details = pd.concat(
        detail_frames,
        ignore_index=True,
    )

    best_detail = details.loc[
        np.isclose(
            details["shrinkage"],
            float(best["shrinkage"]),
        )
        & np.isclose(
            details["incumbent_discount"],
            float(
                best[
                    "incumbent_discount"
                ]
            ),
        )
        & np.isclose(
            details["one_sided_multiplier"],
            float(
                best[
                    "one_sided_multiplier"
                ]
            ),
        )
        & np.isclose(
            details[
                "observation_prior_strength"
            ],
            float(
                best[
                    "observation_prior_strength"
                ]
            ),
        )
        & np.isclose(
            details["cap"],
            float(best["cap"]),
        )
    ].copy()

    failures: list[str] = []

    zero_control = grid.loc[
        grid["shrinkage"].eq(0.0)
    ]

    if zero_control.empty:
        failures.append(
            "Missing zero-WAR control."
        )
    else:
        maximum_zero_difference = (
            zero_control[
                "change_vs_layer_2_mean_absolute_error"
            ]
            .abs()
            .max()
        )

        if maximum_zero_difference > 1e-12:
            failures.append(
                "Zero-WAR control does not reproduce Layer 2."
            )

    if (
        best_detail[
            "candidate_war_adjustment_dem"
        ].abs().max()
        > float(best["cap"]) + 1e-9
    ):
        failures.append(
            "Best-setting adjustment exceeds its cap."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    grid_path = (
        args.output_dir
        / "house_2022_candidate_war_sensitivity.csv"
    )

    best_detail_path = (
        args.output_dir
        / "house_2022_candidate_war_best_detail.csv"
    )

    validation_path = (
        args.output_dir
        / "house_2022_candidate_war_sensitivity_validation.txt"
    )

    grid.to_csv(
        grid_path,
        index=False,
    )

    best_detail.to_csv(
        best_detail_path,
        index=False,
    )

    report_lines = [
        "2022 House Candidate WAR Sensitivity",
        "=" * 36,
        "",
        f"Scored races: {len(frame)}",
        (
            "National environment Dem: "
            f"{national_environment:+.6f}"
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
        "Layer 2 benchmark:",
        (
            "MAE: "
            f"{baseline_scores['mean_absolute_error']:.6f}"
        ),
        (
            "RMSE: "
            f"{baseline_scores['rmse']:.6f}"
        ),
        (
            "Winner accuracy: "
            f"{baseline_scores['winner_accuracy']:.6f}"
        ),
        (
            "Brier: "
            f"{baseline_scores['brier_score']:.6f}"
        ),
        (
            "Log loss: "
            f"{baseline_scores['log_loss']:.6f}"
        ),
        "",
        "Best combined-rank WAR setting:",
        (
            "Shrinkage: "
            f"{float(best['shrinkage']):.4f}"
        ),
        (
            "Incumbent discount: "
            f"{float(best['incumbent_discount']):.4f}"
        ),
        (
            "One-sided multiplier: "
            f"{float(best['one_sided_multiplier']):.4f}"
        ),
        (
            "Observation prior strength: "
            f"{float(best['observation_prior_strength']):.4f}"
        ),
        (
            "Cap: "
            f"{float(best['cap']):.4f}"
        ),
        (
            "MAE: "
            f"{float(best['mean_absolute_error']):.6f}"
        ),
        (
            "MAE change versus Layer 2: "
            f"{float(best['change_vs_layer_2_mean_absolute_error']):+.6f}"
        ),
        (
            "RMSE: "
            f"{float(best['rmse']):.6f}"
        ),
        (
            "RMSE change versus Layer 2: "
            f"{float(best['change_vs_layer_2_rmse']):+.6f}"
        ),
        (
            "Winner accuracy: "
            f"{float(best['winner_accuracy']):.6f}"
        ),
        (
            "Brier change versus Layer 2: "
            f"{float(best['change_vs_layer_2_brier_score']):+.6f}"
        ),
        (
            "Log-loss change versus Layer 2: "
            f"{float(best['change_vs_layer_2_log_loss']):+.6f}"
        ),
        (
            "Expected-seat error: "
            f"{float(best['expected_win_count_error']):.6f}"
        ),
        (
            "Mean absolute WAR adjustment: "
            f"{float(best['mean_absolute_adjustment']):.6f}"
        ),
        (
            "Maximum absolute WAR adjustment: "
            f"{float(best['maximum_absolute_adjustment']):.6f}"
        ),
        "",
        "Methodological limitation:",
        (
            "This calibrates WAR against the 2022 election only. "
            "No production setting should be changed until the same "
            "parameter family is tested across earlier election cycles."
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

    report = "\n".join(
        report_lines
    )

    validation_path.write_text(
        report
    )

    if failures:
        raise RuntimeError(report)

    print(report)
    print()
    print(f"Wrote: {grid_path}")
    print(f"Wrote: {best_detail_path}")
    print(f"Wrote: {validation_path}")


if __name__ == "__main__":
    main()
