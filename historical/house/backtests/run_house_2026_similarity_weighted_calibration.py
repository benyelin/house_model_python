from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


INPUT_PATH = Path(
    "historical/house/backtests/outputs/"
    "production_environment_bakeoff/"
    "house_production_environment_bakeoff_by_cycle.csv"
)

OUTPUT_DIR = Path(
    "historical/house/backtests/outputs/"
    "2026_similarity_weighted_calibration"
)

BASELINE_TYPE = "normalized_partisan_baseline_dem"

# These scenarios vary only the assumed national-cycle relevance.
# District maps and district-level characteristics are deliberately
# excluded because they are already modeled through the baseline,
# elasticity, incumbency, candidate quality, and other race inputs.
SCENARIO_WEIGHTS = {
    "equal_weight": {
        2016: 0.25,
        2018: 0.25,
        2020: 0.25,
        2022: 0.25,
    },
    "midterm_emphasis": {
        2016: 0.15,
        2018: 0.35,
        2020: 0.15,
        2022: 0.35,
    },
    "moderate_trump_midterm_emphasis": {
        2016: 0.10,
        2018: 0.45,
        2020: 0.10,
        2022: 0.35,
    },
    "strong_trump_midterm_emphasis": {
        2016: 0.05,
        2018: 0.60,
        2020: 0.05,
        2022: 0.30,
    },
    "extreme_2018_stress_test": {
        2016: 0.025,
        2018: 0.75,
        2020: 0.025,
        2022: 0.20,
    },
}

# Restrict the final decision table to the plausible region identified
# in the prior calibration while retaining enough range to detect
# whether the 2018 weighting materially favors a higher coefficient.
CANDIDATE_MULTIPLIERS = [
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]

LOWER_IS_BETTER = [
    "weighted_mae",
    "weighted_rmse",
    "weighted_brier",
    "weighted_absolute_cycle_bias",
    "weighted_seat_error",
]

HIGHER_IS_BETTER = [
    "weighted_winner_accuracy",
]


def validate_weights() -> None:
    expected_cycles = {2016, 2018, 2020, 2022}

    for scenario, weights in SCENARIO_WEIGHTS.items():
        if set(weights) != expected_cycles:
            raise ValueError(
                f"{scenario} does not contain exactly "
                f"{sorted(expected_cycles)}"
            )

        total = sum(weights.values())

        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"{scenario} weights sum to {total}, not 1.0"
            )

        if any(weight < 0.0 for weight in weights.values()):
            raise ValueError(
                f"{scenario} contains a negative weight"
            )


def load_results() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing prior bakeoff output: {INPUT_PATH}"
        )

    frame = pd.read_csv(INPUT_PATH)

    required = {
        "baseline_type",
        "house_environment_multiplier",
        "cycle",
        "mean_absolute_error",
        "rmse",
        "brier_score",
        "winner_accuracy",
        "mean_margin_error_dem_bias",
        "expected_win_count_error",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"Input is missing required columns: {sorted(missing)}"
        )

    frame = frame.loc[
        frame["baseline_type"].eq(BASELINE_TYPE)
    ].copy()

    numeric_columns = [
        "house_environment_multiplier",
        "cycle",
        "mean_absolute_error",
        "rmse",
        "brier_score",
        "winner_accuracy",
        "mean_margin_error_dem_bias",
        "expected_win_count_error",
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="raise",
        )

    frame["cycle"] = frame["cycle"].astype(int)

    # Float-safe candidate selection.
    frame["multiplier_key"] = (
        frame["house_environment_multiplier"].round(2)
    )

    frame = frame.loc[
        frame["multiplier_key"].isin(CANDIDATE_MULTIPLIERS)
    ].copy()

    expected_rows = (
        len(CANDIDATE_MULTIPLIERS)
        * len(SCENARIO_WEIGHTS["equal_weight"])
    )

    if len(frame) != expected_rows:
        counts = (
            frame.groupby("multiplier_key")
            .size()
            .to_dict()
        )
        raise ValueError(
            f"Expected {expected_rows} rows but found {len(frame)}. "
            f"Counts by multiplier: {counts}"
        )

    cycle_sets = (
        frame.groupby("multiplier_key")["cycle"]
        .apply(lambda values: set(values))
    )

    expected_cycles = set(
        SCENARIO_WEIGHTS["equal_weight"]
    )

    bad = cycle_sets.loc[
        cycle_sets.apply(lambda value: value != expected_cycles)
    ]

    if not bad.empty:
        raise ValueError(
            "Some multipliers do not contain the complete cycle set: "
            f"{bad.to_dict()}"
        )

    return frame


def weighted_average(
    frame: pd.DataFrame,
    value_column: str,
) -> float:
    return float(
        (
            frame[value_column]
            * frame["scenario_weight"]
        ).sum()
    )


def build_scenario_results(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for scenario, weights in SCENARIO_WEIGHTS.items():
        scenario_frame = frame.copy()

        scenario_frame["scenario_weight"] = (
            scenario_frame["cycle"].map(weights)
        )

        if scenario_frame["scenario_weight"].isna().any():
            raise ValueError(
                f"Missing cycle weight in scenario {scenario}"
            )

        for multiplier, group in scenario_frame.groupby(
            "multiplier_key",
            sort=True,
        ):
            rows.append(
                {
                    "scenario": scenario,
                    "house_environment_multiplier": float(
                        multiplier
                    ),
                    "weighted_mae": weighted_average(
                        group,
                        "mean_absolute_error",
                    ),
                    "weighted_rmse": weighted_average(
                        group,
                        "rmse",
                    ),
                    "weighted_brier": weighted_average(
                        group,
                        "brier_score",
                    ),
                    "weighted_winner_accuracy": (
                        weighted_average(
                            group,
                            "winner_accuracy",
                        )
                    ),
                    "weighted_cycle_bias": weighted_average(
                        group,
                        "mean_margin_error_dem_bias",
                    ),
                    "weighted_absolute_cycle_bias": (
                        weighted_average(
                            group.assign(
                                absolute_cycle_bias=group[
                                    "mean_margin_error_dem_bias"
                                ].abs()
                            ),
                            "absolute_cycle_bias",
                        )
                    ),
                    "weighted_seat_error": weighted_average(
                        group,
                        "expected_win_count_error",
                    ),
                    "worst_cycle_mae": float(
                        group["mean_absolute_error"].max()
                    ),
                    "worst_cycle_seat_error": float(
                        group[
                            "expected_win_count_error"
                        ].max()
                    ),
                }
            )

    results = pd.DataFrame(rows)

    # Rank each metric independently within each weighting scenario.
    for metric in LOWER_IS_BETTER:
        results[f"{metric}_rank"] = (
            results.groupby("scenario")[metric]
            .rank(method="average", ascending=True)
        )

    for metric in HIGHER_IS_BETTER:
        results[f"{metric}_rank"] = (
            results.groupby("scenario")[metric]
            .rank(method="average", ascending=False)
        )

    rank_columns = [
        f"{metric}_rank"
        for metric in LOWER_IS_BETTER + HIGHER_IS_BETTER
    ]

    results["scenario_joint_rank"] = (
        results[rank_columns].mean(axis=1)
    )

    results["scenario_joint_rank_position"] = (
        results.groupby("scenario")[
            "scenario_joint_rank"
        ]
        .rank(method="min", ascending=True)
        .astype(int)
    )

    return results.sort_values(
        [
            "scenario",
            "scenario_joint_rank",
            "weighted_mae",
        ]
    ).reset_index(drop=True)


def build_robustness_summary(
    scenario_results: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        scenario_results.groupby(
            "house_environment_multiplier",
            as_index=False,
        )
        .agg(
            average_scenario_joint_rank=(
                "scenario_joint_rank",
                "mean",
            ),
            median_scenario_joint_rank=(
                "scenario_joint_rank",
                "median",
            ),
            worst_scenario_joint_rank=(
                "scenario_joint_rank",
                "max",
            ),
            average_rank_position=(
                "scenario_joint_rank_position",
                "mean",
            ),
            worst_rank_position=(
                "scenario_joint_rank_position",
                "max",
            ),
            scenarios_won=(
                "scenario_joint_rank_position",
                lambda values: int((values == 1).sum()),
            ),
            average_weighted_mae=(
                "weighted_mae",
                "mean",
            ),
            average_weighted_brier=(
                "weighted_brier",
                "mean",
            ),
            average_weighted_seat_error=(
                "weighted_seat_error",
                "mean",
            ),
            average_weighted_absolute_bias=(
                "weighted_absolute_cycle_bias",
                "mean",
            ),
        )
    )

    return summary.sort_values(
        [
            "average_scenario_joint_rank",
            "worst_rank_position",
            "average_weighted_mae",
        ]
    ).reset_index(drop=True)


def main() -> None:
    validate_weights()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_results()

    scenario_results = build_scenario_results(frame)

    robustness = build_robustness_summary(
        scenario_results
    )

    winners = (
        scenario_results.loc[
            scenario_results[
                "scenario_joint_rank_position"
            ].eq(1)
        ]
        .sort_values("scenario")
        .reset_index(drop=True)
    )

    recommendation = robustness.iloc[[0]].copy()

    scenario_results.to_csv(
        OUTPUT_DIR
        / "house_2026_similarity_weighted_scenario_results.csv",
        index=False,
    )

    robustness.to_csv(
        OUTPUT_DIR
        / "house_2026_similarity_weighted_robustness.csv",
        index=False,
    )

    winners.to_csv(
        OUTPUT_DIR
        / "house_2026_similarity_weighted_scenario_winners.csv",
        index=False,
    )

    recommendation.to_csv(
        OUTPUT_DIR
        / "house_2026_similarity_weighted_recommendation.csv",
        index=False,
    )

    config = {
        "input_path": str(INPUT_PATH),
        "baseline_type": BASELINE_TYPE,
        "candidate_multipliers": CANDIDATE_MULTIPLIERS,
        "scenario_weights": SCENARIO_WEIGHTS,
        "lower_is_better_metrics": LOWER_IS_BETTER,
        "higher_is_better_metrics": HIGHER_IS_BETTER,
        "decision_rule": (
            "Lowest average scenario joint rank, followed by "
            "worst scenario rank position and weighted MAE."
        ),
    }

    with open(
        OUTPUT_DIR
        / "house_2026_similarity_weighted_config.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(config, handle, indent=2)

    scenario_display = [
        "scenario",
        "house_environment_multiplier",
        "weighted_mae",
        "weighted_rmse",
        "weighted_brier",
        "weighted_winner_accuracy",
        "weighted_cycle_bias",
        "weighted_absolute_cycle_bias",
        "weighted_seat_error",
        "scenario_joint_rank",
        "scenario_joint_rank_position",
    ]

    robustness_display = [
        "house_environment_multiplier",
        "average_scenario_joint_rank",
        "median_scenario_joint_rank",
        "worst_scenario_joint_rank",
        "average_rank_position",
        "worst_rank_position",
        "scenarios_won",
        "average_weighted_mae",
        "average_weighted_brier",
        "average_weighted_seat_error",
        "average_weighted_absolute_bias",
    ]

    print()
    print("2026 similarity-weighted House calibration")
    print("=" * 125)
    print(
        f"Candidate multipliers: {CANDIDATE_MULTIPLIERS}"
    )
    print()

    print("Scenario weights")
    print("=" * 125)

    weight_table = pd.DataFrame(
        SCENARIO_WEIGHTS
    ).T

    weight_table.index.name = "scenario"

    print(weight_table.to_string())
    print()

    print("Winner under each scenario")
    print("=" * 125)
    print(
        winners[scenario_display].to_string(index=False)
    )
    print()

    print("All scenario results")
    print("=" * 125)
    print(
        scenario_results[scenario_display]
        .sort_values(
            [
                "scenario",
                "scenario_joint_rank_position",
                "house_environment_multiplier",
            ]
        )
        .to_string(index=False)
    )
    print()

    print("Cross-scenario robustness ranking")
    print("=" * 125)
    print(
        robustness[robustness_display].to_string(
            index=False
        )
    )
    print()

    print("Recommended robust multiplier")
    print("=" * 125)
    print(
        recommendation[robustness_display].to_string(
            index=False
        )
    )
    print()
    print(
        "2026 similarity-weighted calibration validation: "
        "PASSED"
    )


if __name__ == "__main__":
    main()
