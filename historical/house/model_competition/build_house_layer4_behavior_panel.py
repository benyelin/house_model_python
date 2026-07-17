from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_RESULTS_PATH = (
    PROJECT_ROOT
    / "historical/house/warehouse/"
    "house_historical_results_2012_2022.csv"
)

DEFAULT_REGISTRY_PATH = (
    PROJECT_ROOT
    / "historical/house/config/"
    "house_boundary_regimes_2012_2020.csv"
)

DEFAULT_CYCLE_PLAN_PATH = (
    PROJECT_ROOT
    / "historical/house/model_competition/config/"
    "house_layer4_cycle_plan.csv"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "historical/house/model_competition/outputs/"
    "house_layer4_behavior_panel.csv"
)

DEFAULT_HISTORY_DETAIL_PATH = (
    PROJECT_ROOT
    / "historical/house/model_competition/outputs/"
    "house_layer4_behavior_panel_history_detail.csv"
)

DEFAULT_VALIDATION_PATH = (
    PROJECT_ROOT
    / "historical/house/model_competition/outputs/"
    "house_layer4_behavior_panel_validation.txt"
)


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def median_absolute_deviation(
    values: np.ndarray,
) -> float:
    if len(values) == 0:
        return np.nan

    median = float(np.median(values))

    return float(
        np.median(
            np.abs(values - median)
        )
    )


def linear_trend_metrics(
    cycles: np.ndarray,
    margins: np.ndarray,
) -> tuple[float, float, float]:
    if len(margins) < 2:
        return np.nan, np.nan, np.nan

    centered_cycles = (
        cycles.astype(float)
        - float(np.mean(cycles))
    ) / 2.0

    design = np.column_stack(
        [
            np.ones(len(centered_cycles)),
            centered_cycles,
        ]
    )

    coefficients, *_ = np.linalg.lstsq(
        design,
        margins,
        rcond=None,
    )

    predictions = design @ coefficients

    residual_rmse = float(
        np.sqrt(
            np.mean(
                np.square(
                    margins - predictions
                )
            )
        )
    )

    return (
        float(coefficients[0]),
        float(coefficients[1]),
        residual_rmse,
    )


def current_party_streak(
    winners: list[str],
) -> tuple[str | None, int]:
    if not winners:
        return None, 0

    party = winners[-1]
    length = 0

    for winner in reversed(winners):
        if winner == party:
            length += 1
        else:
            break

    return party, length


def longest_party_streak(
    winners: list[str],
    party: str,
) -> int:
    longest = 0
    current = 0

    for winner in winners:
        if winner == party:
            current += 1
            longest = max(
                longest,
                current,
            )
        else:
            current = 0

    return longest


def reliability_label(
    count: int,
) -> str:
    if count >= 4:
        return "high"
    if count == 3:
        return "medium"
    if count == 2:
        return "low"
    if count == 1:
        return "single_cycle"
    return "no_history"


def reliability_score(
    count: int,
) -> float:
    mapping = {
        0: 0.00,
        1: 0.25,
        2: 0.50,
        3: 0.75,
    }

    return mapping.get(
        count,
        1.00,
    )


def assign_boundary_regimes(
    results: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    work = results.copy()

    work["boundary_regime_id"] = (
        work["state"]
        + "_2012_2020_default"
    )

    work["boundary_regime_status"] = (
        "assumed_stable_2012_2020"
    )

    work["boundary_regime_notes"] = (
        "No configured mid-decade statewide boundary break."
    )

    for row in registry.itertuples(index=False):
        mask = (
            work["state"].eq(row.state)
            & work["cycle"].between(
                int(row.start_cycle),
                int(row.end_cycle),
            )
        )

        work.loc[
            mask,
            "boundary_regime_id",
        ] = row.regime_id

        work.loc[
            mask,
            "boundary_regime_status",
        ] = row.boundary_continuity_status

        work.loc[
            mask,
            "boundary_regime_notes",
        ] = row.notes

    return work


def target_regime_id(
    state: str,
    target_cycle: int,
    registry: pd.DataFrame,
) -> tuple[str, str, str]:
    """
    Return the boundary regime applicable to the target election.

    For 2022, all states are marked as a new post-redistricting regime.
    Label continuity with 2012-2020 histories is therefore approximate.
    """
    if target_cycle >= 2022:
        return (
            f"{state}_2022_post_redistricting",
            "post_2020_redistricting_new_regime",
            (
                "Target district uses the post-2020 redistricting map. "
                "Prior same-numbered district history is only an "
                "approximate label match without a geographic crosswalk."
            ),
        )

    matches = registry.loc[
        registry["state"].eq(state)
        & registry["start_cycle"].le(target_cycle)
        & registry["end_cycle"].ge(target_cycle)
    ]

    if not matches.empty:
        selected = matches.iloc[0]

        return (
            str(selected["regime_id"]),
            str(
                selected[
                    "boundary_continuity_status"
                ]
            ),
            str(selected["notes"]),
        )

    return (
        f"{state}_2012_2020_default",
        "assumed_stable_2012_2020",
        "No configured mid-decade statewide boundary break.",
    )


def summarize_prior_history(
    history: pd.DataFrame,
) -> dict[str, object]:
    history = history.sort_values(
        "cycle"
    )

    margins = history[
        "actual_dem_margin"
    ].to_numpy(dtype=float)

    cycles = history[
        "cycle"
    ].to_numpy(dtype=int)

    winners = [
        winner
        for winner in (
            history["actual_winner"]
            .fillna("")
            .astype(str)
            .tolist()
        )
        if winner in {"D", "R"}
    ]

    count = len(history)

    result: dict[str, object] = {
        "prior_scorable_elections": count,
        "prior_first_cycle": (
            int(cycles[0])
            if count
            else np.nan
        ),
        "prior_last_cycle": (
            int(cycles[-1])
            if count
            else np.nan
        ),
        "prior_mean_dem_margin": (
            float(np.mean(margins))
            if count
            else np.nan
        ),
        "prior_median_dem_margin": (
            float(np.median(margins))
            if count
            else np.nan
        ),
        "prior_mean_absolute_margin": (
            float(np.mean(np.abs(margins)))
            if count
            else np.nan
        ),
        "prior_margin_std": (
            float(np.std(margins, ddof=0))
            if count
            else np.nan
        ),
        "prior_margin_mad": (
            median_absolute_deviation(
                margins
            )
        ),
        "prior_margin_range": (
            float(
                np.max(margins)
                - np.min(margins)
            )
            if count
            else np.nan
        ),
        "prior_democratic_wins": (
            winners.count("D")
        ),
        "prior_republican_wins": (
            winners.count("R")
        ),
        "prior_democratic_win_rate": (
            winners.count("D")
            / len(winners)
            if winners
            else np.nan
        ),
        "prior_competitive_within_5_rate": (
            float(
                np.mean(
                    np.abs(margins) <= 5.0
                )
            )
            if count
            else np.nan
        ),
        "prior_competitive_within_10_rate": (
            float(
                np.mean(
                    np.abs(margins) <= 10.0
                )
            )
            if count
            else np.nan
        ),
        "prior_competitive_within_15_rate": (
            float(
                np.mean(
                    np.abs(margins) <= 15.0
                )
            )
            if count
            else np.nan
        ),
        "prior_reliability": (
            reliability_label(count)
        ),
        "prior_reliability_score": (
            reliability_score(count)
        ),
    }

    if len(winners) >= 2:
        result[
            "prior_party_switch_count"
        ] = int(
            sum(
                previous != current
                for previous, current in zip(
                    winners[:-1],
                    winners[1:],
                )
            )
        )
    else:
        result[
            "prior_party_switch_count"
        ] = 0

    result[
        "prior_longest_democratic_streak"
    ] = longest_party_streak(
        winners,
        "D",
    )

    result[
        "prior_longest_republican_streak"
    ] = longest_party_streak(
        winners,
        "R",
    )

    (
        current_streak_party,
        current_streak_length,
    ) = current_party_streak(
        winners
    )

    result[
        "prior_current_party_streak_party"
    ] = current_streak_party

    result[
        "prior_current_party_streak_length"
    ] = current_streak_length

    if count >= 2:
        swings = np.diff(
            margins
        )

        result.update(
            {
                "prior_consecutive_swing_count": (
                    len(swings)
                ),
                "prior_mean_swing_dem": float(
                    np.mean(swings)
                ),
                "prior_mean_absolute_swing": float(
                    np.mean(
                        np.abs(swings)
                    )
                ),
                "prior_swing_std": float(
                    np.std(
                        swings,
                        ddof=0,
                    )
                ),
                "prior_largest_absolute_swing": float(
                    np.max(
                        np.abs(swings)
                    )
                ),
                "prior_first_to_last_margin_change_dem": float(
                    margins[-1]
                    - margins[0]
                ),
            }
        )
    else:
        result.update(
            {
                "prior_consecutive_swing_count": 0,
                "prior_mean_swing_dem": np.nan,
                "prior_mean_absolute_swing": np.nan,
                "prior_swing_std": np.nan,
                "prior_largest_absolute_swing": np.nan,
                "prior_first_to_last_margin_change_dem": np.nan,
            }
        )

    (
        trend_intercept,
        trend_slope,
        trend_residual_rmse,
    ) = linear_trend_metrics(
        cycles,
        margins,
    )

    result[
        "prior_trend_intercept_dem"
    ] = trend_intercept

    result[
        "prior_trend_slope_points_per_election"
    ] = trend_slope

    result[
        "prior_trend_residual_rmse"
    ] = trend_residual_rmse

    return result


def build_panel(
    results: pd.DataFrame,
    registry: pd.DataFrame,
    cycle_plan: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    str,
]:
    required_results = {
        "cycle",
        "race_id",
        "state",
        "district",
        "actual_dem_margin",
        "actual_winner",
        "include_in_major_party_margin_scoring",
        "general_election_party_structure",
    }

    missing_results = sorted(
        required_results
        - set(results.columns)
    )

    if missing_results:
        raise ValueError(
            "Historical results are missing columns: "
            + ", ".join(
                missing_results
            )
        )

    required_plan = {
        "target_cycle",
        "history_start_cycle",
        "history_end_cycle",
        "minimum_prior_scorable_elections",
        "use_for_behavior_training",
        "use_for_retention_selection",
    }

    missing_plan = sorted(
        required_plan
        - set(cycle_plan.columns)
    )

    if missing_plan:
        raise ValueError(
            "Cycle plan is missing columns: "
            + ", ".join(
                missing_plan
            )
        )

    work = results.copy()

    work["cycle"] = pd.to_numeric(
        work["cycle"],
        errors="raise",
    ).astype(int)

    work[
        "actual_dem_margin"
    ] = pd.to_numeric(
        work["actual_dem_margin"],
        errors="coerce",
    )

    work[
        "include_in_major_party_margin_scoring"
    ] = parse_bool_series(
        work[
            "include_in_major_party_margin_scoring"
        ]
    )

    work = assign_boundary_regimes(
        results=work,
        registry=registry,
    )

    cycle_plan = cycle_plan.copy()

    cycle_plan[
        "target_cycle"
    ] = pd.to_numeric(
        cycle_plan["target_cycle"],
        errors="raise",
    ).astype(int)

    cycle_plan[
        "history_start_cycle"
    ] = pd.to_numeric(
        cycle_plan["history_start_cycle"],
        errors="raise",
    ).astype(int)

    cycle_plan[
        "history_end_cycle"
    ] = pd.to_numeric(
        cycle_plan["history_end_cycle"],
        errors="raise",
    ).astype(int)

    cycle_plan[
        "minimum_prior_scorable_elections"
    ] = pd.to_numeric(
        cycle_plan[
            "minimum_prior_scorable_elections"
        ],
        errors="raise",
    ).astype(int)

    cycle_plan[
        "use_for_behavior_training"
    ] = parse_bool_series(
        cycle_plan[
            "use_for_behavior_training"
        ]
    )

    cycle_plan[
        "use_for_retention_selection"
    ] = parse_bool_series(
        cycle_plan[
            "use_for_retention_selection"
        ]
    )

    panel_rows: list[
        dict[str, object]
    ] = []

    history_detail_rows: list[
        dict[str, object]
    ] = []

    for plan_row in cycle_plan.itertuples(
        index=False
    ):
        target_cycle = int(
            plan_row.target_cycle
        )

        target_rows = work.loc[
            work["cycle"].eq(
                target_cycle
            )
        ].copy()

        if len(target_rows) != 435:
            raise ValueError(
                f"Expected 435 target rows for {target_cycle}; "
                f"found {len(target_rows)}."
            )

        for target in target_rows.itertuples(
            index=False
        ):
            (
                target_boundary_regime_id,
                target_boundary_regime_status,
                target_boundary_regime_notes,
            ) = target_regime_id(
                state=str(target.state),
                target_cycle=target_cycle,
                registry=registry,
            )

            candidate_history = work.loc[
                work["race_id"].eq(
                    target.race_id
                )
                & work["cycle"].between(
                    int(
                        plan_row.history_start_cycle
                    ),
                    int(
                        plan_row.history_end_cycle
                    ),
                )
                & work["cycle"].lt(
                    target_cycle
                )
                & work[
                    "include_in_major_party_margin_scoring"
                ]
                & work[
                    "actual_dem_margin"
                ].notna()
            ].copy()

            if target_cycle < 2022:
                same_regime_history = (
                    candidate_history.loc[
                        candidate_history[
                            "boundary_regime_id"
                        ].eq(
                            target_boundary_regime_id
                        )
                    ].copy()
                )

                history_selection_method = (
                    "same_boundary_regime_only"
                )

                target_history_continuity = (
                    "same_regime"
                )
            else:
                same_regime_history = (
                    candidate_history.copy()
                )

                history_selection_method = (
                    "same_district_label_approximation"
                )

                target_history_continuity = (
                    "approximate_across_2022_redistricting"
                )

            metrics = summarize_prior_history(
                same_regime_history
            )

            minimum_required = int(
                plan_row.minimum_prior_scorable_elections
            )

            feature_eligible = (
                bool(
                    plan_row.use_for_behavior_training
                )
                and int(
                    metrics[
                        "prior_scorable_elections"
                    ]
                )
                >= minimum_required
            )

            panel_row: dict[
                str,
                object,
            ] = {
                "target_cycle": target_cycle,
                "race_id": target.race_id,
                "state": target.state,
                "district": target.district,
                "target_actual_dem_margin": (
                    target.actual_dem_margin
                ),
                "target_actual_winner": (
                    target.actual_winner
                ),
                (
                    "target_general_election_"
                    "party_structure"
                ): (
                    target
                    .general_election_party_structure
                ),
                (
                    "target_include_in_"
                    "major_party_margin_scoring"
                ): bool(
                    target
                    .include_in_major_party_margin_scoring
                ),
                "history_start_cycle": int(
                    plan_row.history_start_cycle
                ),
                "history_end_cycle": int(
                    plan_row.history_end_cycle
                ),
                (
                    "minimum_prior_"
                    "scorable_elections"
                ): minimum_required,
                "use_for_behavior_training": bool(
                    plan_row.use_for_behavior_training
                ),
                "use_for_retention_selection": bool(
                    plan_row
                    .use_for_retention_selection
                ),
                "target_boundary_regime_id": (
                    target_boundary_regime_id
                ),
                "target_boundary_regime_status": (
                    target_boundary_regime_status
                ),
                "target_boundary_regime_notes": (
                    target_boundary_regime_notes
                ),
                "target_history_continuity": (
                    target_history_continuity
                ),
                "history_selection_method": (
                    history_selection_method
                ),
                "candidate_prior_rows": len(
                    candidate_history
                ),
                "selected_prior_rows": len(
                    same_regime_history
                ),
                "behavior_feature_eligible": (
                    feature_eligible
                ),
                "behavior_panel_version": "1.0",
            }

            panel_row.update(
                metrics
            )

            panel_rows.append(
                panel_row
            )

            for history_row in (
                same_regime_history
                .sort_values("cycle")
                .itertuples(index=False)
            ):
                history_detail_rows.append(
                    {
                        "target_cycle": (
                            target_cycle
                        ),
                        "target_race_id": (
                            target.race_id
                        ),
                        "history_cycle": (
                            history_row.cycle
                        ),
                        "history_race_id": (
                            history_row.race_id
                        ),
                        "history_state": (
                            history_row.state
                        ),
                        "history_district": (
                            history_row.district
                        ),
                        (
                            "history_actual_"
                            "dem_margin"
                        ): (
                            history_row
                            .actual_dem_margin
                        ),
                        "history_actual_winner": (
                            history_row
                            .actual_winner
                        ),
                        (
                            "history_boundary_"
                            "regime_id"
                        ): (
                            history_row
                            .boundary_regime_id
                        ),
                        (
                            "target_boundary_"
                            "regime_id"
                        ): (
                            target_boundary_regime_id
                        ),
                        (
                            "history_selection_"
                            "method"
                        ): (
                            history_selection_method
                        ),
                    }
                )

    panel = pd.DataFrame(
        panel_rows
    )

    history_detail = pd.DataFrame(
        history_detail_rows
    )

    panel = panel.sort_values(
        [
            "target_cycle",
            "state",
            "district",
        ],
        key=lambda series: (
            pd.to_numeric(
                series,
                errors="coerce",
            )
            if series.name == "district"
            else series
        ),
    ).reset_index(drop=True)

    duplicate_rows = int(
        panel.duplicated(
            [
                "target_cycle",
                "race_id",
            ]
        ).sum()
    )

    future_leakage_rows = int(
        (
            panel[
                "prior_last_cycle"
            ].notna()
            & panel[
                "prior_last_cycle"
            ].ge(
                panel[
                    "target_cycle"
                ]
            )
        ).sum()
    )

    failures: list[str] = []

    expected_rows = (
        len(cycle_plan)
        * 435
    )

    if len(panel) != expected_rows:
        failures.append(
            f"Expected {expected_rows} panel rows; "
            f"found {len(panel)}."
        )

    if duplicate_rows:
        failures.append(
            f"Found {duplicate_rows} duplicate "
            "target-cycle/race rows."
        )

    if future_leakage_rows:
        failures.append(
            f"Found {future_leakage_rows} rows using "
            "same-cycle or future history."
        )

    clean_cycles = panel.loc[
        panel[
            "target_cycle"
        ].lt(2022)
    ]

    invalid_regime_rows = int(
        clean_cycles.loc[
            clean_cycles[
                "selected_prior_rows"
            ].gt(0)
            & ~clean_cycles[
                "target_history_continuity"
            ].eq("same_regime")
        ].shape[0]
    )

    if invalid_regime_rows:
        failures.append(
            "Pre-2022 rows contain history outside "
            "the target boundary regime."
        )

    cycle_summary = (
        panel.groupby(
            "target_cycle"
        )
        .agg(
            rows=(
                "race_id",
                "size",
            ),
            unique_race_ids=(
                "race_id",
                "nunique",
            ),
            target_scorable_races=(
                (
                    "target_include_in_"
                    "major_party_margin_scoring"
                ),
                "sum",
            ),
            behavior_feature_eligible=(
                "behavior_feature_eligible",
                "sum",
            ),
            mean_prior_elections=(
                "prior_scorable_elections",
                "mean",
            ),
            districts_with_no_history=(
                "prior_scorable_elections",
                lambda series: int(
                    series.eq(0).sum()
                ),
            ),
            districts_with_two_plus_history=(
                "prior_scorable_elections",
                lambda series: int(
                    series.ge(2).sum()
                ),
            ),
        )
    )

    reliability_summary = (
        panel.groupby(
            [
                "target_cycle",
                "prior_reliability",
            ],
            dropna=False,
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    report_lines = [
        "House Layer 4A Historical Behavior Panel Validation",
        "=" * 52,
        "",
        f"Rows: {len(panel)}",
        (
            "Unique target-cycle/race rows: "
            f"{panel[['target_cycle', 'race_id']].drop_duplicates().shape[0]}"
        ),
        f"Duplicate rows: {duplicate_rows}",
        f"Future-leakage rows: {future_leakage_rows}",
        f"History-detail rows: {len(history_detail)}",
        "",
        "Cycle summary:",
        cycle_summary.to_string(
            float_format=lambda value: f"{value:.3f}"
        ),
        "",
        "Reliability by target cycle:",
        reliability_summary.to_string(),
        "",
        "History-selection methods:",
        panel[
            "history_selection_method"
        ]
        .value_counts()
        .to_string(),
        "",
        "Target continuity status:",
        panel[
            "target_history_continuity"
        ]
        .value_counts()
        .to_string(),
        "",
        "Important 2022 limitation:",
        (
            "The 2022 target districts use post-redistricting boundaries. "
            "Without a geographic crosswalk, prior behavior is attached "
            "by district label and must be treated as exploratory rather "
            "than a clean same-geography pseudo-out-of-sample test."
        ),
        "",
        "Validation status:",
    ]

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

    return (
        panel,
        history_detail,
        report,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a leakage-safe historical House behavior feature "
            "panel for Layer 4A election-cycle backtesting."
        )
    )

    parser.add_argument(
        "--results-path",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
    )

    parser.add_argument(
        "--registry-path",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
    )

    parser.add_argument(
        "--cycle-plan-path",
        type=Path,
        default=DEFAULT_CYCLE_PLAN_PATH,
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )

    parser.add_argument(
        "--history-detail-path",
        type=Path,
        default=DEFAULT_HISTORY_DETAIL_PATH,
    )

    parser.add_argument(
        "--validation-path",
        type=Path,
        default=DEFAULT_VALIDATION_PATH,
    )

    args = parser.parse_args()

    for path in [
        args.results_path,
        args.registry_path,
        args.cycle_plan_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required input: {path}"
            )

    results = pd.read_csv(
        args.results_path,
        dtype={
            "race_id": str,
            "state": str,
            "district": str,
        },
    )

    registry = pd.read_csv(
        args.registry_path,
        dtype={
            "state": str,
            "regime_id": str,
        },
    )

    cycle_plan = pd.read_csv(
        args.cycle_plan_path,
    )

    (
        panel,
        history_detail,
        report,
    ) = build_panel(
        results=results,
        registry=registry,
        cycle_plan=cycle_plan,
    )

    args.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    panel.to_csv(
        args.output_path,
        index=False,
    )

    history_detail.to_csv(
        args.history_detail_path,
        index=False,
    )

    args.validation_path.write_text(
        report
    )

    print(report)
    print()
    print(f"Wrote: {args.output_path}")
    print(f"Wrote: {args.history_detail_path}")
    print(f"Wrote: {args.validation_path}")


if __name__ == "__main__":
    main()
