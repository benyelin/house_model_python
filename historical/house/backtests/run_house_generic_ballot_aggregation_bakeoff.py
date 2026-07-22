"""Bake off historical generic-ballot aggregation specifications.

The experiment evaluates Election Day generic-ballot estimates against the
actual national Democratic two-party House vote margin.

Scored cycles:
    2018, 2020, 2022

Unscored production-readiness cycle:
    2024

The primary downstream forecast input is:
    shared_environment_margin_dem = 0.90 * generic_ballot_margin_dem

The experiment intentionally uses one aggregated observation per poll. Treating
multiple questions from one poll as independent observations is retained in the
aggregation engine for diagnostics, but is not considered a primary production
candidate here.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any, get_args

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

AGGREGATION_DIR = (
    ROOT
    / "historical"
    / "common"
    / "polling"
    / "aggregation"
)

sys.path.insert(
    0,
    str(AGGREGATION_DIR),
)

from generic_ballot_aggregation import (  # noqa: E402
    DuplicateMode,
    GenericBallotAggregationSpec,
    PartisanMode,
    PopulationMode,
    QuestionSelectionMode,
    RecencyMode,
    aggregate_generic_ballot_snapshot,
    select_snapshot,
)


SNAPSHOT_PATH = (
    ROOT
    / "historical"
    / "common"
    / "polling"
    / "snapshots"
    / "generic_ballot_polling_snapshots.csv"
)

TARGET_PATH = (
    ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "environment_signal_decomposition"
    / "house_environment_national_signal.csv"
)

OUTPUT_DIR = (
    ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "generic_ballot_aggregation_bakeoff"
)

SCORED_CYCLES = [
    2018,
    2020,
    2022,
]

ALL_CYCLES = SCORED_CYCLES + [
    2024,
]

ENVIRONMENT_COEFFICIENT = 0.90


def literal_values(annotation: Any) -> tuple[Any, ...]:
    """Return Literal values from a type alias."""

    return tuple(get_args(annotation))


def require_modes(
    *,
    type_name: str,
    available: tuple[Any, ...],
    required: list[Any],
) -> list[Any]:
    """Ensure the engine supports each mode required by the experiment."""

    missing = [
        value
        for value in required
        if value not in available
    ]

    if missing:
        raise ValueError(
            f"{type_name} is missing required values: "
            f"{missing}. Available values: {available}"
        )

    return required


def specification_id(
    *,
    lookback_days: int,
    recency_mode: str,
    recency_half_life_days: float | None,
    question_selection_mode: str,
    sample_size_weighting: bool,
    pollster_quality_weighting: bool,
    partisan_mode: str,
) -> str:
    """Construct a compact, stable specification identifier."""

    if recency_mode == "equal":
        recency_label = "equal"
    else:
        recency_label = (
            f"hl{int(recency_half_life_days)}"
        )

    return "__".join(
        [
            f"lb{lookback_days}",
            recency_label,
            f"q_{question_selection_mode}",
            (
                "sample_sqrt"
                if sample_size_weighting
                else "sample_equal"
            ),
            (
                "quality_on"
                if pollster_quality_weighting
                else "quality_off"
            ),
            f"partisan_{partisan_mode}",
            "poll_level",
        ]
    )


def complexity_score(
    *,
    recency_mode: str,
    question_selection_mode: str,
    sample_size_weighting: bool,
    pollster_quality_weighting: bool,
    partisan_mode: str,
) -> int:
    """Count methodological choices beyond a simple poll-level average."""

    score = 0

    if recency_mode != "equal":
        score += 1

    if question_selection_mode != "all":
        score += 1

    if sample_size_weighting:
        score += 1

    if pollster_quality_weighting:
        score += 1

    if partisan_mode != "include":
        score += 1

    return score


def build_specification_grid() -> pd.DataFrame:
    """Build the primary aggregation grid."""

    available_recency = literal_values(
        RecencyMode
    )
    available_population = literal_values(
        PopulationMode
    )
    available_partisan = literal_values(
        PartisanMode
    )
    available_question_selection = literal_values(
        QuestionSelectionMode
    )
    available_duplicate = literal_values(
        DuplicateMode
    )

    print("Engine-supported modes")
    print("-" * 88)
    print(f"Recency: {available_recency}")
    print(f"Population: {available_population}")
    print(f"Partisan: {available_partisan}")
    print(
        "Question selection: "
        f"{available_question_selection}"
    )
    print(f"Duplicate: {available_duplicate}")

    require_modes(
        type_name="RecencyMode",
        available=available_recency,
        required=[
            "equal",
            "half_life",
        ],
    )

    require_modes(
        type_name="PopulationMode",
        available=available_population,
        required=[
            "all",
        ],
    )

    require_modes(
        type_name="PartisanMode",
        available=available_partisan,
        required=[
            "include",
            "downweight",
            "exclude",
        ],
    )

    require_modes(
        type_name="QuestionSelectionMode",
        available=available_question_selection,
        required=[
            "all",
            "prefer_lv",
            "prefer_rv",
            "largest_sample",
        ],
    )

    require_modes(
        type_name="DuplicateMode",
        available=available_duplicate,
        required=[
            "poll",
        ],
    )

    lookbacks = [
        21,
        30,
        45,
        60,
        90,
    ]

    recency_configurations = [
        (
            "equal",
            None,
        ),
        (
            "half_life",
            14.0,
        ),
        (
            "half_life",
            21.0,
        ),
        (
            "half_life",
            30.0,
        ),
    ]

    question_selection_modes = [
        "all",
        "prefer_lv",
        "prefer_rv",
        "largest_sample",
    ]

    sample_size_options = [
        False,
        True,
    ]

    pollster_quality_options = [
        False,
        True,
    ]

    partisan_modes = [
        "include",
        "downweight",
        "exclude",
    ]

    rows: list[dict[str, Any]] = []

    for (
        lookback_days,
        recency_configuration,
        question_selection_mode,
        sample_size_weighting,
        pollster_quality_weighting,
        partisan_mode,
    ) in product(
        lookbacks,
        recency_configurations,
        question_selection_modes,
        sample_size_options,
        pollster_quality_options,
        partisan_modes,
    ):
        (
            recency_mode,
            recency_half_life_days,
        ) = recency_configuration

        spec_id = specification_id(
            lookback_days=lookback_days,
            recency_mode=recency_mode,
            recency_half_life_days=(
                recency_half_life_days
            ),
            question_selection_mode=(
                question_selection_mode
            ),
            sample_size_weighting=(
                sample_size_weighting
            ),
            pollster_quality_weighting=(
                pollster_quality_weighting
            ),
            partisan_mode=partisan_mode,
        )

        rows.append(
            {
                "specification_id": spec_id,
                "lookback_days": lookback_days,
                "recency_mode": recency_mode,
                "recency_half_life_days": (
                    recency_half_life_days
                ),
                "population_mode": "all",
                "question_selection_mode": (
                    question_selection_mode
                ),
                "duplicate_mode": "poll",
                "sample_size_weighting": (
                    sample_size_weighting
                ),
                "pollster_quality_weighting": (
                    pollster_quality_weighting
                ),
                "partisan_mode": partisan_mode,
                "partisan_weight": 0.5,
                "complexity_score": complexity_score(
                    recency_mode=recency_mode,
                    question_selection_mode=(
                        question_selection_mode
                    ),
                    sample_size_weighting=(
                        sample_size_weighting
                    ),
                    pollster_quality_weighting=(
                        pollster_quality_weighting
                    ),
                    partisan_mode=partisan_mode,
                ),
            }
        )

    grid = pd.DataFrame(rows)

    if grid["specification_id"].duplicated().any():
        duplicates = grid.loc[
            grid["specification_id"].duplicated(
                keep=False
            ),
            "specification_id",
        ].tolist()

        raise AssertionError(
            "Duplicate specification IDs: "
            f"{duplicates[:10]}"
        )

    return grid


def load_targets() -> pd.DataFrame:
    """Load validated national House vote targets."""

    target = pd.read_csv(
        TARGET_PATH,
        low_memory=False,
    )

    required_columns = {
        "cycle",
        "actual_national_house_margin_dem",
    }

    missing = required_columns - set(
        target.columns
    )

    if missing:
        raise ValueError(
            "Target file is missing columns: "
            f"{sorted(missing)}"
        )

    target["cycle"] = pd.to_numeric(
        target["cycle"],
        errors="raise",
    ).astype(int)

    target[
        "actual_national_house_margin_dem"
    ] = pd.to_numeric(
        target[
            "actual_national_house_margin_dem"
        ],
        errors="raise",
    )

    scored_target = target[
        target["cycle"].isin(SCORED_CYCLES)
    ][
        [
            "cycle",
            "actual_national_house_margin_dem",
        ]
    ].copy()

    missing_cycles = (
        set(SCORED_CYCLES)
        - set(scored_target["cycle"])
    )

    if missing_cycles:
        raise ValueError(
            "Missing target cycles: "
            f"{sorted(missing_cycles)}"
        )

    return scored_target


def build_engine_spec(
    row: pd.Series,
) -> GenericBallotAggregationSpec:
    """Convert a grid row to the engine's dataclass."""

    half_life = row[
        "recency_half_life_days"
    ]

    if pd.isna(half_life):
        half_life = None
    else:
        half_life = float(half_life)

    return GenericBallotAggregationSpec(
        lookback_days=int(
            row["lookback_days"]
        ),
        recency_mode=str(
            row["recency_mode"]
        ),
        recency_half_life_days=half_life,
        population_mode=str(
            row["population_mode"]
        ),
        sample_size_weighting=bool(
            row["sample_size_weighting"]
        ),
        pollster_quality_weighting=bool(
            row["pollster_quality_weighting"]
        ),
        partisan_mode=str(
            row["partisan_mode"]
        ),
        partisan_weight=float(
            row["partisan_weight"]
        ),
        question_selection_mode=str(
            row["question_selection_mode"]
        ),
        duplicate_mode=str(
            row["duplicate_mode"]
        ),
    )


def run_grid(
    *,
    snapshots: pd.DataFrame,
    grid: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all specifications for each available cycle."""

    cycle_snapshots = {
        cycle: select_snapshot(
            snapshots,
            cycle=cycle,
            snapshot_days_before_election=0,
        )
        for cycle in ALL_CYCLES
    }

    result_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    total_runs = len(grid) * len(
        ALL_CYCLES
    )

    completed_runs = 0

    for grid_index, row in grid.iterrows():
        spec = build_engine_spec(row)

        for cycle in ALL_CYCLES:
            completed_runs += 1

            try:
                result = (
                    aggregate_generic_ballot_snapshot(
                        cycle_snapshots[cycle],
                        spec,
                    )
                )
            except Exception as exc:
                failure_rows.append(
                    {
                        "specification_id": row[
                            "specification_id"
                        ],
                        "cycle": cycle,
                        "exception_type": type(
                            exc
                        ).__name__,
                        "exception_message": str(
                            exc
                        ),
                    }
                )
                continue

            result_rows.append(
                {
                    "specification_id": row[
                        "specification_id"
                    ],
                    "cycle": cycle,
                    "estimate_margin_dem": float(
                        result.estimate_margin_dem
                    ),
                    "environment_margin_dem": (
                        ENVIRONMENT_COEFFICIENT
                        * float(
                            result.estimate_margin_dem
                        )
                    ),
                    "input_questions": int(
                        result.input_question_rows
                    ),
                    "retained_questions": int(
                        result.retained_question_rows
                    ),
                    "aggregated_rows": int(
                        result.aggregated_rows
                    ),
                    "unique_polls": int(
                        result.unique_polls
                    ),
                    "unique_pollsters": int(
                        result.unique_pollsters
                    ),
                    "effective_sample_size": float(
                        result.effective_sample_size
                    ),
                    "weighted_mean_poll_age_days": (
                        float(
                            result
                            .weighted_mean_poll_age_days
                        )
                    ),
                }
            )

        if (
            (grid_index + 1) % 100 == 0
            or grid_index + 1 == len(grid)
        ):
            print(
                f"Completed specifications: "
                f"{grid_index + 1:,}/{len(grid):,} "
                f"({completed_runs:,}/{total_runs:,} "
                "cycle-runs)"
            )

    results = pd.DataFrame(result_rows)
    failures = pd.DataFrame(failure_rows)

    return results, failures


def add_scoring(
    *,
    results: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """Add raw and downstream errors for scored cycles."""

    scored = results[
        results["cycle"].isin(
            SCORED_CYCLES
        )
    ].merge(
        targets,
        on="cycle",
        how="left",
        validate="many_to_one",
    )

    if scored[
        "actual_national_house_margin_dem"
    ].isna().any():
        raise AssertionError(
            "Scored rows are missing actual targets."
        )

    scored["raw_error_dem"] = (
        scored["estimate_margin_dem"]
        - scored[
            "actual_national_house_margin_dem"
        ]
    )

    scored["raw_absolute_error"] = (
        scored["raw_error_dem"].abs()
    )

    scored["raw_squared_error"] = (
        scored["raw_error_dem"] ** 2
    )

    scored["environment_error_dem"] = (
        scored["environment_margin_dem"]
        - scored[
            "actual_national_house_margin_dem"
        ]
    )

    scored[
        "environment_absolute_error"
    ] = scored[
        "environment_error_dem"
    ].abs()

    scored[
        "environment_squared_error"
    ] = scored[
        "environment_error_dem"
    ] ** 2

    return scored


def summarize_specifications(
    *,
    scored: pd.DataFrame,
    grid: pd.DataFrame,
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Create one performance row per specification."""

    summary = (
        scored.groupby(
            "specification_id",
            as_index=False,
        )
        .agg(
            scored_cycles=(
                "cycle",
                "nunique",
            ),
            raw_mae=(
                "raw_absolute_error",
                "mean",
            ),
            raw_rmse=(
                "raw_squared_error",
                lambda values: float(
                    np.sqrt(
                        np.mean(values)
                    )
                ),
            ),
            raw_mean_error=(
                "raw_error_dem",
                "mean",
            ),
            raw_max_absolute_error=(
                "raw_absolute_error",
                "max",
            ),
            environment_mae=(
                "environment_absolute_error",
                "mean",
            ),
            environment_rmse=(
                "environment_squared_error",
                lambda values: float(
                    np.sqrt(
                        np.mean(values)
                    )
                ),
            ),
            environment_mean_error=(
                "environment_error_dem",
                "mean",
            ),
            environment_max_absolute_error=(
                "environment_absolute_error",
                "max",
            ),
            mean_effective_sample_size=(
                "effective_sample_size",
                "mean",
            ),
            mean_weighted_poll_age_days=(
                "weighted_mean_poll_age_days",
                "mean",
            ),
            minimum_unique_polls=(
                "unique_polls",
                "min",
            ),
        )
    )

    summary = summary.merge(
        grid,
        on="specification_id",
        how="left",
        validate="one_to_one",
    )

    production_2024 = results[
        results["cycle"].eq(2024)
    ][
        [
            "specification_id",
            "estimate_margin_dem",
            "environment_margin_dem",
            "unique_polls",
            "unique_pollsters",
            "effective_sample_size",
            "weighted_mean_poll_age_days",
        ]
    ].rename(
        columns={
            "estimate_margin_dem": (
                "estimate_2024_margin_dem"
            ),
            "environment_margin_dem": (
                "environment_2024_margin_dem"
            ),
            "unique_polls": (
                "unique_polls_2024"
            ),
            "unique_pollsters": (
                "unique_pollsters_2024"
            ),
            "effective_sample_size": (
                "effective_sample_size_2024"
            ),
            "weighted_mean_poll_age_days": (
                "weighted_mean_poll_age_days_2024"
            ),
        }
    )

    summary = summary.merge(
        production_2024,
        on="specification_id",
        how="left",
        validate="one_to_one",
    )

    summary["simplicity_adjusted_score"] = (
        summary["environment_mae"]
        + 0.03 * summary["complexity_score"]
    )

    summary = summary.sort_values(
        [
            "environment_mae",
            "environment_max_absolute_error",
            "raw_mae",
            "complexity_score",
            "specification_id",
        ],
        ascending=True,
    ).reset_index(drop=True)

    summary["environment_rank"] = (
        np.arange(
            1,
            len(summary) + 1,
        )
    )

    simplicity_order = summary.sort_values(
        [
            "simplicity_adjusted_score",
            "environment_max_absolute_error",
            "raw_mae",
            "complexity_score",
            "specification_id",
        ]
    )[
        "specification_id"
    ].tolist()

    simplicity_rank = {
        specification_id: rank
        for rank, specification_id in enumerate(
            simplicity_order,
            start=1,
        )
    }

    summary["simplicity_adjusted_rank"] = (
        summary["specification_id"].map(
            simplicity_rank
        )
    )

    return summary


def add_pareto_flag(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Flag methods not dominated on accuracy, robustness, and complexity."""

    output = summary.copy()

    metrics = [
        "environment_mae",
        "environment_max_absolute_error",
        "complexity_score",
    ]

    values = output[metrics].to_numpy(
        dtype=float
    )

    pareto = np.ones(
        len(output),
        dtype=bool,
    )

    for index in range(len(output)):
        candidate = values[index]

        dominated = np.any(
            np.all(
                values <= candidate,
                axis=1,
            )
            & np.any(
                values < candidate,
                axis=1,
            )
        )

        pareto[index] = not dominated

    output["pareto_optimal"] = pareto

    return output


def run_leave_one_cycle_out(
    *,
    scored: pd.DataFrame,
    grid: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select a method on two cycles and evaluate it on the held-out cycle."""

    selection_rows: list[dict[str, Any]] = []
    candidate_rows: list[pd.DataFrame] = []

    complexity = grid[
        [
            "specification_id",
            "complexity_score",
        ]
    ]

    for held_out_cycle in SCORED_CYCLES:
        training = scored[
            scored["cycle"].ne(
                held_out_cycle
            )
        ]

        held_out = scored[
            scored["cycle"].eq(
                held_out_cycle
            )
        ]

        training_summary = (
            training.groupby(
                "specification_id",
                as_index=False,
            )
            .agg(
                training_environment_mae=(
                    "environment_absolute_error",
                    "mean",
                ),
                training_environment_max_error=(
                    "environment_absolute_error",
                    "max",
                ),
                training_raw_mae=(
                    "raw_absolute_error",
                    "mean",
                ),
            )
            .merge(
                complexity,
                on="specification_id",
                how="left",
                validate="one_to_one",
            )
        )

        training_summary[
            "training_simplicity_adjusted_score"
        ] = (
            training_summary[
                "training_environment_mae"
            ]
            + 0.03
            * training_summary[
                "complexity_score"
            ]
        )

        training_summary["held_out_cycle"] = (
            held_out_cycle
        )

        candidate_rows.append(
            training_summary
        )

        selected = (
            training_summary.sort_values(
                [
                    "training_simplicity_adjusted_score",
                    "training_environment_max_error",
                    "training_raw_mae",
                    "complexity_score",
                    "specification_id",
                ]
            )
            .iloc[0]
        )

        held_out_selected = held_out[
            held_out["specification_id"].eq(
                selected["specification_id"]
            )
        ]

        if len(held_out_selected) != 1:
            raise AssertionError(
                "Expected exactly one held-out row "
                f"for {held_out_cycle}."
            )

        evaluation = held_out_selected.iloc[0]

        selection_rows.append(
            {
                "held_out_cycle": held_out_cycle,
                "selected_specification_id": (
                    selected[
                        "specification_id"
                    ]
                ),
                "training_environment_mae": (
                    selected[
                        "training_environment_mae"
                    ]
                ),
                "training_environment_max_error": (
                    selected[
                        "training_environment_max_error"
                    ]
                ),
                "training_raw_mae": (
                    selected[
                        "training_raw_mae"
                    ]
                ),
                "complexity_score": (
                    selected[
                        "complexity_score"
                    ]
                ),
                "held_out_estimate_margin_dem": (
                    evaluation[
                        "estimate_margin_dem"
                    ]
                ),
                "held_out_environment_margin_dem": (
                    evaluation[
                        "environment_margin_dem"
                    ]
                ),
                "held_out_actual_margin_dem": (
                    evaluation[
                        "actual_national_house_margin_dem"
                    ]
                ),
                "held_out_raw_error_dem": (
                    evaluation[
                        "raw_error_dem"
                    ]
                ),
                "held_out_raw_absolute_error": (
                    evaluation[
                        "raw_absolute_error"
                    ]
                ),
                "held_out_environment_error_dem": (
                    evaluation[
                        "environment_error_dem"
                    ]
                ),
                (
                    "held_out_environment_"
                    "absolute_error"
                ): evaluation[
                    "environment_absolute_error"
                ],
            }
        )

    selections = pd.DataFrame(
        selection_rows
    )

    candidates = pd.concat(
        candidate_rows,
        ignore_index=True,
    )

    return selections, candidates


def summarize_leave_one_cycle_out(
    selections: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize held-out performance across the three folds."""

    return pd.DataFrame(
        [
            {
                "held_out_cycles": len(
                    selections
                ),
                "loo_raw_mae": selections[
                    "held_out_raw_absolute_error"
                ].mean(),
                "loo_raw_rmse": float(
                    np.sqrt(
                        np.mean(
                            selections[
                                "held_out_raw_error_dem"
                            ]
                            ** 2
                        )
                    )
                ),
                "loo_environment_mae": selections[
                    (
                        "held_out_environment_"
                        "absolute_error"
                    )
                ].mean(),
                "loo_environment_rmse": float(
                    np.sqrt(
                        np.mean(
                            selections[
                                "held_out_environment_error_dem"
                            ]
                            ** 2
                        )
                    )
                ),
                "distinct_selected_specifications": (
                    selections[
                        "selected_specification_id"
                    ].nunique()
                ),
            }
        ]
    )


def main() -> None:
    print(
        "House Generic Ballot Aggregation Bakeoff"
    )
    print("=" * 88)
    print(
        "Scored cycles: "
        + ", ".join(
            str(cycle)
            for cycle in SCORED_CYCLES
        )
    )
    print(
        "Unscored readiness cycle: 2024"
    )
    print(
        "Environment coefficient: "
        f"{ENVIRONMENT_COEFFICIENT:.2f}"
    )
    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not SNAPSHOT_PATH.exists():
        raise FileNotFoundError(
            SNAPSHOT_PATH
        )

    if not TARGET_PATH.exists():
        raise FileNotFoundError(
            TARGET_PATH
        )

    snapshots = pd.read_csv(
        SNAPSHOT_PATH,
        low_memory=False,
    )

    targets = load_targets()
    grid = build_specification_grid()

    print()
    print("Experiment grid")
    print("-" * 88)
    print(
        f"Specifications: {len(grid):,}"
    )
    print(
        "Expected cycle-runs: "
        f"{len(grid) * len(ALL_CYCLES):,}"
    )
    print()

    grid.to_csv(
        OUTPUT_DIR
        / "aggregation_specification_grid.csv",
        index=False,
    )

    results, failures = run_grid(
        snapshots=snapshots,
        grid=grid,
    )

    results.to_csv(
        OUTPUT_DIR
        / "aggregation_cycle_results.csv",
        index=False,
    )

    failures.to_csv(
        OUTPUT_DIR
        / "aggregation_failures.csv",
        index=False,
    )

    print()
    print("Execution results")
    print("-" * 88)
    print(
        f"Successful cycle-runs: "
        f"{len(results):,}"
    )
    print(
        f"Failed cycle-runs: "
        f"{len(failures):,}"
    )

    if not failures.empty:
        print()
        print("Failure summary")
        print(
            failures.groupby(
                [
                    "exception_type",
                    "exception_message",
                ],
                dropna=False,
            )
            .size()
            .reset_index(
                name="count"
            )
            .sort_values(
                "count",
                ascending=False,
            )
            .head(20)
            .to_string(
                index=False
            )
        )

        raise RuntimeError(
            "At least one aggregation run failed. "
            "Inspect aggregation_failures.csv."
        )

    expected_results = (
        len(grid) * len(ALL_CYCLES)
    )

    if len(results) != expected_results:
        raise AssertionError(
            f"Expected {expected_results:,} "
            f"results, found {len(results):,}."
        )

    scored = add_scoring(
        results=results,
        targets=targets,
    )

    scored.to_csv(
        OUTPUT_DIR
        / "aggregation_scored_cycle_results.csv",
        index=False,
    )

    summary = summarize_specifications(
        scored=scored,
        grid=grid,
        results=results,
    )

    summary = add_pareto_flag(
        summary
    )

    summary.to_csv(
        OUTPUT_DIR
        / "aggregation_bakeoff_summary.csv",
        index=False,
    )

    pareto = summary[
        summary["pareto_optimal"]
    ].copy()

    pareto.to_csv(
        OUTPUT_DIR
        / "aggregation_pareto_set.csv",
        index=False,
    )

    loo_selections, loo_candidates = (
        run_leave_one_cycle_out(
            scored=scored,
            grid=grid,
        )
    )

    loo_summary = (
        summarize_leave_one_cycle_out(
            loo_selections
        )
    )

    loo_selections.to_csv(
        OUTPUT_DIR
        / "aggregation_leave_one_cycle_out.csv",
        index=False,
    )

    loo_candidates.to_csv(
        OUTPUT_DIR
        / "aggregation_leave_one_cycle_out_candidates.csv",
        index=False,
    )

    loo_summary.to_csv(
        OUTPUT_DIR
        / "aggregation_leave_one_cycle_out_summary.csv",
        index=False,
    )

    top_environment = summary.head(
        20
    )

    top_simple = (
        summary.sort_values(
            [
                "simplicity_adjusted_rank",
                "environment_max_absolute_error",
                "raw_mae",
            ]
        )
        .head(20)
    )

    print()
    print("Top 20 by downstream environment MAE")
    print("-" * 88)

    display_columns = [
        "environment_rank",
        "specification_id",
        "environment_mae",
        "environment_rmse",
        "environment_max_absolute_error",
        "raw_mae",
        "complexity_score",
        "estimate_2024_margin_dem",
        "environment_2024_margin_dem",
        "minimum_unique_polls",
        "pareto_optimal",
    ]

    print(
        top_environment[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print(
        "Top 20 with simplicity adjustment"
    )
    print("-" * 88)

    simple_display_columns = [
        "simplicity_adjusted_rank",
        "specification_id",
        "simplicity_adjusted_score",
        "environment_mae",
        "environment_max_absolute_error",
        "raw_mae",
        "complexity_score",
        "estimate_2024_margin_dem",
        "environment_2024_margin_dem",
        "pareto_optimal",
    ]

    print(
        top_simple[
            simple_display_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print("Pareto set")
    print("-" * 88)

    print(
        pareto[
            [
                "specification_id",
                "environment_mae",
                "environment_max_absolute_error",
                "raw_mae",
                "complexity_score",
                "estimate_2024_margin_dem",
                "environment_2024_margin_dem",
            ]
        ]
        .sort_values(
            [
                "complexity_score",
                "environment_mae",
                "environment_max_absolute_error",
            ]
        )
        .to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print("Leave-one-cycle-out selections")
    print("-" * 88)

    print(
        loo_selections.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print("Leave-one-cycle-out summary")
    print("-" * 88)

    print(
        loo_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print("Outputs")
    print("-" * 88)

    for path in sorted(
        OUTPUT_DIR.glob("*.csv")
    ):
        print(
            path.relative_to(ROOT)
        )

    print()
    print("Validation")
    print("-" * 88)

    if summary["scored_cycles"].ne(
        len(SCORED_CYCLES)
    ).any():
        raise AssertionError(
            "At least one specification does not "
            "have all scored cycles."
        )

    if not np.isfinite(
        summary[
            [
                "raw_mae",
                "raw_rmse",
                "environment_mae",
                "environment_rmse",
                "estimate_2024_margin_dem",
                "environment_2024_margin_dem",
            ]
        ].to_numpy(
            dtype=float
        )
    ).all():
        raise AssertionError(
            "Non-finite summary values found."
        )

    if pareto.empty:
        raise AssertionError(
            "Pareto set is empty."
        )

    if len(loo_selections) != len(
        SCORED_CYCLES
    ):
        raise AssertionError(
            "Unexpected number of LOO folds."
        )

    print(
        "All specifications have three "
        "scored cycles: PASSED"
    )
    print(
        "All summary metrics finite: PASSED"
    )
    print(
        "Pareto set nonempty: PASSED"
    )
    print(
        "Leave-one-cycle-out folds complete: "
        "PASSED"
    )
    print()
    print(
        "Generic ballot aggregation bakeoff "
        "PASSED."
    )


if __name__ == "__main__":
    main()
