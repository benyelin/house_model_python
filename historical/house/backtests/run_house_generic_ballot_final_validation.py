#!/usr/bin/env python3
"""Final validation of the House generic-ballot aggregation methodology.

Studies
-------
1. Focused refinement around the strongest aggregation family.
2. Partisan-poll inventory, influence, and sensitivity audit.
3. Largest-sample question-selection audit.

This script does not modify production code.
"""

from __future__ import annotations

import math
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import numpy as np
import pandas as pd

from historical.house.backtests import (
    run_house_generic_ballot_aggregation_bakeoff as bakeoff,
)

OUTPUT_DIR = (
    ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "generic_ballot_final_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SCORED_CYCLES = [2018, 2020, 2022]
AUDIT_CYCLES = [2018, 2020, 2022, 2024]

LOOKBACK_VALUES = [18, 21, 24, 27, 30]

# Equal plus seven exponential-decay alternatives:
# 5 lookbacks × 8 recency choices × 2 quality choices = 80 specs.
HALF_LIFE_VALUES = [7, 10, 14, 18, 21, 28, 35]

ENVIRONMENT_COEFFICIENT = 0.90
SIMPLICITY_PENALTY = 0.03

FIXED_QUESTION_SELECTION = "largest_sample"
FIXED_SAMPLE_SIZE_WEIGHTING = True
FIXED_PARTISAN_MODE = "include"
FIXED_DUPLICATE_MODE = "poll"
FIXED_POPULATION_MODE = "all"

REFERENCE_SAMPLE_SIZE = 600.0

POPULATION_PRIORITY = {
    "lv": 0,
    "likely voters": 0,
    "likely voter": 0,
    "rv": 1,
    "registered voters": 1,
    "registered voter": 1,
    "v": 2,
    "voters": 2,
    "voter": 2,
    "a": 3,
    "adults": 3,
    "adult": 3,
}


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_population(value: object) -> str:
    text = normalize_text(value).lower()

    if text in {"lv", "likely voters", "likely voter"}:
        return "LV"

    if text in {
        "rv",
        "registered voters",
        "registered voter",
    }:
        return "RV"

    if text in {"v", "voters", "voter"}:
        return "V"

    if text in {"a", "adults", "adult"}:
        return "A"

    return "UNKNOWN"


def population_priority(value: object) -> int:
    text = normalize_text(value).lower()
    return POPULATION_PRIORITY.get(text, 99)


def normalize_partisan(value: object) -> str:
    text = normalize_text(value).upper()

    if text in {"DEM", "D", "DEMOCRATIC"}:
        return "DEM"

    if text in {"REP", "R", "REPUBLICAN", "GOP"}:
        return "REP"

    return "INDEPENDENT"


def normalize_internal(value: object) -> bool:
    if isinstance(value, bool):
        return value

    text = normalize_text(value).lower()

    return text in {
        "true",
        "1",
        "yes",
        "y",
        "internal",
    }


def safe_numeric(
    series: pd.Series,
    *,
    default: float = np.nan,
) -> pd.Series:
    converted = pd.to_numeric(
        series,
        errors="coerce",
    )

    if not math.isnan(default):
        converted = converted.fillna(default)

    return converted


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    snapshots = pd.read_csv(
        bakeoff.SNAPSHOT_PATH,
        low_memory=False,
    )

    targets = bakeoff.load_targets()

    snapshots["cycle"] = pd.to_numeric(
        snapshots["cycle"],
        errors="coerce",
    ).astype("Int64")

    snapshots["poll_age_days"] = safe_numeric(
        snapshots["poll_age_days"],
    )

    snapshots["sample_size"] = safe_numeric(
        snapshots["sample_size"],
    )

    snapshots["margin_dem"] = safe_numeric(
        snapshots["margin_dem"],
    )

    snapshots["numeric_grade"] = safe_numeric(
        snapshots["numeric_grade"],
    )

    return snapshots, targets


def build_spec(
    *,
    lookback_days: int,
    recency_mode: str,
    half_life_days: float | None,
    quality_weighting: bool,
    partisan_mode: str = FIXED_PARTISAN_MODE,
) -> bakeoff.GenericBallotAggregationSpec:
    return bakeoff.GenericBallotAggregationSpec(
        lookback_days=lookback_days,
        recency_mode=recency_mode,
        recency_half_life_days=half_life_days,
        population_mode=FIXED_POPULATION_MODE,
        sample_size_weighting=(
            FIXED_SAMPLE_SIZE_WEIGHTING
        ),
        pollster_quality_weighting=quality_weighting,
        partisan_mode=partisan_mode,
        partisan_weight=0.5,
        question_selection_mode=(
            FIXED_QUESTION_SELECTION
        ),
        duplicate_mode=FIXED_DUPLICATE_MODE,
    )


def focused_specification_id(
    *,
    lookback_days: int,
    recency_mode: str,
    half_life_days: float | None,
    quality_weighting: bool,
) -> str:
    if recency_mode == "equal":
        recency_label = "equal"
    else:
        recency_label = (
            f"hl{int(half_life_days)}"
        )

    quality_label = (
        "quality_on"
        if quality_weighting
        else "quality_off"
    )

    return "__".join(
        [
            f"lb{lookback_days}",
            recency_label,
            "q_largest_sample",
            "sample_sqrt",
            quality_label,
            "partisan_include",
            "poll_level",
        ]
    )


def run_focused_refinement(
    *,
    snapshots: pd.DataFrame,
    targets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print()
    print("STUDY 1 — Focused aggregation refinement")
    print("=" * 110)

    target_lookup = (
        targets.set_index("cycle")[
            "actual_national_house_margin_dem"
        ]
        .astype(float)
        .to_dict()
    )

    cycle_snapshots = {
        cycle: bakeoff.select_snapshot(
            snapshots,
            cycle=cycle,
            snapshot_days_before_election=0,
        )
        for cycle in AUDIT_CYCLES
    }

    rows: list[dict[str, object]] = []
    expected_specs = 0

    for lookback_days in LOOKBACK_VALUES:
        recency_settings = [
            ("equal", None),
            *[
                ("half_life", float(value))
                for value in HALF_LIFE_VALUES
            ],
        ]

        for recency_mode, half_life_days in (
            recency_settings
        ):
            for quality_weighting in [False, True]:
                expected_specs += 1

                spec_id = focused_specification_id(
                    lookback_days=lookback_days,
                    recency_mode=recency_mode,
                    half_life_days=half_life_days,
                    quality_weighting=(
                        quality_weighting
                    ),
                )

                spec = build_spec(
                    lookback_days=lookback_days,
                    recency_mode=recency_mode,
                    half_life_days=half_life_days,
                    quality_weighting=(
                        quality_weighting
                    ),
                )

                for cycle in AUDIT_CYCLES:
                    result = (
                        bakeoff
                        .aggregate_generic_ballot_snapshot(
                            cycle_snapshots[cycle],
                            spec,
                        )
                    )

                    estimate = float(
                        result.estimate_margin_dem
                    )

                    environment_estimate = (
                        ENVIRONMENT_COEFFICIENT
                        * estimate
                    )

                    row: dict[str, object] = {
                        "specification_id": spec_id,
                        "cycle": cycle,
                        "lookback_days": (
                            lookback_days
                        ),
                        "recency_mode": recency_mode,
                        "recency_half_life_days": (
                            half_life_days
                        ),
                        "pollster_quality_weighting": (
                            quality_weighting
                        ),
                        "estimate_margin_dem": estimate,
                        "environment_margin_dem": (
                            environment_estimate
                        ),
                    }

                    for field_name in [
                        "input_questions",
                        "retained_questions",
                        "aggregated_rows",
                        "unique_polls",
                        "unique_pollsters",
                        "effective_sample_size",
                        "weighted_mean_poll_age_days",
                    ]:
                        row[field_name] = getattr(
                            result,
                            field_name,
                            np.nan,
                        )

                    if cycle in target_lookup:
                        actual = float(
                            target_lookup[cycle]
                        )

                        row[
                            "actual_national_house_margin_dem"
                        ] = actual

                        row["raw_error_dem"] = (
                            estimate - actual
                        )

                        row[
                            "raw_absolute_error"
                        ] = abs(
                            estimate - actual
                        )

                        row[
                            "raw_squared_error"
                        ] = (
                            estimate - actual
                        ) ** 2

                        row[
                            "environment_error_dem"
                        ] = (
                            environment_estimate
                            - actual
                        )

                        row[
                            "environment_absolute_error"
                        ] = abs(
                            environment_estimate
                            - actual
                        )

                        row[
                            "environment_squared_error"
                        ] = (
                            environment_estimate
                            - actual
                        ) ** 2

                    rows.append(row)

    if expected_specs != 80:
        raise AssertionError(
            "Focused grid should contain exactly "
            f"80 specifications, found {expected_specs}."
        )

    cycle_results = pd.DataFrame(rows)

    scored = cycle_results[
        cycle_results["cycle"].isin(SCORED_CYCLES)
    ].copy()

    summary_rows: list[dict[str, object]] = []

    grouped = scored.groupby(
        "specification_id",
        sort=False,
    )

    for spec_id, group in grouped:
        first = group.iloc[0]

        raw_errors = group[
            "raw_error_dem"
        ].astype(float)

        environment_errors = group[
            "environment_error_dem"
        ].astype(float)

        complexity_score = (
            int(
                first["recency_mode"]
                != "equal"
            )
            + int(
                bool(
                    first[
                        "pollster_quality_weighting"
                    ]
                )
            )
        )

        estimate_2024 = float(
            cycle_results.loc[
                (
                    cycle_results[
                        "specification_id"
                    ].eq(spec_id)
                    & cycle_results["cycle"].eq(2024)
                ),
                "estimate_margin_dem",
            ].iloc[0]
        )

        environment_2024 = (
            ENVIRONMENT_COEFFICIENT
            * estimate_2024
        )

        environment_mae = float(
            environment_errors.abs().mean()
        )

        summary_rows.append(
            {
                "specification_id": spec_id,
                "scored_cycles": len(group),
                "lookback_days": int(
                    first["lookback_days"]
                ),
                "recency_mode": str(
                    first["recency_mode"]
                ),
                "recency_half_life_days": (
                    first[
                        "recency_half_life_days"
                    ]
                ),
                "pollster_quality_weighting": (
                    bool(
                        first[
                            "pollster_quality_weighting"
                        ]
                    )
                ),
                "raw_mae": float(
                    raw_errors.abs().mean()
                ),
                "raw_rmse": float(
                    np.sqrt(
                        np.mean(
                            np.square(raw_errors)
                        )
                    )
                ),
                "raw_mean_error": float(
                    raw_errors.mean()
                ),
                "raw_max_absolute_error": float(
                    raw_errors.abs().max()
                ),
                "environment_mae": (
                    environment_mae
                ),
                "environment_rmse": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                environment_errors
                            )
                        )
                    )
                ),
                "environment_mean_error": float(
                    environment_errors.mean()
                ),
                "environment_max_absolute_error": (
                    float(
                        environment_errors
                        .abs()
                        .max()
                    )
                ),
                "complexity_score": (
                    complexity_score
                ),
                "simplicity_adjusted_score": (
                    environment_mae
                    + SIMPLICITY_PENALTY
                    * complexity_score
                ),
                "estimate_2024_margin_dem": (
                    estimate_2024
                ),
                "environment_2024_margin_dem": (
                    environment_2024
                ),
                "mean_effective_sample_size": (
                    float(
                        group[
                            "effective_sample_size"
                        ].mean()
                    )
                ),
                "minimum_unique_polls": int(
                    group["unique_polls"].min()
                ),
                "mean_weighted_poll_age_days": (
                    float(
                        group[
                            "weighted_mean_poll_age_days"
                        ].mean()
                    )
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)

    summary = summary.sort_values(
        [
            "environment_mae",
            "environment_rmse",
            "environment_max_absolute_error",
            "complexity_score",
            "specification_id",
        ],
        kind="stable",
    ).reset_index(drop=True)

    summary["environment_rank"] = (
        np.arange(len(summary)) + 1
    )

    summary[
        "simplicity_adjusted_rank"
    ] = (
        summary[
            "simplicity_adjusted_score"
        ]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )

    summary = bakeoff.add_pareto_flag(
        summary
    )

    cycle_results.to_csv(
        OUTPUT_DIR
        / "focused_refinement_cycle_results.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "focused_refinement_summary.csv",
        index=False,
    )

    top_columns = [
        "environment_rank",
        "specification_id",
        "environment_mae",
        "environment_rmse",
        "environment_max_absolute_error",
        "raw_mae",
        "complexity_score",
        "simplicity_adjusted_score",
        "estimate_2024_margin_dem",
        "environment_2024_margin_dem",
        "minimum_unique_polls",
        "pareto_optimal",
    ]

    print()
    print("Top 20 by environment MAE")
    print("-" * 110)
    print(
        summary[
            top_columns
        ]
        .head(20)
        .to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    simplicity = summary.sort_values(
        [
            "simplicity_adjusted_score",
            "environment_mae",
            "complexity_score",
            "specification_id",
        ],
        kind="stable",
    )

    print()
    print("Top 20 with simplicity adjustment")
    print("-" * 110)
    print(
        simplicity[
            [
                "simplicity_adjusted_rank",
                "specification_id",
                "simplicity_adjusted_score",
                "environment_mae",
                "environment_rmse",
                "environment_max_absolute_error",
                "complexity_score",
                "estimate_2024_margin_dem",
                "environment_2024_margin_dem",
                "pareto_optimal",
            ]
        ]
        .head(20)
        .to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print("Average performance by lookback")
    print("-" * 110)

    lookback_summary = (
        summary.groupby(
            "lookback_days",
            as_index=False,
        )
        .agg(
            specifications=(
                "specification_id",
                "size",
            ),
            mean_environment_mae=(
                "environment_mae",
                "mean",
            ),
            median_environment_mae=(
                "environment_mae",
                "median",
            ),
            best_environment_mae=(
                "environment_mae",
                "min",
            ),
            mean_max_error=(
                "environment_max_absolute_error",
                "mean",
            ),
        )
    )

    print(
        lookback_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print("Average performance by recency")
    print("-" * 110)

    recency_labels = summary.apply(
        lambda row: (
            "equal"
            if row["recency_mode"] == "equal"
            else (
                "half_life_"
                f"{int(row['recency_half_life_days'])}"
            )
        ),
        axis=1,
    )

    recency_summary = (
        summary.assign(
            recency_setting=recency_labels
        )
        .groupby(
            "recency_setting",
            as_index=False,
        )
        .agg(
            specifications=(
                "specification_id",
                "size",
            ),
            mean_environment_mae=(
                "environment_mae",
                "mean",
            ),
            median_environment_mae=(
                "environment_mae",
                "median",
            ),
            best_environment_mae=(
                "environment_mae",
                "min",
            ),
            mean_max_error=(
                "environment_max_absolute_error",
                "mean",
            ),
        )
        .sort_values(
            "best_environment_mae",
        )
    )

    print(
        recency_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    return summary, cycle_results


def eligible_questions(
    snapshot: pd.DataFrame,
    *,
    lookback_days: int,
) -> pd.DataFrame:
    frame = snapshot.copy()

    frame["poll_age_days"] = safe_numeric(
        frame["poll_age_days"],
    )

    frame["sample_size"] = safe_numeric(
        frame["sample_size"],
    )

    frame["margin_dem"] = safe_numeric(
        frame["margin_dem"],
    )

    frame = frame[
        frame["poll_age_days"].between(
            0,
            lookback_days,
            inclusive="both",
        )
    ].copy()

    frame = frame[
        frame["margin_dem"].notna()
    ].copy()

    frame = frame[
        frame["poll_id"].notna()
    ].copy()

    frame["_population_priority"] = (
        frame["population"].map(
            population_priority
        )
    )

    frame["_question_id_text"] = (
        frame["question_id"]
        .fillna("")
        .astype(str)
    )

    return frame


def select_largest_sample_questions(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    selected = frame.copy()

    # Production selection is still used for the actual
    # estimates. This explicit selection table exists to
    # make the question-choice audit transparent.
    selected["_sample_sort"] = (
        selected["sample_size"].fillna(-1)
    )

    selected = selected.sort_values(
        [
            "poll_id",
            "_sample_sort",
            "_population_priority",
            "_question_id_text",
        ],
        ascending=[
            True,
            False,
            True,
            True,
        ],
        kind="stable",
    )

    selected = selected.drop_duplicates(
        subset=["poll_id"],
        keep="first",
    ).copy()

    selected["normalized_population"] = (
        selected["population"].map(
            normalize_population
        )
    )

    selected["partisan_group"] = (
        selected["partisan"].map(
            normalize_partisan
        )
    )

    selected["is_internal"] = (
        selected["internal"].map(
            normalize_internal
        )
    )

    return selected


def base_statistical_weight(
    frame: pd.DataFrame,
    *,
    recency_mode: str,
    half_life_days: float | None,
) -> pd.Series:
    sample_size = (
        frame["sample_size"]
        .fillna(REFERENCE_SAMPLE_SIZE)
        .clip(lower=1.0)
    )

    sample_weight = np.sqrt(
        sample_size
        / REFERENCE_SAMPLE_SIZE
    )

    if recency_mode == "equal":
        recency_weight = pd.Series(
            1.0,
            index=frame.index,
        )
    else:
        if (
            half_life_days is None
            or half_life_days <= 0
        ):
            raise ValueError(
                "Positive half-life required."
            )

        recency_weight = np.power(
            0.5,
            frame["poll_age_days"]
            / float(half_life_days),
        )

    return (
        sample_weight
        * recency_weight
    )


def weighted_margin(
    frame: pd.DataFrame,
    weights: pd.Series,
) -> float:
    usable = (
        frame["margin_dem"].notna()
        & weights.notna()
        & weights.gt(0)
    )

    if not usable.any():
        return float("nan")

    return float(
        np.average(
            frame.loc[
                usable,
                "margin_dem",
            ].astype(float),
            weights=weights.loc[
                usable
            ].astype(float),
        )
    )


def run_partisan_audit(
    *,
    snapshots: pd.DataFrame,
    recommendation: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print()
    print("STUDY 2 — Partisan-poll influence audit")
    print("=" * 110)

    lookback_days = int(
        recommendation["lookback_days"]
    )

    recency_mode = str(
        recommendation["recency_mode"]
    )

    half_life_value = (
        recommendation[
            "recency_half_life_days"
        ]
    )

    half_life_days = (
        None
        if pd.isna(half_life_value)
        else float(half_life_value)
    )

    quality_weighting = bool(
        recommendation[
            "pollster_quality_weighting"
        ]
    )

    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    for cycle in AUDIT_CYCLES:
        snapshot = bakeoff.select_snapshot(
            snapshots,
            cycle=cycle,
            snapshot_days_before_election=0,
        )

        eligible = eligible_questions(
            snapshot,
            lookback_days=lookback_days,
        )

        selected = (
            select_largest_sample_questions(
                eligible
            )
        )

        selected["_base_weight"] = (
            base_statistical_weight(
                selected,
                recency_mode=recency_mode,
                half_life_days=half_life_days,
            )
        )

        selected["_partisan_include_weight"] = (
            selected["_base_weight"]
        )

        selected["_partisan_downweight_weight"] = (
            selected["_base_weight"]
            * np.where(
                selected[
                    "partisan_group"
                ].eq("INDEPENDENT"),
                1.0,
                0.5,
            )
        )

        selected["_partisan_exclude_weight"] = (
            selected["_base_weight"]
            * np.where(
                selected[
                    "partisan_group"
                ].eq("INDEPENDENT"),
                1.0,
                0.0,
            )
        )

        total_weight = float(
            selected[
                "_partisan_include_weight"
            ].sum()
        )

        partisan_weight = float(
            selected.loc[
                selected[
                    "partisan_group"
                ].ne("INDEPENDENT"),
                "_partisan_include_weight",
            ].sum()
        )

        partisan_weight_share = (
            partisan_weight / total_weight
            if total_weight > 0
            else np.nan
        )

        engine_estimates: dict[str, float] = {}
        engine_results: dict[
            str,
            object,
        ] = {}

        for partisan_mode in [
            "include",
            "downweight",
            "exclude",
        ]:
            spec = build_spec(
                lookback_days=lookback_days,
                recency_mode=recency_mode,
                half_life_days=half_life_days,
                quality_weighting=(
                    quality_weighting
                ),
                partisan_mode=partisan_mode,
            )

            result = (
                bakeoff
                .aggregate_generic_ballot_snapshot(
                    snapshot,
                    spec,
                )
            )

            engine_results[
                partisan_mode
            ] = result

            engine_estimates[
                partisan_mode
            ] = float(
                result.estimate_margin_dem
            )

        include_estimate = (
            engine_estimates["include"]
        )

        summary_rows.append(
            {
                "cycle": cycle,
                "lookback_days": (
                    lookback_days
                ),
                "recency_mode": recency_mode,
                "recency_half_life_days": (
                    half_life_days
                ),
                "pollster_quality_weighting": (
                    quality_weighting
                ),
                "eligible_questions": len(
                    eligible
                ),
                "manually_selected_polls": len(
                    selected
                ),
                "independent_selected_polls": int(
                    selected[
                        "partisan_group"
                    ]
                    .eq("INDEPENDENT")
                    .sum()
                ),
                "dem_partisan_selected_polls": int(
                    selected[
                        "partisan_group"
                    ]
                    .eq("DEM")
                    .sum()
                ),
                "rep_partisan_selected_polls": int(
                    selected[
                        "partisan_group"
                    ]
                    .eq("REP")
                    .sum()
                ),
                "internal_selected_polls": int(
                    selected[
                        "is_internal"
                    ].sum()
                ),
                "manual_total_weight": (
                    total_weight
                ),
                "manual_partisan_weight": (
                    partisan_weight
                ),
                "manual_partisan_weight_share": (
                    partisan_weight_share
                ),
                "engine_include_margin_dem": (
                    include_estimate
                ),
                "engine_downweight_margin_dem": (
                    engine_estimates[
                        "downweight"
                    ]
                ),
                "engine_exclude_margin_dem": (
                    engine_estimates[
                        "exclude"
                    ]
                ),
                "downweight_change_from_include": (
                    engine_estimates[
                        "downweight"
                    ]
                    - include_estimate
                ),
                "exclude_change_from_include": (
                    engine_estimates[
                        "exclude"
                    ]
                    - include_estimate
                ),
                "manual_include_margin_dem": (
                    weighted_margin(
                        selected,
                        selected[
                            "_partisan_include_weight"
                        ],
                    )
                ),
                "manual_downweight_margin_dem": (
                    weighted_margin(
                        selected,
                        selected[
                            "_partisan_downweight_weight"
                        ],
                    )
                ),
                "manual_exclude_margin_dem": (
                    weighted_margin(
                        selected,
                        selected[
                            "_partisan_exclude_weight"
                        ],
                    )
                ),
                "engine_input_questions": getattr(
                    engine_results["include"],
                    "input_questions",
                    np.nan,
                ),
                "engine_retained_questions": getattr(
                    engine_results["include"],
                    "retained_questions",
                    np.nan,
                ),
                "engine_aggregated_rows": getattr(
                    engine_results["include"],
                    "aggregated_rows",
                    np.nan,
                ),
                "engine_unique_polls": getattr(
                    engine_results["include"],
                    "unique_polls",
                    np.nan,
                ),
                "engine_unique_pollsters": getattr(
                    engine_results["include"],
                    "unique_pollsters",
                    np.nan,
                ),
                "engine_effective_sample_size": (
                    getattr(
                        engine_results[
                            "include"
                        ],
                        "effective_sample_size",
                        np.nan,
                    )
                ),
            }
        )

        for _, row in selected.iterrows():
            detail_rows.append(
                {
                    "cycle": cycle,
                    "poll_id": row.get(
                        "poll_id"
                    ),
                    "question_id": row.get(
                        "question_id"
                    ),
                    "pollster": row.get(
                        "pollster"
                    ),
                    "display_name": row.get(
                        "display_name"
                    ),
                    "sponsors": row.get(
                        "sponsors"
                    ),
                    "partisan": row.get(
                        "partisan"
                    ),
                    "partisan_group": row.get(
                        "partisan_group"
                    ),
                    "internal": row.get(
                        "internal"
                    ),
                    "is_internal": row.get(
                        "is_internal"
                    ),
                    "population": row.get(
                        "population"
                    ),
                    "sample_size": row.get(
                        "sample_size"
                    ),
                    "poll_age_days": row.get(
                        "poll_age_days"
                    ),
                    "margin_dem": row.get(
                        "margin_dem"
                    ),
                    "base_weight": row.get(
                        "_base_weight"
                    ),
                    "weighted_contribution": (
                        float(
                            row.get(
                                "_base_weight"
                            )
                        )
                        * float(
                            row.get(
                                "margin_dem"
                            )
                        )
                    ),
                    "start_date": row.get(
                        "start_date"
                    ),
                    "end_date": row.get(
                        "end_date"
                    ),
                    "numeric_grade": row.get(
                        "numeric_grade"
                    ),
                    "source_file": row.get(
                        "source_file"
                    ),
                    "source_row_number": row.get(
                        "source_row_number"
                    ),
                }
            )

    partisan_summary = pd.DataFrame(
        summary_rows
    )

    partisan_detail = pd.DataFrame(
        detail_rows
    )

    partisan_summary.to_csv(
        OUTPUT_DIR
        / "partisan_influence_summary.csv",
        index=False,
    )

    partisan_detail.to_csv(
        OUTPUT_DIR
        / "partisan_selected_poll_details.csv",
        index=False,
    )

    print()
    print("Partisan influence by cycle")
    print("-" * 110)

    print(
        partisan_summary[
            [
                "cycle",
                "eligible_questions",
                "manually_selected_polls",
                "independent_selected_polls",
                "dem_partisan_selected_polls",
                "rep_partisan_selected_polls",
                "internal_selected_polls",
                "manual_partisan_weight_share",
                "engine_include_margin_dem",
                "engine_downweight_margin_dem",
                "engine_exclude_margin_dem",
                "downweight_change_from_include",
                "exclude_change_from_include",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    partisan_only = partisan_detail[
        partisan_detail[
            "partisan_group"
        ].ne("INDEPENDENT")
    ].copy()

    print()
    print("Partisan polls selected by the audit")
    print("-" * 110)

    if partisan_only.empty:
        print(
            "No partisan polls entered the selected "
            "question set."
        )
    else:
        print(
            partisan_only[
                [
                    "cycle",
                    "pollster",
                    "sponsors",
                    "partisan_group",
                    "population",
                    "sample_size",
                    "poll_age_days",
                    "margin_dem",
                    "base_weight",
                    "weighted_contribution",
                ]
            ]
            .sort_values(
                [
                    "cycle",
                    "base_weight",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
            .to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.4f}"
                ),
            )
        )

    return partisan_summary, partisan_detail


def run_question_selection_audit(
    *,
    snapshots: pd.DataFrame,
    recommendation: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print()
    print(
        "STUDY 3 — Largest-sample "
        "question-selection audit"
    )
    print("=" * 110)

    lookback_days = int(
        recommendation["lookback_days"]
    )

    question_rows: list[
        dict[str, object]
    ] = []

    poll_rows: list[
        dict[str, object]
    ] = []

    for cycle in AUDIT_CYCLES:
        snapshot = bakeoff.select_snapshot(
            snapshots,
            cycle=cycle,
            snapshot_days_before_election=0,
        )

        eligible = eligible_questions(
            snapshot,
            lookback_days=lookback_days,
        )

        selected = (
            select_largest_sample_questions(
                eligible
            )
        )

        selected_question_ids = set(
            zip(
                selected[
                    "poll_id"
                ].astype(str),
                selected[
                    "question_id"
                ].astype(str),
            )
        )

        selected_by_poll = (
            selected.set_index(
                "poll_id",
                drop=False,
            )
        )

        for poll_id, group in eligible.groupby(
            "poll_id",
            sort=False,
        ):
            group = group.copy()

            max_sample_size = (
                group[
                    "sample_size"
                ].max()
            )

            selected_row = (
                selected_by_poll.loc[poll_id]
            )

            if isinstance(
                selected_row,
                pd.DataFrame,
            ):
                selected_row = (
                    selected_row.iloc[0]
                )

            populations_available = sorted(
                {
                    normalize_population(
                        value
                    )
                    for value in group[
                        "population"
                    ]
                }
            )

            lv_rows = group[
                group["population"].map(
                    normalize_population
                ).eq("LV")
            ]

            if lv_rows.empty:
                prefer_lv_row = (
                    group.sort_values(
                        [
                            "_population_priority",
                            "sample_size",
                        ],
                        ascending=[
                            True,
                            False,
                        ],
                        kind="stable",
                    )
                    .iloc[0]
                )
            else:
                prefer_lv_row = (
                    lv_rows.sort_values(
                        "sample_size",
                        ascending=False,
                        kind="stable",
                    )
                    .iloc[0]
                )

            selected_population = (
                normalize_population(
                    selected_row[
                        "population"
                    ]
                )
            )

            prefer_lv_population = (
                normalize_population(
                    prefer_lv_row[
                        "population"
                    ]
                )
            )

            selected_margin = float(
                selected_row[
                    "margin_dem"
                ]
            )

            prefer_lv_margin = float(
                prefer_lv_row[
                    "margin_dem"
                ]
            )

            poll_rows.append(
                {
                    "cycle": cycle,
                    "poll_id": poll_id,
                    "pollster": selected_row.get(
                        "pollster"
                    ),
                    "sponsors": selected_row.get(
                        "sponsors"
                    ),
                    "questions_available": len(
                        group
                    ),
                    "populations_available": "|".join(
                        populations_available
                    ),
                    "largest_sample_population": (
                        selected_population
                    ),
                    "largest_sample_size": (
                        selected_row[
                            "sample_size"
                        ]
                    ),
                    "largest_sample_margin_dem": (
                        selected_margin
                    ),
                    "prefer_lv_population": (
                        prefer_lv_population
                    ),
                    "prefer_lv_sample_size": (
                        prefer_lv_row[
                            "sample_size"
                        ]
                    ),
                    "prefer_lv_margin_dem": (
                        prefer_lv_margin
                    ),
                    "different_question_selected": (
                        str(
                            selected_row[
                                "question_id"
                            ]
                        )
                        != str(
                            prefer_lv_row[
                                "question_id"
                            ]
                        )
                    ),
                    "margin_difference_largest_minus_prefer_lv": (
                        selected_margin
                        - prefer_lv_margin
                    ),
                    "sample_size_gain_largest_minus_prefer_lv": (
                        float(
                            selected_row[
                                "sample_size"
                            ]
                        )
                        - float(
                            prefer_lv_row[
                                "sample_size"
                            ]
                        )
                    ),
                    "poll_age_days": (
                        selected_row[
                            "poll_age_days"
                        ]
                    ),
                    "numeric_grade": (
                        selected_row[
                            "numeric_grade"
                        ]
                    ),
                    "partisan_group": (
                        normalize_partisan(
                            selected_row[
                                "partisan"
                            ]
                        )
                    ),
                }
            )

            for _, row in group.iterrows():
                identity = (
                    str(
                        row["poll_id"]
                    ),
                    str(
                        row["question_id"]
                    ),
                )

                is_selected = (
                    identity
                    in selected_question_ids
                )

                if is_selected:
                    if (
                        pd.notna(
                            max_sample_size
                        )
                        and float(
                            row[
                                "sample_size"
                            ]
                        )
                        == float(
                            max_sample_size
                        )
                    ):
                        reason = (
                            "largest sample size"
                        )
                    else:
                        reason = (
                            "selected after "
                            "missing-size/tie handling"
                        )
                else:
                    reason = (
                        "smaller sample than "
                        "selected question"
                    )

                question_rows.append(
                    {
                        "cycle": cycle,
                        "poll_id": row.get(
                            "poll_id"
                        ),
                        "question_id": row.get(
                            "question_id"
                        ),
                        "pollster": row.get(
                            "pollster"
                        ),
                        "sponsors": row.get(
                            "sponsors"
                        ),
                        "population": row.get(
                            "population"
                        ),
                        "normalized_population": (
                            normalize_population(
                                row.get(
                                    "population"
                                )
                            )
                        ),
                        "sample_size": row.get(
                            "sample_size"
                        ),
                        "margin_dem": row.get(
                            "margin_dem"
                        ),
                        "poll_age_days": row.get(
                            "poll_age_days"
                        ),
                        "numeric_grade": row.get(
                            "numeric_grade"
                        ),
                        "selected_by_largest_sample": (
                            is_selected
                        ),
                        "selection_reason": reason,
                        "partisan_group": (
                            normalize_partisan(
                                row.get(
                                    "partisan"
                                )
                            )
                        ),
                    }
                )

    question_detail = pd.DataFrame(
        question_rows
    )

    poll_comparison = pd.DataFrame(
        poll_rows
    )

    question_detail.to_csv(
        OUTPUT_DIR
        / "question_selection_details.csv",
        index=False,
    )

    poll_comparison.to_csv(
        OUTPUT_DIR
        / "largest_sample_vs_prefer_lv.csv",
        index=False,
    )

    selected_questions = (
        question_detail[
            question_detail[
                "selected_by_largest_sample"
            ]
        ].copy()
    )

    population_summary = (
        selected_questions.groupby(
            [
                "cycle",
                "normalized_population",
            ],
            as_index=False,
        )
        .agg(
            selected_polls=(
                "poll_id",
                "nunique",
            ),
            mean_sample_size=(
                "sample_size",
                "mean",
            ),
            median_sample_size=(
                "sample_size",
                "median",
            ),
            mean_poll_age_days=(
                "poll_age_days",
                "mean",
            ),
            mean_numeric_grade=(
                "numeric_grade",
                "mean",
            ),
            mean_margin_dem=(
                "margin_dem",
                "mean",
            ),
        )
    )

    population_summary.to_csv(
        OUTPUT_DIR
        / "selected_population_summary.csv",
        index=False,
    )

    comparison_summary = (
        poll_comparison.groupby(
            "cycle",
            as_index=False,
        )
        .agg(
            polls=(
                "poll_id",
                "nunique",
            ),
            polls_with_different_selection=(
                "different_question_selected",
                "sum",
            ),
            mean_sample_size_gain=(
                "sample_size_gain_largest_minus_prefer_lv",
                "mean",
            ),
            median_sample_size_gain=(
                "sample_size_gain_largest_minus_prefer_lv",
                "median",
            ),
            mean_margin_difference=(
                "margin_difference_largest_minus_prefer_lv",
                "mean",
            ),
            mean_absolute_margin_difference=(
                "margin_difference_largest_minus_prefer_lv",
                lambda values: (
                    values.abs().mean()
                ),
            ),
        )
    )

    comparison_summary[
        "different_selection_share"
    ] = (
        comparison_summary[
            "polls_with_different_selection"
        ]
        / comparison_summary["polls"]
    )

    comparison_summary.to_csv(
        OUTPUT_DIR
        / "largest_sample_comparison_summary.csv",
        index=False,
    )

    print()
    print("Population selected by largest-sample rule")
    print("-" * 110)
    print(
        population_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print(
        "Largest-sample versus prefer-LV "
        "selection comparison"
    )
    print("-" * 110)
    print(
        comparison_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    differing = poll_comparison[
        poll_comparison[
            "different_question_selected"
        ]
    ].copy()

    print()
    print(
        "Largest individual differences "
        "between the two rules"
    )
    print("-" * 110)

    if differing.empty:
        print(
            "The two rules selected the same "
            "question for every poll."
        )
    else:
        differing[
            "absolute_margin_difference"
        ] = (
            differing[
                "margin_difference_largest_minus_prefer_lv"
            ].abs()
        )

        print(
            differing.sort_values(
                "absolute_margin_difference",
                ascending=False,
            )[
                [
                    "cycle",
                    "pollster",
                    "questions_available",
                    "populations_available",
                    "largest_sample_population",
                    "largest_sample_size",
                    "largest_sample_margin_dem",
                    "prefer_lv_population",
                    "prefer_lv_sample_size",
                    "prefer_lv_margin_dem",
                    "sample_size_gain_largest_minus_prefer_lv",
                    "margin_difference_largest_minus_prefer_lv",
                ]
            ]
            .head(40)
            .to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.4f}"
                ),
            )
        )

    return question_detail, poll_comparison


def write_recommendation(
    *,
    summary: pd.DataFrame,
) -> pd.Series:
    recommended = (
        summary.sort_values(
            [
                "simplicity_adjusted_score",
                "environment_mae",
                "environment_rmse",
                "complexity_score",
                "specification_id",
            ],
            kind="stable",
        )
        .iloc[0]
        .copy()
    )

    recommendation_frame = pd.DataFrame(
        [
            {
                "recommended_specification_id": (
                    recommended[
                        "specification_id"
                    ]
                ),
                "lookback_days": int(
                    recommended[
                        "lookback_days"
                    ]
                ),
                "recency_mode": (
                    recommended[
                        "recency_mode"
                    ]
                ),
                "recency_half_life_days": (
                    recommended[
                        "recency_half_life_days"
                    ]
                ),
                "question_selection_mode": (
                    FIXED_QUESTION_SELECTION
                ),
                "sample_size_weighting": (
                    FIXED_SAMPLE_SIZE_WEIGHTING
                ),
                "pollster_quality_weighting": (
                    bool(
                        recommended[
                            "pollster_quality_weighting"
                        ]
                    )
                ),
                "partisan_mode": (
                    FIXED_PARTISAN_MODE
                ),
                "duplicate_mode": (
                    FIXED_DUPLICATE_MODE
                ),
                "environment_coefficient": (
                    ENVIRONMENT_COEFFICIENT
                ),
                "environment_mae": (
                    recommended[
                        "environment_mae"
                    ]
                ),
                "environment_rmse": (
                    recommended[
                        "environment_rmse"
                    ]
                ),
                "environment_max_absolute_error": (
                    recommended[
                        "environment_max_absolute_error"
                    ]
                ),
                "complexity_score": (
                    recommended[
                        "complexity_score"
                    ]
                ),
                "simplicity_adjusted_score": (
                    recommended[
                        "simplicity_adjusted_score"
                    ]
                ),
                "estimate_2024_margin_dem": (
                    recommended[
                        "estimate_2024_margin_dem"
                    ]
                ),
                "environment_2024_margin_dem": (
                    recommended[
                        "environment_2024_margin_dem"
                    ]
                ),
            }
        ]
    )

    recommendation_frame.to_csv(
        OUTPUT_DIR
        / "production_recommendation.csv",
        index=False,
    )

    return recommended


def print_recommendation(
    recommendation: pd.Series,
) -> None:
    print()
    print("PROVISIONAL PRODUCTION RECOMMENDATION")
    print("=" * 110)

    half_life_value = (
        recommendation[
            "recency_half_life_days"
        ]
    )

    half_life_text = (
        "N/A"
        if pd.isna(half_life_value)
        else f"{float(half_life_value):.0f} days"
    )

    print(
        f"Specification .............. "
        f"{recommendation['specification_id']}"
    )
    print(
        f"Lookback ................... "
        f"{int(recommendation['lookback_days'])} days"
    )
    print(
        f"Recency mode ............... "
        f"{recommendation['recency_mode']}"
    )
    print(
        f"Recency half-life .......... "
        f"{half_life_text}"
    )
    print(
        f"Question selection ......... "
        f"{FIXED_QUESTION_SELECTION}"
    )
    print(
        f"Sample-size weighting ...... "
        f"{FIXED_SAMPLE_SIZE_WEIGHTING}"
    )
    print(
        f"Pollster-quality weighting . "
        f"{bool(recommendation['pollster_quality_weighting'])}"
    )
    print(
        f"Partisan treatment ......... "
        f"{FIXED_PARTISAN_MODE}"
    )
    print(
        f"Duplicate handling ......... "
        f"{FIXED_DUPLICATE_MODE}"
    )
    print(
        f"Environment MAE ............ "
        f"{float(recommendation['environment_mae']):.4f}"
    )
    print(
        f"Environment RMSE ........... "
        f"{float(recommendation['environment_rmse']):.4f}"
    )
    print(
        f"Maximum absolute error ..... "
        f"{float(recommendation['environment_max_absolute_error']):.4f}"
    )
    print(
        f"2024 raw estimate .......... "
        f"{float(recommendation['estimate_2024_margin_dem']):+.4f}"
    )
    print(
        f"2024 environment estimate .. "
        f"{float(recommendation['environment_2024_margin_dem']):+.4f}"
    )


def main() -> None:
    print(
        "House Generic Ballot Final Validation"
    )
    print("=" * 110)

    snapshots, targets = load_inputs()

    print()
    print("Input integrity")
    print("-" * 110)
    print(
        f"Snapshot rows: {len(snapshots):,}"
    )
    print(
        f"Target rows: {len(targets):,}"
    )
    print(
        "Focused specifications: 80"
    )
    print(
        f"Scored cycles: {SCORED_CYCLES}"
    )
    print(
        f"Audit cycles: {AUDIT_CYCLES}"
    )

    summary, _ = run_focused_refinement(
        snapshots=snapshots,
        targets=targets,
    )

    recommendation = write_recommendation(
        summary=summary,
    )

    print_recommendation(
        recommendation
    )

    run_partisan_audit(
        snapshots=snapshots,
        recommendation=recommendation,
    )

    run_question_selection_audit(
        snapshots=snapshots,
        recommendation=recommendation,
    )

    print()
    print("Outputs")
    print("-" * 110)

    for path in sorted(
        OUTPUT_DIR.glob("*.csv")
    ):
        print(
            f"  - {path.relative_to(ROOT)}"
        )

    print()
    print("Final validation PASSED.")


if __name__ == "__main__":
    main()
