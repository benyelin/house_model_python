from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

COMMON_AGGREGATION = (
    ROOT
    / "historical"
    / "common"
    / "polling"
    / "aggregation"
)

HOUSE_BACKTESTS = (
    ROOT
    / "historical"
    / "house"
    / "backtests"
)

OUTPUT_DIR = (
    HOUSE_BACKTESTS
    / "outputs"
    / "generic_ballot_rv_lv_selection_bakeoff"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

sys.path.insert(
    0,
    str(COMMON_AGGREGATION),
)

sys.path.insert(
    0,
    str(HOUSE_BACKTESTS),
)

from generic_ballot_aggregation import (  # noqa: E402
    GenericBallotAggregationSpec,
    aggregate_generic_ballot_snapshot,
)

import run_house_generic_ballot_aggregation_bakeoff as bakeoff  # noqa: E402
import run_house_generic_ballot_population_switch_bakeoff as population_bakeoff  # noqa: E402


SNAPSHOT_PATH = (
    ROOT
    / "historical"
    / "common"
    / "polling"
    / "snapshots"
    / "generic_ballot_polling_snapshots.csv"
)

CYCLES = [
    2018,
    2020,
    2022,
    2024,
]

MIDTERMS = [
    2018,
    2022,
]

DAYS_OUT = [
    120,
    90,
    75,
    60,
    45,
    30,
    21,
    14,
    7,
    1,
]

ENVIRONMENT_COEFFICIENT = 0.90


def normalize_population(value):
    if pd.isna(value):
        return ""

    value = str(value).strip().lower()

    mapping = {
        "registered voters": "rv",
        "registered voter": "rv",
        "rv": "rv",
        "likely voters": "lv",
        "likely voter": "lv",
        "lv": "lv",
    }

    return mapping.get(
        value,
        value,
    )


def find_column(
    frame,
    candidates,
):
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    return None


def load_snapshots():
    snapshots = pd.read_csv(
        SNAPSHOT_PATH,
        low_memory=False,
    )

    population_col = find_column(
        snapshots,
        [
            "population",
            "population_type",
        ],
    )

    if population_col is None:
        raise RuntimeError(
            "Could not identify population column."
        )

    snapshots[
        "_population_norm"
    ] = snapshots[
        population_col
    ].map(
        normalize_population
    )

    snapshots = snapshots.loc[
        snapshots[
            "_population_norm"
        ].isin(
            [
                "rv",
                "lv",
            ]
        )
    ].copy()

    cycle_col = find_column(
        snapshots,
        [
            "cycle",
            "year",
        ],
    )

    days_col = find_column(
        snapshots,
        [
            "snapshot_days_before_election",
            "days_out",
            "days_until_election",
        ],
    )

    if cycle_col is None:
        raise RuntimeError(
            "Could not identify cycle column."
        )

    if days_col is None:
        raise RuntimeError(
            "Could not identify days-out column."
        )

    snapshots[
        "_cycle"
    ] = pd.to_numeric(
        snapshots[
            cycle_col
        ],
        errors="coerce",
    )

    snapshots[
        "_days_out"
    ] = pd.to_numeric(
        snapshots[
            days_col
        ],
        errors="coerce",
    )

    return snapshots


def load_targets():
    """
    Use the same canonical national House
    two-party targets as the previously
    validated RV/LV switching bakeoff.

    This includes 2018, 2020, 2022, and 2024.
    """

    targets = (
        population_bakeoff
        .build_house_targets()
    )

    target_map = {}

    for _, row in targets.iterrows():
        cycle = pd.to_numeric(
            row["cycle"],
            errors="coerce",
        )

        target = pd.to_numeric(
            row[
                "actual_house_margin_dem"
            ],
            errors="coerce",
        )

        if (
            np.isfinite(cycle)
            and np.isfinite(target)
        ):
            target_map[
                int(cycle)
            ] = float(target)

    return target_map


def build_spec():
    return GenericBallotAggregationSpec(
        lookback_days=21,
        recency_mode="equal",
        recency_half_life_days=None,
        population_mode="all",
        sample_size_weighting=True,
        pollster_quality_weighting=False,
        partisan_mode="include",
        partisan_weight=0.50,
        question_selection_mode="largest_sample",
        duplicate_mode="poll",
    )


def aggregate_standard(
    snapshot,
):
    result = (
        aggregate_generic_ballot_snapshot(
            snapshot,
            build_spec(),
        )
    )

    return {
        "estimate": float(
            result.estimate_margin_dem
        ),
        "unique_polls": int(
            result.unique_polls
        ),
        "effective_sample_size": float(
            result.effective_sample_size
        ),
    }


def identify_poll_column(
    frame,
):
    return find_column(
        frame,
        [
            "poll_id",
            "pollster_id",
        ],
    )


def identify_sample_column(
    frame,
):
    return find_column(
        frame,
        [
            "sample_size",
            "sample",
            "samplesize",
        ],
    )


def largest_sample_rv_lv(
    snapshot,
):
    poll_col = identify_poll_column(
        snapshot
    )

    sample_col = identify_sample_column(
        snapshot
    )

    if poll_col is None:
        raise RuntimeError(
            "Could not identify poll ID column."
        )

    working = snapshot.copy()

    if sample_col is None:
        working[
            "_sample_numeric"
        ] = 0.0
    else:
        working[
            "_sample_numeric"
        ] = pd.to_numeric(
            working[
                sample_col
            ],
            errors="coerce",
        ).fillna(
            0.0
        )

    working[
        "_population_priority"
    ] = working[
        "_population_norm"
    ].map(
        {
            "lv": 1,
            "rv": 0,
        }
    ).fillna(
        -1
    )

    working = working.sort_values(
        [
            poll_col,
            "_sample_numeric",
            "_population_priority",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )

    working = working.drop_duplicates(
        subset=[
            poll_col,
        ],
        keep="first",
    )

    return aggregate_standard(
        working
    )


def weighted_rv_lv_blend(
    snapshot,
):
    rv = snapshot.loc[
        snapshot[
            "_population_norm"
        ].eq(
            "rv"
        )
    ].copy()

    lv = snapshot.loc[
        snapshot[
            "_population_norm"
        ].eq(
            "lv"
        )
    ].copy()

    rv_result = None
    lv_result = None

    if not rv.empty:
        rv_result = aggregate_standard(
            rv
        )

    if not lv.empty:
        lv_result = aggregate_standard(
            lv
        )

    if (
        rv_result is None
        and lv_result is None
    ):
        raise RuntimeError(
            "No RV or LV rows available."
        )

    if rv_result is None:
        return lv_result

    if lv_result is None:
        return rv_result

    rv_weight = max(
        rv_result[
            "effective_sample_size"
        ],
        1e-9,
    )

    lv_weight = max(
        lv_result[
            "effective_sample_size"
        ],
        1e-9,
    )

    estimate = (
        (
            rv_result[
                "estimate"
            ]
            * rv_weight
        )
        +
        (
            lv_result[
                "estimate"
            ]
            * lv_weight
        )
    ) / (
        rv_weight
        + lv_weight
    )

    return {
        "estimate": float(
            estimate
        ),
        "unique_polls": int(
            max(
                rv_result[
                    "unique_polls"
                ],
                lv_result[
                    "unique_polls"
                ],
            )
        ),
        "effective_sample_size": float(
            rv_weight
            + lv_weight
        ),
    }


def evaluate_rule(
    snapshot,
    rule,
):
    if rule == "rv_only":
        selected = snapshot.loc[
            snapshot[
                "_population_norm"
            ].eq(
                "rv"
            )
        ].copy()

        if selected.empty:
            return None

        return aggregate_standard(
            selected
        )

    if rule == "lv_only":
        selected = snapshot.loc[
            snapshot[
                "_population_norm"
            ].eq(
                "lv"
            )
        ].copy()

        if selected.empty:
            return None

        return aggregate_standard(
            selected
        )

    if rule == "rv_lv_largest_sample":
        return largest_sample_rv_lv(
            snapshot
        )

    if rule == "rv_lv_weighted_blend":
        return weighted_rv_lv_blend(
            snapshot
        )

    raise ValueError(
        f"Unknown rule: {rule}"
    )


def summarize(
    predictions,
    cycles,
):
    subset = predictions.loc[
        predictions[
            "cycle"
        ].isin(
            cycles
        )
    ].copy()

    rows = []

    for rule, group in subset.groupby(
        "population_rule"
    ):
        errors = (
            group[
                "environment_estimate"
            ]
            -
            group[
                "actual_house_margin_dem"
            ]
        )

        absolute_errors = errors.abs()

        rows.append(
            {
                "population_rule": rule,
                "observations": len(
                    group
                ),
                "cycles": group[
                    "cycle"
                ].nunique(),
                "environment_mae": (
                    absolute_errors.mean()
                ),
                "environment_rmse": (
                    np.sqrt(
                        np.mean(
                            errors ** 2
                        )
                    )
                ),
                "environment_bias_dem": (
                    errors.mean()
                ),
                "maximum_absolute_error": (
                    absolute_errors.max()
                ),
                "mean_unique_polls": (
                    group[
                        "unique_polls"
                    ].mean()
                ),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "environment_mae"
        )
        .reset_index(
            drop=True
        )
    )


def horizon_summary(
    predictions,
    cycles,
):
    subset = predictions.loc[
        predictions[
            "cycle"
        ].isin(
            cycles
        )
    ].copy()

    rows = []

    for (
        days_out,
        rule,
    ), group in subset.groupby(
        [
            "days_out",
            "population_rule",
        ]
    ):
        errors = (
            group[
                "environment_estimate"
            ]
            -
            group[
                "actual_house_margin_dem"
            ]
        )

        rows.append(
            {
                "days_out": int(
                    days_out
                ),
                "population_rule": rule,
                "cycles": group[
                    "cycle"
                ].nunique(),
                "environment_mae": (
                    errors.abs().mean()
                ),
                "environment_rmse": (
                    np.sqrt(
                        np.mean(
                            errors ** 2
                        )
                    )
                ),
                "environment_bias_dem": (
                    errors.mean()
                ),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "days_out",
                "environment_mae",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


def main():
    print(
        "=" * 132
    )
    print(
        "HOUSE GENERIC BALLOT — "
        "CONTROLLED RV/LV SELECTION BAKEOFF"
    )
    print(
        "=" * 132
    )

    snapshots = load_snapshots()
    targets = load_targets()

    rules = [
        "rv_only",
        "lv_only",
        "rv_lv_largest_sample",
        "rv_lv_weighted_blend",
    ]

    rows = []

    for cycle in CYCLES:
        if cycle not in targets:
            raise RuntimeError(
                f"No target for {cycle}."
            )

        for days_out in DAYS_OUT:
            snapshot = snapshots.loc[
                snapshots[
                    "_cycle"
                ].eq(
                    cycle
                )
                &
                snapshots[
                    "_days_out"
                ].eq(
                    days_out
                )
            ].copy()

            if snapshot.empty:
                raise RuntimeError(
                    f"No snapshot for "
                    f"{cycle} D{days_out}."
                )

            for rule in rules:
                result = evaluate_rule(
                    snapshot,
                    rule,
                )

                if result is None:
                    raise RuntimeError(
                        f"No usable result for "
                        f"{cycle} D{days_out} "
                        f"{rule}."
                    )

                raw_estimate = result[
                    "estimate"
                ]

                environment_estimate = (
                    ENVIRONMENT_COEFFICIENT
                    * raw_estimate
                )

                actual = targets[
                    cycle
                ]

                rows.append(
                    {
                        "cycle": cycle,
                        "days_out": days_out,
                        "population_rule": rule,
                        "raw_estimate": (
                            raw_estimate
                        ),
                        "environment_estimate": (
                            environment_estimate
                        ),
                        "actual_house_margin_dem": (
                            actual
                        ),
                        "error": (
                            environment_estimate
                            - actual
                        ),
                        "absolute_error": abs(
                            environment_estimate
                            - actual
                        ),
                        "unique_polls": result[
                            "unique_polls"
                        ],
                        "effective_sample_size": (
                            result[
                                "effective_sample_size"
                            ]
                        ),
                    }
                )

    predictions = pd.DataFrame(
        rows
    )

    all_summary = summarize(
        predictions,
        CYCLES,
    )

    midterm_summary = summarize(
        predictions,
        MIDTERMS,
    )

    all_horizons = horizon_summary(
        predictions,
        CYCLES,
    )

    midterm_horizons = horizon_summary(
        predictions,
        MIDTERMS,
    )

    print()
    print(
        "OVERALL — ALL FOUR CYCLES"
    )
    print(
        "-" * 132
    )
    print(
        all_summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    print()
    print(
        "OVERALL — MIDTERMS ONLY"
    )
    print(
        "-" * 132
    )
    print(
        midterm_summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    print()
    print(
        "BY HORIZON — ALL FOUR CYCLES"
    )
    print(
        "-" * 132
    )
    print(
        all_horizons.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    print()
    print(
        "BY HORIZON — MIDTERMS ONLY"
    )
    print(
        "-" * 132
    )
    print(
        midterm_horizons.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    validation = pd.DataFrame(
        [
            {
                "check": (
                    "four_cycles_present"
                ),
                "passed": (
                    predictions[
                        "cycle"
                    ].nunique()
                    == 4
                ),
            },
            {
                "check": (
                    "ten_snapshots_per_cycle"
                ),
                "passed": (
                    predictions[
                        [
                            "cycle",
                            "days_out",
                        ]
                    ]
                    .drop_duplicates()
                    .groupby(
                        "cycle"
                    )
                    .size()
                    .eq(
                        10
                    )
                    .all()
                ),
            },
            {
                "check": (
                    "four_rules_present"
                ),
                "passed": (
                    predictions[
                        "population_rule"
                    ].nunique()
                    == 4
                ),
            },
            {
                "check": (
                    "all_estimates_finite"
                ),
                "passed": (
                    np.isfinite(
                        predictions[
                            "environment_estimate"
                        ]
                    ).all()
                ),
            },
            {
                "check": (
                    "only_rv_lv_input_rows"
                ),
                "passed": (
                    snapshots[
                        "_population_norm"
                    ].isin(
                        [
                            "rv",
                            "lv",
                        ]
                    ).all()
                ),
            },
        ]
    )

    print()
    print(
        "VALIDATION"
    )
    print(
        "-" * 132
    )
    print(
        validation.to_string(
            index=False
        )
    )

    if not validation[
        "passed"
    ].all():
        raise RuntimeError(
            "Controlled RV/LV bakeoff "
            "validation FAILED."
        )

    predictions.to_csv(
        OUTPUT_DIR
        / "rv_lv_selection_predictions.csv",
        index=False,
    )

    all_summary.to_csv(
        OUTPUT_DIR
        / "rv_lv_selection_all_cycles_summary.csv",
        index=False,
    )

    midterm_summary.to_csv(
        OUTPUT_DIR
        / "rv_lv_selection_midterm_summary.csv",
        index=False,
    )

    all_horizons.to_csv(
        OUTPUT_DIR
        / "rv_lv_selection_all_cycles_horizons.csv",
        index=False,
    )

    midterm_horizons.to_csv(
        OUTPUT_DIR
        / "rv_lv_selection_midterm_horizons.csv",
        index=False,
    )

    validation.to_csv(
        OUTPUT_DIR
        / "validation.csv",
        index=False,
    )

    print()
    print(
        "Controlled RV/LV selection "
        "bakeoff validation PASSED."
    )

    print()
    print(
        "Outputs:"
    )

    for path in sorted(
        OUTPUT_DIR.glob(
            "*.csv"
        )
    ):
        print(
            "  -",
            path.relative_to(
                ROOT
            ),
        )


if __name__ == "__main__":
    main()
