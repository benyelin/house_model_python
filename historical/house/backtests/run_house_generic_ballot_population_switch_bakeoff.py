from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

AGGREGATION_DIR = (
    ROOT
    / "historical/common/polling/aggregation"
)

if str(AGGREGATION_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(AGGREGATION_DIR),
    )

from generic_ballot_aggregation import (  # noqa: E402
    GenericBallotAggregationSpec,
    aggregate_generic_ballot_snapshot,
)


SNAPSHOT_PATH = (
    ROOT
    / "historical/common/polling/snapshots/"
    / "generic_ballot_polling_snapshots.csv"
)

HOUSE_SOURCE = (
    ROOT
    / "historical/house/raw/2022/source_downloads/"
    / "1976-2024-house.tab"
)

OUTPUT_DIR = (
    ROOT
    / "historical/house/backtests/outputs/"
    / "generic_ballot_population_switch_bakeoff"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CYCLES = [
    2018,
    2020,
    2022,
    2024,
]

MIDTERM_CYCLES = [
    2018,
    2022,
]

PRESIDENTIAL_CYCLES = [
    2020,
    2024,
]

SNAPSHOT_DAYS = [
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


DEM_LABELS = {
    "DEMOCRAT",
    "DEMOCRATIC",
    "DEMOCRATIC-FARMER-LABOR",
}

GOP_LABELS = {
    "REPUBLICAN",
}


def normalize_population(
    series: pd.Series,
) -> pd.Series:
    return (
        series
        .astype("string")
        .str.strip()
        .str.lower()
    )


def production_spec():
    """
    Exact validated generic-ballot aggregation
    architecture, except population is controlled
    externally by filtering the snapshot.
    """

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


def load_snapshots() -> pd.DataFrame:
    snapshots = pd.read_csv(
        SNAPSHOT_PATH,
        low_memory=False,
    )

    snapshots[
        "cycle"
    ] = pd.to_numeric(
        snapshots["cycle"],
        errors="coerce",
    )

    snapshots[
        "snapshot_days_before_election"
    ] = pd.to_numeric(
        snapshots[
            "snapshot_days_before_election"
        ],
        errors="coerce",
    )

    snapshots[
        "population_norm"
    ] = normalize_population(
        snapshots["population"]
    )

    snapshots = snapshots.loc[
        snapshots["cycle"].isin(CYCLES)
        & snapshots[
            "snapshot_days_before_election"
        ].isin(SNAPSHOT_DAYS)
    ].copy()

    return snapshots


def parse_bool_false(
    series: pd.Series,
) -> pd.Series:
    """
    Return True where source values represent FALSE.
    """

    normalized = (
        series
        .astype("string")
        .str.strip()
        .str.upper()
    )

    return normalized.isin(
        [
            "FALSE",
            "0",
            "NO",
            "N",
        ]
    )


def build_house_targets() -> pd.DataFrame:
    """
    Construct national Democratic two-party House
    margin directly from the canonical 1976-2024
    House returns.

    Votes are attributed by reported ballot-line
    party. This avoids incorrectly assigning all
    fusion-ticket votes to one inferred party.
    """

    df = pd.read_csv(
        HOUSE_SOURCE,
        low_memory=False,
    )

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce",
    )

    df["candidatevotes"] = pd.to_numeric(
        df["candidatevotes"],
        errors="coerce",
    )

    df["party_norm"] = (
        df["party"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    df = df.loc[
        df["year"].isin(CYCLES)
    ].copy()

    df = df.loc[
        df["stage"]
        .astype("string")
        .str.upper()
        .eq("GEN")
    ].copy()

    df = df.loc[
        parse_bool_false(
            df["special"]
        )
    ].copy()

    df = df.loc[
        df["mode"]
        .astype("string")
        .str.upper()
        .eq("TOTAL")
    ].copy()

    df["major_party"] = np.select(
        [
            df["party_norm"].isin(
                DEM_LABELS
            ),
            df["party_norm"].isin(
                GOP_LABELS
            ),
        ],
        [
            "D",
            "R",
        ],
        default="OTHER",
    )

    major = df.loc[
        df["major_party"].isin(
            ["D", "R"]
        )
    ].copy()

    national = (
        major.groupby(
            [
                "year",
                "major_party",
            ],
            as_index=False,
        )["candidatevotes"]
        .sum()
        .pivot(
            index="year",
            columns="major_party",
            values="candidatevotes",
        )
        .reset_index()
        .rename(
            columns={
                "D": "dem_votes",
                "R": "gop_votes",
            }
        )
    )

    national.columns.name = None

    national[
        "actual_house_margin_dem"
    ] = (
        (
            national["dem_votes"]
            - national["gop_votes"]
        )
        / (
            national["dem_votes"]
            + national["gop_votes"]
        )
        * 100.0
    )

    national = national.rename(
        columns={
            "year": "cycle",
        }
    )

    return national[
        [
            "cycle",
            "dem_votes",
            "gop_votes",
            "actual_house_margin_dem",
        ]
    ].sort_values(
        "cycle"
    )


def aggregate_one(
    snapshot: pd.DataFrame,
    population: str | None,
):
    """
    population:
        None -> current validated all-population rule
        "rv" -> RV-only
        "lv" -> LV-only
    """

    if population is None:
        selected = snapshot.copy()
        label = "production_all"
    else:
        selected = snapshot.loc[
            snapshot[
                "population_norm"
            ].eq(population)
        ].copy()

        label = population

    if selected.empty:
        return None

    result = aggregate_generic_ballot_snapshot(
        selected,
        production_spec(),
    )

    return {
        "population_rule": label,
        "estimate_margin_dem": float(
            result.estimate_margin_dem
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
        "weighted_mean_poll_age_days": float(
            result.weighted_mean_poll_age_days
        ),
    }


def build_population_estimates(
    snapshots: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    target_lookup = (
        targets.set_index(
            "cycle"
        )[
            "actual_house_margin_dem"
        ]
        .to_dict()
    )

    for cycle in CYCLES:
        for days_out in SNAPSHOT_DAYS:

            snapshot = snapshots.loc[
                snapshots["cycle"].eq(cycle)
                & snapshots[
                    "snapshot_days_before_election"
                ].eq(days_out)
            ].copy()

            if snapshot.empty:
                raise RuntimeError(
                    f"Missing snapshot: "
                    f"{cycle} D{days_out}"
                )

            for population in [
                None,
                "rv",
                "lv",
            ]:
                result = aggregate_one(
                    snapshot,
                    population,
                )

                if result is None:
                    raise RuntimeError(
                        f"No usable {population} "
                        f"polling for {cycle} D{days_out}"
                    )

                estimate = result[
                    "estimate_margin_dem"
                ]

                actual = float(
                    target_lookup[
                        cycle
                    ]
                )

                environment = (
                    ENVIRONMENT_COEFFICIENT
                    * estimate
                )

                row = {
                    "cycle": cycle,
                    "election_type": (
                        "midterm"
                        if cycle in MIDTERM_CYCLES
                        else "presidential"
                    ),
                    "days_out": days_out,
                    "actual_house_margin_dem": actual,
                    "environment_coefficient": (
                        ENVIRONMENT_COEFFICIENT
                    ),
                    "environment_estimate_dem": (
                        environment
                    ),
                    "raw_error_dem": (
                        estimate
                        - actual
                    ),
                    "raw_absolute_error": abs(
                        estimate
                        - actual
                    ),
                    "environment_error_dem": (
                        environment
                        - actual
                    ),
                    "environment_absolute_error": abs(
                        environment
                        - actual
                    ),
                    **result,
                }

                rows.append(
                    row
                )

    return pd.DataFrame(
        rows
    )


def summarize_population_by_horizon(
    estimates: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    subsets = {
        "all_cycles": CYCLES,
        "midterms_only": MIDTERM_CYCLES,
        "presidential_only": PRESIDENTIAL_CYCLES,
    }

    for subset_name, cycles in subsets.items():
        subset = estimates.loc[
            estimates["cycle"].isin(
                cycles
            )
        ]

        for days_out in SNAPSHOT_DAYS:
            day = subset.loc[
                subset["days_out"].eq(
                    days_out
                )
            ]

            for population_rule in [
                "production_all",
                "rv",
                "lv",
            ]:
                group = day.loc[
                    day[
                        "population_rule"
                    ].eq(
                        population_rule
                    )
                ]

                errors = group[
                    "environment_error_dem"
                ].to_numpy(
                    dtype=float
                )

                raw_errors = group[
                    "raw_error_dem"
                ].to_numpy(
                    dtype=float
                )

                rows.append(
                    {
                        "subset": subset_name,
                        "days_out": days_out,
                        "population_rule": (
                            population_rule
                        ),
                        "cycles": len(group),
                        "environment_mae": float(
                            np.mean(
                                np.abs(
                                    errors
                                )
                            )
                        ),
                        "environment_rmse": float(
                            np.sqrt(
                                np.mean(
                                    errors**2
                                )
                            )
                        ),
                        "environment_bias_dem": float(
                            np.mean(
                                errors
                            )
                        ),
                        "raw_mae": float(
                            np.mean(
                                np.abs(
                                    raw_errors
                                )
                            )
                        ),
                        "mean_unique_polls": float(
                            group[
                                "unique_polls"
                            ].mean()
                        ),
                        "minimum_unique_polls": int(
                            group[
                                "unique_polls"
                            ].min()
                        ),
                    }
                )

    return pd.DataFrame(
        rows
    )


def strategy_population(
    strategy: str,
    days_out: int,
):
    if strategy == "production_all":
        return "production_all"

    if strategy == "rv_throughout":
        return "rv"

    if strategy == "lv_throughout":
        return "lv"

    if strategy.startswith(
        "switch_d"
    ):
        switch_day = int(
            strategy.replace(
                "switch_d",
                "",
            )
        )

        # At the switch point and thereafter
        # (closer to Election Day), use LV.
        if days_out <= switch_day:
            return "lv"

        return "rv"

    raise ValueError(
        f"Unknown strategy: {strategy}"
    )


def build_strategy_results(
    estimates: pd.DataFrame,
) -> pd.DataFrame:

    strategies = [
        "production_all",
        "rv_throughout",
        "lv_throughout",
        "switch_d120",
        "switch_d90",
        "switch_d75",
        "switch_d60",
        "switch_d45",
        "switch_d30",
        "switch_d21",
        "switch_d14",
        "switch_d7",
    ]

    lookup = estimates.set_index(
        [
            "cycle",
            "days_out",
            "population_rule",
        ]
    )

    rows = []

    for strategy in strategies:
        for cycle in CYCLES:
            for days_out in SNAPSHOT_DAYS:

                population_rule = (
                    strategy_population(
                        strategy,
                        days_out,
                    )
                )

                source = lookup.loc[
                    (
                        cycle,
                        days_out,
                        population_rule,
                    )
                ]

                rows.append(
                    {
                        "strategy": strategy,
                        "cycle": cycle,
                        "election_type": (
                            "midterm"
                            if cycle in MIDTERM_CYCLES
                            else "presidential"
                        ),
                        "days_out": days_out,
                        "population_used": (
                            population_rule
                        ),
                        "estimate_margin_dem": float(
                            source[
                                "estimate_margin_dem"
                            ]
                        ),
                        "environment_estimate_dem": float(
                            source[
                                "environment_estimate_dem"
                            ]
                        ),
                        "actual_house_margin_dem": float(
                            source[
                                "actual_house_margin_dem"
                            ]
                        ),
                        "environment_error_dem": float(
                            source[
                                "environment_error_dem"
                            ]
                        ),
                        "environment_absolute_error": float(
                            source[
                                "environment_absolute_error"
                            ]
                        ),
                        "raw_absolute_error": float(
                            source[
                                "raw_absolute_error"
                            ]
                        ),
                    }
                )

    return pd.DataFrame(
        rows
    )


def summarize_strategies(
    strategy_results: pd.DataFrame,
) -> pd.DataFrame:

    subsets = {
        "all_cycles": CYCLES,
        "midterms_only": MIDTERM_CYCLES,
        "presidential_only": PRESIDENTIAL_CYCLES,
    }

    rows = []

    for subset_name, cycles in subsets.items():

        subset = strategy_results.loc[
            strategy_results[
                "cycle"
            ].isin(
                cycles
            )
        ]

        rv = subset.loc[
            subset["strategy"].eq(
                "rv_throughout"
            )
        ][
            [
                "cycle",
                "days_out",
                "environment_absolute_error",
            ]
        ].rename(
            columns={
                "environment_absolute_error":
                "rv_absolute_error"
            }
        )

        for strategy in (
            subset[
                "strategy"
            ].drop_duplicates()
        ):

            group = subset.loc[
                subset[
                    "strategy"
                ].eq(
                    strategy
                )
            ].copy()

            errors = group[
                "environment_error_dem"
            ].to_numpy(
                dtype=float
            )

            comparison = group.merge(
                rv,
                on=[
                    "cycle",
                    "days_out",
                ],
                how="left",
                validate="one_to_one",
            )

            comparison[
                "beats_rv"
            ] = (
                comparison[
                    "environment_absolute_error"
                ]
                < comparison[
                    "rv_absolute_error"
                ]
            )

            cycle_mae = (
                group.groupby(
                    "cycle"
                )[
                    "environment_absolute_error"
                ]
                .mean()
            )

            rv_cycle_mae = (
                subset.loc[
                    subset[
                        "strategy"
                    ].eq(
                        "rv_throughout"
                    )
                ]
                .groupby(
                    "cycle"
                )[
                    "environment_absolute_error"
                ]
                .mean()
            )

            aligned = pd.concat(
                [
                    cycle_mae.rename(
                        "candidate"
                    ),
                    rv_cycle_mae.rename(
                        "rv"
                    ),
                ],
                axis=1,
            ).dropna()

            rows.append(
                {
                    "subset": subset_name,
                    "strategy": strategy,
                    "observations": len(group),
                    "cycles": group[
                        "cycle"
                    ].nunique(),
                    "environment_mae": float(
                        np.mean(
                            np.abs(
                                errors
                            )
                        )
                    ),
                    "environment_rmse": float(
                        np.sqrt(
                            np.mean(
                                errors**2
                            )
                        )
                    ),
                    "environment_bias_dem": float(
                        np.mean(
                            errors
                        )
                    ),
                    "maximum_absolute_error": float(
                        np.max(
                            np.abs(
                                errors
                            )
                        )
                    ),
                    "observations_beating_rv": int(
                        comparison[
                            "beats_rv"
                        ].sum()
                    ),
                    "share_beating_rv": float(
                        comparison[
                            "beats_rv"
                        ].mean()
                    ),
                    "cycles_beating_rv": int(
                        (
                            aligned[
                                "candidate"
                            ]
                            < aligned[
                                "rv"
                            ]
                        ).sum()
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def print_horizon_table(
    horizon_summary: pd.DataFrame,
    subset: str,
):
    table = horizon_summary.loc[
        horizon_summary[
            "subset"
        ].eq(
            subset
        )
    ].pivot(
        index="days_out",
        columns="population_rule",
        values="environment_mae",
    ).reset_index()

    table = table.sort_values(
        "days_out",
        ascending=False,
    )

    if (
        "rv" in table.columns
        and "lv" in table.columns
    ):
        table[
            "lv_minus_rv_mae"
        ] = (
            table["lv"]
            - table["rv"]
        )

        table[
            "winner_rv_vs_lv"
        ] = np.where(
            table["lv"] < table["rv"],
            "LV",
            np.where(
                table["rv"] < table["lv"],
                "RV",
                "TIE",
            ),
        )

    print(
        table.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )


def main():
    print("=" * 132)
    print(
        "HOUSE GENERIC BALLOT — "
        "REGISTERED VS LIKELY VOTER "
        "SWITCHING BAKEOFF"
    )
    print("=" * 132)

    snapshots = load_snapshots()
    targets = build_house_targets()

    print()
    print("HOUSE NATIONAL TARGETS")
    print("-" * 132)

    print(
        targets.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    estimates = build_population_estimates(
        snapshots,
        targets,
    )

    horizon_summary = (
        summarize_population_by_horizon(
            estimates
        )
    )

    strategy_results = (
        build_strategy_results(
            estimates
        )
    )

    strategy_summary = (
        summarize_strategies(
            strategy_results
        )
    )

    print()
    print(
        "RV VS LV BY HORIZON — "
        "ALL FOUR CYCLES"
    )
    print("-" * 132)

    print_horizon_table(
        horizon_summary,
        "all_cycles",
    )

    print()
    print(
        "RV VS LV BY HORIZON — "
        "MIDTERMS ONLY (2018, 2022)"
    )
    print("-" * 132)

    print_horizon_table(
        horizon_summary,
        "midterms_only",
    )

    print()
    print(
        "RV VS LV BY HORIZON — "
        "PRESIDENTIAL YEARS ONLY "
        "(2020, 2024)"
    )
    print("-" * 132)

    print_horizon_table(
        horizon_summary,
        "presidential_only",
    )

    print()
    print(
        "SWITCHING STRATEGIES — "
        "ALL FOUR CYCLES"
    )
    print("-" * 132)

    all_strategy = (
        strategy_summary.loc[
            strategy_summary[
                "subset"
            ].eq(
                "all_cycles"
            )
        ]
        .sort_values(
            [
                "environment_mae",
                "environment_rmse",
            ]
        )
    )

    print(
        all_strategy.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    print()
    print(
        "SWITCHING STRATEGIES — "
        "MIDTERMS ONLY"
    )
    print("-" * 132)

    midterm_strategy = (
        strategy_summary.loc[
            strategy_summary[
                "subset"
            ].eq(
                "midterms_only"
            )
        ]
        .sort_values(
            [
                "environment_mae",
                "environment_rmse",
            ]
        )
    )

    print(
        midterm_strategy.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    print()
    print(
        "CYCLE-BY-CYCLE RV / LV MAE"
    )
    print("-" * 132)

    cycle_summary = (
        estimates.loc[
            estimates[
                "population_rule"
            ].isin(
                [
                    "rv",
                    "lv",
                    "production_all",
                ]
            )
        ]
        .groupby(
            [
                "cycle",
                "population_rule",
            ],
            as_index=False,
        )
        .agg(
            environment_mae=(
                "environment_absolute_error",
                "mean",
            ),
            environment_bias_dem=(
                "environment_error_dem",
                "mean",
            ),
        )
    )

    print(
        cycle_summary.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    print()
    print(
        "NEAR-CURRENT-CYCLE CHECKPOINTS "
        "(D90 / D75 / D60)"
    )
    print("-" * 132)

    near_current = (
        horizon_summary.loc[
            horizon_summary[
                "subset"
            ].eq(
                "all_cycles"
            )
            & horizon_summary[
                "days_out"
            ].isin(
                [
                    90,
                    75,
                    60,
                ]
            )
            & horizon_summary[
                "population_rule"
            ].isin(
                [
                    "rv",
                    "lv",
                    "production_all",
                ]
            )
        ]
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
    )

    print(
        near_current.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    estimates.to_csv(
        OUTPUT_DIR
        / "population_estimates.csv",
        index=False,
    )

    horizon_summary.to_csv(
        OUTPUT_DIR
        / "population_horizon_summary.csv",
        index=False,
    )

    strategy_results.to_csv(
        OUTPUT_DIR
        / "switch_strategy_predictions.csv",
        index=False,
    )

    strategy_summary.to_csv(
        OUTPUT_DIR
        / "switch_strategy_summary.csv",
        index=False,
    )

    cycle_summary.to_csv(
        OUTPUT_DIR
        / "population_cycle_summary.csv",
        index=False,
    )

    checks = pd.DataFrame(
        [
            {
                "check": (
                    "four_cycles_present"
                ),
                "passed": (
                    estimates[
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
                    estimates[
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
                        len(
                            SNAPSHOT_DAYS
                        )
                    )
                    .all()
                ),
            },
            {
                "check": (
                    "all_three_population_"
                    "rules_present"
                ),
                "passed": (
                    estimates[
                        [
                            "cycle",
                            "days_out",
                            "population_rule",
                        ]
                    ]
                    .drop_duplicates()
                    .shape[0]
                    == (
                        len(CYCLES)
                        * len(
                            SNAPSHOT_DAYS
                        )
                        * 3
                    )
                ),
            },
            {
                "check": (
                    "all_estimates_finite"
                ),
                "passed": bool(
                    np.isfinite(
                        estimates[
                            "estimate_margin_dem"
                        ]
                    ).all()
                ),
            },
            {
                "check": (
                    "all_targets_finite"
                ),
                "passed": bool(
                    np.isfinite(
                        estimates[
                            "actual_house_margin_dem"
                        ]
                    ).all()
                ),
            },
        ]
    )

    checks.to_csv(
        OUTPUT_DIR
        / "validation.csv",
        index=False,
    )

    print()
    print("VALIDATION")
    print("-" * 132)

    print(
        checks.to_string(
            index=False
        )
    )

    if not checks[
        "passed"
    ].all():
        raise RuntimeError(
            "RV/LV switching bakeoff "
            "validation failed."
        )

    print()
    print(
        "RV/LV switching bakeoff "
        "validation PASSED."
    )

    print()
    print("Outputs:")
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
