#!/usr/bin/env python3
"""
Decompose historical House production fundamentals by additive component.

This diagnostic imports the replay's leakage-safe preparation functions but
does not modify production calculations or canonical replay outputs.

Initial scope:
    2018, 2020, and 2022

The incomplete 2016 Florida presidential baseline is intentionally excluded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from historical.house.backtests.run_house_full_production_replay import (  # noqa: E402
    DEFAULT_MASTER_PATH,
    DEFAULT_WAR_PATH,
    build_production_fundamentals,
    prepare_cycle,
    validate_input,
)
from run_house_dynamic_uncertainty import normalize_fixed_control  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "production_component_decomposition"
)

DEFAULT_CYCLES = (2018, 2020, 2022)


STAGES = [
    (
        "baseline_only",
        "Partisan baseline",
    ),
    (
        "plus_uniform_environment",
        "+ Uniform national environment",
    ),
    (
        "plus_elasticity",
        "+ District elasticity",
    ),
    (
        "plus_state_environment",
        "+ State environment",
    ),
    (
        "plus_incumbency",
        "+ Incumbency",
    ),
    (
        "plus_candidate_quality",
        "+ Candidate quality",
    ),
    (
        "plus_special",
        "+ Special adjustments",
    ),
    (
        "final_fundamentals",
        "+ Poll spillover / final fundamentals",
    ),
]


COMPONENT_COLUMNS = [
    "district_partisan_baseline_dem",
    "house_national_environment_used_dem",
    "district_elasticity",
    "district_environment_adjustment_dem",
    "state_environment_adjustment_dem",
    "incumbency_adjustment_dem",
    "candidate_quality_adjustment_dem",
    "special_adjustment_dem",
    "poll_spillover_adjustment_dem",
    "fundamentals_margin_dem",
    "model_margin_dem",
]


def numeric(
    df: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> pd.Series:
    if column not in df.columns:
        return pd.Series(
            default,
            index=df.index,
            dtype=float,
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).fillna(default)


def add_decomposition_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    out = df.copy()

    baseline = numeric(
        out,
        "district_partisan_baseline_dem",
        default=np.nan,
    )

    uniform_environment = numeric(
        out,
        "house_national_environment_used_dem",
    )

    elasticity = numeric(
        out,
        "district_elasticity",
        default=1.0,
    )

    actual_environment = numeric(
        out,
        "district_environment_adjustment_dem",
    )

    state_environment = numeric(
        out,
        "state_environment_adjustment_dem",
    )

    incumbency = numeric(
        out,
        "incumbency_adjustment_dem",
    )

    candidate_quality = numeric(
        out,
        "candidate_quality_adjustment_dem",
    )

    special = numeric(
        out,
        "special_adjustment_dem",
    )

    poll_spillover = numeric(
        out,
        "poll_spillover_adjustment_dem",
    )

    out["baseline_only"] = baseline

    out["plus_uniform_environment"] = (
        out["baseline_only"]
        + uniform_environment
    )

    # This stage uses the actual production environment adjustment.
    # The incremental change from the prior stage is therefore:
    #
    # national_environment * (district_elasticity - 1)
    out["plus_elasticity"] = (
        out["baseline_only"]
        + actual_environment
    )

    out["plus_state_environment"] = (
        out["plus_elasticity"]
        + state_environment
    )

    out["plus_incumbency"] = (
        out["plus_state_environment"]
        + incumbency
    )

    out["plus_candidate_quality"] = (
        out["plus_incumbency"]
        + candidate_quality
    )

    out["plus_special"] = (
        out["plus_candidate_quality"]
        + special
    )

    out["final_fundamentals"] = (
        out["plus_special"]
        + poll_spillover
    )

    out["uniform_environment_component_dem"] = (
        uniform_environment
    )

    out["elasticity_component_dem"] = (
        actual_environment
        - uniform_environment
    )

    out["state_environment_component_dem"] = (
        state_environment
    )

    out["incumbency_component_dem"] = incumbency

    out["candidate_quality_component_dem"] = (
        candidate_quality
    )

    out["special_component_dem"] = special

    out["poll_spillover_component_dem"] = (
        poll_spillover
    )

    out["decomposition_identity_error"] = (
        out["final_fundamentals"]
        - numeric(
            out,
            "fundamentals_margin_dem",
            default=np.nan,
        )
    )

    out["elasticity_identity_error"] = (
        actual_environment
        - (
            uniform_environment
            * elasticity
        )
    )

    return out


def probability_from_margin(
    margin: pd.Series,
    fixed_error_sd: float,
) -> pd.Series:
    clipped = np.clip(
        pd.to_numeric(
            margin,
            errors="coerce",
        ),
        -100.0,
        100.0,
    )

    probability = 1.0 / (
        1.0
        + np.exp(
            -clipped
            / float(fixed_error_sd)
        )
    )

    return pd.Series(
        probability,
        index=margin.index,
        dtype=float,
    )


def stage_probability(
    df: pd.DataFrame,
    stage_column: str,
    fixed_error_sd: float,
) -> pd.Series:
    normalized = normalize_fixed_control(df)

    probability = probability_from_margin(
        normalized[stage_column],
        fixed_error_sd=fixed_error_sd,
    )

    fixed_party = (
        normalized["party_control_fixed"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    probability.loc[
        fixed_party.eq("D")
    ] = 1.0

    probability.loc[
        fixed_party.eq("R")
    ] = 0.0

    return probability


def summarize_cycle(
    df: pd.DataFrame,
    cycle: int,
    fixed_error_sd: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    previous_expected_seats: float | None = None
    previous_margin_column: str | None = None

    for stage_number, (
        stage_column,
        stage_label,
    ) in enumerate(STAGES, start=1):
        probability = stage_probability(
            df,
            stage_column=stage_column,
            fixed_error_sd=fixed_error_sd,
        )

        expected_seats = float(
            probability.sum()
        )

        margin = pd.to_numeric(
            df[stage_column],
            errors="coerce",
        )

        if previous_expected_seats is None:
            incremental_seats = np.nan
            mean_component_margin = np.nan
            median_component_margin = np.nan
            shifted_dem = np.nan
            shifted_gop = np.nan
        else:
            incremental_seats = (
                expected_seats
                - previous_expected_seats
            )

            component_margin = (
                margin
                - pd.to_numeric(
                    df[previous_margin_column],
                    errors="coerce",
                )
            )

            mean_component_margin = float(
                component_margin.mean()
            )

            median_component_margin = float(
                component_margin.median()
            )

            shifted_dem = int(
                component_margin.gt(1e-12).sum()
            )

            shifted_gop = int(
                component_margin.lt(-1e-12).sum()
            )

        rows.append(
            {
                "cycle": int(cycle),
                "stage_number": int(stage_number),
                "stage": stage_column,
                "stage_label": stage_label,
                "expected_dem_seats": expected_seats,
                "incremental_expected_dem_seats": (
                    incremental_seats
                ),
                "mean_stage_margin_dem": float(
                    margin.mean()
                ),
                "median_stage_margin_dem": float(
                    margin.median()
                ),
                "mean_component_margin_dem": (
                    mean_component_margin
                ),
                "median_component_margin_dem": (
                    median_component_margin
                ),
                "districts_shifted_dem": shifted_dem,
                "districts_shifted_gop": shifted_gop,
                "district_count": int(len(df)),
                "fixed_error_sd": float(
                    fixed_error_sd
                ),
            }
        )

        previous_expected_seats = expected_seats
        previous_margin_column = stage_column

    return pd.DataFrame(rows)


def print_cycle_summary(
    summary: pd.DataFrame,
    cycle: int,
) -> None:
    cycle_summary = summary.loc[
        summary["cycle"].eq(cycle)
    ].copy()

    display_columns = [
        "stage_label",
        "expected_dem_seats",
        "incremental_expected_dem_seats",
        "mean_component_margin_dem",
        "districts_shifted_dem",
        "districts_shifted_gop",
    ]

    print()
    print("=" * 88)
    print(f"{cycle} HOUSE FUNDAMENTALS COMPONENT DECOMPOSITION")
    print("=" * 88)
    print(
        cycle_summary[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:,.3f}"
            ),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose leakage-safe House production "
            "fundamentals by component."
        )
    )

    parser.add_argument(
        "--master-path",
        type=Path,
        default=DEFAULT_MASTER_PATH,
    )

    parser.add_argument(
        "--candidate-war-path",
        type=Path,
        default=DEFAULT_WAR_PATH,
    )

    parser.add_argument(
        "--candidate-quality-weight",
        type=float,
        default=0.0,
        help=(
            "Historical candidate-quality multiplier. "
            "Default matches replay v1: 0.0."
        ),
    )

    parser.add_argument(
        "--fixed-error-sd",
        type=float,
        default=6.5,
    )

    parser.add_argument(
        "--cycles",
        type=int,
        nargs="+",
        default=list(DEFAULT_CYCLES),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    if args.fixed_error_sd <= 0:
        raise ValueError(
            "--fixed-error-sd must be positive."
        )

    master = pd.read_csv(
        args.master_path,
        low_memory=False,
    )

    validate_input(master)

    district_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []

    for cycle in args.cycles:
        prepared, national_environment = prepare_cycle(
            master=master,
            cycle=int(cycle),
            candidate_quality_weight=float(
                args.candidate_quality_weight
            ),
            candidate_war_path=args.candidate_war_path,
        )

        production_df, _, _ = (
            build_production_fundamentals(
                df=prepared,
                cycle=int(cycle),
                national_environment=float(
                    national_environment
                ),
            )
        )

        decomposed = add_decomposition_columns(
            production_df
        )

        max_identity_error = float(
            decomposed[
                "decomposition_identity_error"
            ].abs().max()
        )

        max_elasticity_error = float(
            decomposed[
                "elasticity_identity_error"
            ].abs().max()
        )

        if (
            not np.isfinite(max_identity_error)
            or max_identity_error > 1e-9
        ):
            raise RuntimeError(
                f"Cycle {cycle} decomposition does not "
                "reproduce fundamentals_margin_dem. "
                f"Maximum error: {max_identity_error}"
            )

        if (
            not np.isfinite(max_elasticity_error)
            or max_elasticity_error > 1e-9
        ):
            raise RuntimeError(
                f"Cycle {cycle} environment adjustment "
                "does not equal environment × elasticity. "
                f"Maximum error: {max_elasticity_error}"
            )

        decomposed["forecast_cycle"] = int(cycle)

        keep_columns = [
            column
            for column in [
                "forecast_cycle",
                "cycle",
                "race_id",
                "district_id",
                "state",
                "district",
                "party_control_fixed",
                "general_election_party_structure",
                *COMPONENT_COLUMNS,
                *[
                    stage_column
                    for stage_column, _ in STAGES
                ],
                "uniform_environment_component_dem",
                "elasticity_component_dem",
                "state_environment_component_dem",
                "incumbency_component_dem",
                "candidate_quality_component_dem",
                "special_component_dem",
                "poll_spillover_component_dem",
                "decomposition_identity_error",
                "elasticity_identity_error",
            ]
            if column in decomposed.columns
        ]

        district_frames.append(
            decomposed[keep_columns].copy()
        )

        cycle_summary = summarize_cycle(
            decomposed,
            cycle=int(cycle),
            fixed_error_sd=float(
                args.fixed_error_sd
            ),
        )

        summary_frames.append(cycle_summary)

    district_output = pd.concat(
        district_frames,
        ignore_index=True,
    )

    summary_output = pd.concat(
        summary_frames,
        ignore_index=True,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    district_path = (
        args.output_dir
        / "house_production_component_decomposition_districts.csv"
    )

    summary_path = (
        args.output_dir
        / "house_production_component_decomposition_summary.csv"
    )

    district_output.to_csv(
        district_path,
        index=False,
    )

    summary_output.to_csv(
        summary_path,
        index=False,
    )

    for cycle in args.cycles:
        print_cycle_summary(
            summary_output,
            cycle=int(cycle),
        )

    print()
    print("VALIDATION PASSED")
    print(
        "Every cumulative stage reproduces the "
        "production additive formula."
    )
    print(
        "The final decomposition exactly matches "
        "fundamentals_margin_dem."
    )
    print()
    print(f"District output: {district_path}")
    print(f"Summary output:  {summary_path}")


if __name__ == "__main__":
    main()
