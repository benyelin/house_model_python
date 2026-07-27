#!/usr/bin/env python3
"""
Jointly calibrate House Election Day uncertainty scale and correlation.

This script evaluates two parameters:

    1. marginal race-level posterior total error SD
    2. chamber-wide correlated error SD

For every valid combination, residual district-specific uncertainty is
derived using variance-preserving accounting:

    residual_sd = sqrt(total_sd**2 - correlated_sd**2)

The simulator and scoring functions are imported from the validated
fixed-total correlated-uncertainty sweep.

Common random numbers are used within each historical cycle: every
parameter combination receives the same cycle-specific random seed.
This reduces Monte Carlo noise in comparisons between allocations.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from historical.house.backtests import (  # noqa: E402
    run_house_correlated_uncertainty_sweep as base,
)
from historical.house.backtests import (  # noqa: E402
    run_house_production_replay as replay,
)


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "joint_uncertainty_calibration"
)

LEGACY_TOTAL_SD = math.sqrt(
    5.0625 ** 2
    + 6.1875 ** 2
)

CURRENT_PRODUCTION_TOTAL_SD = 6.75
CURRENT_PRODUCTION_CORRELATED_SD = math.sqrt(23.203125)
CURRENT_PRODUCTION_RESIDUAL_SD = math.sqrt(
    CURRENT_PRODUCTION_TOTAL_SD ** 2
    - CURRENT_PRODUCTION_CORRELATED_SD ** 2
)

DEFAULT_POSTERIOR_TOTAL_GRID = (
    6.00,
    6.25,
    6.50,
    6.75,
    7.00,
    7.25,
    7.50,
    7.75,
    LEGACY_TOTAL_SD,
)

DEFAULT_CORRELATED_GRID = (
    0.0,
    0.5,
    1.0,
    1.5,
    1.9,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    CURRENT_PRODUCTION_CORRELATED_SD,
    5.0,
    5.5,
    6.0,
    6.5,
)


class JointSweepError(RuntimeError):
    """Raised when joint uncertainty calibration is invalid."""


def build_allocations() -> list[dict[str, float]]:
    """Construct all valid unique total/correlated allocations."""
    rows: list[dict[str, float]] = []
    seen: set[tuple[float, float]] = set()

    for posterior_total_sd in DEFAULT_POSTERIOR_TOTAL_GRID:
        posterior_total_sd = float(posterior_total_sd)

        if posterior_total_sd <= 0.0:
            raise JointSweepError(
                "Posterior total SD values must be positive."
            )

        for correlated_sd in DEFAULT_CORRELATED_GRID:
            correlated_sd = float(correlated_sd)

            if correlated_sd < 0.0:
                raise JointSweepError(
                    "Correlated SD values must be nonnegative."
                )

            if correlated_sd > posterior_total_sd + 1e-12:
                continue

            key = (
                round(posterior_total_sd, 12),
                round(correlated_sd, 12),
            )

            if key in seen:
                continue

            seen.add(key)

            residual_sd = math.sqrt(
                max(
                    posterior_total_sd ** 2
                    - correlated_sd ** 2,
                    0.0,
                )
            )

            rows.append(
                {
                    "posterior_total_error_sd": (
                        posterior_total_sd
                    ),
                    "correlated_error_sd": correlated_sd,
                    "residual_district_error_sd": residual_sd,
                }
            )

    if not rows:
        raise JointSweepError(
            "No valid uncertainty allocations were generated."
        )

    return rows


def is_current_production_allocation(
    posterior_total_sd: float,
    correlated_sd: float,
) -> bool:
    return bool(
        math.isclose(
            posterior_total_sd,
            CURRENT_PRODUCTION_TOTAL_SD,
            abs_tol=1e-9,
        )
        and math.isclose(
            correlated_sd,
            CURRENT_PRODUCTION_CORRELATED_SD,
            abs_tol=1e-9,
        )
    )


def is_legacy_total_benchmark(
    posterior_total_sd: float,
    correlated_sd: float,
) -> bool:
    return bool(
        math.isclose(
            posterior_total_sd,
            LEGACY_TOTAL_SD,
            abs_tol=1e-9,
        )
        and math.isclose(
            correlated_sd,
            2.0,
            abs_tol=1e-9,
        )
    )


def summarize_joint(
    cycle_results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize each total/correlated allocation across cycles."""
    rows: list[dict[str, Any]] = []

    group_columns = [
        "allocation_id",
        "posterior_total_error_sd",
        "correlated_error_sd",
        "residual_district_error_sd",
        "is_current_production_allocation",
        "is_legacy_total_best_smoke_benchmark",
    ]

    for keys, frame in cycle_results.groupby(
        group_columns,
        sort=True,
        dropna=False,
    ):
        (
            allocation_id,
            posterior_total_sd,
            correlated_sd,
            residual_sd,
            is_current,
            is_legacy_benchmark,
        ) = keys

        rows.append(
            {
                "allocation_id": allocation_id,
                "posterior_total_error_sd": float(
                    posterior_total_sd
                ),
                "correlated_error_sd": float(
                    correlated_sd
                ),
                "residual_district_error_sd": float(
                    residual_sd
                ),
                "correlated_variance_share": float(
                    correlated_sd ** 2
                    / posterior_total_sd ** 2
                ),
                "is_current_production_allocation": bool(
                    is_current
                ),
                "is_legacy_total_best_smoke_benchmark": bool(
                    is_legacy_benchmark
                ),
                "cycles": int(frame["cycle"].nunique()),
                "mean_crps": float(frame["crps"].mean()),
                "mean_realized_seat_log_loss": float(
                    frame[
                        "realized_seat_log_loss"
                    ].mean()
                ),
                "mean_absolute_expected_seat_error": float(
                    frame[
                        "absolute_expected_seat_error"
                    ].mean()
                ),
                "mean_absolute_median_seat_error": float(
                    frame[
                        "absolute_median_seat_error"
                    ].mean()
                ),
                "rmse_expected_seat_error": float(
                    np.sqrt(
                        np.mean(
                            frame["expected_seat_error"] ** 2
                        )
                    )
                ),
                "mean_simulation_dem_seat_sd": float(
                    frame[
                        "simulation_dem_seat_sd"
                    ].mean()
                ),
                "mean_interval_50_width": float(
                    frame["interval_50_width"].mean()
                ),
                "mean_interval_80_width": float(
                    frame["interval_80_width"].mean()
                ),
                "mean_interval_95_width": float(
                    frame["interval_95_width"].mean()
                ),
                "coverage_50": float(
                    frame["actual_in_50_interval"].mean()
                ),
                "coverage_80": float(
                    frame["actual_in_80_interval"].mean()
                ),
                "coverage_95": float(
                    frame["actual_in_95_interval"].mean()
                ),
                "mean_dem_control_probability": float(
                    frame[
                        "dem_control_probability"
                    ].mean()
                ),
            }
        )

    summary = pd.DataFrame(rows)

    ranking_specs = (
        ("mean_crps", "crps_rank"),
        (
            "mean_realized_seat_log_loss",
            "seat_log_loss_rank",
        ),
        (
            "mean_absolute_median_seat_error",
            "median_error_rank",
        ),
    )

    for metric, rank_column in ranking_specs:
        summary[rank_column] = (
            summary[metric]
            .rank(
                method="min",
                ascending=True,
            )
            .astype(int)
        )

    summary["composite_rank_score"] = (
        summary["crps_rank"]
        + summary["seat_log_loss_rank"]
        + summary["median_error_rank"]
    )

    summary["recommended"] = False

    best_index = (
        summary.sort_values(
            [
                "composite_rank_score",
                "mean_crps",
                "mean_realized_seat_log_loss",
                "posterior_total_error_sd",
                "correlated_error_sd",
            ],
            kind="mergesort",
        )
        .index[0]
    )

    summary.loc[best_index, "recommended"] = True

    return summary.sort_values(
        [
            "composite_rank_score",
            "mean_crps",
            "posterior_total_error_sd",
            "correlated_error_sd",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def profile_best_by_total(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Return the best correlated allocation for each total SD."""
    ranked = summary.sort_values(
        [
            "posterior_total_error_sd",
            "composite_rank_score",
            "mean_crps",
            "mean_realized_seat_log_loss",
            "correlated_error_sd",
        ],
        kind="mergesort",
    )

    return (
        ranked.groupby(
            "posterior_total_error_sd",
            as_index=False,
            sort=True,
        )
        .head(1)
        .reset_index(drop=True)
    )


def profile_best_by_correlation(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Return the best total SD for each correlated SD."""
    ranked = summary.sort_values(
        [
            "correlated_error_sd",
            "composite_rank_score",
            "mean_crps",
            "mean_realized_seat_log_loss",
            "posterior_total_error_sd",
        ],
        kind="mergesort",
    )

    return (
        ranked.groupby(
            "correlated_error_sd",
            as_index=False,
            sort=True,
        )
        .head(1)
        .reset_index(drop=True)
    )


def validate_outputs(
    cycle_results: pd.DataFrame,
    summary: pd.DataFrame,
    allocations: list[dict[str, float]],
) -> list[str]:
    """Validate joint-sweep dimensions and benchmark coverage."""
    checks: list[str] = []

    expected_allocations = len(allocations)
    expected_cycle_rows = (
        expected_allocations
        * len(replay.SUPPORTED_CYCLES)
    )

    if len(cycle_results) != expected_cycle_rows:
        raise JointSweepError(
            f"Expected {expected_cycle_rows} cycle rows; "
            f"found {len(cycle_results)}."
        )

    checks.append(
        f"PASS: cycle rows = {expected_cycle_rows}"
    )

    if len(summary) != expected_allocations:
        raise JointSweepError(
            f"Expected {expected_allocations} allocation rows; "
            f"found {len(summary)}."
        )

    checks.append(
        f"PASS: allocation rows = {expected_allocations}"
    )

    if cycle_results.duplicated(
        ["allocation_id", "cycle"]
    ).any():
        raise JointSweepError(
            "Duplicate allocation/cycle keys found."
        )

    checks.append(
        "PASS: allocation/cycle keys are unique"
    )

    reconstructed = np.sqrt(
        cycle_results["correlated_error_sd"] ** 2
        + cycle_results[
            "residual_district_error_sd"
        ] ** 2
    )

    max_reconstruction_error = float(
        np.max(
            np.abs(
                reconstructed
                - cycle_results[
                    "posterior_total_error_sd"
                ]
            )
        )
    )

    if max_reconstruction_error > 1e-9:
        raise JointSweepError(
            "Variance reconstruction failed; maximum error "
            f"was {max_reconstruction_error:.12f}."
        )

    checks.append(
        "PASS: variance-preserving reconstruction"
    )

    if int(summary["recommended"].sum()) != 1:
        raise JointSweepError(
            "Exactly one allocation must be recommended."
        )

    checks.append(
        "PASS: exactly one allocation is recommended"
    )

    current_rows = summary.loc[
        summary["is_current_production_allocation"]
    ]

    if len(current_rows) != 1:
        raise JointSweepError(
            "Current production allocation is missing "
            "or duplicated."
        )

    checks.append(
        "PASS: current production allocation is represented"
    )

    legacy_rows = summary.loc[
        summary[
            "is_legacy_total_best_smoke_benchmark"
        ]
    ]

    if len(legacy_rows) != 1:
        raise JointSweepError(
            "Legacy-total 2.0 correlated benchmark is "
            "missing or duplicated."
        )

    checks.append(
        "PASS: legacy-total smoke benchmark is represented"
    )

    if summary[
        "posterior_total_error_sd"
    ].nunique() != len(DEFAULT_POSTERIOR_TOTAL_GRID):
        raise JointSweepError(
            "Not all posterior-total grid values are represented."
        )

    checks.append(
        "PASS: all posterior-total grid values are represented"
    )

    return checks


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Jointly sweep House posterior-total and "
            "correlated Election Day uncertainty."
        )
    )

    parser.add_argument(
        "--master-path",
        type=Path,
        default=replay.DEFAULT_MASTER_PATH,
    )

    parser.add_argument(
        "--candidate-war-path",
        type=Path,
        default=replay.DEFAULT_WAR_PATH,
    )

    parser.add_argument(
        "--candidate-quality-weight",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--sims",
        type=int,
        default=20000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260720,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--clean-output",
        action="store_true",
    )

    args = parser.parse_args()

    if args.sims <= 0:
        raise ValueError("--sims must be positive.")

    if not args.master_path.exists():
        raise FileNotFoundError(
            f"Historical master not found: "
            f"{args.master_path}"
        )

    if args.clean_output and args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    master = pd.read_csv(
        args.master_path,
        low_memory=False,
    )

    replay.validate_input(master)

    # Prepare each historical cycle once. The model-margin inputs are
    # invariant across uncertainty allocations, so rebuilding them
    # inside the allocation loop is unnecessary and causes hundreds
    # of repeated reads of the same historical input files.
    prepared_cycles: dict[
        int,
        tuple[pd.DataFrame, pd.Series, str],
    ] = {}

    print("Preparing historical cycles once...")

    for cycle in replay.SUPPORTED_CYCLES:
        (
            prepared_df,
            prepared_model_margin,
            prepared_forecast_source,
        ) = replay.prepare_cycle(
            master=master,
            cycle=cycle,
            candidate_quality_weight=(
                args.candidate_quality_weight
            ),
            candidate_war_path=(
                args.candidate_war_path
            ),
        )

        prepared_cycles[int(cycle)] = (
            prepared_df,
            prepared_model_margin,
            prepared_forecast_source,
        )

        print(
            f"  {cycle}: "
            f"{len(prepared_df)} districts prepared"
        )

    if set(prepared_cycles) != {
        int(cycle)
        for cycle in replay.SUPPORTED_CYCLES
    }:
        raise JointSweepError(
            "Prepared-cycle cache does not match "
            "the supported historical cycles."
        )

    allocations = build_allocations()
    cycle_rows: list[dict[str, Any]] = []

    print(
        f"Valid joint allocations: {len(allocations)}"
    )
    print(
        "Common random numbers: enabled within each cycle"
    )
    print(
        "Historical model inputs: cached once per cycle"
    )

    for allocation_number, allocation in enumerate(
        allocations,
        start=1,
    ):
        posterior_total_sd = float(
            allocation["posterior_total_error_sd"]
        )
        correlated_sd = float(
            allocation["correlated_error_sd"]
        )
        residual_sd = float(
            allocation["residual_district_error_sd"]
        )

        allocation_id = (
            f"total_{posterior_total_sd:.4f}"
            f"_correlated_{correlated_sd:.4f}"
            f"_residual_{residual_sd:.4f}"
        )

        print()
        print("=" * 76)
        print(
            f"Allocation {allocation_number}/"
            f"{len(allocations)}"
        )
        print(
            f"Posterior total SD: {posterior_total_sd:.4f}"
        )
        print(
            f"Correlated SD: {correlated_sd:.4f}"
        )
        print(
            f"Residual district SD: {residual_sd:.4f}"
        )
        print("=" * 76)

        for cycle_index, cycle in enumerate(
            replay.SUPPORTED_CYCLES
        ):
            (
                cached_df,
                cached_model_margin,
                forecast_source,
            ) = prepared_cycles[int(cycle)]

            # Use defensive copies so future simulator changes cannot
            # accidentally mutate the cache shared by allocations.
            df = cached_df.copy()
            model_margin = cached_model_margin.copy()

            # Deliberately independent of allocation_number.
            # Every allocation receives the same random standard-normal
            # streams within a cycle.
            simulation_seed = (
                int(args.seed)
                + cycle_index * 1_000
                + int(cycle)
            )

            simulated_seats = (
                base.simulate_chamber_distribution(
                    df=df,
                    model_margin=model_margin,
                    correlated_sd=correlated_sd,
                    residual_district_sd=residual_sd,
                    n_sims=args.sims,
                    seed=simulation_seed,
                )
            )

            actual_seats = base.actual_dem_seats(df)

            metrics = base.score_distribution(
                simulated_seats=simulated_seats,
                actual_seats=actual_seats,
            )

            row = {
                "allocation_id": allocation_id,
                "cycle": int(cycle),
                "forecast_source": forecast_source,
                "posterior_total_error_sd": (
                    posterior_total_sd
                ),
                "correlated_error_sd": correlated_sd,
                "residual_district_error_sd": residual_sd,
                "correlated_variance_share": float(
                    correlated_sd ** 2
                    / posterior_total_sd ** 2
                ),
                "is_current_production_allocation": (
                    is_current_production_allocation(
                        posterior_total_sd,
                        correlated_sd,
                    )
                ),
                "is_legacy_total_best_smoke_benchmark": (
                    is_legacy_total_benchmark(
                        posterior_total_sd,
                        correlated_sd,
                    )
                ),
                "n_sims": int(args.sims),
                "seed": int(simulation_seed),
                **metrics,
            }

            cycle_rows.append(row)

            print(
                f"{cycle}: "
                f"actual={actual_seats}, "
                f"mean={metrics['expected_dem_seats']:.2f}, "
                f"SD={metrics['simulation_dem_seat_sd']:.2f}, "
                f"CRPS={metrics['crps']:.3f}, "
                f"log={metrics['realized_seat_log_loss']:.3f}"
            )

    cycle_results = pd.DataFrame(cycle_rows)
    summary = summarize_joint(cycle_results)

    best_by_total = profile_best_by_total(summary)
    best_by_correlation = profile_best_by_correlation(
        summary
    )

    validation_checks = validate_outputs(
        cycle_results=cycle_results,
        summary=summary,
        allocations=allocations,
    )

    ranking_columns = [
        "recommended",
        "posterior_total_error_sd",
        "correlated_error_sd",
        "residual_district_error_sd",
        "correlated_variance_share",
        "mean_crps",
        "mean_realized_seat_log_loss",
        "mean_absolute_median_seat_error",
        "mean_simulation_dem_seat_sd",
        "coverage_50",
        "coverage_80",
        "coverage_95",
        "composite_rank_score",
        "is_current_production_allocation",
        "is_legacy_total_best_smoke_benchmark",
    ]

    ranking = summary[ranking_columns].copy()

    print()
    print("=" * 76)
    print("House joint uncertainty ranking")
    print("=" * 76)
    print(
        ranking.head(30).to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    recommended = summary.loc[
        summary["recommended"]
    ].iloc[0]

    print()
    print("Recommended joint allocation")
    print("-" * 76)
    print(
        "Posterior total SD: "
        f"{recommended['posterior_total_error_sd']:.4f}"
    )
    print(
        "Correlated SD: "
        f"{recommended['correlated_error_sd']:.4f}"
    )
    print(
        "Residual district SD: "
        f"{recommended['residual_district_error_sd']:.4f}"
    )
    print(
        "Correlated variance share: "
        f"{recommended['correlated_variance_share']:.4f}"
    )
    print(
        f"Mean CRPS: {recommended['mean_crps']:.4f}"
    )
    print(
        "Mean realized-seat log loss: "
        f"{recommended['mean_realized_seat_log_loss']:.4f}"
    )
    print(
        "Mean simulated seat SD: "
        f"{recommended['mean_simulation_dem_seat_sd']:.4f}"
    )

    cycle_path = (
        args.output_dir
        / "house_joint_uncertainty_by_cycle.csv"
    )
    summary_path = (
        args.output_dir
        / "house_joint_uncertainty_summary.csv"
    )
    ranking_path = (
        args.output_dir
        / "house_joint_uncertainty_ranking.csv"
    )
    by_total_path = (
        args.output_dir
        / "house_joint_uncertainty_best_by_total.csv"
    )
    by_correlation_path = (
        args.output_dir
        / "house_joint_uncertainty_best_by_correlation.csv"
    )
    config_path = (
        args.output_dir
        / "house_joint_uncertainty_config.json"
    )
    validation_path = (
        args.output_dir
        / "house_joint_uncertainty_validation.txt"
    )

    cycle_results.to_csv(
        cycle_path,
        index=False,
    )
    summary.to_csv(
        summary_path,
        index=False,
    )
    ranking.to_csv(
        ranking_path,
        index=False,
    )
    best_by_total.to_csv(
        by_total_path,
        index=False,
    )
    best_by_correlation.to_csv(
        by_correlation_path,
        index=False,
    )

    config = {
        "supported_cycles": list(
            replay.SUPPORTED_CYCLES
        ),
        "posterior_total_sd_grid": [
            float(value)
            for value in DEFAULT_POSTERIOR_TOTAL_GRID
        ],
        "correlated_sd_grid": [
            float(value)
            for value in DEFAULT_CORRELATED_GRID
        ],
        "valid_allocations": len(allocations),
        "n_sims": int(args.sims),
        "seed": int(args.seed),
        "common_random_numbers": True,
        "historical_cycles_cached": True,
        "candidate_quality_weight": float(
            args.candidate_quality_weight
        ),
        "current_production_total_sd": (
            CURRENT_PRODUCTION_TOTAL_SD
        ),
        "current_production_correlated_sd": (
            CURRENT_PRODUCTION_CORRELATED_SD
        ),
        "current_production_residual_sd": (
            CURRENT_PRODUCTION_RESIDUAL_SD
        ),
        "legacy_total_sd": LEGACY_TOTAL_SD,
        "selection_method": (
            "minimum sum of CRPS rank, realized-seat "
            "log-loss rank, and absolute median-seat-error rank"
        ),
    }

    config_path.write_text(
        json.dumps(
            config,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    validation_text = (
        "House Joint Uncertainty Sweep Validation\n"
        + "=" * 48
        + "\n"
        + "\n".join(validation_checks)
        + "\n\nVALIDATION PASSED\n"
    )

    validation_path.write_text(validation_text)

    print()
    print(validation_text)

    print("Wrote:")
    for path in (
        cycle_path,
        summary_path,
        ranking_path,
        by_total_path,
        by_correlation_path,
        config_path,
        validation_path,
    ):
        print(f"  {path}")


if __name__ == "__main__":
    main()
