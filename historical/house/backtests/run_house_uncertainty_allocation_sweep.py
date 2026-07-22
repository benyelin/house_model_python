#!/usr/bin/env python3
"""
Calibrate the House Election Day uncertainty allocation.

This script holds the marginal race-level Election Day uncertainty
constant while sweeping the allocation between:

    - nationally correlated error
    - district-specific independent error

It reuses the leakage-safe model-margin and fixed-control preparation
from run_house_production_replay.py.

Allocation candidates are evaluated using the full simulated chamber
seat distribution rather than race-level probability metrics, which
are nearly invariant when marginal total uncertainty is held fixed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from historical.house.backtests import (  # noqa: E402
    run_house_production_replay as replay,
)


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "uncertainty_allocation_calibration"
)

DEFAULT_TOTAL_SD = math.sqrt(
    5.0625 ** 2
    + 6.1875 ** 2
)

DEFAULT_NATIONAL_GRID = (
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
    5.0,
    5.0625,
)

CURRENT_NATIONAL_SD = 5.0625
CURRENT_DISTRICT_SD = 6.1875

HOUSE_SEATS = 435
HOUSE_CONTROL_THRESHOLD = 218


class AllocationSweepError(RuntimeError):
    """Raised when uncertainty-allocation outputs are invalid."""


def empirical_crps(
    simulated_values: np.ndarray,
    actual_value: float,
) -> float:
    """
    Compute empirical CRPS efficiently.

    CRPS(F, y) =
        E|X - y| - 0.5 E|X - X'|

    The second expectation is calculated from the sorted sample
    without constructing a full pairwise-distance matrix.
    """
    values = np.asarray(
        simulated_values,
        dtype=float,
    )

    if values.ndim != 1 or len(values) == 0:
        raise AllocationSweepError(
            "CRPS requires a nonempty one-dimensional sample."
        )

    first_term = float(
        np.mean(
            np.abs(values - float(actual_value))
        )
    )

    sorted_values = np.sort(values)
    n = len(sorted_values)

    indices = np.arange(
        1,
        n + 1,
        dtype=float,
    )

    pairwise_expectation = float(
        (
            2.0
            / (n ** 2)
            * np.sum(
                (
                    2.0 * indices
                    - n
                    - 1.0
                )
                * sorted_values
            )
        )
    )

    return first_term - 0.5 * pairwise_expectation


def smoothed_realized_seat_log_loss(
    simulated_seats: np.ndarray,
    actual_seats: int,
) -> tuple[float, float]:
    """
    Return the smoothed probability and negative log probability of
    the realized seat count.

    Add-one smoothing is applied over all possible 0-435 Democratic
    seat totals so a finite score is available even when the exact
    realized count is absent from a finite Monte Carlo sample.
    """
    simulated = np.asarray(
        simulated_seats,
        dtype=int,
    )

    realized_count = int(
        np.sum(simulated == int(actual_seats))
    )

    possible_outcomes = HOUSE_SEATS + 1

    probability = (
        realized_count + 1.0
    ) / (
        len(simulated) + possible_outcomes
    )

    return (
        float(probability),
        -float(np.log(probability)),
    )


def actual_dem_seats(
    df: pd.DataFrame,
) -> int:
    winners = (
        df["actual_winner"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    value = int(winners.eq("D").sum())

    if not 0 <= value <= HOUSE_SEATS:
        raise AllocationSweepError(
            f"Invalid actual Democratic seat count: {value}."
        )

    return value


def simulate_chamber_distribution(
    df: pd.DataFrame,
    model_margin: pd.Series,
    national_sd: float,
    district_sd: float,
    n_sims: int,
    seed: int,
) -> np.ndarray:
    """
    Run the same national-plus-district Election Day simulation used
    by production replay v1.
    """
    replay_df = replay.normalize_fixed_control(
        df.copy()
    )

    replay_df["poll_count"] = 0

    poll_counts = pd.to_numeric(
        replay_df["poll_count"],
        errors="coerce",
    ).fillna(0)

    settings = replay.read_settings()

    district_multipliers = poll_counts.apply(
        lambda value: replay.polling_district_multiplier(
            value,
            settings,
        )
    ).to_numpy(dtype=float)

    district_sds = (
        float(district_sd)
        * district_multipliers
    )

    rng = np.random.default_rng(seed)

    national_errors = rng.normal(
        0.0,
        float(national_sd),
        size=n_sims,
    )

    district_errors = rng.normal(
        0.0,
        district_sds.reshape(
            1,
            len(replay_df),
        ),
        size=(
            n_sims,
            len(replay_df),
        ),
    )

    margins = pd.to_numeric(
        model_margin,
        errors="raise",
    ).to_numpy(dtype=float)

    simulated_margins = (
        margins.reshape(
            1,
            len(replay_df),
        )
        + national_errors.reshape(
            n_sims,
            1,
        )
        + district_errors
    )

    fixed = (
        replay_df["party_control_fixed"]
        .fillna("")
        .astype(str)
        .str.upper()
        .to_numpy()
    )

    dem_wins = simulated_margins > 0.0
    dem_wins[:, fixed == "D"] = True
    dem_wins[:, fixed == "R"] = False

    return dem_wins.sum(axis=1).astype(int)


def score_distribution(
    simulated_seats: np.ndarray,
    actual_seats: int,
) -> dict[str, float]:
    simulated = np.asarray(
        simulated_seats,
        dtype=float,
    )

    expected_seats = float(
        np.mean(simulated)
    )

    median_seats = float(
        np.median(simulated)
    )

    seat_sd = float(
        np.std(
            simulated,
            ddof=1,
        )
    )

    percentiles = {
        "p025": float(
            np.percentile(
                simulated,
                2.5,
            )
        ),
        "p10": float(
            np.percentile(
                simulated,
                10,
            )
        ),
        "p25": float(
            np.percentile(
                simulated,
                25,
            )
        ),
        "p50": float(
            np.percentile(
                simulated,
                50,
            )
        ),
        "p75": float(
            np.percentile(
                simulated,
                75,
            )
        ),
        "p90": float(
            np.percentile(
                simulated,
                90,
            )
        ),
        "p975": float(
            np.percentile(
                simulated,
                97.5,
            )
        ),
    }

    realized_probability, realized_log_loss = (
        smoothed_realized_seat_log_loss(
            simulated.astype(int),
            actual_seats,
        )
    )

    return {
        "actual_dem_seats": int(actual_seats),
        "expected_dem_seats": expected_seats,
        "median_dem_seats": median_seats,
        "simulation_dem_seat_sd": seat_sd,
        "expected_seat_error": (
            expected_seats - actual_seats
        ),
        "absolute_expected_seat_error": abs(
            expected_seats - actual_seats
        ),
        "median_seat_error": (
            median_seats - actual_seats
        ),
        "absolute_median_seat_error": abs(
            median_seats - actual_seats
        ),
        "crps": empirical_crps(
            simulated,
            actual_seats,
        ),
        "realized_seat_probability": (
            realized_probability
        ),
        "realized_seat_log_loss": (
            realized_log_loss
        ),
        "dem_control_probability": float(
            np.mean(
                simulated
                >= HOUSE_CONTROL_THRESHOLD
            )
        ),
        **percentiles,
        "interval_50_width": (
            percentiles["p75"]
            - percentiles["p25"]
        ),
        "interval_80_width": (
            percentiles["p90"]
            - percentiles["p10"]
        ),
        "interval_95_width": (
            percentiles["p975"]
            - percentiles["p025"]
        ),
        "actual_in_50_interval": float(
            percentiles["p25"]
            <= actual_seats
            <= percentiles["p75"]
        ),
        "actual_in_80_interval": float(
            percentiles["p10"]
            <= actual_seats
            <= percentiles["p90"]
        ),
        "actual_in_95_interval": float(
            percentiles["p025"]
            <= actual_seats
            <= percentiles["p975"]
        ),
    }


def summarize_allocations(
    cycle_results: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    group_columns = [
        "allocation_id",
        "national_error_sd",
        "district_error_sd",
        "marginal_total_error_sd",
        "is_current_production",
        "is_empirical_residual_allocation",
    ]

    for keys, frame in cycle_results.groupby(
        group_columns,
        sort=True,
        dropna=False,
    ):
        (
            allocation_id,
            national_sd,
            district_sd,
            marginal_sd,
            is_current,
            is_empirical,
        ) = keys

        rows.append(
            {
                "allocation_id": allocation_id,
                "national_error_sd": float(
                    national_sd
                ),
                "district_error_sd": float(
                    district_sd
                ),
                "marginal_total_error_sd": float(
                    marginal_sd
                ),
                "is_current_production": bool(
                    is_current
                ),
                "is_empirical_residual_allocation": bool(
                    is_empirical
                ),
                "cycles": int(
                    frame["cycle"].nunique()
                ),
                "mean_crps": float(
                    frame["crps"].mean()
                ),
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
                            frame[
                                "expected_seat_error"
                            ] ** 2
                        )
                    )
                ),
                "mean_simulation_dem_seat_sd": float(
                    frame[
                        "simulation_dem_seat_sd"
                    ].mean()
                ),
                "mean_interval_50_width": float(
                    frame[
                        "interval_50_width"
                    ].mean()
                ),
                "mean_interval_80_width": float(
                    frame[
                        "interval_80_width"
                    ].mean()
                ),
                "mean_interval_95_width": float(
                    frame[
                        "interval_95_width"
                    ].mean()
                ),
                "coverage_50": float(
                    frame[
                        "actual_in_50_interval"
                    ].mean()
                ),
                "coverage_80": float(
                    frame[
                        "actual_in_80_interval"
                    ].mean()
                ),
                "coverage_95": float(
                    frame[
                        "actual_in_95_interval"
                    ].mean()
                ),
                "mean_dem_control_probability": float(
                    frame[
                        "dem_control_probability"
                    ].mean()
                ),
            }
        )

    summary = pd.DataFrame(rows)

    summary["crps_rank"] = (
        summary["mean_crps"]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )

    summary["seat_log_loss_rank"] = (
        summary["mean_realized_seat_log_loss"]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )

    summary["median_error_rank"] = (
        summary[
            "mean_absolute_median_seat_error"
        ]
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
                "national_error_sd",
            ],
            kind="mergesort",
        )
        .index[0]
    )

    summary.loc[
        best_index,
        "recommended",
    ] = True

    return summary.sort_values(
        [
            "composite_rank_score",
            "mean_crps",
            "national_error_sd",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def validate_outputs(
    cycle_results: pd.DataFrame,
    summary: pd.DataFrame,
    expected_allocations: int,
) -> list[str]:
    checks: list[str] = []

    expected_cycle_rows = (
        expected_allocations
        * len(replay.SUPPORTED_CYCLES)
    )

    if len(cycle_results) != expected_cycle_rows:
        raise AllocationSweepError(
            f"Expected {expected_cycle_rows} cycle rows; "
            f"found {len(cycle_results)}."
        )

    checks.append(
        f"PASS: cycle rows = {expected_cycle_rows}"
    )

    if len(summary) != expected_allocations:
        raise AllocationSweepError(
            f"Expected {expected_allocations} allocation rows; "
            f"found {len(summary)}."
        )

    checks.append(
        f"PASS: allocation rows = {expected_allocations}"
    )

    if cycle_results.duplicated(
        [
            "allocation_id",
            "cycle",
        ]
    ).any():
        raise AllocationSweepError(
            "Duplicate allocation/cycle keys found."
        )

    checks.append(
        "PASS: allocation/cycle keys are unique"
    )

    total_sds = pd.to_numeric(
        cycle_results[
            "marginal_total_error_sd"
        ],
        errors="raise",
    )

    if (
        total_sds.max()
        - total_sds.min()
        > 1e-9
    ):
        raise AllocationSweepError(
            "Marginal total SD changed across allocations."
        )

    checks.append(
        "PASS: marginal total SD is fixed"
    )

    if int(summary["recommended"].sum()) != 1:
        raise AllocationSweepError(
            "Exactly one allocation must be recommended."
        )

    checks.append(
        "PASS: exactly one allocation is recommended"
    )

    current_rows = summary.loc[
        summary["is_current_production"]
    ]

    if len(current_rows) != 1:
        raise AllocationSweepError(
            "Current production allocation is missing or duplicated."
        )

    checks.append(
        "PASS: current production allocation is represented"
    )

    empirical_rows = summary.loc[
        summary[
            "is_empirical_residual_allocation"
        ]
    ]

    if len(empirical_rows) != 1:
        raise AllocationSweepError(
            "Empirical residual allocation is missing or duplicated."
        )

    checks.append(
        "PASS: empirical residual allocation is represented"
    )

    return checks


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep House national-versus-district "
            "Election Day uncertainty allocation."
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
        "--total-sd",
        type=float,
        default=DEFAULT_TOTAL_SD,
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

    if args.total_sd <= 0.0:
        raise ValueError("--total-sd must be positive.")

    if not args.master_path.exists():
        raise FileNotFoundError(
            f"Historical master not found: "
            f"{args.master_path}"
        )

    if args.clean_output and args.output_dir.exists():
        import shutil

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

    allocation_rows: list[dict[str, float]] = []

    seen: set[float] = set()

    for national_sd in DEFAULT_NATIONAL_GRID:
        national_sd = float(national_sd)

        if national_sd in seen:
            continue

        seen.add(national_sd)

        if national_sd > args.total_sd:
            raise AllocationSweepError(
                "National SD cannot exceed total SD: "
                f"{national_sd:.6f} > {args.total_sd:.6f}"
            )

        district_sd = math.sqrt(
            max(
                0.0,
                args.total_sd ** 2
                - national_sd ** 2,
            )
        )

        allocation_rows.append(
            {
                "national_error_sd": national_sd,
                "district_error_sd": district_sd,
            }
        )

    cycle_rows: list[dict[str, Any]] = []

    for allocation_number, allocation in enumerate(
        allocation_rows,
        start=1,
    ):
        national_sd = float(
            allocation["national_error_sd"]
        )

        district_sd = float(
            allocation["district_error_sd"]
        )

        allocation_id = (
            f"national_{national_sd:.4f}"
            f"_district_{district_sd:.4f}"
        )

        print()
        print("=" * 72)
        print(
            f"Allocation {allocation_number}/"
            f"{len(allocation_rows)}"
        )
        print(
            f"National SD: {national_sd:.4f}"
        )
        print(
            f"District SD: {district_sd:.4f}"
        )
        print(
            f"Total SD: {args.total_sd:.4f}"
        )
        print("=" * 72)

        for cycle_index, cycle in enumerate(
            replay.SUPPORTED_CYCLES
        ):
            df, model_margin, forecast_source = (
                replay.prepare_cycle(
                    master=master,
                    cycle=cycle,
                    candidate_quality_weight=(
                        args.candidate_quality_weight
                    ),
                    candidate_war_path=(
                        args.candidate_war_path
                    ),
                )
            )

            simulation_seed = (
                int(args.seed)
                + allocation_number * 100_000
                + cycle_index * 1_000
                + int(cycle)
            )

            simulated_seats = (
                simulate_chamber_distribution(
                    df=df,
                    model_margin=model_margin,
                    national_sd=national_sd,
                    district_sd=district_sd,
                    n_sims=args.sims,
                    seed=simulation_seed,
                )
            )

            actual_seats = actual_dem_seats(df)

            metrics = score_distribution(
                simulated_seats=simulated_seats,
                actual_seats=actual_seats,
            )

            row = {
                "allocation_id": allocation_id,
                "cycle": int(cycle),
                "forecast_source": forecast_source,
                "national_error_sd": national_sd,
                "district_error_sd": district_sd,
                "marginal_total_error_sd": float(
                    math.sqrt(
                        national_sd ** 2
                        + district_sd ** 2
                    )
                ),
                "is_current_production": bool(
                    math.isclose(
                        national_sd,
                        CURRENT_NATIONAL_SD,
                        abs_tol=1e-9,
                    )
                ),
                "is_empirical_residual_allocation": bool(
                    math.isclose(
                        national_sd,
                        1.9,
                        abs_tol=1e-9,
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

    summary = summarize_allocations(
        cycle_results
    )

    validation_checks = validate_outputs(
        cycle_results=cycle_results,
        summary=summary,
        expected_allocations=len(allocation_rows),
    )

    ranking_columns = [
        "recommended",
        "national_error_sd",
        "district_error_sd",
        "mean_crps",
        "mean_realized_seat_log_loss",
        "mean_absolute_median_seat_error",
        "mean_simulation_dem_seat_sd",
        "coverage_50",
        "coverage_80",
        "coverage_95",
        "composite_rank_score",
        "is_current_production",
        "is_empirical_residual_allocation",
    ]

    ranking = summary[
        ranking_columns
    ].copy()

    print()
    print("=" * 72)
    print("House uncertainty allocation ranking")
    print("=" * 72)
    print(
        ranking.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    recommended = summary.loc[
        summary["recommended"]
    ].iloc[0]

    print()
    print("Recommended allocation")
    print("-" * 72)
    print(
        f"National SD: "
        f"{recommended['national_error_sd']:.4f}"
    )
    print(
        f"District SD: "
        f"{recommended['district_error_sd']:.4f}"
    )
    print(
        f"Total SD: "
        f"{recommended['marginal_total_error_sd']:.4f}"
    )
    print(
        f"Mean CRPS: "
        f"{recommended['mean_crps']:.4f}"
    )
    print(
        f"Mean realized-seat log loss: "
        f"{recommended['mean_realized_seat_log_loss']:.4f}"
    )
    print(
        f"Mean simulated seat SD: "
        f"{recommended['mean_simulation_dem_seat_sd']:.4f}"
    )

    cycle_path = (
        args.output_dir
        / "house_uncertainty_allocation_by_cycle.csv"
    )

    summary_path = (
        args.output_dir
        / "house_uncertainty_allocation_summary.csv"
    )

    ranking_path = (
        args.output_dir
        / "house_uncertainty_allocation_ranking.csv"
    )

    config_path = (
        args.output_dir
        / "house_uncertainty_allocation_config.json"
    )

    validation_path = (
        args.output_dir
        / "house_uncertainty_allocation_validation.txt"
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

    config = {
        "supported_cycles": list(
            replay.SUPPORTED_CYCLES
        ),
        "total_sd": float(args.total_sd),
        "national_sd_grid": [
            float(row["national_error_sd"])
            for row in allocation_rows
        ],
        "district_sd_grid": [
            float(row["district_error_sd"])
            for row in allocation_rows
        ],
        "n_sims": int(args.sims),
        "seed": int(args.seed),
        "candidate_quality_weight": float(
            args.candidate_quality_weight
        ),
        "selection_method": (
            "minimum sum of CRPS rank, realized-seat "
            "log-loss rank, and absolute median-seat-error rank"
        ),
        "current_production_national_sd": (
            CURRENT_NATIONAL_SD
        ),
        "current_production_district_sd": (
            CURRENT_DISTRICT_SD
        ),
        "empirical_residual_national_sd": 1.9,
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
        "House Uncertainty Allocation Sweep Validation\n"
        + "=" * 48
        + "\n"
        + "\n".join(validation_checks)
        + "\n\nVALIDATION PASSED\n"
    )

    validation_path.write_text(
        validation_text
    )

    print()
    print(validation_text)

    print("Wrote:")
    for path in [
        cycle_path,
        summary_path,
        ranking_path,
        config_path,
        validation_path,
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
