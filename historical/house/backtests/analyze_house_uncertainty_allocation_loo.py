#!/usr/bin/env python3
"""
Analyze leave-one-cycle-out stability for the House uncertainty sweep.

This script does not rerun simulations. It reads the saved cycle-level
allocation results and re-ranks allocations after excluding each
historical election cycle in turn.

It also produces a dedicated post-2016 ranking using only
2018, 2020, and 2022.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "uncertainty_allocation_calibration"
    / "house_uncertainty_allocation_by_cycle.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "uncertainty_allocation_calibration"
    / "loo_stability"
)


class LOOValidationError(RuntimeError):
    """Raised when leave-one-cycle-out outputs are invalid."""


def rank_subset(
    frame: pd.DataFrame,
    subset_name: str,
    excluded_cycle: int | None,
) -> pd.DataFrame:
    group_columns = [
        "allocation_id",
        "national_error_sd",
        "district_error_sd",
        "marginal_total_error_sd",
        "is_current_production",
        "is_empirical_residual_allocation",
    ]

    rows: list[dict[str, object]] = []

    for keys, allocation in frame.groupby(
        group_columns,
        sort=True,
        dropna=False,
    ):
        (
            allocation_id,
            national_sd,
            district_sd,
            total_sd,
            is_current,
            is_empirical,
        ) = keys

        rows.append(
            {
                "subset_name": subset_name,
                "excluded_cycle": excluded_cycle,
                "allocation_id": allocation_id,
                "national_error_sd": float(national_sd),
                "district_error_sd": float(district_sd),
                "marginal_total_error_sd": float(total_sd),
                "is_current_production": bool(is_current),
                "is_empirical_residual_allocation": bool(
                    is_empirical
                ),
                "cycles_used": int(
                    allocation["cycle"].nunique()
                ),
                "mean_crps": float(
                    allocation["crps"].mean()
                ),
                "mean_realized_seat_log_loss": float(
                    allocation[
                        "realized_seat_log_loss"
                    ].mean()
                ),
                "mean_absolute_expected_seat_error": float(
                    allocation[
                        "absolute_expected_seat_error"
                    ].mean()
                ),
                "mean_absolute_median_seat_error": float(
                    allocation[
                        "absolute_median_seat_error"
                    ].mean()
                ),
                "rmse_expected_seat_error": float(
                    np.sqrt(
                        np.mean(
                            allocation[
                                "expected_seat_error"
                            ] ** 2
                        )
                    )
                ),
                "mean_simulation_dem_seat_sd": float(
                    allocation[
                        "simulation_dem_seat_sd"
                    ].mean()
                ),
                "coverage_50": float(
                    allocation[
                        "actual_in_50_interval"
                    ].mean()
                ),
                "coverage_80": float(
                    allocation[
                        "actual_in_80_interval"
                    ].mean()
                ),
                "coverage_95": float(
                    allocation[
                        "actual_in_95_interval"
                    ].mean()
                ),
            }
        )

    ranking = pd.DataFrame(rows)

    ranking["crps_rank"] = (
        ranking["mean_crps"]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )

    ranking["seat_log_loss_rank"] = (
        ranking["mean_realized_seat_log_loss"]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )

    ranking["median_error_rank"] = (
        ranking[
            "mean_absolute_median_seat_error"
        ]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )

    ranking["composite_rank_score"] = (
        ranking["crps_rank"]
        + ranking["seat_log_loss_rank"]
        + ranking["median_error_rank"]
    )

    ranking = ranking.sort_values(
        [
            "composite_rank_score",
            "mean_crps",
            "mean_realized_seat_log_loss",
            "national_error_sd",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    ranking["subset_rank"] = (
        np.arange(len(ranking)) + 1
    )

    ranking["recommended"] = False
    ranking.loc[0, "recommended"] = True

    return ranking


def build_stability_summary(
    all_rankings: pd.DataFrame,
) -> pd.DataFrame:
    loo_only = all_rankings.loc[
        all_rankings["subset_name"].str.startswith(
            "exclude_"
        )
    ].copy()

    rows: list[dict[str, object]] = []

    group_columns = [
        "allocation_id",
        "national_error_sd",
        "district_error_sd",
        "is_current_production",
        "is_empirical_residual_allocation",
    ]

    for keys, allocation in loo_only.groupby(
        group_columns,
        sort=True,
        dropna=False,
    ):
        (
            allocation_id,
            national_sd,
            district_sd,
            is_current,
            is_empirical,
        ) = keys

        rows.append(
            {
                "allocation_id": allocation_id,
                "national_error_sd": float(national_sd),
                "district_error_sd": float(district_sd),
                "is_current_production": bool(is_current),
                "is_empirical_residual_allocation": bool(
                    is_empirical
                ),
                "loo_mean_rank": float(
                    allocation["subset_rank"].mean()
                ),
                "loo_median_rank": float(
                    allocation["subset_rank"].median()
                ),
                "loo_best_rank": int(
                    allocation["subset_rank"].min()
                ),
                "loo_worst_rank": int(
                    allocation["subset_rank"].max()
                ),
                "loo_times_recommended": int(
                    allocation["recommended"].sum()
                ),
                "loo_mean_crps": float(
                    allocation["mean_crps"].mean()
                ),
                "loo_mean_log_loss": float(
                    allocation[
                        "mean_realized_seat_log_loss"
                    ].mean()
                ),
            }
        )

    summary = pd.DataFrame(rows)

    return summary.sort_values(
        [
            "loo_mean_rank",
            "loo_worst_rank",
            "loo_mean_crps",
            "national_error_sd",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def validate(
    input_frame: pd.DataFrame,
    rankings: pd.DataFrame,
    stability: pd.DataFrame,
) -> list[str]:
    checks: list[str] = []

    cycles = sorted(
        input_frame["cycle"]
        .astype(int)
        .unique()
        .tolist()
    )

    if cycles != [2016, 2018, 2020, 2022]:
        raise LOOValidationError(
            f"Unexpected cycles: {cycles}"
        )

    checks.append(
        "PASS: expected cycles = 2016, 2018, 2020, 2022"
    )

    allocation_count = int(
        input_frame["allocation_id"].nunique()
    )

    expected_subsets = 6

    expected_rows = (
        allocation_count
        * expected_subsets
    )

    if len(rankings) != expected_rows:
        raise LOOValidationError(
            f"Expected {expected_rows} ranking rows; "
            f"found {len(rankings)}."
        )

    checks.append(
        f"PASS: ranking rows = {expected_rows}"
    )

    recommendations = (
        rankings.groupby("subset_name")[
            "recommended"
        ].sum()
    )

    if not recommendations.eq(1).all():
        raise LOOValidationError(
            "Each subset must have exactly one recommendation."
        )

    checks.append(
        "PASS: one recommendation per subset"
    )

    if len(stability) != allocation_count:
        raise LOOValidationError(
            "Stability summary does not contain one row "
            "per allocation."
        )

    checks.append(
        f"PASS: stability rows = {allocation_count}"
    )

    return checks


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze leave-one-cycle-out stability for "
            "House uncertainty allocation."
        )
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    if not args.input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input_path}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = pd.read_csv(
        args.input_path,
        low_memory=False,
    )

    required = {
        "allocation_id",
        "cycle",
        "national_error_sd",
        "district_error_sd",
        "marginal_total_error_sd",
        "crps",
        "realized_seat_log_loss",
        "absolute_expected_seat_error",
        "absolute_median_seat_error",
        "expected_seat_error",
        "simulation_dem_seat_sd",
        "actual_in_50_interval",
        "actual_in_80_interval",
        "actual_in_95_interval",
        "is_current_production",
        "is_empirical_residual_allocation",
    }

    missing = sorted(
        required - set(frame.columns)
    )

    if missing:
        raise LOOValidationError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    cycles = sorted(
        frame["cycle"]
        .astype(int)
        .unique()
        .tolist()
    )

    ranking_frames: list[pd.DataFrame] = []

    ranking_frames.append(
        rank_subset(
            frame=frame,
            subset_name="all_cycles",
            excluded_cycle=None,
        )
    )

    for excluded_cycle in cycles:
        subset = frame.loc[
            frame["cycle"].ne(excluded_cycle)
        ].copy()

        ranking_frames.append(
            rank_subset(
                frame=subset,
                subset_name=(
                    f"exclude_{excluded_cycle}"
                ),
                excluded_cycle=excluded_cycle,
            )
        )

    post_2016 = frame.loc[
        frame["cycle"].isin(
            [2018, 2020, 2022]
        )
    ].copy()

    ranking_frames.append(
        rank_subset(
            frame=post_2016,
            subset_name="post_2016_only",
            excluded_cycle=2016,
        )
    )

    rankings = pd.concat(
        ranking_frames,
        ignore_index=True,
    )

    stability = build_stability_summary(
        rankings
    )

    checks = validate(
        input_frame=frame,
        rankings=rankings,
        stability=stability,
    )

    recommendation_table = (
        rankings.loc[
            rankings["recommended"],
            [
                "subset_name",
                "excluded_cycle",
                "national_error_sd",
                "district_error_sd",
                "mean_crps",
                "mean_realized_seat_log_loss",
                "mean_absolute_median_seat_error",
                "mean_simulation_dem_seat_sd",
                "coverage_50",
                "coverage_80",
                "coverage_95",
            ],
        ]
        .sort_values(
            "subset_name",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    print()
    print("=" * 78)
    print("Recommended allocation by historical subset")
    print("=" * 78)
    print(
        recommendation_table.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print("=" * 78)
    print("Leave-one-cycle-out stability ranking")
    print("=" * 78)
    print(
        stability[
            [
                "national_error_sd",
                "district_error_sd",
                "loo_mean_rank",
                "loo_median_rank",
                "loo_best_rank",
                "loo_worst_rank",
                "loo_times_recommended",
                "loo_mean_crps",
                "loo_mean_log_loss",
                "is_current_production",
                "is_empirical_residual_allocation",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    rankings_path = (
        args.output_dir
        / "house_uncertainty_allocation_loo_rankings.csv"
    )

    recommendations_path = (
        args.output_dir
        / "house_uncertainty_allocation_loo_recommendations.csv"
    )

    stability_path = (
        args.output_dir
        / "house_uncertainty_allocation_loo_stability.csv"
    )

    validation_path = (
        args.output_dir
        / "house_uncertainty_allocation_loo_validation.txt"
    )

    rankings.to_csv(
        rankings_path,
        index=False,
    )

    recommendation_table.to_csv(
        recommendations_path,
        index=False,
    )

    stability.to_csv(
        stability_path,
        index=False,
    )

    validation_text = (
        "House Uncertainty Allocation LOO Validation\n"
        + "=" * 49
        + "\n"
        + "\n".join(checks)
        + "\n\nVALIDATION PASSED\n"
    )

    validation_path.write_text(
        validation_text
    )

    print()
    print(validation_text)

    print("Wrote:")
    for path in [
        rankings_path,
        recommendations_path,
        stability_path,
        validation_path,
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
