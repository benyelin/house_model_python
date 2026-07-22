#!/usr/bin/env python3
"""
Calibrate the House Election Day national-versus-district variance split.

This sweep holds each district's validated marginal Election Day uncertainty
constant while changing only the share of variance assigned to a common
national error.

The experiment therefore isolates chamber-level correlation without changing
the underlying margin forecasts or marginal race-level probabilities.
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


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from historical.house.backtests.run_house_production_replay import (  # noqa: E402
    DEFAULT_MASTER_PATH,
    DEFAULT_WAR_PATH,
    HOUSE_CONTROL_THRESHOLD,
    SUPPORTED_CYCLES,
    prepare_cycle,
    validate_input,
)
from run_house_dynamic_uncertainty import (  # noqa: E402
    get_dynamic_sds,
    read_settings,
)


DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "distribution_calibration"
)

DEFAULT_VARIANCE_SHARES = (
    0.00,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
)

EPSILON = 1e-15


class DistributionCalibrationError(RuntimeError):
    """Raised when the calibration experiment fails validation."""


def parse_variance_shares(raw: str) -> list[float]:
    shares: list[float] = []

    for token in raw.split(","):
        token = token.strip()

        if not token:
            continue

        value = float(token)

        if not 0.0 <= value < 1.0:
            raise ValueError(
                "Every national variance share must be in [0, 1). "
                f"Received: {value}"
            )

        shares.append(value)

    if not shares:
        raise ValueError("At least one variance share is required.")

    return sorted(set(shares))


def resolve_fixed_control(df: pd.DataFrame) -> np.ndarray:
    """
    Return fixed-control party labels for all districts.

    The production replay's prepare_cycle function should already provide
    party_control_fixed. This fallback reconstructs the field from the
    uncontested indicators if necessary.
    """

    if "party_control_fixed" in df.columns:
        fixed = (
            df["party_control_fixed"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )
    else:
        fixed = pd.Series("", index=df.index, dtype="object")

    if "uncontested_dem" in df.columns:
        dem_uncontested = (
            df["uncontested_dem"]
            .fillna(False)
            .astype(bool)
        )
        fixed.loc[dem_uncontested] = "D"

    if "uncontested_gop" in df.columns:
        gop_uncontested = (
            df["uncontested_gop"]
            .fillna(False)
            .astype(bool)
        )
        fixed.loc[gop_uncontested] = "R"

    invalid = sorted(
        set(fixed.unique())
        - {"", "D", "R"}
    )

    if invalid:
        raise DistributionCalibrationError(
            f"Unexpected fixed-control values: {invalid}"
        )

    return fixed.to_numpy()


def actual_dem_seat_count(df: pd.DataFrame) -> int:
    actual_winner = (
        df["actual_winner"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    count = int(actual_winner.eq("D").sum())

    if not 0 <= count <= len(df):
        raise DistributionCalibrationError(
            f"Invalid actual Democratic seat count: {count}"
        )

    return count


def empirical_crps(
    samples: np.ndarray,
    observed: float,
) -> float:
    """
    Compute CRPS for an empirical forecast distribution.

    CRPS =
        mean(|X - y|)
        - 0.5 * mean(|X - X'|)

    The pairwise term is evaluated from sorted samples in O(n log n),
    avoiding an infeasible n-by-n distance matrix.
    """

    values = np.asarray(
        samples,
        dtype=float,
    )

    if values.ndim != 1 or len(values) == 0:
        raise DistributionCalibrationError(
            "CRPS samples must be a nonempty one-dimensional array."
        )

    if not np.isfinite(values).all():
        raise DistributionCalibrationError(
            "CRPS samples contain non-finite values."
        )

    ordered = np.sort(values)
    n = len(ordered)

    mean_absolute_observation_error = float(
        np.mean(
            np.abs(
                ordered - float(observed)
            )
        )
    )

    coefficients = (
        2.0 * np.arange(1, n + 1)
        - n
        - 1.0
    )

    half_expected_pairwise_difference = float(
        np.sum(
            coefficients * ordered
        )
        / (n * n)
    )

    return float(
        mean_absolute_observation_error
        - half_expected_pairwise_difference
    )


def simulate_split(
    *,
    df: pd.DataFrame,
    model_margin: pd.Series,
    total_sd: float,
    national_variance_share: float,
    national_standard_draws: np.ndarray,
    district_standard_draws: np.ndarray,
    actual_dem_seats: int,
) -> dict[str, Any]:
    n_sims, n_districts = district_standard_draws.shape

    if len(df) != n_districts:
        raise DistributionCalibrationError(
            "District draw width does not match cycle rows."
        )

    national_sd = total_sd * math.sqrt(
        national_variance_share
    )
    district_sd = total_sd * math.sqrt(
        1.0 - national_variance_share
    )

    base_margin = pd.to_numeric(
        model_margin,
        errors="raise",
    ).to_numpy(dtype=float)

    if not np.isfinite(base_margin).all():
        raise DistributionCalibrationError(
            "Model margins contain non-finite values."
        )

    simulated_margins = (
        base_margin.reshape(1, n_districts)
        + national_sd
        * national_standard_draws.reshape(n_sims, 1)
        + district_sd
        * district_standard_draws
    )

    dem_wins = simulated_margins > 0.0
    fixed = resolve_fixed_control(df)

    dem_wins[:, fixed == "D"] = True
    dem_wins[:, fixed == "R"] = False

    dem_seats = dem_wins.sum(axis=1).astype(int)

    expected_dem_seats = float(np.mean(dem_seats))
    median_dem_seats = float(np.median(dem_seats))
    seat_sd = float(np.std(dem_seats, ddof=1))

    control_probability = float(
        np.mean(
            dem_seats >= HOUSE_CONTROL_THRESHOLD
        )
    )

    seat_crps = empirical_crps(
        samples=dem_seats,
        observed=float(actual_dem_seats),
    )

    seats_below_actual = float(
        np.mean(
            dem_seats < actual_dem_seats
        )
    )
    seats_equal_actual = float(
        np.mean(
            dem_seats == actual_dem_seats
        )
    )

    # Midrank empirical percentile. This handles the discreteness of
    # integer seat totals more fairly than assigning all ties below or above.
    actual_seat_percentile = float(
        seats_below_actual
        + 0.5 * seats_equal_actual
    )

    return {
        "national_variance_share": float(
            national_variance_share
        ),
        "district_variance_share": float(
            1.0 - national_variance_share
        ),
        "implied_pairwise_correlation": float(
            national_variance_share
        ),
        "marginal_total_error_sd": float(total_sd),
        "national_error_sd": float(national_sd),
        "district_error_sd": float(district_sd),
        "expected_dem_seats": expected_dem_seats,
        "median_dem_seats": median_dem_seats,
        "simulation_dem_seat_sd": seat_sd,
        "dem_seats_p05": float(
            np.percentile(dem_seats, 5)
        ),
        "dem_seats_p10": float(
            np.percentile(dem_seats, 10)
        ),
        "dem_seats_p25": float(
            np.percentile(dem_seats, 25)
        ),
        "dem_seats_p50": float(
            np.percentile(dem_seats, 50)
        ),
        "dem_seats_p75": float(
            np.percentile(dem_seats, 75)
        ),
        "dem_seats_p90": float(
            np.percentile(dem_seats, 90)
        ),
        "dem_seats_p95": float(
            np.percentile(dem_seats, 95)
        ),
        "dem_control_probability": control_probability,
        "seat_distribution_crps": float(
            seat_crps
        ),
        "actual_seat_percentile": float(
            actual_seat_percentile
        ),
        "actual_percentile_distance_from_median": float(
            abs(
                actual_seat_percentile - 0.5
            )
        ),
        "actual_below_p05": int(
            actual_dem_seats
            < np.percentile(
                dem_seats,
                5,
            )
        ),
        "actual_above_p95": int(
            actual_dem_seats
            > np.percentile(
                dem_seats,
                95,
            )
        ),
        "actual_outside_90_interval": int(
            (
                actual_dem_seats
                < np.percentile(
                    dem_seats,
                    5,
                )
            )
            or (
                actual_dem_seats
                > np.percentile(
                    dem_seats,
                    95,
                )
            )
        ),
    }


def add_observed_scores(
    row: dict[str, Any],
    *,
    actual_dem_seats: int,
) -> dict[str, Any]:
    out = dict(row)

    actual_dem_control = int(
        actual_dem_seats >= HOUSE_CONTROL_THRESHOLD
    )
    probability = float(
        out["dem_control_probability"]
    )
    clipped_probability = float(
        np.clip(
            probability,
            EPSILON,
            1.0 - EPSILON,
        )
    )

    chamber_brier = float(
        (
            probability
            - actual_dem_control
        )
        ** 2
    )

    chamber_log_loss = -float(
        actual_dem_control
        * math.log(clipped_probability)
        + (1 - actual_dem_control)
        * math.log(1.0 - clipped_probability)
    )

    expected_seat_error = float(
        out["expected_dem_seats"]
        - actual_dem_seats
    )

    assigned_probability_to_actual_control = (
        probability
        if actual_dem_control
        else 1.0 - probability
    )

    predicted_dem_control = int(
        probability >= 0.5
    )

    out.update(
        {
            "actual_dem_seats": int(
                actual_dem_seats
            ),
            "actual_dem_control": actual_dem_control,
            "predicted_dem_control": (
                predicted_dem_control
            ),
            "correct_control_call": int(
                predicted_dem_control
                == actual_dem_control
            ),
            "probability_assigned_to_actual_control": float(
                assigned_probability_to_actual_control
            ),
            "chamber_brier_score": chamber_brier,
            "chamber_log_loss": chamber_log_loss,
            "expected_seat_error": expected_seat_error,
            "absolute_expected_seat_error": abs(
                expected_seat_error
            ),
            "squared_expected_seat_error": (
                expected_seat_error**2
            ),
            "actual_in_50_interval": int(
                out["dem_seats_p25"]
                <= actual_dem_seats
                <= out["dem_seats_p75"]
            ),
            "actual_in_80_interval": int(
                out["dem_seats_p10"]
                <= actual_dem_seats
                <= out["dem_seats_p90"]
            ),
            "actual_in_90_interval": int(
                out["dem_seats_p05"]
                <= actual_dem_seats
                <= out["dem_seats_p95"]
            ),
        }
    )

    return out


def summarize_shares(
    cycle_results: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for share, frame in cycle_results.groupby(
        "national_variance_share",
        sort=True,
    ):
        rows.append(
            {
                "national_variance_share": float(
                    share
                ),
                "district_variance_share": float(
                    1.0 - share
                ),
                "implied_pairwise_correlation": float(
                    share
                ),
                "national_error_sd": float(
                    frame["national_error_sd"].iloc[0]
                ),
                "district_error_sd": float(
                    frame["district_error_sd"].iloc[0]
                ),
                "cycles": int(len(frame)),
                "mean_chamber_brier_score": float(
                    frame[
                        "chamber_brier_score"
                    ].mean()
                ),
                "mean_chamber_log_loss": float(
                    frame[
                        "chamber_log_loss"
                    ].mean()
                ),
                "control_call_accuracy": float(
                    frame[
                        "correct_control_call"
                    ].mean()
                ),
                "mean_probability_assigned_to_actual_control": float(
                    frame[
                        "probability_assigned_to_actual_control"
                    ].mean()
                ),
                "mean_absolute_expected_seat_error": float(
                    frame[
                        "absolute_expected_seat_error"
                    ].mean()
                ),
                "rmse_expected_seats": float(
                    np.sqrt(
                        frame[
                            "squared_expected_seat_error"
                        ].mean()
                    )
                ),
                "mean_expected_seat_error_dem_bias": float(
                    frame[
                        "expected_seat_error"
                    ].mean()
                ),
                "mean_simulation_dem_seat_sd": float(
                    frame[
                        "simulation_dem_seat_sd"
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
                "coverage_90": float(
                    frame[
                        "actual_in_90_interval"
                    ].mean()
                ),
                "mean_seat_distribution_crps": float(
                    frame[
                        "seat_distribution_crps"
                    ].mean()
                ),
                "mean_actual_seat_percentile": float(
                    frame[
                        "actual_seat_percentile"
                    ].mean()
                ),
                "mean_actual_percentile_distance_from_median": float(
                    frame[
                        "actual_percentile_distance_from_median"
                    ].mean()
                ),
                "outcome_below_p05_rate": float(
                    frame[
                        "actual_below_p05"
                    ].mean()
                ),
                "outcome_above_p95_rate": float(
                    frame[
                        "actual_above_p95"
                    ].mean()
                ),
                "outcome_outside_90_interval_rate": float(
                    frame[
                        "actual_outside_90_interval"
                    ].mean()
                ),
            }
        )

    summary = pd.DataFrame(rows)

    summary["brier_rank"] = (
        summary["mean_chamber_brier_score"]
        .rank(method="min")
        .astype(int)
    )
    summary["log_loss_rank"] = (
        summary["mean_chamber_log_loss"]
        .rank(method="min")
        .astype(int)
    )
    summary["seat_rmse_rank"] = (
        summary["rmse_expected_seats"]
        .rank(method="min")
        .astype(int)
    )

    summary["combined_rank_score"] = (
        summary["brier_rank"]
        + summary["log_loss_rank"]
        + summary["seat_rmse_rank"]
    )

    summary["crps_rank"] = (
        summary[
            "mean_seat_distribution_crps"
        ]
        .rank(
            method="min"
        )
        .astype(int)
    )

    summary["distribution_recommended"] = False

    distribution_best_index = (
        summary.sort_values(
            [
                "mean_seat_distribution_crps",
                "mean_actual_percentile_distance_from_median",
                "national_variance_share",
            ]
        )
        .index[0]
    )

    summary.loc[
        distribution_best_index,
        "distribution_recommended",
    ] = True

    summary["recommended"] = False

    best_index = (
        summary.sort_values(
            [
                "combined_rank_score",
                "mean_chamber_log_loss",
                "mean_chamber_brier_score",
                "rmse_expected_seats",
                "national_variance_share",
            ]
        )
        .index[0]
    )

    summary.loc[best_index, "recommended"] = True

    return summary.sort_values(
        "national_variance_share"
    ).reset_index(drop=True)


def make_leave_one_cycle_out(
    cycle_results: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    cycles = sorted(
        cycle_results["cycle"]
        .astype(int)
        .unique()
        .tolist()
    )

    for held_out_cycle in cycles:
        training = cycle_results.loc[
            cycle_results["cycle"].ne(
                held_out_cycle
            )
        ].copy()

        training_summary = summarize_shares(
            training
        )

        selected = training_summary.loc[
            training_summary["recommended"]
        ].iloc[0]

        selected_share = float(
            selected["national_variance_share"]
        )

        held_out = cycle_results.loc[
            cycle_results["cycle"].eq(
                held_out_cycle
            )
            & cycle_results[
                "national_variance_share"
            ].eq(selected_share)
        ]

        if len(held_out) != 1:
            raise DistributionCalibrationError(
                "Expected one held-out result for "
                f"cycle={held_out_cycle}, "
                f"share={selected_share}; "
                f"found {len(held_out)}."
            )

        result = held_out.iloc[0]

        rows.append(
            {
                "held_out_cycle": int(
                    held_out_cycle
                ),
                "selected_national_variance_share": (
                    selected_share
                ),
                "selected_district_variance_share": float(
                    1.0 - selected_share
                ),
                "training_combined_rank_score": int(
                    selected[
                        "combined_rank_score"
                    ]
                ),
                "training_mean_chamber_brier_score": float(
                    selected[
                        "mean_chamber_brier_score"
                    ]
                ),
                "training_mean_chamber_log_loss": float(
                    selected[
                        "mean_chamber_log_loss"
                    ]
                ),
                "held_out_actual_dem_seats": int(
                    result["actual_dem_seats"]
                ),
                "held_out_expected_dem_seats": float(
                    result["expected_dem_seats"]
                ),
                "held_out_dem_control_probability": float(
                    result[
                        "dem_control_probability"
                    ]
                ),
                "held_out_actual_dem_control": int(
                    result["actual_dem_control"]
                ),
                "held_out_correct_control_call": int(
                    result["correct_control_call"]
                ),
                "held_out_chamber_brier_score": float(
                    result["chamber_brier_score"]
                ),
                "held_out_chamber_log_loss": float(
                    result["chamber_log_loss"]
                ),
                "held_out_absolute_expected_seat_error": float(
                    result[
                        "absolute_expected_seat_error"
                    ]
                ),
                "held_out_simulation_dem_seat_sd": float(
                    result[
                        "simulation_dem_seat_sd"
                    ]
                ),
                "held_out_actual_in_50_interval": int(
                    result[
                        "actual_in_50_interval"
                    ]
                ),
                "held_out_actual_in_80_interval": int(
                    result[
                        "actual_in_80_interval"
                    ]
                ),
                "held_out_actual_in_90_interval": int(
                    result[
                        "actual_in_90_interval"
                    ]
                ),
            }
        )

    return pd.DataFrame(rows)


def validate_results(
    *,
    cycle_results: pd.DataFrame,
    summary: pd.DataFrame,
    shares: list[float],
    total_sd: float,
) -> list[str]:
    checks: list[str] = []

    expected_rows = (
        len(SUPPORTED_CYCLES)
        * len(shares)
    )

    if len(cycle_results) != expected_rows:
        raise DistributionCalibrationError(
            f"Expected {expected_rows} cycle rows; "
            f"found {len(cycle_results)}."
        )

    checks.append(
        f"PASS: cycle-result rows = {len(cycle_results)}"
    )

    duplicate_count = int(
        cycle_results.duplicated(
            [
                "cycle",
                "national_variance_share",
            ]
        ).sum()
    )

    if duplicate_count:
        raise DistributionCalibrationError(
            "Duplicate cycle/share rows found."
        )

    checks.append(
        "PASS: cycle/share keys are unique"
    )

    reconstructed_variance = (
        cycle_results["national_error_sd"] ** 2
        + cycle_results["district_error_sd"] ** 2
    )

    if not np.allclose(
        reconstructed_variance,
        total_sd**2,
        rtol=0.0,
        atol=1e-10,
    ):
        raise DistributionCalibrationError(
            "Marginal variance changed across the sweep."
        )

    checks.append(
        "PASS: marginal variance is fixed across shares"
    )

    probabilities = cycle_results[
        "dem_control_probability"
    ]

    if (
        probabilities.isna().any()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise DistributionCalibrationError(
            "Invalid chamber probabilities found."
        )

    checks.append(
        "PASS: chamber probabilities are finite and within [0, 1]"
    )

    expected_shares = sorted(shares)
    observed_shares = sorted(
        summary[
            "national_variance_share"
        ].tolist()
    )

    if not np.allclose(
        expected_shares,
        observed_shares,
        rtol=0.0,
        atol=1e-12,
    ):
        raise DistributionCalibrationError(
            "Summary variance-share grid differs "
            "from requested grid."
        )

    checks.append(
        f"PASS: summary rows = {len(summary)}"
    )

    if int(summary["recommended"].sum()) != 1:
        raise DistributionCalibrationError(
            "Exactly one summary row must be recommended."
        )

    checks.append(
        "PASS: exactly one pooled recommendation"
    )

    if (
        cycle_results[
            "seat_distribution_crps"
        ].isna().any()
        or (
            cycle_results[
                "seat_distribution_crps"
            ]
            < 0
        ).any()
    ):
        raise DistributionCalibrationError(
            "Invalid CRPS values found."
        )

    checks.append(
        "PASS: full-distribution CRPS values are finite and nonnegative"
    )

    percentiles = cycle_results[
        "actual_seat_percentile"
    ]

    if (
        percentiles.isna().any()
        or (percentiles < 0).any()
        or (percentiles > 1).any()
    ):
        raise DistributionCalibrationError(
            "Invalid observed-seat percentiles found."
        )

    checks.append(
        "PASS: observed-seat percentiles are within [0, 1]"
    )

    if int(
        summary[
            "distribution_recommended"
        ].sum()
    ) != 1:
        raise DistributionCalibrationError(
            "Exactly one CRPS-based recommendation is required."
        )

    checks.append(
        "PASS: exactly one CRPS-based distribution recommendation"
    )

    return checks


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the House national-versus-district "
            "Election Day variance split."
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
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=100000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260720,
    )
    parser.add_argument(
        "--variance-shares",
        default=",".join(
            str(value)
            for value in DEFAULT_VARIANCE_SHARES
        ),
        help=(
            "Comma-separated national variance shares. "
            "Example: 0.1,0.2,0.3,0.4,0.5"
        ),
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

    shares = parse_variance_shares(
        args.variance_shares
    )

    if not args.master_path.exists():
        raise FileNotFoundError(
            f"Historical master not found: "
            f"{args.master_path}"
        )

    if (
        args.clean_output
        and args.output_dir.exists()
    ):
        shutil.rmtree(args.output_dir)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    master = pd.read_csv(
        args.master_path,
        low_memory=False,
    )
    validate_input(master)

    settings = read_settings()

    national_sd, _, _, district_sd = (
        get_dynamic_sds(
            settings,
            days_out=0,
        )
    )

    total_sd = float(
        math.sqrt(
            national_sd**2
            + district_sd**2
        )
    )

    print(
        "House Distribution Calibration"
    )
    print("=" * 72)
    print(
        f"Validated marginal Election Day SD: "
        f"{total_sd:.6f}"
    )
    print(
        "National variance shares: "
        + ", ".join(
            f"{share:.2f}"
            for share in shares
        )
    )
    print(
        f"Simulations per cycle/share: "
        f"{args.sims:,}"
    )

    cycle_rows: list[dict[str, Any]] = []
    forecast_sources: dict[str, str] = {}

    for cycle in SUPPORTED_CYCLES:
        print()
        print("-" * 72)
        print(f"Preparing cycle {cycle}")
        print("-" * 72)

        df, model_margin, forecast_source = (
            prepare_cycle(
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

        forecast_sources[str(cycle)] = (
            forecast_source
        )

        n_districts = len(df)

        if n_districts != 435:
            raise DistributionCalibrationError(
                f"Cycle {cycle} has "
                f"{n_districts} rows, expected 435."
            )

        actual_dem_seats = (
            actual_dem_seat_count(df)
        )

        rng = np.random.default_rng(
            args.seed + int(cycle)
        )

        # Common random numbers across all variance shares.
        national_standard_draws = rng.normal(
            0.0,
            1.0,
            size=args.sims,
        )
        district_standard_draws = rng.normal(
            0.0,
            1.0,
            size=(
                args.sims,
                n_districts,
            ),
        )

        print(
            f"Actual Democratic seats: "
            f"{actual_dem_seats}"
        )

        for share in shares:
            simulation = simulate_split(
                df=df,
                model_margin=model_margin,
                total_sd=total_sd,
                national_variance_share=share,
                national_standard_draws=(
                    national_standard_draws
                ),
                district_standard_draws=(
                    district_standard_draws
                ),
                actual_dem_seats=(
                    actual_dem_seats
                ),
            )

            scored = add_observed_scores(
                simulation,
                actual_dem_seats=(
                    actual_dem_seats
                ),
            )

            scored.update(
                {
                    "cycle": int(cycle),
                    "forecast_source": (
                        forecast_source
                    ),
                    "n_sims": int(args.sims),
                    "seed": int(
                        args.seed + int(cycle)
                    ),
                }
            )

            cycle_rows.append(scored)

            print(
                f"share={share:0.2f}  "
                f"national_sd="
                f"{scored['national_error_sd']:0.4f}  "
                f"district_sd="
                f"{scored['district_error_sd']:0.4f}  "
                f"expected="
                f"{scored['expected_dem_seats']:0.2f}  "
                f"control="
                f"{scored['dem_control_probability']:0.3f}  "
                f"seat_sd="
                f"{scored['simulation_dem_seat_sd']:0.2f}"
            )

    cycle_results = pd.DataFrame(
        cycle_rows
    ).sort_values(
        [
            "national_variance_share",
            "cycle",
        ]
    ).reset_index(drop=True)

    summary = summarize_shares(
        cycle_results
    )

    loo = make_leave_one_cycle_out(
        cycle_results
    )

    validation_checks = validate_results(
        cycle_results=cycle_results,
        summary=summary,
        shares=shares,
        total_sd=total_sd,
    )

    cycle_path = (
        args.output_dir
        / "house_distribution_calibration_by_cycle.csv"
    )
    summary_path = (
        args.output_dir
        / "house_distribution_calibration_summary.csv"
    )
    loo_path = (
        args.output_dir
        / "house_distribution_calibration_loo.csv"
    )
    config_path = (
        args.output_dir
        / "house_distribution_calibration_config.json"
    )
    validation_path = (
        args.output_dir
        / "house_distribution_calibration_validation.txt"
    )

    cycle_results.to_csv(
        cycle_path,
        index=False,
    )
    summary.to_csv(
        summary_path,
        index=False,
    )
    loo.to_csv(
        loo_path,
        index=False,
    )

    config = {
        "calibration_version": (
            "house_distribution_calibration_v2"
        ),
        "master_path": str(args.master_path),
        "candidate_war_path": str(
            args.candidate_war_path
        ),
        "candidate_quality_weight": float(
            args.candidate_quality_weight
        ),
        "cycles": list(SUPPORTED_CYCLES),
        "n_sims_per_cycle_share": int(
            args.sims
        ),
        "base_seed": int(args.seed),
        "variance_shares": shares,
        "validated_marginal_election_day_sd": (
            total_sd
        ),
        "components_included": [
            "national_error",
            "district_error",
        ],
        "components_deferred": [
            "region_error_groups",
            "demographic_error_groups",
            "historical_polling",
            "polling_variance_reduction",
        ],
        "common_random_numbers_across_shares": True,
        "forecast_sources": forecast_sources,
        "recommendation_rule": (
            "Lowest combined rank across pooled chamber "
            "Brier score, chamber log loss, and expected-seat "
            "RMSE; ties broken by log loss, Brier, seat RMSE, "
            "then lower national variance share."
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

    validation_text = "\n".join(
        [
            "House Distribution Calibration Validation",
            "=" * 40,
            *validation_checks,
            "",
            "VALIDATION PASSED",
            "",
        ]
    )

    validation_path.write_text(
        validation_text
    )

    display_columns = [
        "national_variance_share",
        "national_error_sd",
        "district_error_sd",
        "mean_chamber_brier_score",
        "mean_chamber_log_loss",
        "control_call_accuracy",
        "mean_probability_assigned_to_actual_control",
        "rmse_expected_seats",
        "mean_simulation_dem_seat_sd",
        "coverage_50",
        "coverage_80",
        "coverage_90",
        "mean_seat_distribution_crps",
        "mean_actual_seat_percentile",
        "mean_actual_percentile_distance_from_median",
        "outcome_outside_90_interval_rate",
        "crps_rank",
        "distribution_recommended",
        "combined_rank_score",
        "recommended",
    ]

    print()
    print("=" * 72)
    print("Pooled correlation calibration")
    print("=" * 72)
    print(
        summary[
            display_columns
        ].to_string(index=False)
    )

    print()
    print("=" * 72)
    print("Leave-one-cycle-out selections")
    print("=" * 72)
    print(loo.to_string(index=False))

    print()
    print(validation_text)

    print("Wrote:")
    for path in [
        cycle_path,
        summary_path,
        loo_path,
        config_path,
        validation_path,
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
