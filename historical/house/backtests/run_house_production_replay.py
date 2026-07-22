#!/usr/bin/env python3
"""
Run the first leakage-safe House production replay.

This replay compares two probability specifications using identical
historical model margins:

1. canonical_fixed_6_5
   The canonical historical scorer's fixed 6.5-point logistic scale.

2. production_election_day_v1
   Election Day production uncertainty using:
       - national correlated error
       - district-specific error
       - no historical polling
       - no synthetic region or demographic group assignments

The canonical historical runner and its regression benchmark are not
modified.

This first replay intentionally isolates the uncertainty conversion.
Candidate WAR can be enabled through the existing leakage-safe overlay.
"""

from __future__ import annotations

import os

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtr


PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from historical.house.backtests.run_house_historical_backtest import (  # noqa: E402
    build_candidate_quality_overlay,
    build_model_margin,
    score_backtest,
)
from run_house_dynamic_uncertainty import (  # noqa: E402
    get_dynamic_sds,
    normalize_fixed_control,
    polling_district_multiplier,
    read_settings,
)


SUPPORTED_CYCLES = (2016, 2018, 2020, 2022)

DEFAULT_MASTER_PATH = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "warehouse"
    / "house_historical_backtest_inputs_2016_2022.csv"
)

DEFAULT_WAR_PATH = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "warehouse"
    / "house_historical_candidate_war_2016_2022.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "production_replay_v1"
)

CANONICAL_SPEC = "canonical_fixed_6_5"
PRODUCTION_SPEC = "production_election_day_v1"

HOUSE_CONTROL_THRESHOLD = 218


class ReplayValidationError(RuntimeError):
    """Raised when production replay outputs violate their contract."""


def validate_input(master: pd.DataFrame) -> None:
    required = {
        "forecast_cycle",
        "cycle",
        "race_id",
        "actual_dem_margin",
        "actual_winner",
    }

    missing = sorted(required - set(master.columns))

    if missing:
        raise ReplayValidationError(
            "Historical master is missing required columns: "
            + ", ".join(missing)
        )

    forecast_cycle = pd.to_numeric(
        master["forecast_cycle"],
        errors="coerce",
    )

    available_cycles = sorted(
        forecast_cycle.dropna().astype(int).unique().tolist()
    )

    if available_cycles != list(SUPPORTED_CYCLES):
        raise ReplayValidationError(
            f"Expected cycles {list(SUPPORTED_CYCLES)}; "
            f"found {available_cycles}."
        )

    for cycle in SUPPORTED_CYCLES:
        cycle_rows = master.loc[forecast_cycle.eq(cycle)].copy()

        if len(cycle_rows) != 435:
            raise ReplayValidationError(
                f"Cycle {cycle} expected 435 rows; "
                f"found {len(cycle_rows)}."
            )

        if cycle_rows["race_id"].duplicated().any():
            duplicates = (
                cycle_rows.loc[
                    cycle_rows["race_id"].duplicated(False),
                    "race_id",
                ]
                .astype(str)
                .tolist()
            )

            raise ReplayValidationError(
                f"Cycle {cycle} contains duplicate race IDs: "
                + ", ".join(duplicates[:20])
            )


def prepare_cycle(
    master: pd.DataFrame,
    cycle: int,
    candidate_quality_weight: float,
    candidate_war_path: Path,
) -> tuple[pd.DataFrame, pd.Series, str]:
    forecast_cycle = pd.to_numeric(
        master["forecast_cycle"],
        errors="coerce",
    )

    df = master.loc[
        forecast_cycle.eq(cycle)
    ].copy()

    df = df.sort_values(
        ["state", "district", "race_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    df["candidate_quality_weight"] = float(
        candidate_quality_weight
    )

    # Candidate WAR is intentionally disabled in the initial
    # uncertainty-only replay. Do not require the external WAR
    # warehouse when its multiplier is exactly zero.
    if float(candidate_quality_weight) == 0.0:
        candidate_war = pd.Series(
            0.0,
            index=df.index,
            dtype=float,
        )
    else:
        if not candidate_war_path.exists():
            raise FileNotFoundError(
                "Historical candidate WAR table not found: "
                f"{candidate_war_path}"
            )

        candidate_war = build_candidate_quality_overlay(
            df,
            war_path=candidate_war_path,
        )

    df["candidate_war_adjustment_dem"] = (
        pd.to_numeric(
            candidate_war,
            errors="coerce",
        ).fillna(0.0)
    )

    df["candidate_quality_contribution_dem"] = (
        df["candidate_war_adjustment_dem"]
        * df["candidate_quality_weight"]
    )

    if "candidate_quality_adjustment_dem" in df.columns:
        existing_candidate_quality = pd.to_numeric(
            df["candidate_quality_adjustment_dem"],
            errors="coerce",
        ).fillna(0.0)
    else:
        existing_candidate_quality = pd.Series(
            0.0,
            index=df.index,
            dtype=float,
        )

    df["candidate_quality_adjustment_dem"] = (
        existing_candidate_quality
        + df["candidate_quality_contribution_dem"]
    )

    model_margin, forecast_source = build_model_margin(df)

    if model_margin is None:
        raise ReplayValidationError(
            f"Unable to construct model margins for {cycle}."
        )

    model_margin = pd.to_numeric(
        model_margin,
        errors="coerce",
    )

    if model_margin.isna().any():
        missing_count = int(model_margin.isna().sum())

        raise ReplayValidationError(
            f"Cycle {cycle} contains {missing_count} missing "
            "model margins."
        )

    return df, model_margin, str(forecast_source)


def run_canonical_spec(
    df: pd.DataFrame,
    model_margin: pd.Series,
    fixed_error_sd: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    canonical_df = df.copy()

    # Ensure the canonical scorer performs its own fixed-scale
    # probability conversion rather than inheriting any stale values.
    canonical_df = canonical_df.drop(
        columns=[
            "dem_win_probability",
            "total_error_sd",
        ],
        errors="ignore",
    )

    results, summary, calibration = score_backtest(
        df=canonical_df,
        model_margin=model_margin,
        default_error_sd=fixed_error_sd,
    )

    results.insert(0, "replay_spec", CANONICAL_SPEC)
    summary.insert(0, "replay_spec", CANONICAL_SPEC)
    calibration.insert(0, "replay_spec", CANONICAL_SPEC)

    summary["uncertainty_days_out"] = np.nan
    summary["national_error_sd"] = 0.0
    summary["region_error_sd"] = 0.0
    summary["demographic_error_sd"] = 0.0
    summary["district_error_sd"] = float(fixed_error_sd)
    summary["marginal_total_error_sd"] = float(fixed_error_sd)
    summary["n_sims"] = 0
    summary["median_dem_seats"] = np.nan
    summary["dem_seats_p25"] = np.nan
    summary["dem_seats_p50"] = np.nan
    summary["dem_seats_p75"] = np.nan
    summary["dem_control_probability"] = np.nan

    return results, summary, calibration


def simulate_production_v1(
    df: pd.DataFrame,
    model_margin: pd.Series,
    settings: dict[str, float],
    n_sims: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Run Election Day production uncertainty without fabricated
    historical region or demographic groups.

    Included:
      - national error
      - district-specific error

    Excluded in v1:
      - region error
      - demographic error
      - polling-based district variance reductions
    """
    replay_df = normalize_fixed_control(df.copy())

    replay_df["poll_count"] = 0

    national_sd, region_sd, demographic_sd, district_sd = (
        get_dynamic_sds(
            settings,
            days_out=0,
        )
    )

    # Optional research-only overrides allow uncertainty-allocation
    # experiments to use the exact production replay implementation.
    #
    # Normal production behavior is unchanged when these environment
    # variables are absent.
    national_sd_override_raw = os.environ.get(
        "HOUSE_REPLAY_NATIONAL_SD"
    )
    district_sd_override_raw = os.environ.get(
        "HOUSE_REPLAY_DISTRICT_SD"
    )

    if national_sd_override_raw is not None:
        national_sd = float(national_sd_override_raw)

    if district_sd_override_raw is not None:
        district_sd = float(district_sd_override_raw)

    if float(national_sd) < 0.0:
        raise ReplayValidationError(
            "HOUSE_REPLAY_NATIONAL_SD must be nonnegative."
        )

    if float(district_sd) <= 0.0:
        raise ReplayValidationError(
            "HOUSE_REPLAY_DISTRICT_SD must be positive."
        )

    # This first replay deliberately omits group effects because the
    # canonical historical warehouse does not yet contain defensible
    # historical region/demographic error-group assignments.
    region_sd_used = 0.0
    demographic_sd_used = 0.0

    poll_counts = pd.to_numeric(
        replay_df["poll_count"],
        errors="coerce",
    ).fillna(0)

    district_multipliers = poll_counts.apply(
        lambda value: polling_district_multiplier(
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
        district_sds.reshape(1, len(replay_df)),
        size=(n_sims, len(replay_df)),
    )

    margins = model_margin.to_numpy(dtype=float)

    simulated_margins = (
        margins.reshape(1, len(replay_df))
        + national_errors.reshape(n_sims, 1)
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

    dem_seats = dem_wins.sum(axis=1)

    effective_total_sd = np.sqrt(
        float(national_sd) ** 2
        + district_sds ** 2
    )

    # Use the exact marginal normal probability for race-level
    # scoring. Correlated simulation remains the source of the
    # chamber-level seat distribution and control probability.
    #
    # This avoids Monte Carlo probabilities of exactly 0 or 1 for
    # non-fixed races and makes Brier/log-loss results deterministic.
    dem_probabilities = ndtr(
        margins / effective_total_sd
    )

    dem_probabilities[fixed == "D"] = 1.0
    dem_probabilities[fixed == "R"] = 0.0

    replay_df["dem_win_probability"] = (
        dem_probabilities
    )

    replay_df["total_error_sd"] = effective_total_sd
    replay_df["uncertainty_days_out"] = 0
    replay_df["national_error_sd"] = float(national_sd)
    replay_df["region_error_sd"] = region_sd_used
    replay_df["demographic_error_sd"] = (
        demographic_sd_used
    )
    replay_df["district_error_sd"] = float(district_sd)
    replay_df["effective_district_error_sd"] = district_sds

    simulation_summary = {
        "uncertainty_days_out": 0,
        "national_error_sd": float(national_sd),
        "configured_region_error_sd": float(region_sd),
        "configured_demographic_error_sd": float(
            demographic_sd
        ),
        "region_error_sd": region_sd_used,
        "demographic_error_sd": demographic_sd_used,
        "district_error_sd": float(district_sd),
        "marginal_total_error_sd": float(
            np.sqrt(
                float(national_sd) ** 2
                + float(district_sd) ** 2
            )
        ),
        "n_sims": int(n_sims),
        "expected_dem_seats_from_simulation": float(
            dem_seats.mean()
        ),
        "simulation_dem_seat_sd": float(
            np.std(dem_seats, ddof=1)
        ),
        "simulation_expected_seat_standard_error": float(
            np.std(dem_seats, ddof=1)
            / np.sqrt(n_sims)
        ),
        "median_dem_seats": float(
            np.median(dem_seats)
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
        "dem_control_probability": float(
            np.mean(
                dem_seats >= HOUSE_CONTROL_THRESHOLD
            )
        ),
    }

    return replay_df, simulation_summary


def run_production_spec(
    df: pd.DataFrame,
    model_margin: pd.Series,
    settings: dict[str, float],
    n_sims: int,
    seed: int,
    fixed_error_sd: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    replay_df, simulation_summary = simulate_production_v1(
        df=df,
        model_margin=model_margin,
        settings=settings,
        n_sims=n_sims,
        seed=seed,
    )

    results, summary, calibration = score_backtest(
        df=replay_df,
        model_margin=model_margin,
        default_error_sd=fixed_error_sd,
    )

    results.insert(0, "replay_spec", PRODUCTION_SPEC)
    summary.insert(0, "replay_spec", PRODUCTION_SPEC)
    calibration.insert(0, "replay_spec", PRODUCTION_SPEC)

    for key, value in simulation_summary.items():
        summary[key] = value

    # The scorer's expected-seat value should equal the sum of the
    # race probabilities, which should closely match the direct
    # simulation mean.
    difference = abs(
        float(summary.loc[0, "expected_dem_seats"])
        - float(
            simulation_summary[
                "expected_dem_seats_from_simulation"
            ]
        )
    )

    simulation_standard_error = float(
        simulation_summary[
            "simulation_expected_seat_standard_error"
        ]
    )

    # The analytic expected-seat total and Monte Carlo simulation mean
    # need not be identical. Allow five Monte Carlo standard errors,
    # with a small absolute floor for exceptionally narrow simulated
    # seat distributions.
    simulation_tolerance = max(
        0.10,
        5.0 * simulation_standard_error,
    )

    summary["analytic_minus_simulated_expected_seats"] = (
        float(summary.loc[0, "expected_dem_seats"])
        - float(
            simulation_summary[
                "expected_dem_seats_from_simulation"
            ]
        )
    )
    summary["simulation_validation_tolerance"] = (
        simulation_tolerance
    )

    if difference > simulation_tolerance:
        raise ReplayValidationError(
            "Production analytic expected seats differ from the "
            "direct simulation mean by more than the Monte Carlo "
            "tolerance. "
            f"Difference: {difference:.6f}; "
            f"standard error: {simulation_standard_error:.6f}; "
            f"tolerance: {simulation_tolerance:.6f}"
        )

    return results, summary, calibration


def make_comparison(
    summaries: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "winner_accuracy",
        "mean_abs_margin_error",
        "median_abs_margin_error",
        "rmse_margin_error",
        "mean_margin_error_dem_bias",
        "brier_score",
        "log_loss",
        "expected_dem_seats",
        "expected_seat_error",
    ]

    canonical = (
        summaries.loc[
            summaries["replay_spec"].eq(
                CANONICAL_SPEC
            ),
            ["cycle", *metrics],
        ]
        .set_index("cycle")
        .sort_index()
    )

    production = (
        summaries.loc[
            summaries["replay_spec"].eq(
                PRODUCTION_SPEC
            ),
            ["cycle", *metrics],
        ]
        .set_index("cycle")
        .sort_index()
    )

    if not canonical.index.equals(production.index):
        raise ReplayValidationError(
            "Canonical and production cycle keys differ."
        )

    rows: list[dict[str, Any]] = []

    for cycle in canonical.index:
        row: dict[str, Any] = {
            "cycle": int(cycle),
        }

        for metric in metrics:
            canonical_value = float(
                canonical.loc[cycle, metric]
            )
            production_value = float(
                production.loc[cycle, metric]
            )

            row[f"canonical_{metric}"] = canonical_value
            row[f"production_{metric}"] = production_value
            row[f"production_minus_canonical_{metric}"] = (
                production_value - canonical_value
            )

        rows.append(row)

    return pd.DataFrame(rows)


def make_overall_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for spec, frame in results.groupby(
        "replay_spec",
        sort=True,
    ):
        scored = frame.loc[
            frame["include_in_scoring"].astype(bool)
        ].copy()

        probabilities = pd.to_numeric(
            scored["dem_win_probability"],
            errors="raise",
        ).clip(1e-15, 1.0 - 1e-15)

        actual = (
            scored["actual_dem_win"]
            .astype(bool)
            .astype(float)
        )

        log_loss = -float(
            np.mean(
                actual * np.log(probabilities)
                + (1.0 - actual)
                * np.log(1.0 - probabilities)
            )
        )

        rows.append(
            {
                "replay_spec": spec,
                "cycles": int(
                    frame["cycle"].nunique()
                ),
                "all_races": int(len(frame)),
                "scored_races": int(len(scored)),
                "winner_accuracy": float(
                    scored["correct_winner"].mean()
                ),
                "mean_abs_margin_error": float(
                    scored["abs_margin_error"].mean()
                ),
                "rmse_margin_error": float(
                    np.sqrt(
                        scored[
                            "squared_margin_error"
                        ].mean()
                    )
                ),
                "mean_margin_error_dem_bias": float(
                    scored["margin_error"].mean()
                ),
                "brier_score": float(
                    scored["brier_score"].mean()
                ),
                "log_loss": log_loss,
            }
        )

    return pd.DataFrame(rows)


def validate_outputs(
    results: pd.DataFrame,
    summaries: pd.DataFrame,
) -> list[str]:
    checks: list[str] = []

    expected_result_rows = (
        len(SUPPORTED_CYCLES)
        * 435
        * 2
    )

    if len(results) != expected_result_rows:
        raise ReplayValidationError(
            f"Expected {expected_result_rows} result rows; "
            f"found {len(results)}."
        )

    checks.append(
        f"PASS: result rows = {expected_result_rows:,}"
    )

    duplicate_keys = results.duplicated(
        ["replay_spec", "cycle", "race_id"]
    )

    if duplicate_keys.any():
        raise ReplayValidationError(
            "Duplicate replay-spec/cycle/race keys found."
        )

    checks.append(
        "PASS: replay-spec/cycle/race keys are unique"
    )

    probabilities = pd.to_numeric(
        results["dem_win_probability"],
        errors="coerce",
    )

    if probabilities.isna().any():
        raise ReplayValidationError(
            "Missing replay probabilities found."
        )

    if not probabilities.between(0.0, 1.0).all():
        raise ReplayValidationError(
            "Replay probabilities fall outside [0, 1]."
        )

    checks.append(
        "PASS: all probabilities are finite and within [0, 1]"
    )

    expected_summary_rows = (
        len(SUPPORTED_CYCLES)
        * 2
    )

    if len(summaries) != expected_summary_rows:
        raise ReplayValidationError(
            f"Expected {expected_summary_rows} summary rows; "
            f"found {len(summaries)}."
        )

    checks.append(
        f"PASS: summary rows = {expected_summary_rows}"
    )

    production = results.loc[
        results["replay_spec"].eq(PRODUCTION_SPEC)
    ]

    fixed_d = production[
        "party_control_fixed"
    ].fillna("").astype(str).str.upper().eq("D")

    fixed_r = production[
        "party_control_fixed"
    ].fillna("").astype(str).str.upper().eq("R")

    if not (
        production.loc[
            fixed_d,
            "dem_win_probability",
        ].eq(1.0).all()
    ):
        raise ReplayValidationError(
            "Fixed Democratic seats do not all have "
            "probability 1.0."
        )

    if not (
        production.loc[
            fixed_r,
            "dem_win_probability",
        ].eq(0.0).all()
    ):
        raise ReplayValidationError(
            "Fixed Republican seats do not all have "
            "probability 0.0."
        )

    checks.append(
        "PASS: fixed-control probabilities are exact"
    )

    checks.append(
        "PASS: canonical scorer and benchmark files were not modified"
    )

    return checks


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run leakage-safe House production replay v1."
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
            "Leakage-safe historical candidate WAR multiplier. "
            "Default: 0.0 so v1 isolates uncertainty."
        ),
    )

    parser.add_argument(
        "--fixed-error-sd",
        type=float,
        default=6.5,
    )

    parser.add_argument(
        "--sims",
        type=int,
        default=20000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260719,
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

    validate_input(master)

    settings = read_settings()

    result_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []

    forecast_sources: dict[str, str] = {}

    for cycle in SUPPORTED_CYCLES:
        print()
        print("=" * 72)
        print(f"House production replay: {cycle}")
        print("=" * 72)

        df, model_margin, forecast_source = prepare_cycle(
            master=master,
            cycle=cycle,
            candidate_quality_weight=(
                args.candidate_quality_weight
            ),
            candidate_war_path=args.candidate_war_path,
        )

        forecast_sources[str(cycle)] = forecast_source

        canonical_results, canonical_summary, canonical_calibration = (
            run_canonical_spec(
                df=df,
                model_margin=model_margin,
                fixed_error_sd=args.fixed_error_sd,
            )
        )

        production_results, production_summary, production_calibration = (
            run_production_spec(
                df=df,
                model_margin=model_margin,
                settings=settings,
                n_sims=args.sims,
                seed=args.seed + cycle,
                fixed_error_sd=args.fixed_error_sd,
            )
        )

        result_frames.extend(
            [
                canonical_results,
                production_results,
            ]
        )

        summary_frames.extend(
            [
                canonical_summary,
                production_summary,
            ]
        )

        calibration_frames.extend(
            [
                canonical_calibration,
                production_calibration,
            ]
        )

        print(
            pd.concat(
                [
                    canonical_summary,
                    production_summary,
                ],
                ignore_index=True,
            )[
                [
                    "replay_spec",
                    "cycle",
                    "brier_score",
                    "log_loss",
                    "expected_dem_seats",
                    "actual_dem_seats",
                    "expected_seat_error",
                ]
            ].to_string(index=False)
        )

    results = pd.concat(
        result_frames,
        ignore_index=True,
    )

    summaries = pd.concat(
        summary_frames,
        ignore_index=True,
    )

    calibration = pd.concat(
        calibration_frames,
        ignore_index=True,
    )

    comparison = make_comparison(summaries)
    overall = make_overall_summary(results)

    validation_checks = validate_outputs(
        results=results,
        summaries=summaries,
    )

    results_path = (
        args.output_dir
        / "house_production_replay_predictions.csv"
    )

    summaries_path = (
        args.output_dir
        / "house_production_replay_by_cycle.csv"
    )

    comparison_path = (
        args.output_dir
        / "house_production_replay_comparison.csv"
    )

    overall_path = (
        args.output_dir
        / "house_production_replay_summary.csv"
    )

    calibration_path = (
        args.output_dir
        / "house_production_replay_calibration.csv"
    )

    config_path = (
        args.output_dir
        / "house_production_replay_config.json"
    )

    validation_path = (
        args.output_dir
        / "house_production_replay_validation.txt"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    summaries.to_csv(
        summaries_path,
        index=False,
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    overall.to_csv(
        overall_path,
        index=False,
    )

    calibration.to_csv(
        calibration_path,
        index=False,
    )

    config = {
        "replay_version": "house_production_replay_v1",
        "master_path": str(args.master_path),
        "candidate_war_path": str(
            args.candidate_war_path
        ),
        "candidate_quality_weight": float(
            args.candidate_quality_weight
        ),
        "fixed_error_sd": float(
            args.fixed_error_sd
        ),
        "n_sims": int(args.sims),
        "base_seed": int(args.seed),
        "cycles": list(SUPPORTED_CYCLES),
        "specifications": [
            CANONICAL_SPEC,
            PRODUCTION_SPEC,
        ],
        "production_days_out": 0,
        "production_components_included": [
            "national_error",
            "district_error",
        ],
        "production_components_deferred": [
            "region_error_groups",
            "demographic_error_groups",
            "historical_polling",
            "polling_variance_reduction",
        ],
        "forecast_sources": forecast_sources,
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
            "House Production Replay v1 Validation",
            "=" * 44,
            *validation_checks,
            "",
            "VALIDATION PASSED",
            "",
        ]
    )

    validation_path.write_text(
        validation_text
    )

    print()
    print("=" * 72)
    print("Overall replay comparison")
    print("=" * 72)
    print(overall.to_string(index=False))

    print()
    print("=" * 72)
    print("Cycle-level production minus canonical")
    print("=" * 72)

    display_columns = [
        "cycle",
        "production_minus_canonical_brier_score",
        "production_minus_canonical_log_loss",
        "production_minus_canonical_expected_seat_error",
    ]

    print(
        comparison[display_columns]
        .to_string(index=False)
    )

    print()
    print(validation_text)

    print("Wrote:")
    for path in [
        results_path,
        summaries_path,
        comparison_path,
        overall_path,
        calibration_path,
        config_path,
        validation_path,
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
