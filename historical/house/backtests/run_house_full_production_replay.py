#!/usr/bin/env python3
"""
Run the leakage-safe full House production replay.

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

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


import os

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from historical.house.backtests import (
    run_house_baseline_environment_bakeoff
    as baseline_environment_bakeoff,
)
from historical.house.common.partisan_baseline import (
    add_normalized_partisan_baseline,
)
from recalculate_house_fundamentals import (
    recalculate_house_fundamentals_dataframe,
)
from scipy.special import ndtr

from house_simulation import (
    run_simulation as run_shared_house_simulation,
)
from run_house_model import HouseModelConfig


PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from historical.house.backtests.run_house_historical_backtest import (  # noqa: E402
    build_candidate_quality_overlay,
    build_model_margin,
    build_scoring_mask,
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
    / "full_production_replay"
)

LEGACY_SPEC = "legacy_fixed_6_5"
PRODUCTION_FUNDAMENTALS_SPEC = (
    "production_fundamentals_fixed_6_5"
)
PRODUCTION_SPEC = "production_election_day_v1"
PRODUCTION_SHARED_SPEC = (
    "production_shared_uncertainty_v2"
)

REPLAY_SPECS = (
    LEGACY_SPEC,
    PRODUCTION_FUNDAMENTALS_SPEC,
    PRODUCTION_SPEC,
    PRODUCTION_SHARED_SPEC,
)

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


def add_validated_historical_partisan_baseline(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct the historical normalized partisan baseline using the
    identical validated national presidential-margin lookup used by
    the established House baseline bakeoff.
    """
    validated_inputs = (
        baseline_environment_bakeoff.load_inputs()
    )

    lookup = validated_inputs[
        [
            "presidential_result_year",
            "national_pres_margin_dem",
        ]
    ].drop_duplicates()

    before = df.loc[
        df["state"].astype(str).str.upper().eq("FL")
        & pd.to_numeric(
            df["forecast_cycle"],
            errors="coerce",
        ).eq(2016)
    ].copy()

    print()
    print("FLORIDA BASELINE INPUT DIAGNOSTIC")
    print("-" * 72)

    diagnostic_columns = [
        column
        for column in [
            "forecast_cycle",
            "cycle",
            "race_id",
            "district_id",
            "state",
            "district",
            "district_pres_margin_dem",
            "presidential_result_year",
            "national_pres_margin_dem",
            "boundary_cycle",
            "boundary_compatibility",
            "baseline_selection_method",
            "district_partisan_baseline_dem",
        ]
        if column in before.columns
    ]

    print(before[diagnostic_columns].to_string(index=False))

    out = add_normalized_partisan_baseline(
        df,
        lookup,
    )

    after = out.loc[
        out["state"].astype(str).str.upper().eq("FL")
        & pd.to_numeric(
            out["forecast_cycle"],
            errors="coerce",
        ).eq(2016)
    ].copy()

    print()
    print("FLORIDA BASELINE OUTPUT DIAGNOSTIC")
    print("-" * 72)

    diagnostic_columns = [
        column
        for column in [
            "forecast_cycle",
            "cycle",
            "race_id",
            "district_id",
            "state",
            "district",
            "district_pres_margin_dem",
            "presidential_result_year",
            "national_pres_margin_dem",
            "boundary_cycle",
            "boundary_compatibility",
            "baseline_selection_method",
            "district_partisan_baseline_dem",
        ]
        if column in after.columns
    ]

    print(after[diagnostic_columns].to_string(index=False))

    return out


def adapt_replay_inputs_to_production(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Translate leakage-safe historical House inputs into the
    production fundamentals interface.

    This adapter performs no forecasting calculations. Historical
    district partisanship remains in the warehouse's validated
    district_partisan_baseline_dem column.

    The current production-specific presidential fields are left
    unavailable so historical replays cannot use future election
    results. The production fundamentals calculator will preserve
    the supplied historical district partisan baseline.
    """
    out = df.copy()

    required = [
        "state",
        "district",
        "district_id",
        "district_partisan_baseline_dem",
        "dem_is_incumbent",
        "gop_is_incumbent",
    ]

    missing = [
        column
        for column in required
        if column not in out.columns
    ]

    if missing:
        raise ReplayValidationError(
            "Historical replay input is missing adapter columns: "
            + ", ".join(missing)
        )

    # These fields describe the current production baseline blend.
    # They must remain unavailable during historical replays because
    # using them would leak future presidential results.
    out["pres_2024_margin_dem"] = np.nan
    out["pres_2020_margin_dem"] = np.nan

    # Translate canonical historical candidate-registry flags into
    # the names consumed by production fundamentals.
    out["dem_candidate_is_incumbent"] = (
        out["dem_is_incumbent"].fillna(False)
    )
    out["gop_candidate_is_incumbent"] = (
        out["gop_is_incumbent"].fillna(False)
    )

    out["replay_presidential_baseline_mode"] = (
        "historical_precomputed"
    )

    return out


def prepare_cycle(
    master: pd.DataFrame,
    cycle: int,
    candidate_quality_weight: float,
    candidate_war_path: Path,
) -> tuple[pd.DataFrame, float]:
    """
    Prepare leakage-safe historical inputs for one election cycle.

    This function performs input selection, normalization, schema
    adaptation, and historical candidate-quality preparation. It does
    not calculate a forecast margin.
    """
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

    df = add_validated_historical_partisan_baseline(
        df
    )

    df = adapt_replay_inputs_to_production(df)

    df["candidate_quality_weight"] = float(
        candidate_quality_weight
    )

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

    historical_environment = pd.to_numeric(
        df["national_environment_margin_dem"],
        errors="coerce",
    ).dropna()

    unique_environments = np.sort(
        historical_environment.unique()
    )

    if len(unique_environments) != 1:
        raise ReplayValidationError(
            f"Cycle {cycle} must contain exactly one historical "
            "national environment; found "
            f"{unique_environments.tolist()}."
        )

    national_environment = float(
        unique_environments[0]
    )

    return df, national_environment


def build_production_fundamentals(
    df: pd.DataFrame,
    cycle: int,
    national_environment: float,
) -> tuple[pd.DataFrame, pd.Series, str]:
    """
    Run the production fundamentals calculator on prepared historical
    inputs and return its district-level Democratic margins.
    """
    production_df = df.copy()

    env_metadata = {
        "national_environment_source_path": (
            f"historical production replay cycle {cycle}"
        ),
        "national_environment_margin_dem": float(
            national_environment
        ),
        "environment_formula_version": (
            "generic_ballot_only_0_90_v1"
        ),
        "forecast_cycle": int(cycle),
    }

    production_df = recalculate_house_fundamentals_dataframe(
        production_df,
        national_environment=float(national_environment),
        env_metadata=env_metadata,
    )

    if "model_margin_dem" not in production_df.columns:
        raise ReplayValidationError(
            "Production fundamentals did not return "
            "model_margin_dem."
        )

    model_margin = pd.to_numeric(
        production_df["model_margin_dem"],
        errors="coerce",
    )

    scoring_mask = build_scoring_mask(production_df)

    missing_scored_margin = (
        scoring_mask
        & model_margin.isna()
    )

    if missing_scored_margin.any():
        missing_count = int(
            missing_scored_margin.sum()
        )

        examples = (
            production_df.loc[
                missing_scored_margin,
                "race_id",
            ]
            .astype(str)
            .head(20)
            .tolist()
        )

        raise ReplayValidationError(
            f"Cycle {cycle} contains {missing_count} missing "
            "production model margins among scored races. "
            f"Examples: {examples}"
        )

    forecast_source = (
        "recalculate_house_fundamentals_dataframe"
    )

    return production_df, model_margin, forecast_source

def run_fixed_probability_spec(
    df: pd.DataFrame,
    model_margin: pd.Series,
    fixed_error_sd: float,
    replay_spec: str,
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

    results.insert(0, "replay_spec", replay_spec)
    summary.insert(0, "replay_spec", replay_spec)
    calibration.insert(0, "replay_spec", replay_spec)

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



def prepare_shared_simulation_table(
    df: pd.DataFrame,
    model_margin: pd.Series,
    fallback_margin: pd.Series,
) -> pd.DataFrame:
    """
    Prepare leakage-safe historical rows for the shared production
    House simulation engine.

    This adapter does not calculate fundamentals. It supplies the
    simulation-only columns that the live production race-table
    preparation ordinarily creates after fundamentals are complete.

    Historical group assignments must be present or reconstructable
    from historical covariates. The adapter deliberately refuses to
    create a single fabricated unknown group.
    """
    out = normalize_fixed_control(df.copy())

    production_margin = pd.to_numeric(
        model_margin,
        errors="coerce",
    )

    legacy_fallback_margin = pd.to_numeric(
        fallback_margin,
        errors="coerce",
    )

    if len(production_margin) != len(out):
        raise ReplayValidationError(
            "Production-margin length does not match the "
            "historical race table."
        )

    if len(legacy_fallback_margin) != len(out):
        raise ReplayValidationError(
            "Fallback-margin length does not match the "
            "historical race table."
        )

    # Assign positionally because both margin series were constructed
    # from this cycle's prepared race table. This avoids accidental
    # index-alignment errors if a prior transformation retained a
    # nonconsecutive index.
    production_margin = pd.Series(
        production_margin.to_numpy(),
        index=out.index,
        dtype=float,
    )

    legacy_fallback_margin = pd.Series(
        legacy_fallback_margin.to_numpy(),
        index=out.index,
        dtype=float,
    )

    fallback_mask = production_margin.isna()

    # Option B is intentionally narrow: only the known 2016 Florida
    # baseline gap may use the legacy-margin fallback. Any new missing
    # production margins elsewhere should still stop the replay.
    if fallback_mask.any():
        fallback_cycle = pd.to_numeric(
            out.loc[
                fallback_mask,
                "forecast_cycle",
            ],
            errors="coerce",
        )

        fallback_state = (
            out.loc[
                fallback_mask,
                "state",
            ]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.strip()
        )

        allowed_fallback = (
            fallback_cycle.eq(2016)
            & fallback_state.eq("FL")
        )

        if not allowed_fallback.all():
            unexpected_rows = out.loc[
                fallback_mask
                & ~allowed_fallback.reindex(
                    out.index,
                    fill_value=False,
                ),
                "district_id",
            ].astype(str).head(20).tolist()

            raise ReplayValidationError(
                "Missing production margins were found outside "
                "the approved 2016 Florida fallback: "
                + ", ".join(unexpected_rows)
            )

        missing_fallback = (
            fallback_mask
            & legacy_fallback_margin.isna()
        )

        if missing_fallback.any():
            bad_rows = out.loc[
                missing_fallback,
                "district_id",
            ].astype(str).head(20).tolist()

            raise ReplayValidationError(
                "Legacy fallback margins are also missing for: "
                + ", ".join(bad_rows)
            )

    out["production_model_margin_dem"] = production_margin
    out["legacy_fallback_margin_dem"] = (
        legacy_fallback_margin
    )
    out["shared_margin_fallback_used"] = fallback_mask
    out["shared_margin_source"] = np.where(
        fallback_mask,
        "legacy_margin_fallback",
        "production_fundamentals",
    )

    out["model_margin_dem"] = production_margin.where(
        ~fallback_mask,
        legacy_fallback_margin,
    )

    if out["model_margin_dem"].isna().any():
        bad_rows = out.loc[
            out["model_margin_dem"].isna(),
            "district_id",
        ].astype(str).head(20).tolist()

        raise ReplayValidationError(
            "Shared simulation still has missing margins after "
            "the approved fallback: "
            + ", ".join(bad_rows)
        )

    fallback_count = int(fallback_mask.sum())

    if fallback_count:
        fallback_ids = out.loc[
            fallback_mask,
            "district_id",
        ].astype(str).tolist()

        print()
        print("SHARED REPLAY MARGIN FALLBACK")
        print("-" * 72)
        print(
            "Production margins replaced with leakage-safe "
            f"legacy margins: {fallback_count}"
        )
        print(
            "Fallback districts: "
            + ", ".join(fallback_ids)
        )

    required_direct_groups = [
        "state_error_group",
        "region_error_group",
        "district_type_error_group",
    ]

    missing_direct_groups = [
        column
        for column in required_direct_groups
        if column not in out.columns
    ]

    if missing_direct_groups:
        raise ReplayValidationError(
            "Shared production replay is missing historical "
            "simulation-group columns: "
            + ", ".join(missing_direct_groups)
        )

    for column in required_direct_groups:
        normalized = (
            out[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        if normalized.eq("").any():
            bad_rows = out.loc[
                normalized.eq(""),
                "district_id",
            ].astype(str).head(20).tolist()

            raise ReplayValidationError(
                f"Historical group column {column} contains blank "
                "assignments: "
                + ", ".join(bad_rows)
            )

        out[column] = normalized

    # Match the live production grouping definition whenever the
    # explicit historical error-group column is unavailable.
    if "education_race_error_group" in out.columns:
        education_race_group = (
            out["education_race_error_group"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
    else:
        source_columns = [
            "college_share_tier",
            "white_share_tier",
        ]

        historical_tiers_available = all(
            column in out.columns
            for column in source_columns
        )

        if historical_tiers_available:
            college = (
                out["college_share_tier"]
                .fillna("")
                .astype(str)
                .str.strip()
            )
            white = (
                out["white_share_tier"]
                .fillna("")
                .astype(str)
                .str.strip()
            )

            complete_tiers = (
                college.ne("")
                & white.ne("")
            )

            # Use genuine historical groups where available. Rows
            # without historical demographic tiers receive unique
            # inert labels. The shared historical replay config sets
            # education_race_error_sd to zero, so these labels do not
            # affect any simulation draw.
            education_race_group = pd.Series(
                "historical_demographics_unavailable__"
                + out["district_id"].astype(str),
                index=out.index,
                dtype=str,
            )

            education_race_group.loc[
                complete_tiers
            ] = (
                college.loc[complete_tiers]
                + " College / "
                + white.loc[complete_tiers]
                + " White"
            )
        else:
            education_race_group = (
                "historical_demographics_unavailable__"
                + out["district_id"].astype(str)
            )

    invalid_education_group = (
        education_race_group.eq("")
        | education_race_group.str.lower().isin(
            {
                "unknown",
                "unknown education/race",
                "nan",
                "none",
            }
        )
    )

    if invalid_education_group.any():
        bad_rows = out.loc[
            invalid_education_group,
            "district_id",
        ].astype(str).head(20).tolist()

        raise ReplayValidationError(
            "Historical education/race groups are unavailable for: "
            + ", ".join(bad_rows)
        )

    out["education_race_error_group"] = (
        education_race_group
    )

    # The current shared production simulator does not draw the older
    # aggregate demographic-error component. This column is retained
    # only because the production summary contract reports its count.
    if "demographic_error_group" not in out.columns:
        out["demographic_error_group"] = (
            out["education_race_error_group"]
        )
    else:
        out["demographic_error_group"] = (
            out["demographic_error_group"]
            .fillna(out["education_race_error_group"])
            .astype(str)
            .str.strip()
        )

    # Historical v2 intentionally contains no district polling.
    # Election-Day production begins with a 4.75-point posterior SD;
    # the shared engine then applies the variance-preserving residual
    # floor exactly as it does in the live model.
    out["bayesian_polling_weight"] = 0.0
    out["district_posterior_sd"] = 4.75

    defaults = {
        "general_election_party_structure": "D_vs_R",
        "election_system": "standard",
        "other_candidate": "",
        "imported_national_environment_margin_dem": np.nan,
        "house_environment_multiplier": np.nan,
        "house_national_environment_used_dem": np.nan,
    }

    for column, default in defaults.items():
        if column not in out.columns:
            out[column] = default

    out["general_election_party_structure"] = (
        out["general_election_party_structure"]
        .fillna("D_vs_R")
        .astype(str)
        .str.strip()
    )

    out["election_system"] = (
        out["election_system"]
        .fillna("standard")
        .astype(str)
        .str.strip()
    )

    out["other_candidate"] = (
        out["other_candidate"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return out


def run_production_shared_spec(
    df: pd.DataFrame,
    model_margin: pd.Series,
    fallback_margin: pd.Series,
    n_sims: int,
    seed: int,
    fixed_error_sd: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the current production House simulation engine against
    leakage-safe historical fundamentals.
    """
    replay_table = prepare_shared_simulation_table(
        df=df,
        model_margin=model_margin,
        fallback_margin=fallback_margin,
    )

    config = HouseModelConfig(
        n_sims=int(n_sims),
        seed=int(seed),
        majority_threshold=HOUSE_CONTROL_THRESHOLD,

        # Historical education/race group assignments have not yet
        # been constructed on cycle-appropriate district boundaries.
        # Setting this component to zero avoids current-boundary
        # leakage while retaining every other shared production
        # uncertainty component.
        education_race_error_sd=0.0,
    )

    (
        race_stats,
        seat_distribution,
        simulation_draws,
        shared_summary,
    ) = run_shared_house_simulation(
        race_table=replay_table,
        days_out=0,
        config=config,
    )

    replay_df = race_stats.copy()

    audit_columns = [
        "district_id",
        "production_model_margin_dem",
        "legacy_fallback_margin_dem",
        "shared_margin_fallback_used",
        "shared_margin_source",
    ]

    missing_audit_columns = [
        column
        for column in audit_columns[1:]
        if column not in replay_df.columns
    ]

    if missing_audit_columns:
        audit = replay_table[
            audit_columns
        ].copy()

        replay_df = replay_df.merge(
            audit,
            on="district_id",
            how="left",
            validate="one_to_one",
        )

    if "simulated_dem_win_probability" not in replay_df.columns:
        raise ReplayValidationError(
            "Shared production simulation did not return "
            "simulated_dem_win_probability."
        )

    replay_df["dem_win_probability"] = pd.to_numeric(
        replay_df["simulated_dem_win_probability"],
        errors="coerce",
    )

    if replay_df["dem_win_probability"].isna().any():
        raise ReplayValidationError(
            "Shared production simulation returned missing "
            "race probabilities."
        )

    # Supply a deterministic marginal-SD diagnostic for the canonical
    # scorer. The scorer uses dem_win_probability when it is present.
    replay_df["total_error_sd"] = float(
        shared_summary["total_error_sd"]
    )

    results, summary, calibration = score_backtest(
        df=replay_df,
        model_margin=model_margin,
        default_error_sd=fixed_error_sd,
    )

    results.insert(
        0,
        "replay_spec",
        PRODUCTION_SHARED_SPEC,
    )
    summary.insert(
        0,
        "replay_spec",
        PRODUCTION_SHARED_SPEC,
    )
    calibration.insert(
        0,
        "replay_spec",
        PRODUCTION_SHARED_SPEC,
    )

    dem_seats = pd.to_numeric(
        simulation_draws["dem_seats"],
        errors="raise",
    )

    simulated_expected_seats = float(
        dem_seats.mean()
    )
    simulated_seat_sd = float(
        dem_seats.std(ddof=1)
    )
    simulation_standard_error = float(
        simulated_seat_sd / np.sqrt(n_sims)
    )

    fallback_count = int(
        replay_table[
            "shared_margin_fallback_used"
        ].sum()
    )

    fallback_districts = ",".join(
        replay_table.loc[
            replay_table[
                "shared_margin_fallback_used"
            ],
            "district_id",
        ].astype(str)
    )

    summary_values: dict[str, Any] = {
        "uncertainty_days_out": 0,
        "shared_margin_fallback_count": fallback_count,
        "shared_margin_fallback_districts": (
            fallback_districts
        ),
        "national_error_sd": float(
            shared_summary["national_error_sd"]
        ),
        "state_error_sd": float(
            shared_summary["state_error_sd"]
        ),
        "region_error_sd": float(
            shared_summary["region_error_sd"]
        ),
        "district_type_error_sd": float(
            shared_summary["district_type_error_sd"]
        ),
        "education_race_error_sd": float(
            shared_summary["education_race_error_sd"]
        ),
        "historical_education_race_component_deferred": True,
        # Compatibility alias used by older replay output readers.
        "demographic_error_sd": float(
            shared_summary["education_race_error_sd"]
        ),
        "district_error_sd": float(
            shared_summary[
                "district_specific_error_sd_floor"
            ]
        ),
        "marginal_total_error_sd": float(
            shared_summary["total_error_sd"]
        ),
        "n_sims": int(n_sims),
        "expected_dem_seats_from_simulation": (
            simulated_expected_seats
        ),
        "simulation_dem_seat_sd": simulated_seat_sd,
        "simulation_expected_seat_standard_error": (
            simulation_standard_error
        ),
        "median_dem_seats": float(
            dem_seats.median()
        ),
        "dem_seats_p25": float(
            dem_seats.quantile(0.25)
        ),
        "dem_seats_p50": float(
            dem_seats.quantile(0.50)
        ),
        "dem_seats_p75": float(
            dem_seats.quantile(0.75)
        ),
        "dem_control_probability": float(
            shared_summary["dem_majority_probability"]
        ),
        "shared_state_error_groups": int(
            shared_summary["state_error_groups"]
        ),
        "shared_region_error_groups": int(
            shared_summary["region_error_groups"]
        ),
        "shared_district_type_error_groups": int(
            shared_summary["district_type_error_groups"]
        ),
        "shared_education_race_error_groups": int(
            shared_summary[
                "education_race_error_groups"
            ]
        ),
        "shared_grouped_variance": float(
            config.total_error_sd ** 2
            - shared_summary[
                "district_specific_error_sd_floor"
            ] ** 2
        ),
    }

    for key, value in summary_values.items():
        summary[key] = value

    analytic_expected_seats = float(
        summary.loc[0, "expected_dem_seats"]
    )

    difference = abs(
        analytic_expected_seats
        - simulated_expected_seats
    )

    tolerance = max(
        0.10,
        5.0 * simulation_standard_error,
    )

    summary[
        "analytic_minus_simulated_expected_seats"
    ] = (
        analytic_expected_seats
        - simulated_expected_seats
    )
    summary[
        "simulation_validation_tolerance"
    ] = tolerance

    if difference > tolerance:
        raise ReplayValidationError(
            "Shared production expected seats differ from the "
            "direct simulation mean by more than the Monte Carlo "
            "tolerance. "
            f"Difference: {difference:.6f}; "
            f"standard error: "
            f"{simulation_standard_error:.6f}; "
            f"tolerance: {tolerance:.6f}"
        )

    fixed = (
        replay_df["party_control_fixed"]
        .fillna("")
        .astype(str)
        .str.upper()
    )

    if not replay_df.loc[
        fixed.eq("D"),
        "dem_win_probability",
    ].eq(1.0).all():
        raise ReplayValidationError(
            "Shared simulation fixed Democratic seats are not "
            "all probability 1.0."
        )

    if not replay_df.loc[
        fixed.eq("R"),
        "dem_win_probability",
    ].eq(0.0).all():
        raise ReplayValidationError(
            "Shared simulation fixed Republican seats are not "
            "all probability 0.0."
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

    def select_spec(spec: str) -> pd.DataFrame:
        selected = (
            summaries.loc[
                summaries["replay_spec"].eq(spec),
                ["cycle", *metrics],
            ]
            .set_index("cycle")
            .sort_index()
        )

        if selected.empty:
            raise ReplayValidationError(
                f"No summary rows found for replay specification: {spec}"
            )

        if not selected.index.is_unique:
            raise ReplayValidationError(
                f"Duplicate cycle rows found for replay specification: {spec}"
            )

        return selected

    legacy = select_spec(LEGACY_SPEC)
    production_fundamentals = select_spec(
        PRODUCTION_FUNDAMENTALS_SPEC
    )
    production = select_spec(PRODUCTION_SPEC)

    if not legacy.index.equals(
        production_fundamentals.index
    ):
        raise ReplayValidationError(
            "Legacy and production-fundamentals cycle keys differ."
        )

    if not legacy.index.equals(production.index):
        raise ReplayValidationError(
            "Legacy and production cycle keys differ."
        )

    rows: list[dict[str, Any]] = []

    for cycle in legacy.index:
        row: dict[str, Any] = {
            "cycle": int(cycle),
        }

        for metric in metrics:
            legacy_value = float(
                legacy.loc[cycle, metric]
            )
            production_fundamentals_value = float(
                production_fundamentals.loc[
                    cycle,
                    metric,
                ]
            )
            production_value = float(
                production.loc[cycle, metric]
            )

            row[f"legacy_{metric}"] = legacy_value
            row[
                f"production_fundamentals_{metric}"
            ] = production_fundamentals_value
            row[f"production_{metric}"] = production_value

            row[
                f"production_fundamentals_minus_legacy_{metric}"
            ] = (
                production_fundamentals_value
                - legacy_value
            )

            row[
                f"production_minus_production_fundamentals_{metric}"
            ] = (
                production_value
                - production_fundamentals_value
            )

            row[
                f"production_minus_legacy_{metric}"
            ] = (
                production_value
                - legacy_value
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
        * len(REPLAY_SPECS)
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
        * len(REPLAY_SPECS)
    )

    if len(summaries) != expected_summary_rows:
        raise ReplayValidationError(
            f"Expected {expected_summary_rows} summary rows; "
            f"found {len(summaries)}."
        )

    checks.append(
        f"PASS: summary rows = {expected_summary_rows}"
    )

    observed_specs = tuple(
        sorted(results["replay_spec"].unique())
    )

    expected_specs = tuple(
        sorted(REPLAY_SPECS)
    )

    if observed_specs != expected_specs:
        raise ReplayValidationError(
            "Replay specifications do not match. "
            f"Expected {expected_specs}; "
            f"found {observed_specs}."
        )

    checks.append(
        "PASS: replay specification set is complete"
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
            "Run leakage-safe full House production replay."
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

        prepared_df, national_environment = prepare_cycle(
            master=master,
            cycle=cycle,
            candidate_quality_weight=(
                args.candidate_quality_weight
            ),
            candidate_war_path=args.candidate_war_path,
        )

        legacy_margin, legacy_forecast_source = build_model_margin(
            prepared_df.copy()
        )

        df, model_margin, forecast_source = (
            build_production_fundamentals(
                df=prepared_df,
                cycle=cycle,
                national_environment=national_environment,
            )
        )

        print()
        print(f"Margin comparison ({cycle})")
        print("-" * 40)
        print(f"Legacy source:     {legacy_forecast_source}")
        print(f"Production source: {forecast_source}")

        if legacy_margin is None:
            print("Legacy margin unavailable.")
        else:
            production_numeric = pd.to_numeric(
                model_margin,
                errors="coerce",
            )
            legacy_numeric = pd.to_numeric(
                legacy_margin,
                errors="coerce",
            )

            comparable_mask = (
                production_numeric.notna()
                & legacy_numeric.notna()
            )

            comparable_delta = (
                production_numeric.loc[comparable_mask]
                - legacy_numeric.loc[comparable_mask]
            )

            print(f"Rows compared:     {len(comparable_delta):,}")

            if comparable_delta.empty:
                print("No finite margin pairs were available.")
            else:
                print(
                    f"Mean delta:        "
                    f"{comparable_delta.mean():.6f}"
                )
                print(
                    f"Median delta:      "
                    f"{comparable_delta.median():.6f}"
                )
                print(
                    f"Max |delta|:       "
                    f"{comparable_delta.abs().max():.6f}"
                )
                print(
                    f"95th pct |delta|:  "
                    f"{comparable_delta.abs().quantile(0.95):.6f}"
                )

        forecast_sources[str(cycle)] = forecast_source

        if legacy_margin is None:
            raise ReplayValidationError(
                f"Cycle {cycle} legacy margin unavailable: "
                f"{legacy_forecast_source}"
            )

        # Resolve the historically valid production margin once before
        # running the production replay specifications. This reuses the
        # shared replay's audited, leakage-safe fallback logic, including
        # the narrow 2016 Florida exception.
        effective_production_table = prepare_shared_simulation_table(
            df=df,
            model_margin=model_margin,
            fallback_margin=legacy_margin,
        )

        effective_production_margin = pd.to_numeric(
            effective_production_table["model_margin_dem"],
            errors="coerce",
        )

        if effective_production_margin.isna().any():
            bad_rows = effective_production_table.loc[
                effective_production_margin.isna(),
                "district_id",
            ].astype(str).head(20).tolist()

            raise ReplayValidationError(
                "Effective production margin remains missing after "
                "the approved historical fallback: "
                + ", ".join(bad_rows)
            )

        # Preserve the fallback audit columns on the common production
        # dataframe so downstream replay outputs can identify which
        # central estimates required the historical exception.
        fallback_audit_columns = [
            "production_model_margin_dem",
            "legacy_fallback_margin_dem",
            "shared_margin_fallback_used",
            "shared_margin_source",
        ]

        for column in fallback_audit_columns:
            if column in effective_production_table.columns:
                df[column] = effective_production_table[
                    column
                ].to_numpy()

        legacy_results, legacy_summary, legacy_calibration = (
            run_fixed_probability_spec(
                df=prepared_df,
                model_margin=legacy_margin,
                fixed_error_sd=args.fixed_error_sd,
                replay_spec=LEGACY_SPEC,
            )
        )

        (
            production_fundamentals_results,
            production_fundamentals_summary,
            production_fundamentals_calibration,
        ) = run_fixed_probability_spec(
            df=df,
            model_margin=effective_production_margin,
            fixed_error_sd=args.fixed_error_sd,
            replay_spec=PRODUCTION_FUNDAMENTALS_SPEC,
        )

        production_results, production_summary, production_calibration = (
            run_production_spec(
                df=df,
                model_margin=effective_production_margin,
                settings=settings,
                n_sims=args.sims,
                seed=args.seed + cycle,
                fixed_error_sd=args.fixed_error_sd,
            )
        )

        (
            production_shared_results,
            production_shared_summary,
            production_shared_calibration,
        ) = run_production_shared_spec(
            df=df,
            model_margin=model_margin,
            fallback_margin=legacy_margin,
            n_sims=args.sims,
            seed=args.seed + cycle,
            fixed_error_sd=args.fixed_error_sd,
        )

        result_frames.extend(
            [
                legacy_results,
                production_fundamentals_results,
                production_results,
                production_shared_results,
            ]
        )

        summary_frames.extend(
            [
                legacy_summary,
                production_fundamentals_summary,
                production_summary,
                production_shared_summary,
            ]
        )

        calibration_frames.extend(
            [
                legacy_calibration,
                production_fundamentals_calibration,
                production_calibration,
                production_shared_calibration,
            ]
        )

        print(
            pd.concat(
                [
                    legacy_summary,
                    production_fundamentals_summary,
                    production_summary,
                    production_shared_summary,
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

    probability_columns = [
        column
        for column in results.columns
        if "prob" in column.lower()
    ]

    print()
    print("MISSING PROBABILITY DIAGNOSTIC")
    print("-" * 72)
    print(f"Probability columns: {probability_columns}")

    for column in probability_columns:
        missing_mask = results[column].isna()

        if not missing_mask.any():
            continue

        print()
        print(
            f"{column}: {int(missing_mask.sum())} missing rows"
        )

        diagnostic_columns = [
            candidate
            for candidate in [
                "cycle",
                "replay_spec",
                "district_id",
                "state",
                "district",
                "model_margin_dem",
                "district_partisan_baseline_dem",
                "pres_2024_margin_dem",
                "pres_2020_margin_dem",
                column,
            ]
            if candidate in results.columns
        ]

        print(
            results.loc[
                missing_mask,
                diagnostic_columns,
            ]
            .head(100)
            .to_string(index=False)
        )

    probability_columns = [
        column
        for column in results.columns
        if "prob" in column.lower()
    ]

    print()
    print("MISSING PROBABILITY DIAGNOSTIC")
    print("-" * 72)
    print(f"Probability columns: {probability_columns}")

    for column in probability_columns:
        missing_mask = results[column].isna()

        if not missing_mask.any():
            continue

        print()
        print(
            f"{column}: {int(missing_mask.sum())} missing rows"
        )

        diagnostic_columns = [
            candidate
            for candidate in [
                "cycle",
                "replay_spec",
                "district_id",
                "state",
                "district",
                "model_margin_dem",
                "district_partisan_baseline_dem",
                "pres_2024_margin_dem",
                "pres_2020_margin_dem",
                column,
            ]
            if candidate in results.columns
        ]

        print(
            results.loc[
                missing_mask,
                diagnostic_columns,
            ]
            .head(100)
            .to_string(index=False)
        )

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
        "replay_version": "house_full_production_replay_v2",
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
        "specifications": list(REPLAY_SPECS),
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
        "shared_v2_components_included": [
            "national_error",
            "state_error_groups",
            "region_error_groups",
            "district_type_error_groups",
            "variance_preserving_district_error",
            "fixed_party_control",
        ],
        "shared_v2_components_deferred": [
            "historical_education_race_error_groups",
            "historical_polling",
            "polling_variance_reduction",
        ],
        "shared_v2_margin_fallback_policy": {
            "allowed_cycle": 2016,
            "allowed_state": "FL",
            "fallback_source": (
                "legacy_leakage_safe_model_margin"
            ),
            "reason": (
                "Missing 2012 presidential district baselines "
                "on the 2016 Florida House boundaries"
            ),
            "fallback_is_applied_before_shared_simulation": True,
        },
        "shared_v2_production_config": {
            "total_error_sd": HouseModelConfig.total_error_sd,
            "national_error_share": (
                HouseModelConfig.national_error_share
            ),
            "state_error_sd": HouseModelConfig.state_error_sd,
            "region_error_sd": HouseModelConfig.region_error_sd,
            "district_type_error_sd": (
                HouseModelConfig.district_type_error_sd
            ),
            "production_education_race_error_sd": (
                HouseModelConfig.education_race_error_sd
            ),
            "historical_replay_education_race_error_sd": 0.0,
            "majority_threshold": HOUSE_CONTROL_THRESHOLD,
        },
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
            "House Full Production Replay Validation",
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
    print("Cycle-level replay decomposition")
    print("=" * 72)

    display_columns = [
        "cycle",

        "production_fundamentals_minus_legacy_brier_score",
        "production_minus_production_fundamentals_brier_score",
        "production_minus_legacy_brier_score",

        "production_fundamentals_minus_legacy_log_loss",
        "production_minus_production_fundamentals_log_loss",
        "production_minus_legacy_log_loss",

        "production_fundamentals_minus_legacy_expected_seat_error",
        "production_minus_production_fundamentals_expected_seat_error",
        "production_minus_legacy_expected_seat_error",
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
