#!/usr/bin/env python3
"""
Shared House Monte Carlo simulation engine.

This module contains the production simulation implementation used by
both the current-cycle House forecast and, after replay integration,
the historical production replay.

The initial extraction is intentionally behavior-preserving. Simulation
mathematics should not be changed during the shared-helper refactor.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def draw_group_errors(rng, groups, n_sims, sd):
    labels = pd.Series(groups).fillna("Unknown").astype(str)
    unique_groups = sorted(labels.unique().tolist())

    group_draws = {
        group: rng.normal(0.0, sd, size=n_sims)
        for group in unique_groups
    }

    return np.column_stack([group_draws[group] for group in labels])


def run_simulation(race_table, days_out, config):
    rng = np.random.default_rng(config.seed)

    n_districts = len(race_table)

    total_sd = config.total_error_sd

    if days_out > 180:
        total_sd *= 1.15
    elif days_out > 120:
        total_sd *= 1.05
    elif days_out < 30:
        total_sd *= 0.80

    national_sd = total_sd * config.national_error_share

    base_margins = race_table["model_margin_dem"].to_numpy(dtype=float).reshape(1, n_districts)

    national_error = rng.normal(0.0, national_sd, size=(config.n_sims, 1))

    state_error = draw_group_errors(
        rng,
        race_table["state_error_group"],
        config.n_sims,
        config.state_error_sd,
    )

    region_error = draw_group_errors(
        rng,
        race_table["region_error_group"],
        config.n_sims,
        config.region_error_sd,
    )

    district_type_error = draw_group_errors(
        rng,
        race_table["district_type_error_group"],
        config.n_sims,
        config.district_type_error_sd,
    )

    education_race_error = draw_group_errors(
        rng,
        race_table["education_race_error_group"],
        config.n_sims,
        config.education_race_error_sd,
    )

    grouped_variance = (
        national_sd ** 2
        + config.state_error_sd ** 2
        + config.region_error_sd ** 2
        + config.district_type_error_sd ** 2
        + config.education_race_error_sd ** 2
    )

    remaining_sd = np.sqrt(max(total_sd ** 2 - grouped_variance, 2.5 ** 2))

    district_posterior_sd = race_table["district_posterior_sd"].to_numpy(dtype=float)
    district_specific_sd = np.maximum(district_posterior_sd, remaining_sd)

    district_error = rng.normal(
        0.0,
        district_specific_sd.reshape(1, n_districts),
        size=(config.n_sims, n_districts),
    )

    simulated_margins = (
        base_margins
        + national_error
        + state_error
        + region_error
        + district_type_error
        + education_race_error
        + district_error
    )

    dem_wins = simulated_margins > 0

    # Apply fixed party-control outcomes for same-party general elections
    # or explicit party-control overrides.
    fixed = race_table.get("party_control_fixed", pd.Series([""] * n_districts))
    fixed = fixed.fillna("").astype(str).str.upper().to_numpy()

    fixed_dem = fixed == "D"
    fixed_gop = fixed == "R"

    if fixed_dem.any():
        dem_wins[:, fixed_dem] = True

    if fixed_gop.any():
        dem_wins[:, fixed_gop] = False

    dem_seats = dem_wins.sum(axis=1)

    # NOTE:
    #
    # The simulator reports Jeffreys-smoothed district probabilities
    # rather than raw Monte Carlo frequencies. This prevents finite
    # simulation runs from producing artificial 0% and 100%
    # probabilities that distort log-loss evaluation.
    #
    # The underlying simulation draws, expected seats, seat
    # distribution, and majority probability remain unchanged.
    dem_win_counts = dem_wins.sum(axis=0)

    raw_district_win_probs = (
        dem_win_counts / float(config.n_sims)
    )

    district_win_probs = (
        dem_win_counts + 0.5
    ) / (
        float(config.n_sims) + 1.0
    )

    # Fixed-party races are structural certainties rather than
    # Monte Carlo estimates, so retain exact 0% or 100% values.
    district_win_probs[fixed_dem] = 1.0
    district_win_probs[fixed_gop] = 0.0

    avg_simulated_margin = simulated_margins.mean(axis=0)

    seat_distribution = (
        pd.Series(dem_seats)
        .value_counts(normalize=True)
        .sort_index()
        .reset_index()
    )
    seat_distribution.columns = ["dem_seats", "probability"]

    race_stats = race_table.copy()
    # The simulation probability is the authoritative party-control
    # probability. This is especially important for same-party and
    # unopposed general elections, where the unconstrained probability
    # implied by model_margin_dem is not a valid seat-control probability.
    race_stats["raw_simulated_dem_win_probability"] = (
        raw_district_win_probs
    )
    race_stats["simulated_dem_win_probability"] = (
        district_win_probs
    )
    race_stats["dem_win_probability"] = (
        district_win_probs
    )
    race_stats["avg_simulated_margin_dem"] = avg_simulated_margin
    race_stats["margin_p25_dem"] = np.percentile(simulated_margins, 25, axis=0)
    race_stats["margin_p50_dem"] = np.percentile(simulated_margins, 50, axis=0)
    race_stats["margin_p75_dem"] = np.percentile(simulated_margins, 75, axis=0)

    majority_prob = float((dem_seats >= config.majority_threshold).mean())

    summary = {
        "n_sims": config.n_sims,
        "days_out": days_out,
        "expected_dem_seats": float(dem_seats.mean()),
        "median_dem_seats": float(np.median(dem_seats)),
        "dem_seats_p25": float(np.percentile(dem_seats, 25)),
        "dem_seats_p50": float(np.percentile(dem_seats, 50)),
        "dem_seats_p75": float(np.percentile(dem_seats, 75)),
        "dem_majority_probability": majority_prob,
        "majority_threshold": config.majority_threshold,
        "total_error_sd": total_sd,
        "national_error_sd": national_sd,
        "state_error_sd": config.state_error_sd,
        "region_error_sd": config.region_error_sd,
        "district_type_error_sd": config.district_type_error_sd,
        "education_race_error_sd": config.education_race_error_sd,
        "district_specific_error_sd_floor": remaining_sd,
        "national_error_share": config.national_error_share,
        "imported_national_environment_margin": (
            float(
                race_table[
                    "imported_national_environment_margin_dem"
                ].dropna().iloc[0]
            )
            if "imported_national_environment_margin_dem"
            in race_table.columns
            and race_table[
                "imported_national_environment_margin_dem"
            ].notna().any()
            else np.nan
        ),
        "house_environment_multiplier": (
            float(
                race_table[
                    "house_environment_multiplier"
                ].dropna().iloc[0]
            )
            if "house_environment_multiplier" in race_table.columns
            and race_table[
                "house_environment_multiplier"
            ].notna().any()
            else np.nan
        ),
        "house_adjusted_national_environment": (
            float(
                race_table[
                    "house_national_environment_used_dem"
                ].dropna().iloc[0]
            )
            if "house_national_environment_used_dem"
            in race_table.columns
            and race_table[
                "house_national_environment_used_dem"
            ].notna().any()
            else (
                float(
                    race_table[
                        "national_environment_margin_dem"
                    ].dropna().iloc[0]
                )
                if "national_environment_margin_dem"
                in race_table.columns
                and race_table[
                    "national_environment_margin_dem"
                ].notna().any()
                else np.nan
            )
        ),
        # Legacy alias retained for compatibility. This is the
        # House-adjusted value actually used in district forecasts.
        "national_environment_margin": (
            float(
                race_table[
                    "house_national_environment_used_dem"
                ].dropna().iloc[0]
            )
            if "house_national_environment_used_dem"
            in race_table.columns
            and race_table[
                "house_national_environment_used_dem"
            ].notna().any()
            else (
                float(
                    race_table[
                        "national_environment_margin_dem"
                    ].dropna().iloc[0]
                )
                if "national_environment_margin_dem"
                in race_table.columns
                and race_table[
                    "national_environment_margin_dem"
                ].notna().any()
                else np.nan
            )
        ),
        "average_polling_weight": float(race_table["bayesian_polling_weight"].mean()),
        "districts_with_polling": int((race_table["bayesian_polling_weight"] > 0).sum()),
        "state_error_groups": int(race_table["state_error_group"].nunique()),
        "region_error_groups": int(race_table["region_error_group"].nunique()),
        "district_type_error_groups": int(race_table["district_type_error_group"].nunique()),
        "education_race_error_groups": int(race_table["education_race_error_group"].nunique()),
        "demographic_error_groups": int(race_table["demographic_error_group"].nunique()),
        "fixed_dem_control_districts": int((race_table["party_control_fixed"] == "D").sum()),
        "fixed_gop_control_districts": int((race_table["party_control_fixed"] == "R").sum()),
        "d_unopposed_districts": int((race_table["general_election_party_structure"] == "D_unopposed").sum()),
        "r_unopposed_districts": int((race_table["general_election_party_structure"] == "R_unopposed").sum()),
        "d_vs_d_districts": int((race_table["general_election_party_structure"] == "D_vs_D").sum()),
        "r_vs_r_districts": int((race_table["general_election_party_structure"] == "R_vs_R").sum()),
        "other_candidate_districts": int(race_table["other_candidate"].fillna("").astype(str).str.strip().ne("").sum()) if "other_candidate" in race_table.columns else 0,
        "top_two_districts": int((race_table["election_system"] == "top_two").sum()),
        "top_four_rcv_districts": int((race_table["election_system"] == "top_four_rcv").sum()),
    }

    simulation_draws = pd.DataFrame({
        "simulation": np.arange(1, config.n_sims + 1),
        "dem_seats": dem_seats,
    })

    return race_stats, seat_distribution, simulation_draws, summary
