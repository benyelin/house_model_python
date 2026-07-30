from pathlib import Path
from dataclasses import dataclass
from datetime import date
import argparse
import numpy as np
import pandas as pd

INPUTS = Path("inputs")
OUTPUTS = Path("outputs")

HOUSE_INPUT_PATH = INPUTS / "house_race_inputs.csv"

ELECTION_DAY = date(2026, 11, 3)


@dataclass
class HouseModelConfig:
    n_sims: int = 20000
    seed: int = 20260529

    total_error_sd: float = 7.5
    national_error_share: float = 0.55

    state_error_sd: float = 1.75
    region_error_sd: float = 1.25
    district_type_error_sd: float = 1.00
    education_race_error_sd: float = 0.75

    probability_scale: float = 6.0
    majority_threshold: int = 218


def compute_days_out(today=None):
    if today is None:
        today = date.today()
    return max(0, (ELECTION_DAY - today).days)


def cycle_polling_cap(days_out):
    if days_out > 180:
        return 0.12
    if days_out > 120:
        return 0.18
    if days_out > 60:
        return 0.35
    if days_out > 30:
        return 0.50
    return 0.70


def poll_count_multiplier(poll_count):
    """Convert effective/quality poll count into a polling-weight multiplier.

    This intentionally supports fractional effective counts. A district with
    1.05 quality-adjusted polls should behave like a little more than one poll,
    not like a fully mature polling average.
    """
    try:
        poll_count = float(poll_count)
    except Exception:
        poll_count = 0.0

    if poll_count <= 0:
        return 0.0

    # Anchor points:
    # 0 polls -> 0.00
    # 1 poll  -> 0.30
    # 2 polls -> 0.55
    # 3 polls -> 0.75
    # 4+      -> 1.00
    if poll_count <= 1:
        return 0.30 * poll_count

    if poll_count <= 2:
        return 0.30 + (poll_count - 1.0) * (0.55 - 0.30)

    if poll_count <= 3:
        return 0.55 + (poll_count - 2.0) * (0.75 - 0.55)

    if poll_count <= 4:
        return 0.75 + (poll_count - 3.0) * (1.00 - 0.75)

    return 1.0


def normalize_bool(x):
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in ["true", "1", "yes", "y"]


def safe_numeric(df, col, default=np.nan):
    if col not in df.columns:
        df[col] = default
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def ensure_text(df, col, default):
    if col not in df.columns:
        df[col] = default
    df[col] = df[col].fillna(default).astype(str).str.strip()
    return df


def prepare_house_table(df, days_out, config):
    out = df.copy()

    required = [
        "district_id",
        "state",
        "district",
        "fundamentals_margin_dem",
    ]

    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"house_race_inputs.csv missing required columns: {missing}")

    out["state"] = out["state"].astype(str).str.strip().str.upper()
    out["district_id"] = out["district_id"].astype(str).str.strip()

    numeric_cols = [
        "fundamentals_margin_dem",
        "polling_margin_dem",
        "poll_count",
        "effective_poll_count",
        "largest_pollster_weight_share",
        "district_partisan_baseline_dem",
        "district_environment_adjustment_dem",
        "state_environment_adjustment_dem",
        "incumbency_adjustment_dem",
        "candidate_quality_adjustment_dem",
        "special_adjustment_dem",
        "district_elasticity",
        "national_environment_margin_dem",
    ]

    for col in numeric_cols:
        out = safe_numeric(out, col)

    if "polling_active" not in out.columns:
        out["polling_active"] = False

    out["polling_active_bool"] = out["polling_active"].apply(normalize_bool)

    has_polling = (
        out["polling_active_bool"]
        & out["polling_margin_dem"].notna()
    )

    out["poll_count"] = out["poll_count"].fillna(0.0)
    out["effective_poll_count"] = out["effective_poll_count"].fillna(out["poll_count"])
    out["largest_pollster_weight_share"] = out["largest_pollster_weight_share"].fillna(0.0)

    if "only_partisan_or_internal_polls" not in out.columns:
        out["only_partisan_or_internal_polls"] = False

    out["only_partisan_or_internal_polls_bool"] = out[
        "only_partisan_or_internal_polls"
    ].apply(normalize_bool)

    # Polling quality adjustment:
    # - start with Kish effective poll count
    # - penalize averages dominated by one pollster
    # - penalize averages made only from partisan/internal polls
    out["poll_quality_count"] = out["effective_poll_count"].copy()

    concentration_penalty = np.where(
        out["largest_pollster_weight_share"] >= 0.95,
        0.70,
        np.where(out["largest_pollster_weight_share"] >= 0.75, 0.85, 1.00),
    )

    partisan_only_penalty = np.where(
        out["only_partisan_or_internal_polls_bool"],
        0.75,
        1.00,
    )

    out["poll_quality_count"] = (
        out["poll_quality_count"]
        * concentration_penalty
        * partisan_only_penalty
    )

    out["poll_quality_count"] = np.where(
        has_polling,
        out["poll_quality_count"],
        0.0,
    )

    cap = cycle_polling_cap(days_out)

    out["poll_count_multiplier"] = out["poll_quality_count"].apply(poll_count_multiplier)

    out["bayesian_polling_weight"] = np.where(
        has_polling,
        cap * out["poll_count_multiplier"],
        0.0,
    )

    out["bayesian_polling_weight"] = out["bayesian_polling_weight"].clip(
        lower=0.0,
        upper=cap,
    )

    out["bayesian_fundamentals_weight"] = 1.0 - out["bayesian_polling_weight"]

    out["bayesian_model_margin_dem"] = (
        out["fundamentals_margin_dem"] * out["bayesian_fundamentals_weight"]
        + out["polling_margin_dem"].fillna(0.0) * out["bayesian_polling_weight"]
    )

    out["model_margin_dem"] = out["bayesian_model_margin_dem"]

    if days_out > 180:
        base_sd = 8.5
    elif days_out > 120:
        base_sd = 8.0
    elif days_out > 60:
        base_sd = 6.75
    elif days_out > 30:
        base_sd = 5.75
    else:
        base_sd = 4.75

    out["district_posterior_sd"] = base_sd
    out["district_posterior_sd"] = (
        out["district_posterior_sd"]
        * (1.0 - 0.25 * out["bayesian_polling_weight"])
    )

    for col, default in [
        ("state_error_group", None),
        ("region_error_group", "Unknown Region"),
        ("district_type_error_group", "Mixed"),
        ("education_race_error_group", "Unknown Education/Race"),
        ("demographic_error_group", "Unknown Demographic"),
        ("election_system", "standard"),
        ("general_election_party_structure", "unresolved"),
        ("party_control_override", ""),
        ("election_system_notes", ""),
        ("other_candidate", ""),
        ("college_share_tier", "Unknown"),
        ("white_share_tier", "Unknown"),
        ("black_share_tier", "Unknown"),
        ("hispanic_share_tier", "Unknown"),
        ("median_income_tier", "Unknown"),
        ("region", "Unknown Region"),
        ("district_type", "Mixed"),
    ]:
        if col not in out.columns:
            out[col] = default

    out["state_error_group"] = out["state_error_group"].fillna(out["state"]).astype(str).str.strip().str.upper()
    out["region"] = out["region"].fillna("Unknown Region").astype(str).str.strip()
    out["district_type"] = out["district_type"].fillna("Mixed").astype(str).str.strip()
    out["region_error_group"] = out["region_error_group"].fillna(out["region"]).astype(str).str.strip()
    out["district_type_error_group"] = out["district_type_error_group"].fillna(out["district_type"]).astype(str).str.strip()

    out["college_share_tier"] = out["college_share_tier"].fillna("Unknown").astype(str).str.strip()
    out["white_share_tier"] = out["white_share_tier"].fillna("Unknown").astype(str).str.strip()
    out["black_share_tier"] = out["black_share_tier"].fillna("Unknown").astype(str).str.strip()
    out["hispanic_share_tier"] = out["hispanic_share_tier"].fillna("Unknown").astype(str).str.strip()
    out["median_income_tier"] = out["median_income_tier"].fillna("Unknown").astype(str).str.strip()

    out["education_race_error_group"] = (
        out["education_race_error_group"]
        .fillna(out["college_share_tier"] + " College / " + out["white_share_tier"] + " White")
        .astype(str)
        .str.strip()
    )

    out["demographic_error_group"] = (
        out["demographic_error_group"]
        .fillna(
            out["college_share_tier"] + " College / "
            + out["white_share_tier"] + " White / "
            + out["black_share_tier"] + " Black / "
            + out["hispanic_share_tier"] + " Hispanic / "
            + out["median_income_tier"] + " Income"
        )
        .astype(str)
        .str.strip()
    )

    out["election_system"] = out["election_system"].fillna("standard").astype(str).str.strip()
    out["general_election_party_structure"] = (
        out["general_election_party_structure"]
        .fillna("unresolved")
        .astype(str)
        .str.strip()
    )
    out["party_control_override"] = (
        out["party_control_override"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    out["election_system_notes"] = out["election_system_notes"].fillna("").astype(str).str.strip()

    out["party_control_fixed"] = ""

    # Automatic fixed party-control logic.
    # Party Control Override remains an escape hatch, but ordinary same-party
    # and uncontested races are inferred from General Election Party Structure.
    out.loc[
        out["general_election_party_structure"].isin(["D_vs_D", "D_unopposed"]),
        "party_control_fixed",
    ] = "D"

    out.loc[
        out["general_election_party_structure"].isin(["R_vs_R", "R_unopposed"]),
        "party_control_fixed",
    ] = "R"

    # Explicit override wins over automatic inference.
    out.loc[out["party_control_override"].isin(["D", "R"]), "party_control_fixed"] = out.loc[
        out["party_control_override"].isin(["D", "R"]),
        "party_control_override",
    ]

    out["pre_sim_dem_win_probability"] = 1 / (
        1 + np.exp(-out["model_margin_dem"] / config.probability_scale)
    )

    out.loc[out["party_control_fixed"] == "D", "pre_sim_dem_win_probability"] = 1.0
    out.loc[out["party_control_fixed"] == "R", "pre_sim_dem_win_probability"] = 0.0

    return out


def draw_group_errors(
    rng,
    groups,
    n_sims,
    sd,
):
    """Compatibility wrapper for the shared simulation helper."""
    from house_simulation import (
        draw_group_errors as shared_draw_group_errors,
    )

    return shared_draw_group_errors(
        rng,
        groups,
        n_sims,
        sd,
    )


def run_simulation(
    race_table,
    days_out,
    config,
):
    """Run the shared production House simulation engine."""
    from house_simulation import (
        run_simulation as shared_run_simulation,
    )

    return shared_run_simulation(
        race_table,
        days_out,
        config,
    )



def run_forecast(input_path, output_dir, config, today=None):
    if today is None:
        today = date.today()

    days_out = compute_days_out(today)

    if not input_path.exists():
        raise FileNotFoundError(f"Could not find {input_path}")

    df = pd.read_csv(input_path)

    table = prepare_house_table(df, days_out, config)

    race_stats, seat_distribution, simulation_draws, summary = run_simulation(
        table,
        days_out,
        config,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Temporary live-output consistency diagnostics.
    race_probability_sum = float(
        pd.to_numeric(
            race_stats["simulated_dem_win_probability"],
            errors="coerce",
        ).sum()
    )
    summary_expected_seats = float(summary["expected_dem_seats"])
    draw_expected_seats = float(
        pd.to_numeric(
            simulation_draws["dem_seats"],
            errors="coerce",
        ).mean()
    )
    draw_majority_probability = float(
        (
            pd.to_numeric(
                simulation_draws["dem_seats"],
                errors="coerce",
            )
            >= config.majority_threshold
        ).mean()
    )
    summary_majority_probability = float(
        summary["dem_majority_probability"]
    )

    print()
    print("IN-MEMORY HOUSE SIMULATION CONSISTENCY CHECK")
    print("-" * 52)
    print(
        "Sum of district win probabilities: "
        f"{race_probability_sum:.9f}"
    )
    print(
        "Summary expected Democratic seats: "
        f"{summary_expected_seats:.9f}"
    )
    print(
        "Raw-draw expected Democratic seats: "
        f"{draw_expected_seats:.9f}"
    )
    print(
        "Summary Democratic majority probability: "
        f"{summary_majority_probability:.9%}"
    )
    print(
        "Raw-draw Democratic majority probability: "
        f"{draw_majority_probability:.9%}"
    )
    print(
        "District-sum minus summary expected seats: "
        f"{race_probability_sum - summary_expected_seats:+.12f}"
    )
    print(
        "Draw mean minus summary expected seats: "
        f"{draw_expected_seats - summary_expected_seats:+.12f}"
    )
    print(
        "Draw majority minus summary majority: "
        f"{draw_majority_probability - summary_majority_probability:+.12f}"
    )

    if not np.isclose(
        race_probability_sum,
        summary_expected_seats,
        atol=1e-9,
        rtol=0.0,
    ):
        raise RuntimeError(
            "In-memory inconsistency: district probability sum does not "
            "equal summary expected seats."
        )

    if not np.isclose(
        draw_expected_seats,
        summary_expected_seats,
        atol=1e-9,
        rtol=0.0,
    ):
        raise RuntimeError(
            "In-memory inconsistency: raw draw mean does not equal "
            "summary expected seats."
        )

    if not np.isclose(
        draw_majority_probability,
        summary_majority_probability,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            "In-memory inconsistency: raw draw majority probability does "
            "not equal summary majority probability."
        )

    race_stats.to_csv(output_dir / "house_race_stats.csv", index=False)
    seat_distribution.to_csv(output_dir / "house_seat_distribution.csv", index=False)
    simulation_draws.to_csv(output_dir / "house_simulation_draws.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "house_forecast_summary.csv", index=False)

    updated = df.copy()

    writeback_cols = [
        "bayesian_polling_weight",
        "bayesian_fundamentals_weight",
        "bayesian_model_margin_dem",
        "district_posterior_sd",
        "model_margin_dem",
        "pre_sim_dem_win_probability",
        "simulated_dem_win_probability",
        "avg_simulated_margin_dem",
    ]

    for col in writeback_cols:
        if col in race_stats.columns:
            updated[col] = race_stats[col].values

    updated["dem_win_probability"] = race_stats["simulated_dem_win_probability"].values
    updated.to_csv(input_path, index=False)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run House forecast simulation.")
    parser.add_argument("--sims", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--input", default=str(HOUSE_INPUT_PATH))
    parser.add_argument("--output-dir", default=str(OUTPUTS))
    parser.add_argument("--today", default=None, help="Optional YYYY-MM-DD date.")

    args = parser.parse_args()

    if args.today:
        today = date.fromisoformat(args.today)
    else:
        today = date.today()

    config = HouseModelConfig(
        n_sims=args.sims,
        seed=args.seed,
    )

    summary = run_forecast(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        config=config,
        today=today,
    )

    print()
    print("House forecast complete")
    print("-----------------------")
    print(f"Expected Dem seats:     {summary['expected_dem_seats']:.2f}")
    print(f"Median Dem seats:       {summary['median_dem_seats']:.0f}")
    print(f"Dem majority odds:      {summary['dem_majority_probability']:.1%}")
    print(f"Days out:               {summary['days_out']}")
    print(
        "Imported national environment:       "
        f"{summary['imported_national_environment_margin']:+.2f}"
    )
    print(
        "House environment multiplier:        "
        f"x{summary['house_environment_multiplier']:.2f}"
    )
    print(
        "House-adjusted national environment: "
        f"{summary['house_adjusted_national_environment']:+.2f}"
    )
    print(f"Districts with polling: {summary['districts_with_polling']}")
    print(f"Education/race groups:  {summary['education_race_error_groups']}")
    print()
    print(f"Outputs saved to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
