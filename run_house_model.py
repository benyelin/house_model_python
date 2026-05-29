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

    # Error structure. House races are more numerous and more correlated
    # through national environment than individual candidate effects.
    total_error_sd: float = 7.5
    national_error_share: float = 0.65

    # Logistic probability scale for pre-simulation district probability.
    probability_scale: float = 6.0

    # House majority threshold.
    majority_threshold: int = 218

    # Current Democratic seats after 2024, if needed for reference only.
    # The simulation itself counts all 435 races from district probabilities.
    baseline_dem_seats: int = 0


def compute_days_out(today=None):
    if today is None:
        today = date.today()
    return max(0, (ELECTION_DAY - today).days)


def cycle_polling_cap(days_out):
    """
    Same general idea as Senate:
    early polling receives a lower maximum weight.
    """
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
    """
    More polls = more trust.
    """
    try:
        poll_count = float(poll_count)
    except Exception:
        poll_count = 0.0

    if poll_count <= 0:
        return 0.0
    if poll_count == 1:
        return 0.30
    if poll_count == 2:
        return 0.55
    if poll_count == 3:
        return 0.75
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
        "polling_active",
        "district_partisan_baseline_dem",
        "district_environment_adjustment_dem",
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

    # A race is poll-active only if polling_active is true and polling_margin_dem is present.
    has_polling = (
        out["polling_active_bool"]
        & out["polling_margin_dem"].notna()
    )

    if "poll_count" not in out.columns:
        out["poll_count"] = 0.0

    out["poll_count"] = out["poll_count"].fillna(0.0)

    cap = cycle_polling_cap(days_out)

    out["poll_count_multiplier"] = out["poll_count"].apply(poll_count_multiplier)

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

    # First version: model margin equals Bayesian model margin.
    out["model_margin_dem"] = out["bayesian_model_margin_dem"]

    # District uncertainty. Wider early, narrower later.
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

    # Incumbents/open seats can have different district-specific uncertainty later.
    out["district_posterior_sd"] = base_sd

    # Polling lowers uncertainty modestly, but not too much.
    out["district_posterior_sd"] = (
        out["district_posterior_sd"]
        * (1.0 - 0.25 * out["bayesian_polling_weight"])
    )

    out["pre_sim_dem_win_probability"] = 1 / (
        1 + np.exp(-out["model_margin_dem"] / config.probability_scale)
    )

    return out


def run_simulation(race_table, days_out, config):
    rng = np.random.default_rng(config.seed)

    n_districts = len(race_table)

    total_sd = config.total_error_sd

    # Allow total error to narrow slightly closer to Election Day.
    if days_out > 180:
        total_sd *= 1.15
    elif days_out > 120:
        total_sd *= 1.05
    elif days_out < 30:
        total_sd *= 0.80

    national_sd = total_sd * config.national_error_share
    district_sd_floor = total_sd * np.sqrt(1 - config.national_error_share ** 2)

    district_sd = np.maximum(
        race_table["district_posterior_sd"].to_numpy(dtype=float),
        district_sd_floor,
    )

    base_margins = race_table["model_margin_dem"].to_numpy(dtype=float).reshape(1, n_districts)

    national_error = rng.normal(
        0.0,
        national_sd,
        size=(config.n_sims, 1),
    )

    district_error = rng.normal(
        0.0,
        district_sd.reshape(1, n_districts),
        size=(config.n_sims, n_districts),
    )

    simulated_margins = base_margins + national_error + district_error

    dem_wins = simulated_margins > 0
    dem_seats = dem_wins.sum(axis=1)

    district_win_probs = dem_wins.mean(axis=0)
    avg_simulated_margin = simulated_margins.mean(axis=0)

    seat_distribution = (
        pd.Series(dem_seats)
        .value_counts(normalize=True)
        .sort_index()
        .reset_index()
    )
    seat_distribution.columns = ["dem_seats", "probability"]

    race_stats = race_table.copy()
    race_stats["simulated_dem_win_probability"] = district_win_probs
    race_stats["avg_simulated_margin_dem"] = avg_simulated_margin

    majority_prob = float((dem_seats >= config.majority_threshold).mean())

    summary = {
        "n_sims": config.n_sims,
        "days_out": days_out,
        "expected_dem_seats": float(dem_seats.mean()),
        "median_dem_seats": float(np.median(dem_seats)),
        "dem_majority_probability": majority_prob,
        "majority_threshold": config.majority_threshold,
        "total_error_sd": total_sd,
        "national_error_sd": national_sd,
        "district_error_sd_floor": district_sd_floor,
        "national_error_share": config.national_error_share,
        "national_environment_margin": (
            float(race_table["national_environment_margin_dem"].dropna().iloc[0])
            if "national_environment_margin_dem" in race_table.columns
            and race_table["national_environment_margin_dem"].notna().any()
            else np.nan
        ),
        "average_polling_weight": float(race_table["bayesian_polling_weight"].mean()),
        "districts_with_polling": int((race_table["bayesian_polling_weight"] > 0).sum()),
    }

    simulation_draws = pd.DataFrame({
        "simulation": np.arange(1, config.n_sims + 1),
        "dem_seats": dem_seats,
    })

    return race_stats, seat_distribution, simulation_draws, summary


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

    race_stats.to_csv(output_dir / "house_race_stats.csv", index=False)
    seat_distribution.to_csv(output_dir / "house_seat_distribution.csv", index=False)
    simulation_draws.to_csv(output_dir / "house_simulation_draws.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "house_forecast_summary.csv", index=False)

    # Also write model margin/probability back into house_race_inputs.csv for dashboard convenience.
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
    print(f"National environment:   {summary['national_environment_margin']:+.2f}")
    print(f"Districts with polling: {summary['districts_with_polling']}")
    print()
    print(f"Outputs saved to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
