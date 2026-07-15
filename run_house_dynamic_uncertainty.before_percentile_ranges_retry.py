from pathlib import Path
import argparse
from datetime import date, datetime
import numpy as np
import pandas as pd

INPUTS = Path("inputs")
OUTPUTS = Path("outputs")

RACE_INPUTS = INPUTS / "house_race_inputs.csv"
SETTINGS = INPUTS / "house_calibration_settings.csv"

RACE_STATS_OUT = OUTPUTS / "house_race_stats.csv"
SEAT_DIST_OUT = OUTPUTS / "house_seat_distribution.csv"
SEAT_DIST_ALIAS_OUT = OUTPUTS / "house_simulated_seat_distribution.csv"
SUMMARY_OUT = OUTPUTS / "house_forecast_summary.csv"
UNCERTAINTY_AUDIT_OUT = OUTPUTS / "house_uncertainty_audit.csv"

HOUSE_CONTROL_THRESHOLD = 218
DEFAULT_ELECTION_DAY = "2026-11-03"


def read_settings():
    if not SETTINGS.exists():
        return {}

    df = pd.read_csv(SETTINGS)

    if df.empty or "setting" not in df.columns or "value" not in df.columns:
        return {}

    out = {}

    for _, row in df.iterrows():
        key = str(row["setting"]).strip()
        try:
            value = float(row["value"])
        except Exception:
            continue
        out[key] = value

    return out


def setting(settings, key, default):
    return float(settings.get(key, default))


def parse_date(value):
    if isinstance(value, date):
        return value

    return datetime.strptime(str(value), "%Y-%m-%d").date()


def compute_days_out(today, election_day):
    today = parse_date(today)
    election_day = parse_date(election_day)
    return max(0, (election_day - today).days)


def interpolate_by_days_out(days_out, floor, d30, d90, d180):
    """
    Linear schedule:
      0 days: floor
      30 days: d30
      90 days: d90
      180+ days: d180
    """
    d = float(days_out)

    if d >= 180:
        return d180

    if d >= 90:
        t = (d - 90) / 90
        return d90 + t * (d180 - d90)

    if d >= 30:
        t = (d - 30) / 60
        return d30 + t * (d90 - d30)

    t = d / 30
    return floor + t * (d30 - floor)


def get_dynamic_sds(settings, days_out):
    national_sd = interpolate_by_days_out(
        days_out,
        setting(settings, "house_uncertainty_national_sd_floor", 2.25),
        setting(settings, "house_uncertainty_national_sd_30", 2.75),
        setting(settings, "house_uncertainty_national_sd_90", 3.5),
        setting(settings, "house_uncertainty_national_sd_180", 4.5),
    )

    region_sd = interpolate_by_days_out(
        days_out,
        setting(settings, "house_uncertainty_region_sd_floor", 1.00),
        setting(settings, "house_uncertainty_region_sd_30", 1.25),
        setting(settings, "house_uncertainty_region_sd_90", 1.75),
        setting(settings, "house_uncertainty_region_sd_180", 2.25),
    )

    demographic_sd = interpolate_by_days_out(
        days_out,
        setting(settings, "house_uncertainty_demographic_sd_floor", 0.75),
        setting(settings, "house_uncertainty_demographic_sd_30", 1.00),
        setting(settings, "house_uncertainty_demographic_sd_90", 1.25),
        setting(settings, "house_uncertainty_demographic_sd_180", 1.75),
    )

    district_sd = interpolate_by_days_out(
        days_out,
        setting(settings, "house_uncertainty_district_sd_floor", 2.75),
        setting(settings, "house_uncertainty_district_sd_30", 3.25),
        setting(settings, "house_uncertainty_district_sd_90", 3.75),
        setting(settings, "house_uncertainty_district_sd_180", 4.75),
    )

    return national_sd, region_sd, demographic_sd, district_sd


def infer_rating(prob):
    p = float(prob)

    if p >= 0.95:
        return "Safe D"
    if p >= 0.85:
        return "Likely D"
    if p >= 0.65:
        return "Lean D"
    if p >= 0.55:
        return "Tilt D"
    if p > 0.45:
        return "Toss-Up"
    if p > 0.35:
        return "Tilt R"
    if p > 0.15:
        return "Lean R"
    if p > 0.05:
        return "Likely R"
    return "Safe R"


def fmt_margin(x):
    try:
        x = float(x)
    except Exception:
        return ""

    if pd.isna(x):
        return ""
    if x > 0:
        return f"D+{x:.1f}"
    if x < 0:
        return f"R+{abs(x):.1f}"
    return "Even"


def normalize_fixed_control(df):
    out = df.copy()

    if "party_control_fixed" not in out.columns:
        out["party_control_fixed"] = ""

    out["party_control_fixed"] = out["party_control_fixed"].fillna("").astype(str).str.strip().str.upper()

    if "general_election_party_structure" in out.columns:
        structure = out["general_election_party_structure"].fillna("").astype(str).str.strip()

        out.loc[
            structure.isin(["D_vs_D", "D_unopposed"]),
            "party_control_fixed",
        ] = "D"

        out.loc[
            structure.isin(["R_vs_R", "R_unopposed"]),
            "party_control_fixed",
        ] = "R"

    if "party_control_override" in out.columns:
        override = out["party_control_override"].fillna("").astype(str).str.strip().str.upper()
        out.loc[override.isin(["D", "R"]), "party_control_fixed"] = override[override.isin(["D", "R"])]

    return out


def choose_model_margin(df):
    out = df.copy()

    candidates = [
        "bayesian_model_margin_dem",
        "model_margin_dem",
        "fundamentals_margin_dem",
        "district_partisan_baseline_dem",
    ]

    margin = pd.Series(np.nan, index=out.index, dtype=float)

    for col in candidates:
        if col in out.columns:
            margin = margin.combine_first(pd.to_numeric(out[col], errors="coerce"))

    out["model_margin_dem"] = margin.fillna(0.0)

    return out


def polling_district_multiplier(poll_count, settings):
    try:
        n = int(float(poll_count))
    except Exception:
        n = 0

    if n >= 3:
        return setting(settings, "house_polling_district_error_multiplier_3plus_polls", 0.70)
    if n == 2:
        return setting(settings, "house_polling_district_error_multiplier_2_polls", 0.80)
    if n == 1:
        return setting(settings, "house_polling_district_error_multiplier_1_poll", 0.90)
    return 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=20000)
    parser.add_argument("--today", default=None)
    parser.add_argument("--election-day", default=DEFAULT_ELECTION_DAY)
    parser.add_argument("--seed", type=int, default=20260603)
    args = parser.parse_args()

    if not RACE_INPUTS.exists():
        raise FileNotFoundError("inputs/house_race_inputs.csv not found.")

    today = args.today or date.today().isoformat()
    days_out = compute_days_out(today, args.election_day)

    settings = read_settings()
    national_sd, region_sd, demographic_sd, district_sd = get_dynamic_sds(settings, days_out)

    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(RACE_INPUTS)
    df = normalize_fixed_control(df)
    df = choose_model_margin(df)

    if "district_id" not in df.columns:
        raise ValueError("house_race_inputs.csv must include district_id.")

    n_districts = len(df)
    n_sims = int(args.sims)

    region_col = "region_error_group" if "region_error_group" in df.columns else "region"
    if region_col not in df.columns:
        df["region"] = "National"
        region_col = "region"

    demo_col = "education_race_error_group"
    if demo_col not in df.columns:
        demo_col = "demographic_error_group" if "demographic_error_group" in df.columns else None

    if demo_col is None:
        df["demographic_error_group"] = "All Districts"
        demo_col = "demographic_error_group"

    df[region_col] = df[region_col].fillna("Unknown Region").astype(str)
    df[demo_col] = df[demo_col].fillna("Unknown Demographic Group").astype(str)

    regions = sorted(df[region_col].unique().tolist())
    demos = sorted(df[demo_col].unique().tolist())

    region_index = {g: i for i, g in enumerate(regions)}
    demo_index = {g: i for i, g in enumerate(demos)}

    region_ids = df[region_col].map(region_index).to_numpy()
    demo_ids = df[demo_col].map(demo_index).to_numpy()

    national_errors = rng.normal(0, national_sd, size=n_sims)
    region_errors = rng.normal(0, region_sd, size=(n_sims, len(regions)))
    demo_errors = rng.normal(0, demographic_sd, size=(n_sims, len(demos)))

    poll_counts = pd.to_numeric(
        df["poll_count"] if "poll_count" in df.columns else 0,
        errors="coerce",
    ).fillna(0)

    district_sd_multiplier = poll_counts.apply(lambda n: polling_district_multiplier(n, settings)).to_numpy()
    district_sds = district_sd * district_sd_multiplier

    district_errors = rng.normal(
        0,
        district_sds.reshape(1, n_districts),
        size=(n_sims, n_districts),
    )

    base_margin = pd.to_numeric(df["model_margin_dem"], errors="coerce").fillna(0.0).to_numpy()

    simulated_margins = (
        base_margin.reshape(1, n_districts)
        + national_errors.reshape(n_sims, 1)
        + region_errors[:, region_ids]
        + demo_errors[:, demo_ids]
        + district_errors
    )

    fixed = df["party_control_fixed"].fillna("").astype(str).str.upper().to_numpy()

    dem_wins = simulated_margins > 0
    dem_wins[:, fixed == "D"] = True
    dem_wins[:, fixed == "R"] = False

    dem_seats_by_sim = dem_wins.sum(axis=1)

    dem_prob = dem_wins.mean(axis=0)
    avg_sim_margin = simulated_margins.mean(axis=0)

    df["simulated_dem_win_prob"] = dem_prob
    df["dem_win_probability"] = dem_prob
    df["avg_simulated_margin_dem"] = avg_sim_margin
    df["model_margin_label"] = df["model_margin_dem"].apply(fmt_margin)
    df["rating"] = df["dem_win_probability"].apply(infer_rating)

    # Preserve fixed-control ratings.
    df.loc[fixed == "D", "dem_win_probability"] = 1.0
    df.loc[fixed == "D", "simulated_dem_win_prob"] = 1.0
    df.loc[fixed == "D", "rating"] = "Safe D"

    df.loc[fixed == "R", "dem_win_probability"] = 0.0
    df.loc[fixed == "R", "simulated_dem_win_prob"] = 0.0
    df.loc[fixed == "R", "rating"] = "Safe R"

    df["uncertainty_days_out"] = days_out
    df["national_error_sd"] = national_sd
    df["region_error_sd"] = region_sd
    df["demographic_error_sd"] = demographic_sd
    df["district_error_sd"] = district_sd
    df["effective_district_error_sd"] = district_sds

    seat_counts = pd.Series(dem_seats_by_sim).value_counts().sort_index()
    seat_dist = pd.DataFrame(
        {
            "dem_seats": seat_counts.index.astype(int),
            "count": seat_counts.values.astype(int),
        }
    )
    seat_dist["probability"] = seat_dist["count"] / n_sims

    expected_dem_seats = float(np.mean(dem_seats_by_sim))
    median_dem_seats = float(np.median(dem_seats_by_sim))
    dem_control_probability = float(np.mean(dem_seats_by_sim >= HOUSE_CONTROL_THRESHOLD))

    total_error_sd = float(
        np.sqrt(
            national_sd ** 2
            + region_sd ** 2
            + demographic_sd ** 2
            + district_sd ** 2
        )
    )

    implied_correlation = float(
        (national_sd ** 2 + region_sd ** 2 + demographic_sd ** 2)
        / max(total_error_sd ** 2, 0.000001)
    )

    summary = pd.DataFrame(
        [
            {
                "n_sims": n_sims,
                "days_out": days_out,
                "expected_dem_seats": expected_dem_seats,
                "median_dem_seats": median_dem_seats,
                "dem_control_probability": dem_control_probability,
                "dem_control_threshold": HOUSE_CONTROL_THRESHOLD,
                "national_error_sd": national_sd,
                "region_error_sd": region_sd,
                "demographic_error_sd": demographic_sd,
                "district_error_sd": district_sd,
                "total_error_sd": total_error_sd,
                "implied_correlation": implied_correlation,
                "districts_with_polling": int((poll_counts > 0).sum()),
                "region_groups": len(regions),
                "demographic_groups": len(demos),
                "uncertainty_engine": "dynamic_correlated_house_v1",
            }
        ]
    )

    uncertainty_audit = pd.DataFrame(
        [
            {
                "days_out": days_out,
                "national_error_sd": national_sd,
                "region_error_sd": region_sd,
                "demographic_error_sd": demographic_sd,
                "district_error_sd": district_sd,
                "total_error_sd": total_error_sd,
                "implied_correlation": implied_correlation,
                "polling_district_error_multiplier_1_poll": setting(settings, "house_polling_district_error_multiplier_1_poll", 0.90),
                "polling_district_error_multiplier_2_polls": setting(settings, "house_polling_district_error_multiplier_2_polls", 0.80),
                "polling_district_error_multiplier_3plus_polls": setting(settings, "house_polling_district_error_multiplier_3plus_polls", 0.70),
            }
        ]
    )

    OUTPUTS.mkdir(exist_ok=True)

    df.to_csv(RACE_STATS_OUT, index=False)
    seat_dist.to_csv(SEAT_DIST_OUT, index=False)
    seat_dist.to_csv(SEAT_DIST_ALIAS_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)
    uncertainty_audit.to_csv(UNCERTAINTY_AUDIT_OUT, index=False)

    print()
    print("Dynamic House uncertainty forecast complete")
    print("------------------------------------------")
    print(f"Days out:               {days_out}")
    print(f"Simulations:            {n_sims}")
    print(f"Expected Dem seats:     {expected_dem_seats:.2f}")
    print(f"Median Dem seats:       {median_dem_seats:.0f}")
    print(f"Dem majority odds:      {dem_control_probability:.1%}")
    print(f"National error SD:      {national_sd:.2f}")
    print(f"Region error SD:        {region_sd:.2f}")
    print(f"Demographic error SD:   {demographic_sd:.2f}")
    print(f"District error SD:      {district_sd:.2f}")
    print(f"Total approx error SD:  {total_error_sd:.2f}")
    print()
    print("Outputs saved to outputs/")


if __name__ == "__main__":
    main()
