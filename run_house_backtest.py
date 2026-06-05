from pathlib import Path
import argparse
import numpy as np
import pandas as pd

BACKTESTS = Path("backtests")
OUTPUTS = Path("outputs")

DEFAULT_SIMS = 20000


def fmt_margin(x):
    if pd.isna(x):
        return ""
    x = float(x)
    if x > 0:
        return f"D+{x:.1f}"
    if x < 0:
        return f"R+{abs(x):.1f}"
    return "Even"


def rating_from_prob(p):
    if pd.isna(p):
        return "Unknown"
    p = float(p)
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


def win_prob_from_margin(margin, total_error_sd):
    # Logistic approximation. Keeps backtest simple and comparable.
    # Scale chosen so larger uncertainty produces flatter probabilities.
    scale = max(float(total_error_sd), 1.0)
    return 1 / (1 + np.exp(-float(margin) / scale))


def load_cycle(cycle):
    path = BACKTESTS / f"house_{cycle}_inputs.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Create this input file before running the backtest."
        )

    df = pd.read_csv(path)

    required = [
        "district_id",
        "actual_dem_margin",
        "district_pres_margin_dem",
        "national_environment_margin_dem",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    return df


def run_cycle(cycle, total_error_sd, environment_multiplier=1.0):
    df = load_cycle(cycle).copy()

    numeric_defaults = {
        "district_pres_margin_dem": 0.0,
        "national_environment_margin_dem": 0.0,
        "district_elasticity": 1.0,
        "incumbency_adjustment_dem": 0.0,
        "candidate_quality_adjustment_dem": 0.0,
        "special_adjustment_dem": 0.0,
        "actual_dem_margin": 0.0,
    }

    for col, default in numeric_defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    df["environment_multiplier"] = float(environment_multiplier)

    df["district_environment_adjustment_dem"] = (
        df["national_environment_margin_dem"]
        * df["environment_multiplier"]
        * df["district_elasticity"]
    )

    df["model_margin_dem"] = (
        df["district_pres_margin_dem"]
        + df["district_environment_adjustment_dem"]
        + df["incumbency_adjustment_dem"]
        + df["candidate_quality_adjustment_dem"]
        + df["special_adjustment_dem"]
    )

    df["dem_win_probability"] = df["model_margin_dem"].apply(
        lambda m: win_prob_from_margin(m, total_error_sd)
    )

    df["predicted_dem_win"] = df["dem_win_probability"] >= 0.5
    df["actual_dem_win"] = df["actual_dem_margin"] > 0
    df["correct_winner"] = df["predicted_dem_win"] == df["actual_dem_win"]

    df["margin_error"] = df["model_margin_dem"] - df["actual_dem_margin"]
    df["abs_margin_error"] = df["margin_error"].abs()
    df["brier_score"] = (
        df["dem_win_probability"] - df["actual_dem_win"].astype(float)
    ) ** 2

    df["model_margin_label"] = df["model_margin_dem"].apply(fmt_margin)
    df["actual_margin_label"] = df["actual_dem_margin"].apply(fmt_margin)
    df["rating"] = df["dem_win_probability"].apply(rating_from_prob)
    df["cycle"] = cycle

    return df


def summarize(results):
    rows = []

    for cycle, g in results.groupby("cycle"):
        rows.append(
            {
                "cycle": cycle,
                "environment_multiplier": g["environment_multiplier"].iloc[0] if "environment_multiplier" in g.columns else np.nan,
                "districts": len(g),
                "winner_accuracy": g["correct_winner"].mean(),
                "mean_abs_margin_error": g["abs_margin_error"].mean(),
                "median_abs_margin_error": g["abs_margin_error"].median(),
                "mean_margin_error_dem_bias": g["margin_error"].mean(),
                "brier_score": g["brier_score"].mean(),
                "actual_dem_seats": int(g["actual_dem_win"].sum()),
                "predicted_dem_seats": int(g["predicted_dem_win"].sum()),
                "expected_dem_seats": g["dem_win_probability"].sum(),
            }
        )

    return pd.DataFrame(rows)


def summarize_by_rating(results):
    rows = []

    for (cycle, rating), g in results.groupby(["cycle", "rating"]):
        rows.append(
            {
                "cycle": cycle,
                "rating": rating,
                "districts": len(g),
                "avg_dem_probability": g["dem_win_probability"].mean(),
                "actual_dem_win_rate": g["actual_dem_win"].mean(),
                "winner_accuracy": g["correct_winner"].mean(),
                "mean_margin_error_dem_bias": g["margin_error"].mean(),
                "mean_abs_margin_error": g["abs_margin_error"].mean(),
                "brier_score": g["brier_score"].mean(),
            }
        )

    return pd.DataFrame(rows)


def summarize_by_state(results):
    rows = []

    temp = results.copy()
    temp["state"] = temp["district_id"].astype(str).str.extract(r"^([A-Z]{2})")

    for (cycle, state), g in temp.groupby(["cycle", "state"]):
        rows.append(
            {
                "cycle": cycle,
                "state": state,
                "districts": len(g),
                "winner_accuracy": g["correct_winner"].mean(),
                "mean_margin_error_dem_bias": g["margin_error"].mean(),
                "mean_abs_margin_error": g["abs_margin_error"].mean(),
                "actual_dem_wins": int(g["actual_dem_win"].sum()),
                "predicted_dem_wins": int(g["predicted_dem_win"].sum()),
                "expected_dem_wins": g["dem_win_probability"].sum(),
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", nargs="+", default=["2018", "2022"])
    parser.add_argument("--total-error-sd", type=float, default=6.5)
    parser.add_argument("--environment-multiplier", type=float, default=1.0)
    args = parser.parse_args()

    all_results = []

    for cycle in args.cycles:
        result = run_cycle(cycle, args.total_error_sd, args.environment_multiplier)
        all_results.append(result)

    results = pd.concat(all_results, ignore_index=True)
    summary = summarize(results)

    BACKTESTS.mkdir(exist_ok=True)

    rating_summary = summarize_by_rating(results)
    state_summary = summarize_by_state(results)

    results.to_csv(BACKTESTS / "house_backtest_results.csv", index=False)
    summary.to_csv(BACKTESTS / "house_backtest_summary.csv", index=False)
    rating_summary.to_csv(BACKTESTS / "house_backtest_rating_summary.csv", index=False)
    state_summary.to_csv(BACKTESTS / "house_backtest_state_summary.csv", index=False)

    print()
    print("House backtest summary")
    print("----------------------")
    print(summary.to_string(index=False))

    print()
    print("Rating calibration")
    print("------------------")
    print(rating_summary.to_string(index=False))

    print()
    print("State/cycle bias")
    print("----------------")
    print(state_summary.sort_values(["cycle", "mean_margin_error_dem_bias"]).to_string(index=False))

    print()
    print("Worst margin misses")
    print("-------------------")
    show_cols = [
        "cycle",
        "district_id",
        "model_margin_label",
        "actual_margin_label",
        "margin_error",
        "dem_win_probability",
        "rating",
        "correct_winner",
    ]
    print(
        results.sort_values("abs_margin_error", ascending=False)
        .head(25)[show_cols]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
