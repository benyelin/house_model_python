import subprocess
import sys
from pathlib import Path
import pandas as pd

BACKTESTS = Path("backtests")

ENVIRONMENT_MULTIPLIERS = [0.75, 0.85, 0.95, 1.00]
TOTAL_ERROR_SDS = [6.5, 7.5, 8.5, 9.5]

summary_rows = []
rating_rows = []
state_rows = []

for env_mult in ENVIRONMENT_MULTIPLIERS:
    for error_sd in TOTAL_ERROR_SDS:
        print()
        print("=" * 84)
        print(f"Testing environment_multiplier={env_mult:.2f}, total_error_sd={error_sd:.1f}")
        print("=" * 84)

        subprocess.run(
            [
                sys.executable,
                "run_house_backtest.py",
                "--cycles",
                "2018",
                "2022",
                "--total-error-sd",
                str(error_sd),
                "--environment-multiplier",
                str(env_mult),
            ],
            check=True,
        )

        summary = pd.read_csv(BACKTESTS / "house_backtest_summary.csv")
        summary["tested_environment_multiplier"] = env_mult
        summary["tested_total_error_sd"] = error_sd
        summary_rows.append(summary)

        rating_path = BACKTESTS / "house_backtest_rating_summary.csv"
        if rating_path.exists():
            rating = pd.read_csv(rating_path)
            rating["tested_environment_multiplier"] = env_mult
            rating["tested_total_error_sd"] = error_sd
            rating_rows.append(rating)

        state_path = BACKTESTS / "house_backtest_state_summary.csv"
        if state_path.exists():
            state = pd.read_csv(state_path)
            state["tested_environment_multiplier"] = env_mult
            state["tested_total_error_sd"] = error_sd
            state_rows.append(state)

comparison = pd.concat(summary_rows, ignore_index=True)
comparison.to_csv(BACKTESTS / "house_backtest_env_uncertainty_grid.csv", index=False)

if rating_rows:
    rating_comparison = pd.concat(rating_rows, ignore_index=True)
    rating_comparison.to_csv(
        BACKTESTS / "house_backtest_env_uncertainty_rating_grid.csv",
        index=False,
    )

if state_rows:
    state_comparison = pd.concat(state_rows, ignore_index=True)
    state_comparison.to_csv(
        BACKTESTS / "house_backtest_env_uncertainty_state_grid.csv",
        index=False,
    )

display_cols = [
    "tested_environment_multiplier",
    "tested_total_error_sd",
    "cycle",
    "districts",
    "winner_accuracy",
    "mean_abs_margin_error",
    "mean_margin_error_dem_bias",
    "brier_score",
    "actual_dem_seats",
    "predicted_dem_seats",
    "expected_dem_seats",
]

print()
print("Environment / uncertainty grid")
print("------------------------------")
print(comparison[display_cols].to_string(index=False))

print()
print("Best by cycle, lowest Brier score")
print("---------------------------------")
best_by_cycle = (
    comparison.sort_values("brier_score")
    .groupby("cycle", as_index=False)
    .head(5)
)
print(best_by_cycle[display_cols].to_string(index=False))

print()
print("Best combined average Brier score across 2018 and 2022")
print("------------------------------------------------------")
combined = (
    comparison
    .groupby(["tested_environment_multiplier", "tested_total_error_sd"], as_index=False)
    .agg(
        avg_brier_score=("brier_score", "mean"),
        avg_abs_margin_error=("mean_abs_margin_error", "mean"),
        avg_winner_accuracy=("winner_accuracy", "mean"),
        avg_dem_bias=("mean_margin_error_dem_bias", "mean"),
    )
    .sort_values("avg_brier_score")
)
print(combined.to_string(index=False))
