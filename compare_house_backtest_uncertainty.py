import subprocess
import sys
import pandas as pd
from pathlib import Path

BACKTESTS = Path("backtests")
VALUES = [5.5, 6.5, 7.5, 8.5, 9.5]

rows = []

for sd in VALUES:
    print()
    print("=" * 72)
    print(f"Testing total_error_sd = {sd}")
    print("=" * 72)

    subprocess.run(
        [sys.executable, "run_house_backtest.py", "--cycles", "2018", "2022", "--total-error-sd", str(sd)],
        check=True,
    )

    summary = pd.read_csv(BACKTESTS / "house_backtest_summary.csv")
    summary["tested_total_error_sd"] = sd
    rows.append(summary)

comparison = pd.concat(rows, ignore_index=True)

comparison.to_csv(BACKTESTS / "house_backtest_uncertainty_comparison.csv", index=False)

display_cols = [
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
print("Uncertainty comparison")
print("----------------------")
print(comparison[display_cols].to_string(index=False))
