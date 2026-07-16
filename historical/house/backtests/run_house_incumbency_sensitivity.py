from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RUNNER_PATH = (
    PROJECT_ROOT
    / "historical/house/backtests/"
    "run_house_layered_backtest.py"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "historical/house/backtests/outputs/layered/"
    "house_2022_layered_backtest_summary.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "historical/house/backtests/outputs/layered/"
    "house_2022_incumbency_sensitivity.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--minimum",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--maximum",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--step",
        type=float,
        default=0.25,
    )

    args = parser.parse_args()

    if args.step <= 0:
        raise ValueError("Step must be positive.")

    bonuses = []

    value = args.minimum

    while value <= args.maximum + 1e-9:
        bonuses.append(round(value, 6))
        value += args.step

    rows = []

    for bonus in bonuses:
        command = [
            "python3",
            str(RUNNER_PATH),
            "--incumbency-bonus",
            str(bonus),
        ]

        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

        if completed.returncode != 0:
            print(completed.stdout)
            print(completed.stderr)

            raise RuntimeError(
                f"Backtest failed for incumbency bonus {bonus}."
            )

        summary = pd.read_csv(SUMMARY_PATH)

        layer_2 = summary.loc[
            summary["model_name"].eq(
                "layer_2_plus_incumbency"
            )
        ]

        if len(layer_2) != 1:
            raise ValueError(
                "Could not identify exactly one Layer 2 summary row."
            )

        result = layer_2.iloc[0]

        rows.append(
            {
                "incumbency_bonus": bonus,
                "mean_absolute_error": result[
                    "mean_absolute_error"
                ],
                "median_absolute_error": result[
                    "median_absolute_error"
                ],
                "rmse": result["rmse"],
                "winner_accuracy": result[
                    "winner_accuracy"
                ],
                "brier_score": result[
                    "brier_score"
                ],
                "log_loss": result["log_loss"],
                "mean_margin_error_dem_bias": result[
                    "mean_margin_error_dem_bias"
                ],
                "predicted_dem_wins": result[
                    "predicted_dem_wins_in_scored_sample"
                ],
                "expected_dem_wins": result[
                    "expected_dem_wins_in_scored_sample"
                ],
            }
        )

        print(
            f"Bonus {bonus:>4.2f}: "
            f"MAE={float(result['mean_absolute_error']):.4f}, "
            f"RMSE={float(result['rmse']):.4f}, "
            f"Brier={float(result['brier_score']):.5f}"
        )

    results = pd.DataFrame(rows)

    results["mae_rank"] = (
        results["mean_absolute_error"]
        .rank(method="min")
        .astype(int)
    )

    results["rmse_rank"] = (
        results["rmse"]
        .rank(method="min")
        .astype(int)
    )

    results["brier_rank"] = (
        results["brier_score"]
        .rank(method="min")
        .astype(int)
    )

    results["combined_rank"] = (
        results[
            [
                "mae_rank",
                "rmse_rank",
                "brier_rank",
            ]
        ].sum(axis=1)
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("Incumbency sensitivity results")
    print("------------------------------")
    print(
        results.sort_values(
            [
                "combined_rank",
                "mean_absolute_error",
            ]
        ).to_string(index=False)
    )

    print()
    print("Best by MAE:")
    print(
        results.nsmallest(
            5,
            "mean_absolute_error",
        )[
            [
                "incumbency_bonus",
                "mean_absolute_error",
                "rmse",
                "brier_score",
                "winner_accuracy",
            ]
        ].to_string(index=False)
    )

    print()
    print("Best by RMSE:")
    print(
        results.nsmallest(
            5,
            "rmse",
        )[
            [
                "incumbency_bonus",
                "mean_absolute_error",
                "rmse",
                "brier_score",
                "winner_accuracy",
            ]
        ].to_string(index=False)
    )

    print()
    print(f"Wrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
