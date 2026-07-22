#!/usr/bin/env python3
"""
Analyze leakage-safe House forecast-margin residuals.

This script does not tune or modify the production model. It establishes the
empirical Election Day forecast-error distribution using the same historical
rows, candidate-quality handling, and model-margin builder as the House
production replay.

Residual orientation:
    residual_dem = actual_dem_margin - model_margin_dem

Positive residual:
    Democrats performed better than forecast.

Negative residual:
    Republicans performed better than forecast.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from historical.house.backtests.run_house_historical_backtest import (  # noqa: E402
    build_scoring_mask,
)
from historical.house.backtests.run_house_production_replay import (  # noqa: E402
    DEFAULT_MASTER_PATH,
    DEFAULT_WAR_PATH,
    SUPPORTED_CYCLES,
    prepare_cycle,
    validate_input,
)


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "uncertainty_residual_analysis"
)


class ResidualAnalysisError(RuntimeError):
    """Raised when residual-analysis validation fails."""


def safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan

    return result if np.isfinite(result) else math.nan


def summarize_residuals(
    df: pd.DataFrame,
    group_label: str,
    group_value: str,
) -> dict[str, Any]:
    residual = pd.to_numeric(
        df["forecast_residual_dem"],
        errors="coerce",
    ).dropna()

    absolute_error = residual.abs()
    squared_error = residual.pow(2)

    if residual.empty:
        return {
            "group_label": group_label,
            "group_value": group_value,
            "races": 0,
            "mean_error_dem": math.nan,
            "median_error_dem": math.nan,
            "mae": math.nan,
            "rmse": math.nan,
            "residual_sd": math.nan,
            "residual_skew": math.nan,
            "residual_excess_kurtosis": math.nan,
            "p05": math.nan,
            "p10": math.nan,
            "p25": math.nan,
            "p50": math.nan,
            "p75": math.nan,
            "p90": math.nan,
            "p95": math.nan,
            "within_3_points": math.nan,
            "within_5_points": math.nan,
            "within_7_5_points": math.nan,
            "within_10_points": math.nan,
            "dem_outperformed_share": math.nan,
        }

    return {
        "group_label": group_label,
        "group_value": group_value,
        "races": int(len(residual)),
        "mean_error_dem": float(residual.mean()),
        "median_error_dem": float(residual.median()),
        "mae": float(absolute_error.mean()),
        "rmse": float(np.sqrt(squared_error.mean())),
        "residual_sd": (
            float(residual.std(ddof=1))
            if len(residual) > 1
            else math.nan
        ),
        "residual_skew": (
            float(skew(residual, bias=False))
            if len(residual) > 2
            else math.nan
        ),
        "residual_excess_kurtosis": (
            float(kurtosis(residual, fisher=True, bias=False))
            if len(residual) > 3
            else math.nan
        ),
        "p05": float(residual.quantile(0.05)),
        "p10": float(residual.quantile(0.10)),
        "p25": float(residual.quantile(0.25)),
        "p50": float(residual.quantile(0.50)),
        "p75": float(residual.quantile(0.75)),
        "p90": float(residual.quantile(0.90)),
        "p95": float(residual.quantile(0.95)),
        "within_3_points": float(absolute_error.le(3.0).mean()),
        "within_5_points": float(absolute_error.le(5.0).mean()),
        "within_7_5_points": float(absolute_error.le(7.5).mean()),
        "within_10_points": float(absolute_error.le(10.0).mean()),
        "dem_outperformed_share": float(residual.gt(0.0).mean()),
    }


def add_analysis_groups(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    predicted_margin = pd.to_numeric(
        result["model_margin_dem"],
        errors="coerce",
    )

    result["predicted_abs_margin"] = predicted_margin.abs()

    result["predicted_competitiveness"] = pd.cut(
        result["predicted_abs_margin"],
        bins=[
            -np.inf,
            2.5,
            5.0,
            10.0,
            20.0,
            np.inf,
        ],
        labels=[
            "Toss-up: ≤2.5",
            "Lean: 2.5–5",
            "Likely: 5–10",
            "Solid: 10–20",
            "Safe: >20",
        ],
        right=True,
        include_lowest=True,
    )

    result["predicted_winner"] = np.where(
        predicted_margin > 0.0,
        "D",
        np.where(predicted_margin < 0.0, "R", "TIE"),
    )

    result["actual_winner_from_margin"] = np.where(
        pd.to_numeric(
            result["actual_dem_margin"],
            errors="coerce",
        )
        > 0.0,
        "D",
        "R",
    )

    result["winner_correct"] = (
        result["predicted_winner"]
        == result["actual_winner_from_margin"]
    )

    if "open_seat" in result.columns:
        open_seat = result["open_seat"]

        if open_seat.dtype == bool:
            result["seat_type"] = np.where(
                open_seat,
                "Open",
                "Incumbent-running",
            )
        else:
            normalized = (
                open_seat
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
            )

            result["seat_type"] = np.where(
                normalized.isin(
                    {
                        "true",
                        "1",
                        "yes",
                        "y",
                        "open",
                    }
                ),
                "Open",
                "Incumbent-running",
            )
    else:
        result["seat_type"] = "Unknown"

    return result


def build_group_summaries(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    rows.append(
        summarize_residuals(
            df,
            group_label="overall",
            group_value="all_cycles",
        )
    )

    for cycle, group in df.groupby("cycle", sort=True):
        rows.append(
            summarize_residuals(
                group,
                group_label="cycle",
                group_value=str(int(cycle)),
            )
        )

    for bucket, group in df.groupby(
        "predicted_competitiveness",
        observed=False,
        sort=False,
    ):
        if group.empty:
            continue

        rows.append(
            summarize_residuals(
                group,
                group_label="predicted_competitiveness",
                group_value=str(bucket),
            )
        )

    for winner, group in df.groupby(
        "predicted_winner",
        sort=True,
    ):
        rows.append(
            summarize_residuals(
                group,
                group_label="predicted_winner",
                group_value=str(winner),
            )
        )

    for seat_type, group in df.groupby(
        "seat_type",
        sort=True,
    ):
        rows.append(
            summarize_residuals(
                group,
                group_label="seat_type",
                group_value=str(seat_type),
            )
        )

    return pd.DataFrame(rows)


def build_cycle_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for cycle, group in df.groupby("cycle", sort=True):
        forecast_residual = pd.to_numeric(
            group["forecast_residual_dem"],
            errors="coerce",
        ).dropna()

        rows.append(
            {
                "cycle": int(cycle),
                "scored_races": int(len(group)),
                "mean_model_margin_dem": float(
                    pd.to_numeric(
                        group["model_margin_dem"],
                        errors="coerce",
                    ).mean()
                ),
                "mean_actual_margin_dem": float(
                    pd.to_numeric(
                        group["actual_dem_margin"],
                        errors="coerce",
                    ).mean()
                ),
                "mean_residual_dem": float(
                    forecast_residual.mean()
                ),
                "mae": float(
                    forecast_residual.abs().mean()
                ),
                "rmse": float(
                    np.sqrt(
                        forecast_residual.pow(2).mean()
                    )
                ),
                "residual_sd": float(
                    forecast_residual.std(ddof=1)
                ),
                "winner_accuracy": float(
                    group["winner_correct"].mean()
                ),
                "actual_dem_seats_scored": int(
                    group[
                        "actual_winner_from_margin"
                    ].eq("D").sum()
                ),
                "predicted_dem_seats_scored": int(
                    group["predicted_winner"].eq("D").sum()
                ),
                "predicted_minus_actual_dem_seats": int(
                    group["predicted_winner"].eq("D").sum()
                    - group[
                        "actual_winner_from_margin"
                    ].eq("D").sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze historical House model-margin residuals "
            "using the production replay population."
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
            "Candidate WAR multiplier. The uncertainty-only "
            "baseline remains 0.0."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.master_path.exists():
        raise FileNotFoundError(
            f"Historical master not found: {args.master_path}"
        )

    master = pd.read_csv(
        args.master_path,
        low_memory=False,
    )

    validate_input(master)

    residual_frames: list[pd.DataFrame] = []
    forecast_sources: dict[str, str] = {}

    for cycle in SUPPORTED_CYCLES:
        df, model_margin, forecast_source = prepare_cycle(
            master=master,
            cycle=cycle,
            candidate_quality_weight=(
                args.candidate_quality_weight
            ),
            candidate_war_path=args.candidate_war_path,
        )

        scoring_mask = build_scoring_mask(df)

        actual_margin = pd.to_numeric(
            df["actual_dem_margin"],
            errors="coerce",
        )

        valid = (
            scoring_mask
            & actual_margin.notna()
            & model_margin.notna()
        )

        scored = df.loc[valid].copy()
        scored["model_margin_dem"] = (
            model_margin.loc[valid].astype(float)
        )
        scored["actual_dem_margin"] = (
            actual_margin.loc[valid].astype(float)
        )
        scored["forecast_residual_dem"] = (
            scored["actual_dem_margin"]
            - scored["model_margin_dem"]
        )
        scored["absolute_error"] = (
            scored["forecast_residual_dem"].abs()
        )
        scored["squared_error"] = (
            scored["forecast_residual_dem"].pow(2)
        )
        scored["forecast_source"] = forecast_source

        scored = add_analysis_groups(scored)

        if scored.empty:
            raise ResidualAnalysisError(
                f"Cycle {cycle} contains no scorable rows."
            )

        residual_frames.append(scored)
        forecast_sources[str(cycle)] = forecast_source

        cycle_rmse = np.sqrt(
            scored["squared_error"].mean()
        )

        print()
        print("=" * 72)
        print(f"House residual analysis: {cycle}")
        print("=" * 72)
        print(f"Scored races: {len(scored)}")
        print(
            "Mean residual (D): "
            f"{scored['forecast_residual_dem'].mean():.4f}"
        )
        print(
            f"MAE: {scored['absolute_error'].mean():.4f}"
        )
        print(f"RMSE: {cycle_rmse:.4f}")
        print(
            "Residual SD: "
            f"{scored['forecast_residual_dem'].std(ddof=1):.4f}"
        )
        print(
            "Winner accuracy: "
            f"{scored['winner_correct'].mean():.4%}"
        )

    residuals = pd.concat(
        residual_frames,
        ignore_index=True,
    )

    expected_rows = {
        2016: 345,
        2018: 393,
        2020: 407,
        2022: 400,
    }

    actual_rows = (
        residuals.groupby("cycle")
        .size()
        .astype(int)
        .to_dict()
    )

    if actual_rows != expected_rows:
        raise ResidualAnalysisError(
            "Canonical scoring population changed. "
            f"Expected {expected_rows}; found {actual_rows}."
        )

    if residuals["forecast_residual_dem"].isna().any():
        raise ResidualAnalysisError(
            "Residual output contains missing values."
        )

    if not np.allclose(
        residuals["forecast_residual_dem"],
        (
            residuals["actual_dem_margin"]
            - residuals["model_margin_dem"]
        ),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ResidualAnalysisError(
            "Residual arithmetic validation failed."
        )

    summary = build_group_summaries(residuals)
    cycle_summary = build_cycle_summary(residuals)

    overall = summary.loc[
        summary["group_label"].eq("overall")
    ].iloc[0]

    config = {
        "analysis_version": (
            "house_uncertainty_residual_analysis_v1"
        ),
        "master_path": str(args.master_path),
        "candidate_war_path": str(
            args.candidate_war_path
        ),
        "candidate_quality_weight": float(
            args.candidate_quality_weight
        ),
        "cycles": list(SUPPORTED_CYCLES),
        "forecast_sources": forecast_sources,
        "scored_rows_by_cycle": {
            str(key): int(value)
            for key, value in actual_rows.items()
        },
        "total_scored_rows": int(len(residuals)),
        "residual_definition": (
            "actual_dem_margin - model_margin_dem"
        ),
    }

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    residuals_path = (
        args.output_dir
        / "house_uncertainty_residuals.csv"
    )
    summary_path = (
        args.output_dir
        / "house_uncertainty_residual_summary.csv"
    )
    cycle_summary_path = (
        args.output_dir
        / "house_uncertainty_residual_cycle_summary.csv"
    )
    config_path = (
        args.output_dir
        / "house_uncertainty_residual_config.json"
    )
    validation_path = (
        args.output_dir
        / "house_uncertainty_residual_validation.txt"
    )

    residuals.to_csv(
        residuals_path,
        index=False,
    )
    summary.to_csv(
        summary_path,
        index=False,
    )
    cycle_summary.to_csv(
        cycle_summary_path,
        index=False,
    )
    config_path.write_text(
        json.dumps(
            config,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    validation_lines = [
        "House Uncertainty Residual Analysis v1",
        "=" * 46,
        "",
        (
            "PASS: canonical scoring rows equal "
            f"{expected_rows}"
        ),
        (
            "PASS: total scored rows equal "
            f"{len(residuals)}"
        ),
        "PASS: no missing forecast residuals",
        "PASS: residual arithmetic reproduced exactly",
        (
            "PASS: residual orientation is "
            "actual_dem_margin - model_margin_dem"
        ),
        "",
        "VALIDATION PASSED",
        "",
    ]

    validation_path.write_text(
        "\n".join(validation_lines)
    )

    print()
    print("=" * 72)
    print("Overall historical forecast residuals")
    print("=" * 72)
    print(f"Scored races: {int(overall['races'])}")
    print(
        "Mean error (D): "
        f"{safe_float(overall['mean_error_dem']):.4f}"
    )
    print(
        "Median error (D): "
        f"{safe_float(overall['median_error_dem']):.4f}"
    )
    print(f"MAE: {safe_float(overall['mae']):.4f}")
    print(f"RMSE: {safe_float(overall['rmse']):.4f}")
    print(
        "Residual SD: "
        f"{safe_float(overall['residual_sd']):.4f}"
    )
    print(
        "Skew: "
        f"{safe_float(overall['residual_skew']):.4f}"
    )
    print(
        "Excess kurtosis: "
        f"{safe_float(overall['residual_excess_kurtosis']):.4f}"
    )
    print(
        "Within 5 points: "
        f"{safe_float(overall['within_5_points']):.2%}"
    )
    print(
        "Within 7.5 points: "
        f"{safe_float(overall['within_7_5_points']):.2%}"
    )
    print(
        "Within 10 points: "
        f"{safe_float(overall['within_10_points']):.2%}"
    )

    print()
    print("Cycle summary")
    print("-" * 72)
    print(
        cycle_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print()
    print("Wrote:")
    for path in [
        residuals_path,
        summary_path,
        cycle_summary_path,
        config_path,
        validation_path,
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
