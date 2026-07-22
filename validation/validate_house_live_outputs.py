#!/usr/bin/env python3

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_SEATS_TOLERANCE = 1e-9
PROBABILITY_TOLERANCE = 1e-12
DEFAULT_TIMESTAMP_TOLERANCE_SECONDS = 5.0


def require_columns(
    frame: pd.DataFrame,
    required: list[str],
    source: Path,
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{source} is missing required columns: {missing}"
        )


def numeric_column(
    frame: pd.DataFrame,
    column: str,
    source: Path,
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")

    if values.isna().any():
        bad_count = int(values.isna().sum())
        raise ValueError(
            f"{source}: column {column!r} contains "
            f"{bad_count} nonnumeric or missing values."
        )

    return values.astype(float)


def validate_outputs(
    output_dir: Path,
    majority_threshold: int,
    timestamp_tolerance_seconds: float,
) -> None:
    paths = {
        "race stats": output_dir / "house_race_stats.csv",
        "seat distribution": output_dir / "house_seat_distribution.csv",
        "simulation draws": output_dir / "house_simulation_draws.csv",
        "forecast summary": output_dir / "house_forecast_summary.csv",
    }

    missing = [
        str(path)
        for path in paths.values()
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required House output files:\n  - "
            + "\n  - ".join(missing)
        )

    race = pd.read_csv(paths["race stats"])
    distribution = pd.read_csv(paths["seat distribution"])
    draws = pd.read_csv(paths["simulation draws"])
    summary = pd.read_csv(paths["forecast summary"])

    if summary.empty:
        raise ValueError(
            f"{paths['forecast summary']} contains no rows."
        )

    if len(summary) != 1:
        raise ValueError(
            f"{paths['forecast summary']} should contain exactly one row; "
            f"found {len(summary)}."
        )

    require_columns(
        race,
        ["simulated_dem_win_probability"],
        paths["race stats"],
    )
    require_columns(
        distribution,
        ["dem_seats", "probability"],
        paths["seat distribution"],
    )
    require_columns(
        draws,
        ["dem_seats"],
        paths["simulation draws"],
    )
    require_columns(
        summary,
        [
            "expected_dem_seats",
            "dem_majority_probability",
        ],
        paths["forecast summary"],
    )

    race_probabilities = numeric_column(
        race,
        "simulated_dem_win_probability",
        paths["race stats"],
    )
    distribution_seats = numeric_column(
        distribution,
        "dem_seats",
        paths["seat distribution"],
    )
    distribution_probabilities = numeric_column(
        distribution,
        "probability",
        paths["seat distribution"],
    )
    draw_seats = numeric_column(
        draws,
        "dem_seats",
        paths["simulation draws"],
    )

    summary_row = summary.iloc[0]
    summary_expected = float(summary_row["expected_dem_seats"])
    summary_control = float(
        summary_row["dem_majority_probability"]
    )

    district_expected = float(race_probabilities.sum())
    draw_expected = float(draw_seats.mean())
    distribution_expected = float(
        (
            distribution_seats
            * distribution_probabilities
        ).sum()
    )

    draw_control = float(
        (draw_seats >= majority_threshold).mean()
    )
    distribution_control = float(
        distribution_probabilities[
            distribution_seats >= majority_threshold
        ].sum()
    )

    distribution_probability_sum = float(
        distribution_probabilities.sum()
    )

    expected_values = {
        "district probability sum": district_expected,
        "forecast summary": summary_expected,
        "simulation draw mean": draw_expected,
        "seat-distribution mean": distribution_expected,
    }

    control_values = {
        "forecast summary": summary_control,
        "simulation draws": draw_control,
        "seat distribution": distribution_control,
    }

    failures: list[str] = []

    for label, value in expected_values.items():
        if not np.isclose(
            value,
            summary_expected,
            atol=EXPECTED_SEATS_TOLERANCE,
            rtol=0.0,
        ):
            failures.append(
                f"Expected-seat mismatch: {label}={value:.12f}, "
                f"summary={summary_expected:.12f}"
            )

    for label, value in control_values.items():
        if not np.isclose(
            value,
            summary_control,
            atol=PROBABILITY_TOLERANCE,
            rtol=0.0,
        ):
            failures.append(
                f"Control-probability mismatch: {label}={value:.12f}, "
                f"summary={summary_control:.12f}"
            )

    if not np.isclose(
        distribution_probability_sum,
        1.0,
        atol=PROBABILITY_TOLERANCE,
        rtol=0.0,
    ):
        failures.append(
            "Seat-distribution probabilities do not sum to 1: "
            f"{distribution_probability_sum:.12f}"
        )

    modification_times = {
        label: path.stat().st_mtime
        for label, path in paths.items()
    }
    timestamp_spread = (
        max(modification_times.values())
        - min(modification_times.values())
    )

    if timestamp_spread > timestamp_tolerance_seconds:
        failures.append(
            "Output timestamps span "
            f"{timestamp_spread:.3f} seconds, exceeding the "
            f"{timestamp_tolerance_seconds:.3f}-second tolerance."
        )

    print("HOUSE LIVE OUTPUT VALIDATION")
    print("=" * 64)

    for label, path in paths.items():
        stat = path.stat()
        timestamp = datetime.fromtimestamp(stat.st_mtime)
        print(
            f"{label:<20} "
            f"{timestamp:%Y-%m-%d %H:%M:%S}."
            f"{stat.st_mtime_ns % 1_000_000_000:09d}"
        )

    print()
    print("Expected Democratic seats")
    print("-" * 64)
    for label, value in expected_values.items():
        print(f"{label:<28} {value:.9f}")

    print()
    print("Democratic control probability")
    print("-" * 64)
    for label, value in control_values.items():
        print(f"{label:<28} {value:.9%}")

    print()
    print(
        "Seat-distribution probability sum: "
        f"{distribution_probability_sum:.12f}"
    )
    print(
        "Output timestamp spread:            "
        f"{timestamp_spread:.6f} seconds"
    )

    if failures:
        print()
        print("VALIDATION FAILED")
        print("-" * 64)
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print()
    print("VALIDATION PASSED")
    print(
        "All House outputs describe the same simulation run."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate consistency across live House forecast outputs."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
    )
    parser.add_argument(
        "--majority-threshold",
        type=int,
        default=218,
    )
    parser.add_argument(
        "--timestamp-tolerance-seconds",
        type=float,
        default=DEFAULT_TIMESTAMP_TOLERANCE_SECONDS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_outputs(
        output_dir=args.output_dir,
        majority_threshold=args.majority_threshold,
        timestamp_tolerance_seconds=(
            args.timestamp_tolerance_seconds
        ),
    )


if __name__ == "__main__":
    main()
