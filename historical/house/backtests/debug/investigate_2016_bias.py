#!/usr/bin/env python3
"""
Investigate the House production replay's 2016 Democratic overprediction.

This diagnostic examines:

- Overall margin bias
- Expected-versus-actual Democratic seats
- Incumbency groups
- Competitive-seat performance
- State-level bias
- Largest individual errors
- Available model-component columns

It does not modify any model files or outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_REPLAY_PATHS = [
    REPOSITORY_ROOT
    / "historical/house/backtests/outputs/production_replay_v1"
    / "house_production_replay_predictions.csv",
    REPOSITORY_ROOT
    / "historical/house/backtests/outputs/production_replay"
    / "house_production_replay_predictions.csv",
]

DEFAULT_CYCLE = 2016
DEFAULT_SPEC = "production_election_day_v1"


class BiasInvestigationError(RuntimeError):
    """Raised when the diagnostic cannot safely continue."""


def find_default_replay_path() -> Path:
    for path in DEFAULT_REPLAY_PATHS:
        if path.exists():
            return path

    searched = "\n".join(
        f"  - {path}"
        for path in DEFAULT_REPLAY_PATHS
    )

    raise BiasInvestigationError(
        "Could not find the production replay predictions file.\n"
        "Searched:\n"
        f"{searched}\n\n"
        "Supply the file explicitly with --input."
    )


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> None:
    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]

    if missing:
        raise BiasInvestigationError(
            "Replay file is missing required columns:\n  - "
            + "\n  - ".join(missing)
        )


def normalize_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    return normalized.isin(
        {
            "true",
            "1",
            "yes",
            "y",
            "t",
        }
    )


def first_available(
    frame: pd.DataFrame,
    candidates: Iterable[str],
) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def print_heading(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def print_subheading(title: str) -> None:
    print()
    print("-" * 88)
    print(title)
    print("-" * 88)


def safe_group_summary(
    frame: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    return (
        frame.groupby(
            group_column,
            dropna=False,
        )
        .agg(
            races=("race_id", "size"),
            predicted_dem_wins=("predicted_dem_win", "sum"),
            actual_dem_wins=("actual_dem_win", "sum"),
            seat_error=("seat_error", "sum"),
            mean_model_margin=("model_margin_dem", "mean"),
            mean_actual_margin=("actual_dem_margin", "mean"),
            mean_margin_error=("margin_error", "mean"),
            median_margin_error=("margin_error", "median"),
            mean_absolute_error=("abs_margin_error", "mean"),
            rmse=(
                "squared_margin_error",
                lambda values: float(
                    np.sqrt(
                        np.mean(values)
                    )
                ),
            ),
        )
        .sort_values(
            [
                "mean_margin_error",
                "races",
            ],
            ascending=[
                False,
                False,
            ],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Investigate systematic bias in a House production replay cycle."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Path to house_production_replay_predictions.csv. "
            "If omitted, the standard output locations are searched."
        ),
    )

    parser.add_argument(
        "--cycle",
        type=int,
        default=DEFAULT_CYCLE,
        help=f"Election cycle to inspect. Default: {DEFAULT_CYCLE}.",
    )

    parser.add_argument(
        "--spec",
        default=DEFAULT_SPEC,
        help=(
            "Replay specification to inspect. "
            f"Default: {DEFAULT_SPEC}."
        ),
    )

    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Number of largest errors to display. Default: 30.",
    )

    args = parser.parse_args()

    replay_path = (
        args.input.expanduser().resolve()
        if args.input is not None
        else find_default_replay_path()
    )

    if not replay_path.exists():
        raise BiasInvestigationError(
            f"Replay predictions file does not exist: {replay_path}"
        )

    frame = pd.read_csv(
        replay_path,
        low_memory=False,
    )

    require_columns(
        frame,
        [
            "cycle",
            "race_id",
            "model_margin_dem",
            "actual_dem_margin",
        ],
    )

    cycle_frame = frame.loc[
        frame["cycle"].eq(args.cycle)
    ].copy()

    if cycle_frame.empty:
        available_cycles = sorted(
            pd.to_numeric(
                frame["cycle"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        raise BiasInvestigationError(
            f"No rows found for cycle {args.cycle}. "
            f"Available cycles: {available_cycles}"
        )

    if "replay_spec" in cycle_frame.columns:
        available_specs = sorted(
            cycle_frame["replay_spec"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        cycle_frame = cycle_frame.loc[
            cycle_frame["replay_spec"].eq(args.spec)
        ].copy()

        if cycle_frame.empty:
            raise BiasInvestigationError(
                f"No rows found for cycle {args.cycle} "
                f"and replay specification {args.spec!r}. "
                f"Available specifications: {available_specs}"
            )

    if "include_in_scoring" in cycle_frame.columns:
        scoring_mask = normalize_boolean(
            cycle_frame["include_in_scoring"]
        )
        cycle_frame = cycle_frame.loc[
            scoring_mask
        ].copy()

    if cycle_frame.empty:
        raise BiasInvestigationError(
            "No scorable rows remain after filtering."
        )

    numeric_columns = [
        "model_margin_dem",
        "actual_dem_margin",
    ]

    for column in numeric_columns:
        cycle_frame[column] = pd.to_numeric(
            cycle_frame[column],
            errors="coerce",
        )

    invalid_margin_rows = cycle_frame[
        numeric_columns
    ].isna().any(axis=1)

    if invalid_margin_rows.any():
        print(
            "Warning: dropping "
            f"{int(invalid_margin_rows.sum())} rows "
            "with missing model or actual margins."
        )

        cycle_frame = cycle_frame.loc[
            ~invalid_margin_rows
        ].copy()

    if cycle_frame.empty:
        raise BiasInvestigationError(
            "No valid rows remain after removing missing margins."
        )

    cycle_frame["margin_error"] = (
        cycle_frame["model_margin_dem"]
        - cycle_frame["actual_dem_margin"]
    )

    cycle_frame["abs_margin_error"] = (
        cycle_frame["margin_error"].abs()
    )

    cycle_frame["squared_margin_error"] = (
        cycle_frame["margin_error"] ** 2
    )

    cycle_frame["predicted_dem_win"] = (
        cycle_frame["model_margin_dem"] > 0
    ).astype(int)

    cycle_frame["actual_dem_win"] = (
        cycle_frame["actual_dem_margin"] > 0
    ).astype(int)

    cycle_frame["seat_error"] = (
        cycle_frame["predicted_dem_win"]
        - cycle_frame["actual_dem_win"]
    )

    expected_dem_seats = int(
        cycle_frame["predicted_dem_win"].sum()
    )

    actual_dem_seats = int(
        cycle_frame["actual_dem_win"].sum()
    )

    mean_margin_error = float(
        cycle_frame["margin_error"].mean()
    )

    median_margin_error = float(
        cycle_frame["margin_error"].median()
    )

    mae = float(
        cycle_frame["abs_margin_error"].mean()
    )

    rmse = float(
        np.sqrt(
            cycle_frame[
                "squared_margin_error"
            ].mean()
        )
    )

    print_heading(
        f"House Production Replay Bias Investigation — {args.cycle}"
    )

    print(f"Input file: {replay_path}")
    print(f"Replay specification: {args.spec}")
    print(f"Scored races: {len(cycle_frame)}")

    print_subheading("Overall results")

    print(f"Predicted Democratic seats: {expected_dem_seats}")
    print(f"Actual Democratic seats:    {actual_dem_seats}")
    print(
        "Democratic seat error:      "
        f"{expected_dem_seats - actual_dem_seats:+d}"
    )
    print()
    print(
        "Mean Democratic margin bias:   "
        f"{mean_margin_error:+.3f}"
    )
    print(
        "Median Democratic margin bias: "
        f"{median_margin_error:+.3f}"
    )
    print(f"Mean absolute error:             {mae:.3f}")
    print(f"Margin RMSE:                     {rmse:.3f}")

    if mean_margin_error > 0:
        print()
        print(
            "Interpretation: positive margin error means the replay "
            "was too Democratic on average."
        )
    elif mean_margin_error < 0:
        print()
        print(
            "Interpretation: negative margin error means the replay "
            "was too Republican on average."
        )

    print_subheading("Seat-call confusion matrix")

    confusion = pd.crosstab(
        cycle_frame["actual_dem_win"],
        cycle_frame["predicted_dem_win"],
        rownames=["Actual Democratic win"],
        colnames=["Predicted Democratic win"],
        dropna=False,
    )

    print(confusion.to_string())

    incumbency_column = first_available(
        cycle_frame,
        [
            "incumbent_configuration",
            "incumbency_configuration",
            "incumbency_status",
        ],
    )

    if incumbency_column is not None:
        print_subheading(
            f"Bias by incumbency configuration ({incumbency_column})"
        )

        print(
            safe_group_summary(
                cycle_frame,
                incumbency_column,
            ).to_string(
                float_format=lambda value: f"{value:.3f}"
            )
        )

    state_column = first_available(
        cycle_frame,
        [
            "state",
            "state_abbreviation",
            "state_code",
        ],
    )

    if state_column is not None:
        print_subheading(
            f"States with largest Democratic overprediction ({state_column})"
        )

        state_summary = safe_group_summary(
            cycle_frame,
            state_column,
        )

        print(
            state_summary.head(15).to_string(
                float_format=lambda value: f"{value:.3f}"
            )
        )

        print_subheading(
            f"States with largest Republican overprediction ({state_column})"
        )

        print(
            state_summary.tail(15)
            .sort_values(
                "mean_margin_error",
                ascending=True,
            )
            .to_string(
                float_format=lambda value: f"{value:.3f}"
            )
        )

    print_subheading("Performance by actual competitiveness")

    cycle_frame["actual_competitiveness_band"] = pd.cut(
        cycle_frame["actual_dem_margin"].abs(),
        bins=[
            -np.inf,
            5,
            10,
            20,
            np.inf,
        ],
        labels=[
            "Within 5 points",
            "5–10 points",
            "10–20 points",
            "More than 20 points",
        ],
        right=True,
    )

    print(
        safe_group_summary(
            cycle_frame,
            "actual_competitiveness_band",
        ).to_string(
            float_format=lambda value: f"{value:.3f}"
        )
    )

    component_candidates = [
        "district_pres_margin_dem",
        "presidential_baseline_margin_dem",
        "national_environment_margin_dem",
        "national_environment_adjustment_dem",
        "district_environment_adjustment_dem",
        "incumbency_adjustment_dem",
        "candidate_quality_adjustment_dem",
        "candidate_quality_margin_dem",
        "district_elasticity",
        "fundamentals_margin_dem",
        "final_fundamentals_margin_dem",
    ]

    component_columns = [
        column
        for column in component_candidates
        if column in cycle_frame.columns
    ]

    display_columns = [
        "race_id",
    ]

    if state_column is not None:
        display_columns.append(
            state_column
        )

    if incumbency_column is not None:
        display_columns.append(
            incumbency_column
        )

    display_columns.extend(
        component_columns
    )

    display_columns.extend(
        [
            "model_margin_dem",
            "actual_dem_margin",
            "margin_error",
            "predicted_dem_win",
            "actual_dem_win",
        ]
    )

    display_columns = list(
        dict.fromkeys(display_columns)
    )

    print_subheading(
        f"Largest Democratic overpredictions — top {args.top}"
    )

    print(
        cycle_frame.sort_values(
            "margin_error",
            ascending=False,
        )[
            display_columns
        ]
        .head(args.top)
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print_subheading(
        f"Largest Republican overpredictions — top {min(args.top, 20)}"
    )

    print(
        cycle_frame.sort_values(
            "margin_error",
            ascending=True,
        )[
            display_columns
        ]
        .head(
            min(
                args.top,
                20,
            )
        )
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print_subheading("Available component columns")

    if component_columns:
        for column in component_columns:
            values = pd.to_numeric(
                cycle_frame[column],
                errors="coerce",
            )

            print(
                f"{column}: "
                f"nonmissing={int(values.notna().sum())}, "
                f"mean={values.mean():.3f}, "
                f"min={values.min():.3f}, "
                f"max={values.max():.3f}"
            )
    else:
        print(
            "No recognized component-breakdown columns were found "
            "in the replay predictions file."
        )

    print()
    print("Diagnostic completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except BiasInvestigationError as exc:
        raise SystemExit(
            f"\nERROR: {exc}"
        ) from exc
