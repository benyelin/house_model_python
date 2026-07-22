#!/usr/bin/env python3
"""
Autopsy historical House production forecasts by cycle.

The analysis reuses run_house_production_replay.prepare_cycle() so the
diagnostic is based on the same leakage-safe model margins used in the
production replay.

Outputs:
    1. Cycle-level forecast summary
    2. Cycle-level component averages
    3. District-level forecast errors
    4. 2016 largest misses
    5. Detected-column inventory

This is primarily a diagnostic script. It deliberately reports missing
component columns rather than silently treating unavailable components
as observed zeros.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from historical.house.backtests import (  # noqa: E402
    run_house_production_replay as replay,
)


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "forecast_autopsy"
)

DEFAULT_TOTAL_SD = math.sqrt(
    5.0625 ** 2
    + 6.1875 ** 2
)

HOUSE_CONTROL_THRESHOLD = 218


COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "district_id": (
        "district_id",
        "district",
        "race_id",
        "canonical_district_id",
    ),
    "state": (
        "state",
        "state_abbreviation",
        "state_po",
    ),
    "district_number": (
        "district_number",
        "district_num",
        "district_code",
    ),
    "actual_margin_dem": (
        "actual_margin_dem",
        "actual_margin",
        "dem_margin",
        "general_margin_dem",
        "two_party_margin_dem",
        "actual_dem_margin",
    ),
    "actual_dem_share": (
        "actual_dem_share",
        "dem_two_party_share",
        "dem_vote_share",
        "general_dem_share",
    ),
    "actual_gop_share": (
        "actual_gop_share",
        "gop_two_party_share",
        "gop_vote_share",
        "general_gop_share",
        "actual_rep_share",
    ),
    "actual_winner": (
        "actual_winner",
        "winner",
        "winner_party",
    ),
    "generic_ballot_margin_dem": (
        "generic_ballot_margin_dem",
        "generic_ballot_dem_margin",
        "generic_ballot_margin",
        "election_day_generic_ballot_margin_dem",
    ),
    "national_environment_dem": (
        "national_environment_dem",
        "national_environment",
        "environment_margin_dem",
        "national_environment_margin_dem",
    ),
    "presidential_baseline_dem": (
        "district_partisan_baseline_dem",
        "presidential_baseline_dem",
        "district_presidential_baseline_dem",
        "baseline_margin_dem",
        "presidential_margin_dem",
    ),
    "environment_adjustment_dem": (
        "district_environment_adjustment_dem",
        "environment_adjustment_dem",
        "national_environment_adjustment_dem",
    ),
    "state_adjustment_dem": (
        "state_environment_adjustment_dem",
        "state_adjustment_dem",
    ),
    "elasticity_adjustment_dem": (
        "district_elasticity_adjustment_dem",
        "elasticity_adjustment_dem",
    ),
    "incumbency_adjustment_dem": (
        "incumbency_adjustment_dem",
        "district_incumbency_adjustment_dem",
        "incumbency_bonus_dem",
    ),
    "candidate_quality_adjustment_dem": (
        "candidate_quality_adjustment_dem",
        "district_candidate_quality_adjustment_dem",
        "candidate_war_adjustment_dem",
        "candidate_quality_dem",
    ),
    "polling_adjustment_dem": (
        "polling_adjustment_dem",
        "district_polling_adjustment_dem",
        "poll_adjustment_dem",
    ),
    "fundamentals_margin_dem": (
        "fundamentals_margin_dem",
        "district_fundamentals_margin_dem",
        "forecast_fundamentals_margin_dem",
    ),
}


COMPONENT_KEYS = (
    "presidential_baseline_dem",
    "environment_adjustment_dem",
    "state_adjustment_dem",
    "elasticity_adjustment_dem",
    "incumbency_adjustment_dem",
    "candidate_quality_adjustment_dem",
    "polling_adjustment_dem",
    "fundamentals_margin_dem",
)


class AutopsyError(RuntimeError):
    """Raised when the forecast autopsy cannot be completed safely."""


def find_column(
    columns: Iterable[str],
    candidates: Iterable[str],
) -> str | None:
    available = set(columns)

    for candidate in candidates:
        if candidate in available:
            return candidate

    return None


def numeric_series(
    frame: pd.DataFrame,
    column: str | None,
) -> pd.Series:
    if column is None:
        return pd.Series(
            np.nan,
            index=frame.index,
            dtype=float,
        )

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    )


def derive_actual_margin(
    frame: pd.DataFrame,
    detected: dict[str, str | None],
) -> tuple[pd.Series, str]:
    direct_column = detected["actual_margin_dem"]

    if direct_column is not None:
        return (
            numeric_series(
                frame,
                direct_column,
            ),
            direct_column,
        )

    dem_share_column = detected["actual_dem_share"]
    gop_share_column = detected["actual_gop_share"]

    if (
        dem_share_column is not None
        and gop_share_column is not None
    ):
        margin = (
            numeric_series(
                frame,
                dem_share_column,
            )
            - numeric_series(
                frame,
                gop_share_column,
            )
        )

        return (
            margin,
            (
                f"{dem_share_column}"
                f" - {gop_share_column}"
            ),
        )

    raise AutopsyError(
        "Could not identify an actual Democratic margin column "
        "or a Democratic/GOP vote-share pair."
    )


def derive_district_id(
    frame: pd.DataFrame,
    detected: dict[str, str | None],
) -> pd.Series:
    district_id_column = detected["district_id"]

    if district_id_column is not None:
        return (
            frame[district_id_column]
            .fillna("")
            .astype(str)
        )

    state_column = detected["state"]
    district_number_column = detected["district_number"]

    if (
        state_column is not None
        and district_number_column is not None
    ):
        state = (
            frame[state_column]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.strip()
        )

        district = (
            frame[district_number_column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        return state + "-" + district

    return pd.Series(
        [
            f"row_{index}"
            for index in range(len(frame))
        ],
        index=frame.index,
        dtype=str,
    )


def normal_dem_probability(
    margin: float,
    total_sd: float,
) -> float:
    if not np.isfinite(margin):
        return float("nan")

    if total_sd <= 0.0:
        return float(margin > 0.0)

    return float(
        NormalDist().cdf(
            margin / total_sd
        )
    )


def build_cycle_autopsy(
    master: pd.DataFrame,
    cycle: int,
    candidate_quality_weight: float,
    candidate_war_path: Path,
    total_sd: float,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    pd.DataFrame,
    list[dict[str, Any]],
]:
    df, model_margin, forecast_source = (
        replay.prepare_cycle(
            master=master,
            cycle=cycle,
            candidate_quality_weight=(
                candidate_quality_weight
            ),
            candidate_war_path=candidate_war_path,
        )
    )

    df = replay.normalize_fixed_control(
        df.copy()
    )

    detected = {
        key: find_column(
            df.columns,
            candidates,
        )
        for key, candidates
        in COLUMN_CANDIDATES.items()
    }

    actual_margin, actual_margin_source = (
        derive_actual_margin(
            df,
            detected,
        )
    )

    model_margin_numeric = pd.to_numeric(
        model_margin,
        errors="coerce",
    )

    if model_margin_numeric.isna().any():
        raise AutopsyError(
            f"{cycle}: model margin contains missing values."
        )

    district_ids = derive_district_id(
        df,
        detected,
    )

    fixed_party = (
        df["party_control_fixed"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    dem_probabilities = model_margin_numeric.apply(
        lambda margin: normal_dem_probability(
            margin,
            total_sd,
        )
    )

    dem_probabilities.loc[
        fixed_party.eq("D")
    ] = 1.0

    dem_probabilities.loc[
        fixed_party.eq("R")
    ] = 0.0

    actual_winner_column = detected["actual_winner"]

    if actual_winner_column is None:
        actual_dem_win = (
            actual_margin > 0.0
        ).astype(float)
    else:
        winners = (
            df[actual_winner_column]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.strip()
        )

        actual_dem_win = winners.eq("D").astype(float)

    expected_dem_seats = float(
        dem_probabilities.sum()
    )

    actual_dem_seats = int(
        actual_dem_win.sum()
    )

    district_output = pd.DataFrame(
        {
            "cycle": int(cycle),
            "district_id": district_ids,
            "forecast_source": forecast_source,
            "model_margin_dem": model_margin_numeric,
            "actual_margin_dem": actual_margin,
            "margin_error_dem": (
                model_margin_numeric
                - actual_margin
            ),
            "absolute_margin_error": (
                model_margin_numeric
                - actual_margin
            ).abs(),
            "dem_win_probability": dem_probabilities,
            "actual_dem_win": actual_dem_win,
            "probability_error_dem": (
                dem_probabilities
                - actual_dem_win
            ),
            "party_control_fixed": fixed_party,
        }
    )

    for key in COMPONENT_KEYS:
        source_column = detected[key]

        district_output[key] = numeric_series(
            df,
            source_column,
        )

    component_rows: list[dict[str, Any]] = []

    for key in COMPONENT_KEYS:
        source_column = detected[key]
        values = district_output[key]

        component_rows.append(
            {
                "cycle": int(cycle),
                "component": key,
                "source_column": (
                    source_column
                    if source_column is not None
                    else ""
                ),
                "available": bool(
                    source_column is not None
                ),
                "nonmissing_rows": int(
                    values.notna().sum()
                ),
                "mean": (
                    float(values.mean())
                    if values.notna().any()
                    else np.nan
                ),
                "median": (
                    float(values.median())
                    if values.notna().any()
                    else np.nan
                ),
                "mean_absolute_value": (
                    float(values.abs().mean())
                    if values.notna().any()
                    else np.nan
                ),
                "minimum": (
                    float(values.min())
                    if values.notna().any()
                    else np.nan
                ),
                "maximum": (
                    float(values.max())
                    if values.notna().any()
                    else np.nan
                ),
            }
        )

    generic_ballot_column = detected[
        "generic_ballot_margin_dem"
    ]

    national_environment_column = detected[
        "national_environment_dem"
    ]

    generic_ballot = numeric_series(
        df,
        generic_ballot_column,
    )

    national_environment = numeric_series(
        df,
        national_environment_column,
    )

    scorable_margin_rows = (
        actual_margin.notna()
        & model_margin_numeric.notna()
    )

    cycle_summary = {
        "cycle": int(cycle),
        "forecast_source": forecast_source,
        "district_rows": int(len(df)),
        "scorable_margin_rows": int(
            scorable_margin_rows.sum()
        ),
        "generic_ballot_source_column": (
            generic_ballot_column or ""
        ),
        "generic_ballot_margin_dem": (
            float(generic_ballot.mean())
            if generic_ballot.notna().any()
            else np.nan
        ),
        "national_environment_source_column": (
            national_environment_column or ""
        ),
        "national_environment_dem": (
            float(national_environment.mean())
            if national_environment.notna().any()
            else np.nan
        ),
        "actual_margin_source": actual_margin_source,
        "mean_model_margin_dem": float(
            model_margin_numeric.mean()
        ),
        "mean_actual_margin_dem": float(
            actual_margin.loc[
                scorable_margin_rows
            ].mean()
        ),
        "mean_margin_error_dem": float(
            district_output.loc[
                scorable_margin_rows,
                "margin_error_dem",
            ].mean()
        ),
        "margin_mae": float(
            district_output.loc[
                scorable_margin_rows,
                "absolute_margin_error",
            ].mean()
        ),
        "margin_rmse": float(
            np.sqrt(
                np.mean(
                    district_output.loc[
                        scorable_margin_rows,
                        "margin_error_dem",
                    ] ** 2
                )
            )
        ),
        "expected_dem_seats": (
            expected_dem_seats
        ),
        "actual_dem_seats": (
            actual_dem_seats
        ),
        "expected_seat_error_dem": (
            expected_dem_seats
            - actual_dem_seats
        ),
        "deterministic_dem_seats": int(
            (
                model_margin_numeric > 0.0
            ).sum()
        ),
        "total_marginal_sd": float(
            total_sd
        ),
    }

    inventory_rows: list[dict[str, Any]] = []

    for key, candidates in COLUMN_CANDIDATES.items():
        inventory_rows.append(
            {
                "cycle": int(cycle),
                "logical_field": key,
                "detected_column": (
                    detected[key] or ""
                ),
                "found": bool(
                    detected[key] is not None
                ),
                "candidate_names": " | ".join(
                    candidates
                ),
            }
        )

    return (
        cycle_summary,
        component_rows,
        district_output,
        inventory_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a cycle and district-level autopsy "
            "of historical House production forecasts."
        )
    )

    parser.add_argument(
        "--master-path",
        type=Path,
        default=replay.DEFAULT_MASTER_PATH,
    )

    parser.add_argument(
        "--candidate-war-path",
        type=Path,
        default=replay.DEFAULT_WAR_PATH,
    )

    parser.add_argument(
        "--candidate-quality-weight",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--total-sd",
        type=float,
        default=DEFAULT_TOTAL_SD,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--clean-output",
        action="store_true",
    )

    args = parser.parse_args()

    if args.total_sd <= 0.0:
        raise ValueError(
            "--total-sd must be positive."
        )

    if not args.master_path.exists():
        raise FileNotFoundError(
            f"Historical master not found: "
            f"{args.master_path}"
        )

    if args.clean_output and args.output_dir.exists():
        import shutil

        shutil.rmtree(
            args.output_dir
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    master = pd.read_csv(
        args.master_path,
        low_memory=False,
    )

    replay.validate_input(
        master
    )

    cycle_summaries: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    district_frames: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, Any]] = []

    for cycle in replay.SUPPORTED_CYCLES:
        (
            cycle_summary,
            cycle_components,
            district_output,
            cycle_inventory,
        ) = build_cycle_autopsy(
            master=master,
            cycle=int(cycle),
            candidate_quality_weight=(
                args.candidate_quality_weight
            ),
            candidate_war_path=(
                args.candidate_war_path
            ),
            total_sd=float(args.total_sd),
        )

        cycle_summaries.append(
            cycle_summary
        )

        component_rows.extend(
            cycle_components
        )

        district_frames.append(
            district_output
        )

        inventory_rows.extend(
            cycle_inventory
        )

    cycle_summary_df = pd.DataFrame(
        cycle_summaries
    )

    components_df = pd.DataFrame(
        component_rows
    )

    districts_df = pd.concat(
        district_frames,
        ignore_index=True,
    )

    inventory_df = pd.DataFrame(
        inventory_rows
    )

    misses_2016 = (
        districts_df.loc[
            districts_df["cycle"].eq(2016)
            & districts_df[
                "actual_margin_dem"
            ].notna()
        ]
        .sort_values(
            [
                "absolute_margin_error",
                "district_id",
            ],
            ascending=[
                False,
                True,
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    cycle_path = (
        args.output_dir
        / "house_forecast_autopsy_cycle_summary.csv"
    )

    components_path = (
        args.output_dir
        / "house_forecast_autopsy_components.csv"
    )

    districts_path = (
        args.output_dir
        / "house_forecast_autopsy_districts.csv"
    )

    misses_path = (
        args.output_dir
        / "house_forecast_autopsy_2016_largest_misses.csv"
    )

    inventory_path = (
        args.output_dir
        / "house_forecast_autopsy_column_inventory.csv"
    )

    validation_path = (
        args.output_dir
        / "house_forecast_autopsy_validation.txt"
    )

    cycle_summary_df.to_csv(
        cycle_path,
        index=False,
    )

    components_df.to_csv(
        components_path,
        index=False,
    )

    districts_df.to_csv(
        districts_path,
        index=False,
    )

    misses_2016.to_csv(
        misses_path,
        index=False,
    )

    inventory_df.to_csv(
        inventory_path,
        index=False,
    )

    expected_cycles = list(
        replay.SUPPORTED_CYCLES
    )

    observed_cycles = (
        cycle_summary_df["cycle"]
        .astype(int)
        .tolist()
    )

    if observed_cycles != expected_cycles:
        raise AutopsyError(
            "Unexpected cycle ordering or missing cycles: "
            f"{observed_cycles}"
        )

    if districts_df.duplicated(
        [
            "cycle",
            "district_id",
        ]
    ).any():
        duplicates = (
            districts_df.loc[
                districts_df.duplicated(
                    [
                        "cycle",
                        "district_id",
                    ],
                    keep=False,
                ),
                [
                    "cycle",
                    "district_id",
                ],
            ]
            .sort_values(
                [
                    "cycle",
                    "district_id",
                ]
            )
        )

        duplicate_path = (
            args.output_dir
            / "house_forecast_autopsy_duplicate_ids.csv"
        )

        duplicates.to_csv(
            duplicate_path,
            index=False,
        )

        print()
        print(
            "WARNING: duplicate cycle/district IDs found."
        )
        print(
            f"Details written to: {duplicate_path}"
        )

    validation_lines = [
        (
            "PASS: cycles = "
            + ", ".join(
                str(cycle)
                for cycle in expected_cycles
            )
        ),
        (
            "PASS: cycle summary rows = "
            f"{len(cycle_summary_df)}"
        ),
        (
            "PASS: district rows = "
            f"{len(districts_df)}"
        ),
        (
            "PASS: 2016 scorable rows = "
            f"{len(misses_2016)}"
        ),
    ]

    validation_text = (
        "House Forecast Autopsy Validation\n"
        + "=" * 42
        + "\n"
        + "\n".join(validation_lines)
        + "\n\nVALIDATION PASSED\n"
    )

    validation_path.write_text(
        validation_text
    )

    print()
    print("=" * 100)
    print("House historical forecast autopsy: cycle summary")
    print("=" * 100)

    summary_columns = [
        "cycle",
        "generic_ballot_margin_dem",
        "national_environment_dem",
        "mean_model_margin_dem",
        "mean_actual_margin_dem",
        "mean_margin_error_dem",
        "margin_mae",
        "margin_rmse",
        "expected_dem_seats",
        "actual_dem_seats",
        "expected_seat_error_dem",
    ]

    print(
        cycle_summary_df[
            summary_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.3f}"
            ),
        )
    )

    print()
    print("=" * 100)
    print("Detected component averages")
    print("=" * 100)

    component_display = (
        components_df.loc[
            components_df["available"],
            [
                "cycle",
                "component",
                "source_column",
                "nonmissing_rows",
                "mean",
                "median",
                "mean_absolute_value",
                "minimum",
                "maximum",
            ],
        ]
    )

    if component_display.empty:
        print(
            "No recognized component columns were found."
        )
    else:
        print(
            component_display.to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.3f}"
                ),
            )
        )

    print()
    print("=" * 100)
    print("2016 largest district-level margin misses")
    print("=" * 100)

    miss_display_columns = [
        "district_id",
        "model_margin_dem",
        "actual_margin_dem",
        "margin_error_dem",
        "absolute_margin_error",
        "dem_win_probability",
        "actual_dem_win",
    ]

    available_component_columns = [
        column
        for column in COMPONENT_KEYS
        if (
            column in misses_2016.columns
            and misses_2016[
                column
            ].notna().any()
        )
    ]

    print(
        misses_2016[
            miss_display_columns
            + available_component_columns
        ]
        .head(30)
        .to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.3f}"
            ),
        )
    )

    missing_fields = (
        inventory_df.loc[
            ~inventory_df["found"],
            "logical_field",
        ]
        .drop_duplicates()
        .tolist()
    )

    print()
    print("Missing recognized fields")
    print("-" * 100)

    if missing_fields:
        for field in missing_fields:
            print(f"  - {field}")
    else:
        print("  None")

    print()
    print(validation_text)

    print("Wrote:")
    for path in [
        cycle_path,
        components_path,
        districts_path,
        misses_path,
        inventory_path,
        validation_path,
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
