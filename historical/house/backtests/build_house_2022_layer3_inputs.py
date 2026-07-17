from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_BACKTEST_INPUT_PATH = (
    PROJECT_ROOT
    / "historical/house/backtests/outputs/"
    "house_2022_backtest_inputs_provisional.csv"
)

DEFAULT_ELASTICITY_PATH = (
    PROJECT_ROOT
    / "historical/house/elasticity/"
    "house_district_elasticity.csv"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "historical/house/backtests/outputs/"
    "house_2022_backtest_inputs_layer3.csv"
)

DEFAULT_VALIDATION_PATH = (
    PROJECT_ROOT
    / "historical/house/backtests/outputs/"
    "house_2022_backtest_inputs_layer3_validation.txt"
)


def build_layer3_inputs(
    races: pd.DataFrame,
    elasticity: pd.DataFrame,
    fallback_elasticity: float,
) -> tuple[pd.DataFrame, str]:
    """
    Join historical elasticity estimates onto the 2022 backtest input.

    District-number matches across the 2020-to-2022 redistricting boundary
    are treated as approximations, not exact geographic continuity.
    """
    required_race_columns = {
        "cycle",
        "race_id",
        "state",
        "district",
        "district_pres_margin_dem",
        "actual_dem_margin",
        "general_election_party_structure",
        "district_elasticity",
    }

    missing_race_columns = sorted(
        required_race_columns - set(races.columns)
    )

    if missing_race_columns:
        raise ValueError(
            "Backtest input is missing required columns: "
            + ", ".join(missing_race_columns)
        )

    required_elasticity_columns = {
        "race_id",
        "raw_elasticity",
        "shrunk_elasticity",
        "shrink_target",
        "shrinkage_strength",
        "observation_count",
        "residual_rmse",
        "low_information_estimate",
    }

    missing_elasticity_columns = sorted(
        required_elasticity_columns - set(elasticity.columns)
    )

    if missing_elasticity_columns:
        raise ValueError(
            "Elasticity table is missing required columns: "
            + ", ".join(missing_elasticity_columns)
        )

    if len(races) != 435:
        raise ValueError(
            f"Expected 435 backtest rows; found {len(races)}."
        )

    if races["race_id"].duplicated().any():
        raise ValueError(
            "Duplicate race IDs found in the 2022 backtest input."
        )

    if elasticity["race_id"].duplicated().any():
        raise ValueError(
            "Duplicate race IDs found in the elasticity table."
        )

    if not np.isfinite(fallback_elasticity):
        raise ValueError("fallback_elasticity must be finite.")

    elasticity_keep = elasticity[
        [
            "race_id",
            "raw_elasticity",
            "shrunk_elasticity",
            "shrink_target",
            "shrinkage_strength",
            "observation_count",
            "residual_rmse",
            "low_information_estimate",
        ]
    ].copy()

    elasticity_keep = elasticity_keep.rename(
        columns={
            "raw_elasticity": "district_elasticity_raw",
            "shrunk_elasticity": "district_elasticity_shrunk",
            "shrink_target": "elasticity_shrink_target",
            "shrinkage_strength": (
                "elasticity_shrinkage_strength"
            ),
            "observation_count": (
                "elasticity_observation_count"
            ),
            "residual_rmse": "elasticity_residual_rmse",
            "low_information_estimate": (
                "elasticity_low_information_estimate"
            ),
        }
    )

    # The provisional file intentionally contains a blank placeholder
    # named district_elasticity. Remove it before adding the selected
    # historical estimate.
    combined = races.drop(
        columns=["district_elasticity"],
    ).merge(
        elasticity_keep,
        on="race_id",
        how="left",
        validate="one_to_one",
    )

    combined["elasticity_match_status"] = np.where(
        combined["district_elasticity_shrunk"].notna(),
        "district_label_match_across_redistricting",
        "fallback_no_historical_district_label",
    )

    combined["elasticity_source"] = np.where(
        combined["district_elasticity_shrunk"].notna(),
        (
            "2012-2020 consecutive House swings; constrained OLS; "
            "district-number match applied across 2022 redistricting"
        ),
        "fallback elasticity",
    )

    combined["district_elasticity"] = pd.to_numeric(
        combined["district_elasticity_shrunk"],
        errors="coerce",
    ).fillna(fallback_elasticity)

    combined["elasticity_fallback_used"] = (
        combined["district_elasticity_shrunk"].isna()
    )

    combined["elasticity_boundary_compatibility"] = np.where(
        combined["elasticity_fallback_used"],
        "no historical district-label estimate",
        "approximate label match; geography not guaranteed continuous",
    )

    numeric_columns = [
        "district_elasticity_raw",
        "district_elasticity_shrunk",
        "district_elasticity",
        "elasticity_shrink_target",
        "elasticity_shrinkage_strength",
        "elasticity_observation_count",
        "elasticity_residual_rmse",
    ]

    for column in numeric_columns:
        combined[column] = pd.to_numeric(
            combined[column],
            errors="coerce",
        )

    matched = int(
        combined["district_elasticity_shrunk"].notna().sum()
    )
    fallback_count = int(
        combined["elasticity_fallback_used"].sum()
    )
    missing_selected = int(
        combined["district_elasticity"].isna().sum()
    )
    nonfinite_selected = int(
        (
            combined["district_elasticity"].notna()
            & ~np.isfinite(combined["district_elasticity"])
        ).sum()
    )
    duplicate_races = int(
        combined["race_id"].duplicated().sum()
    )

    failures: list[str] = []

    if len(combined) != 435:
        failures.append(
            f"Expected 435 rows after merge; found {len(combined)}."
        )

    if duplicate_races:
        failures.append(
            f"Found {duplicate_races} duplicate race IDs."
        )

    if missing_selected:
        failures.append(
            f"Found {missing_selected} missing selected elasticities."
        )

    if nonfinite_selected:
        failures.append(
            f"Found {nonfinite_selected} nonfinite selected elasticities."
        )

    selected_summary = combined["district_elasticity"].describe()

    match_counts = (
        combined["elasticity_match_status"]
        .value_counts(dropna=False)
    )

    state_fallback_counts = (
        combined.loc[
            combined["elasticity_fallback_used"],
            "state",
        ]
        .value_counts()
        .sort_index()
    )

    report_lines = [
        "2022 House Layer 3 Input Validation",
        "=" * 35,
        "",
        f"Rows: {len(combined)}",
        f"Unique race IDs: {combined['race_id'].nunique()}",
        f"Historical district-label matches: {matched}",
        f"Fallback elasticities used: {fallback_count}",
        f"Fallback elasticity: {fallback_elasticity:.6f}",
        f"Missing selected elasticities: {missing_selected}",
        f"Nonfinite selected elasticities: {nonfinite_selected}",
        "",
        "Match status counts:",
        match_counts.to_string(),
        "",
        "Selected elasticity summary:",
        selected_summary.to_string(
            float_format=lambda value: f"{value:.6f}"
        ),
        "",
        "Fallback counts by state:",
        (
            state_fallback_counts.to_string()
            if not state_fallback_counts.empty
            else "None"
        ),
        "",
        "Boundary warning:",
        (
            "Historical elasticity estimates use 2012-2020 district "
            "labels. Matching 2022 race IDs are approximate label "
            "matches across redistricting and do not guarantee geographic "
            "continuity."
        ),
        "",
        "Validation status:",
    ]

    if failures:
        report_lines.append("FAILED")
        report_lines.extend(
            f"- {failure}"
            for failure in failures
        )
    else:
        report_lines.append("PASSED")

    report = "\n".join(report_lines)

    if failures:
        raise RuntimeError(report)

    return combined, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Join historical district elasticity estimates onto "
            "the 2022 House backtest input."
        )
    )
    parser.add_argument(
        "--backtest-input-path",
        type=Path,
        default=DEFAULT_BACKTEST_INPUT_PATH,
    )
    parser.add_argument(
        "--elasticity-path",
        type=Path,
        default=DEFAULT_ELASTICITY_PATH,
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument(
        "--validation-path",
        type=Path,
        default=DEFAULT_VALIDATION_PATH,
    )
    parser.add_argument(
        "--fallback-elasticity",
        type=float,
        default=None,
        help=(
            "Fallback for 2022 district labels without a historical "
            "estimate. Default: the elasticity table's shrink target."
        ),
    )
    args = parser.parse_args()

    if not args.backtest_input_path.exists():
        raise FileNotFoundError(
            f"Missing backtest input: {args.backtest_input_path}"
        )

    if not args.elasticity_path.exists():
        raise FileNotFoundError(
            f"Missing elasticity table: {args.elasticity_path}"
        )

    races = pd.read_csv(
        args.backtest_input_path,
        dtype={"race_id": str},
    )

    elasticity = pd.read_csv(
        args.elasticity_path,
        dtype={"race_id": str},
    )

    shrink_targets = pd.to_numeric(
        elasticity["shrink_target"],
        errors="coerce",
    ).dropna().unique()

    if len(shrink_targets) != 1:
        raise ValueError(
            "Elasticity table must contain exactly one finite "
            "shrink target."
        )

    fallback_elasticity = (
        float(shrink_targets[0])
        if args.fallback_elasticity is None
        else float(args.fallback_elasticity)
    )

    combined, report = build_layer3_inputs(
        races=races,
        elasticity=elasticity,
        fallback_elasticity=fallback_elasticity,
    )

    args.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.validation_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        args.output_path,
        index=False,
    )
    args.validation_path.write_text(report)

    print(report)
    print()
    print(f"Wrote: {args.output_path}")
    print(f"Wrote: {args.validation_path}")


if __name__ == "__main__":
    main()
