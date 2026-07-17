from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PROFILE_PATH = (
    PROJECT_ROOT
    / "historical/house/warehouse/"
    "house_structural_profile.csv"
)

DEFAULT_BEHAVIOR_PATH = (
    PROJECT_ROOT
    / "historical/house/warehouse/"
    "house_district_behavior.csv"
)

DEFAULT_ELASTICITY_PATH = (
    PROJECT_ROOT
    / "historical/house/elasticity/"
    "house_district_elasticity.csv"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "historical/house/warehouse/"
    "house_district_master_features.csv"
)

DEFAULT_VALIDATION_PATH = (
    PROJECT_ROOT
    / "historical/house/warehouse/"
    "house_district_master_features_validation.txt"
)


EXPECTED_DRA_MISSING = {
    "AK-AL",
    "DE-AL",
    "ND-AL",
    "SD-AL",
    "VT-AL",
    "WY-AL",
}


def parse_bool_series(
    series: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def reliability_score(
    series: pd.Series,
) -> pd.Series:
    mapping = {
        "high": 1.00,
        "medium": 0.75,
        "low": 0.50,
        "insufficient_single_cycle": 0.25,
        "no_scorable_history": 0.00,
    }

    return (
        series.map(mapping)
        .fillna(0.0)
        .astype(float)
    )


def build_master_features(
    profile: pd.DataFrame,
    behavior: pd.DataFrame,
    elasticity: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    required_profile = {
        "race_id",
        "state",
        "district",
        "region",
        "district_type",
        "dra_data_available",
        "pres_2020_margin_dem",
        "pres_2024_margin_dem",
        "presidential_margin_average_2020_2024_dem",
        "presidential_swing_2020_to_2024_dem",
        "dra_composite_margin_dem",
    }

    required_behavior = {
        "race_id",
        "selected_boundary_regime_id",
        "selected_regime_reliability",
        "selected_regime_scorable_elections",
        "selected_regime_mean_dem_margin",
        "selected_regime_margin_std",
        "selected_regime_mean_absolute_margin",
        "selected_regime_mean_absolute_swing",
        "selected_regime_party_switch_count",
        "selected_regime_competitive_within_10_rate",
        "selected_regime_trend_slope_points_per_election",
        "selected_regime_trend_residual_rmse",
    }

    required_elasticity = {
        "race_id",
        "observation_count",
        "raw_elasticity",
        "shrunk_elasticity",
        "residual_rmse",
        "low_information_estimate",
    }

    missing_profile = sorted(
        required_profile - set(profile.columns)
    )

    missing_behavior = sorted(
        required_behavior - set(behavior.columns)
    )

    missing_elasticity = sorted(
        required_elasticity - set(elasticity.columns)
    )

    if missing_profile:
        raise ValueError(
            "Structural profile is missing columns: "
            + ", ".join(missing_profile)
        )

    if missing_behavior:
        raise ValueError(
            "District behavior table is missing columns: "
            + ", ".join(missing_behavior)
        )

    if missing_elasticity:
        raise ValueError(
            "Elasticity table is missing columns: "
            + ", ".join(missing_elasticity)
        )

    if len(profile) != 435:
        raise ValueError(
            f"Expected 435 structural-profile rows; found {len(profile)}."
        )

    if len(behavior) != 435:
        raise ValueError(
            f"Expected 435 behavior rows; found {len(behavior)}."
        )

    for name, frame in [
        ("profile", profile),
        ("behavior", behavior),
        ("elasticity", elasticity),
    ]:
        if frame["race_id"].duplicated().any():
            raise ValueError(
                f"Duplicate race IDs found in {name} table."
            )

    behavior_feature_columns = [
        column
        for column in behavior.columns
        if column not in {
            "state",
            "district",
        }
    ]

    required_elasticity_columns = [
        "race_id",
        "observation_count",
        "raw_elasticity",
        "shrunk_elasticity",
        "residual_rmse",
        "low_information_estimate",
        "national_swing_sum_squares",
    ]

    optional_elasticity_columns = [
        column
        for column in [
            "shrinkage_target",
            "shrinkage_strength",
        ]
        if column in elasticity.columns
    ]

    elasticity_keep = elasticity[
        [
            *required_elasticity_columns,
            *optional_elasticity_columns,
        ]
    ].copy()

    elasticity_keep = elasticity_keep.rename(
        columns={
            "observation_count": (
                "historical_elasticity_observation_count"
            ),
            "raw_elasticity": (
                "historical_raw_elasticity"
            ),
            "shrunk_elasticity": (
                "historical_shrunk_elasticity"
            ),
            "residual_rmse": (
                "historical_elasticity_residual_rmse"
            ),
            "low_information_estimate": (
                "historical_elasticity_low_information"
            ),
            "national_swing_sum_squares": (
                "historical_elasticity_information"
            ),
            "shrinkage_target": (
                "historical_elasticity_shrinkage_target"
            ),
            "shrinkage_strength": (
                "historical_elasticity_shrinkage_strength"
            ),
        }
    )

    master = profile.merge(
        behavior[behavior_feature_columns],
        on="race_id",
        how="left",
        validate="one_to_one",
    )

    master = master.merge(
        elasticity_keep,
        on="race_id",
        how="left",
        validate="one_to_one",
    )

    master["dra_data_available"] = parse_bool_series(
        master["dra_data_available"]
    )

    master[
        "historical_elasticity_low_information"
    ] = parse_bool_series(
        master[
            "historical_elasticity_low_information"
        ]
    )

    numeric_columns = [
        "selected_regime_scorable_elections",
        "selected_regime_mean_dem_margin",
        "selected_regime_margin_std",
        "selected_regime_mean_absolute_margin",
        "selected_regime_mean_absolute_swing",
        "selected_regime_party_switch_count",
        "selected_regime_competitive_within_10_rate",
        "selected_regime_trend_slope_points_per_election",
        "selected_regime_trend_residual_rmse",
        "historical_elasticity_observation_count",
        "historical_raw_elasticity",
        "historical_shrunk_elasticity",
        "historical_elasticity_residual_rmse",
        "historical_elasticity_information",
    ]

    for column in numeric_columns:
        master[column] = pd.to_numeric(
            master[column],
            errors="coerce",
        )

    # ---------------------------------------------------------------
    # Feature-family availability and reliability
    # ---------------------------------------------------------------

    master["behavior_data_available"] = (
        master["selected_regime_scorable_elections"]
        .fillna(0)
        .gt(0)
    )

    master["behavior_multi_cycle_available"] = (
        master["selected_regime_scorable_elections"]
        .fillna(0)
        .ge(2)
    )

    master["behavior_high_or_medium_reliability"] = (
        master["selected_regime_reliability"]
        .isin({"high", "medium"})
    )

    master["behavior_reliability_score"] = (
        reliability_score(
            master["selected_regime_reliability"]
        )
    )

    master["historical_elasticity_available"] = (
        master["historical_shrunk_elasticity"]
        .notna()
    )

    master[
        "historical_elasticity_reliable"
    ] = (
        master[
            "historical_elasticity_available"
        ]
        & ~master[
            "historical_elasticity_low_information"
        ]
        & master[
            "historical_elasticity_observation_count"
        ].fillna(0).ge(3)
    )

    master["core_structural_features_available"] = (
        master[
            [
                "pres_2020_margin_dem",
                "pres_2024_margin_dem",
                "presidential_margin_average_2020_2024_dem",
                "presidential_swing_2020_to_2024_dem",
            ]
        ]
        .notna()
        .all(axis=1)
    )

    master["complete_master_feature_row"] = (
        master["dra_data_available"]
        & master["behavior_multi_cycle_available"]
        & master["historical_elasticity_available"]
        & master["core_structural_features_available"]
    )

    # ---------------------------------------------------------------
    # Initial district-DNA summary features
    # ---------------------------------------------------------------

    master["historical_behavior_competitiveness"] = (
        100.0
        - master[
            "selected_regime_mean_absolute_margin"
        ]
    ).clip(
        lower=0.0,
        upper=100.0,
    )

    master["historical_behavior_volatility_index"] = (
        master["selected_regime_margin_std"]
        .clip(lower=0.0)
    )

    master["historical_behavior_swing_index"] = (
        master[
            "selected_regime_mean_absolute_swing"
        ]
        .clip(lower=0.0)
    )

    master["historical_behavior_predictability_index"] = (
        100.0
        - (
            5.0
            * master[
                "selected_regime_trend_residual_rmse"
            ]
        )
    ).clip(
        lower=0.0,
        upper=100.0,
    )

    master["district_dna_feature_version"] = "1.0"

    master["district_dna_stage"] = (
        "structural_behavior_elasticity_foundation"
    )

    master["district_dna_notes"] = (
        "Master feature warehouse combines current structural data, "
        "boundary-regime-aware House behavior, and historical district "
        "elasticity estimates. Composite indexes are descriptive only "
        "until validated in out-of-sample model comparisons."
    )

    master = master.sort_values(
        ["state", "district"],
        key=lambda series: (
            pd.to_numeric(
                series,
                errors="coerce",
            )
            if series.name == "district"
            else series
        ),
    ).reset_index(drop=True)

    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------

    duplicate_race_ids = int(
        master["race_id"].duplicated().sum()
    )

    dra_rows = int(
        master["dra_data_available"].sum()
    )

    behavior_rows = int(
        master["behavior_data_available"].sum()
    )

    behavior_multi_cycle_rows = int(
        master["behavior_multi_cycle_available"].sum()
    )

    elasticity_rows = int(
        master["historical_elasticity_available"].sum()
    )

    reliable_elasticity_rows = int(
        master[
            "historical_elasticity_reliable"
        ].sum()
    )

    complete_rows = int(
        master["complete_master_feature_row"].sum()
    )

    missing_dra_ids = set(
        master.loc[
            ~master["dra_data_available"],
            "race_id",
        ]
    )

    failures: list[str] = []

    if len(master) != 435:
        failures.append(
            f"Expected 435 master rows; found {len(master)}."
        )

    if master["race_id"].nunique() != 435:
        failures.append(
            "Expected 435 unique race IDs."
        )

    if duplicate_race_ids:
        failures.append(
            f"Found {duplicate_race_ids} duplicate race IDs."
        )

    if dra_rows != 429:
        failures.append(
            f"Expected 429 DRA-covered rows; found {dra_rows}."
        )

    if missing_dra_ids != EXPECTED_DRA_MISSING:
        failures.append(
            "DRA-missing race IDs do not match expected at-large seats."
        )

    if behavior_rows != 424:
        failures.append(
            "Expected 424 districts with at least one scorable "
            f"behavior observation; found {behavior_rows}."
        )

    if elasticity_rows != 405:
        failures.append(
            "Expected 405 historical elasticity estimates; "
            f"found {elasticity_rows}."
        )

    if master[
        "behavior_reliability_score"
    ].lt(0).any() or master[
        "behavior_reliability_score"
    ].gt(1).any():
        failures.append(
            "Behavior reliability scores fall outside [0, 1]."
        )

    report_lines = [
        "House District Master Feature Validation",
        "=" * 40,
        "",
        f"Rows: {len(master)}",
        f"Unique race IDs: {master['race_id'].nunique()}",
        f"Duplicate race IDs: {duplicate_race_ids}",
        "",
        "Feature-family coverage:",
        f"DRA demographic coverage: {dra_rows}/435",
        f"Behavior history available: {behavior_rows}/435",
        (
            "Behavior multi-cycle history available: "
            f"{behavior_multi_cycle_rows}/435"
        ),
        f"Historical elasticity available: {elasticity_rows}/435",
        (
            "Reliable historical elasticity estimates: "
            f"{reliable_elasticity_rows}/435"
        ),
        (
            "Complete master feature rows: "
            f"{complete_rows}/435"
        ),
        "",
        "Behavior reliability:",
        master[
            "selected_regime_reliability"
        ]
        .value_counts(dropna=False)
        .to_string(),
        "",
        "Historical elasticity observation counts:",
        master[
            "historical_elasticity_observation_count"
        ]
        .value_counts(dropna=False)
        .sort_index()
        .to_string(),
        "",
        "District-DNA descriptive feature summary:",
        master[
            [
                "historical_behavior_competitiveness",
                "historical_behavior_volatility_index",
                "historical_behavior_swing_index",
                "historical_behavior_predictability_index",
                "historical_shrunk_elasticity",
                "behavior_reliability_score",
            ]
        ]
        .describe()
        .transpose()
        .to_string(
            float_format=lambda value: f"{value:.4f}"
        ),
        "",
        "Boundary treatment:",
        (
            "Historical behavior metrics are imported from the "
            "boundary-regime-aware behavior warehouse. No configured "
            "mid-decade boundary break is crossed."
        ),
        "",
        "Modeling note:",
        (
            "The descriptive District DNA indexes are not yet production "
            "forecast components. Each must earn inclusion through "
            "out-of-sample model comparison and election backtesting."
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

    return master, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the canonical House district master feature warehouse."
        )
    )

    parser.add_argument(
        "--profile-path",
        type=Path,
        default=DEFAULT_PROFILE_PATH,
    )

    parser.add_argument(
        "--behavior-path",
        type=Path,
        default=DEFAULT_BEHAVIOR_PATH,
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

    args = parser.parse_args()

    for path in [
        args.profile_path,
        args.behavior_path,
        args.elasticity_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required input: {path}"
            )

    profile = pd.read_csv(
        args.profile_path,
        dtype={
            "race_id": str,
            "state": str,
            "district": str,
        },
    )

    behavior = pd.read_csv(
        args.behavior_path,
        dtype={
            "race_id": str,
            "state": str,
            "district": str,
        },
    )

    elasticity = pd.read_csv(
        args.elasticity_path,
        dtype={
            "race_id": str,
        },
    )

    master, report = build_master_features(
        profile=profile,
        behavior=behavior,
        elasticity=elasticity,
    )

    args.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.validation_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    master.to_csv(
        args.output_path,
        index=False,
    )

    args.validation_path.write_text(
        report
    )

    print(report)
    print()
    print(f"Wrote: {args.output_path}")
    print(f"Wrote: {args.validation_path}")


if __name__ == "__main__":
    main()
