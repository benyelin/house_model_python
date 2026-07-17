from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "historical/house/warehouse/"
    "house_enriched_district_characteristics.csv"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "historical/house/warehouse/"
    "house_structural_profile.csv"
)

DEFAULT_VALIDATION_PATH = (
    PROJECT_ROOT
    / "historical/house/warehouse/"
    "house_structural_profile_validation.txt"
)


EXPECTED_AT_LARGE_MISSING_DRA = {
    "AK-AL",
    "DE-AL",
    "ND-AL",
    "SD-AL",
    "VT-AL",
    "WY-AL",
}


IDENTIFIER_COLUMNS = [
    "race_id",
    "state",
    "district",
]

CATEGORICAL_COLUMNS = [
    "region",
    "district_type",
    "dra_population_dataset",
    "dra_data_status",
]

DEMOGRAPHIC_COLUMNS = [
    "total_population",
    "voting_age_population",
    "white_population_share",
    "black_population_share",
    "hispanic_population_share",
    "asian_population_share",
    "native_population_share",
    "pacific_population_share",
    "white_vap_share",
    "black_vap_share",
    "hispanic_vap_share",
    "asian_vap_share",
    "native_vap_share",
    "pacific_vap_share",
]

POLITICAL_SOURCE_COLUMNS = [
    "pres_2020_margin_dem",
    "pres_2024_margin_dem",
    "dra_composite_margin_dem",
    "dra_composite_dem_two_party_share",
]

PROVENANCE_COLUMNS = [
    "dra_map_year",
    "dra_source_zip",
    "dra_data_available",
]


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


def clean_category(
    series: pd.Series,
) -> pd.Series:
    return (
        series.fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )


def build_structural_profile(
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    required_columns = {
        *IDENTIFIER_COLUMNS,
        *CATEGORICAL_COLUMNS,
        *DEMOGRAPHIC_COLUMNS,
        *POLITICAL_SOURCE_COLUMNS,
        *PROVENANCE_COLUMNS,
    }

    missing_columns = sorted(
        required_columns - set(source.columns)
    )

    if missing_columns:
        raise ValueError(
            "Enriched characteristics table is missing columns: "
            + ", ".join(missing_columns)
        )

    if len(source) != 435:
        raise ValueError(
            f"Expected 435 source rows; found {len(source)}."
        )

    if source["race_id"].duplicated().any():
        raise ValueError(
            "Duplicate race IDs found in source data."
        )

    profile = source[
        [
            *IDENTIFIER_COLUMNS,
            *CATEGORICAL_COLUMNS,
            *DEMOGRAPHIC_COLUMNS,
            *POLITICAL_SOURCE_COLUMNS,
            *PROVENANCE_COLUMNS,
        ]
    ].copy()

    profile["dra_data_available"] = parse_bool_series(
        profile["dra_data_available"]
    )

    for column in CATEGORICAL_COLUMNS:
        profile[column] = clean_category(
            profile[column]
        )

    numeric_columns = [
        *DEMOGRAPHIC_COLUMNS,
        *POLITICAL_SOURCE_COLUMNS,
        "dra_map_year",
    ]

    for column in numeric_columns:
        profile[column] = pd.to_numeric(
            profile[column],
            errors="coerce",
        )

    # -----------------------------------------------------------------
    # Presidential structural features
    # -----------------------------------------------------------------

    profile["presidential_margin_average_2020_2024_dem"] = (
        profile[
            [
                "pres_2020_margin_dem",
                "pres_2024_margin_dem",
            ]
        ].mean(axis=1)
    )

    profile["presidential_swing_2020_to_2024_dem"] = (
        profile["pres_2024_margin_dem"]
        - profile["pres_2020_margin_dem"]
    )

    profile["presidential_margin_range_2020_2024"] = (
        profile[
            [
                "pres_2020_margin_dem",
                "pres_2024_margin_dem",
            ]
        ].max(axis=1)
        - profile[
            [
                "pres_2020_margin_dem",
                "pres_2024_margin_dem",
            ]
        ].min(axis=1)
    )

    profile["presidential_average_absolute_margin"] = (
        profile[
            [
                "pres_2020_margin_dem",
                "pres_2024_margin_dem",
            ]
        ]
        .abs()
        .mean(axis=1)
    )

    profile["presidential_baseline_competitiveness"] = (
        100.0
        - profile[
            "presidential_average_absolute_margin"
        ]
    ).clip(
        lower=0.0,
        upper=100.0,
    )

    profile["presidential_party_lean"] = np.select(
        [
            profile[
                "presidential_margin_average_2020_2024_dem"
            ].gt(0),
            profile[
                "presidential_margin_average_2020_2024_dem"
            ].lt(0),
        ],
        [
            "Democratic",
            "Republican",
        ],
        default="Even",
    )

    profile["presidential_trend_direction"] = np.select(
        [
            profile[
                "presidential_swing_2020_to_2024_dem"
            ].gt(0.5),
            profile[
                "presidential_swing_2020_to_2024_dem"
            ].lt(-0.5),
        ],
        [
            "Democratic",
            "Republican",
        ],
        default="Stable",
    )

    # -----------------------------------------------------------------
    # DRA composite structural features
    # -----------------------------------------------------------------

    profile["dra_composite_minus_presidential_average_dem"] = (
        profile["dra_composite_margin_dem"]
        - profile[
            "presidential_margin_average_2020_2024_dem"
        ]
    )

    profile["dra_composite_absolute_margin"] = (
        profile["dra_composite_margin_dem"].abs()
    )

    profile["dra_composite_competitiveness"] = (
        100.0
        - profile[
            "dra_composite_absolute_margin"
        ]
    ).clip(
        lower=0.0,
        upper=100.0,
    )

    # -----------------------------------------------------------------
    # Initial structural-profile metadata
    # -----------------------------------------------------------------

    profile["structural_profile_version"] = "1.0"

    profile["structural_profile_stage"] = (
        "foundation_existing_validated_sources"
    )

    profile["political_baseline_method"] = (
        "mean of 2020 and 2024 Democratic presidential margins"
    )

    profile["demographic_source"] = np.where(
        profile["dra_data_available"],
        "Dave's Redistricting district-data export",
        "DRA unavailable for at-large district",
    )

    profile["structural_profile_notes"] = np.where(
        profile["dra_data_available"],
        (
            "Foundation profile includes DRA demographics, "
            "DRA composite election margin, and presidential results."
        ),
        (
            "At-large district retained without DRA demographic or "
            "composite-election data; future state-level fallback pending."
        ),
    )

    profile = profile.sort_values(
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

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    duplicate_race_ids = int(
        profile["race_id"].duplicated().sum()
    )

    dra_available_rows = int(
        profile["dra_data_available"].sum()
    )

    dra_missing_rows = int(
        (~profile["dra_data_available"]).sum()
    )

    dra_missing_ids = set(
        profile.loc[
            ~profile["dra_data_available"],
            "race_id",
        ]
    )

    missing_presidential_baselines = int(
        profile[
            "presidential_margin_average_2020_2024_dem"
        ].isna().sum()
    )

    missing_dra_composite = int(
        profile["dra_composite_margin_dem"]
        .isna()
        .sum()
    )

    share_columns = [
        column
        for column in DEMOGRAPHIC_COLUMNS
        if column.endswith("_share")
    ]

    invalid_share_rows = int(
        (
            profile[share_columns].lt(0)
            | profile[share_columns].gt(100)
        ).any(axis=1).sum()
    )

    failures: list[str] = []

    if len(profile) != 435:
        failures.append(
            f"Expected 435 profile rows; found {len(profile)}."
        )

    if profile["race_id"].nunique() != 435:
        failures.append(
            "Expected 435 unique race IDs."
        )

    if duplicate_race_ids:
        failures.append(
            f"Found {duplicate_race_ids} duplicate race IDs."
        )

    if dra_available_rows != 429:
        failures.append(
            "Expected 429 DRA-covered districts; "
            f"found {dra_available_rows}."
        )

    if dra_missing_rows != 6:
        failures.append(
            "Expected 6 DRA-missing districts; "
            f"found {dra_missing_rows}."
        )

    if dra_missing_ids != EXPECTED_AT_LARGE_MISSING_DRA:
        failures.append(
            "DRA-missing districts do not match the expected "
            "six at-large race IDs."
        )

    if missing_presidential_baselines:
        failures.append(
            "Found "
            f"{missing_presidential_baselines} missing presidential "
            "structural baselines."
        )

    if missing_dra_composite != 6:
        failures.append(
            "Expected six missing DRA composite margins; "
            f"found {missing_dra_composite}."
        )

    if invalid_share_rows:
        failures.append(
            f"Found {invalid_share_rows} rows with invalid shares."
        )

    report_lines = [
        "House Structural District Profile Validation",
        "=" * 44,
        "",
        f"Rows: {len(profile)}",
        f"Unique race IDs: {profile['race_id'].nunique()}",
        f"Duplicate race IDs: {duplicate_race_ids}",
        f"DRA-covered districts: {dra_available_rows}",
        f"DRA-missing districts: {dra_missing_rows}",
        (
            "DRA-missing race IDs: "
            + ", ".join(sorted(dra_missing_ids))
        ),
        (
            "Missing presidential structural baselines: "
            f"{missing_presidential_baselines}"
        ),
        (
            "Missing DRA composite margins: "
            f"{missing_dra_composite}"
        ),
        f"Rows with invalid percentage shares: {invalid_share_rows}",
        "",
        "Presidential structural-feature summary:",
        profile[
            [
                "presidential_margin_average_2020_2024_dem",
                "presidential_swing_2020_to_2024_dem",
                "presidential_average_absolute_margin",
                "presidential_baseline_competitiveness",
            ]
        ]
        .describe()
        .transpose()
        .to_string(
            float_format=lambda value: f"{value:.4f}"
        ),
        "",
        "Presidential party-lean counts:",
        profile[
            "presidential_party_lean"
        ]
        .value_counts()
        .to_string(),
        "",
        "Presidential trend-direction counts:",
        profile[
            "presidential_trend_direction"
        ]
        .value_counts()
        .to_string(),
        "",
        "Profile stage:",
        (
            "Foundation using existing validated DRA and presidential "
            "data. Future layers will add geography, historical House "
            "performance, volatility, and party-switch measures."
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

    return profile, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the canonical House structural district profile "
            "from validated district characteristics."
        )
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
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

    if not args.input_path.exists():
        raise FileNotFoundError(
            f"Missing enriched characteristics file: {args.input_path}"
        )

    source = pd.read_csv(
        args.input_path,
        dtype={
            "race_id": str,
            "state": str,
            "district": str,
        },
    )

    profile, report = build_structural_profile(
        source
    )

    args.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.validation_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile.to_csv(
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
