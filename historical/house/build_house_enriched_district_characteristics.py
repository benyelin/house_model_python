from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CHARACTERISTICS_PATH = (
    PROJECT_ROOT
    / "historical/house/warehouse/"
    "house_district_characteristics.csv"
)

DEFAULT_DRA_PATH = (
    PROJECT_ROOT
    / "historical/house/warehouse/"
    "house_dra_district_characteristics.csv"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "historical/house/warehouse/"
    "house_enriched_district_characteristics.csv"
)

DEFAULT_VALIDATION_PATH = (
    PROJECT_ROOT
    / "historical/house/warehouse/"
    "house_enriched_district_characteristics_validation.txt"
)


EXPECTED_DRA_MISSING_RACE_IDS = {
    "AK-AL",
    "DE-AL",
    "ND-AL",
    "SD-AL",
    "VT-AL",
    "WY-AL",
}


DRA_COLUMNS = [
    "dra_map_year",
    "dra_population_dataset",
    "dra_source_zip",
    "total_population",
    "white_population",
    "hispanic_population",
    "black_population",
    "asian_population",
    "native_population",
    "pacific_population",
    "voting_age_population",
    "white_vap",
    "hispanic_vap",
    "black_vap",
    "asian_vap",
    "native_vap",
    "pacific_vap",
    "dra_composite_total_votes",
    "dra_composite_dem_votes",
    "dra_composite_rep_votes",
    "white_population_share",
    "hispanic_population_share",
    "black_population_share",
    "asian_population_share",
    "native_population_share",
    "pacific_population_share",
    "white_vap_share",
    "hispanic_vap_share",
    "black_vap_share",
    "asian_vap_share",
    "native_vap_share",
    "pacific_vap_share",
    "dra_composite_dem_two_party_share",
    "dra_composite_margin_dem",
]


def build_enriched_characteristics(
    characteristics: pd.DataFrame,
    dra: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    required_characteristics = {
        "race_id",
        "state",
        "district",
        "region",
        "district_type",
        "pres_2020_margin_dem",
        "pres_2024_margin_dem",
    }

    missing_characteristics = sorted(
        required_characteristics
        - set(characteristics.columns)
    )

    if missing_characteristics:
        raise ValueError(
            "Canonical characteristics table is missing columns: "
            + ", ".join(missing_characteristics)
        )

    required_dra = {
        "race_id",
        *DRA_COLUMNS,
    }

    missing_dra = sorted(
        required_dra - set(dra.columns)
    )

    if missing_dra:
        raise ValueError(
            "DRA characteristics table is missing columns: "
            + ", ".join(missing_dra)
        )

    if len(characteristics) != 435:
        raise ValueError(
            "Expected 435 canonical characteristic rows; "
            f"found {len(characteristics)}."
        )

    if len(dra) != 429:
        raise ValueError(
            "Expected 429 mapped-state DRA rows; "
            f"found {len(dra)}."
        )

    if characteristics["race_id"].duplicated().any():
        raise ValueError(
            "Duplicate race IDs found in canonical characteristics."
        )

    if dra["race_id"].duplicated().any():
        raise ValueError(
            "Duplicate race IDs found in DRA characteristics."
        )

    dra_keep = dra[
        [
            "race_id",
            *DRA_COLUMNS,
        ]
    ].copy()

    enriched = characteristics.merge(
        dra_keep,
        on="race_id",
        how="left",
        validate="one_to_one",
    )

    enriched["dra_data_available"] = (
        enriched["total_population"].notna()
        & enriched["voting_age_population"].notna()
        & enriched["dra_composite_margin_dem"].notna()
    )

    enriched["dra_data_status"] = np.where(
        enriched["dra_data_available"],
        "available",
        "at_large_state_fallback_pending",
    )

    enriched["presidential_swing_2020_to_2024_dem"] = (
        pd.to_numeric(
            enriched["pres_2024_margin_dem"],
            errors="coerce",
        )
        - pd.to_numeric(
            enriched["pres_2020_margin_dem"],
            errors="coerce",
        )
    )

    enriched["dra_composite_minus_2020_margin_dem"] = (
        pd.to_numeric(
            enriched["dra_composite_margin_dem"],
            errors="coerce",
        )
        - pd.to_numeric(
            enriched["pres_2020_margin_dem"],
            errors="coerce",
        )
    )

    enriched["dra_composite_minus_2024_margin_dem"] = (
        pd.to_numeric(
            enriched["dra_composite_margin_dem"],
            errors="coerce",
        )
        - pd.to_numeric(
            enriched["pres_2024_margin_dem"],
            errors="coerce",
        )
    )

    duplicate_race_ids = int(
        enriched["race_id"].duplicated().sum()
    )

    available_rows = int(
        enriched["dra_data_available"].sum()
    )

    missing_rows = int(
        (~enriched["dra_data_available"]).sum()
    )

    missing_ids = set(
        enriched.loc[
            ~enriched["dra_data_available"],
            "race_id",
        ]
    )

    invalid_share_rows = int(
        (
            enriched[
                [
                    "white_population_share",
                    "hispanic_population_share",
                    "black_population_share",
                    "asian_population_share",
                    "native_population_share",
                    "pacific_population_share",
                    "white_vap_share",
                    "hispanic_vap_share",
                    "black_vap_share",
                    "asian_vap_share",
                    "native_vap_share",
                    "pacific_vap_share",
                    "dra_composite_dem_two_party_share",
                ]
            ].lt(0)
            | enriched[
                [
                    "white_population_share",
                    "hispanic_population_share",
                    "black_population_share",
                    "asian_population_share",
                    "native_population_share",
                    "pacific_population_share",
                    "white_vap_share",
                    "hispanic_vap_share",
                    "black_vap_share",
                    "asian_vap_share",
                    "native_vap_share",
                    "pacific_vap_share",
                    "dra_composite_dem_two_party_share",
                ]
            ].gt(100)
        ).any(axis=1).sum()
    )

    failures: list[str] = []

    if len(enriched) != 435:
        failures.append(
            f"Expected 435 enriched rows; found {len(enriched)}."
        )

    if duplicate_race_ids:
        failures.append(
            f"Found {duplicate_race_ids} duplicate race IDs."
        )

    if available_rows != 429:
        failures.append(
            f"Expected 429 DRA-covered districts; found {available_rows}."
        )

    if missing_rows != 6:
        failures.append(
            f"Expected 6 DRA-missing districts; found {missing_rows}."
        )

    if missing_ids != EXPECTED_DRA_MISSING_RACE_IDS:
        failures.append(
            "DRA-missing race IDs do not match the six expected "
            "at-large districts."
        )

    if invalid_share_rows:
        failures.append(
            f"Found {invalid_share_rows} rows with invalid shares."
        )

    population_dataset_counts = (
        enriched.loc[
            enriched["dra_data_available"],
            "dra_population_dataset",
        ]
        .value_counts(dropna=False)
    )

    report_lines = [
        "House Enriched District Characteristics Validation",
        "=" * 48,
        "",
        f"Rows: {len(enriched)}",
        f"Unique race IDs: {enriched['race_id'].nunique()}",
        f"Duplicate race IDs: {duplicate_race_ids}",
        f"DRA-covered districts: {available_rows}",
        f"DRA-missing districts: {missing_rows}",
        (
            "DRA-missing race IDs: "
            + ", ".join(sorted(missing_ids))
        ),
        f"Rows with invalid percentage shares: {invalid_share_rows}",
        "",
        "DRA population datasets:",
        population_dataset_counts.to_string(),
        "",
        "Continuous feature coverage:",
    ]

    feature_columns = [
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
        "dra_composite_margin_dem",
        "presidential_swing_2020_to_2024_dem",
        "dra_composite_minus_2020_margin_dem",
        "dra_composite_minus_2024_margin_dem",
    ]

    for column in feature_columns:
        report_lines.append(
            f"{column}: "
            f"{int(enriched[column].notna().sum())}/435"
        )

    report_lines.extend(
        [
            "",
            "Known limitation:",
            (
                "Six at-large districts do not have DRA map exports. "
                "They remain in the warehouse with explicit missing "
                "DRA values and should be excluded or imputed during "
                "model-specific preprocessing."
            ),
            "",
            "Validation status:",
        ]
    )

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

    enriched = enriched.sort_values(
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

    return enriched, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge canonical House district characteristics with "
            "continuous Dave's Redistricting App features."
        )
    )

    parser.add_argument(
        "--characteristics-path",
        type=Path,
        default=DEFAULT_CHARACTERISTICS_PATH,
    )

    parser.add_argument(
        "--dra-path",
        type=Path,
        default=DEFAULT_DRA_PATH,
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

    if not args.characteristics_path.exists():
        raise FileNotFoundError(
            "Missing canonical characteristics table: "
            f"{args.characteristics_path}"
        )

    if not args.dra_path.exists():
        raise FileNotFoundError(
            f"Missing DRA characteristics table: {args.dra_path}"
        )

    characteristics = pd.read_csv(
        args.characteristics_path,
        dtype={
            "race_id": str,
            "state": str,
            "district": str,
        },
    )

    dra = pd.read_csv(
        args.dra_path,
        dtype={
            "race_id": str,
            "state": str,
            "district": str,
        },
    )

    enriched, report = build_enriched_characteristics(
        characteristics=characteristics,
        dra=dra,
    )

    args.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.validation_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    enriched.to_csv(
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
