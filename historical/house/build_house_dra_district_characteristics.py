from __future__ import annotations

import argparse
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "historical/house/raw/dra_district_data"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "historical/house/warehouse/"
    "house_dra_district_characteristics.csv"
)

DEFAULT_VALIDATION_PATH = (
    PROJECT_ROOT
    / "historical/house/warehouse/"
    "house_dra_district_characteristics_validation.txt"
)


ZIP_NAME_PATTERN = re.compile(
    r"^(?P<state>[A-Z]{2})-"
    r"(?P<map_year>\d{4})-"
    r"Congressional-district-data\.zip$",
    flags=re.IGNORECASE,
)

EXPECTED_DISTRICTS = {
    "MD": 8,
    "TX": 38,
}


def percentage(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    numerator = pd.to_numeric(
        numerator,
        errors="coerce",
    )

    denominator = pd.to_numeric(
        denominator,
        errors="coerce",
    )

    return np.where(
        denominator.gt(0),
        100.0 * numerator / denominator,
        np.nan,
    )


def find_column(
    columns: list[str],
    candidates: list[str],
) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate

    return None


def read_dra_zip(
    zip_path: Path,
) -> tuple[pd.DataFrame, str, str]:
    match = ZIP_NAME_PATTERN.match(
        zip_path.name
    )

    if not match:
        raise ValueError(
            "ZIP filename must follow "
            "ST-YEAR-Congressional-district-data.zip: "
            f"{zip_path.name}"
        )

    state = match.group("state").upper()
    map_year = match.group("map_year")

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())

        required_files = {
            "district-data.csv",
            "README.txt",
            "LICENSE.txt",
        }

        missing_files = sorted(
            required_files - names
        )

        if missing_files:
            raise ValueError(
                f"{zip_path.name} is missing files: "
                + ", ".join(missing_files)
            )

        raw = archive.read(
            "district-data.csv"
        )

    df = pd.read_csv(
        io.BytesIO(raw)
    )

    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
    )

    # DRA includes an unassigned geography row labelled "Un".
    df = df.loc[
        ~df["Label"].str.upper().eq("UN")
    ].copy()

    df["district"] = pd.to_numeric(
        df["Label"],
        errors="coerce",
    )

    if df["district"].isna().any():
        bad_labels = sorted(
            df.loc[
                df["district"].isna(),
                "Label",
            ].unique()
        )

        raise ValueError(
            f"{zip_path.name} contains invalid district labels: "
            f"{bad_labels}"
        )

    df["district"] = (
        df["district"]
        .astype(int)
        .astype(str)
    )

    df["state"] = state
    df["map_year"] = int(map_year)
    df["race_id"] = (
        df["state"]
        + "-"
        + df["district"]
    )

    columns = list(df.columns)

    total_population_column = find_column(
        columns,
        [
            "T_20_CENS_ADJ_Total",
            "T_20_CENS_Total",
            "T_20_ACS_Total",
        ],
    )

    if total_population_column is None:
        raise ValueError(
            f"{zip_path.name} has no supported total-population column."
        )

    if "_CENS_ADJ_" in total_population_column:
        population_dataset = "2020 Census adjusted"
        population_prefix = "T_20_CENS_ADJ"
    elif "_CENS_" in total_population_column:
        population_dataset = "2020 Census"
        population_prefix = "T_20_CENS"
    elif "_ACS_" in total_population_column:
        population_dataset = "ACS"
        population_prefix = "T_20_ACS"
    else:
        raise RuntimeError(
            "Could not identify population dataset."
        )

    required_source_columns = [
        f"{population_prefix}_Total",
        f"{population_prefix}_White",
        f"{population_prefix}_Hispanic",
        f"{population_prefix}_Black",
        f"{population_prefix}_Asian",
        f"{population_prefix}_Native",
        f"{population_prefix}_Pacific",
        "V_20_VAP_Total",
        "V_20_VAP_White",
        "V_20_VAP_Hispanic",
        "V_20_VAP_Black",
        "V_20_VAP_Asian",
        "V_20_VAP_Native",
        "V_20_VAP_Pacific",
        "E_16-20_COMP_Total",
        "E_16-20_COMP_Dem",
        "E_16-20_COMP_Rep",
    ]

    missing_columns = [
        column
        for column in required_source_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{zip_path.name} is missing expected columns: "
            + ", ".join(missing_columns)
        )

    output = pd.DataFrame(
        {
            "race_id": df["race_id"],
            "state": df["state"],
            "district": df["district"],
            "dra_map_year": df["map_year"],
            "dra_population_dataset": population_dataset,
            "dra_source_zip": zip_path.name,
            "total_population": pd.to_numeric(
                df[f"{population_prefix}_Total"],
                errors="coerce",
            ),
            "white_population": pd.to_numeric(
                df[f"{population_prefix}_White"],
                errors="coerce",
            ),
            "hispanic_population": pd.to_numeric(
                df[f"{population_prefix}_Hispanic"],
                errors="coerce",
            ),
            "black_population": pd.to_numeric(
                df[f"{population_prefix}_Black"],
                errors="coerce",
            ),
            "asian_population": pd.to_numeric(
                df[f"{population_prefix}_Asian"],
                errors="coerce",
            ),
            "native_population": pd.to_numeric(
                df[f"{population_prefix}_Native"],
                errors="coerce",
            ),
            "pacific_population": pd.to_numeric(
                df[f"{population_prefix}_Pacific"],
                errors="coerce",
            ),
            "voting_age_population": pd.to_numeric(
                df["V_20_VAP_Total"],
                errors="coerce",
            ),
            "white_vap": pd.to_numeric(
                df["V_20_VAP_White"],
                errors="coerce",
            ),
            "hispanic_vap": pd.to_numeric(
                df["V_20_VAP_Hispanic"],
                errors="coerce",
            ),
            "black_vap": pd.to_numeric(
                df["V_20_VAP_Black"],
                errors="coerce",
            ),
            "asian_vap": pd.to_numeric(
                df["V_20_VAP_Asian"],
                errors="coerce",
            ),
            "native_vap": pd.to_numeric(
                df["V_20_VAP_Native"],
                errors="coerce",
            ),
            "pacific_vap": pd.to_numeric(
                df["V_20_VAP_Pacific"],
                errors="coerce",
            ),
            "dra_composite_total_votes": pd.to_numeric(
                df["E_16-20_COMP_Total"],
                errors="coerce",
            ),
            "dra_composite_dem_votes": pd.to_numeric(
                df["E_16-20_COMP_Dem"],
                errors="coerce",
            ),
            "dra_composite_rep_votes": pd.to_numeric(
                df["E_16-20_COMP_Rep"],
                errors="coerce",
            ),
        }
    )

    output["white_population_share"] = percentage(
        output["white_population"],
        output["total_population"],
    )

    output["hispanic_population_share"] = percentage(
        output["hispanic_population"],
        output["total_population"],
    )

    output["black_population_share"] = percentage(
        output["black_population"],
        output["total_population"],
    )

    output["asian_population_share"] = percentage(
        output["asian_population"],
        output["total_population"],
    )

    output["native_population_share"] = percentage(
        output["native_population"],
        output["total_population"],
    )

    output["pacific_population_share"] = percentage(
        output["pacific_population"],
        output["total_population"],
    )

    output["white_vap_share"] = percentage(
        output["white_vap"],
        output["voting_age_population"],
    )

    output["hispanic_vap_share"] = percentage(
        output["hispanic_vap"],
        output["voting_age_population"],
    )

    output["black_vap_share"] = percentage(
        output["black_vap"],
        output["voting_age_population"],
    )

    output["asian_vap_share"] = percentage(
        output["asian_vap"],
        output["voting_age_population"],
    )

    output["native_vap_share"] = percentage(
        output["native_vap"],
        output["voting_age_population"],
    )

    output["pacific_vap_share"] = percentage(
        output["pacific_vap"],
        output["voting_age_population"],
    )

    two_party_votes = (
        output["dra_composite_dem_votes"]
        + output["dra_composite_rep_votes"]
    )

    output["dra_composite_dem_two_party_share"] = percentage(
        output["dra_composite_dem_votes"],
        two_party_votes,
    )

    output["dra_composite_margin_dem"] = np.where(
        two_party_votes.gt(0),
        100.0
        * (
            output["dra_composite_dem_votes"]
            - output["dra_composite_rep_votes"]
        )
        / two_party_votes,
        np.nan,
    )

    return output, state, population_dataset


def build_dataset(
    input_dir: Path,
) -> tuple[pd.DataFrame, str]:
    zip_paths = sorted(
        input_dir.glob(
            "*-Congressional-district-data.zip"
        )
    )

    if not zip_paths:
        raise FileNotFoundError(
            f"No DRA ZIP exports found in {input_dir}."
        )

    frames: list[pd.DataFrame] = []
    state_summaries: list[dict[str, object]] = []

    for zip_path in zip_paths:
        frame, state, population_dataset = (
            read_dra_zip(zip_path)
        )

        expected = EXPECTED_DISTRICTS.get(state)

        if expected is not None and len(frame) != expected:
            raise ValueError(
                f"{state} expected {expected} districts; "
                f"found {len(frame)}."
            )

        frames.append(frame)

        state_summaries.append(
            {
                "state": state,
                "districts": len(frame),
                "map_year": int(
                    frame["dra_map_year"].iloc[0]
                ),
                "population_dataset": population_dataset,
                "source_zip": zip_path.name,
            }
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined = combined.sort_values(
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

    duplicate_race_ids = int(
        combined["race_id"].duplicated().sum()
    )

    missing_population = int(
        combined["total_population"].isna().sum()
    )

    invalid_population = int(
        combined["total_population"].le(0).sum()
    )

    share_columns = [
        column
        for column in combined.columns
        if column.endswith("_share")
    ]

    invalid_share_rows = int(
        (
            combined[share_columns].lt(0)
            | combined[share_columns].gt(100)
        ).any(axis=1).sum()
    )

    failures: list[str] = []

    if duplicate_race_ids:
        failures.append(
            f"Found {duplicate_race_ids} duplicate race IDs."
        )

    if missing_population:
        failures.append(
            f"Found {missing_population} missing populations."
        )

    if invalid_population:
        failures.append(
            f"Found {invalid_population} nonpositive populations."
        )

    if invalid_share_rows:
        failures.append(
            f"Found {invalid_share_rows} rows with shares outside 0–100."
        )

    state_summary = pd.DataFrame(
        state_summaries
    )

    report_lines = [
        "House DRA District Characteristics Validation",
        "=" * 45,
        "",
        f"ZIP exports processed: {len(zip_paths)}",
        f"District rows: {len(combined)}",
        f"Unique race IDs: {combined['race_id'].nunique()}",
        f"Duplicate race IDs: {duplicate_race_ids}",
        f"Missing total populations: {missing_population}",
        f"Nonpositive total populations: {invalid_population}",
        f"Rows with invalid percentage shares: {invalid_share_rows}",
        "",
        "States processed:",
        state_summary.to_string(index=False),
        "",
        "Population dataset warning:",
        (
            "DRA exports may use different total-population datasets "
            "by state or map. The dra_population_dataset field must be "
            "preserved, and adjusted and unadjusted totals should not "
            "be treated as perfectly interchangeable."
        ),
        "",
        "License note:",
        (
            "Raw DRA exports include CC BY-SA 4.0 licensing and a "
            "prohibition on selling the data. Preserve each ZIP, "
            "README.txt, and LICENSE.txt."
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
            "Import Dave's Redistricting district-data ZIP exports "
            "into a canonical House characteristics dataset."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
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

    dataset, report = build_dataset(
        args.input_dir
    )

    args.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.validation_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_csv(
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
