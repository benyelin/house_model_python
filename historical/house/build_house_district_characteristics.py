from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "inputs"
    / "House Model Data.xlsx"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "warehouse"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "house_district_characteristics.csv"
)

VALIDATION_PATH = (
    OUTPUT_DIR
    / "house_district_characteristics_validation.txt"
)


AT_LARGE_STATES = {
    "AK",
    "DE",
    "DC",
    "ND",
    "SD",
    "VT",
    "WY",
}


def normalize_district(value: object) -> str:
    """
    Convert district labels into canonical House IDs.

    Examples:
        1  -> "1"
        AL -> "AL"
        0  -> "AL" (at-large states only)
    """
    if pd.isna(value):
        return ""

    text = str(value).strip().upper()

    if text in {"AL", "AT LARGE", "AT-LARGE"}:
        return "AL"

    try:
        number = int(float(text))
    except ValueError:
        return text

    if number == 0:
        return "AL"

    return str(number)


def build_characteristics(input_path: Path) -> pd.DataFrame:

    df = pd.read_excel(
        input_path,
        sheet_name="House Model Data",
    )

    df["state"] = (
        df["State"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["district"] = (
        df["District"]
        .apply(normalize_district)
    )

    #
    # Force canonical AL labels
    #
    df.loc[
        df["state"].isin(AT_LARGE_STATES),
        "district",
    ] = "AL"

    df["race_id"] = (
        df["state"]
        + "-"
        + df["district"]
    )

    output = pd.DataFrame(
        {
            "race_id": df["race_id"],
            "state": df["state"],
            "district": df["district"],
            "region": df["Region"],
            "district_type": df["District Type"],
            "college_share_tier": df["College Share Tier"],
            "white_share_tier": df["White Share Tier"],
            "black_share_tier": df["Black Share Tier"],
            "hispanic_share_tier": df["Hispanic Share Tier"],
            "median_income_tier": df["Median Income Tier"],
            "pres_2020_margin_dem": pd.to_numeric(
                df["2020 Margin"],
                errors="coerce",
            ),
            "pres_2024_margin_dem": pd.to_numeric(
                df["2024 Margin"],
                errors="coerce",
            ),
        }
    )

    return output


def validate(df: pd.DataFrame) -> str:

    failures: list[str] = []
    warnings: list[str] = []

    if len(df) != 435:
        failures.append(
            f"Expected 435 rows; found {len(df)}."
        )

    duplicate_count = int(
        df["race_id"].duplicated().sum()
    )

    if duplicate_count:
        failures.append(
            f"Found {duplicate_count} duplicate race IDs."
        )

    missing_2020 = int(
        df["pres_2020_margin_dem"].isna().sum()
    )

    if missing_2020:
        failures.append(
            f"{missing_2020} districts missing 2020 margin."
        )

    missing_2024 = int(
        df["pres_2024_margin_dem"].isna().sum()
    )

    if missing_2024:
        warnings.append(
            f"{missing_2024} districts missing 2024 margin."
        )

    tier_columns = [
        "college_share_tier",
        "white_share_tier",
        "black_share_tier",
        "hispanic_share_tier",
        "median_income_tier",
    ]

    report = [
        "House District Characteristics Validation",
        "========================================",
        "",
        f"Rows: {len(df)}",
        f"Unique race IDs: {df['race_id'].nunique()}",
        f"Duplicate race IDs: {duplicate_count}",
        f"Missing 2020 margins: {missing_2020}",
        f"Missing 2024 margins: {missing_2024}",
        "",
        "Missing characteristic values:",
    ]

    for column in tier_columns:
        missing = int(df[column].isna().sum())

        report.append(
            f"{column}: {missing}"
        )

    report.append("")
    report.append("Validation status:")

    if failures:
        report.append("FAILED")
        report.extend(f"- {x}" for x in failures)
    else:
        report.append("PASSED")

    if warnings:
        report.append("")
        report.append("Warnings:")
        report.extend(f"- {x}" for x in warnings)

    return "\n".join(report)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
    )

    args = parser.parse_args()

    warehouse = build_characteristics(
        args.input_path,
    )

    report = validate(
        warehouse,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    warehouse.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    VALIDATION_PATH.write_text(
        report,
    )

    print(report)
    print()
    print(f"Wrote: {OUTPUT_PATH}")
    print(f"Wrote: {VALIDATION_PATH}")


if __name__ == "__main__":
    main()
