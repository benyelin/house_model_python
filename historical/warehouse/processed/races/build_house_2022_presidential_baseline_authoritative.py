from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]

RAW_PATH = (
    PROJECT_ROOT
    / "historical/house/raw/2022/source_downloads/"
    "presidential_baselines/"
    "2020_presidential_results_by_2022_cd.csv"
)

MASTER_PATH = (
    PROJECT_ROOT
    / "historical/house/master/house_2022_master.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "historical/warehouse/processed/races/"
    "house_2022_presidential_baseline.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "historical/warehouse/validation/"
    "house_2022_presidential_baseline_validation.txt"
)


def normalize_race_id(value: object) -> str:
    text = str(value).strip().upper()

    if "-" not in text:
        raise ValueError(
            f"Invalid district identifier: {value}"
        )

    state, district = text.split("-", 1)

    if district == "AL":
        return f"{state}-AL"

    district_number = int(district)

    return f"{state}-{district_number}"


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Missing raw baseline file: {RAW_PATH}"
        )

    if not MASTER_PATH.exists():
        raise FileNotFoundError(
            f"Missing historical master: {MASTER_PATH}"
        )

    raw = pd.read_csv(
        RAW_PATH,
        dtype={"District": str},
    )

    master = pd.read_csv(
        MASTER_PATH,
        dtype={"race_id": str},
    )

    required_columns = [
        "District",
        "Biden",
        "Trump",
        "Margin",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in raw.columns
    ]

    if missing_columns:
        raise ValueError(
            "Raw presidential baseline file is missing: "
            + ", ".join(missing_columns)
        )

    baseline = raw[
        [
            "District",
            "Incumbent",
            "Party",
            "Biden",
            "Trump",
            "Margin",
        ]
    ].copy()

    baseline["race_id"] = baseline[
        "District"
    ].apply(normalize_race_id)

    baseline["biden_vote_share"] = pd.to_numeric(
        baseline["Biden"],
        errors="coerce",
    )

    baseline["trump_vote_share"] = pd.to_numeric(
        baseline["Trump"],
        errors="coerce",
    )

    baseline["district_pres_margin_dem"] = pd.to_numeric(
        baseline["Margin"],
        errors="coerce",
    )

    baseline = baseline.rename(
        columns={
            "Incumbent": "source_incumbent",
            "Party": "source_incumbent_party",
        }
    )

    baseline["presidential_result_year"] = 2020
    baseline["boundary_cycle"] = 2022
    baseline["boundary_compatibility"] = (
        "authoritative_2022_boundary"
    )
    baseline["include_in_2022_backtest"] = True
    baseline["source_organization"] = (
        "Daily Kos Elections / The Downballot"
    )
    baseline["source_local_path"] = str(
        RAW_PATH.relative_to(PROJECT_ROOT)
    )
    baseline["source_description"] = (
        "2020 presidential election results calculated "
        "for congressional districts used in the 2022 elections"
    )

    keep_columns = [
        "race_id",
        "source_incumbent",
        "source_incumbent_party",
        "biden_vote_share",
        "trump_vote_share",
        "district_pres_margin_dem",
        "presidential_result_year",
        "boundary_cycle",
        "boundary_compatibility",
        "include_in_2022_backtest",
        "source_organization",
        "source_local_path",
        "source_description",
    ]

    baseline = baseline[
        keep_columns
    ].copy()

    if len(baseline) != 435:
        raise ValueError(
            f"Expected 435 baseline rows; found {len(baseline)}."
        )

    if baseline["race_id"].duplicated().any():
        duplicates = baseline.loc[
            baseline["race_id"].duplicated(False),
            "race_id",
        ].tolist()

        raise ValueError(
            "Duplicate presidential baseline race IDs: "
            + ", ".join(duplicates)
        )

    comparison = master[
        ["race_id"]
    ].merge(
        baseline,
        on="race_id",
        how="left",
        validate="one_to_one",
    )

    failed_joins = comparison.loc[
        comparison["district_pres_margin_dem"].isna(),
        ["race_id"],
    ]

    if not failed_joins.empty:
        raise ValueError(
            "Historical master races missing baseline joins:\n"
            + failed_joins.to_string(index=False)
        )

    extra_baselines = baseline.loc[
        ~baseline["race_id"].isin(
            master["race_id"]
        ),
        ["race_id"],
    ]

    if not extra_baselines.empty:
        raise ValueError(
            "Baseline contains race IDs absent from master:\n"
            + extra_baselines.to_string(index=False)
        )

    margin_residual = (
        comparison["biden_vote_share"]
        - comparison["trump_vote_share"]
        - comparison["district_pres_margin_dem"]
    )

    max_margin_residual = float(
        margin_residual.abs().max()
    )

    if max_margin_residual > 0.11:
        raise ValueError(
            "Published margin does not match Biden minus Trump. "
            f"Maximum residual: {max_margin_residual}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    VALIDATION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    validation_lines = [
        "2022 House Presidential Baseline Validation",
        "=" * 43,
        "",
        f"Rows: {len(comparison)}",
        f"Unique race IDs: {comparison['race_id'].nunique()}",
        (
            "Numeric Democratic margins: "
            f"{comparison['district_pres_margin_dem'].notna().sum()}"
        ),
        f"Failed joins: {len(failed_joins)}",
        (
            "Maximum Biden-minus-Trump margin residual: "
            f"{max_margin_residual:.6f}"
        ),
        (
            "Minimum Democratic presidential margin: "
            f"{comparison['district_pres_margin_dem'].min():+.1f}"
        ),
        (
            "Maximum Democratic presidential margin: "
            f"{comparison['district_pres_margin_dem'].max():+.1f}"
        ),
        "",
        "Boundary status:",
        (
            "All values are calculated on congressional boundaries "
            "used in the 2022 House election."
        ),
        "",
        "Validation status:",
        "PASSED",
    ]

    validation_text = "\n".join(
        validation_lines
    )

    VALIDATION_PATH.write_text(
        validation_text
    )

    print(validation_text)
    print()
    print(comparison.head(15).to_string(index=False))
    print()
    print(f"Wrote: {OUTPUT_PATH}")
    print(f"Wrote: {VALIDATION_PATH}")


if __name__ == "__main__":
    main()
