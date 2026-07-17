from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SOURCE_DIR = (
    PROJECT_ROOT
    / "historical/house/raw/shared/source_downloads/"
    "presidential_by_cd"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "historical/house/warehouse/"
    "house_presidential_results_by_boundary.csv"
)

DEFAULT_BASELINE_OUTPUT_PATH = (
    PROJECT_ROOT
    / "historical/house/warehouse/"
    "house_historical_presidential_baselines.csv"
)

DEFAULT_VALIDATION_PATH = (
    PROJECT_ROOT
    / "historical/house/warehouse/"
    "house_presidential_results_by_boundary_validation.txt"
)


@dataclass(frozen=True)
class SourceSpec:
    boundary_cycle: int
    filename: str
    result_years: tuple[int, ...]


SOURCE_SPECS = (
    SourceSpec(
        boundary_cycle=2016,
        filename="presidential_results_on_2016_districts.csv",
        result_years=(2008, 2012, 2016),
    ),
    SourceSpec(
        boundary_cycle=2018,
        filename="presidential_results_on_2018_districts.csv",
        result_years=(2008, 2012, 2016),
    ),
    SourceSpec(
        boundary_cycle=2020,
        filename="presidential_results_on_2020_districts.csv",
        result_years=(2008, 2012, 2016, 2020),
    ),
)

CANDIDATES_BY_YEAR = {
    2008: ("Obama", "McCain"),
    2012: ("Obama", "Romney"),
    2016: ("Clinton", "Trump"),
    2020: ("Biden", "Trump"),
}

FORECAST_BASELINE_PLAN = {
    2016: {
        "boundary_cycle": 2016,
        "presidential_result_year": 2012,
    },
    2018: {
        "boundary_cycle": 2018,
        "presidential_result_year": 2016,
    },
    2020: {
        "boundary_cycle": 2020,
        "presidential_result_year": 2016,
    },
}


def normalize_race_id(value: object) -> str:
    text = str(value).strip().upper()

    text = (
        text.replace("–", "-")
        .replace("—", "-")
        .replace("_", "-")
        .replace(" ", "")
    )

    # Accept any state-level at-large district identifier.
    # Montana was also at-large during the 2016, 2018, and 2020
    # boundary cycles, so a hard-coded six-state list is incomplete.
    at_large_match = re.fullmatch(
        r"([A-Z]{2})-AL",
        text,
    )

    if at_large_match:
        return f"{at_large_match.group(1)}-AL"

    match = re.fullmatch(
        r"([A-Z]{2})-(?:CD)?0*(\d+)",
        text,
    )

    if not match:
        raise ValueError(
            f"Could not normalize district identifier: {value!r}"
        )

    state, district = match.groups()

    return f"{state}-{int(district)}"


def read_source_table(
    path: Path,
) -> pd.DataFrame:
    """
    These CSV exports contain two header rows:

        row 1: year group labels
        row 2: CD / candidate labels

    Reading with header=None preserves both rows and avoids pandas'
    automatic duplicate-column renaming.
    """
    raw = pd.read_csv(
        path,
        header=None,
        dtype=str,
        low_memory=False,
    )

    if len(raw) != 437:
        raise ValueError(
            f"{path.name}: expected 437 physical rows "
            f"(2 headers + 435 districts); found {len(raw)}."
        )

    return raw


def locate_year_columns(
    raw: pd.DataFrame,
    expected_years: tuple[int, ...],
) -> dict[int, tuple[int, int]]:
    year_row = (
        raw.iloc[0]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    candidate_row = (
        raw.iloc[1]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    located: dict[int, tuple[int, int]] = {}

    for year in expected_years:
        dem_candidate, gop_candidate = (
            CANDIDATES_BY_YEAR[year]
        )

        year_columns = [
            int(column)
            for column, value in year_row.items()
            if value == str(year)
        ]

        if len(year_columns) != 1:
            raise ValueError(
                f"Expected one starting column for {year}; "
                f"found {year_columns}."
            )

        dem_column = year_columns[0]
        gop_column = dem_column + 1

        actual_dem = candidate_row.iloc[dem_column]
        actual_gop = candidate_row.iloc[gop_column]

        if (
            actual_dem.lower()
            != dem_candidate.lower()
            or actual_gop.lower()
            != gop_candidate.lower()
        ):
            raise ValueError(
                f"{year} candidate columns do not match expectations: "
                f"{actual_dem!r}, {actual_gop!r}."
            )

        located[year] = (
            dem_column,
            gop_column,
        )

    return located


def parse_source(
    source_dir: Path,
    spec: SourceSpec,
) -> pd.DataFrame:
    path = source_dir / spec.filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing presidential source file: {path}"
        )

    raw = read_source_table(path)

    district_label = (
        str(raw.iloc[1, 0])
        .strip()
        .lower()
    )

    if district_label not in {
        "cd",
        "district",
    }:
        raise ValueError(
            f"{path.name}: unexpected district header "
            f"{raw.iloc[1, 0]!r}."
        )

    year_columns = locate_year_columns(
        raw=raw,
        expected_years=spec.result_years,
    )

    district_rows = raw.iloc[2:].copy()

    if len(district_rows) != 435:
        raise ValueError(
            f"{path.name}: expected 435 district rows; "
            f"found {len(district_rows)}."
        )

    race_ids = district_rows.iloc[:, 0].apply(
        normalize_race_id
    )

    if race_ids.duplicated().any():
        duplicates = sorted(
            race_ids.loc[
                race_ids.duplicated(
                    keep=False
                )
            ].unique()
        )

        raise ValueError(
            f"{path.name}: duplicate race IDs: "
            + ", ".join(duplicates)
        )

    frames: list[pd.DataFrame] = []

    for result_year, (
        dem_column,
        gop_column,
    ) in year_columns.items():
        dem_share = pd.to_numeric(
            district_rows.iloc[:, dem_column],
            errors="coerce",
        )

        gop_share = pd.to_numeric(
            district_rows.iloc[:, gop_column],
            errors="coerce",
        )

        frame = pd.DataFrame(
            {
                "boundary_cycle": spec.boundary_cycle,
                "presidential_result_year": result_year,
                "race_id": race_ids.to_numpy(),
                "state": race_ids.str.split(
                    "-",
                    n=1,
                ).str[0].to_numpy(),
                "district": race_ids.str.split(
                    "-",
                    n=1,
                ).str[1].to_numpy(),
                "dem_presidential_candidate": (
                    CANDIDATES_BY_YEAR[
                        result_year
                    ][0]
                ),
                "gop_presidential_candidate": (
                    CANDIDATES_BY_YEAR[
                        result_year
                    ][1]
                ),
                "dem_presidential_share": (
                    dem_share.to_numpy()
                ),
                "gop_presidential_share": (
                    gop_share.to_numpy()
                ),
                "district_pres_margin_dem": (
                    dem_share
                    - gop_share
                ).to_numpy(),
                "major_party_share_total": (
                    dem_share
                    + gop_share
                ).to_numpy(),
                "source_name": (
                    "Daily Kos Elections presidential "
                    "results by congressional district"
                ),
                "source_file": spec.filename,
                "source_boundary_description": (
                    f"Congressional district lines used "
                    f"in the {spec.boundary_cycle} elections"
                ),
                "warehouse_version": "1.0",
            }
        )

        frame[
            "presidential_result_available"
        ] = (
            frame[
                "dem_presidential_share"
            ].notna()
            & frame[
                "gop_presidential_share"
            ].notna()
        )

        frames.append(frame)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def build_forecast_baselines(
    warehouse: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for forecast_cycle, plan in (
        FORECAST_BASELINE_PLAN.items()
    ):
        selected = warehouse.loc[
            warehouse["boundary_cycle"].eq(
                plan["boundary_cycle"]
            )
            & warehouse[
                "presidential_result_year"
            ].eq(
                plan[
                    "presidential_result_year"
                ]
            )
        ].copy()

        selected.insert(
            0,
            "forecast_cycle",
            forecast_cycle,
        )

        selected[
            "boundary_compatibility"
        ] = "same_election_boundary_set"

        selected[
            "presidential_baseline_available"
        ] = selected[
            "presidential_result_available"
        ]

        selected[
            "baseline_selection_method"
        ] = (
            "most_recent_completed_presidential_"
            "election_on_target_house_boundaries"
        )

        frames.append(selected)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def build_validation_report(
    warehouse: pd.DataFrame,
    baselines: pd.DataFrame,
) -> tuple[str, list[str]]:
    failures: list[str] = []

    expected_warehouse_rows = (
        sum(
            len(spec.result_years)
            for spec in SOURCE_SPECS
        )
        * 435
    )

    duplicate_warehouse = int(
        warehouse.duplicated(
            [
                "boundary_cycle",
                "presidential_result_year",
                "race_id",
            ]
        ).sum()
    )

    duplicate_baselines = int(
        baselines.duplicated(
            [
                "forecast_cycle",
                "race_id",
            ]
        ).sum()
    )

    if len(warehouse) != expected_warehouse_rows:
        failures.append(
            f"Expected {expected_warehouse_rows} warehouse rows; "
            f"found {len(warehouse)}."
        )

    if duplicate_warehouse:
        failures.append(
            f"Found {duplicate_warehouse} duplicate warehouse rows."
        )

    if len(baselines) != 3 * 435:
        failures.append(
            f"Expected 1,305 forecast-baseline rows; "
            f"found {len(baselines)}."
        )

    if duplicate_baselines:
        failures.append(
            f"Found {duplicate_baselines} duplicate baseline rows."
        )

    invalid_shares = int(
        (
            warehouse[
                "dem_presidential_share"
            ].notna()
            & (
                warehouse[
                    "dem_presidential_share"
                ].lt(0)
                | warehouse[
                    "dem_presidential_share"
                ].gt(100)
            )
        ).sum()
        + (
            warehouse[
                "gop_presidential_share"
            ].notna()
            & (
                warehouse[
                    "gop_presidential_share"
                ].lt(0)
                | warehouse[
                    "gop_presidential_share"
                ].gt(100)
            )
        ).sum()
    )

    if invalid_shares:
        failures.append(
            f"Found {invalid_shares} invalid presidential shares."
        )

    required_baseline_missing = (
        baselines.groupby(
            "forecast_cycle"
        )[
            "presidential_baseline_available"
        ]
        .apply(
            lambda values: int(
                (~values).sum()
            )
        )
    )

    if required_baseline_missing.sum():
        for cycle, missing_count in (
            required_baseline_missing.items()
        ):
            if missing_count:
                failures.append(
                    f"{cycle}: {missing_count} required "
                    "presidential baselines are missing."
                )

    combination_summary = (
        warehouse.groupby(
            [
                "boundary_cycle",
                "presidential_result_year",
            ]
        )
        .agg(
            rows=("race_id", "size"),
            unique_race_ids=(
                "race_id",
                "nunique",
            ),
            available_results=(
                "presidential_result_available",
                "sum",
            ),
            missing_results=(
                "presidential_result_available",
                lambda values: int(
                    (~values).sum()
                ),
            ),
            mean_dem_margin=(
                "district_pres_margin_dem",
                "mean",
            ),
            minimum_dem_margin=(
                "district_pres_margin_dem",
                "min",
            ),
            maximum_dem_margin=(
                "district_pres_margin_dem",
                "max",
            ),
        )
    )

    baseline_summary = (
        baselines.groupby(
            "forecast_cycle"
        )
        .agg(
            rows=("race_id", "size"),
            unique_race_ids=(
                "race_id",
                "nunique",
            ),
            presidential_result_year=(
                "presidential_result_year",
                "first",
            ),
            boundary_cycle=(
                "boundary_cycle",
                "first",
            ),
            available_baselines=(
                "presidential_baseline_available",
                "sum",
            ),
            missing_baselines=(
                "presidential_baseline_available",
                lambda values: int(
                    (~values).sum()
                ),
            ),
            mean_dem_margin=(
                "district_pres_margin_dem",
                "mean",
            ),
        )
    )

    report_lines = [
        "House Presidential Baseline Warehouse Validation",
        "=" * 48,
        "",
        f"Warehouse rows: {len(warehouse)}",
        (
            "Unique boundary/result-year/race rows: "
            f"{warehouse[['boundary_cycle', 'presidential_result_year', 'race_id']].drop_duplicates().shape[0]}"
        ),
        (
            "Duplicate warehouse rows: "
            f"{duplicate_warehouse}"
        ),
        "",
        "Boundary/result-year coverage:",
        combination_summary.to_string(
            float_format=lambda value: f"{value:.4f}"
        ),
        "",
        f"Forecast-baseline rows: {len(baselines)}",
        (
            "Duplicate forecast-cycle/race rows: "
            f"{duplicate_baselines}"
        ),
        "",
        "Forecast baseline coverage:",
        baseline_summary.to_string(
            float_format=lambda value: f"{value:.4f}"
        ),
        "",
        "Baseline mapping:",
        "2016 forecast -> 2012 presidential result on 2016 district lines",
        "2018 forecast -> 2016 presidential result on 2018 district lines",
        "2020 forecast -> 2016 presidential result on 2020 district lines",
        "",
        "Temporal note:",
        (
            "The 2020 presidential result contained in the 2020-boundary "
            "source is warehoused for future analysis but is not used as "
            "a baseline for the simultaneous 2020 House election."
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

    return "\n".join(report_lines), failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a canonical warehouse of presidential results "
            "on historical congressional district boundaries."
        )
    )

    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )

    parser.add_argument(
        "--baseline-output-path",
        type=Path,
        default=DEFAULT_BASELINE_OUTPUT_PATH,
    )

    parser.add_argument(
        "--validation-path",
        type=Path,
        default=DEFAULT_VALIDATION_PATH,
    )

    args = parser.parse_args()

    frames = [
        parse_source(
            source_dir=args.source_dir,
            spec=spec,
        )
        for spec in SOURCE_SPECS
    ]

    warehouse = pd.concat(
        frames,
        ignore_index=True,
    ).sort_values(
        [
            "boundary_cycle",
            "presidential_result_year",
            "state",
            "district",
        ]
    ).reset_index(drop=True)

    baselines = build_forecast_baselines(
        warehouse
    ).sort_values(
        [
            "forecast_cycle",
            "state",
            "district",
        ]
    ).reset_index(drop=True)

    report, failures = (
        build_validation_report(
            warehouse=warehouse,
            baselines=baselines,
        )
    )

    args.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    warehouse.to_csv(
        args.output_path,
        index=False,
    )

    baselines.to_csv(
        args.baseline_output_path,
        index=False,
    )

    args.validation_path.write_text(
        report
    )

    print(report)
    print()
    print(f"Wrote: {args.output_path}")
    print(f"Wrote: {args.baseline_output_path}")
    print(f"Wrote: {args.validation_path}")

    if failures:
        raise RuntimeError(report)


if __name__ == "__main__":
    main()
