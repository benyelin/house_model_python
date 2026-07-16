from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ingest_house_results import RAW_PATH, build_cycle


DEFAULT_CYCLES = (2012, 2014, 2016, 2018, 2020, 2022)

PROCESSED_DIR = Path("historical/house/processed")
WAREHOUSE_DIR = Path("historical/house/warehouse")


def parse_cycles(value: str) -> tuple[int, ...]:
    """Parse a comma-separated collection of even-year election cycles."""
    cycles: list[int] = []

    for item in value.split(","):
        item = item.strip()

        if not item:
            continue

        try:
            cycle = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid cycle value: {item!r}"
            ) from exc

        if cycle < 1976 or cycle > 2024 or cycle % 2 != 0:
            raise argparse.ArgumentTypeError(
                f"Cycle must be an even election year from 1976 through "
                f"2024: {cycle}"
            )

        cycles.append(cycle)

    if not cycles:
        raise argparse.ArgumentTypeError("At least one cycle is required.")

    if len(cycles) != len(set(cycles)):
        raise argparse.ArgumentTypeError("Duplicate cycles are not allowed.")

    return tuple(sorted(cycles))


def build_historical_results(
    cycles: tuple[int, ...],
    raw_path: Path,
) -> tuple[pd.DataFrame, str]:
    """Build and combine normalized district results for multiple cycles."""
    frames: list[pd.DataFrame] = []
    cycle_reports: list[str] = []

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)

    for cycle in cycles:
        print(f"Building {cycle} House results...")

        results, report = build_cycle(
            cycle=cycle,
            raw_path=raw_path,
        )

        cycle_results_path = (
            PROCESSED_DIR / f"house_{cycle}_results.csv"
        )
        cycle_validation_path = (
            PROCESSED_DIR / f"house_{cycle}_results_validation.txt"
        )

        results.to_csv(cycle_results_path, index=False)
        cycle_validation_path.write_text(report)

        frames.append(results)
        cycle_reports.append(report)

    warehouse = pd.concat(
        frames,
        ignore_index=True,
        verify_integrity=False,
    )

    warehouse = warehouse.sort_values(
        ["cycle", "state", "district"],
        key=lambda series: series.map(
            lambda value: (
                0
                if str(value) == "AL"
                else int(value)
                if str(value).isdigit()
                else 999
            )
        )
        if series.name == "district"
        else series,
    ).reset_index(drop=True)

    duplicate_cycle_races = int(
        warehouse.duplicated(["cycle", "race_id"]).sum()
    )
    missing_margins = int(
        warehouse["actual_dem_margin"].isna().sum()
    )
    unexpected_cycle_counts = {
        int(cycle): int(count)
        for cycle, count in warehouse.groupby("cycle").size().items()
        if int(count) != 435
    }

    failures: list[str] = []

    if duplicate_cycle_races:
        failures.append(
            f"Found {duplicate_cycle_races} duplicate cycle/race records."
        )

    if missing_margins:
        failures.append(
            f"Found {missing_margins} records with missing margins."
        )

    if unexpected_cycle_counts:
        failures.append(
            "Unexpected race counts by cycle: "
            f"{unexpected_cycle_counts}"
        )

    report_lines = [
        "Historical House Results Warehouse Validation",
        "=" * 45,
        "",
        f"Cycles: {', '.join(str(cycle) for cycle in cycles)}",
        f"Cycle count: {len(cycles)}",
        f"Warehouse rows: {len(warehouse)}",
        f"Expected rows: {435 * len(cycles)}",
        f"Duplicate cycle/race records: {duplicate_cycle_races}",
        f"Missing margins: {missing_margins}",
        "",
        "Rows by cycle:",
        warehouse.groupby("cycle").size().to_string(),
        "",
        "Major-party-contested races by cycle:",
        warehouse.groupby("cycle")[
            "major_party_contested"
        ].sum().astype(int).to_string(),
        "",
        "Validation status:",
    ]

    if failures:
        report_lines.append("FAILED")
        report_lines.extend(f"- {failure}" for failure in failures)
    else:
        report_lines.append("PASSED")

    report = "\n".join(report_lines)

    if failures:
        raise RuntimeError(report)

    return warehouse, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build normalized historical House results for multiple cycles."
        )
    )
    parser.add_argument(
        "--cycles",
        type=parse_cycles,
        default=DEFAULT_CYCLES,
        help=(
            "Comma-separated election cycles. "
            "Default: 2012,2014,2016,2018,2020,2022"
        ),
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=RAW_PATH,
    )
    args = parser.parse_args()

    warehouse, report = build_historical_results(
        cycles=args.cycles,
        raw_path=args.raw_path,
    )

    warehouse_path = (
        WAREHOUSE_DIR / "house_historical_results_2012_2022.csv"
    )
    validation_path = (
        WAREHOUSE_DIR
        / "house_historical_results_2012_2022_validation.txt"
    )

    warehouse.to_csv(warehouse_path, index=False)
    validation_path.write_text(report)

    print()
    print(report)
    print()
    print(f"Wrote: {warehouse_path}")
    print(f"Wrote: {validation_path}")


if __name__ == "__main__":
    main()
