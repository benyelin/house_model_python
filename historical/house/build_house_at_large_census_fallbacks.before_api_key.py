from __future__ import annotations

import argparse
import csv
import json
import ssl
import subprocess
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import certifi


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "historical/house/raw/dra_district_data/"
    "at_large_state_fallbacks.csv"
)

CENSUS_API_BASE = "https://api.census.gov/data/2020/dec/pl"

AT_LARGE_STATES = {
    "AK": "02",
    "DE": "10",
    "ND": "38",
    "SD": "46",
    "VT": "50",
    "WY": "56",
}

# P2 and P4 give mutually exclusive Hispanic and non-Hispanic
# race categories for total population and age 18+ population.
VARIABLES = {
    "NAME": "state_name",
    "P2_001N": "total_population",
    "P2_002N": "hispanic_population",
    "P2_005N": "white_population",
    "P2_006N": "black_population",
    "P2_007N": "native_population",
    "P2_008N": "asian_population",
    "P2_009N": "pacific_population",
    "P4_001N": "voting_age_population",
    "P4_002N": "hispanic_vap",
    "P4_005N": "white_vap",
    "P4_006N": "black_vap",
    "P4_007N": "native_vap",
    "P4_008N": "asian_vap",
    "P4_009N": "pacific_vap",
}

OUTPUT_COLUMNS = [
    "race_id",
    "state",
    "district",
    "dra_map_year",
    "dra_population_dataset",
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
    "source_name",
    "source_notes",
]


def fetch_state_rows() -> list[dict[str, str]]:
    params = {
        "get": ",".join(VARIABLES),
        "for": "state:*",
    }

    url = f"{CENSUS_API_BASE}?{urlencode(params)}"

    result = subprocess.run(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--max-time",
            "60",
            url,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Census API request failed through curl. "
            f"Exit code: {result.returncode}. "
            f"Response: {result.stderr.strip()}"
        )

    response_text = result.stdout.strip()

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        preview = response_text[:1000]

        raise RuntimeError(
            "Census API returned a non-JSON response. "
            f"Response preview: {preview!r}"
        ) from exc

    if not payload or len(payload) < 2:
        raise RuntimeError(
            "Census API returned no state data."
        )

    headers = payload[0]

    rows = [
        dict(zip(headers, values, strict=True))
        for values in payload[1:]
    ]

    return rows


def build_fallback_rows() -> list[dict[str, object]]:
    census_rows = fetch_state_rows()

    by_fips = {
        row["state"]: row
        for row in census_rows
    }

    output: list[dict[str, object]] = []

    for state, state_fips in AT_LARGE_STATES.items():
        if state_fips not in by_fips:
            raise RuntimeError(
                f"Census API did not return state FIPS {state_fips} "
                f"for {state}."
            )

        source = by_fips[state_fips]

        row: dict[str, object] = {
            "race_id": f"{state}-AL",
            "state": state,
            "district": "AL",
            "dra_map_year": 2022,
            "dra_population_dataset": "2020 Census",
            "dra_composite_total_votes": "",
            "dra_composite_dem_votes": "",
            "dra_composite_rep_votes": "",
            "source_name": (
                "U.S. Census Bureau 2020 P.L. 94-171 API"
            ),
            "source_notes": (
                "At-large congressional district equals the entire "
                "state. Population fields use P2 and P4 mutually "
                "exclusive Hispanic/non-Hispanic categories. "
                "DRA composite election fields unavailable."
            ),
        }

        for census_variable, output_column in VARIABLES.items():
            if census_variable == "NAME":
                continue

            try:
                row[output_column] = int(source[census_variable])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Invalid {census_variable} value for {state}: "
                    f"{source.get(census_variable)!r}"
                ) from exc

        output.append(row)

    return output


def validate(rows: list[dict[str, object]]) -> str:
    failures: list[str] = []

    race_ids = [
        str(row["race_id"])
        for row in rows
    ]

    if len(rows) != 6:
        failures.append(
            f"Expected 6 fallback rows; found {len(rows)}."
        )

    if len(set(race_ids)) != len(race_ids):
        failures.append(
            "Duplicate fallback race IDs found."
        )

    expected_ids = {
        f"{state}-AL"
        for state in AT_LARGE_STATES
    }

    if set(race_ids) != expected_ids:
        failures.append(
            "Fallback race IDs do not match expected at-large states."
        )

    numeric_columns = [
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
    ]

    for row in rows:
        race_id = str(row["race_id"])

        for column in numeric_columns:
            value = row.get(column)

            if not isinstance(value, int) or value < 0:
                failures.append(
                    f"{race_id}: invalid {column}={value!r}."
                )

        if int(row["total_population"]) <= 0:
            failures.append(
                f"{race_id}: total population is nonpositive."
            )

        if int(row["voting_age_population"]) <= 0:
            failures.append(
                f"{race_id}: voting-age population is nonpositive."
            )

        if int(row["voting_age_population"]) > int(
            row["total_population"]
        ):
            failures.append(
                f"{race_id}: voting-age population exceeds total."
            )

        population_categories = sum(
            int(row[column])
            for column in [
                "white_population",
                "hispanic_population",
                "black_population",
                "asian_population",
                "native_population",
                "pacific_population",
            ]
        )

        if population_categories > int(row["total_population"]):
            failures.append(
                f"{race_id}: listed population categories exceed total."
            )

        vap_categories = sum(
            int(row[column])
            for column in [
                "white_vap",
                "hispanic_vap",
                "black_vap",
                "asian_vap",
                "native_vap",
                "pacific_vap",
            ]
        )

        if vap_categories > int(row["voting_age_population"]):
            failures.append(
                f"{race_id}: listed VAP categories exceed total VAP."
            )

    report_lines = [
        "House At-Large Census Fallback Validation",
        "=" * 41,
        "",
        f"Fallback rows: {len(rows)}",
        f"Unique race IDs: {len(set(race_ids))}",
        "Race IDs: " + ", ".join(sorted(race_ids)),
        "",
        "Source:",
        "U.S. Census Bureau 2020 P.L. 94-171 API",
        "",
        "Definitions:",
        (
            "Total-population categories use P2; voting-age categories "
            "use P4. Hispanic is reported separately, while the race "
            "categories are non-Hispanic single-race counts."
        ),
        "",
        "Political-data status:",
        (
            "DRA composite election fields remain blank because they "
            "are not supplied by the Census API."
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

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Populate demographic fallback rows for at-large House "
            "districts from the 2020 Census P.L. 94-171 API."
        )
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )

    args = parser.parse_args()

    rows = build_fallback_rows()
    report = validate(rows)

    args.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=OUTPUT_COLUMNS,
        )
        writer.writeheader()
        writer.writerows(rows)

    validation_path = args.output_path.with_name(
        "at_large_state_fallbacks_validation.txt"
    )

    validation_path.write_text(
        report,
        encoding="utf-8",
    )

    print(report)
    print()
    print(f"Wrote: {args.output_path}")
    print(f"Wrote: {validation_path}")


if __name__ == "__main__":
    main()
