#!/usr/bin/env python3
"""
Build the canonical historical generic-ballot polling warehouse.

Source:
    historical/house/polling/raw/fivethirtyeight_archive/
        generic_ballot_polls_historical.csv

Outputs:
    historical/common/polling/warehouse/
        generic_ballot_polling_warehouse.csv
        generic_ballot_polling_warehouse_manifest.json

    historical/common/polling/validation/
        generic_ballot_polling_validation.json
        generic_ballot_polling_by_cycle.csv
        generic_ballot_polling_population_counts.csv
        generic_ballot_polling_pollster_counts.csv
        generic_ballot_polling_exclusions.csv

Design principles:
    - Raw source is immutable.
    - Every source question becomes at most one warehouse row.
    - Poll margins are Democratic minus Republican.
    - Dates are normalized before eligibility checks.
    - Post-election and malformed records are excluded explicitly.
    - Exclusions are preserved for auditability.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]

RAW_PATH = (
    REPOSITORY_ROOT
    / "historical"
    / "house"
    / "polling"
    / "raw"
    / "fivethirtyeight_archive"
    / "generic_ballot_polls_historical.csv"
)

WAREHOUSE_DIR = (
    REPOSITORY_ROOT
    / "historical"
    / "common"
    / "polling"
    / "warehouse"
)

VALIDATION_DIR = (
    REPOSITORY_ROOT
    / "historical"
    / "common"
    / "polling"
    / "validation"
)

WAREHOUSE_PATH = (
    WAREHOUSE_DIR
    / "generic_ballot_polling_warehouse.csv"
)

MANIFEST_PATH = (
    WAREHOUSE_DIR
    / "generic_ballot_polling_warehouse_manifest.json"
)

VALIDATION_PATH = (
    VALIDATION_DIR
    / "generic_ballot_polling_validation.json"
)

BY_CYCLE_PATH = (
    VALIDATION_DIR
    / "generic_ballot_polling_by_cycle.csv"
)

POPULATION_COUNTS_PATH = (
    VALIDATION_DIR
    / "generic_ballot_polling_population_counts.csv"
)

POLLSTER_COUNTS_PATH = (
    VALIDATION_DIR
    / "generic_ballot_polling_pollster_counts.csv"
)

EXCLUSIONS_PATH = (
    VALIDATION_DIR
    / "generic_ballot_polling_exclusions.csv"
)


REQUIRED_SOURCE_COLUMNS = {
    "poll_id",
    "question_id",
    "pollster_id",
    "pollster",
    "start_date",
    "end_date",
    "question_id",
    "sample_size",
    "population",
    "cycle",
    "election_date",
    "stage",
    "dem",
    "rep",
}


WAREHOUSE_COLUMNS = [
    "source",
    "source_file",
    "source_row_number",
    "source_row_hash",
    "poll_id",
    "question_id",
    "race_id",
    "cycle",
    "election_date",
    "stage",
    "office_type",
    "seat_name",
    "pollster_id",
    "pollster",
    "display_name",
    "pollster_rating_id",
    "pollster_rating_name",
    "numeric_grade",
    "pollscore",
    "transparency_score",
    "methodology",
    "start_date",
    "end_date",
    "field_period_days",
    "days_before_election",
    "sample_size",
    "population",
    "population_full",
    "tracking",
    "internal",
    "partisan",
    "sponsors",
    "notes",
    "url",
    "url_article",
    "url_topline",
    "url_crosstab",
    "dem_pct",
    "rep_pct",
    "ind_pct",
    "two_party_total",
    "margin_dem",
    "two_party_margin_dem",
    "valid_major_party_values",
    "pre_election",
]


def scalar_to_json(value: Any) -> Any:
    """Convert pandas/numpy values into JSON-safe Python scalars."""
    if value is None:
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()

    if pd.isna(value):
        return None

    return value


def normalize_text(series: pd.Series) -> pd.Series:
    """Trim strings while retaining missing values."""
    result = series.astype("string").str.strip()
    return result.mask(result.eq(""))


def normalize_population(series: pd.Series) -> pd.Series:
    """
    Normalize FiveThirtyEight population codes.

    lv = likely voters
    rv = registered voters
    v  = voters
    a  = adults
    """
    normalized = normalize_text(series).str.lower()

    mappings = {
        "likely voters": "lv",
        "likely voter": "lv",
        "registered voters": "rv",
        "registered voter": "rv",
        "voters": "v",
        "voter": "v",
        "adults": "a",
        "adult": "a",
    }

    normalized = normalized.replace(mappings)

    allowed = {"lv", "rv", "v", "a"}

    return normalized.where(
        normalized.isin(allowed),
        "unknown",
    )


def normalize_boolean(series: pd.Series) -> pd.Series:
    """Normalize mixed boolean-like source values."""
    text = normalize_text(series).str.lower()

    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "y": True,
        "false": False,
        "0": False,
        "no": False,
        "n": False,
    }

    return text.map(mapping).astype("boolean")


def parse_date(series: pd.Series) -> pd.Series:
    """
    Parse the date formats used in the recovered archive.

    FiveThirtyEight historically used both:
        M/D/YY
        M/D/YYYY
        YYYY-MM-DD
    """
    text = normalize_text(series)

    parsed = pd.to_datetime(
        text,
        errors="coerce",
        format="mixed",
    )

    return parsed.dt.normalize()


def calculate_source_hash(
    frame: pd.DataFrame,
) -> pd.Series:
    """
    Create a deterministic content hash for each raw source row.

    This allows future rebuilds to identify source changes independently
    of row ordering.
    """
    hash_columns = sorted(frame.columns)

    def hash_row(row: pd.Series) -> str:
        payload = {
            column: scalar_to_json(row[column])
            for column in hash_columns
        }

        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")

        return hashlib.sha256(encoded).hexdigest()

    return frame.apply(hash_row, axis=1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for chunk in iter(
            lambda: file_handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def add_exclusion_reason(
    reasons: pd.Series,
    condition: pd.Series,
    label: str,
) -> pd.Series:
    """
    Append a semicolon-delimited reason to records matching condition.
    """
    reasons = reasons.copy()

    existing = reasons.fillna("")

    reasons.loc[condition] = np.where(
        existing.loc[condition].eq(""),
        label,
        existing.loc[condition] + ";" + label,
    )

    return reasons


def build_warehouse(
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalize the recovered FiveThirtyEight generic-ballot archive.
    """
    frame = source.copy()

    frame.columns = [
        str(column).strip()
        for column in frame.columns
    ]

    missing_columns = sorted(
        REQUIRED_SOURCE_COLUMNS
        - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Source file is missing required columns: "
            + ", ".join(missing_columns)
        )

    frame.insert(
        0,
        "source_row_number",
        np.arange(2, len(frame) + 2),
    )

    frame["source_row_hash"] = calculate_source_hash(
        frame.drop(
            columns=["source_row_number"],
        )
    )

    normalized = pd.DataFrame(
        index=frame.index
    )

    normalized["source"] = (
        "FiveThirtyEight historical polling archive"
    )

    normalized["source_file"] = str(
        RAW_PATH.relative_to(REPOSITORY_ROOT)
    )

    normalized["source_row_number"] = (
        frame["source_row_number"]
    )

    normalized["source_row_hash"] = (
        frame["source_row_hash"]
    )

    integer_columns = [
        "poll_id",
        "question_id",
        "race_id",
        "cycle",
        "pollster_id",
        "pollster_rating_id",
    ]

    for column in integer_columns:
        if column in frame.columns:
            normalized[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            ).astype("Int64")
        else:
            normalized[column] = pd.Series(
                pd.NA,
                index=frame.index,
                dtype="Int64",
            )

    text_columns = [
        "stage",
        "office_type",
        "seat_name",
        "pollster",
        "display_name",
        "pollster_rating_name",
        "methodology",
        "population_full",
        "sponsors",
        "notes",
        "url",
        "url_article",
        "url_topline",
        "url_crosstab",
        "partisan",
    ]

    for column in text_columns:
        if column in frame.columns:
            normalized[column] = normalize_text(
                frame[column]
            )
        else:
            normalized[column] = pd.Series(
                pd.NA,
                index=frame.index,
                dtype="string",
            )

    numeric_columns = [
        "numeric_grade",
        "pollscore",
        "transparency_score",
        "sample_size",
    ]

    for column in numeric_columns:
        if column in frame.columns:
            normalized[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )
        else:
            normalized[column] = np.nan

    normalized["start_date"] = parse_date(
        frame["start_date"]
    )

    normalized["end_date"] = parse_date(
        frame["end_date"]
    )

    normalized["election_date"] = parse_date(
        frame["election_date"]
    )

    normalized["population"] = normalize_population(
        frame["population"]
    )

    normalized["tracking"] = normalize_boolean(
        frame["tracking"]
        if "tracking" in frame.columns
        else pd.Series(
            pd.NA,
            index=frame.index,
        )
    )

    normalized["internal"] = normalize_boolean(
        frame["internal"]
        if "internal" in frame.columns
        else pd.Series(
            pd.NA,
            index=frame.index,
        )
    )

    normalized["dem_pct"] = pd.to_numeric(
        frame["dem"],
        errors="coerce",
    )

    normalized["rep_pct"] = pd.to_numeric(
        frame["rep"],
        errors="coerce",
    )

    normalized["ind_pct"] = pd.to_numeric(
        frame["ind"],
        errors="coerce",
    ) if "ind" in frame.columns else np.nan

    normalized["field_period_days"] = (
        normalized["end_date"]
        - normalized["start_date"]
    ).dt.days + 1

    normalized["days_before_election"] = (
        normalized["election_date"]
        - normalized["end_date"]
    ).dt.days

    normalized["two_party_total"] = (
        normalized["dem_pct"]
        + normalized["rep_pct"]
    )

    normalized["margin_dem"] = (
        normalized["dem_pct"]
        - normalized["rep_pct"]
    )

    normalized["two_party_margin_dem"] = np.where(
        normalized["two_party_total"].gt(0),
        (
            100.0
            * normalized["margin_dem"]
            / normalized["two_party_total"]
        ),
        np.nan,
    )

    normalized["valid_major_party_values"] = (
        normalized["dem_pct"].between(
            0,
            100,
            inclusive="both",
        )
        & normalized["rep_pct"].between(
            0,
            100,
            inclusive="both",
        )
        & normalized["two_party_total"].between(
            1,
            100,
            inclusive="both",
        )
    )

    normalized["pre_election"] = (
        normalized["end_date"]
        <= normalized["election_date"]
    )

    exclusion_reason = pd.Series(
        "",
        index=normalized.index,
        dtype="string",
    )

    exclusion_reason = add_exclusion_reason(
        exclusion_reason,
        normalized["poll_id"].isna(),
        "missing_poll_id",
    )

    exclusion_reason = add_exclusion_reason(
        exclusion_reason,
        normalized["question_id"].isna(),
        "missing_question_id",
    )

    exclusion_reason = add_exclusion_reason(
        exclusion_reason,
        normalized["cycle"].isna(),
        "missing_cycle",
    )

    exclusion_reason = add_exclusion_reason(
        exclusion_reason,
        normalized["start_date"].isna(),
        "invalid_start_date",
    )

    exclusion_reason = add_exclusion_reason(
        exclusion_reason,
        normalized["end_date"].isna(),
        "invalid_end_date",
    )

    exclusion_reason = add_exclusion_reason(
        exclusion_reason,
        normalized["election_date"].isna(),
        "invalid_election_date",
    )

    exclusion_reason = add_exclusion_reason(
        exclusion_reason,
        (
            normalized["start_date"].notna()
            & normalized["end_date"].notna()
            & normalized["start_date"].gt(
                normalized["end_date"]
            )
        ),
        "start_date_after_end_date",
    )

    exclusion_reason = add_exclusion_reason(
        exclusion_reason,
        ~normalized["valid_major_party_values"],
        "invalid_major_party_values",
    )

    exclusion_reason = add_exclusion_reason(
        exclusion_reason,
        (
            normalized["end_date"].notna()
            & normalized["election_date"].notna()
            & normalized["end_date"].gt(
                normalized["election_date"]
            )
        ),
        "post_election_poll",
    )

    exclusion_reason = add_exclusion_reason(
        exclusion_reason,
        (
            normalized["sample_size"].notna()
            & normalized["sample_size"].le(0)
        ),
        "nonpositive_sample_size",
    )

    normalized["exclusion_reason"] = (
        exclusion_reason.mask(
            exclusion_reason.eq("")
        )
    )

    included = normalized[
        normalized["exclusion_reason"].isna()
    ].copy()

    excluded = normalized[
        normalized["exclusion_reason"].notna()
    ].copy()

    included = included[
        WAREHOUSE_COLUMNS
    ].sort_values(
        [
            "cycle",
            "end_date",
            "poll_id",
            "question_id",
        ],
        kind="stable",
    ).reset_index(drop=True)

    excluded_columns = [
        *WAREHOUSE_COLUMNS,
        "exclusion_reason",
    ]

    excluded = excluded[
        excluded_columns
    ].sort_values(
        [
            "cycle",
            "end_date",
            "poll_id",
            "question_id",
        ],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    return included, excluded


def validate_warehouse(
    source: pd.DataFrame,
    warehouse: pd.DataFrame,
    excluded: pd.DataFrame,
) -> dict[str, Any]:
    """
    Run structural, temporal, and statistical warehouse validation.
    """
    duplicate_question_mask = warehouse.duplicated(
        subset=["question_id"],
        keep=False,
    )

    duplicate_poll_question_mask = (
        warehouse.duplicated(
            subset=[
                "poll_id",
                "question_id",
            ],
            keep=False,
        )
    )

    date_order_valid = (
        warehouse["start_date"]
        <= warehouse["end_date"]
    ).all()

    pre_election_valid = (
        warehouse["end_date"]
        <= warehouse["election_date"]
    ).all()

    margin_identity_valid = np.allclose(
        warehouse["margin_dem"],
        (
            warehouse["dem_pct"]
            - warehouse["rep_pct"]
        ),
        equal_nan=True,
    )

    two_party_margin_identity_valid = np.allclose(
        warehouse["two_party_margin_dem"],
        (
            100.0
            * warehouse["margin_dem"]
            / warehouse["two_party_total"]
        ),
        equal_nan=True,
    )

    source_reconciliation_valid = (
        len(source)
        == len(warehouse) + len(excluded)
    )

    cycle_election_dates = (
        warehouse.groupby(
            "cycle",
            dropna=False,
        )["election_date"]
        .nunique(
            dropna=False
        )
    )

    validation = {
        "status": "PASSED",
        "source_rows": int(len(source)),
        "warehouse_rows": int(len(warehouse)),
        "excluded_rows": int(len(excluded)),
        "source_reconciliation_valid": bool(
            source_reconciliation_valid
        ),
        "unique_poll_ids": int(
            warehouse["poll_id"].nunique(
                dropna=True
            )
        ),
        "unique_question_ids": int(
            warehouse["question_id"].nunique(
                dropna=True
            )
        ),
        "duplicate_question_rows": int(
            duplicate_question_mask.sum()
        ),
        "duplicate_poll_question_rows": int(
            duplicate_poll_question_mask.sum()
        ),
        "missing_poll_ids": int(
            warehouse["poll_id"].isna().sum()
        ),
        "missing_question_ids": int(
            warehouse["question_id"].isna().sum()
        ),
        "missing_cycle": int(
            warehouse["cycle"].isna().sum()
        ),
        "missing_start_date": int(
            warehouse["start_date"].isna().sum()
        ),
        "missing_end_date": int(
            warehouse["end_date"].isna().sum()
        ),
        "missing_election_date": int(
            warehouse["election_date"].isna().sum()
        ),
        "date_order_valid": bool(
            date_order_valid
        ),
        "pre_election_valid": bool(
            pre_election_valid
        ),
        "margin_identity_valid": bool(
            margin_identity_valid
        ),
        "two_party_margin_identity_valid": bool(
            two_party_margin_identity_valid
        ),
        "invalid_major_party_value_rows": int(
            (
                ~warehouse[
                    "valid_major_party_values"
                ]
            ).sum()
        ),
        "cycles": sorted(
            int(value)
            for value in (
                warehouse["cycle"]
                .dropna()
                .unique()
            )
        ),
        "cycle_election_date_counts": {
            str(int(cycle)): int(count)
            for cycle, count in (
                cycle_election_dates.items()
            )
        },
        "minimum_poll_end_date": (
            warehouse["end_date"]
            .min()
            .date()
            .isoformat()
            if not warehouse.empty
            else None
        ),
        "maximum_poll_end_date": (
            warehouse["end_date"]
            .max()
            .date()
            .isoformat()
            if not warehouse.empty
            else None
        ),
        "minimum_margin_dem": (
            float(
                warehouse["margin_dem"].min()
            )
            if not warehouse.empty
            else None
        ),
        "maximum_margin_dem": (
            float(
                warehouse["margin_dem"].max()
            )
            if not warehouse.empty
            else None
        ),
        "mean_margin_dem": (
            float(
                warehouse["margin_dem"].mean()
            )
            if not warehouse.empty
            else None
        ),
        "median_margin_dem": (
            float(
                warehouse["margin_dem"].median()
            )
            if not warehouse.empty
            else None
        ),
        "exclusion_reason_counts": (
            excluded["exclusion_reason"]
            .value_counts(dropna=False)
            .to_dict()
            if not excluded.empty
            else {}
        ),
    }

    hard_failures = {
        "source_reconciliation_valid": (
            not source_reconciliation_valid
        ),
        "duplicate_question_rows": (
            duplicate_question_mask.any()
        ),
        "duplicate_poll_question_rows": (
            duplicate_poll_question_mask.any()
        ),
        "missing_poll_ids": (
            warehouse["poll_id"].isna().any()
        ),
        "missing_question_ids": (
            warehouse["question_id"].isna().any()
        ),
        "missing_cycle": (
            warehouse["cycle"].isna().any()
        ),
        "date_order_valid": (
            not date_order_valid
        ),
        "pre_election_valid": (
            not pre_election_valid
        ),
        "margin_identity_valid": (
            not margin_identity_valid
        ),
        "two_party_margin_identity_valid": (
            not two_party_margin_identity_valid
        ),
        "invalid_major_party_value_rows": (
            (
                ~warehouse[
                    "valid_major_party_values"
                ]
            ).any()
        ),
    }

    failed_checks = [
        name
        for name, failed
        in hard_failures.items()
        if failed
    ]

    validation["failed_checks"] = (
        failed_checks
    )

    if failed_checks:
        validation["status"] = "FAILED"

    return validation


def write_diagnostics(
    warehouse: pd.DataFrame,
) -> None:
    by_cycle = (
        warehouse.groupby(
            "cycle",
            dropna=False,
        )
        .agg(
            rows=("question_id", "size"),
            unique_polls=(
                "poll_id",
                "nunique",
            ),
            unique_questions=(
                "question_id",
                "nunique",
            ),
            unique_pollsters=(
                "pollster_id",
                "nunique",
            ),
            first_poll_end_date=(
                "end_date",
                "min",
            ),
            last_poll_end_date=(
                "end_date",
                "max",
            ),
            election_date=(
                "election_date",
                "max",
            ),
            mean_margin_dem=(
                "margin_dem",
                "mean",
            ),
            median_margin_dem=(
                "margin_dem",
                "median",
            ),
            mean_sample_size=(
                "sample_size",
                "mean",
            ),
        )
        .reset_index()
        .sort_values("cycle")
    )

    by_cycle.to_csv(
        BY_CYCLE_PATH,
        index=False,
    )

    population_counts = (
        warehouse.groupby(
            [
                "cycle",
                "population",
            ],
            dropna=False,
        )
        .size()
        .rename("rows")
        .reset_index()
        .sort_values(
            [
                "cycle",
                "rows",
                "population",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
    )

    population_counts.to_csv(
        POPULATION_COUNTS_PATH,
        index=False,
    )

    pollster_counts = (
        warehouse.groupby(
            [
                "pollster_id",
                "pollster",
            ],
            dropna=False,
        )
        .agg(
            rows=("question_id", "size"),
            unique_polls=(
                "poll_id",
                "nunique",
            ),
            first_end_date=(
                "end_date",
                "min",
            ),
            last_end_date=(
                "end_date",
                "max",
            ),
            mean_numeric_grade=(
                "numeric_grade",
                "mean",
            ),
            mean_pollscore=(
                "pollscore",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "rows",
                "pollster",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    pollster_counts.to_csv(
        POLLSTER_COUNTS_PATH,
        index=False,
    )


def main() -> int:
    print(
        "Generic Ballot Polling Warehouse Builder"
    )
    print("=" * 48)

    if not RAW_PATH.exists():
        print(
            "ERROR: Raw source file not found:"
        )
        print(RAW_PATH)
        return 1

    WAREHOUSE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source = pd.read_csv(
        RAW_PATH,
        low_memory=False,
    )

    print(f"Source rows: {len(source):,}")
    print(
        f"Source columns: "
        f"{len(source.columns):,}"
    )

    warehouse, excluded = build_warehouse(
        source
    )

    validation = validate_warehouse(
        source,
        warehouse,
        excluded,
    )

    warehouse.to_csv(
        WAREHOUSE_PATH,
        index=False,
        date_format="%Y-%m-%d",
    )

    excluded.to_csv(
        EXCLUSIONS_PATH,
        index=False,
        date_format="%Y-%m-%d",
    )

    write_diagnostics(warehouse)

    manifest = {
        "builder": str(
            Path(__file__).resolve().relative_to(
                REPOSITORY_ROOT
            )
        ),
        "source_file": str(
            RAW_PATH.relative_to(
                REPOSITORY_ROOT
            )
        ),
        "source_file_sha256": sha256_file(
            RAW_PATH
        ),
        "warehouse_file": str(
            WAREHOUSE_PATH.relative_to(
                REPOSITORY_ROOT
            )
        ),
        "warehouse_file_sha256": sha256_file(
            WAREHOUSE_PATH
        ),
        "source_rows": int(len(source)),
        "warehouse_rows": int(
            len(warehouse)
        ),
        "excluded_rows": int(
            len(excluded)
        ),
        "warehouse_columns": (
            WAREHOUSE_COLUMNS
        ),
        "cycles": validation["cycles"],
        "validation_status": (
            validation["status"]
        ),
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n"
    )

    VALIDATION_PATH.write_text(
        json.dumps(
            validation,
            indent=2,
            default=scalar_to_json,
        )
        + "\n"
    )

    print()
    print("Build results")
    print("-" * 48)
    print(
        f"Warehouse rows: "
        f"{len(warehouse):,}"
    )
    print(
        f"Excluded rows: "
        f"{len(excluded):,}"
    )
    print(
        f"Unique polls: "
        f"{warehouse['poll_id'].nunique():,}"
    )
    print(
        f"Unique questions: "
        f"{warehouse['question_id'].nunique():,}"
    )
    print(
        "Cycles: "
        + ", ".join(
            str(cycle)
            for cycle in validation["cycles"]
        )
    )

    print()
    print("Rows by cycle")
    print("-" * 48)

    cycle_counts = (
        warehouse["cycle"]
        .value_counts()
        .sort_index()
    )

    print(cycle_counts.to_string())

    print()
    print("Population counts")
    print("-" * 48)

    print(
        warehouse["population"]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()
    print("Margin summary")
    print("-" * 48)

    print(
        warehouse["margin_dem"]
        .describe()
        .to_string()
    )

    print()
    print("Validation")
    print("-" * 48)
    print(
        f"Status: "
        f"{validation['status']}"
    )
    print(
        "Failed checks: "
        + (
            ", ".join(
                validation[
                    "failed_checks"
                ]
            )
            if validation[
                "failed_checks"
            ]
            else "None"
        )
    )

    print()
    print("Outputs")
    print("-" * 48)

    for path in [
        WAREHOUSE_PATH,
        MANIFEST_PATH,
        VALIDATION_PATH,
        BY_CYCLE_PATH,
        POPULATION_COUNTS_PATH,
        POLLSTER_COUNTS_PATH,
        EXCLUSIONS_PATH,
    ]:
        print(
            path.relative_to(
                REPOSITORY_ROOT
            )
        )

    if validation["status"] != "PASSED":
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
