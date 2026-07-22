#!/usr/bin/env python3
"""
Build leakage-free historical generic-ballot polling snapshots.

Each snapshot contains all canonical warehouse questions that were fully
available by a specified number of days before Election Day.

This script performs selection only. It does not:
    - average polls,
    - weight pollsters,
    - select one population,
    - exclude partisan/internal polls,
    - apply recency decay,
    - collapse multiple questions from the same poll.

Outputs:
    historical/common/polling/snapshots/
        generic_ballot_polling_snapshots.csv
        generic_ballot_snapshot_summary.csv
        generic_ballot_snapshot_manifest.json

    historical/common/polling/validation/
        generic_ballot_snapshot_validation.json
        generic_ballot_snapshot_increment_counts.csv
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]

WAREHOUSE_PATH = (
    REPOSITORY_ROOT
    / "historical"
    / "common"
    / "polling"
    / "warehouse"
    / "generic_ballot_polling_warehouse.csv"
)

SNAPSHOT_DIR = (
    REPOSITORY_ROOT
    / "historical"
    / "common"
    / "polling"
    / "snapshots"
)

VALIDATION_DIR = (
    REPOSITORY_ROOT
    / "historical"
    / "common"
    / "polling"
    / "validation"
)

SNAPSHOT_PATH = (
    SNAPSHOT_DIR
    / "generic_ballot_polling_snapshots.csv"
)

SUMMARY_PATH = (
    SNAPSHOT_DIR
    / "generic_ballot_snapshot_summary.csv"
)

MANIFEST_PATH = (
    SNAPSHOT_DIR
    / "generic_ballot_snapshot_manifest.json"
)

VALIDATION_PATH = (
    VALIDATION_DIR
    / "generic_ballot_snapshot_validation.json"
)

INCREMENT_PATH = (
    VALIDATION_DIR
    / "generic_ballot_snapshot_increment_counts.csv"
)


SNAPSHOT_DAYS = [
    180,
    150,
    120,
    90,
    75,
    60,
    45,
    30,
    21,
    14,
    10,
    7,
    5,
    3,
    1,
    0,
]


IDENTITY_COLUMNS = [
    "cycle",
    "snapshot_days_before_election",
    "question_id",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for chunk in iter(
            lambda: file_handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def json_safe(value: Any) -> Any:
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


def load_warehouse() -> pd.DataFrame:
    if not WAREHOUSE_PATH.exists():
        raise FileNotFoundError(
            f"Warehouse not found: {WAREHOUSE_PATH}"
        )

    frame = pd.read_csv(
        WAREHOUSE_PATH,
        low_memory=False,
    )

    required_columns = {
        "poll_id",
        "question_id",
        "cycle",
        "start_date",
        "end_date",
        "election_date",
        "days_before_election",
        "margin_dem",
    }

    missing = sorted(
        required_columns - set(frame.columns)
    )

    if missing:
        raise ValueError(
            "Warehouse missing required columns: "
            + ", ".join(missing)
        )

    for column in [
        "start_date",
        "end_date",
        "election_date",
    ]:
        frame[column] = pd.to_datetime(
            frame[column],
            errors="coerce",
        ).dt.normalize()

    frame["cycle"] = pd.to_numeric(
        frame["cycle"],
        errors="coerce",
    ).astype("Int64")

    frame["poll_id"] = pd.to_numeric(
        frame["poll_id"],
        errors="coerce",
    ).astype("Int64")

    frame["question_id"] = pd.to_numeric(
        frame["question_id"],
        errors="coerce",
    ).astype("Int64")

    frame["days_before_election"] = (
        pd.to_numeric(
            frame["days_before_election"],
            errors="coerce",
        )
    )

    return frame


def build_snapshots(
    warehouse: pd.DataFrame,
) -> pd.DataFrame:
    snapshot_frames: list[pd.DataFrame] = []

    for cycle in sorted(
        warehouse["cycle"].dropna().unique()
    ):
        cycle_frame = warehouse[
            warehouse["cycle"].eq(cycle)
        ].copy()

        election_dates = (
            cycle_frame["election_date"]
            .dropna()
            .unique()
        )

        if len(election_dates) != 1:
            raise ValueError(
                f"Cycle {int(cycle)} has "
                f"{len(election_dates)} election dates."
            )

        election_date = pd.Timestamp(
            election_dates[0]
        )

        for snapshot_days in SNAPSHOT_DAYS:
            snapshot_date = (
                election_date
                - pd.Timedelta(
                    days=snapshot_days
                )
            )

            eligible = cycle_frame[
                cycle_frame["end_date"].le(
                    snapshot_date
                )
            ].copy()

            eligible.insert(
                0,
                "snapshot_days_before_election",
                snapshot_days,
            )

            eligible.insert(
                1,
                "snapshot_date",
                snapshot_date,
            )

            eligible["poll_age_days"] = (
                snapshot_date
                - eligible["end_date"]
            ).dt.days

            eligible["first_available_snapshot_days"] = (
                eligible[
                    "days_before_election"
                ]
                .apply(
                    lambda days_before:
                    max(
                        (
                            day
                            for day in SNAPSHOT_DAYS
                            if day <= days_before
                        ),
                        default=np.nan,
                    )
                )
            )

            snapshot_frames.append(
                eligible
            )

    if not snapshot_frames:
        return pd.DataFrame()

    snapshots = pd.concat(
        snapshot_frames,
        ignore_index=True,
    )

    snapshots = snapshots.sort_values(
        [
            "cycle",
            "snapshot_days_before_election",
            "end_date",
            "poll_id",
            "question_id",
        ],
        ascending=[
            True,
            False,
            True,
            True,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)

    return snapshots


def build_summary(
    snapshots: pd.DataFrame,
) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame()

    summary = (
        snapshots.groupby(
            [
                "cycle",
                "snapshot_days_before_election",
                "snapshot_date",
                "election_date",
            ],
            dropna=False,
        )
        .agg(
            question_rows=(
                "question_id",
                "size",
            ),
            unique_questions=(
                "question_id",
                "nunique",
            ),
            unique_polls=(
                "poll_id",
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
            latest_poll_end_date=(
                "end_date",
                "max",
            ),
            minimum_poll_age_days=(
                "poll_age_days",
                "min",
            ),
            maximum_poll_age_days=(
                "poll_age_days",
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
            lv_questions=(
                "population",
                lambda values:
                int(values.eq("lv").sum()),
            ),
            rv_questions=(
                "population",
                lambda values:
                int(values.eq("rv").sum()),
            ),
            voter_questions=(
                "population",
                lambda values:
                int(values.eq("v").sum()),
            ),
            adult_questions=(
                "population",
                lambda values:
                int(values.eq("a").sum()),
            ),
            partisan_questions=(
                "partisan",
                lambda values:
                int(values.notna().sum()),
            ),
            internal_questions=(
                "internal",
                lambda values:
                int(
                    values.astype("string")
                    .str.lower()
                    .eq("true")
                    .sum()
                ),
            ),
        )
        .reset_index()
        .sort_values(
            [
                "cycle",
                "snapshot_days_before_election",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    return summary


def build_increment_counts(
    snapshots: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for cycle in sorted(
        snapshots["cycle"].dropna().unique()
    ):
        cycle_frame = snapshots[
            snapshots["cycle"].eq(cycle)
        ]

        prior_questions: set[int] = set()

        for snapshot_days in sorted(
            SNAPSHOT_DAYS,
            reverse=True,
        ):
            current = cycle_frame[
                cycle_frame[
                    "snapshot_days_before_election"
                ].eq(snapshot_days)
            ]

            current_questions = set(
                int(value)
                for value in (
                    current["question_id"]
                    .dropna()
                    .unique()
                )
            )

            new_questions = (
                current_questions
                - prior_questions
            )

            records.append(
                {
                    "cycle": int(cycle),
                    "snapshot_days_before_election": (
                        snapshot_days
                    ),
                    "question_rows": int(
                        len(current)
                    ),
                    "unique_questions": int(
                        len(current_questions)
                    ),
                    "new_questions_since_prior_snapshot": int(
                        len(new_questions)
                    ),
                }
            )

            prior_questions = current_questions

    return pd.DataFrame(records)


def validate_snapshots(
    warehouse: pd.DataFrame,
    snapshots: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, Any]:
    duplicate_rows = snapshots.duplicated(
        subset=IDENTITY_COLUMNS,
        keep=False,
    )

    availability_valid = (
        snapshots["end_date"]
        <= snapshots["snapshot_date"]
    ).all()

    nonnegative_age_valid = (
        snapshots["poll_age_days"]
        .ge(0)
        .all()
    )

    snapshot_date_identity_valid = (
        snapshots["snapshot_date"]
        == (
            snapshots["election_date"]
            - pd.to_timedelta(
                snapshots[
                    "snapshot_days_before_election"
                ],
                unit="D",
            )
        )
    ).all()

    source_question_ids = set(
        int(value)
        for value in (
            warehouse["question_id"]
            .dropna()
            .unique()
        )
    )

    snapshot_question_ids = set(
        int(value)
        for value in (
            snapshots["question_id"]
            .dropna()
            .unique()
        )
    )

    unknown_question_ids = sorted(
        snapshot_question_ids
        - source_question_ids
    )

    expected_snapshot_pairs = {
        (int(cycle), int(snapshot_days))
        for cycle in (
            warehouse["cycle"]
            .dropna()
            .unique()
        )
        for snapshot_days in SNAPSHOT_DAYS
    }

    actual_snapshot_pairs = {
        (
            int(row.cycle),
            int(
                row.snapshot_days_before_election
            ),
        )
        for row in summary.itertuples()
    }

    missing_snapshot_pairs = sorted(
        expected_snapshot_pairs
        - actual_snapshot_pairs
    )

    monotonic_failures: list[dict[str, Any]] = []

    for cycle in sorted(
        warehouse["cycle"].dropna().unique()
    ):
        cycle_summary = (
            summary[
                summary["cycle"].eq(cycle)
            ]
            .sort_values(
                "snapshot_days_before_election",
                ascending=False,
            )
        )

        counts = (
            cycle_summary["unique_questions"]
            .tolist()
        )

        if any(
            later < earlier
            for earlier, later in zip(
                counts,
                counts[1:],
            )
        ):
            monotonic_failures.append(
                {
                    "cycle": int(cycle),
                    "counts": [
                        int(value)
                        for value in counts
                    ],
                }
            )

    zero_day_counts_match: dict[str, bool] = {}

    for cycle in sorted(
        warehouse["cycle"].dropna().unique()
    ):
        warehouse_count = int(
            warehouse[
                warehouse["cycle"].eq(cycle)
            ]["question_id"].nunique()
        )

        zero_day_count = int(
            snapshots[
                snapshots["cycle"].eq(cycle)
                & snapshots[
                    "snapshot_days_before_election"
                ].eq(0)
            ]["question_id"].nunique()
        )

        zero_day_counts_match[
            str(int(cycle))
        ] = (
            warehouse_count
            == zero_day_count
        )

    failed_checks: list[str] = []

    checks = {
        "duplicate_snapshot_question_rows": (
            not duplicate_rows.any()
        ),
        "availability_valid": (
            availability_valid
        ),
        "nonnegative_poll_age_valid": (
            nonnegative_age_valid
        ),
        "snapshot_date_identity_valid": (
            snapshot_date_identity_valid
        ),
        "no_unknown_question_ids": (
            len(unknown_question_ids) == 0
        ),
        "all_snapshot_pairs_present": (
            len(missing_snapshot_pairs) == 0
        ),
        "monotonic_accumulation_valid": (
            len(monotonic_failures) == 0
        ),
        "zero_day_counts_match_warehouse": (
            all(
                zero_day_counts_match.values()
            )
        ),
    }

    for name, passed in checks.items():
        if not passed:
            failed_checks.append(name)

    return {
        "status": (
            "PASSED"
            if not failed_checks
            else "FAILED"
        ),
        "warehouse_rows": int(
            len(warehouse)
        ),
        "snapshot_rows": int(
            len(snapshots)
        ),
        "summary_rows": int(
            len(summary)
        ),
        "cycles": sorted(
            int(value)
            for value in (
                warehouse["cycle"]
                .dropna()
                .unique()
            )
        ),
        "snapshot_days": SNAPSHOT_DAYS,
        "duplicate_snapshot_question_rows": int(
            duplicate_rows.sum()
        ),
        "availability_valid": bool(
            availability_valid
        ),
        "nonnegative_poll_age_valid": bool(
            nonnegative_age_valid
        ),
        "snapshot_date_identity_valid": bool(
            snapshot_date_identity_valid
        ),
        "unknown_question_ids": (
            unknown_question_ids
        ),
        "missing_snapshot_pairs": [
            {
                "cycle": cycle,
                "snapshot_days_before_election": days,
            }
            for cycle, days in (
                missing_snapshot_pairs
            )
        ],
        "monotonic_accumulation_failures": (
            monotonic_failures
        ),
        "zero_day_counts_match_warehouse": (
            zero_day_counts_match
        ),
        "failed_checks": failed_checks,
    }


def main() -> int:
    print(
        "Generic Ballot Historical Snapshot Builder"
    )
    print("=" * 52)

    SNAPSHOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    warehouse = load_warehouse()

    print(
        f"Warehouse rows: "
        f"{len(warehouse):,}"
    )

    snapshots = build_snapshots(
        warehouse
    )

    summary = build_summary(
        snapshots
    )

    increments = build_increment_counts(
        snapshots
    )

    validation = validate_snapshots(
        warehouse,
        snapshots,
        summary,
    )

    snapshots.to_csv(
        SNAPSHOT_PATH,
        index=False,
        date_format="%Y-%m-%d",
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
        date_format="%Y-%m-%d",
    )

    increments.to_csv(
        INCREMENT_PATH,
        index=False,
    )

    VALIDATION_PATH.write_text(
        json.dumps(
            validation,
            indent=2,
            default=json_safe,
        )
        + "\n"
    )

    manifest = {
        "builder": str(
            Path(__file__).resolve().relative_to(
                REPOSITORY_ROOT
            )
        ),
        "warehouse_file": str(
            WAREHOUSE_PATH.relative_to(
                REPOSITORY_ROOT
            )
        ),
        "warehouse_file_sha256": sha256_file(
            WAREHOUSE_PATH
        ),
        "snapshot_file": str(
            SNAPSHOT_PATH.relative_to(
                REPOSITORY_ROOT
            )
        ),
        "snapshot_file_sha256": sha256_file(
            SNAPSHOT_PATH
        ),
        "snapshot_days": SNAPSHOT_DAYS,
        "warehouse_rows": int(
            len(warehouse)
        ),
        "snapshot_rows": int(
            len(snapshots)
        ),
        "summary_rows": int(
            len(summary)
        ),
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

    print()
    print("Snapshot results")
    print("-" * 52)
    print(
        f"Snapshot rows: "
        f"{len(snapshots):,}"
    )
    print(
        f"Summary rows: "
        f"{len(summary):,}"
    )

    print()
    print("Unique questions by snapshot")
    print("-" * 52)

    display = summary[
        [
            "cycle",
            "snapshot_days_before_election",
            "unique_questions",
            "unique_polls",
            "latest_poll_end_date",
        ]
    ]

    print(
        display.to_string(index=False)
    )

    print()
    print("Validation")
    print("-" * 52)
    print(
        f"Status: "
        f"{validation['status']}"
    )
    print(
        "Failed checks: "
        + (
            ", ".join(
                validation["failed_checks"]
            )
            if validation["failed_checks"]
            else "None"
        )
    )

    print()
    print("Outputs")
    print("-" * 52)

    for path in [
        SNAPSHOT_PATH,
        SUMMARY_PATH,
        MANIFEST_PATH,
        VALIDATION_PATH,
        INCREMENT_PATH,
    ]:
        print(
            path.relative_to(
                REPOSITORY_ROOT
            )
        )

    return (
        0
        if validation["status"] == "PASSED"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
