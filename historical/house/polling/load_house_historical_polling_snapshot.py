from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_SNAPSHOT_PATH = (
    REPO_ROOT
    / "historical"
    / "house"
    / "polling"
    / "processed"
    / "house_historical_polling_snapshots.csv"
)

SUPPORTED_CYCLES = (
    2018,
    2020,
    2022,
)

SUPPORTED_DAYS_OUT = (
    120,
    90,
    60,
    30,
    14,
    7,
    1,
)

POLLING_COLUMNS = [
    "district_id",
    "snapshot_date",
    "days_out",
    "polling_margin_dem",
    "poll_count",
    "polling_active",
    "latest_poll_end_date",
    "avg_poll_age_days",
    "total_poll_weight",
    "effective_poll_count",
    "largest_pollster_weight_share",
    "only_partisan_or_internal_polls",
    "polling_notes",
]


def load_house_historical_polling_snapshot(
    *,
    cycle: int,
    days_out: int,
    snapshot_path: Path = DEFAULT_SNAPSHOT_PATH,
) -> pd.DataFrame:
    """
    Load one validated 435-row historical House polling snapshot.

    This function performs no modeling and no weighting. It only
    selects and validates one cycle/days-out slice from the canonical
    historical polling snapshot warehouse.
    """
    cycle = int(cycle)
    days_out = int(days_out)

    if cycle not in SUPPORTED_CYCLES:
        raise ValueError(
            f"Unsupported polling cycle {cycle}. "
            f"Expected one of {SUPPORTED_CYCLES}."
        )

    if days_out not in SUPPORTED_DAYS_OUT:
        raise ValueError(
            f"Unsupported days_out {days_out}. "
            f"Expected one of {SUPPORTED_DAYS_OUT}."
        )

    if not snapshot_path.exists():
        raise FileNotFoundError(
            "Historical House polling snapshot warehouse "
            f"not found: {snapshot_path}"
        )

    warehouse = pd.read_csv(
        snapshot_path,
        low_memory=False,
    )

    required = {
        "forecast_cycle",
        *POLLING_COLUMNS,
    }

    missing = sorted(
        required
        - set(warehouse.columns)
    )

    if missing:
        raise ValueError(
            "Historical polling warehouse is missing "
            "required columns: "
            + ", ".join(missing)
        )

    forecast_cycle = pd.to_numeric(
        warehouse["forecast_cycle"],
        errors="coerce",
    )

    warehouse_days_out = pd.to_numeric(
        warehouse["days_out"],
        errors="coerce",
    )

    snapshot = warehouse.loc[
        forecast_cycle.eq(cycle)
        & warehouse_days_out.eq(days_out),
        POLLING_COLUMNS,
    ].copy()

    snapshot = snapshot.sort_values(
        "district_id",
        kind="mergesort",
    ).reset_index(drop=True)

    if len(snapshot) != 435:
        raise ValueError(
            f"Cycle {cycle}, days_out {days_out} "
            f"expected 435 polling rows; found {len(snapshot)}."
        )

    if snapshot[
        "district_id"
    ].duplicated().any():
        duplicates = (
            snapshot.loc[
                snapshot[
                    "district_id"
                ].duplicated(False),
                "district_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Historical polling snapshot contains "
            "duplicate district IDs: "
            + ", ".join(
                duplicates[:20]
            )
        )

    snapshot["district_id"] = (
        snapshot["district_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if snapshot[
        "district_id"
    ].eq("").any():
        raise ValueError(
            "Historical polling snapshot contains "
            "blank district IDs."
        )

    poll_count = pd.to_numeric(
        snapshot["poll_count"],
        errors="coerce",
    ).fillna(0)

    polling_active = (
        snapshot["polling_active"]
        .fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            {
                "true",
                "1",
                "yes",
                "y",
            }
        )
    )

    if not polling_active.eq(
        poll_count.gt(0)
    ).all():
        raise ValueError(
            "Historical polling snapshot contains "
            "polling-active flags inconsistent with poll counts."
        )

    return snapshot
