from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from pollster_registry import normalize_pollster_key


POLL_PATH = Path("inputs/house_manual_polls.csv")
REGISTRY_PATH = Path("inputs/pollster_registry.csv")


def parse_active(series: pd.Series) -> pd.Series:
    return (
        series.fillna(True)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def sync_house_pollster_registry() -> int:
    if not POLL_PATH.exists():
        raise FileNotFoundError(POLL_PATH)

    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(REGISTRY_PATH)

    polls = pd.read_csv(
        POLL_PATH,
        low_memory=False,
    )

    registry = pd.read_csv(
        REGISTRY_PATH,
        low_memory=False,
    )

    if "pollster" not in polls.columns:
        raise ValueError(
            "House manual polls are missing the pollster column."
        )

    required_registry_columns = {
        "canonical_pollster": "",
        "normalized_pollster_key": "",
        "aliases": "",
        "pollster_house_effect_dem": 0.0,
        "house_effect_confidence": "low",
        "house_effect_notes": "",
        "active": True,
    }

    for column, default in required_registry_columns.items():
        if column not in registry.columns:
            registry[column] = default

    registry["canonical_pollster"] = (
        registry["canonical_pollster"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    registry["normalized_pollster_key"] = (
        registry["normalized_pollster_key"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    existing_keys: set[str] = set()

    for _, row in registry.iterrows():
        values = [
            row.get("canonical_pollster", ""),
            row.get("normalized_pollster_key", ""),
        ]

        values.extend(
            alias.strip()
            for alias in str(
                row.get("aliases", "")
            ).split("|")
            if alias.strip()
        )

        for value in values:
            key = normalize_pollster_key(value)

            if key:
                existing_keys.add(key)

    raw_pollsters = (
        polls["pollster"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    additions: list[dict[str, object]] = []

    for pollster in raw_pollsters:
        if not pollster:
            continue

        key = normalize_pollster_key(pollster)

        if not key or key in existing_keys:
            continue

        additions.append(
            {
                "canonical_pollster": pollster,
                "normalized_pollster_key": key,
                "aliases": pollster,
                "pollster_house_effect_dem": 0.0,
                "house_effect_confidence": "low",
                "house_effect_notes": (
                    "Automatically added from House manual polls; "
                    "house effect pending review."
                ),
                "active": True,
            }
        )

        existing_keys.add(key)

    if not additions:
        print(
            "House pollster registry already contains every "
            "nonblank manual-poll pollster."
        )
        return 0

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup = REGISTRY_PATH.with_name(
        f"{REGISTRY_PATH.stem}."
        f"before_house_auto_sync_{timestamp}"
        f"{REGISTRY_PATH.suffix}"
    )

    registry.to_csv(
        backup,
        index=False,
    )

    updated = pd.concat(
        [
            registry,
            pd.DataFrame(additions),
        ],
        ignore_index=True,
    )

    updated = updated.sort_values(
        "canonical_pollster",
        key=lambda values: (
            values.fillna("")
            .astype(str)
            .str.lower()
        ),
        kind="mergesort",
    ).reset_index(drop=True)

    updated.to_csv(
        REGISTRY_PATH,
        index=False,
    )

    print(f"Backup written to: {backup}")
    print(
        f"Added {len(additions)} new House pollster(s):"
    )

    for row in additions:
        print(
            f"  - {row['canonical_pollster']}"
        )

    return len(additions)


if __name__ == "__main__":
    sync_house_pollster_registry()
