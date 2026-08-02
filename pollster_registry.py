from __future__ import annotations

from pathlib import Path
import re
import unicodedata

import pandas as pd


DEFAULT_REGISTRY_PATH = Path(
    "inputs/pollster_registry.csv"
)


def normalize_pollster_key(value: object) -> str:
    """
    Produce a conservative matching key for pollster names.

    Handles capitalization, punctuation, slash, ampersand, hyphen,
    underscore, and whitespace differences. It deliberately avoids
    aggressive fuzzy matching.
    """
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()

    text = text.replace("&", " and ")
    text = re.sub(r"[/_-]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_pollster_registry(
    path: Path = DEFAULT_REGISTRY_PATH,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Pollster registry not found: {path}"
        )

    registry = pd.read_csv(path, low_memory=False)

    required = {
        "canonical_pollster",
        "normalized_pollster_key",
        "aliases",
        "pollster_house_effect_dem",
    }

    missing = required - set(registry.columns)

    if missing:
        raise ValueError(
            "Pollster registry is missing columns: "
            + ", ".join(sorted(missing))
        )

    registry = registry.copy()

    if "active" not in registry.columns:
        registry["active"] = True

    active = (
        registry["active"]
        .fillna(True)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )

    registry = registry.loc[active].copy()

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

    registry["pollster_house_effect_dem"] = (
        pd.to_numeric(
            registry["pollster_house_effect_dem"],
            errors="coerce",
        )
        .fillna(0.0)
    )

    return registry


def build_alias_lookup(
    registry: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    lookup: dict[str, dict[str, object]] = {}

    for _, row in registry.iterrows():
        canonical = str(
            row["canonical_pollster"]
        ).strip()

        aliases = [
            canonical,
            str(row.get("normalized_pollster_key", "")),
        ]

        aliases.extend(
            alias.strip()
            for alias in str(
                row.get("aliases", "")
            ).split("|")
            if alias.strip()
        )

        record = {
            "canonical_pollster": canonical,
            "pollster_house_effect_dem": float(
                row["pollster_house_effect_dem"]
            ),
            "house_effect_confidence": str(
                row.get(
                    "house_effect_confidence",
                    "",
                )
            ).strip(),
            "house_effect_notes": str(
                row.get(
                    "house_effect_notes",
                    "",
                )
            ).strip(),
        }

        for alias in aliases:
            key = normalize_pollster_key(alias)

            if not key:
                continue

            existing = lookup.get(key)

            if (
                existing is not None
                and existing["canonical_pollster"]
                != canonical
            ):
                raise ValueError(
                    "Conflicting registry aliases for normalized "
                    f"key {key!r}: "
                    f"{existing['canonical_pollster']!r} and "
                    f"{canonical!r}"
                )

            lookup[key] = record

    return lookup


def apply_pollster_registry(
    polls: pd.DataFrame,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> pd.DataFrame:
    if "pollster" not in polls.columns:
        raise ValueError(
            "Poll data must contain a pollster column."
        )

    registry = load_pollster_registry(registry_path)
    lookup = build_alias_lookup(registry)

    out = polls.copy()

    out["pollster_raw"] = (
        out["pollster"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    out["pollster_normalized_key"] = (
        out["pollster_raw"].map(
            normalize_pollster_key
        )
    )

    records = []

    for key in out["pollster_normalized_key"]:
        if not key:
            records.append(
                {
                    "canonical_pollster": "",
                    "pollster_match_method": (
                        "missing_pollster"
                    ),
                    "pollster_house_effect_dem": 0.0,
                    "house_effect_confidence": "",
                    "house_effect_notes": "",
                }
            )
            continue

        record = lookup.get(key)

        if record is None:
            records.append(
                {
                    "canonical_pollster": "",
                    "pollster_match_method": (
                        "unmatched"
                    ),
                    "pollster_house_effect_dem": 0.0,
                    "house_effect_confidence": "",
                    "house_effect_notes": "",
                }
            )
        else:
            records.append(
                {
                    **record,
                    "pollster_match_method": (
                        "normalized_alias"
                    ),
                }
            )

    matched = pd.DataFrame(
        records,
        index=out.index,
    )

    for column in matched.columns:
        out[column] = matched[column]

    return out
