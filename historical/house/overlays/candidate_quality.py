"""Candidate-quality overlay for historical House backtests.

This module joins the leakage-safe historical candidate WAR warehouse onto
canonical race inputs without modifying the factual historical warehouse.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_WAR_PATH = Path(
    "historical/house/backtests/outputs/"
    "candidate_war/house_historical_candidate_war.csv"
)

RACE_KEY_COLUMNS = ["cycle", "race_id"]
WAR_VALUE_COLUMN = "candidate_war_adjustment_dem"


class CandidateQualityOverlayError(ValueError):
    """Raised when the candidate-quality overlay fails validation."""


def _normalize_cycle(series: pd.Series) -> pd.Series:
    """Return cycles as nullable integers, rejecting invalid values."""
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.isna().any():
        examples = series.loc[numeric.isna()].head(10).tolist()
        raise CandidateQualityOverlayError(
            "Candidate-quality overlay contains invalid cycle values. "
            f"Examples: {examples}"
        )

    return numeric.astype("int64")


def _normalize_race_id(series: pd.Series) -> pd.Series:
    """Return trimmed race identifiers, rejecting blank values."""
    normalized = series.astype("string").str.strip()

    invalid = normalized.isna() | normalized.eq("")

    if invalid.any():
        raise CandidateQualityOverlayError(
            "Candidate-quality overlay contains blank race_id values."
        )

    return normalized.astype(str)


def validate_race_keys(
    frame: pd.DataFrame,
    *,
    frame_name: str,
) -> pd.DataFrame:
    """Validate and normalize the canonical race key columns."""
    validated = frame.copy()

    if "cycle" not in validated.columns:
        if "forecast_cycle" in validated.columns:
            validated = validated.rename(
                columns={"forecast_cycle": "cycle"}
            )
        elif "target_cycle" in validated.columns:
            validated = validated.rename(
                columns={"target_cycle": "cycle"}
            )

    missing = [
        column
        for column in RACE_KEY_COLUMNS
        if column not in validated.columns
    ]

    if missing:
        raise CandidateQualityOverlayError(
            f"{frame_name} is missing required key columns: {missing}"
        )

    validated["cycle"] = _normalize_cycle(validated["cycle"])
    validated["race_id"] = _normalize_race_id(validated["race_id"])

    duplicate_mask = validated.duplicated(
        RACE_KEY_COLUMNS,
        keep=False,
    )

    if duplicate_mask.any():
        examples = (
            validated.loc[duplicate_mask, RACE_KEY_COLUMNS]
            .sort_values(RACE_KEY_COLUMNS)
            .head(20)
            .to_dict("records")
        )

        raise CandidateQualityOverlayError(
            f"{frame_name} contains duplicate cycle/race_id keys. "
            f"Examples: {examples}"
        )

    return validated


def load_candidate_quality_table(
    war_path: Path | str = DEFAULT_WAR_PATH,
) -> pd.DataFrame:
    """Load and validate the leakage-safe historical WAR table."""
    path = Path(war_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Historical candidate WAR table not found: {path}"
        )

    war = pd.read_csv(
        path,
        dtype={"race_id": "string"},
        low_memory=False,
    )

    war = validate_race_keys(
        war,
        frame_name="Historical candidate WAR table",
    )

    if WAR_VALUE_COLUMN not in war.columns:
        raise CandidateQualityOverlayError(
            "Historical candidate WAR table is missing required column "
            f"{WAR_VALUE_COLUMN!r}."
        )

    war[WAR_VALUE_COLUMN] = pd.to_numeric(
        war[WAR_VALUE_COLUMN],
        errors="coerce",
    )

    nonfinite = (
        war[WAR_VALUE_COLUMN].notna()
        & ~np.isfinite(war[WAR_VALUE_COLUMN])
    )

    if nonfinite.any():
        examples = (
            war.loc[
                nonfinite,
                RACE_KEY_COLUMNS + [WAR_VALUE_COLUMN],
            ]
            .head(20)
            .to_dict("records")
        )

        raise CandidateQualityOverlayError(
            "Historical candidate WAR table contains non-finite "
            f"adjustments. Examples: {examples}"
        )

    if "future_war_used" in war.columns:
        future_used = (
            war["future_war_used"]
            .astype("string")
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes", "y"})
        )

        if future_used.any():
            examples = (
                war.loc[
                    future_used,
                    RACE_KEY_COLUMNS,
                ]
                .head(20)
                .to_dict("records")
            )

            raise CandidateQualityOverlayError(
                "Historical candidate WAR table indicates future WAR "
                f"usage. Examples: {examples}"
            )

    return war[
        RACE_KEY_COLUMNS + [WAR_VALUE_COLUMN]
    ].copy()


def build_candidate_quality_overlay(
    races: pd.DataFrame,
    *,
    war_table: pd.DataFrame | None = None,
    war_path: Path | str = DEFAULT_WAR_PATH,
    fill_missing: float = 0.0,
) -> pd.Series:
    """Return WAR adjustments aligned exactly to the supplied race frame.

    The returned Series:
    - has the same index as ``races``;
    - preserves the race row count and order;
    - uses zero for unmatched races by default;
    - is named ``candidate_war_adjustment_dem``.
    """
    validated_races = validate_race_keys(
        races,
        frame_name="Canonical race frame",
    )

    if war_table is None:
        validated_war = load_candidate_quality_table(war_path)
    else:
        validated_war = validate_race_keys(
            war_table,
            frame_name="Candidate WAR table",
        )

        if WAR_VALUE_COLUMN not in validated_war.columns:
            raise CandidateQualityOverlayError(
                "Candidate WAR table is missing required column "
                f"{WAR_VALUE_COLUMN!r}."
            )

        validated_war[WAR_VALUE_COLUMN] = pd.to_numeric(
            validated_war[WAR_VALUE_COLUMN],
            errors="coerce",
        )

        validated_war = validated_war[
            RACE_KEY_COLUMNS + [WAR_VALUE_COLUMN]
        ].copy()

    left = validated_races[
        RACE_KEY_COLUMNS
    ].copy()

    left["_overlay_row_order"] = np.arange(
        len(left),
        dtype=np.int64,
    )

    merged = left.merge(
        validated_war,
        on=RACE_KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if len(merged) != len(races):
        raise CandidateQualityOverlayError(
            "Candidate-quality merge changed the race row count: "
            f"{len(races)} before, {len(merged)} after."
        )

    merged = merged.sort_values(
        "_overlay_row_order",
        kind="stable",
    )

    adjustment = (
        pd.to_numeric(
            merged[WAR_VALUE_COLUMN],
            errors="coerce",
        )
        .fillna(float(fill_missing))
        .astype(float)
    )

    if not np.isfinite(adjustment.to_numpy()).all():
        raise CandidateQualityOverlayError(
            "Candidate-quality overlay produced non-finite values."
        )

    return pd.Series(
        adjustment.to_numpy(),
        index=races.index,
        name=WAR_VALUE_COLUMN,
        dtype=float,
    )
