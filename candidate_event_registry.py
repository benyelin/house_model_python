from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parent
    / "inputs"
    / "candidate_event_registry.csv"
)

MAX_ABSOLUTE_ADJUSTMENT = 3.0
ARITHMETIC_TOLERANCE = 1e-9

REQUIRED_COLUMNS = [
    "event_id",
    "cycle",
    "chamber",
    "race_id",
    "candidate_name",
    "candidate_party",
    "event_type",
    "event_scope",
    "event_status",
    "severity_category",
    "credibility_level",
    "media_salience",
    "reported_date",
    "effective_date",
    "review_date",
    "expiration_date",
    "baseline_event_adjustment_dem",
    "analyst_modifier_dem",
    "candidate_event_adjustment_dem",
    "confidence",
    "source_summary",
    "source_url",
    "candidate_response",
    "analyst_rationale",
    "polling_supersession_mode",
    "active",
]

VALID_CHAMBERS = {
    "house",
    "senate",
}

VALID_PARTIES = {
    "D",
    "R",
    "I",
    "OTHER",
    "UNKNOWN",
}

VALID_EVENT_TYPES = {
    "personal_controversy",
    "consensual_affair",
    "ethics_allegation",
    "financial_misconduct",
    "harassment_allegation",
    "assault_allegation",
    "domestic_abuse_allegation",
    "criminal_investigation",
    "criminal_charge",
    "criminal_conviction",
    "health_event",
    "withdrawal",
    "death",
    "replacement_nominee",
    "party_switch",
    "major_endorsement",
    "other",
}

VALID_EVENT_SCOPES = {
    "candidate_personal",
    "campaign",
    "ballot_access",
    "nomination",
    "structural",
    "legal",
    "health",
    "other",
}


VALID_EVENT_STATUSES = {
    "alleged",
    "reported",
    "investigated",
    "charged",
    "admitted",
    "substantiated",
    "convicted",
    "resolved",
    "withdrawn",
    "deceased",
    "replaced",
    "other",
}

VALID_SEVERITY_CATEGORIES = {
    "minor",
    "moderate",
    "serious",
    "severe",
    "structural",
}

VALID_CREDIBILITY_LEVELS = {
    "low",
    "medium",
    "high",
    "very_high",
}

VALID_MEDIA_SALIENCE = {
    "low",
    "medium",
    "high",
    "very_high",
}

VALID_CONFIDENCE_LEVELS = {
    "low",
    "medium",
    "high",
}

VALID_POLLING_SUPERSESSION_MODES = {
    "manual_review",
    "not_applicable",
}

STRUCTURAL_EVENT_TYPES = {
    "withdrawal",
    "death",
    "replacement_nominee",
}


@dataclass(frozen=True)
class CandidateEventValidationResult:
    rows: int
    active_rows: int
    nonzero_active_rows: int
    chambers: tuple[str, ...]
    maximum_absolute_adjustment: float


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""

    return str(value).strip()


def normalize_chamber(value: object) -> str:
    return normalize_text(value).lower()


def normalize_party(value: object) -> str:
    text = normalize_text(value).upper()

    aliases = {
        "DEMOCRAT": "D",
        "DEMOCRATIC": "D",
        "REPUBLICAN": "R",
        "GOP": "R",
        "INDEPENDENT": "I",
    }

    return aliases.get(text, text)


def normalize_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value

    if value is None or pd.isna(value):
        return False

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def normalize_event_id(value: object) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9_-]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def _require_columns(frame: pd.DataFrame) -> None:
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            "Candidate event registry is missing required columns: "
            + ", ".join(missing)
        )


def _parse_dates(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    out = frame.copy()

    for column in [
        "reported_date",
        "effective_date",
        "review_date",
        "expiration_date",
    ]:
        out[column] = pd.to_datetime(
            out[column],
            errors="coerce",
        )

    return out


def _validate_allowed_values(
    frame: pd.DataFrame,
    column: str,
    allowed: set[str],
) -> None:
    values = (
        frame[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    invalid = sorted(
        set(values[values.ne("")])
        - allowed
    )

    if invalid:
        raise ValueError(
            f"Invalid {column} value(s): {invalid}. "
            f"Allowed values: {sorted(allowed)}"
        )


def _validate_nonblank(
    frame: pd.DataFrame,
    columns: Iterable[str],
    mask: pd.Series,
    label: str,
) -> None:
    for column in columns:
        blank = (
            frame[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
        )

        bad = frame.loc[
            mask & blank,
            ["event_id", column],
        ]

        if not bad.empty:
            raise ValueError(
                f"{label} rows require nonblank {column}. "
                f"Examples:\n{bad.head(20).to_string(index=False)}"
            )


def validate_candidate_event_registry(
    frame: pd.DataFrame,
    *,
    as_of: date | None = None,
) -> CandidateEventValidationResult:
    _require_columns(frame)

    out = frame.copy()

    out["event_id"] = out["event_id"].apply(
        normalize_event_id
    )
    out["chamber"] = out["chamber"].apply(
        normalize_chamber
    )
    out["candidate_party"] = out[
        "candidate_party"
    ].apply(normalize_party)
    out["active"] = out["active"].apply(
        normalize_bool
    )

    for column in [
        "event_type",
        "event_scope",
        "event_status",
        "severity_category",
        "credibility_level",
        "media_salience",
        "confidence",
        "polling_supersession_mode",
    ]:
        out[column] = (
            out[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

    if out["event_id"].eq("").any():
        raise ValueError(
            "Every candidate event row requires event_id."
        )

    duplicates = out.loc[
        out["event_id"].duplicated(keep=False),
        ["event_id", "chamber", "race_id"],
    ]

    if not duplicates.empty:
        raise ValueError(
            "Duplicate event_id values found:\n"
            + duplicates.to_string(index=False)
        )

    _validate_allowed_values(
        out,
        "chamber",
        VALID_CHAMBERS,
    )
    _validate_allowed_values(
        out,
        "candidate_party",
        VALID_PARTIES,
    )
    _validate_allowed_values(
        out,
        "event_type",
        VALID_EVENT_TYPES,
    )
    _validate_allowed_values(
        out,
        "event_scope",
        VALID_EVENT_SCOPES,
    )
    _validate_allowed_values(
        out,
        "event_status",
        VALID_EVENT_STATUSES,
    )
    _validate_allowed_values(
        out,
        "severity_category",
        VALID_SEVERITY_CATEGORIES,
    )
    _validate_allowed_values(
        out,
        "credibility_level",
        VALID_CREDIBILITY_LEVELS,
    )
    _validate_allowed_values(
        out,
        "media_salience",
        VALID_MEDIA_SALIENCE,
    )
    _validate_allowed_values(
        out,
        "confidence",
        VALID_CONFIDENCE_LEVELS,
    )
    _validate_allowed_values(
        out,
        "polling_supersession_mode",
        VALID_POLLING_SUPERSESSION_MODES,
    )

    for column in [
        "cycle",
        "baseline_event_adjustment_dem",
        "analyst_modifier_dem",
        "candidate_event_adjustment_dem",
    ]:
        out[column] = pd.to_numeric(
            out[column],
            errors="coerce",
        )

    if out["cycle"].isna().any():
        raise ValueError(
            "Every candidate event row requires numeric cycle."
        )

    numeric_columns = [
        "baseline_event_adjustment_dem",
        "analyst_modifier_dem",
        "candidate_event_adjustment_dem",
    ]

    for column in numeric_columns:
        if out[column].isna().any():
            raise ValueError(
                f"Every candidate event row requires numeric "
                f"{column}."
            )

    reconstructed = (
        out["baseline_event_adjustment_dem"]
        + out["analyst_modifier_dem"]
    )

    arithmetic_error = (
        reconstructed
        - out["candidate_event_adjustment_dem"]
    ).abs()

    if arithmetic_error.gt(
        ARITHMETIC_TOLERANCE
    ).any():
        bad = out.loc[
            arithmetic_error.gt(
                ARITHMETIC_TOLERANCE
            ),
            [
                "event_id",
                "baseline_event_adjustment_dem",
                "analyst_modifier_dem",
                "candidate_event_adjustment_dem",
            ],
        ]

        raise ValueError(
            "Candidate event adjustment does not equal "
            "baseline plus analyst modifier:\n"
            + bad.to_string(index=False)
        )

    excessive = out[
        "candidate_event_adjustment_dem"
    ].abs().gt(MAX_ABSOLUTE_ADJUSTMENT)

    if excessive.any():
        bad = out.loc[
            excessive,
            [
                "event_id",
                "candidate_event_adjustment_dem",
            ],
        ]

        raise ValueError(
            "Candidate event adjustments exceed the "
            f"+/-{MAX_ABSOLUTE_ADJUSTMENT:.1f} cap:\n"
            + bad.to_string(index=False)
        )

    active = out["active"]
    nonzero_active = (
        active
        & out[
            "candidate_event_adjustment_dem"
        ].abs().gt(ARITHMETIC_TOLERANCE)
    )

    _validate_nonblank(
        out,
        [
            "race_id",
            "candidate_name",
            "candidate_party",
            "event_type",
            "event_scope",
            "event_status",
            "severity_category",
            "credibility_level",
            "media_salience",
            "effective_date",
            "review_date",
            "confidence",
            "source_summary",
            "candidate_response",
            "analyst_rationale",
            "polling_supersession_mode",
        ],
        nonzero_active,
        "Active nonzero candidate-event",
    )

    out = _parse_dates(out)

    if out.loc[
        nonzero_active,
        "effective_date",
    ].isna().any():
        raise ValueError(
            "Active nonzero events require valid effective_date."
        )

    if out.loc[
        nonzero_active,
        "review_date",
    ].isna().any():
        raise ValueError(
            "Active nonzero events require valid review_date."
        )

    review_before_effective = (
        nonzero_active
        & out["review_date"].lt(
            out["effective_date"]
        )
    )

    if review_before_effective.any():
        raise ValueError(
            "review_date cannot precede effective_date."
        )

    expiration_before_effective = (
        out["expiration_date"].notna()
        & out["effective_date"].notna()
        & out["expiration_date"].lt(
            out["effective_date"]
        )
    )

    if expiration_before_effective.any():
        raise ValueError(
            "expiration_date cannot precede effective_date."
        )

    today = as_of or date.today()
    today_timestamp = pd.Timestamp(today)

    overdue_review = (
        nonzero_active
        & out["review_date"].lt(today_timestamp)
    )

    if overdue_review.any():
        bad = out.loc[
            overdue_review,
            [
                "event_id",
                "review_date",
                "candidate_event_adjustment_dem",
            ],
        ]

        raise ValueError(
            "Active candidate events are past their mandatory "
            "review date:\n"
            + bad.to_string(index=False)
        )

    structural_with_numeric_penalty = (
        out["event_type"].isin(
            STRUCTURAL_EVENT_TYPES
        )
        & out[
            "candidate_event_adjustment_dem"
        ].abs().gt(ARITHMETIC_TOLERANCE)
    )

    if structural_with_numeric_penalty.any():
        bad = out.loc[
            structural_with_numeric_penalty,
            [
                "event_id",
                "event_type",
                "candidate_event_adjustment_dem",
            ],
        ]

        raise ValueError(
            "Withdrawal, death, and replacement events require "
            "separate structural handling and cannot use a normal "
            "numeric candidate-event adjustment:\n"
            + bad.to_string(index=False)
        )

    maximum_absolute = float(
        out[
            "candidate_event_adjustment_dem"
        ].abs().max()
    ) if len(out) else 0.0

    chambers = tuple(
        sorted(
            out.loc[
                out["chamber"].ne(""),
                "chamber",
            ].unique()
        )
    )

    return CandidateEventValidationResult(
        rows=len(out),
        active_rows=int(active.sum()),
        nonzero_active_rows=int(
            nonzero_active.sum()
        ),
        chambers=chambers,
        maximum_absolute_adjustment=maximum_absolute,
    )


def load_candidate_event_registry(
    registry_path: Path | str = DEFAULT_REGISTRY_PATH,
    *,
    as_of: date | None = None,
) -> pd.DataFrame:
    path = Path(registry_path)

    if not path.exists():
        raise FileNotFoundError(path)

    frame = pd.read_csv(
        path,
        low_memory=False,
    )

    validate_candidate_event_registry(
        frame,
        as_of=as_of,
    )

    out = frame.copy()

    out["event_id"] = out["event_id"].apply(
        normalize_event_id
    )
    out["chamber"] = out["chamber"].apply(
        normalize_chamber
    )
    out["candidate_party"] = out[
        "candidate_party"
    ].apply(normalize_party)
    out["active"] = out["active"].apply(
        normalize_bool
    )

    for column in [
        "baseline_event_adjustment_dem",
        "analyst_modifier_dem",
        "candidate_event_adjustment_dem",
    ]:
        out[column] = pd.to_numeric(
            out[column],
            errors="coerce",
        ).fillna(0.0)

    out = _parse_dates(out)

    return out


def active_candidate_events(
    chamber: str,
    *,
    cycle: int,
    as_of: date,
    registry_path: Path | str = DEFAULT_REGISTRY_PATH,
) -> pd.DataFrame:
    frame = load_candidate_event_registry(
        registry_path,
        as_of=as_of,
    )

    chamber_normalized = normalize_chamber(
        chamber
    )

    mask = (
        frame["active"]
        & frame["chamber"].eq(
            chamber_normalized
        )
        & frame["cycle"].eq(int(cycle))
        & frame["effective_date"].le(
            pd.Timestamp(as_of)
        )
        & (
            frame["expiration_date"].isna()
            | frame["expiration_date"].ge(
                pd.Timestamp(as_of)
            )
        )
    )

    return frame.loc[mask].copy()


def summarize_candidate_events(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "chamber",
                "cycle",
                "race_id",
                "candidate_event_adjustment_dem",
                "candidate_event_count",
                "candidate_event_ids",
                "candidate_event_summary",
            ]
        )

    grouped = (
        frame.groupby(
            [
                "chamber",
                "cycle",
                "race_id",
            ],
            dropna=False,
        )
        .agg(
            candidate_event_adjustment_dem=(
                "candidate_event_adjustment_dem",
                "sum",
            ),
            candidate_event_count=(
                "event_id",
                "count",
            ),
            candidate_event_ids=(
                "event_id",
                lambda values: "|".join(
                    values.astype(str)
                ),
            ),
            candidate_event_summary=(
                "source_summary",
                lambda values: " | ".join(
                    values.astype(str)
                ),
            ),
        )
        .reset_index()
    )

    grouped[
        "candidate_event_adjustment_dem"
    ] = grouped[
        "candidate_event_adjustment_dem"
    ].clip(
        lower=-MAX_ABSOLUTE_ADJUSTMENT,
        upper=MAX_ABSOLUTE_ADJUSTMENT,
    )

    return grouped
