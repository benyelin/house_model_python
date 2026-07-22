"""Reusable historical generic-ballot polling aggregation engine.

This module converts a leakage-free polling snapshot into one aggregated
Democratic generic-ballot margin under an explicit aggregation specification.

The module intentionally contains no model-selection or bakeoff logic.
Backtests and production code should both call this same aggregation engine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd


RecencyMode = Literal[
    "equal",
    "half_life",
]

PopulationMode = Literal[
    "all",
    "lv_preferred",
    "weighted",
]

PartisanMode = Literal[
    "include",
    "downweight",
    "exclude",
]

DuplicateMode = Literal[
    "question",
    "poll",
]

QuestionSelectionMode = Literal[
    "all",
    "prefer_lv",
    "prefer_rv",
    "largest_sample",
]


@dataclass(frozen=True)
class GenericBallotAggregationSpec:
    """Configuration for one generic-ballot aggregation calculation."""

    lookback_days: int | None = None

    recency_mode: RecencyMode = "equal"
    recency_half_life_days: float | None = None

    population_mode: PopulationMode = "all"
    likely_voter_weight: float = 1.00
    registered_voter_weight: float = 0.85
    voter_weight: float = 0.75
    adult_weight: float = 0.60
    unknown_population_weight: float = 0.60

    sample_size_weighting: bool = False
    pollster_quality_weighting: bool = False

    partisan_mode: PartisanMode = "include"
    partisan_weight: float = 0.50

    question_selection_mode: QuestionSelectionMode = "all"

    duplicate_mode: DuplicateMode = "question"

    minimum_effective_weight: float = 1e-12


@dataclass(frozen=True)
class GenericBallotAggregationResult:
    """Output and diagnostics for one aggregated polling snapshot."""

    cycle: int
    snapshot_days_before_election: int
    snapshot_date: str

    estimate_margin_dem: float

    input_question_rows: int
    retained_question_rows: int
    aggregated_rows: int

    unique_polls: int
    unique_pollsters: int

    total_weight: float
    effective_sample_size: float

    weighted_mean_poll_age_days: float
    oldest_poll_age_days: float
    newest_poll_age_days: float

    specification: dict


REQUIRED_COLUMNS = {
    "cycle",
    "snapshot_days_before_election",
    "snapshot_date",
    "poll_id",
    "question_id",
    "pollster",
    "poll_age_days",
    "sample_size",
    "population",
    "numeric_grade",
    "pollscore",
    "partisan",
    "internal",
    "two_party_margin_dem",
}


def _normalize_bool(series: pd.Series) -> pd.Series:
    """Convert common boolean representations to a Boolean Series."""

    normalized = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    return normalized.isin(
        {
            "true",
            "1",
            "yes",
            "y",
            "t",
        }
    )


def _normalize_population(value: object) -> str:
    """Normalize FiveThirtyEight population labels."""

    if pd.isna(value):
        return "unknown"

    normalized = str(value).strip().lower()

    mapping = {
        "lv": "lv",
        "likely voters": "lv",
        "likely voter": "lv",
        "rv": "rv",
        "registered voters": "rv",
        "registered voter": "rv",
        "v": "v",
        "voters": "v",
        "voter": "v",
        "a": "a",
        "adults": "a",
        "adult": "a",
    }

    return mapping.get(
        normalized,
        normalized or "unknown",
    )


def _validate_spec(
    spec: GenericBallotAggregationSpec,
) -> None:
    """Validate aggregation specification values."""

    if (
        spec.lookback_days is not None
        and spec.lookback_days < 0
    ):
        raise ValueError(
            "lookback_days must be nonnegative or None."
        )

    if spec.recency_mode == "half_life":
        if (
            spec.recency_half_life_days is None
            or spec.recency_half_life_days <= 0
        ):
            raise ValueError(
                "Positive recency_half_life_days is required "
                "when recency_mode='half_life'."
            )

    if spec.recency_mode == "equal":
        if spec.recency_half_life_days is not None:
            raise ValueError(
                "recency_half_life_days must be None when "
                "recency_mode='equal'."
            )

    for field_name in [
        "likely_voter_weight",
        "registered_voter_weight",
        "voter_weight",
        "adult_weight",
        "unknown_population_weight",
        "partisan_weight",
        "minimum_effective_weight",
    ]:
        value = getattr(spec, field_name)

        if value < 0:
            raise ValueError(
                f"{field_name} must be nonnegative."
            )

    if spec.minimum_effective_weight <= 0:
        raise ValueError(
            "minimum_effective_weight must be positive."
        )


def _validate_snapshot(
    snapshot: pd.DataFrame,
) -> None:
    """Validate the structure of one polling snapshot."""

    missing = sorted(
        REQUIRED_COLUMNS.difference(snapshot.columns)
    )

    if missing:
        raise ValueError(
            "Snapshot is missing required columns: "
            + ", ".join(missing)
        )

    identity_columns = [
        "cycle",
        "snapshot_days_before_election",
        "snapshot_date",
    ]

    for column in identity_columns:
        if snapshot[column].nunique(dropna=False) != 1:
            raise ValueError(
                "Aggregation requires exactly one snapshot. "
                f"Column {column!r} contains multiple values."
            )


def _select_questions_within_poll(
    frame: pd.DataFrame,
    mode: QuestionSelectionMode,
) -> pd.DataFrame:
    """Select representative questions within each poll.

    Selection occurs separately within each poll. This avoids the global
    behavior of using only likely-voter questions across an entire snapshot
    merely because at least one poll reported a likely-voter sample.

    Modes
    -----
    all
        Retain every question.

    prefer_lv
        Within each poll, retain likely-voter questions when available.
        Otherwise retain registered-voter questions, then generic voter
        questions, then adult questions, and finally all unknown questions.

    prefer_rv
        Within each poll, retain registered-voter questions when available.
        Otherwise retain likely-voter questions, then generic voter
        questions, then adult questions, and finally all unknown questions.

    largest_sample
        Within each poll, retain the question row or rows with the largest
        valid sample size. If no valid sample size exists, retain all rows.
    """

    if mode == "all":
        return frame.copy()

    selected_groups: list[pd.DataFrame] = []

    for _, group in frame.groupby(
        "poll_id",
        dropna=False,
        sort=False,
    ):
        group = group.copy()

        normalized_population = group[
            "population"
        ].map(_normalize_population)

        if mode == "prefer_lv":
            preference_order = [
                "lv",
                "rv",
                "v",
                "a",
            ]

            selected = None

            for population_code in preference_order:
                matches = group[
                    normalized_population.eq(
                        population_code
                    )
                ]

                if not matches.empty:
                    selected = matches
                    break

            if selected is None:
                selected = group

            selected_groups.append(selected)
            continue

        if mode == "prefer_rv":
            preference_order = [
                "rv",
                "lv",
                "v",
                "a",
            ]

            selected = None

            for population_code in preference_order:
                matches = group[
                    normalized_population.eq(
                        population_code
                    )
                ]

                if not matches.empty:
                    selected = matches
                    break

            if selected is None:
                selected = group

            selected_groups.append(selected)
            continue

        if mode == "largest_sample":
            numeric_sample = pd.to_numeric(
                group["sample_size"],
                errors="coerce",
            )

            valid_sample = numeric_sample[
                numeric_sample.gt(0)
            ]

            if valid_sample.empty:
                selected_groups.append(group)
                continue

            largest_sample = valid_sample.max()

            selected_groups.append(
                group[
                    numeric_sample.eq(
                        largest_sample
                    )
                ]
            )
            continue

        raise ValueError(
            "Unsupported question_selection_mode: "
            f"{mode}"
        )

    if not selected_groups:
        return frame.iloc[0:0].copy()

    return pd.concat(
        selected_groups,
        ignore_index=False,
    ).copy()


def _population_weight(
    population: pd.Series,
    spec: GenericBallotAggregationSpec,
) -> pd.Series:
    """Construct population weights."""

    normalized = population.map(
        _normalize_population
    )

    if spec.population_mode == "all":
        return pd.Series(
            1.0,
            index=population.index,
            dtype=float,
        )

    if spec.population_mode == "lv_preferred":
        has_lv = normalized.eq("lv").any()

        if has_lv:
            return normalized.eq("lv").astype(float)

        has_rv = normalized.eq("rv").any()

        if has_rv:
            return normalized.eq("rv").astype(float)

        has_voter = normalized.eq("v").any()

        if has_voter:
            return normalized.eq("v").astype(float)

        return pd.Series(
            1.0,
            index=population.index,
            dtype=float,
        )

    if spec.population_mode == "weighted":
        mapping = {
            "lv": spec.likely_voter_weight,
            "rv": spec.registered_voter_weight,
            "v": spec.voter_weight,
            "a": spec.adult_weight,
        }

        return (
            normalized.map(mapping)
            .fillna(spec.unknown_population_weight)
            .astype(float)
        )

    raise ValueError(
        f"Unsupported population_mode: "
        f"{spec.population_mode}"
    )


def _recency_weight(
    poll_age_days: pd.Series,
    spec: GenericBallotAggregationSpec,
) -> pd.Series:
    """Construct recency weights."""

    ages = pd.to_numeric(
        poll_age_days,
        errors="coerce",
    )

    if spec.recency_mode == "equal":
        return pd.Series(
            1.0,
            index=ages.index,
            dtype=float,
        )

    if spec.recency_mode == "half_life":
        half_life = float(
            spec.recency_half_life_days
        )

        return np.power(
            0.5,
            ages / half_life,
        )

    raise ValueError(
        f"Unsupported recency_mode: "
        f"{spec.recency_mode}"
    )


def _sample_size_weight(
    sample_size: pd.Series,
    enabled: bool,
) -> pd.Series:
    """Construct conservative sample-size weights.

    Square-root weighting prevents very large polls from dominating an
    aggregation while still allowing larger samples to receive more weight.
    We normalize around a nominal sample size of 1,000.
    """

    if not enabled:
        return pd.Series(
            1.0,
            index=sample_size.index,
            dtype=float,
        )

    numeric = pd.to_numeric(
        sample_size,
        errors="coerce",
    )

    valid_median = numeric[
        numeric.gt(0)
    ].median()

    fallback = (
        float(valid_median)
        if pd.notna(valid_median)
        else 1000.0
    )

    numeric = (
        numeric.where(numeric.gt(0), fallback)
        .fillna(fallback)
    )

    return np.sqrt(
        numeric / 1000.0
    )


def _quality_weight(
    frame: pd.DataFrame,
    enabled: bool,
) -> pd.Series:
    """Construct conservative pollster-quality weights.

    numeric_grade is preferred when available. FiveThirtyEight grades are
    transformed to a modest multiplier rather than being treated as a direct
    probability. pollscore is used only as a fallback signal.
    """

    if not enabled:
        return pd.Series(
            1.0,
            index=frame.index,
            dtype=float,
        )

    grade = pd.to_numeric(
        frame["numeric_grade"],
        errors="coerce",
    )

    pollscore = pd.to_numeric(
        frame["pollscore"],
        errors="coerce",
    )

    grade_weight = (
        0.50
        + grade.clip(lower=0.0, upper=3.0) / 3.0
    )

    pollscore_weight = (
        1.0
        - pollscore.clip(
            lower=-2.0,
            upper=2.0,
        )
        * 0.10
    )

    quality = grade_weight.where(
        grade.notna(),
        pollscore_weight,
    )

    return (
        quality.fillna(1.0)
        .clip(lower=0.25, upper=1.50)
    )


def _partisan_weight(
    frame: pd.DataFrame,
    spec: GenericBallotAggregationSpec,
) -> pd.Series:
    """Construct partisan/internal-poll treatment weights."""

    partisan = _normalize_bool(
        frame["partisan"]
    )

    internal = _normalize_bool(
        frame["internal"]
    )

    flagged = partisan | internal

    if spec.partisan_mode == "include":
        return pd.Series(
            1.0,
            index=frame.index,
            dtype=float,
        )

    if spec.partisan_mode == "downweight":
        return pd.Series(
            np.where(
                flagged,
                spec.partisan_weight,
                1.0,
            ),
            index=frame.index,
            dtype=float,
        )

    if spec.partisan_mode == "exclude":
        return (~flagged).astype(float)

    raise ValueError(
        f"Unsupported partisan_mode: "
        f"{spec.partisan_mode}"
    )


def _collapse_poll_questions(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse multiple questions from one poll to one poll estimate.

    Each poll contributes its within-poll weighted mean margin. The poll-level
    weight is the mean question weight, preventing a poll with several
    alternative population questions from receiving several times the weight
    of a poll reporting one question.
    """

    rows: list[dict] = []

    group_columns = [
        "cycle",
        "snapshot_days_before_election",
        "snapshot_date",
        "poll_id",
    ]

    for keys, group in frame.groupby(
        group_columns,
        dropna=False,
        sort=False,
    ):
        weights = group["_weight"].to_numpy(
            dtype=float
        )

        margins = group[
            "two_party_margin_dem"
        ].to_numpy(dtype=float)

        positive = (
            np.isfinite(weights)
            & np.isfinite(margins)
            & (weights > 0)
        )

        if not positive.any():
            continue

        valid_weights = weights[positive]
        valid_margins = margins[positive]

        poll_margin = float(
            np.average(
                valid_margins,
                weights=valid_weights,
            )
        )

        poll_weight = float(
            np.mean(valid_weights)
        )

        poll_age = float(
            np.average(
                group.loc[
                    positive,
                    "poll_age_days",
                ],
                weights=valid_weights,
            )
        )

        rows.append(
            {
                "cycle": int(keys[0]),
                "snapshot_days_before_election": int(
                    keys[1]
                ),
                "snapshot_date": str(keys[2]),
                "poll_id": keys[3],
                "pollster": group[
                    "pollster"
                ].iloc[0],
                "two_party_margin_dem": poll_margin,
                "poll_age_days": poll_age,
                "_weight": poll_weight,
            }
        )

    return pd.DataFrame(rows)


def aggregate_generic_ballot_snapshot(
    snapshot: pd.DataFrame,
    spec: GenericBallotAggregationSpec,
) -> GenericBallotAggregationResult:
    """Aggregate one leakage-free generic-ballot snapshot."""

    _validate_spec(spec)
    _validate_snapshot(snapshot)

    frame = snapshot.copy()

    input_question_rows = len(frame)

    frame["two_party_margin_dem"] = pd.to_numeric(
        frame["two_party_margin_dem"],
        errors="coerce",
    )

    frame["poll_age_days"] = pd.to_numeric(
        frame["poll_age_days"],
        errors="coerce",
    )

    frame = frame[
        frame["two_party_margin_dem"].notna()
        & frame["poll_age_days"].notna()
        & frame["poll_age_days"].ge(0)
    ].copy()

    if spec.lookback_days is not None:
        frame = frame[
            frame["poll_age_days"].le(
                spec.lookback_days
            )
        ].copy()

    if frame.empty:
        raise ValueError(
            "No valid polling questions remain after "
            "lookback and validity filtering."
        )

    frame = _select_questions_within_poll(
        frame,
        spec.question_selection_mode,
    )

    if frame.empty:
        raise ValueError(
            "No polling questions remain after "
            "within-poll question selection."
        )

    frame["_recency_weight"] = _recency_weight(
        frame["poll_age_days"],
        spec,
    )

    frame["_population_weight"] = _population_weight(
        frame["population"],
        spec,
    )

    frame["_sample_size_weight"] = _sample_size_weight(
        frame["sample_size"],
        spec.sample_size_weighting,
    )

    frame["_quality_weight"] = _quality_weight(
        frame,
        spec.pollster_quality_weighting,
    )

    frame["_partisan_weight"] = _partisan_weight(
        frame,
        spec,
    )

    component_columns = [
        "_recency_weight",
        "_population_weight",
        "_sample_size_weight",
        "_quality_weight",
        "_partisan_weight",
    ]

    frame["_weight"] = frame[
        component_columns
    ].prod(axis=1)

    frame = frame[
        frame["_weight"].gt(
            spec.minimum_effective_weight
        )
        & np.isfinite(frame["_weight"])
    ].copy()

    retained_question_rows = len(frame)

    if frame.empty:
        raise ValueError(
            "No positive-weight polling questions remain."
        )

    if spec.duplicate_mode == "poll":
        aggregate_frame = _collapse_poll_questions(
            frame
        )
    elif spec.duplicate_mode == "question":
        aggregate_frame = frame.copy()
    else:
        raise ValueError(
            f"Unsupported duplicate_mode: "
            f"{spec.duplicate_mode}"
        )

    if aggregate_frame.empty:
        raise ValueError(
            "No aggregated polling rows remain."
        )

    weights = aggregate_frame[
        "_weight"
    ].to_numpy(dtype=float)

    margins = aggregate_frame[
        "two_party_margin_dem"
    ].to_numpy(dtype=float)

    estimate = float(
        np.average(
            margins,
            weights=weights,
        )
    )

    total_weight = float(weights.sum())

    effective_sample_size = float(
        total_weight**2
        / np.square(weights).sum()
    )

    ages = aggregate_frame[
        "poll_age_days"
    ].to_numpy(dtype=float)

    weighted_mean_age = float(
        np.average(
            ages,
            weights=weights,
        )
    )

    cycle = int(
        snapshot["cycle"].iloc[0]
    )

    snapshot_days = int(
        snapshot[
            "snapshot_days_before_election"
        ].iloc[0]
    )

    snapshot_date = str(
        snapshot["snapshot_date"].iloc[0]
    )

    return GenericBallotAggregationResult(
        cycle=cycle,
        snapshot_days_before_election=snapshot_days,
        snapshot_date=snapshot_date,
        estimate_margin_dem=estimate,
        input_question_rows=input_question_rows,
        retained_question_rows=retained_question_rows,
        aggregated_rows=len(aggregate_frame),
        unique_polls=int(
            aggregate_frame["poll_id"].nunique()
        ),
        unique_pollsters=int(
            aggregate_frame["pollster"].nunique()
        ),
        total_weight=total_weight,
        effective_sample_size=effective_sample_size,
        weighted_mean_poll_age_days=weighted_mean_age,
        oldest_poll_age_days=float(ages.max()),
        newest_poll_age_days=float(ages.min()),
        specification=asdict(spec),
    )


def select_snapshot(
    snapshots: pd.DataFrame,
    *,
    cycle: int,
    snapshot_days_before_election: int,
) -> pd.DataFrame:
    """Select exactly one historical snapshot from the snapshot warehouse."""

    selected = snapshots[
        pd.to_numeric(
            snapshots["cycle"],
            errors="coerce",
        ).eq(cycle)
        & pd.to_numeric(
            snapshots[
                "snapshot_days_before_election"
            ],
            errors="coerce",
        ).eq(snapshot_days_before_election)
    ].copy()

    if selected.empty:
        raise ValueError(
            "No snapshot found for "
            f"cycle={cycle}, "
            "snapshot_days_before_election="
            f"{snapshot_days_before_election}."
        )

    return selected
