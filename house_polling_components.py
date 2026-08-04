from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from pollster_registry import apply_pollster_registry


AT_LARGE_STATES = {
    "AK",
    "DE",
    "ND",
    "SD",
    "VT",
    "WY",
}

POLLSTER_GRADE_WEIGHTS = {
    "A+": 1.15,
    "A": 1.10,
    "A-": 1.05,
    "B+": 1.00,
    "B": 0.95,
    "B-": 0.90,
    "C+": 0.85,
    "C": 0.80,
    "C-": 0.75,
    "D": 0.65,
    "Unknown": 0.85,
    "": 0.85,
}

SAMPLE_TYPE_WEIGHTS = {
    "LV": 1.00,
    "RV": 0.85,
    "A": 0.70,
    "Other": 0.75,
    "": 0.75,
}

POLL_INPUT_COLUMNS = [
    "race",
    "state",
    "district",
    "district_id",
    "pollster",
    "pollster_grade",
    "manual_house_effect_adjustment_dem",
    "sponsor",
    "poll_sponsor_type",
    "partisan_sponsor_party",
    "is_internal_poll",
    "start_date",
    "end_date",
    "sample_size",
    "sample_type",
    "dem_candidate",
    "gop_candidate",
    "ind_candidate",
    "other_candidate",
    "dem_pct",
    "gop_pct",
    "ind_pct",
    "other_pct",
    "undecided_pct",
    "notes",
]

CLEAN_POLL_COLUMNS = [
    "race",
    "state",
    "district",
    "district_id",
    "pollster",
    "pollster_grade",
    "pollster_raw",
    "pollster_normalized_key",
    "canonical_pollster",
    "pollster_match_method",
    "pollster_house_effect_dem",
    "manual_house_effect_override_dem",
    "effective_house_effect_dem",
    "house_effect_source",
    "house_effect_confidence",
    "house_effect_notes",
    "house_effect_dem",
    "start_date",
    "end_date",
    "sample_size",
    "sample_type",
    "dem_candidate",
    "gop_candidate",
    "dem_pct",
    "gop_pct",
    "raw_margin_dem",
    "polling_margin_dem",
    "poll_age_days",
    "recency_weight",
    "sample_size_weight",
    "pollster_grade_weight",
    "sample_type_weight",
    "poll_weight",
    "notes",
]

AVERAGE_COLUMNS = [
    "district_id",
    "state",
    "district",
    "polling_margin_dem",
    "poll_count",
    "latest_poll_end_date",
    "avg_poll_age_days",
    "total_poll_weight",
    "polling_notes",
    "effective_poll_count",
    "largest_pollster_weight_share",
    "only_partisan_or_internal_polls",
]


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""

    return str(value).strip()


def normalize_state(value: object) -> str:
    return clean_text(value).upper()


def normalize_district_value(
    state: object,
    district: object,
) -> str:
    state_text = normalize_state(state)
    district_text = clean_text(
        district
    ).upper()

    if district_text in {
        "",
        "NAN",
        "NONE",
    }:
        return ""

    if (
        state_text in AT_LARGE_STATES
        and district_text
        in {
            "1",
            "01",
            "AL",
            "AT-LARGE",
            "AT LARGE",
            "AT_LARGE",
        }
    ):
        return "AL"

    if district_text.isdigit():
        return str(
            int(district_text)
        )

    return district_text


def normalize_district_id(
    state: object,
    district: object,
) -> str:
    state_text = normalize_state(
        state
    )
    district_text = (
        normalize_district_value(
            state,
            district,
        )
    )

    if (
        state_text == ""
        or district_text == ""
    ):
        return ""

    return (
        f"{state_text}-"
        f"{district_text}"
    )


def normalize_existing_district_id(
    raw_district_id: object,
    state: object = "",
    district: object = "",
) -> str:
    raw = clean_text(
        raw_district_id
    ).upper()

    if raw and "-" in raw:
        state_part, district_part = (
            raw.split("-", 1)
        )

        return normalize_district_id(
            state_part,
            district_part,
        )

    if raw and state:
        return normalize_district_id(
            state,
            raw,
        )

    return normalize_district_id(
        state,
        district,
    )


def safe_numeric(
    series: pd.Series,
    default: float = np.nan,
) -> pd.Series:
    return pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(default)


def recency_weight(
    end_date: pd.Timestamp,
    as_of: date,
) -> float:
    if pd.isna(end_date):
        return 0.50

    age = max(
        0,
        (
            as_of
            - end_date.date()
        ).days,
    )

    return float(
        np.exp(
            -age / 75.0
        )
    )


def sample_size_weight(
    sample_size: object,
) -> float:
    if (
        pd.isna(sample_size)
        or float(sample_size) <= 0
    ):
        return 0.60

    return float(
        np.sqrt(
            float(sample_size)
            / 600.0
        )
    )


def grade_weight(
    grade: object,
) -> float:
    grade_text = clean_text(
        grade
    )

    return POLLSTER_GRADE_WEIGHTS.get(
        grade_text,
        POLLSTER_GRADE_WEIGHTS[
            "Unknown"
        ],
    )


def sample_type_weight(
    sample_type: object,
) -> float:
    sample_text = clean_text(
        sample_type
    ).upper()

    if (
        sample_text
        in SAMPLE_TYPE_WEIGHTS
    ):
        return SAMPLE_TYPE_WEIGHTS[
            sample_text
        ]

    return SAMPLE_TYPE_WEIGHTS[
        "Other"
    ]


def normalize_sponsor_type(
    row: pd.Series,
) -> str:
    raw_type = clean_text(
        row.get(
            "poll_sponsor_type",
            "",
        )
    ).lower()

    is_internal = clean_text(
        row.get(
            "is_internal_poll",
            "",
        )
    ).lower()

    if is_internal in {
        "true",
        "1",
        "yes",
        "y",
    }:
        return "internal"

    if raw_type in {
        "internal",
        "campaign internal",
        "campaign",
    }:
        return "internal"

    if raw_type in {
        "partisan",
        "party",
        "aligned",
        "sponsored",
    }:
        return "partisan"

    if raw_type in {
        "neutral",
        "nonpartisan",
        "independent",
        "public",
    }:
        return "neutral"

    return "unknown"


def sponsor_weight(
    sponsor_type: object,
) -> float:
    sponsor_text = clean_text(
        sponsor_type
    ).lower()

    if sponsor_text == "neutral":
        return 1.00

    if sponsor_text == "partisan":
        return 0.80

    if sponsor_text == "internal":
        return 0.65

    return 0.85


def partisan_sponsor_adjustment(
    row: pd.Series,
) -> float:
    sponsor_type = clean_text(
        row.get(
            "sponsor_classification",
            "",
        )
    ).lower()

    sponsor_party = clean_text(
        row.get(
            "partisan_sponsor_party",
            "",
        )
    ).upper()

    if sponsor_type == "internal":
        magnitude = 1.5
    elif sponsor_type == "partisan":
        magnitude = 1.0
    else:
        return 0.0

    if sponsor_party in {
        "D",
        "DEM",
        "DEMOCRAT",
        "DEMOCRATIC",
    }:
        return -magnitude

    if sponsor_party in {
        "R",
        "REP",
        "REPUBLICAN",
        "GOP",
    }:
        return magnitude

    return 0.0


def kish_effective_count(
    weights: pd.Series,
) -> float:
    numeric = pd.to_numeric(
        weights,
        errors="coerce",
    ).fillna(0.0)

    total = numeric.sum()
    squared_total = (
        numeric ** 2
    ).sum()

    if (
        total <= 0
        or squared_total <= 0
    ):
        return 0.0

    return float(
        (total ** 2)
        / squared_total
    )


def largest_pollster_share(
    group: pd.DataFrame,
) -> float:
    total = group[
        "poll_weight"
    ].sum()

    if total <= 0:
        return 0.0

    pollster_norm = (
        group["pollster"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace(
            "",
            "Unknown",
        )
    )

    by_pollster = (
        group.assign(
            pollster_norm=pollster_norm
        )
        .groupby(
            "pollster_norm"
        )["poll_weight"]
        .sum()
    )

    return float(
        by_pollster.max()
        / total
    )


def ensure_poll_columns(
    polls: pd.DataFrame,
) -> pd.DataFrame:
    out = polls.copy()

    for column in POLL_INPUT_COLUMNS:
        if column not in out.columns:
            out[column] = ""

    return out[
        POLL_INPUT_COLUMNS
    ].copy()


def prepare_house_poll_questions(
    polls: pd.DataFrame,
    races: pd.DataFrame,
    *,
    as_of: date,
    registry_path: Path,
) -> tuple[
    pd.DataFrame,
    list[str],
    int,
]:
    """
    Normalize, validate, adjust, and weight House poll rows.

    Returns:
        prepared polls,
        unmatched district IDs,
        dropped-row count.
    """
    out = ensure_poll_columns(
        polls
    )

    # Historical snapshots can legitimately contain no eligible
    # polls. Return the empty production-schema frame before calling
    # the pollster registry, whose empty-input result does not include
    # the registry-derived columns expected below.
    if out.empty:
        for column, default in [
            ("pollster_raw", ""),
            ("pollster_normalized_key", ""),
            ("canonical_pollster", ""),
            ("pollster_match_method", ""),
            ("pollster_house_effect_dem", 0.0),
            ("house_effect_confidence", ""),
            ("house_effect_notes", ""),
            ("manual_house_effect_override_dem", np.nan),
            ("effective_house_effect_dem", 0.0),
            ("house_effect_source", ""),
            ("house_effect_dem", 0.0),
            ("raw_margin_dem", np.nan),
            ("sponsor_classification", ""),
            ("sponsor_weight", np.nan),
            ("partisan_sponsor_adjustment_dem", 0.0),
            ("polling_margin_dem", np.nan),
            ("poll_age_days", np.nan),
            ("recency_weight", np.nan),
            ("sample_size_weight", np.nan),
            ("pollster_grade_weight", np.nan),
            ("sample_type_weight", np.nan),
            ("poll_weight", np.nan),
        ]:
            out[column] = default

        return (
            out,
            [],
            0,
        )

    out = apply_pollster_registry(
        out,
        registry_path=registry_path,
    )

    poll_level_override = (
        pd.to_numeric(
            out[
                "manual_house_effect_adjustment_dem"
            ],
            errors="coerce",
        )
    )

    registry_house_effect = (
        pd.to_numeric(
            out[
                "pollster_house_effect_dem"
            ],
            errors="coerce",
        ).fillna(0.0)
    )

    out[
        "manual_house_effect_override_dem"
    ] = poll_level_override

    out[
        "effective_house_effect_dem"
    ] = poll_level_override.where(
        poll_level_override.notna(),
        registry_house_effect,
    )

    out[
        "house_effect_source"
    ] = "pollster_registry"

    out.loc[
        poll_level_override.notna(),
        "house_effect_source",
    ] = "poll_level_override"

    out["house_effect_dem"] = (
        pd.to_numeric(
            out[
                "effective_house_effect_dem"
            ],
            errors="coerce",
        ).fillna(0.0)
    )

    out["state"] = out[
        "state"
    ].apply(normalize_state)

    out["district"] = out.apply(
        lambda row: (
            normalize_district_value(
                row.get(
                    "state",
                    "",
                ),
                row.get(
                    "district",
                    "",
                ),
            )
        ),
        axis=1,
    )

    out["district_id"] = out.apply(
        lambda row: (
            normalize_existing_district_id(
                row.get(
                    "district_id",
                    "",
                ),
                row.get(
                    "state",
                    "",
                ),
                row.get(
                    "district",
                    "",
                ),
            )
        ),
        axis=1,
    )

    for column in [
        "dem_pct",
        "gop_pct",
    ]:
        out[column] = safe_numeric(
            out[column]
        )

    for column in [
        "ind_pct",
        "other_pct",
        "undecided_pct",
    ]:
        out[column] = safe_numeric(
            out[column],
            default=0.0,
        )

    out["sample_size"] = safe_numeric(
        out["sample_size"],
        default=np.nan,
    )

    out["start_date"] = (
        pd.to_datetime(
            out["start_date"],
            errors="coerce",
        )
    )

    out["end_date"] = (
        pd.to_datetime(
            out["end_date"],
            errors="coerce",
        )
    )

    valid_districts = set(
        races[
            "district_id"
        ]
        .dropna()
        .astype(str)
    )

    unmatched = sorted(
        set(
            out[
                "district_id"
            ]
            .dropna()
            .astype(str)
        )
        - valid_districts
        - {""}
    )

    out = out.loc[
        out[
            "district_id"
        ].isin(valid_districts)
    ].copy()

    usable = (
        out["dem_pct"].notna()
        & out["gop_pct"].notna()
        & out["district_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    dropped = int(
        len(out)
        - usable.sum()
    )

    out = out.loc[
        usable
    ].copy()

    if out.empty:
        return (
            out,
            unmatched,
            dropped,
        )

    out["raw_margin_dem"] = (
        out["dem_pct"]
        - out["gop_pct"]
    )

    out[
        "sponsor_classification"
    ] = out.apply(
        normalize_sponsor_type,
        axis=1,
    )

    out["sponsor_weight"] = out[
        "sponsor_classification"
    ].apply(sponsor_weight)

    out[
        "partisan_sponsor_adjustment_dem"
    ] = out.apply(
        partisan_sponsor_adjustment,
        axis=1,
    )

    out["polling_margin_dem"] = (
        out["raw_margin_dem"]
        - out["house_effect_dem"]
        + out[
            "partisan_sponsor_adjustment_dem"
        ]
    )

    out["poll_age_days"] = out[
        "end_date"
    ].apply(
        lambda value: (
            max(
                0,
                (
                    as_of
                    - value.date()
                ).days,
            )
            if pd.notna(value)
            else np.nan
        )
    )

    out["recency_weight"] = out[
        "end_date"
    ].apply(
        lambda value: recency_weight(
            value,
            as_of,
        )
    )

    out[
        "sample_size_weight"
    ] = out[
        "sample_size"
    ].apply(sample_size_weight)

    out[
        "pollster_grade_weight"
    ] = out[
        "pollster_grade"
    ].apply(grade_weight)

    out[
        "sample_type_weight"
    ] = out[
        "sample_type"
    ].apply(sample_type_weight)

    out["poll_weight"] = (
        out["recency_weight"]
        * out[
            "sample_size_weight"
        ]
        * out[
            "pollster_grade_weight"
        ]
        * out[
            "sample_type_weight"
        ]
        * out["sponsor_weight"]
    )

    out["poll_weight"] = out[
        "poll_weight"
    ].clip(lower=0.05)

    return (
        out,
        unmatched,
        dropped,
    )


def aggregate_house_poll_questions(
    prepared_polls: pd.DataFrame,
    *,
    notes_prefix: str = (
        "Manual House polling average"
    ),
) -> pd.DataFrame:
    """
    Aggregate prepared poll rows into district-level polling fields.
    """
    if prepared_polls.empty:
        return pd.DataFrame(
            columns=AVERAGE_COLUMNS
        )

    rows: list[dict[str, object]] = []

    for district_id, group in (
        prepared_polls.groupby(
            "district_id"
        )
    ):
        total_weight = group[
            "poll_weight"
        ].sum()

        if total_weight <= 0:
            polling_margin = group[
                "polling_margin_dem"
            ].mean()
        else:
            polling_margin = float(
                (
                    group[
                        "polling_margin_dem"
                    ]
                    * group[
                        "poll_weight"
                    ]
                ).sum()
                / total_weight
            )

        latest_end = group[
            "end_date"
        ].max()

        avg_age = group[
            "poll_age_days"
        ].mean()

        pollsters = ", ".join(
            group[
                "pollster"
            ]
            .fillna("")
            .astype(str)
            .replace(
                "",
                "Unknown",
            )
            .tolist()
        )

        only_partisan_or_internal = bool(
            group[
                "sponsor_classification"
            ]
            .fillna("unknown")
            .astype(str)
            .str.lower()
            .isin(
                [
                    "partisan",
                    "internal",
                ]
            )
            .all()
        )

        rows.append(
            {
                "district_id": district_id,
                "state": group[
                    "state"
                ].iloc[0],
                "district": group[
                    "district"
                ].iloc[0],
                "polling_margin_dem":
                    polling_margin,
                "poll_count": len(group),
                "latest_poll_end_date": (
                    latest_end
                    .date()
                    .isoformat()
                    if pd.notna(
                        latest_end
                    )
                    else ""
                ),
                "avg_poll_age_days":
                    avg_age,
                "total_poll_weight":
                    total_weight,
                "polling_notes": (
                    f"{notes_prefix} from "
                    f"{len(group)} poll(s): "
                    f"{pollsters}"
                ),
                "effective_poll_count":
                    kish_effective_count(
                        group[
                            "poll_weight"
                        ]
                    ),
                "largest_pollster_weight_share":
                    largest_pollster_share(
                        group
                    ),
                "only_partisan_or_internal_polls":
                    only_partisan_or_internal,
            }
        )

    return pd.DataFrame(
        rows,
        columns=AVERAGE_COLUMNS,
    )


def clean_poll_output(
    prepared_polls: pd.DataFrame,
) -> pd.DataFrame:
    if prepared_polls.empty:
        return pd.DataFrame(
            columns=CLEAN_POLL_COLUMNS
        )

    missing = [
        column
        for column in CLEAN_POLL_COLUMNS
        if column
        not in prepared_polls.columns
    ]

    if missing:
        raise ValueError(
            "Prepared poll data are missing "
            "clean-output columns: "
            + ", ".join(missing)
        )

    return prepared_polls[
        CLEAN_POLL_COLUMNS
    ].copy()
