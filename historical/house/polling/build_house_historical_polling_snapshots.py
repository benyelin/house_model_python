from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )

from house_polling_components import (
    aggregate_house_poll_questions,
    normalize_district_id,
    prepare_house_poll_questions,
)


RAW_POLL_PATH = (
    REPO_ROOT
    / "historical"
    / "house"
    / "polling"
    / "raw"
    / "fivethirtyeight_archive"
    / "house_polls_historical.csv"
)

HISTORICAL_RACE_PATH = (
    REPO_ROOT
    / "historical"
    / "house"
    / "warehouse"
    / "house_historical_backtest_inputs_2016_2022.csv"
)

POLLSTER_REGISTRY_PATH = (
    REPO_ROOT
    / "inputs"
    / "pollster_registry.csv"
)

OUTPUT_DIR = (
    REPO_ROOT
    / "historical"
    / "house"
    / "polling"
    / "processed"
)

VALIDATION_DIR = (
    REPO_ROOT
    / "historical"
    / "house"
    / "polling"
    / "validation"
)

QUESTION_OUTPUT_PATH = (
    OUTPUT_DIR
    / "house_historical_poll_questions.csv"
)

SNAPSHOT_OUTPUT_PATH = (
    OUTPUT_DIR
    / "house_historical_polling_snapshots.csv"
)

SUMMARY_OUTPUT_PATH = (
    OUTPUT_DIR
    / "house_historical_polling_snapshot_summary.csv"
)

VALIDATION_OUTPUT_PATH = (
    VALIDATION_DIR
    / "house_historical_polling_snapshot_validation.txt"
)

SUPPORTED_CYCLES = (
    2018,
    2020,
    2022,
)

SNAPSHOT_DAYS_OUT = (
    120,
    90,
    60,
    30,
    14,
    7,
    1,
)

CANONICAL_ELECTION_DATES = {
    2018: pd.Timestamp("2018-11-06"),
    2020: pd.Timestamp("2020-11-03"),
    2022: pd.Timestamp("2022-11-08"),
}

STATE_ABBREVIATIONS = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}

PARTY_MAP = {
    "DEM": "D",
    "DEMOCRAT": "D",
    "DEMOCRATIC": "D",
    "D": "D",
    "REP": "R",
    "REPUBLICAN": "R",
    "GOP": "R",
    "R": "R",
}

POPULATION_PRIORITY = {
    "lv": 1,
    "rv": 2,
    "v": 3,
    "a": 4,
}

# Montana elected one statewide House member through 2020.
# Following reapportionment, MT-1 and MT-2 first apply in 2022.
CYCLE_DISTRICT_ID_OVERRIDES = {
    (2018, "MT", 1): "MT-AL",
    (2020, "MT", 1): "MT-AL",
}

OUTPUT_POLLING_COLUMNS = [
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


def parse_bool(
    value: object,
) -> bool:
    if value is None or pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    return (
        str(value)
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
            "y",
        }
    )


def numeric_grade_to_letter(
    value: object,
) -> str:
    """
    Translate FiveThirtyEight's approximately 0–3 numeric
    grade scale into the production House letter-grade buckets.

    This mapping can later be tested against a no-grade control.
    """
    grade = pd.to_numeric(
        value,
        errors="coerce",
    )

    if pd.isna(grade):
        return "Unknown"

    grade = float(grade)

    if grade >= 2.85:
        return "A+"
    if grade >= 2.55:
        return "A"
    if grade >= 2.25:
        return "A-"
    if grade >= 1.95:
        return "B+"
    if grade >= 1.65:
        return "B"
    if grade >= 1.35:
        return "B-"
    if grade >= 1.05:
        return "C+"
    if grade >= 0.75:
        return "C"
    if grade >= 0.45:
        return "C-"

    return "D"


def normalize_population(
    value: object,
) -> str:
    text = (
        ""
        if value is None or pd.isna(value)
        else str(value).strip().lower()
    )

    if text == "lv":
        return "LV"
    if text == "rv":
        return "RV"
    if text == "a":
        return "A"

    return "Other"


def normalize_party(
    value: object,
) -> str:
    text = (
        ""
        if value is None or pd.isna(value)
        else str(value).strip().upper()
    )

    return PARTY_MAP.get(
        text,
        "",
    )


def parse_archive_dates(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    out = frame.copy()

    for column in [
        "start_date",
        "end_date",
        "election_date",
    ]:
        out[column] = pd.to_datetime(
            out[column],
            format="%m/%d/%y",
            errors="coerce",
        )

    out["created_at"] = pd.to_datetime(
        out["created_at"],
        errors="coerce",
    )

    return out


def validate_zero_historical_house_effects() -> None:
    registry = pd.read_csv(
        POLLSTER_REGISTRY_PATH,
        low_memory=False,
    )

    if (
        "pollster_house_effect_dem"
        not in registry.columns
    ):
        return

    effects = pd.to_numeric(
        registry[
            "pollster_house_effect_dem"
        ],
        errors="coerce",
    ).fillna(0.0)

    if effects.ne(0.0).any():
        raise RuntimeError(
            "The production pollster registry now contains "
            "nonzero House effects. Historical replay must not "
            "silently apply present-cycle house-effect judgments. "
            "Add an explicit historical house-effect policy before "
            "rebuilding snapshots."
        )


def build_question_level_archive(
    raw: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    archive = parse_archive_dates(
        raw
    )

    archive["cycle"] = pd.to_numeric(
        archive["cycle"],
        errors="coerce",
    )

    archive["pct"] = pd.to_numeric(
        archive["pct"],
        errors="coerce",
    )

    archive["sample_size"] = (
        pd.to_numeric(
            archive["sample_size"],
            errors="coerce",
        )
    )

    archive["major_party"] = (
        archive["party"].apply(
            normalize_party
        )
    )

    archive["state_abbreviation"] = (
        archive["state"].map(
            STATE_ABBREVIATIONS
        )
    )

    archive["seat_number_numeric"] = (
        pd.to_numeric(
            archive["seat_number"],
            errors="coerce",
        )
    )

    def archive_district_id(
        row: pd.Series,
    ) -> str:
        state = row[
            "state_abbreviation"
        ]

        seat_number = row[
            "seat_number_numeric"
        ]

        cycle = pd.to_numeric(
            row["cycle"],
            errors="coerce",
        )

        if (
            pd.isna(state)
            or pd.isna(seat_number)
            or pd.isna(cycle)
            or not float(
                seat_number
            ).is_integer()
        ):
            return ""

        seat_number_int = int(
            seat_number
        )

        cycle_int = int(
            cycle
        )

        override = (
            CYCLE_DISTRICT_ID_OVERRIDES.get(
                (
                    cycle_int,
                    str(state),
                    seat_number_int,
                )
            )
        )

        if override is not None:
            return override

        return normalize_district_id(
            state,
            seat_number_int,
        )

    archive["district_id"] = (
        archive.apply(
            archive_district_id,
            axis=1,
        )
    )

    regular = archive.loc[
        archive["cycle"].isin(
            SUPPORTED_CYCLES
        )
        & archive["office_type"]
        .fillna("")
        .astype(str)
        .str.contains(
            "House",
            case=False,
            regex=False,
        )
        & archive["stage"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("general")
        & ~archive[
            "hypothetical"
        ].apply(parse_bool)
        & ~archive[
            "ranked_choice_reallocated"
        ].apply(parse_bool)
        & archive[
            "district_id"
        ].ne("")
    ].copy()

    question_keys = [
        "cycle",
        "race_id",
        "poll_id",
        "question_id",
    ]

    metadata = (
        regular.groupby(
            question_keys,
            dropna=False,
        )
        .agg(
            district_id=(
                "district_id",
                "first",
            ),
            state=(
                "state_abbreviation",
                "first",
            ),
            district=(
                "seat_number_numeric",
                "first",
            ),
            pollster=(
                "pollster",
                "first",
            ),
            numeric_grade=(
                "numeric_grade",
                "first",
            ),
            sponsors=(
                "sponsors",
                "first",
            ),
            sponsor_candidate_party=(
                "sponsor_candidate_party",
                "first",
            ),
            internal=(
                "internal",
                "first",
            ),
            partisan=(
                "partisan",
                "first",
            ),
            start_date=(
                "start_date",
                "first",
            ),
            end_date=(
                "end_date",
                "first",
            ),
            created_at=(
                "created_at",
                "first",
            ),
            election_date=(
                "election_date",
                "first",
            ),
            sample_size=(
                "sample_size",
                "first",
            ),
            population=(
                "population",
                "first",
            ),
            notes=(
                "notes",
                "first",
            ),
            answer_rows=(
                "answer",
                "size",
            ),
            dem_answer_rows=(
                "major_party",
                lambda values: int(
                    values.eq("D").sum()
                ),
            ),
            gop_answer_rows=(
                "major_party",
                lambda values: int(
                    values.eq("R").sum()
                ),
            ),
        )
        .reset_index()
    )

    usable_keys = metadata.loc[
        metadata[
            "dem_answer_rows"
        ].eq(1)
        & metadata[
            "gop_answer_rows"
        ].eq(1),
        question_keys,
    ]

    usable_answers = regular.merge(
        usable_keys,
        on=question_keys,
        how="inner",
        validate="many_to_one",
    )

    major_answers = usable_answers.loc[
        usable_answers[
            "major_party"
        ].isin(
            [
                "D",
                "R",
            ]
        )
    ].copy()

    pivot_pct = (
        major_answers.pivot_table(
            index=question_keys,
            columns="major_party",
            values="pct",
            aggfunc="first",
        )
        .rename(
            columns={
                "D": "dem_pct",
                "R": "gop_pct",
            }
        )
        .reset_index()
    )

    pivot_candidate = (
        major_answers.pivot_table(
            index=question_keys,
            columns="major_party",
            values="candidate_name",
            aggfunc="first",
        )
        .rename(
            columns={
                "D": "dem_candidate",
                "R": "gop_candidate",
            }
        )
        .reset_index()
    )

    questions = (
        metadata.merge(
            pivot_pct,
            on=question_keys,
            how="inner",
            validate="one_to_one",
        )
        .merge(
            pivot_candidate,
            on=question_keys,
            how="inner",
            validate="one_to_one",
        )
    )

    questions["population_key"] = (
        questions["population"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    questions[
        "population_priority"
    ] = questions[
        "population_key"
    ].map(
        POPULATION_PRIORITY
    ).fillna(5)

    # One poll should not receive extra influence solely because it
    # reported several population variants or alternate questionnaires.
    # Prefer LV, then RV, then V, then adults, with question_id as the
    # deterministic tie-breaker.
    selected = (
        questions.sort_values(
            [
                "cycle",
                "district_id",
                "poll_id",
                "population_priority",
                "question_id",
            ],
            kind="mergesort",
        )
        .drop_duplicates(
            [
                "cycle",
                "district_id",
                "poll_id",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    return questions, selected


def archive_questions_to_production_schema(
    selected: pd.DataFrame,
) -> pd.DataFrame:
    out = pd.DataFrame(
        index=selected.index
    )

    out["race"] = selected[
        "district_id"
    ]

    out["state"] = selected[
        "state"
    ]

    out["district"] = selected[
        "district"
    ]

    out["district_id"] = selected[
        "district_id"
    ]

    out["pollster"] = selected[
        "pollster"
    ].fillna("")

    out["pollster_grade"] = (
        selected["numeric_grade"].apply(
            numeric_grade_to_letter
        )
    )

    out[
        "manual_house_effect_adjustment_dem"
    ] = np.nan

    out["sponsor"] = selected[
        "sponsors"
    ].fillna("")

    internal = selected[
        "internal"
    ].apply(parse_bool)

    partisan_raw = (
        selected["partisan"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    partisan_flag = (
        partisan_raw
        .str.lower()
        .isin(
            {
                "true",
                "1",
                "yes",
                "y",
            }
        )
        | partisan_raw.ne("")
    )

    out["poll_sponsor_type"] = np.where(
        internal,
        "internal",
        np.where(
            partisan_flag,
            "partisan",
            "neutral",
        ),
    )

    sponsor_party = selected[
        "sponsor_candidate_party"
    ].apply(normalize_party)

    # In some FiveThirtyEight versions, the partisan field itself
    # contains the sponsoring party rather than a Boolean.
    partisan_party = selected[
        "partisan"
    ].apply(normalize_party)

    out[
        "partisan_sponsor_party"
    ] = sponsor_party.where(
        sponsor_party.ne(""),
        partisan_party,
    )

    out["is_internal_poll"] = (
        internal
    )

    out["start_date"] = selected[
        "start_date"
    ]

    out["end_date"] = selected[
        "end_date"
    ]

    out["sample_size"] = selected[
        "sample_size"
    ]

    out["sample_type"] = selected[
        "population"
    ].apply(normalize_population)

    out["dem_candidate"] = selected[
        "dem_candidate"
    ].fillna("")

    out["gop_candidate"] = selected[
        "gop_candidate"
    ].fillna("")

    out["ind_candidate"] = ""
    out["other_candidate"] = ""

    out["dem_pct"] = selected[
        "dem_pct"
    ]

    out["gop_pct"] = selected[
        "gop_pct"
    ]

    out["ind_pct"] = 0.0
    out["other_pct"] = 0.0
    out["undecided_pct"] = 0.0

    out["notes"] = selected[
        "notes"
    ].fillna("")

    return out


def initialize_snapshot_rows(
    races: pd.DataFrame,
    *,
    cycle: int,
    snapshot_date: pd.Timestamp,
    days_out: int,
) -> pd.DataFrame:
    out = races[
        [
            "forecast_cycle",
            "race_id",
            "state",
            "district",
        ]
    ].copy()

    out = out.rename(
        columns={
            "race_id": "district_id",
        }
    )

    out["snapshot_date"] = (
        snapshot_date.date().isoformat()
    )

    out["days_out"] = days_out

    out["polling_margin_dem"] = np.nan
    out["poll_count"] = 0
    out["polling_active"] = False
    out["latest_poll_end_date"] = ""
    out["avg_poll_age_days"] = np.nan
    out["total_poll_weight"] = 0.0
    out["effective_poll_count"] = 0.0
    out[
        "largest_pollster_weight_share"
    ] = 0.0
    out[
        "only_partisan_or_internal_polls"
    ] = False
    out["polling_notes"] = ""

    return out


def build_snapshots(
    selected: pd.DataFrame,
    historical_races: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    snapshot_frames = []
    summary_rows = []

    for cycle in SUPPORTED_CYCLES:
        cycle_races = (
            historical_races.loc[
                historical_races[
                    "forecast_cycle"
                ].eq(cycle)
            ]
            .drop_duplicates(
                [
                    "forecast_cycle",
                    "race_id",
                ]
            )
            .copy()
        )

        if len(cycle_races) != 435:
            raise RuntimeError(
                f"Cycle {cycle} expected 435 canonical races; "
                f"found {len(cycle_races)}."
            )

        cycle_questions = selected.loc[
            selected["cycle"].eq(
                cycle
            )
        ].copy()

        # Use the known federal general-election date for each
        # supported cycle. Archive and warehouse fields can include
        # specials, runoffs, delayed contests, or malformed dates.
        if cycle not in CANONICAL_ELECTION_DATES:
            raise RuntimeError(
                f"No canonical election date configured for {cycle}."
            )

        election_date = CANONICAL_ELECTION_DATES[
            cycle
        ].normalize()

        cycle_questions[
            "election_date_normalized"
        ] = pd.to_datetime(
            cycle_questions[
                "election_date"
            ],
            errors="coerce",
        ).dt.normalize()

        off_calendar_questions = (
            cycle_questions[
                "election_date_normalized"
            ].notna()
            & cycle_questions[
                "election_date_normalized"
            ].ne(election_date)
        )

        excluded_off_calendar_count = int(
            off_calendar_questions.sum()
        )

        cycle_questions = cycle_questions.loc[
            ~off_calendar_questions
        ].copy()

        for days_out in (
            SNAPSHOT_DAYS_OUT
        ):
            snapshot_date = (
                election_date
                - pd.Timedelta(
                    days=days_out
                )
            )

            # The poll must have completed and been entered into the
            # archive by the snapshot date. Requiring both protects
            # against retrospective publication leakage.
            eligible = (
                cycle_questions.loc[
                    cycle_questions[
                        "end_date"
                    ].notna()
                    & cycle_questions[
                        "created_at"
                    ].notna()
                    & cycle_questions[
                        "end_date"
                    ].le(
                        snapshot_date
                    )
                    & cycle_questions[
                        "created_at"
                    ].dt.normalize().le(
                        snapshot_date
                    )
                    & cycle_questions[
                        "end_date"
                    ].lt(
                        election_date
                    )
                ]
                .copy()
            )

            production_polls = (
                archive_questions_to_production_schema(
                    eligible
                )
            )

            race_adapter = (
                cycle_races[
                    [
                        "race_id",
                        "state",
                        "district",
                    ]
                ]
                .rename(
                    columns={
                        "race_id":
                            "district_id",
                    }
                )
                .copy()
            )

            prepared, unmatched, dropped = (
                prepare_house_poll_questions(
                    production_polls,
                    race_adapter,
                    as_of=snapshot_date.date(),
                    registry_path=(
                        POLLSTER_REGISTRY_PATH
                    ),
                )
            )

            averages = (
                aggregate_house_poll_questions(
                    prepared,
                    notes_prefix=(
                        "Historical House polling average"
                    ),
                )
            )

            snapshot = (
                initialize_snapshot_rows(
                    cycle_races,
                    cycle=cycle,
                    snapshot_date=snapshot_date,
                    days_out=days_out,
                )
            )

            if not averages.empty:
                merge_columns = [
                    "district_id",
                    *[
                        column
                        for column
                        in OUTPUT_POLLING_COLUMNS
                        if column
                        != "polling_active"
                    ],
                ]

                snapshot = snapshot.merge(
                    averages[
                        merge_columns
                    ],
                    on="district_id",
                    how="left",
                    suffixes=(
                        "",
                        "_new",
                    ),
                    validate="one_to_one",
                )

                for column in [
                    column
                    for column
                    in OUTPUT_POLLING_COLUMNS
                    if column
                    != "polling_active"
                ]:
                    new_column = (
                        f"{column}_new"
                    )

                    if (
                        new_column
                        in snapshot.columns
                    ):
                        snapshot[column] = (
                            snapshot[
                                new_column
                            ].combine_first(
                                snapshot[
                                    column
                                ]
                            )
                        )

                        snapshot = (
                            snapshot.drop(
                                columns=[
                                    new_column
                                ]
                            )
                        )

            snapshot["poll_count"] = (
                pd.to_numeric(
                    snapshot[
                        "poll_count"
                    ],
                    errors="coerce",
                )
                .fillna(0)
                .astype(int)
            )

            snapshot[
                "polling_active"
            ] = snapshot[
                "poll_count"
            ].gt(0)

            for column in [
                "total_poll_weight",
                "effective_poll_count",
                "largest_pollster_weight_share",
            ]:
                snapshot[column] = (
                    pd.to_numeric(
                        snapshot[column],
                        errors="coerce",
                    ).fillna(0.0)
                )

            snapshot[
                "only_partisan_or_internal_polls"
            ] = snapshot[
                "only_partisan_or_internal_polls"
            ].fillna(False).astype(bool)

            snapshot_frames.append(
                snapshot
            )

            summary_rows.append(
                {
                    "forecast_cycle":
                        cycle,
                    "snapshot_date":
                        snapshot_date
                        .date()
                        .isoformat(),
                    "days_out":
                        days_out,
                    "eligible_poll_questions":
                        len(eligible),
                    "prepared_poll_questions":
                        len(prepared),
                    "districts_with_polling":
                        int(
                            snapshot[
                                "polling_active"
                            ].sum()
                        ),
                    "unmatched_poll_districts":
                        len(unmatched),
                    "dropped_poll_questions":
                        dropped,
                    "excluded_off_calendar_questions":
                        excluded_off_calendar_count,
                    "canonical_election_date":
                        election_date.date().isoformat(),
                    "mean_poll_count_polled":
                        (
                            float(
                                snapshot.loc[
                                    snapshot[
                                        "polling_active"
                                    ],
                                    "poll_count",
                                ].mean()
                            )
                            if snapshot[
                                "polling_active"
                            ].any()
                            else 0.0
                        ),
                    "mean_effective_poll_count_polled":
                        (
                            float(
                                snapshot.loc[
                                    snapshot[
                                        "polling_active"
                                    ],
                                    "effective_poll_count",
                                ].mean()
                            )
                            if snapshot[
                                "polling_active"
                            ].any()
                            else 0.0
                        ),
                }
            )

    return (
        pd.concat(
            snapshot_frames,
            ignore_index=True,
        ),
        pd.DataFrame(
            summary_rows
        ),
    )


def validate_outputs(
    selected: pd.DataFrame,
    snapshots: pd.DataFrame,
    summary: pd.DataFrame,
) -> list[str]:
    checks = []

    expected_snapshot_rows = (
        len(SUPPORTED_CYCLES)
        * len(SNAPSHOT_DAYS_OUT)
        * 435
    )

    if (
        len(snapshots)
        != expected_snapshot_rows
    ):
        raise RuntimeError(
            "Historical polling snapshot row count "
            f"should be {expected_snapshot_rows:,}; "
            f"found {len(snapshots):,}."
        )

    checks.append(
        "PASS: snapshot warehouse contains "
        f"{expected_snapshot_rows:,} rows"
    )

    duplicate_snapshot_keys = (
        snapshots.duplicated(
            [
                "forecast_cycle",
                "snapshot_date",
                "district_id",
            ]
        )
    )

    if duplicate_snapshot_keys.any():
        raise RuntimeError(
            "Duplicate cycle/snapshot/district "
            "rows found."
        )

    checks.append(
        "PASS: cycle/snapshot/district keys "
        "are unique"
    )

    counts = (
        snapshots.groupby(
            [
                "forecast_cycle",
                "snapshot_date",
            ]
        )
        .size()
    )

    if not counts.eq(435).all():
        raise RuntimeError(
            "At least one polling snapshot does "
            "not contain 435 House races."
        )

    checks.append(
        "PASS: every historical snapshot "
        "contains 435 races"
    )

    if snapshots[
        "poll_count"
    ].lt(0).any():
        raise RuntimeError(
            "Negative poll counts found."
        )

    checks.append(
        "PASS: all poll counts are nonnegative"
    )

    polling_active_expected = (
        snapshots["poll_count"].gt(0)
    )

    if not snapshots[
        "polling_active"
    ].eq(
        polling_active_expected
    ).all():
        raise RuntimeError(
            "polling_active does not equal "
            "poll_count > 0."
        )

    checks.append(
        "PASS: polling-active flags agree "
        "with poll counts"
    )

    selected_duplicate_keys = (
        selected.duplicated(
            [
                "cycle",
                "district_id",
                "poll_id",
            ]
        )
    )

    if selected_duplicate_keys.any():
        raise RuntimeError(
            "Selected historical poll questions "
            "contain duplicate poll/race rows."
        )

    checks.append(
        "PASS: one selected question per "
        "cycle/race/poll"
    )

    if (
        summary[
            "unmatched_poll_districts"
        ].gt(0).any()
    ):
        bad = summary.loc[
            summary[
                "unmatched_poll_districts"
            ].gt(0)
        ]

        raise RuntimeError(
            "Historical snapshot builder found "
            "unmatched poll districts:\n"
            + bad.to_string(index=False)
        )

    checks.append(
        "PASS: every prepared poll maps to "
        "a canonical historical district"
    )

    expected_summary_rows = (
        len(SUPPORTED_CYCLES)
        * len(SNAPSHOT_DAYS_OUT)
    )

    if (
        len(summary)
        != expected_summary_rows
    ):
        raise RuntimeError(
            "Historical polling summary row "
            f"count should be {expected_summary_rows}; "
            f"found {len(summary)}."
        )

    checks.append(
        "PASS: snapshot summary is complete"
    )

    final_snapshot_days_out = min(
        SNAPSHOT_DAYS_OUT
    )

    final_coverage = summary.loc[
        summary["days_out"].eq(
            final_snapshot_days_out
        )
    ].copy()

    missing_cycle_coverage = (
        final_coverage.loc[
            final_coverage[
                "districts_with_polling"
            ].le(0)
        ]
    )

    if not missing_cycle_coverage.empty:
        raise RuntimeError(
            "At least one supported cycle has zero "
            "polling coverage at the final snapshot:\n"
            + missing_cycle_coverage[
                [
                    "forecast_cycle",
                    "snapshot_date",
                    "eligible_poll_questions",
                    "excluded_off_calendar_questions",
                ]
            ].to_string(index=False)
        )

    observed_cycles = set(
        pd.to_numeric(
            final_coverage[
                "forecast_cycle"
            ],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )

    expected_cycles = set(
        SUPPORTED_CYCLES
    )

    if observed_cycles != expected_cycles:
        raise RuntimeError(
            "Final-snapshot coverage is missing supported "
            f"cycles. Expected {sorted(expected_cycles)}; "
            f"found {sorted(observed_cycles)}."
        )

    checks.append(
        "PASS: every supported cycle has polling "
        "coverage at the final snapshot"
    )

    return checks


def main() -> None:
    validate_zero_historical_house_effects()

    raw = pd.read_csv(
        RAW_POLL_PATH,
        low_memory=False,
    )

    historical_races = pd.read_csv(
        HISTORICAL_RACE_PATH,
        low_memory=False,
    )

    all_usable_questions, selected = (
        build_question_level_archive(
            raw
        )
    )

    snapshots, summary = (
        build_snapshots(
            selected,
            historical_races,
        )
    )

    checks = validate_outputs(
        selected,
        snapshots,
        summary,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected.to_csv(
        QUESTION_OUTPUT_PATH,
        index=False,
    )

    snapshots.to_csv(
        SNAPSHOT_OUTPUT_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )

    report_lines = [
        "HOUSE HISTORICAL POLLING SNAPSHOT VALIDATION",
        "=" * 92,
        "",
        f"Raw archive answer rows:        {len(raw):,}",
        (
            "Usable D-vs-R questions before "
            f"poll-level deduplication: {len(all_usable_questions):,}"
        ),
        (
            "Selected independent poll/race "
            f"questions:    {len(selected):,}"
        ),
        f"Snapshot warehouse rows:       {len(snapshots):,}",
        f"Snapshot summary rows:         {len(summary):,}",
        "",
        "SELECTED QUESTIONS BY CYCLE",
        "-" * 92,
        selected.groupby("cycle")
        .size()
        .to_string(),
        "",
        "SNAPSHOT COVERAGE",
        "-" * 92,
        summary.to_string(
            index=False
        ),
        "",
        "VALIDATION CHECKS",
        "-" * 92,
        *checks,
        "",
        "VALIDATION STATUS: PASSED",
    ]

    report = "\n".join(
        report_lines
    )

    VALIDATION_OUTPUT_PATH.write_text(
        report
    )

    print(report)
    print()
    print(
        f"Wrote: {QUESTION_OUTPUT_PATH}"
    )
    print(
        f"Wrote: {SNAPSHOT_OUTPUT_PATH}"
    )
    print(
        f"Wrote: {SUMMARY_OUTPUT_PATH}"
    )
    print(
        f"Wrote: {VALIDATION_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
