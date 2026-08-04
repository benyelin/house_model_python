from pathlib import Path
from datetime import date
import pandas as pd

from pollster_registry import apply_pollster_registry
from house_polling_components import (
    aggregate_house_poll_questions,
    clean_poll_output,
    prepare_house_poll_questions,
)
import numpy as np

INPUTS = Path("inputs")

HOUSE_RACE_INPUTS = INPUTS / "house_race_inputs.csv"
HOUSE_MANUAL_POLLS = INPUTS / "house_manual_polls.csv"
OUTPUT_CLEAN_POLLS = INPUTS / "house_manual_polls_clean.csv"
OUTPUT_AVERAGES = INPUTS / "house_polling_averages_generated.csv"

AT_LARGE_STATES = {"AK", "DE", "ND", "SD", "VT", "WY"}


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


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_state(x):
    return clean_text(x).upper()


def normalize_district_value(state, district):
    state = normalize_state(state)
    district = clean_text(district).upper()

    if district in ["", "NAN", "NONE"]:
        return ""

    if state in AT_LARGE_STATES and district in ["1", "01", "AL", "AT-LARGE", "AT LARGE", "AT_LARGE"]:
        return "AL"

    if district.isdigit():
        return str(int(district))

    return district


def normalize_district_id(state, district):
    state = normalize_state(state)
    district = normalize_district_value(state, district)

    if state == "" or district == "":
        return ""

    return f"{state}-{district}"


def normalize_existing_district_id(raw_district_id, state="", district=""):
    raw = clean_text(raw_district_id).upper()

    if raw and "-" in raw:
        state_part, district_part = raw.split("-", 1)
        return normalize_district_id(state_part, district_part)

    if raw and state:
        return normalize_district_id(state, raw)

    return normalize_district_id(state, district)


def safe_numeric(series, default=np.nan):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def recency_weight(end_date, as_of):
    """
    Smooth recency decay.

    Roughly:
      0 days old:    1.00
      30 days old:   ~0.67
      60 days old:   ~0.45
      90 days old:   ~0.30
    """
    if pd.isna(end_date):
        return 0.50

    age = max(0, (as_of - end_date.date()).days)

    return float(np.exp(-age / 75.0))


def sample_size_weight(n):
    if pd.isna(n) or n <= 0:
        return 0.60

    # sqrt scaling prevents very large polls from overwhelming everything.
    return float(np.sqrt(n / 600.0))


def grade_weight(grade):
    grade = clean_text(grade)

    return POLLSTER_GRADE_WEIGHTS.get(grade, POLLSTER_GRADE_WEIGHTS["Unknown"])


def sample_type_weight(sample_type):
    sample_type = clean_text(sample_type).upper()

    if sample_type in SAMPLE_TYPE_WEIGHTS:
        return SAMPLE_TYPE_WEIGHTS[sample_type]

    return SAMPLE_TYPE_WEIGHTS["Other"]


def normalize_sponsor_type(row):
    raw_type = clean_text(row.get("poll_sponsor_type", "")).lower()
    is_internal = clean_text(row.get("is_internal_poll", "")).lower()

    if is_internal in {"true", "1", "yes", "y"}:
        return "internal"
    if raw_type in {"internal", "campaign internal", "campaign"}:
        return "internal"
    if raw_type in {"partisan", "party", "aligned", "sponsored"}:
        return "partisan"
    if raw_type in {"neutral", "nonpartisan", "independent", "public"}:
        return "neutral"

    return "unknown"


def sponsor_weight(sponsor_type):
    sponsor_type = clean_text(sponsor_type).lower()

    if sponsor_type == "neutral":
        return 1.00
    if sponsor_type == "partisan":
        return 0.80
    if sponsor_type == "internal":
        return 0.65

    return 0.85


def partisan_sponsor_adjustment(row):
    sponsor_type = clean_text(row.get("sponsor_classification", "")).lower()
    sponsor_party = clean_text(row.get("partisan_sponsor_party", "")).upper()

    if sponsor_type == "internal":
        magnitude = 1.5
    elif sponsor_type == "partisan":
        magnitude = 1.0
    else:
        return 0.0

    # Adjustment is against the sponsor party.
    if sponsor_party in {"D", "DEM", "DEMOCRAT", "DEMOCRATIC"}:
        return -magnitude
    if sponsor_party in {"R", "REP", "REPUBLICAN", "GOP"}:
        return magnitude

    return 0.0


def kish_effective_count(weights):
    weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    total = weights.sum()
    squared_total = (weights ** 2).sum()

    if total <= 0 or squared_total <= 0:
        return 0.0

    return float((total ** 2) / squared_total)


def largest_pollster_share(group):
    total = group["poll_weight"].sum()

    if total <= 0:
        return 0.0

    pollster_norm = (
        group["pollster"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )

    by_pollster = (
        group.assign(pollster_norm=pollster_norm)
        .groupby("pollster_norm")["poll_weight"]
        .sum()
    )

    return float(by_pollster.max() / total)


def ensure_poll_columns(polls):
    cols = [
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

    for col in cols:
        if col not in polls.columns:
            polls[col] = ""

    return polls[cols].copy()


def empty_poll_outputs(races):
    races = races.copy()

    for col, default in [
        ("polling_margin_dem", np.nan),
        ("poll_count", 0),
        ("polling_active", False),
        ("latest_poll_end_date", ""),
        ("avg_poll_age_days", np.nan),
        ("total_poll_weight", 0.0),
        ("effective_poll_count", 0.0),
        ("largest_pollster_weight_share", 0.0),
        ("only_partisan_or_internal_polls", False),
        ("polling_notes", ""),
    ]:
        races[col] = default

    races.to_csv(HOUSE_RACE_INPUTS, index=False)

    empty_avg = pd.DataFrame(
        columns=[
            "district_id",
            "state",
            "district",
            "polling_margin_dem",
            "poll_count",
            "latest_poll_end_date",
            "avg_poll_age_days",
            "total_poll_weight",
            "polling_notes",
        ]
    )

    empty_avg.to_csv(OUTPUT_AVERAGES, index=False)

    return races, empty_avg


def main():
    as_of = date.today()

    if not HOUSE_RACE_INPUTS.exists():
        raise FileNotFoundError("inputs/house_race_inputs.csv not found. Run import/recalculate pipeline first.")

    races = pd.read_csv(HOUSE_RACE_INPUTS)

    if races.empty:
        raise ValueError("inputs/house_race_inputs.csv is empty.")

    if "district_id" not in races.columns:
        raise ValueError("house_race_inputs.csv must include district_id.")

    races["state"] = races["state"].fillna("").astype(str).str.strip().str.upper()
    races["district"] = races["district"].fillna("").astype(str).str.strip()
    races["district_id"] = races.apply(
        lambda row: normalize_district_id(row.get("state", ""), row.get("district", "")),
        axis=1,
    )

    if not HOUSE_MANUAL_POLLS.exists():
        print("No house_manual_polls.csv found. Creating empty polling fields.")
        empty_poll_outputs(races)
        print(f"Wrote {OUTPUT_AVERAGES}")
        return

    polls = pd.read_csv(HOUSE_MANUAL_POLLS)

    if polls.empty:
        print("house_manual_polls.csv is empty. Clearing polling fields.")
        empty_poll_outputs(races)
        print(f"Wrote {OUTPUT_AVERAGES}")
        return

    polls, unmatched, dropped = (
        prepare_house_poll_questions(
            polls,
            races,
            as_of=as_of,
            registry_path=(
                INPUTS
                / "pollster_registry.csv"
            ),
        )
    )

    if unmatched:
        print(
            "WARNING: Some polls do not match any "
            "district_id in house_race_inputs.csv:"
        )
        for district_id in unmatched[:30]:
            print(f"  - {district_id}")

    if dropped:
        print(
            "WARNING: Dropping "
            f"{dropped} polls with missing/invalid "
            "district_id or Dem/GOP percentages."
        )

    if polls.empty:
        print(
            "No usable polls after validation. "
            "Clearing polling fields."
        )
        empty_poll_outputs(races)
        return

    clean_polls = clean_poll_output(
        polls
    )

    averages = (
        aggregate_house_poll_questions(
            polls,
            notes_prefix=(
                "Manual House polling average"
            ),
        )
    )

    clean_polls.to_csv(
        OUTPUT_CLEAN_POLLS,
        index=False,
    )

    averages.to_csv(
        OUTPUT_AVERAGES,
        index=False,
    )

    # Clear existing poll fields first.
    for col, default in [
        ("polling_margin_dem", np.nan),
        ("poll_count", 0),
        ("polling_active", False),
        ("latest_poll_end_date", ""),
        ("avg_poll_age_days", np.nan),
        ("total_poll_weight", 0.0),
        ("effective_poll_count", 0.0),
        ("largest_pollster_weight_share", 0.0),
        ("only_partisan_or_internal_polls", False),
        ("polling_notes", ""),
    ]:
        races[col] = default

    races = races.merge(
        averages[
            [
                "district_id",
                "polling_margin_dem",
                "poll_count",
                "latest_poll_end_date",
                "avg_poll_age_days",
                "total_poll_weight",
                "polling_notes",
            ]
        ],
        on="district_id",
        how="left",
        suffixes=("", "_new"),
    )

    for col in [
        "polling_margin_dem",
        "poll_count",
        "latest_poll_end_date",
        "avg_poll_age_days",
        "total_poll_weight",
        "polling_notes",
    ]:
        new_col = f"{col}_new"
        if new_col in races.columns:
            races[col] = races[new_col].combine_first(races[col])
            races = races.drop(columns=[new_col])

    races["poll_count"] = pd.to_numeric(races["poll_count"], errors="coerce").fillna(0).astype(int)
    races["polling_active"] = races["poll_count"] > 0


    # Final safety sync: copy polling diagnostics from generated averages

    # into house_race_inputs.csv immediately before saving.

    for diagnostic_col, default in [

        ("effective_poll_count", 0.0),

        ("largest_pollster_weight_share", 0.0),

        ("only_partisan_or_internal_polls", False),

    ]:

        if diagnostic_col not in races.columns:

            races[diagnostic_col] = default

    

        if diagnostic_col in averages.columns:

            diagnostic_map = dict(

                zip(

                    averages["district_id"].astype(str),

                    averages[diagnostic_col],

                )

            )

            races[diagnostic_col] = (

                races["district_id"]

                .astype(str)

                .map(diagnostic_map)

                .fillna(default)

            )

    

    races["effective_poll_count"] = pd.to_numeric(

        races["effective_poll_count"],

        errors="coerce",

    ).fillna(0.0)

    

    races["largest_pollster_weight_share"] = pd.to_numeric(

        races["largest_pollster_weight_share"],

        errors="coerce",

    ).fillna(0.0)

    

    races.to_csv(HOUSE_RACE_INPUTS, index=False)

    print(f"Ingested {len(polls)} usable manual House poll(s).")
    print(f"Districts with polling: {(races['polling_active'] == True).sum()}")
    print(f"Wrote clean polls: {OUTPUT_CLEAN_POLLS}")
    print(f"Wrote polling averages: {OUTPUT_AVERAGES}")
    print(f"Updated race inputs: {HOUSE_RACE_INPUTS}")

    print()
    print("Polling averages:")
    print(averages.to_string(index=False))


if __name__ == "__main__":
    main()
