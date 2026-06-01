from pathlib import Path
from datetime import date
import pandas as pd
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


def ensure_poll_columns(polls):
    cols = [
        "race",
        "state",
        "district",
        "district_id",
        "pollster",
        "pollster_grade",
        "house_effect_dem",
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

    polls = ensure_poll_columns(polls)

    polls["state"] = polls["state"].apply(normalize_state)
    polls["district"] = polls.apply(
        lambda row: normalize_district_value(row.get("state", ""), row.get("district", "")),
        axis=1,
    )
    polls["district_id"] = polls.apply(
        lambda row: normalize_existing_district_id(
            row.get("district_id", ""),
            row.get("state", ""),
            row.get("district", ""),
        ),
        axis=1,
    )

    polls["dem_pct"] = safe_numeric(polls["dem_pct"])
    polls["gop_pct"] = safe_numeric(polls["gop_pct"])
    polls["ind_pct"] = safe_numeric(polls["ind_pct"], default=0.0)
    polls["other_pct"] = safe_numeric(polls["other_pct"], default=0.0)
    polls["undecided_pct"] = safe_numeric(polls["undecided_pct"], default=0.0)
    polls["house_effect_dem"] = safe_numeric(polls["house_effect_dem"], default=0.0)
    polls["sample_size"] = safe_numeric(polls["sample_size"], default=np.nan)

    polls["start_date"] = pd.to_datetime(polls["start_date"], errors="coerce")
    polls["end_date"] = pd.to_datetime(polls["end_date"], errors="coerce")

    # Validate matching districts.
    valid_districts = set(races["district_id"].dropna().astype(str))
    unmatched = sorted(set(polls["district_id"].dropna().astype(str)) - valid_districts - {""})

    if unmatched:
        print("WARNING: Some polls do not match any district_id in house_race_inputs.csv:")
        for d in unmatched[:30]:
            print(f"  - {d}")

    polls = polls[polls["district_id"].isin(valid_districts)].copy()

    # Drop rows that do not have usable Dem/GOP percentages.
    usable = (
        polls["dem_pct"].notna()
        & polls["gop_pct"].notna()
        & polls["district_id"].fillna("").astype(str).str.strip().ne("")
    )

    dropped = len(polls) - usable.sum()

    if dropped:
        print(f"WARNING: Dropping {dropped} polls with missing/invalid district_id or Dem/GOP percentages.")

    polls = polls[usable].copy()

    if polls.empty:
        print("No usable polls after validation. Clearing polling fields.")
        empty_poll_outputs(races)
        return

    polls["raw_margin_dem"] = polls["dem_pct"] - polls["gop_pct"]

    # Positive house effect means pollster is Dem-leaning, so subtract from margin.
    polls["polling_margin_dem"] = polls["raw_margin_dem"] - polls["house_effect_dem"]

    polls["poll_age_days"] = polls["end_date"].apply(
        lambda d: max(0, (as_of - d.date()).days) if pd.notna(d) else np.nan
    )

    polls["recency_weight"] = polls["end_date"].apply(lambda d: recency_weight(d, as_of))
    polls["sample_size_weight"] = polls["sample_size"].apply(sample_size_weight)
    polls["pollster_grade_weight"] = polls["pollster_grade"].apply(grade_weight)
    polls["sample_type_weight"] = polls["sample_type"].apply(sample_type_weight)

    polls["poll_weight"] = (
        polls["recency_weight"]
        * polls["sample_size_weight"]
        * polls["pollster_grade_weight"]
        * polls["sample_type_weight"]
    )

    # Avoid zero weights.
    polls["poll_weight"] = polls["poll_weight"].clip(lower=0.05)

    clean_cols = [
        "race",
        "state",
        "district",
        "district_id",
        "pollster",
        "pollster_grade",
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

    polls[clean_cols].to_csv(OUTPUT_CLEAN_POLLS, index=False)

    rows = []

    for district_id, group in polls.groupby("district_id"):
        total_weight = group["poll_weight"].sum()

        if total_weight <= 0:
            polling_margin = group["polling_margin_dem"].mean()
        else:
            polling_margin = (
                group["polling_margin_dem"] * group["poll_weight"]
            ).sum() / total_weight

        latest_end = group["end_date"].max()

        avg_age = group["poll_age_days"].mean()

        state = group["state"].iloc[0]
        district = group["district"].iloc[0]

        pollsters = ", ".join(
            group["pollster"].fillna("").astype(str).replace("", "Unknown").tolist()
        )

        rows.append(
            {
                "district_id": district_id,
                "state": state,
                "district": district,
                "polling_margin_dem": polling_margin,
                "poll_count": len(group),
                "latest_poll_end_date": latest_end.date().isoformat() if pd.notna(latest_end) else "",
                "avg_poll_age_days": avg_age,
                "total_poll_weight": total_weight,
                "polling_notes": f"Manual House polling average from {len(group)} poll(s): {pollsters}",
            }
        )

    averages = pd.DataFrame(rows)

    averages.to_csv(OUTPUT_AVERAGES, index=False)

    # Clear existing poll fields first.
    for col, default in [
        ("polling_margin_dem", np.nan),
        ("poll_count", 0),
        ("polling_active", False),
        ("latest_poll_end_date", ""),
        ("avg_poll_age_days", np.nan),
        ("total_poll_weight", 0.0),
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
