from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


MIT_PATH = Path(
    "historical/house/raw/2022/source_downloads/"
    "1976-2024-house.tab"
)

FEC_PATH = Path(
    "historical/house/raw/2022/source_downloads/"
    "fec_candidate_master/cn.txt"
)

RESULTS_DIR = Path("historical/house/processed")

OVERRIDE_PATH = Path(
    "historical/shared/overrides/"
    "candidate_match_overrides.csv"
)


FEC_COLUMNS = [
    "candidate_id",
    "candidate_name",
    "party_affiliation",
    "election_year",
    "candidate_state",
    "candidate_office",
    "candidate_district",
    "incumbent_challenger_status",
    "candidate_status",
    "principal_campaign_committee_id",
    "street_1",
    "street_2",
    "city",
    "mailing_state",
    "zip",
]


NICKNAME_GROUPS = [
    {"ROBERT", "BOB", "BOBBY", "ROB"},
    {"WILLIAM", "BILL", "BILLY", "WILL"},
    {"RICHARD", "RICK", "RICH", "DICK"},
    {"JAMES", "JIM", "JIMMY"},
    {"JOHN", "JACK", "JOHNNY"},
    {"JOSEPH", "JOE", "JOEY"},
    {"THOMAS", "TOM", "TOMMY"},
    {"MICHAEL", "MIKE", "MICKEY"},
    {"CHARLES", "CHUCK", "CHARLIE"},
    {"EDWARD", "ED", "EDDIE", "TED"},
    {"DANIEL", "DAN", "DANNY"},
    {"DAVID", "DAVE"},
    {"DONALD", "DON"},
    {"RONALD", "RON"},
    {"PATRICK", "PAT"},
    {"KATHERINE", "KATHRYN", "KATHLEEN", "KATE", "KATIE", "KATHY"},
    {"ELIZABETH", "LIZ", "BETH", "BETTY"},
    {"MARGARET", "MAGGIE", "MEG", "PEGGY"},
    {"JENNIFER", "JEN", "JENNY"},
    {"ANTHONY", "TONY"},
    {"ALEXANDER", "ALEX"},
    {"BENJAMIN", "BEN"},
    {"CHRISTOPHER", "CHRIS"},
    {"NICHOLAS", "NICK"},
    {"MATTHEW", "MATT"},
    {"TIMOTHY", "TIM"},
    {"STEPHEN", "STEVEN", "STEVE"},
    {"LAWRENCE", "LARRY"},
    {"JONATHAN", "JON"},
    {"JEFFREY", "JEFF"},
    {"GREGORY", "GREG"},
    {"RAYMOND", "RAY"},
    {"VINCENT", "VINCE"},
    {"JESUS", "CHUY"},
]


def ascii_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")


def clean_name(value: object) -> str:
    text = ascii_text(value).upper()

    text = text.replace("’", "'")
    text = re.sub(r"\([^)]*\)", " ", text)

    text = re.sub(
        r"\b(JR|SR|II|III|IV|V|DR|MR|MRS|MS|HON|REV)\b",
        " ",
        text,
    )

    text = re.sub(r"[^A-Z0-9, ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,")

    return text


def parse_name(value: object, fec_format: bool = False) -> dict[str, object]:
    clean = clean_name(value)

    if fec_format and "," in clean:
        last_part, remaining = clean.split(",", 1)
        last_tokens = last_part.split()
        other_tokens = remaining.strip().split()
        tokens = other_tokens + last_tokens
    else:
        tokens = clean.replace(",", " ").split()

    tokens = [
        token for token in tokens
        if token not in {"JR", "SR", "II", "III", "IV"}
    ]

    if not tokens:
        return {
            "clean": clean,
            "tokens": [],
            "first": "",
            "last": "",
            "middle": [],
        }

    return {
        "clean": clean,
        "tokens": tokens,
        "first": tokens[0],
        "last": tokens[-1],
        "middle": tokens[1:-1],
    }


def nickname_equivalent(left: str, right: str) -> bool:
    if left == right:
        return True

    for group in NICKNAME_GROUPS:
        if left in group and right in group:
            return True

    return False


def token_overlap(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)

    if not left_set or not right_set:
        return 0.0

    return len(left_set & right_set) / len(left_set | right_set)


def name_match_score(
    mit_name: object,
    fec_name: object,
) -> tuple[float, str]:
    mit = parse_name(mit_name, fec_format=False)
    fec = parse_name(fec_name, fec_format=True)

    if not mit["tokens"] or not fec["tokens"]:
        return 0.0, "missing_name"

    score = 0.0
    reasons = []

    if mit["clean"].replace(",", "") == fec["clean"].replace(",", ""):
        return 100.0, "identical_clean_name"

    if mit["last"] == fec["last"]:
        score += 55.0
        reasons.append("last_exact")
    elif (
        mit["last"].startswith(fec["last"])
        or fec["last"].startswith(mit["last"])
    ):
        score += 35.0
        reasons.append("last_prefix")
    else:
        return 0.0, "last_name_mismatch"

    mit_first = mit["first"]
    fec_first = fec["first"]

    # Some candidates campaign under a middle name or nickname:
    # BARRY MOORE vs FELIX BARRY MOORE,
    # AUSTIN SCOTT vs JAMES AUSTIN SCOTT, etc.
    fec_given_tokens = fec["tokens"][:-1]
    mit_given_tokens = mit["tokens"][:-1]

    first_match = nickname_equivalent(mit_first, fec_first)

    alternate_given_match = any(
        nickname_equivalent(mit_first, token)
        for token in fec_given_tokens
    ) or any(
        nickname_equivalent(fec_first, token)
        for token in mit_given_tokens
    )

    if first_match:
        score += 30.0
        reasons.append("first_match")
    elif alternate_given_match:
        score += 27.0
        reasons.append("alternate_given_name_match")
    elif (
        mit_first[:1]
        and mit_first[:1] == fec_first[:1]
    ):
        score += 12.0
        reasons.append("first_initial")
    else:
        score -= 10.0
        reasons.append("first_mismatch")

    overlap = token_overlap(
        mit["tokens"],
        fec["tokens"],
    )
    score += overlap * 15.0

    if overlap >= 0.5:
        reasons.append("token_overlap")

    return score, "+".join(reasons)


def normalize_party(value: object) -> str:
    text = clean_name(value)

    if text in {
        "DEMOCRAT",
        "DEMOCRATIC",
        "DEMOCRATIC FARMER LABOR",
        "DEM",
        "DFL",
    }:
        return "D"

    if text in {"REPUBLICAN", "REP"}:
        return "R"

    return "OTHER"


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def district_label(value: object) -> str:
    number = int(float(value))
    return "AL" if number == 0 else str(number)


def load_mit_candidates(cycle: int) -> pd.DataFrame:
    df = pd.read_csv(MIT_PATH, sep=",", low_memory=False)

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce",
    )

    df["candidatevotes"] = pd.to_numeric(
        df["candidatevotes"],
        errors="coerce",
    ).fillna(0).astype(int)

    df["totalvotes"] = pd.to_numeric(
        df["totalvotes"],
        errors="coerce",
    ).fillna(0).astype(int)

    df = df.loc[
        df["year"].eq(cycle)
        & df["office"].astype(str).str.upper().eq("US HOUSE")
        & df["stage"].astype(str).str.upper().eq("GEN")
        & ~df["special"].apply(parse_bool)
        & df["mode"].astype(str).str.upper().eq("TOTAL")
        & ~df["writein"].apply(parse_bool)
    ].copy()

    df["state"] = (
        df["state_po"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["district_label"] = df["district"].apply(
        district_label
    )

    df["race_id"] = (
        df["state"]
        + "-"
        + df["district_label"]
    )

    df["party_code"] = df["party"].apply(
        normalize_party
    )

    return df.loc[
        df["party_code"].isin(["D", "R"])
    ].copy()


def load_fec_candidates(cycle: int) -> pd.DataFrame:
    df = pd.read_csv(
        FEC_PATH,
        sep="|",
        names=FEC_COLUMNS,
        dtype=str,
        encoding="latin-1",
    )

    df = df.loc[
        df["candidate_office"].eq("H")
        & df["election_year"].eq(str(cycle))
    ].copy()

    df["state"] = (
        df["candidate_state"]
        .fillna("")
        .str.strip()
        .str.upper()
    )

    df["district_label"] = (
        df["candidate_district"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lstrip("0")
        .replace("", "AL")
    )

    df["race_id"] = (
        df["state"]
        + "-"
        + df["district_label"]
    )

    df["party_code"] = df["party_affiliation"].apply(
        normalize_party
    )

    return df.loc[
        df["party_code"].isin(["D", "R"])
    ].copy()


def load_overrides(cycle: int) -> pd.DataFrame:
    if not OVERRIDE_PATH.exists():
        return pd.DataFrame()

    overrides = pd.read_csv(
        OVERRIDE_PATH,
        dtype=str,
    ).fillna("")

    if overrides.empty:
        return overrides

    return overrides.loc[
        overrides["cycle"].astype(str).eq(str(cycle))
        & overrides["chamber"].str.upper().eq("HOUSE")
    ].copy()


def choose_match(
    mit_row: pd.Series,
    fec: pd.DataFrame,
    overrides: pd.DataFrame,
) -> dict[str, object]:
    override = overrides.loc[
        overrides["race_id"].eq(mit_row["race_id"])
        & overrides["party_code"].eq(mit_row["party_code"])
        & overrides["source_candidate_name"]
        .str.upper()
        .eq(str(mit_row["candidate"]).upper())
    ]

    if not override.empty:
        selected_override = override.iloc[0]

        fec_row = fec.loc[
            fec["candidate_id"].eq(
                selected_override["target_candidate_id"]
            )
        ]

        if not fec_row.empty:
            row = fec_row.iloc[0]

            return {
                "fec_row": row,
                "match_score": 100.0,
                "match_method": "manual_override",
                "match_reason": selected_override.get(
                    "notes",
                    "",
                ),
                "candidate_count_considered": 1,
                "second_best_score": np.nan,
            }

    candidates = fec.loc[
        fec["race_id"].eq(mit_row["race_id"])
        & fec["party_code"].eq(mit_row["party_code"])
    ].copy()

    if candidates.empty:
        return {
            "fec_row": None,
            "match_score": 0.0,
            "match_method": "unmatched",
            "match_reason": "no_same_race_party_candidates",
            "candidate_count_considered": 0,
            "second_best_score": np.nan,
        }

    scored = []

    for index, row in candidates.iterrows():
        score, reason = name_match_score(
            mit_row["candidate"],
            row["candidate_name"],
        )

        # Prefer active/statutory candidates when otherwise similar.
        # This is only a small tie-breaker, not a substitute for a name match.
        status_bonus = 0.0

        if str(row.get("candidate_status", "")).strip() == "C":
            status_bonus = 2.0
        elif str(row.get("candidate_status", "")).strip() == "P":
            status_bonus = 1.0

        scored.append(
            {
                "index": index,
                "score": score + status_bonus,
                "raw_name_score": score,
                "reason": reason,
            }
        )

    scored = sorted(
        scored,
        key=lambda item: item["score"],
        reverse=True,
    )

    best = scored[0]

    second_score = (
        scored[1]["score"]
        if len(scored) > 1
        else np.nan
    )

    best_row = candidates.loc[best["index"]]

    score_gap = (
        best["score"] - second_score
        if not pd.isna(second_score)
        else best["score"]
    )

    raw_name_score = float(best.get("raw_name_score", best["score"]))

    if raw_name_score >= 85 and score_gap >= 8:
        method = "high_confidence_automatic"
        retained_row = best_row
    elif raw_name_score >= 70 and score_gap >= 12:
        method = "probable_automatic"
        retained_row = best_row
    elif raw_name_score >= 45:
        method = "review_required"
        retained_row = best_row
    else:
        # Never attach a clearly nonmatching FEC identity merely because
        # it is the highest-scoring candidate in the same race and party.
        method = "unmatched"
        retained_row = None

    return {
        "fec_row": retained_row,
        "match_score": raw_name_score,
        "match_method": method,
        "match_reason": best["reason"],
        "candidate_count_considered": len(candidates),
        "second_best_score": second_score,
    }


def build_registry(cycle: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    mit = load_mit_candidates(cycle)
    fec = load_fec_candidates(cycle)
    overrides = load_overrides(cycle)

    registry_rows = []
    report_rows = []

    for _, mit_row in mit.iterrows():
        match = choose_match(
            mit_row,
            fec,
            overrides,
        )

        fec_row = match["fec_row"]

        if fec_row is None:
            fec_candidate_id = ""
            fec_candidate_name = ""
            incumbent_status = ""
            candidate_status = ""
            committee_id = ""
        else:
            fec_candidate_id = fec_row["candidate_id"]
            fec_candidate_name = fec_row["candidate_name"]
            incumbent_status = fec_row[
                "incumbent_challenger_status"
            ]
            candidate_status = fec_row["candidate_status"]
            committee_id = fec_row[
                "principal_campaign_committee_id"
            ]

        candidate_uid = (
            f"HOUSE_{cycle}_"
            f"{mit_row['race_id'].replace('-', '_')}_"
            f"{mit_row['party_code']}_"
            f"{re.sub(r'[^A-Z0-9]+', '_', clean_name(mit_row['candidate']))}"
        ).strip("_")

        vote_share = (
            mit_row["candidatevotes"]
            / mit_row["totalvotes"]
            if mit_row["totalvotes"] > 0
            else np.nan
        )

        registry_rows.append(
            {
                "candidate_uid": candidate_uid,
                "cycle": cycle,
                "chamber": "House",
                "race_id": mit_row["race_id"],
                "state": mit_row["state"],
                "district": mit_row["district_label"],
                "party_code": mit_row["party_code"],
                "candidate_name_canonical": mit_row["candidate"],
                "candidate_name_mit": mit_row["candidate"],
                "candidate_name_fec": fec_candidate_name,
                "fec_candidate_id": fec_candidate_id,
                "principal_campaign_committee_id": committee_id,
                "incumbent_challenger_status": incumbent_status,
                "is_incumbent": incumbent_status == "I",
                "is_challenger": incumbent_status == "C",
                "is_open_seat_candidate": incumbent_status == "O",
                "fec_candidate_status": candidate_status,
                "candidate_vote_total": mit_row["candidatevotes"],
                "candidate_vote_share": vote_share,
                "match_method": match["match_method"],
                "match_score": match["match_score"],
                "match_reason": match["match_reason"],
                "second_best_score": match[
                    "second_best_score"
                ],
                "candidate_count_considered": match[
                    "candidate_count_considered"
                ],
                "source_results": (
                    "MIT Election Data and Science Lab"
                ),
                "source_candidate_identity": (
                    "Federal Election Commission"
                ),
            }
        )

        report_rows.append(
            {
                "race_id": mit_row["race_id"],
                "party_code": mit_row["party_code"],
                "mit_candidate_name": mit_row["candidate"],
                "fec_candidate_name": fec_candidate_name,
                "fec_candidate_id": fec_candidate_id,
                "incumbent_challenger_status": incumbent_status,
                "match_method": match["match_method"],
                "match_score": match["match_score"],
                "second_best_score": match[
                    "second_best_score"
                ],
                "candidate_count_considered": match[
                    "candidate_count_considered"
                ],
                "match_reason": match["match_reason"],
            }
        )

    registry = pd.DataFrame(registry_rows)
    report = pd.DataFrame(report_rows)

    registry = registry.sort_values(
        ["state", "district", "party_code", "candidate_name_canonical"]
    )

    report = report.sort_values(
        ["match_method", "match_score"],
        ascending=[True, False],
    )

    return registry, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", type=int, default=2022)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    registry, report = build_registry(args.cycle)

    registry_path = (
        RESULTS_DIR
        / f"house_candidate_registry_{args.cycle}.csv"
    )

    report_path = (
        RESULTS_DIR
        / f"house_candidate_registry_{args.cycle}_match_report.csv"
    )

    validation_path = (
        RESULTS_DIR
        / f"house_candidate_registry_{args.cycle}_validation.txt"
    )

    registry.to_csv(registry_path, index=False)
    report.to_csv(report_path, index=False)

    method_counts = (
        registry["match_method"]
        .value_counts(dropna=False)
    )

    high_confidence = registry["match_method"].isin(
        [
            "manual_override",
            "high_confidence_automatic",
        ]
    )

    usable = registry["match_method"].isin(
        [
            "manual_override",
            "high_confidence_automatic",
            "probable_automatic",
        ]
    )

    validation_lines = [
        f"{args.cycle} House Candidate Registry Validation",
        "=" * 42,
        "",
        f"Registry rows: {len(registry)}",
        f"Unique candidate UIDs: {registry['candidate_uid'].nunique()}",
        f"Unique races: {registry['race_id'].nunique()}",
        f"Matched to FEC ID: {registry['fec_candidate_id'].ne('').sum()}",
        f"High-confidence matches: {high_confidence.sum()}",
        f"Usable automatic/manual matches: {usable.sum()}",
        f"Review-required or unmatched: {(~usable).sum()}",
        "",
        "Match methods:",
        method_counts.to_string(),
        "",
        "Incumbency codes among usable matches:",
        registry.loc[
            usable,
            "incumbent_challenger_status",
        ].value_counts(dropna=False).to_string(),
    ]

    validation_text = "\n".join(validation_lines)
    validation_path.write_text(validation_text)

    print(validation_text)
    print()
    print(f"Wrote: {registry_path}")
    print(f"Wrote: {report_path}")
    print(f"Wrote: {validation_path}")


if __name__ == "__main__":
    main()
