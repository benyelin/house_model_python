from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RAW_PATH = Path(
    "historical/house/raw/2022/source_downloads/"
    "1976-2024-house.tab"
)

PROCESSED_DIR = Path("historical/house/processed")


def normalize_district(state_po: str, district: object) -> tuple[str, str]:
    state = str(state_po).strip().upper()

    try:
        district_num = int(float(district))
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid district value for {state}: {district!r}"
        )

    if district_num == 0:
        district_label = "AL"
    else:
        district_label = str(district_num)

    return state, district_label


def standardize_party(value: object) -> str:
    text = str(value).strip().upper()

    if text in {
        "DEMOCRAT",
        "DEMOCRATIC",
        "DEMOCRATIC-FARMER-LABOR",
    }:
        return "D"

    if text == "REPUBLICAN":
        return "R"

    return "OTHER"


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def classify_party_structure(group: pd.DataFrame) -> dict[str, object]:
    named = group.loc[~group["is_writein"]].copy()

    dem_named = named.loc[named["party_code"].eq("D")]
    gop_named = named.loc[named["party_code"].eq("R")]
    other_named = named.loc[named["party_code"].eq("OTHER")]

    dem_count = len(dem_named)
    gop_count = len(gop_named)
    other_count = len(other_named)
    named_count = len(named)

    has_dem = dem_count > 0
    has_gop = gop_count > 0

    if has_dem and has_gop:
        structure = "D_vs_R"
    elif dem_count >= 2 and not has_gop:
        structure = "D_vs_D"
    elif gop_count >= 2 and not has_dem:
        structure = "R_vs_R"
    elif has_dem and other_count > 0:
        structure = "D_vs_Other"
    elif has_gop and other_count > 0:
        structure = "R_vs_Other"
    elif has_dem and named_count == 1:
        structure = "D_unopposed"
    elif has_gop and named_count == 1:
        structure = "R_unopposed"
    elif not has_dem and not has_gop and other_count >= 2:
        structure = "Other_vs_Other"
    elif named_count == 1:
        structure = "Other_unopposed"
    else:
        structure = "Unresolved"

    return {
        "has_dem_candidate": has_dem,
        "has_gop_candidate": has_gop,
        "major_party_contested": bool(has_dem and has_gop),
        "named_candidate_count": named_count,
        "named_dem_count": dem_count,
        "named_gop_count": gop_count,
        "named_other_count": other_count,
        "general_election_party_structure": structure,
        "uncontested": structure in {
            "D_unopposed",
            "R_unopposed",
            "Other_unopposed",
        },
    }


def choose_candidate(group: pd.DataFrame, party_code: str) -> pd.Series | None:
    subset = group.loc[group["party_code"].eq(party_code)].copy()

    if subset.empty:
        return None

    subset = subset.sort_values(
        ["candidatevotes", "candidate"],
        ascending=[False, True],
    )

    return subset.iloc[0]


def build_cycle(cycle: int, raw_path: Path) -> tuple[pd.DataFrame, str]:
    raw = pd.read_csv(raw_path, sep=",", low_memory=False)

    required = [
        "year",
        "state",
        "state_po",
        "office",
        "district",
        "stage",
        "special",
        "candidate",
        "party",
        "writein",
        "mode",
        "candidatevotes",
        "totalvotes",
        "unofficial",
    ]

    missing = [column for column in required if column not in raw.columns]

    if missing:
        raise ValueError(
            f"Raw dataset is missing required columns: {missing}"
        )

    raw["year"] = pd.to_numeric(raw["year"], errors="coerce")

    cycle_rows = raw.loc[
        raw["year"].eq(cycle)
        & raw["office"].astype(str).str.upper().eq("US HOUSE")
        & raw["stage"].astype(str).str.upper().eq("GEN")
        & ~raw["special"].astype(str).str.lower().isin(["true", "1"])
        & raw["mode"].astype(str).str.upper().eq("TOTAL")
    ].copy()

    cycle_rows["candidatevotes"] = pd.to_numeric(
        cycle_rows["candidatevotes"],
        errors="coerce",
    ).fillna(0).astype(int)

    cycle_rows["totalvotes"] = pd.to_numeric(
        cycle_rows["totalvotes"],
        errors="coerce",
    ).fillna(0).astype(int)

    cycle_rows["candidate"] = (
        cycle_rows["candidate"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    cycle_rows["party_code"] = cycle_rows["party"].apply(
        standardize_party
    )

    cycle_rows["is_writein"] = cycle_rows["writein"].apply(
        parse_bool
    )

    normalized = cycle_rows.apply(
        lambda row: normalize_district(
            row["state_po"],
            row["district"],
        ),
        axis=1,
        result_type="expand",
    )

    normalized.columns = ["state", "district"]

    cycle_rows["state"] = normalized["state"]
    cycle_rows["district"] = normalized["district"]

    cycle_rows["race_id"] = (
        cycle_rows["state"]
        + "-"
        + cycle_rows["district"]
    )

    output_rows = []
    warnings = []

    for race_id, group in cycle_rows.groupby("race_id", sort=True):
        state = group["state"].iloc[0]
        district = group["district"].iloc[0]

        structure = classify_party_structure(group)

        dem = choose_candidate(group, "D")
        gop = choose_candidate(group, "R")

        dem_votes = int(dem["candidatevotes"]) if dem is not None else 0
        gop_votes = int(gop["candidatevotes"]) if gop is not None else 0

        other_votes = int(
            group.loc[
                ~group["party_code"].isin(["D", "R"]),
                "candidatevotes",
            ].sum()
        )

        reported_total_votes = int(group["totalvotes"].max())
        summed_candidate_votes = int(group["candidatevotes"].sum())

        total_votes = max(
            reported_total_votes,
            summed_candidate_votes,
        )

        if total_votes <= 0:
            actual_dem_margin = np.nan
            dem_vote_share = np.nan
            gop_vote_share = np.nan
        else:
            actual_dem_margin = (
                100.0 * (dem_votes - gop_votes) / total_votes
            )
            dem_vote_share = 100.0 * dem_votes / total_votes
            gop_vote_share = 100.0 * gop_votes / total_votes

        if dem_votes > gop_votes and dem_votes > other_votes:
            actual_winner = "D"
        elif gop_votes > dem_votes and gop_votes > other_votes:
            actual_winner = "R"
        else:
            highest = group.sort_values(
                "candidatevotes",
                ascending=False,
            ).iloc[0]

            actual_winner = (
                highest["party_code"]
                if highest["party_code"] in {"D", "R"}
                else "Other"
            )

        dem_candidate = (
            str(dem["candidate"]).strip()
            if dem is not None
            else ""
        )

        gop_candidate = (
            str(gop["candidate"]).strip()
            if gop is not None
            else ""
        )

        other_candidate_names = [
            str(value).strip()
            for value in group.loc[
                ~group["party_code"].isin(["D", "R"])
                & ~group["is_writein"],
                "candidate",
            ].tolist()
            if str(value).strip()
        ]

        if dem is None:
            warnings.append(f"{race_id}: no Democratic candidate")

        if gop is None:
            warnings.append(f"{race_id}: no Republican candidate")

        if reported_total_votes != summed_candidate_votes:
            warnings.append(
                f"{race_id}: reported total {reported_total_votes} "
                f"differs from summed candidate votes "
                f"{summed_candidate_votes}"
            )

        output_rows.append(
            {
                "cycle": cycle,
                "chamber": "House",
                "race_id": race_id,
                "district_id": race_id,
                "state": state,
                "district": district,
                "election_date": f"{cycle}-11-08",
                "election_type": "General",
                "dem_candidate": dem_candidate,
                "gop_candidate": gop_candidate,
                "other_candidates": " | ".join(other_candidate_names),
                "dem_vote_total": dem_votes,
                "gop_vote_total": gop_votes,
                "other_vote_total": other_votes,
                "total_vote": total_votes,
                "dem_vote_share": dem_vote_share,
                "gop_vote_share": gop_vote_share,
                "actual_dem_margin": actual_dem_margin,
                "actual_winner": actual_winner,
                "has_dem_candidate": structure["has_dem_candidate"],
                "has_gop_candidate": structure["has_gop_candidate"],
                "major_party_contested": structure["major_party_contested"],
                "named_candidate_count": structure["named_candidate_count"],
                "named_dem_count": structure["named_dem_count"],
                "named_gop_count": structure["named_gop_count"],
                "named_other_count": structure["named_other_count"],
                "general_election_party_structure": structure[
                    "general_election_party_structure"
                ],
                "uncontested": structure["uncontested"],
                "uncontested_dem": bool(
                    structure["general_election_party_structure"]
                    == "D_unopposed"
                ),
                "uncontested_gop": bool(
                    structure["general_election_party_structure"]
                    == "R_unopposed"
                ),
                "include_in_major_party_margin_scoring": bool(
                    structure["major_party_contested"]
                ),
                "special_handling_notes": (
                    ""
                    if structure["general_election_party_structure"]
                    == "D_vs_R"
                    else (
                        "Nonstandard general-election party structure: "
                        + str(
                            structure[
                                "general_election_party_structure"
                            ]
                        )
                    )
                ),
                "source_name": (
                    "MIT Election Data and Science Lab, "
                    "U.S. House 1976-2024"
                ),
                "source_url": "doi:10.7910/DVN/IG0UN2",
                "retrieved_date": "2026-07-15",
            }
        )

    results = pd.DataFrame(output_rows).sort_values(
        ["state", "district"],
        key=lambda series: series.map(
            lambda value: (
                0 if str(value) == "AL"
                else int(value)
                if str(value).isdigit()
                else 999
            )
        )
        if series.name == "district"
        else series,
    )

    duplicate_races = int(results["race_id"].duplicated().sum())
    missing_margins = int(results["actual_dem_margin"].isna().sum())
    missing_winners = int(results["actual_winner"].eq("").sum())
    missing_dem = int(results["dem_candidate"].eq("").sum())
    missing_gop = int(results["gop_candidate"].eq("").sum())

    expected_districts = 435

    report_lines = [
        f"{cycle} House Results Validation",
        "=" * 34,
        "",
        f"Raw candidate-level rows: {len(cycle_rows)}",
        f"Processed race rows: {len(results)}",
        f"Expected race rows: {expected_districts}",
        f"Unique race IDs: {results['race_id'].nunique()}",
        f"Duplicate race IDs: {duplicate_races}",
        f"Missing Democratic candidates: {missing_dem}",
        f"Missing Republican candidates: {missing_gop}",
        f"Missing actual margins: {missing_margins}",
        f"Missing winners: {missing_winners}",
        f"Major-party-contested races: {int(results['major_party_contested'].sum())}",
        f"Nonstandard party structures: {int((~results['major_party_contested']).sum())}",
        f"Actually uncontested races: {int(results['uncontested'].sum())}",
        f"Uncontested Democratic races: {int(results['uncontested_dem'].sum())}",
        f"Uncontested Republican races: {int(results['uncontested_gop'].sum())}",
        "",
        "Party structure counts:",
        results[
            "general_election_party_structure"
        ].value_counts(dropna=False).to_string(),
        "",
        f"Total Democratic votes: {int(results['dem_vote_total'].sum())}",
        f"Total Republican votes: {int(results['gop_vote_total'].sum())}",
        f"Total other votes: {int(results['other_vote_total'].sum())}",
        f"Total votes: {int(results['total_vote'].sum())}",
        "",
        "Validation status:",
    ]

    failures = []

    if len(results) != expected_districts:
        failures.append(
            f"Expected {expected_districts} races but found {len(results)}"
        )

    if duplicate_races:
        failures.append(f"Found {duplicate_races} duplicate race IDs")

    if missing_margins:
        failures.append(f"Found {missing_margins} missing margins")

    if missing_winners:
        failures.append(f"Found {missing_winners} missing winners")

    if failures:
        report_lines.append("FAILED")
        report_lines.extend(f"- {failure}" for failure in failures)
    else:
        report_lines.append("PASSED")

    report_lines.extend(
        [
            "",
            "Warnings:",
        ]
    )

    if warnings:
        report_lines.extend(f"- {warning}" for warning in warnings)
    else:
        report_lines.append("- None")

    report = "\n".join(report_lines)

    if failures:
        raise RuntimeError(report)

    return results, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", type=int, default=2022)
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=RAW_PATH,
    )
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    results, report = build_cycle(
        cycle=args.cycle,
        raw_path=args.raw_path,
    )

    results_path = (
        PROCESSED_DIR
        / f"house_{args.cycle}_results.csv"
    )

    validation_path = (
        PROCESSED_DIR
        / f"house_{args.cycle}_results_validation.txt"
    )

    results.to_csv(results_path, index=False)
    validation_path.write_text(report)

    print(report)
    print()
    print(f"Wrote: {results_path}")
    print(f"Wrote: {validation_path}")


if __name__ == "__main__":
    main()
