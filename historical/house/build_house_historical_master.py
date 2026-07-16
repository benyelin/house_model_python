from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parents[2]

RESULTS_PATH = (
    BASE
    / "historical/house/processed/house_2022_results.csv"
)

REGISTRY_PATH = (
    BASE
    / "historical/house/processed/"
    "house_candidate_registry_2022.csv"
)

OUTPUT_PATH = (
    BASE
    / "historical/house/master/house_2022_master.csv"
)

VALIDATION_PATH = (
    BASE
    / "historical/house/master/"
    "house_2022_master_validation.txt"
)


def build_candidate_side(
    registry: pd.DataFrame,
    party_code: str,
    prefix: str,
) -> pd.DataFrame:
    side = registry.loc[
        registry["party_code"].eq(party_code)
    ].copy()

    keep = [
        "race_id",
        "candidate_name_mit",
        "candidate_uid",
        "fec_candidate_id",
        "principal_campaign_committee_id",
        "incumbent_challenger_status",
        "is_incumbent",
        "is_challenger",
        "is_open_seat_candidate",
        "fec_candidate_status",
        "match_method",
        "match_score",
    ]

    missing = [
        column for column in keep
        if column not in side.columns
    ]

    if missing:
        raise ValueError(
            f"Registry missing required columns: {missing}"
        )

    side = side[keep].copy()

    rename = {
        "candidate_name_mit": f"{prefix}_candidate_join_name",
        "candidate_uid": f"{prefix}_candidate_uid",
        "fec_candidate_id": f"{prefix}_fec_candidate_id",
        "principal_campaign_committee_id": (
            f"{prefix}_principal_campaign_committee_id"
        ),
        "incumbent_challenger_status": f"{prefix}_fec_status",
        "is_incumbent": f"{prefix}_is_incumbent",
        "is_challenger": f"{prefix}_is_challenger",
        "is_open_seat_candidate": (
            f"{prefix}_is_open_seat_candidate"
        ),
        "fec_candidate_status": f"{prefix}_fec_candidate_status",
        "match_method": f"{prefix}_match_method",
        "match_score": f"{prefix}_match_score",
    }

    side = side.rename(columns=rename)

    duplicate_keys = side.duplicated(
        ["race_id", f"{prefix}_candidate_join_name"],
        keep=False,
    )

    if duplicate_keys.any():
        duplicates = side.loc[
            duplicate_keys,
            ["race_id", f"{prefix}_candidate_join_name"],
        ]

        raise ValueError(
            "Duplicate registry candidate join keys found:\n"
            + duplicates.to_string(index=False)
        )

    return side


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing results file: {RESULTS_PATH}"
        )

    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Missing registry file: {REGISTRY_PATH}"
        )

    results = pd.read_csv(
        RESULTS_PATH,
        dtype={
            "race_id": str,
            "dem_candidate": str,
            "gop_candidate": str,
        },
    )

    registry = pd.read_csv(
        REGISTRY_PATH,
        dtype={
            "race_id": str,
            "candidate_name_mit": str,
            "party_code": str,
        },
    )

    dem = build_candidate_side(
        registry=registry,
        party_code="D",
        prefix="dem",
    )

    gop = build_candidate_side(
        registry=registry,
        party_code="R",
        prefix="gop",
    )

    master = results.merge(
        dem,
        left_on=["race_id", "dem_candidate"],
        right_on=["race_id", "dem_candidate_join_name"],
        how="left",
        validate="one_to_one",
    )

    master = master.merge(
        gop,
        left_on=["race_id", "gop_candidate"],
        right_on=["race_id", "gop_candidate_join_name"],
        how="left",
        validate="one_to_one",
    )

    master["dem_is_incumbent"] = (
        master["dem_is_incumbent"]
        .fillna(False)
        .astype(bool)
    )

    master["gop_is_incumbent"] = (
        master["gop_is_incumbent"]
        .fillna(False)
        .astype(bool)
    )

    dem_open_code = master["dem_fec_status"].fillna("").eq("O")
    gop_open_code = master["gop_fec_status"].fillna("").eq("O")

    both_incumbents = (
        master["dem_is_incumbent"]
        & master["gop_is_incumbent"]
    )

    any_incumbent = (
        master["dem_is_incumbent"]
        | master["gop_is_incumbent"]
    )

    any_open_code = dem_open_code | gop_open_code

    master["incumbency_status"] = "unresolved"

    master.loc[
        both_incumbents,
        "incumbency_status",
    ] = "member_vs_member"

    master.loc[
        any_incumbent & ~both_incumbents,
        "incumbency_status",
    ] = "confirmed_incumbent_running"

    master.loc[
        ~any_incumbent & any_open_code,
        "incumbency_status",
    ] = "confirmed_open"

    # Nullable Boolean: unresolved races should not silently become False.
    master["open_seat"] = pd.Series(
        pd.NA,
        index=master.index,
        dtype="boolean",
    )

    master.loc[
        master["incumbency_status"].eq("confirmed_open"),
        "open_seat",
    ] = True

    master.loc[
        master["incumbency_status"].isin(
            [
                "confirmed_incumbent_running",
                "member_vs_member",
            ]
        ),
        "open_seat",
    ] = False

    master["incumbent_party"] = ""

    master.loc[
        master["dem_is_incumbent"]
        & ~master["gop_is_incumbent"],
        "incumbent_party",
    ] = "D"

    master.loc[
        master["gop_is_incumbent"]
        & ~master["dem_is_incumbent"],
        "incumbent_party",
    ] = "R"

    master.loc[
        both_incumbents,
        "incumbent_party",
    ] = "Both"

    master["incumbency_data_complete"] = (
        master["incumbency_status"].ne("unresolved")
    )

    if len(master) != 435:
        raise ValueError(
            f"Expected 435 master rows; found {len(master)}."
        )

    if master["race_id"].duplicated().any():
        duplicates = master.loc[
            master["race_id"].duplicated(False),
            "race_id",
        ].tolist()

        raise ValueError(
            "Duplicate master race IDs: "
            + ", ".join(duplicates)
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    master.to_csv(OUTPUT_PATH, index=False)

    missing_dem_registry = int(
        (
            master["dem_candidate"].fillna("").ne("")
            & master["dem_candidate_uid"].isna()
        ).sum()
    )

    missing_gop_registry = int(
        (
            master["gop_candidate"].fillna("").ne("")
            & master["gop_candidate_uid"].isna()
        ).sum()
    )

    validation_lines = [
        "2022 House Historical Master Validation",
        "=" * 39,
        "",
        f"Master rows: {len(master)}",
        f"Unique race IDs: {master['race_id'].nunique()}",
        f"Columns: {len(master.columns)}",
        (
            "Democratic candidates lacking registry joins: "
            f"{missing_dem_registry}"
        ),
        (
            "Republican candidates lacking registry joins: "
            f"{missing_gop_registry}"
        ),
        (
            "Races with complete available FEC identity data: "
            f"{int(master['incumbency_data_complete'].sum())}"
        ),
        (
            "Democratic incumbents identified: "
            f"{int(master['dem_is_incumbent'].sum())}"
        ),
        (
            "Republican incumbents identified: "
            f"{int(master['gop_is_incumbent'].sum())}"
        ),
        (
            "Confirmed open seats: "
            f"{int(master['open_seat'].fillna(False).sum())}"
        ),
        (
            "Unresolved incumbency classifications: "
            f"{int(master['incumbency_status'].eq('unresolved').sum())}"
        ),
        "",
        "Incumbency status counts:",
        master[
            "incumbency_status"
        ].value_counts(dropna=False).to_string(),
        "",
        "Important limitation:",
        (
            "Unresolved races retain a missing open_seat value rather than "
            "being silently classified as open. These races require either "
            "a candidate-identity override or a separate historical-status "
            "override."
        ),
    ]

    validation_text = "\n".join(validation_lines)
    VALIDATION_PATH.write_text(validation_text)

    print(validation_text)
    print()
    print(f"Wrote: {OUTPUT_PATH}")
    print(f"Wrote: {VALIDATION_PATH}")


if __name__ == "__main__":
    main()
