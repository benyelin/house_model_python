from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


WAREHOUSE_PATH = Path(
    "historical/house/warehouse/"
    "house_historical_results_2012_2022.csv"
)

OUTPUT_DIR = Path("historical/house/elasticity")
OUTPUT_PATH = OUTPUT_DIR / "house_district_swing_observations_2012_2020.csv"
VALIDATION_PATH = (
    OUTPUT_DIR / "house_district_swing_observations_2012_2020_validation.txt"
)

BOUNDARY_ERA_CYCLES = (2012, 2014, 2016, 2018, 2020)


def calculate_two_party_margin(
    dem_votes: pd.Series,
    gop_votes: pd.Series,
) -> pd.Series:
    """Return Democratic two-party margin on a -100 to +100 scale."""
    denominator = dem_votes + gop_votes

    return np.where(
        denominator.gt(0),
        100.0 * (dem_votes - gop_votes) / denominator,
        np.nan,
    )


def build_transition(
    warehouse: pd.DataFrame,
    cycle_from: int,
    cycle_to: int,
) -> pd.DataFrame:
    """Build district swing observations for one consecutive-cycle pair."""
    required_columns = {
        "cycle",
        "race_id",
        "state",
        "district",
        "dem_vote_total",
        "gop_vote_total",
        "actual_winner",
        "major_party_contested",
        "include_in_major_party_margin_scoring",
        "general_election_party_structure",
    }

    missing = sorted(required_columns - set(warehouse.columns))

    if missing:
        raise ValueError(
            f"Warehouse is missing required columns: {missing}"
        )

    usable = warehouse.loc[
        warehouse["include_in_major_party_margin_scoring"].eq(True)
        & warehouse["major_party_contested"].eq(True)
        & warehouse["cycle"].isin([cycle_from, cycle_to])
    ].copy()

    usable["dem_vote_total"] = pd.to_numeric(
        usable["dem_vote_total"],
        errors="coerce",
    )
    usable["gop_vote_total"] = pd.to_numeric(
        usable["gop_vote_total"],
        errors="coerce",
    )

    usable["district_two_party_margin_dem"] = calculate_two_party_margin(
        usable["dem_vote_total"],
        usable["gop_vote_total"],
    )

    previous = usable.loc[
        usable["cycle"].eq(cycle_from)
    ].copy()

    current = usable.loc[
        usable["cycle"].eq(cycle_to)
    ].copy()

    previous = previous.rename(
        columns={
            "dem_vote_total": "dem_vote_total_from",
            "gop_vote_total": "gop_vote_total_from",
            "actual_winner": "actual_winner_from",
            "general_election_party_structure": (
                "party_structure_from"
            ),
            "district_two_party_margin_dem": (
                "district_margin_from"
            ),
        }
    )

    current = current.rename(
        columns={
            "dem_vote_total": "dem_vote_total_to",
            "gop_vote_total": "gop_vote_total_to",
            "actual_winner": "actual_winner_to",
            "general_election_party_structure": (
                "party_structure_to"
            ),
            "district_two_party_margin_dem": (
                "district_margin_to"
            ),
        }
    )

    previous_columns = [
        "race_id",
        "state",
        "district",
        "dem_vote_total_from",
        "gop_vote_total_from",
        "actual_winner_from",
        "party_structure_from",
        "district_margin_from",
    ]

    current_columns = [
        "race_id",
        "state",
        "district",
        "dem_vote_total_to",
        "gop_vote_total_to",
        "actual_winner_to",
        "party_structure_to",
        "district_margin_to",
    ]

    merged = previous[previous_columns].merge(
        current[current_columns],
        on=["race_id", "state", "district"],
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise RuntimeError(
            f"No common scorable districts for {cycle_from}->{cycle_to}."
        )

    national_dem_votes_from = merged["dem_vote_total_from"].sum()
    national_gop_votes_from = merged["gop_vote_total_from"].sum()
    national_dem_votes_to = merged["dem_vote_total_to"].sum()
    national_gop_votes_to = merged["gop_vote_total_to"].sum()

    national_margin_from = float(
        calculate_two_party_margin(
            pd.Series([national_dem_votes_from]),
            pd.Series([national_gop_votes_from]),
        )[0]
    )

    national_margin_to = float(
        calculate_two_party_margin(
            pd.Series([national_dem_votes_to]),
            pd.Series([national_gop_votes_to]),
        )[0]
    )

    national_swing = national_margin_to - national_margin_from

    merged.insert(0, "cycle_from", cycle_from)
    merged.insert(1, "cycle_to", cycle_to)
    merged.insert(2, "transition", f"{cycle_from}_to_{cycle_to}")

    merged["district_swing_dem"] = (
        merged["district_margin_to"]
        - merged["district_margin_from"]
    )

    merged["national_common_district_margin_from"] = (
        national_margin_from
    )
    merged["national_common_district_margin_to"] = national_margin_to
    merged["national_swing_dem"] = national_swing

    merged["raw_swing_ratio"] = np.where(
        abs(national_swing) > 1e-12,
        merged["district_swing_dem"] / national_swing,
        np.nan,
    )

    merged["same_boundary_era"] = True
    merged["eligible_for_elasticity_estimation"] = (
        merged["district_margin_from"].notna()
        & merged["district_margin_to"].notna()
        & np.isfinite(merged["district_swing_dem"])
        & np.isfinite(merged["national_swing_dem"])
        & merged["national_swing_dem"].abs().gt(0.25)
    )

    return merged


def build_swing_dataset(
    warehouse: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """Build all consecutive transitions within the 2012-2020 map era."""
    frames: list[pd.DataFrame] = []

    for cycle_from, cycle_to in zip(
        BOUNDARY_ERA_CYCLES[:-1],
        BOUNDARY_ERA_CYCLES[1:],
    ):
        transition = build_transition(
            warehouse=warehouse,
            cycle_from=cycle_from,
            cycle_to=cycle_to,
        )
        frames.append(transition)

    observations = pd.concat(
        frames,
        ignore_index=True,
    )

    observations = observations.sort_values(
        ["cycle_from", "state", "district"],
        key=lambda series: series.map(
            lambda value: (
                0
                if str(value) == "AL"
                else int(value)
                if str(value).isdigit()
                else 999
            )
        )
        if series.name == "district"
        else series,
    ).reset_index(drop=True)

    duplicate_observations = int(
        observations.duplicated(
            ["cycle_from", "cycle_to", "race_id"]
        ).sum()
    )

    missing_district_swings = int(
        observations["district_swing_dem"].isna().sum()
    )

    missing_national_swings = int(
        observations["national_swing_dem"].isna().sum()
    )

    invalid_margin_bounds = int(
        (
            observations[
                ["district_margin_from", "district_margin_to"]
            ].abs()
            > 100.000001
        ).any(axis=1).sum()
    )

    transition_counts = observations.groupby(
        "transition"
    ).size()

    expected_transition_counts = {
        "2012_to_2014": 328,
        "2014_to_2016": 319,
        "2016_to_2018": 342,
        "2018_to_2020": 374,
    }

    actual_transition_counts = {
        str(key): int(value)
        for key, value in transition_counts.items()
    }

    failures: list[str] = []

    if duplicate_observations:
        failures.append(
            f"Found {duplicate_observations} duplicate observations."
        )

    if missing_district_swings:
        failures.append(
            f"Found {missing_district_swings} missing district swings."
        )

    if missing_national_swings:
        failures.append(
            f"Found {missing_national_swings} missing national swings."
        )

    if invalid_margin_bounds:
        failures.append(
            f"Found {invalid_margin_bounds} margins outside [-100, 100]."
        )

    if actual_transition_counts != expected_transition_counts:
        failures.append(
            "Transition counts differ from the previously audited "
            f"coverage. Expected {expected_transition_counts}; "
            f"found {actual_transition_counts}."
        )

    eligible_count = int(
        observations["eligible_for_elasticity_estimation"].sum()
    )

    report_lines = [
        "House District Swing Dataset Validation",
        "=" * 39,
        "",
        "Boundary era: 2012-2020 congressional districts",
        f"Observations: {len(observations)}",
        f"Eligible observations: {eligible_count}",
        f"Unique districts: {observations['race_id'].nunique()}",
        f"Duplicate transition/district observations: "
        f"{duplicate_observations}",
        f"Missing district swings: {missing_district_swings}",
        f"Missing national swings: {missing_national_swings}",
        f"Margins outside [-100, 100]: {invalid_margin_bounds}",
        "",
        "Observations by transition:",
        transition_counts.to_string(),
        "",
        "National two-party movement by transition:",
    ]

    transition_summary = (
        observations.groupby("transition", as_index=False)
        .agg(
            common_districts=("race_id", "size"),
            national_margin_from=(
                "national_common_district_margin_from",
                "first",
            ),
            national_margin_to=(
                "national_common_district_margin_to",
                "first",
            ),
            national_swing_dem=("national_swing_dem", "first"),
            mean_district_swing_dem=(
                "district_swing_dem",
                "mean",
            ),
            median_district_swing_dem=(
                "district_swing_dem",
                "median",
            ),
        )
    )

    report_lines.append(
        transition_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    report_lines.extend(
        [
            "",
            "District observation counts:",
            observations.groupby("race_id").size()
            .value_counts()
            .sort_index()
            .to_string(),
            "",
            "Validation status:",
        ]
    )

    if failures:
        report_lines.append("FAILED")
        report_lines.extend(
            f"- {failure}"
            for failure in failures
        )
    else:
        report_lines.append("PASSED")

    report = "\n".join(report_lines)

    if failures:
        raise RuntimeError(report)

    return observations, report


def main() -> None:
    warehouse = pd.read_csv(WAREHOUSE_PATH)

    observations, report = build_swing_dataset(warehouse)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    observations.to_csv(OUTPUT_PATH, index=False)
    VALIDATION_PATH.write_text(report)

    print(report)
    print()
    print(f"Wrote: {OUTPUT_PATH}")
    print(f"Wrote: {VALIDATION_PATH}")


if __name__ == "__main__":
    main()
