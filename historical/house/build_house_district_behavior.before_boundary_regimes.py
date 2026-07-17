from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RESULTS_PATH = (
    PROJECT_ROOT
    / "historical/house/warehouse/"
    "house_historical_results_2012_2022.csv"
)

DEFAULT_PROFILE_PATH = (
    PROJECT_ROOT
    / "historical/house/warehouse/"
    "house_structural_profile.csv"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "historical/house/warehouse/"
    "house_district_behavior.csv"
)

DEFAULT_VALIDATION_PATH = (
    PROJECT_ROOT
    / "historical/house/warehouse/"
    "house_district_behavior_validation.txt"
)


PRE_REDISTRICTING_CYCLES = (
    2012,
    2014,
    2016,
    2018,
    2020,
)

ALL_CYCLES = (
    2012,
    2014,
    2016,
    2018,
    2020,
    2022,
)


def parse_bool_series(
    series: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def median_absolute_deviation(
    values: np.ndarray,
) -> float:
    if len(values) == 0:
        return np.nan

    median = float(np.median(values))

    return float(
        np.median(
            np.abs(values - median)
        )
    )


def longest_streak(
    winners: list[str],
    party: str,
) -> int:
    longest = 0
    current = 0

    for winner in winners:
        if winner == party:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def current_streak(
    winners: list[str],
) -> tuple[str | None, int]:
    if not winners:
        return None, 0

    party = winners[-1]
    length = 0

    for winner in reversed(winners):
        if winner == party:
            length += 1
        else:
            break

    return party, length


def linear_trend_metrics(
    cycles: np.ndarray,
    margins: np.ndarray,
) -> tuple[float, float, float]:
    if len(margins) < 2:
        return np.nan, np.nan, np.nan

    centered_cycles = (
        cycles.astype(float)
        - float(np.mean(cycles))
    ) / 2.0

    design = np.column_stack(
        [
            np.ones(len(centered_cycles)),
            centered_cycles,
        ]
    )

    coefficients, *_ = np.linalg.lstsq(
        design,
        margins,
        rcond=None,
    )

    intercept = float(coefficients[0])
    slope_per_cycle = float(coefficients[1])

    predictions = design @ coefficients

    residual_rmse = float(
        np.sqrt(
            np.mean(
                np.square(
                    margins - predictions
                )
            )
        )
    )

    return (
        intercept,
        slope_per_cycle,
        residual_rmse,
    )


def summarize_history(
    group: pd.DataFrame,
    cycles: tuple[int, ...],
    prefix: str,
) -> dict[str, object]:
    subset = group.loc[
        group["cycle"].isin(cycles)
        & group[
            "include_in_major_party_margin_scoring"
        ]
    ].sort_values("cycle")

    margins = (
        subset["actual_dem_margin"]
        .to_numpy(dtype=float)
    )

    observed_cycles = (
        subset["cycle"]
        .to_numpy(dtype=int)
    )

    winners = (
        subset["actual_winner"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    valid_winners = [
        winner
        for winner in winners
        if winner in {"D", "R"}
    ]

    result: dict[str, object] = {
        f"{prefix}_scorable_elections": len(subset),
        f"{prefix}_first_scorable_cycle": (
            int(observed_cycles[0])
            if len(observed_cycles)
            else np.nan
        ),
        f"{prefix}_last_scorable_cycle": (
            int(observed_cycles[-1])
            if len(observed_cycles)
            else np.nan
        ),
        f"{prefix}_mean_dem_margin": (
            float(np.mean(margins))
            if len(margins)
            else np.nan
        ),
        f"{prefix}_median_dem_margin": (
            float(np.median(margins))
            if len(margins)
            else np.nan
        ),
        f"{prefix}_margin_std": (
            float(np.std(margins, ddof=0))
            if len(margins)
            else np.nan
        ),
        f"{prefix}_margin_mad": (
            median_absolute_deviation(margins)
        ),
        f"{prefix}_mean_absolute_margin": (
            float(np.mean(np.abs(margins)))
            if len(margins)
            else np.nan
        ),
        f"{prefix}_minimum_dem_margin": (
            float(np.min(margins))
            if len(margins)
            else np.nan
        ),
        f"{prefix}_maximum_dem_margin": (
            float(np.max(margins))
            if len(margins)
            else np.nan
        ),
        f"{prefix}_margin_range": (
            float(np.max(margins) - np.min(margins))
            if len(margins)
            else np.nan
        ),
        f"{prefix}_democratic_wins": valid_winners.count("D"),
        f"{prefix}_republican_wins": valid_winners.count("R"),
        f"{prefix}_democratic_win_rate": (
            valid_winners.count("D") / len(valid_winners)
            if valid_winners
            else np.nan
        ),
        f"{prefix}_competitive_within_2_count": (
            int(np.sum(np.abs(margins) <= 2.0))
            if len(margins)
            else 0
        ),
        f"{prefix}_competitive_within_5_count": (
            int(np.sum(np.abs(margins) <= 5.0))
            if len(margins)
            else 0
        ),
        f"{prefix}_competitive_within_10_count": (
            int(np.sum(np.abs(margins) <= 10.0))
            if len(margins)
            else 0
        ),
        f"{prefix}_competitive_within_15_count": (
            int(np.sum(np.abs(margins) <= 15.0))
            if len(margins)
            else 0
        ),
        f"{prefix}_competitive_within_5_rate": (
            float(np.mean(np.abs(margins) <= 5.0))
            if len(margins)
            else np.nan
        ),
        f"{prefix}_competitive_within_10_rate": (
            float(np.mean(np.abs(margins) <= 10.0))
            if len(margins)
            else np.nan
        ),
        f"{prefix}_competitive_within_15_rate": (
            float(np.mean(np.abs(margins) <= 15.0))
            if len(margins)
            else np.nan
        ),
    }

    if len(valid_winners) >= 2:
        party_switches = sum(
            previous != current
            for previous, current in zip(
                valid_winners[:-1],
                valid_winners[1:],
            )
        )
    else:
        party_switches = 0

    result[
        f"{prefix}_party_switch_count"
    ] = party_switches

    result[
        f"{prefix}_longest_democratic_streak"
    ] = longest_streak(
        valid_winners,
        "D",
    )

    result[
        f"{prefix}_longest_republican_streak"
    ] = longest_streak(
        valid_winners,
        "R",
    )

    streak_party, streak_length = current_streak(
        valid_winners
    )

    result[
        f"{prefix}_current_party_streak_party"
    ] = streak_party

    result[
        f"{prefix}_current_party_streak_length"
    ] = streak_length

    if len(margins) >= 2:
        swings = np.diff(margins)

        result.update(
            {
                f"{prefix}_consecutive_swing_count": len(swings),
                f"{prefix}_mean_swing_dem": float(
                    np.mean(swings)
                ),
                f"{prefix}_mean_absolute_swing": float(
                    np.mean(np.abs(swings))
                ),
                f"{prefix}_swing_std": float(
                    np.std(swings, ddof=0)
                ),
                f"{prefix}_largest_absolute_swing": float(
                    np.max(np.abs(swings))
                ),
                f"{prefix}_first_to_last_margin_change_dem": float(
                    margins[-1] - margins[0]
                ),
            }
        )
    else:
        result.update(
            {
                f"{prefix}_consecutive_swing_count": 0,
                f"{prefix}_mean_swing_dem": np.nan,
                f"{prefix}_mean_absolute_swing": np.nan,
                f"{prefix}_swing_std": np.nan,
                f"{prefix}_largest_absolute_swing": np.nan,
                f"{prefix}_first_to_last_margin_change_dem": np.nan,
            }
        )

    (
        trend_intercept,
        trend_slope,
        trend_residual_rmse,
    ) = linear_trend_metrics(
        observed_cycles,
        margins,
    )

    result[
        f"{prefix}_trend_intercept_dem"
    ] = trend_intercept

    result[
        f"{prefix}_trend_slope_points_per_election"
    ] = trend_slope

    result[
        f"{prefix}_trend_residual_rmse"
    ] = trend_residual_rmse

    return result


def build_behavior_dataset(
    results: pd.DataFrame,
    profile: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    required_results = {
        "cycle",
        "race_id",
        "actual_dem_margin",
        "actual_winner",
        "include_in_major_party_margin_scoring",
        "major_party_contested",
        "general_election_party_structure",
    }

    missing_results = sorted(
        required_results - set(results.columns)
    )

    if missing_results:
        raise ValueError(
            "Historical results warehouse is missing columns: "
            + ", ".join(missing_results)
        )

    required_profile = {
        "race_id",
        "state",
        "district",
    }

    missing_profile = sorted(
        required_profile - set(profile.columns)
    )

    if missing_profile:
        raise ValueError(
            "Structural profile is missing columns: "
            + ", ".join(missing_profile)
        )

    if len(results) != 2610:
        raise ValueError(
            f"Expected 2,610 historical rows; found {len(results)}."
        )

    if len(profile) != 435:
        raise ValueError(
            f"Expected 435 profile rows; found {len(profile)}."
        )

    if results.duplicated(
        ["cycle", "race_id"]
    ).any():
        raise ValueError(
            "Duplicate cycle/race records found."
        )

    if profile["race_id"].duplicated().any():
        raise ValueError(
            "Duplicate structural-profile race IDs found."
        )

    work = results.copy()

    work["cycle"] = pd.to_numeric(
        work["cycle"],
        errors="raise",
    ).astype(int)

    work["actual_dem_margin"] = pd.to_numeric(
        work["actual_dem_margin"],
        errors="coerce",
    )

    work[
        "include_in_major_party_margin_scoring"
    ] = parse_bool_series(
        work[
            "include_in_major_party_margin_scoring"
        ]
    )

    rows: list[dict[str, object]] = []

    grouped = {
        race_id: group.copy()
        for race_id, group in work.groupby(
            "race_id",
            sort=False,
        )
    }

    for _, profile_row in profile.iterrows():
        race_id = str(
            profile_row["race_id"]
        )

        group = grouped.get(
            race_id,
            pd.DataFrame(
                columns=work.columns
            ),
        )

        row: dict[str, object] = {
            "race_id": race_id,
            "state": profile_row["state"],
            "district": profile_row["district"],
            "historical_label_appearance_count": int(
                group["cycle"].nunique()
            ),
        }

        row.update(
            summarize_history(
                group=group,
                cycles=PRE_REDISTRICTING_CYCLES,
                prefix="pre2022",
            )
        )

        row.update(
            summarize_history(
                group=group,
                cycles=ALL_CYCLES,
                prefix="full_label_history",
            )
        )

        row_2020 = group.loc[
            group["cycle"].eq(2020)
        ]

        row_2022 = group.loc[
            group["cycle"].eq(2022)
        ]

        for cycle, cycle_row in [
            (2020, row_2020),
            (2022, row_2022),
        ]:
            if cycle_row.empty:
                row[
                    f"house_{cycle}_actual_dem_margin"
                ] = np.nan
                row[
                    f"house_{cycle}_actual_winner"
                ] = None
                row[
                    f"house_{cycle}_scorable"
                ] = False
            else:
                selected = cycle_row.iloc[0]

                row[
                    f"house_{cycle}_actual_dem_margin"
                ] = selected[
                    "actual_dem_margin"
                ]

                row[
                    f"house_{cycle}_actual_winner"
                ] = selected[
                    "actual_winner"
                ]

                row[
                    f"house_{cycle}_scorable"
                ] = bool(
                    selected[
                        "include_in_major_party_margin_scoring"
                    ]
                )

        if (
            pd.notna(
                row[
                    "house_2020_actual_dem_margin"
                ]
            )
            and pd.notna(
                row[
                    "house_2022_actual_dem_margin"
                ]
            )
        ):
            row[
                "house_2020_to_2022_label_swing_dem"
            ] = (
                float(
                    row[
                        "house_2022_actual_dem_margin"
                    ]
                )
                - float(
                    row[
                        "house_2020_actual_dem_margin"
                    ]
                )
            )
        else:
            row[
                "house_2020_to_2022_label_swing_dem"
            ] = np.nan

        row["behavior_boundary_era"] = (
            "2012-2020 primary metrics"
        )

        row["full_label_history_continuity_status"] = (
            "approximate_label_match_across_2022_redistricting"
        )

        row["behavior_profile_version"] = "1.0"

        row["behavior_profile_notes"] = (
            "Primary pre2022 metrics use scorable 2012-2020 House "
            "elections. Full-label-history metrics include 2022 but "
            "do not guarantee geographic continuity after redistricting."
        )

        rows.append(row)

    behavior = pd.DataFrame(rows)

    behavior = behavior.sort_values(
        ["state", "district"],
        key=lambda series: (
            pd.to_numeric(
                series,
                errors="coerce",
            )
            if series.name == "district"
            else series
        ),
    ).reset_index(drop=True)

    duplicate_race_ids = int(
        behavior["race_id"].duplicated().sum()
    )

    missing_pre2022_history = int(
        behavior[
            "pre2022_scorable_elections"
        ].eq(0).sum()
    )

    complete_pre2022_histories = int(
        behavior[
            "pre2022_scorable_elections"
        ].eq(5).sum()
    )

    complete_label_appearances = int(
        behavior[
            "historical_label_appearance_count"
        ].eq(6).sum()
    )

    failures: list[str] = []

    if len(behavior) != 435:
        failures.append(
            f"Expected 435 behavior rows; found {len(behavior)}."
        )

    if behavior["race_id"].nunique() != 435:
        failures.append(
            "Expected 435 unique race IDs."
        )

    if duplicate_race_ids:
        failures.append(
            f"Found {duplicate_race_ids} duplicate race IDs."
        )

    if behavior[
        "pre2022_scorable_elections"
    ].gt(5).any():
        failures.append(
            "Found a district with more than five pre-2022 "
            "scorable elections."
        )

    if behavior[
        "full_label_history_scorable_elections"
    ].gt(6).any():
        failures.append(
            "Found a district with more than six full-history "
            "scorable elections."
        )

    report_lines = [
        "House District Behavior Validation",
        "=" * 34,
        "",
        f"Rows: {len(behavior)}",
        f"Unique race IDs: {behavior['race_id'].nunique()}",
        f"Duplicate race IDs: {duplicate_race_ids}",
        (
            "District labels appearing in all six cycles: "
            f"{complete_label_appearances}"
        ),
        (
            "Districts with all five pre-2022 scorable elections: "
            f"{complete_pre2022_histories}"
        ),
        (
            "Districts with no pre-2022 scorable history: "
            f"{missing_pre2022_history}"
        ),
        "",
        "Pre-2022 scorable-election counts:",
        behavior[
            "pre2022_scorable_elections"
        ]
        .value_counts()
        .sort_index()
        .to_string(),
        "",
        "Full-label-history scorable-election counts:",
        behavior[
            "full_label_history_scorable_elections"
        ]
        .value_counts()
        .sort_index()
        .to_string(),
        "",
        "Pre-2022 behavior summary:",
        behavior[
            [
                "pre2022_mean_dem_margin",
                "pre2022_margin_std",
                "pre2022_mean_absolute_margin",
                "pre2022_mean_absolute_swing",
                "pre2022_party_switch_count",
                "pre2022_competitive_within_10_rate",
                "pre2022_trend_slope_points_per_election",
                "pre2022_trend_residual_rmse",
            ]
        ]
        .describe()
        .transpose()
        .to_string(
            float_format=lambda value: f"{value:.4f}"
        ),
        "",
        "Boundary treatment:",
        (
            "Primary behavior metrics stop at 2020. The 2022 result "
            "is retained separately, and full-label-history metrics "
            "are explicitly marked as approximate across redistricting."
        ),
        "",
        "Validation status:",
    ]

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

    return behavior, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a reusable historical House district behavior "
            "warehouse from 2012-2022 election results."
        )
    )

    parser.add_argument(
        "--results-path",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
    )

    parser.add_argument(
        "--profile-path",
        type=Path,
        default=DEFAULT_PROFILE_PATH,
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )

    parser.add_argument(
        "--validation-path",
        type=Path,
        default=DEFAULT_VALIDATION_PATH,
    )

    args = parser.parse_args()

    if not args.results_path.exists():
        raise FileNotFoundError(
            f"Missing historical results: {args.results_path}"
        )

    if not args.profile_path.exists():
        raise FileNotFoundError(
            f"Missing structural profile: {args.profile_path}"
        )

    results = pd.read_csv(
        args.results_path,
        dtype={
            "race_id": str,
            "district": str,
        },
    )

    profile = pd.read_csv(
        args.profile_path,
        dtype={
            "race_id": str,
            "state": str,
            "district": str,
        },
    )

    behavior, report = build_behavior_dataset(
        results=results,
        profile=profile,
    )

    args.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.validation_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    behavior.to_csv(
        args.output_path,
        index=False,
    )

    args.validation_path.write_text(
        report
    )

    print(report)
    print()
    print(f"Wrote: {args.output_path}")
    print(f"Wrote: {args.validation_path}")


if __name__ == "__main__":
    main()
