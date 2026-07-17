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

DEFAULT_REGISTRY_PATH = (
    PROJECT_ROOT
    / "historical/house/config/"
    "house_boundary_regimes_2012_2020.csv"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "historical/house/warehouse/"
    "house_district_behavior.csv"
)

DEFAULT_REGIME_DETAIL_PATH = (
    PROJECT_ROOT
    / "historical/house/warehouse/"
    "house_district_behavior_regime_detail.csv"
)

DEFAULT_VALIDATION_PATH = (
    PROJECT_ROOT
    / "historical/house/warehouse/"
    "house_district_behavior_validation.txt"
)


PRE2022_CYCLES = (2012, 2014, 2016, 2018, 2020)


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def median_absolute_deviation(values: np.ndarray) -> float:
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
        float(coefficients[0]),
        float(coefficients[1]),
        residual_rmse,
    )


def assign_boundary_regimes(
    results: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    work = results.copy()

    work["boundary_regime_id"] = (
        work["state"]
        + "_2012_2020_default"
    )

    work["boundary_regime_status"] = (
        "assumed_stable_2012_2020"
    )

    work["boundary_regime_notes"] = (
        "No known mid-decade statewide boundary break configured."
    )

    for row in registry.itertuples(index=False):
        mask = (
            work["state"].eq(row.state)
            & work["cycle"].between(
                int(row.start_cycle),
                int(row.end_cycle),
            )
        )

        work.loc[
            mask,
            "boundary_regime_id",
        ] = row.regime_id

        work.loc[
            mask,
            "boundary_regime_status",
        ] = row.boundary_continuity_status

        work.loc[
            mask,
            "boundary_regime_notes",
        ] = row.notes

    return work


def summarize_subset(
    subset: pd.DataFrame,
    prefix: str,
) -> dict[str, object]:
    subset = subset.sort_values("cycle")

    margins = (
        subset["actual_dem_margin"]
        .to_numpy(dtype=float)
    )

    cycles = (
        subset["cycle"]
        .to_numpy(dtype=int)
    )

    winners = [
        winner
        for winner in (
            subset["actual_winner"]
            .fillna("")
            .astype(str)
            .tolist()
        )
        if winner in {"D", "R"}
    ]

    result: dict[str, object] = {
        f"{prefix}_scorable_elections": len(subset),
        f"{prefix}_first_cycle": (
            int(cycles[0])
            if len(cycles)
            else np.nan
        ),
        f"{prefix}_last_cycle": (
            int(cycles[-1])
            if len(cycles)
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
        f"{prefix}_margin_range": (
            float(np.max(margins) - np.min(margins))
            if len(margins)
            else np.nan
        ),
        f"{prefix}_democratic_wins": winners.count("D"),
        f"{prefix}_republican_wins": winners.count("R"),
        f"{prefix}_democratic_win_rate": (
            winners.count("D") / len(winners)
            if winners
            else np.nan
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

    result[f"{prefix}_party_switch_count"] = (
        sum(
            previous != current
            for previous, current in zip(
                winners[:-1],
                winners[1:],
            )
        )
        if len(winners) >= 2
        else 0
    )

    result[
        f"{prefix}_longest_democratic_streak"
    ] = longest_streak(
        winners,
        "D",
    )

    result[
        f"{prefix}_longest_republican_streak"
    ] = longest_streak(
        winners,
        "R",
    )

    streak_party, streak_length = current_streak(
        winners
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
        cycles,
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


def select_latest_usable_regime(
    district_regimes: pd.DataFrame,
) -> pd.Series:
    usable = district_regimes.loc[
        district_regimes[
            "regime_scorable_elections"
        ].ge(2)
    ].copy()

    if usable.empty:
        return district_regimes.sort_values(
            [
                "regime_last_cycle",
                "regime_scorable_elections",
            ],
            ascending=[
                False,
                False,
            ],
        ).iloc[0]

    return usable.sort_values(
        [
            "regime_last_cycle",
            "regime_scorable_elections",
            "regime_first_cycle",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).iloc[0]


def reliability_label(
    scorable_elections: int,
) -> str:
    if scorable_elections >= 4:
        return "high"
    if scorable_elections == 3:
        return "medium"
    if scorable_elections == 2:
        return "low"
    return "insufficient_single_cycle"


def build_behavior_dataset(
    results: pd.DataFrame,
    profile: pd.DataFrame,
    registry: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    required_results = {
        "cycle",
        "race_id",
        "state",
        "actual_dem_margin",
        "actual_winner",
        "include_in_major_party_margin_scoring",
    }

    missing_results = sorted(
        required_results - set(results.columns)
    )

    if missing_results:
        raise ValueError(
            "Historical results warehouse is missing columns: "
            + ", ".join(missing_results)
        )

    required_registry = {
        "state",
        "regime_id",
        "start_cycle",
        "end_cycle",
        "boundary_continuity_status",
        "notes",
    }

    missing_registry = sorted(
        required_registry - set(registry.columns)
    )

    if missing_registry:
        raise ValueError(
            "Boundary registry is missing columns: "
            + ", ".join(missing_registry)
        )

    if len(results) != 2610:
        raise ValueError(
            f"Expected 2,610 historical rows; found {len(results)}."
        )

    if len(profile) != 435:
        raise ValueError(
            f"Expected 435 structural-profile rows; found {len(profile)}."
        )

    if results.duplicated(
        ["cycle", "race_id"]
    ).any():
        raise ValueError(
            "Duplicate cycle/race records found."
        )

    work = results.loc[
        results["cycle"].isin(PRE2022_CYCLES)
    ].copy()

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

    work = assign_boundary_regimes(
        results=work,
        registry=registry,
    )

    scorable = work.loc[
        work[
            "include_in_major_party_margin_scoring"
        ]
    ].copy()

    regime_rows: list[dict[str, object]] = []

    for (
        race_id,
        regime_id,
    ), subset in scorable.groupby(
        [
            "race_id",
            "boundary_regime_id",
        ],
        sort=False,
    ):
        subset = subset.sort_values("cycle")

        base = {
            "race_id": race_id,
            "state": subset["state"].iloc[0],
            "boundary_regime_id": regime_id,
            "boundary_regime_status": (
                subset[
                    "boundary_regime_status"
                ].iloc[0]
            ),
            "boundary_regime_notes": (
                subset[
                    "boundary_regime_notes"
                ].iloc[0]
            ),
        }

        metrics = summarize_subset(
            subset=subset,
            prefix="regime",
        )

        base.update(metrics)

        base["regime_reliability"] = reliability_label(
            int(
                base[
                    "regime_scorable_elections"
                ]
            )
        )

        regime_rows.append(base)

    regime_detail = pd.DataFrame(
        regime_rows
    )

    behavior_rows: list[dict[str, object]] = []

    for profile_row in profile.itertuples(
        index=False
    ):
        race_id = str(profile_row.race_id)

        district_regimes = regime_detail.loc[
            regime_detail["race_id"].eq(race_id)
        ].copy()

        row: dict[str, object] = {
            "race_id": race_id,
            "state": profile_row.state,
            "district": profile_row.district,
            "boundary_regime_count": len(
                district_regimes
            ),
        }

        if district_regimes.empty:
            row.update(
                {
                    "selected_boundary_regime_id": None,
                    "selected_boundary_regime_status": None,
                    "selected_boundary_regime_notes": None,
                    "selected_regime_reliability": "no_scorable_history",
                }
            )

            empty = summarize_subset(
                subset=scorable.iloc[0:0],
                prefix="selected_regime",
            )

            row.update(empty)
        else:
            selected = select_latest_usable_regime(
                district_regimes
            )

            row.update(
                {
                    "selected_boundary_regime_id": (
                        selected[
                            "boundary_regime_id"
                        ]
                    ),
                    "selected_boundary_regime_status": (
                        selected[
                            "boundary_regime_status"
                        ]
                    ),
                    "selected_boundary_regime_notes": (
                        selected[
                            "boundary_regime_notes"
                        ]
                    ),
                    "selected_regime_reliability": (
                        selected[
                            "regime_reliability"
                        ]
                    ),
                }
            )

            selected_metrics = {
                key.replace(
                    "regime_",
                    "selected_regime_",
                    1,
                ): value
                for key, value in selected.items()
                if key.startswith("regime_")
                and key != "regime_reliability"
            }

            row.update(
                selected_metrics
            )

        row[
            "behavior_boundary_method"
        ] = (
            "latest stable regime with at least two scorable elections; "
            "otherwise latest available single-cycle regime"
        )

        row[
            "behavior_profile_version"
        ] = "2.0"

        row[
            "behavior_profile_notes"
        ] = (
            "Swing, trend, volatility, and party-switch metrics are "
            "calculated only within the selected boundary regime."
        )

        behavior_rows.append(row)

    behavior = pd.DataFrame(
        behavior_rows
    )

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

    reliability_counts = (
        behavior[
            "selected_regime_reliability"
        ]
        .value_counts(dropna=False)
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

    if (
        behavior[
            "selected_regime_scorable_elections"
        ]
        .dropna()
        .gt(5)
        .any()
    ):
        failures.append(
            "Selected regime has more than five scorable elections."
        )

    report_lines = [
        "House District Behavior Validation",
        "=" * 34,
        "",
        f"Rows: {len(behavior)}",
        f"Unique race IDs: {behavior['race_id'].nunique()}",
        f"Duplicate race IDs: {duplicate_race_ids}",
        f"District-regime detail rows: {len(regime_detail)}",
        "",
        "Selected-regime reliability:",
        reliability_counts.to_string(),
        "",
        "Selected-regime scorable-election counts:",
        behavior[
            "selected_regime_scorable_elections"
        ]
        .value_counts(dropna=False)
        .sort_index()
        .to_string(),
        "",
        "Selected-regime behavior summary:",
        behavior[
            [
                "selected_regime_mean_dem_margin",
                "selected_regime_margin_std",
                "selected_regime_mean_absolute_margin",
                "selected_regime_mean_absolute_swing",
                "selected_regime_party_switch_count",
                "selected_regime_competitive_within_10_rate",
                "selected_regime_trend_slope_points_per_election",
                "selected_regime_trend_residual_rmse",
            ]
        ]
        .describe()
        .transpose()
        .to_string(
            float_format=lambda value: f"{value:.4f}"
        ),
        "",
        "Configured boundary-break states:",
        ", ".join(
            sorted(
                registry["state"].unique()
            )
        ),
        "",
        "Boundary treatment:",
        (
            "No behavior statistic crosses a configured boundary-regime "
            "break. Latest regimes with at least two scorable elections "
            "are preferred; single-cycle regimes are retained but marked "
            "insufficient for volatility and trend estimation."
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

    report = "\n".join(
        report_lines
    )

    if failures:
        raise RuntimeError(
            report
        )

    return behavior, regime_detail, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build boundary-regime-aware historical House district "
            "behavior metrics."
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
        "--registry-path",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )

    parser.add_argument(
        "--regime-detail-path",
        type=Path,
        default=DEFAULT_REGIME_DETAIL_PATH,
    )

    parser.add_argument(
        "--validation-path",
        type=Path,
        default=DEFAULT_VALIDATION_PATH,
    )

    args = parser.parse_args()

    results = pd.read_csv(
        args.results_path,
        dtype={
            "race_id": str,
            "state": str,
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

    registry = pd.read_csv(
        args.registry_path,
        dtype={
            "state": str,
            "regime_id": str,
        },
    )

    behavior, regime_detail, report = (
        build_behavior_dataset(
            results=results,
            profile=profile,
            registry=registry,
        )
    )

    args.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    behavior.to_csv(
        args.output_path,
        index=False,
    )

    regime_detail.to_csv(
        args.regime_detail_path,
        index=False,
    )

    args.validation_path.write_text(
        report
    )

    print(report)
    print()
    print(f"Wrote: {args.output_path}")
    print(f"Wrote: {args.regime_detail_path}")
    print(f"Wrote: {args.validation_path}")


if __name__ == "__main__":
    main()
