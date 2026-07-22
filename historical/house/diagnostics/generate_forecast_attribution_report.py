#!/usr/bin/env python3
"""Compare two completed House forecasts without rerunning the model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_OLD_RACES = Path(
    "diagnostics/production_replay_20260719/house_race_stats_before.csv"
)
DEFAULT_NEW_RACES = Path("outputs/house_race_stats.csv")

DEFAULT_OLD_SUMMARY = Path(
    "diagnostics/production_replay_20260719/house_forecast_summary_before.csv"
)
DEFAULT_NEW_SUMMARY = Path("outputs/house_forecast_summary.csv")

DEFAULT_OUTPUT_DIR = Path("outputs/house_forecast_attribution")


IDENTITY_COLUMNS = [
    "district_id",
    "state",
    "district",
    "dem_candidate",
    "gop_candidate",
]

# These are stored model quantities. Some are nested rather than perfectly
# additive, so the report distinguishes direct component changes from the
# exact final-margin change.
COMPONENT_COLUMNS = [
    "district_partisan_baseline_dem",
    "district_elasticity",
    "house_national_environment_used_dem",
    "district_environment_adjustment_dem",
    "state_environment_adjustment_dem",
    "incumbency_adjustment_dem",
    "candidate_quality_adjustment_dem_before_war",
    "candidate_war_adjustment_dem",
    "candidate_quality_adjustment_dem",
    "special_adjustment_dem",
    "poll_spillover_adjustment_dem",
    "fundamentals_margin_dem_before_poll_spillover",
    "fundamentals_margin_dem",
    "polling_margin_dem",
    "bayesian_polling_weight",
    "bayesian_model_margin_dem",
    "model_margin_dem",
    "pre_sim_dem_win_probability",
    "simulated_dem_win_probability",
    "dem_win_probability",
    "avg_simulated_margin_dem",
    "district_posterior_sd",
]

# Prefer primitive or conceptually distinct quantities when assigning a
# district's primary cause. Do not include nested totals such as fundamentals
# or final model margin in this ranking.
PRIMARY_CAUSE_COLUMNS = [
    "district_partisan_baseline_dem",
    "district_elasticity",
    "house_national_environment_used_dem",
    "district_environment_adjustment_dem",
    "state_environment_adjustment_dem",
    "incumbency_adjustment_dem",
    "candidate_quality_adjustment_dem_before_war",
    "candidate_war_adjustment_dem",
    "special_adjustment_dem",
    "poll_spillover_adjustment_dem",
    "polling_margin_dem",
    "bayesian_polling_weight",
    "district_posterior_sd",
]

DISPLAY_NAMES = {
    "district_partisan_baseline_dem": "Partisan baseline",
    "district_elasticity": "District elasticity",
    "house_national_environment_used_dem": "National environment",
    "district_environment_adjustment_dem": "District environment adjustment",
    "state_environment_adjustment_dem": "State adjustment",
    "incumbency_adjustment_dem": "Incumbency",
    "candidate_quality_adjustment_dem_before_war": "Candidate quality before WAR",
    "candidate_war_adjustment_dem": "Candidate WAR",
    "candidate_quality_adjustment_dem": "Total candidate quality",
    "special_adjustment_dem": "Special adjustment",
    "poll_spillover_adjustment_dem": "Poll spillover",
    "fundamentals_margin_dem_before_poll_spillover": "Fundamentals before spillover",
    "fundamentals_margin_dem": "Fundamentals margin",
    "polling_margin_dem": "Polling margin",
    "bayesian_polling_weight": "Polling weight",
    "bayesian_model_margin_dem": "Bayesian model margin",
    "model_margin_dem": "Model margin",
    "pre_sim_dem_win_probability": "Pre-simulation probability",
    "simulated_dem_win_probability": "Simulated probability",
    "dem_win_probability": "Democratic win probability",
    "avg_simulated_margin_dem": "Average simulated margin",
    "district_posterior_sd": "District posterior SD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two completed House forecasts."
    )
    parser.add_argument("--old-races", type=Path, default=DEFAULT_OLD_RACES)
    parser.add_argument("--new-races", type=Path, default=DEFAULT_NEW_RACES)
    parser.add_argument("--old-summary", type=Path, default=DEFAULT_OLD_SUMMARY)
    parser.add_argument("--new-summary", type=Path, default=DEFAULT_NEW_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--surprise-threshold",
        type=float,
        default=2.0,
        help="Flag absolute final-margin changes at or above this threshold.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def normalize_district_id(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", "", regex=True)
    )


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def first_existing(columns: Iterable[str], available: set[str]) -> str | None:
    for column in columns:
        if column in available:
            return column
    return None


def read_races(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)

    if "district_id" not in frame.columns:
        raise ValueError(f"{label} is missing required column district_id")

    frame = frame.copy()
    frame["district_id"] = normalize_district_id(frame["district_id"])

    duplicated = frame["district_id"].duplicated(keep=False)
    if duplicated.any():
        examples = sorted(frame.loc[duplicated, "district_id"].dropna().unique())
        raise ValueError(
            f"{label} contains duplicate district_id values: {examples[:20]}"
        )

    return frame


def read_summary(path: Path) -> pd.Series | None:
    if not path.exists():
        return None

    frame = pd.read_csv(path)
    if frame.empty:
        return None

    return frame.iloc[0]


def select_identity(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    merged = pd.DataFrame({"district_id": sorted(set(old["district_id"]) | set(new["district_id"]))})

    old_indexed = old.set_index("district_id")
    new_indexed = new.set_index("district_id")

    for column in IDENTITY_COLUMNS[1:]:
        new_values = (
            new_indexed[column]
            if column in new_indexed.columns
            else pd.Series(index=new_indexed.index, dtype="object")
        )
        old_values = (
            old_indexed[column]
            if column in old_indexed.columns
            else pd.Series(index=old_indexed.index, dtype="object")
        )

        merged[column] = (
            merged["district_id"].map(new_values)
            .combine_first(merged["district_id"].map(old_values))
        )

    return merged


def build_comparison(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    comparison = select_identity(old, new)

    old_indexed = old.set_index("district_id")
    new_indexed = new.set_index("district_id")

    available_components = [
        column
        for column in COMPONENT_COLUMNS
        if column in old.columns or column in new.columns
    ]

    for column in available_components:
        old_values = (
            numeric(old_indexed[column])
            if column in old_indexed.columns
            else pd.Series(index=old_indexed.index, dtype=float)
        )
        new_values = (
            numeric(new_indexed[column])
            if column in new_indexed.columns
            else pd.Series(index=new_indexed.index, dtype=float)
        )

        comparison[f"old_{column}"] = comparison["district_id"].map(old_values)
        comparison[f"new_{column}"] = comparison["district_id"].map(new_values)
        comparison[f"delta_{column}"] = (
            comparison[f"new_{column}"] - comparison[f"old_{column}"]
        )

    final_margin_column = first_existing(
        ["model_margin_dem", "bayesian_model_margin_dem", "fundamentals_margin_dem"],
        set(available_components),
    )

    probability_column = first_existing(
        [
            "simulated_dem_win_probability",
            "dem_win_probability",
            "pre_sim_dem_win_probability",
        ],
        set(available_components),
    )

    if final_margin_column is None:
        raise ValueError(
            "Neither forecast contains a usable final margin column."
        )

    comparison["old_margin_dem"] = comparison[f"old_{final_margin_column}"]
    comparison["new_margin_dem"] = comparison[f"new_{final_margin_column}"]
    comparison["delta_margin_dem"] = comparison[f"delta_{final_margin_column}"]
    comparison["abs_delta_margin"] = comparison["delta_margin_dem"].abs()

    if probability_column is not None:
        comparison["old_dem_probability"] = comparison[
            f"old_{probability_column}"
        ]
        comparison["new_dem_probability"] = comparison[
            f"new_{probability_column}"
        ]
        comparison["delta_dem_probability"] = comparison[
            f"delta_{probability_column}"
        ]
        comparison["abs_delta_probability"] = comparison[
            "delta_dem_probability"
        ].abs()

    # Exact polling/blending effect relative to the stored fundamentals margin.
    needed = {
        "old_model_margin_dem",
        "new_model_margin_dem",
        "old_fundamentals_margin_dem",
        "new_fundamentals_margin_dem",
    }
    if needed.issubset(comparison.columns):
        comparison["old_polling_blend_effect_dem"] = (
            comparison["old_model_margin_dem"]
            - comparison["old_fundamentals_margin_dem"]
        )
        comparison["new_polling_blend_effect_dem"] = (
            comparison["new_model_margin_dem"]
            - comparison["new_fundamentals_margin_dem"]
        )
        comparison["delta_polling_blend_effect_dem"] = (
            comparison["new_polling_blend_effect_dem"]
            - comparison["old_polling_blend_effect_dem"]
        )

    primary_candidates = [
        column
        for column in PRIMARY_CAUSE_COLUMNS
        if f"delta_{column}" in comparison.columns
    ]

    if "delta_polling_blend_effect_dem" in comparison.columns:
        primary_candidates.append("polling_blend_effect_dem")

    def primary_cause(row: pd.Series) -> str:
        values: list[tuple[str, float]] = []

        for column in primary_candidates:
            delta_column = f"delta_{column}"
            value = row.get(delta_column)

            if pd.notna(value):
                values.append((column, float(value)))

        if not values:
            return "Unclassified"

        winner, amount = max(values, key=lambda item: abs(item[1]))

        if abs(amount) < 1e-9:
            return "No component change"

        return DISPLAY_NAMES.get(winner, winner.replace("_", " ").title())

    comparison["primary_cause"] = comparison.apply(primary_cause, axis=1)

    old_present = set(old["district_id"])
    new_present = set(new["district_id"])

    comparison["snapshot_status"] = np.select(
        [
            comparison["district_id"].isin(old_present)
            & comparison["district_id"].isin(new_present),
            comparison["district_id"].isin(new_present),
        ],
        ["matched", "new_only"],
        default="old_only",
    )

    return comparison.sort_values(
        ["abs_delta_margin", "district_id"],
        ascending=[False, True],
        na_position="last",
    )


def build_component_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []

    delta_columns = [
        column
        for column in comparison.columns
        if column.startswith("delta_")
        and column
        not in {
            "delta_margin_dem",
            "delta_dem_probability",
        }
    ]

    for delta_column in delta_columns:
        component = delta_column.removeprefix("delta_")
        values = numeric(comparison[delta_column]).dropna()

        if values.empty:
            continue

        rows.append(
            {
                "component": component,
                "display_name": DISPLAY_NAMES.get(
                    component,
                    component.replace("_", " ").title(),
                ),
                "districts_compared": int(values.shape[0]),
                "signed_total_change": float(values.sum()),
                "mean_change": float(values.mean()),
                "median_change": float(values.median()),
                "mean_absolute_change": float(values.abs().mean()),
                "maximum_absolute_change": float(values.abs().max()),
                "districts_changed_0_1_plus": int((values.abs() >= 0.1).sum()),
                "districts_changed_0_5_plus": int((values.abs() >= 0.5).sum()),
                "districts_changed_1_0_plus": int((values.abs() >= 1.0).sum()),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    return summary.sort_values(
        ["mean_absolute_change", "maximum_absolute_change"],
        ascending=False,
    )


def summary_value(series: pd.Series | None, column: str) -> float | None:
    if series is None or column not in series.index:
        return None

    value = pd.to_numeric(pd.Series([series[column]]), errors="coerce").iloc[0]
    return None if pd.isna(value) else float(value)


def fmt_number(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def fmt_probability(value: float | None) -> str:
    if value is None:
        return "N/A"

    # Stored probabilities normally use 0–1. Protect against percentage-form data.
    normalized = value / 100.0 if value > 1.0 else value
    return f"{normalized:.1%}"


def signed(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:+.{digits}f}"


def build_text_report(
    comparison: pd.DataFrame,
    component_summary: pd.DataFrame,
    old_summary: pd.Series | None,
    new_summary: pd.Series | None,
    old_races_path: Path,
    new_races_path: Path,
    threshold: float,
) -> str:
    lines: list[str] = []

    old_expected = summary_value(old_summary, "expected_dem_seats")
    new_expected = summary_value(new_summary, "expected_dem_seats")
    old_majority = summary_value(old_summary, "dem_majority_probability")
    new_majority = summary_value(new_summary, "dem_majority_probability")

    expected_delta = (
        None
        if old_expected is None or new_expected is None
        else new_expected - old_expected
    )
    majority_delta = (
        None
        if old_majority is None or new_majority is None
        else new_majority - old_majority
    )

    matched = comparison.loc[comparison["snapshot_status"] == "matched"].copy()
    surprises = matched.loc[matched["abs_delta_margin"] >= threshold].copy()

    lines.extend(
        [
            "=" * 78,
            "HOUSE FORECAST ATTRIBUTION REPORT",
            "=" * 78,
            "",
            f"Old race snapshot: {old_races_path}",
            f"New race snapshot: {new_races_path}",
            f"Matched districts: {len(matched)}",
            f"Old-only districts: {(comparison['snapshot_status'] == 'old_only').sum()}",
            f"New-only districts: {(comparison['snapshot_status'] == 'new_only').sum()}",
            "",
            "CHAMBER SUMMARY",
            "-" * 78,
            (
                "Expected Democratic seats: "
                f"{fmt_number(old_expected)} -> {fmt_number(new_expected)} "
                f"({signed(expected_delta)})"
            ),
            (
                "Democratic majority probability: "
                f"{fmt_probability(old_majority)} -> {fmt_probability(new_majority)} "
                f"({signed(None if majority_delta is None else majority_delta * 100, 1)} pp)"
            ),
            "",
            "DISTRICT MOVEMENT SUMMARY",
            "-" * 78,
            f"Mean absolute margin change: {matched['abs_delta_margin'].mean():.3f}",
            f"Median absolute margin change: {matched['abs_delta_margin'].median():.3f}",
            f"Maximum absolute margin change: {matched['abs_delta_margin'].max():.3f}",
            (
                f"Districts moving at least {threshold:.1f} points: "
                f"{len(surprises)}"
            ),
            "",
            "LARGEST DEMOCRATIC DECLINES",
            "-" * 78,
        ]
    )

    declines = matched.sort_values("delta_margin_dem").head(15)
    for _, row in declines.iterrows():
        lines.append(
            f"{row['district_id']:<8} "
            f"{row['old_margin_dem']:>7.2f} -> {row['new_margin_dem']:>7.2f} "
            f"({row['delta_margin_dem']:+6.2f})  "
            f"{row['primary_cause']}"
        )

    lines.extend(["", "LARGEST DEMOCRATIC GAINS", "-" * 78])

    gains = matched.sort_values("delta_margin_dem", ascending=False).head(15)
    for _, row in gains.iterrows():
        lines.append(
            f"{row['district_id']:<8} "
            f"{row['old_margin_dem']:>7.2f} -> {row['new_margin_dem']:>7.2f} "
            f"({row['delta_margin_dem']:+6.2f})  "
            f"{row['primary_cause']}"
        )

    lines.extend(["", "LARGEST COMPONENT MOVEMENT", "-" * 78])

    for _, row in component_summary.head(20).iterrows():
        lines.append(
            f"{row['display_name']:<38} "
            f"mean abs={row['mean_absolute_change']:.3f}  "
            f"max abs={row['maximum_absolute_change']:.3f}  "
            f"1+ point districts={int(row['districts_changed_1_0_plus'])}"
        )

    lines.extend(
        [
            "",
            "INTERPRETATION NOTE",
            "-" * 78,
            (
                "The final margin change is exact. Individual stored model columns "
                "are diagnostic quantities and are not always mutually exclusive "
                "or additive. For example, the fundamentals margin contains several "
                "earlier adjustments. The primary-cause label therefore identifies "
                "the largest distinct stored input change; it is not a Shapley-value "
                "or causal decomposition."
            ),
            "",
            "=" * 78,
        ]
    )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    require_file(args.old_races, "Old race snapshot")
    require_file(args.new_races, "New race snapshot")

    old_races = read_races(args.old_races, "Old race snapshot")
    new_races = read_races(args.new_races, "New race snapshot")

    old_summary = read_summary(args.old_summary)
    new_summary = read_summary(args.new_summary)

    comparison = build_comparison(old_races, new_races)
    component_summary = build_component_summary(comparison)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    district_output = args.output_dir / "forecast_change_by_district.csv"
    component_output = args.output_dir / "forecast_component_summary.csv"
    surprise_output = args.output_dir / "forecast_surprise_districts.csv"
    text_output = args.output_dir / "house_forecast_attribution_report.txt"

    comparison.to_csv(district_output, index=False)
    component_summary.to_csv(component_output, index=False)

    comparison.loc[
        comparison["abs_delta_margin"] >= args.surprise_threshold
    ].to_csv(surprise_output, index=False)

    report = build_text_report(
        comparison=comparison,
        component_summary=component_summary,
        old_summary=old_summary,
        new_summary=new_summary,
        old_races_path=args.old_races,
        new_races_path=args.new_races,
        threshold=args.surprise_threshold,
    )
    text_output.write_text(report + "\n")

    print(report)
    print()
    print("Outputs written:")
    print(f"  {district_output}")
    print(f"  {component_output}")
    print(f"  {surprise_output}")
    print(f"  {text_output}")


if __name__ == "__main__":
    main()
