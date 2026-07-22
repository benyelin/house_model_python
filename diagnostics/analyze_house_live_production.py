#!/usr/bin/env python3
"""Audit the current live House production forecast.

Reads:
    inputs/house_race_inputs.csv
    outputs/house_race_stats.csv
    outputs/house_forecast_summary.csv

Writes:
    outputs/production_audit/current/
    outputs/production_audit/snapshots/<timestamp>/

The audit:
- reconciles fundamentals components;
- reconciles Bayesian polling arithmetic;
- summarizes component magnitudes;
- measures polling movement from fundamentals;
- identifies the districts most responsible for chamber movement;
- records a timestamped snapshot;
- compares the current forecast with the preceding snapshot when available.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "inputs" / "house_race_inputs.csv"
RACE_STATS_PATH = ROOT / "outputs" / "house_race_stats.csv"
SUMMARY_PATH = ROOT / "outputs" / "house_forecast_summary.csv"
AUDIT_ROOT = ROOT / "outputs" / "production_audit"

EXPECTED_DISTRICTS = 435
PROBABILITY_SCALE = 6.0
NUMERIC_TOLERANCE = 1e-8

COMPONENTS = [
    ("partisan_baseline", "district_partisan_baseline_dem"),
    ("national_environment", "district_environment_adjustment_dem"),
    ("state_environment", "state_environment_adjustment_dem"),
    ("incumbency", "incumbency_adjustment_dem"),
    ("candidate_quality", "candidate_quality_adjustment_dem"),
    ("special_adjustment", "special_adjustment_dem"),
]

JOIN_COLUMNS = [
    "district_id",
    "state",
    "district",
    "district_partisan_baseline_dem",
    "district_environment_adjustment_dem",
    "state_environment_adjustment_dem",
    "incumbency_adjustment_dem",
    "candidate_quality_adjustment_dem",
    "special_adjustment_dem",
    "fundamentals_margin_dem",
    "polling_margin_dem",
    "poll_count",
    "effective_poll_count",
    "polling_active",
    "national_environment_margin_dem",
    "district_elasticity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=INPUT_PATH)
    parser.add_argument("--race-stats", type=Path, default=RACE_STATS_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--output-root", type=Path, default=AUDIT_ROOT)
    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help="Write current audit only; do not save a timestamped snapshot.",
    )
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def numeric(series: pd.Series, default: float = np.nan) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def as_bool(series: pd.Series) -> pd.Series:
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = set(columns)
    return next((candidate for candidate in candidates if candidate in available), None)


def logistic_probability(margin: pd.Series, scale: float = PROBABILITY_SCALE) -> pd.Series:
    values = numeric(margin)
    z = np.clip(values / scale, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def fmt(value: object, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"

    if not math.isfinite(value):
        return "NA"

    return f"{value:.{digits}f}"


def read_inputs(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)

    if "district_id" not in frame.columns:
        raise ValueError(f"{path} does not contain district_id.")

    frame["district_id"] = frame["district_id"].astype(str).str.strip()

    if frame["district_id"].duplicated().any():
        duplicates = frame.loc[
            frame["district_id"].duplicated(keep=False), "district_id"
        ].tolist()
        raise ValueError(f"Duplicate input districts: {duplicates[:20]}")

    return frame


def read_race_stats(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)

    if "district_id" not in frame.columns:
        raise ValueError(f"{path} does not contain district_id.")

    frame["district_id"] = frame["district_id"].astype(str).str.strip()

    if frame["district_id"].duplicated().any():
        duplicates = frame.loc[
            frame["district_id"].duplicated(keep=False), "district_id"
        ].tolist()
        raise ValueError(f"Duplicate race-stat districts: {duplicates[:20]}")

    return frame


def merge_live_data(inputs: pd.DataFrame, race_stats: pd.DataFrame) -> pd.DataFrame:
    selected_inputs = [
        column for column in JOIN_COLUMNS
        if column in inputs.columns
    ]

    merged = race_stats.merge(
        inputs[selected_inputs],
        on="district_id",
        how="outer",
        suffixes=("_stats", "_input"),
        indicator=True,
        validate="one_to_one",
    )

    for base in JOIN_COLUMNS:
        if base == "district_id":
            continue

        stats_col = f"{base}_stats"
        input_col = f"{base}_input"

        if stats_col in merged.columns and input_col in merged.columns:
            merged[base] = merged[stats_col].combine_first(merged[input_col])
        elif stats_col in merged.columns:
            merged[base] = merged[stats_col]
        elif input_col in merged.columns:
            merged[base] = merged[input_col]

    return merged


def add_derived_fields(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()

    numeric_columns = [
        column
        for _, column in COMPONENTS
    ] + [
        "fundamentals_margin_dem",
        "polling_margin_dem",
        "model_margin_dem",
        "bayesian_model_margin_dem",
        "bayesian_polling_weight",
        "bayesian_fundamentals_weight",
        "poll_count",
        "effective_poll_count",
        "poll_quality_count",
        "poll_count_multiplier",
        "dem_win_probability",
        "district_posterior_sd",
        "district_elasticity",
        "national_environment_margin_dem",
    ]

    for column in numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    if "polling_active" in out.columns:
        out["polling_active_bool"] = as_bool(out["polling_active"])
    else:
        out["polling_active_bool"] = False

    component_columns = [
        column for _, column in COMPONENTS
        if column in out.columns
    ]

    if component_columns:
        out["audit_fundamentals_component_sum_dem"] = (
            out[component_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .sum(axis=1)
        )
    else:
        out["audit_fundamentals_component_sum_dem"] = np.nan

    if "fundamentals_margin_dem" in out.columns:
        out["fundamentals_component_residual_dem"] = (
            out["fundamentals_margin_dem"]
            - out["audit_fundamentals_component_sum_dem"]
        )
    else:
        out["fundamentals_component_residual_dem"] = np.nan

    if "bayesian_polling_weight" in out.columns:
        polling_weight = out["bayesian_polling_weight"].fillna(0.0)
    else:
        polling_weight = pd.Series(0.0, index=out.index)

    if "bayesian_fundamentals_weight" in out.columns:
        fundamentals_weight = out["bayesian_fundamentals_weight"].fillna(1.0)
    else:
        fundamentals_weight = 1.0 - polling_weight

    out["audit_polling_weight"] = polling_weight
    out["audit_fundamentals_weight"] = fundamentals_weight
    out["weight_sum"] = polling_weight + fundamentals_weight

    fundamentals = (
        out["fundamentals_margin_dem"]
        if "fundamentals_margin_dem" in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    polling = (
        out["polling_margin_dem"]
        if "polling_margin_dem" in out.columns
        else pd.Series(np.nan, index=out.index)
    )

    out["audit_bayesian_margin_dem"] = (
        fundamentals * fundamentals_weight
        + polling.fillna(0.0) * polling_weight
    )

    stored_model_column = first_existing(
        out.columns,
        ["model_margin_dem", "bayesian_model_margin_dem"],
    )

    if stored_model_column:
        out["stored_model_margin_dem"] = out[stored_model_column]
        out["bayesian_margin_residual_dem"] = (
            out["stored_model_margin_dem"]
            - out["audit_bayesian_margin_dem"]
        )
    else:
        out["stored_model_margin_dem"] = np.nan
        out["bayesian_margin_residual_dem"] = np.nan

    out["polling_raw_gap_dem"] = polling - fundamentals
    out["polling_weighted_margin_shift_dem"] = (
        out["stored_model_margin_dem"] - fundamentals
    )

    probability_column = first_existing(
        out.columns,
        [
            "dem_win_probability",
            "simulated_dem_win_probability",
            "win_probability_dem",
        ],
    )

    if probability_column:
        out["production_dem_win_probability"] = out[probability_column]
    else:
        out["production_dem_win_probability"] = logistic_probability(
            out["stored_model_margin_dem"]
        )

    # This is deliberately labeled an analytic estimate. It provides a common,
    # transparent scale for assessing how polling changes expected seats.
    out["analytic_fundamentals_probability"] = logistic_probability(fundamentals)
    out["analytic_post_poll_probability"] = logistic_probability(
        out["stored_model_margin_dem"]
    )
    out["analytic_polling_seat_effect"] = (
        out["analytic_post_poll_probability"]
        - out["analytic_fundamentals_probability"]
    )

    out["competitive_distance"] = out["stored_model_margin_dem"].abs()

    return out


def build_component_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for label, column in COMPONENTS:
        if column not in frame.columns:
            continue

        values = pd.to_numeric(frame[column], errors="coerce")

        rows.append(
            {
                "component": label,
                "source_column": column,
                "nonmissing_rows": int(values.notna().sum()),
                "nonzero_rows": int(values.fillna(0.0).abs().gt(NUMERIC_TOLERANCE).sum()),
                "mean": values.mean(),
                "median": values.median(),
                "mean_absolute_value": values.abs().mean(),
                "standard_deviation": values.std(),
                "minimum": values.min(),
                "maximum": values.max(),
                "sum": values.sum(),
            }
        )

    return pd.DataFrame(rows)


def extract_summary_metrics(
    summary: pd.DataFrame,
    frame: pd.DataFrame,
) -> dict[str, object]:
    metrics: dict[str, object] = {}

    if not summary.empty:
        source = summary.iloc[0].to_dict()
        for key, value in source.items():
            if pd.notna(value):
                metrics[str(key)] = value

    metrics["audit_district_count"] = len(frame)
    metrics["audit_expected_dem_seats_from_race_probabilities"] = (
        frame["production_dem_win_probability"].sum()
    )
    metrics["audit_analytic_fundamentals_expected_dem_seats"] = (
        frame["analytic_fundamentals_probability"].sum()
    )
    metrics["audit_analytic_post_poll_expected_dem_seats"] = (
        frame["analytic_post_poll_probability"].sum()
    )
    metrics["audit_analytic_polling_expected_seat_effect"] = (
        frame["analytic_polling_seat_effect"].sum()
    )
    metrics["audit_active_polling_districts"] = int(
        frame["audit_polling_weight"].gt(NUMERIC_TOLERANCE).sum()
    )
    metrics["audit_tossups_model_margin_within_2"] = int(
        frame["competitive_distance"].le(2.0).sum()
    )
    metrics["audit_competitive_model_margin_within_5"] = int(
        frame["competitive_distance"].le(5.0).sum()
    )

    return metrics


def validate(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    passes: list[str] = []
    failures: list[str] = []

    if len(frame) == EXPECTED_DISTRICTS:
        passes.append(f"district rows = {EXPECTED_DISTRICTS}")
    else:
        failures.append(
            f"district rows = {len(frame)}; expected {EXPECTED_DISTRICTS}"
        )

    if frame["district_id"].nunique() == len(frame):
        passes.append("district_id values are unique")
    else:
        failures.append("district_id values are not unique")

    merge_counts = frame["_merge"].value_counts(dropna=False).to_dict()
    both_count = int(merge_counts.get("both", 0))
    left_only_count = int(merge_counts.get("left_only", 0))
    right_only_count = int(merge_counts.get("right_only", 0))

    if (
        both_count == len(frame)
        and left_only_count == 0
        and right_only_count == 0
    ):
        passes.append("input and race-stat district coverage matches exactly")
    else:
        failures.append(f"input/race-stat merge coverage = {merge_counts}")

    weight_residual = (frame["weight_sum"] - 1.0).abs()
    max_weight_residual = weight_residual.max()

    if pd.notna(max_weight_residual) and max_weight_residual <= NUMERIC_TOLERANCE:
        passes.append(
            f"Bayesian weights sum to 1; max residual = {max_weight_residual:.3e}"
        )
    else:
        failures.append(
            f"Bayesian weights do not sum to 1; max residual = {max_weight_residual}"
        )

    margin_residual = frame["bayesian_margin_residual_dem"].abs().dropna()

    if len(margin_residual) == len(frame) and margin_residual.max() <= NUMERIC_TOLERANCE:
        passes.append(
            "stored model margin reconciles to Bayesian arithmetic; "
            f"max residual = {margin_residual.max():.3e}"
        )
    else:
        failures.append(
            "stored model margin does not fully reconcile; "
            f"nonmissing={len(margin_residual)}, "
            f"max residual={margin_residual.max() if len(margin_residual) else 'NA'}"
        )

    probabilities = frame["production_dem_win_probability"]
    invalid_probability = probabilities.notna() & (
        probabilities.lt(0.0) | probabilities.gt(1.0)
    )

    if not invalid_probability.any():
        passes.append("all stored Democratic win probabilities are within [0, 1]")
    else:
        failures.append(
            f"{int(invalid_probability.sum())} probabilities fall outside [0, 1]"
        )

    if frame["fundamentals_margin_dem"].notna().all():
        passes.append("all districts have fundamentals margins")
    else:
        failures.append(
            f"{int(frame['fundamentals_margin_dem'].isna().sum())} "
            "districts lack fundamentals margins"
        )

    if frame["stored_model_margin_dem"].notna().all():
        passes.append("all districts have model margins")
    else:
        failures.append(
            f"{int(frame['stored_model_margin_dem'].isna().sum())} "
            "districts lack model margins"
        )

    return passes, failures


def select_output_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "district_id",
        "state",
        "district",
        "district_partisan_baseline_dem",
        "district_environment_adjustment_dem",
        "state_environment_adjustment_dem",
        "incumbency_adjustment_dem",
        "candidate_quality_adjustment_dem",
        "special_adjustment_dem",
        "audit_fundamentals_component_sum_dem",
        "fundamentals_margin_dem",
        "fundamentals_component_residual_dem",
        "polling_active_bool",
        "polling_margin_dem",
        "poll_count",
        "effective_poll_count",
        "poll_quality_count",
        "audit_polling_weight",
        "audit_fundamentals_weight",
        "weight_sum",
        "audit_bayesian_margin_dem",
        "stored_model_margin_dem",
        "bayesian_margin_residual_dem",
        "polling_raw_gap_dem",
        "polling_weighted_margin_shift_dem",
        "production_dem_win_probability",
        "analytic_fundamentals_probability",
        "analytic_post_poll_probability",
        "analytic_polling_seat_effect",
        "district_posterior_sd",
    ]

    return [column for column in preferred if column in frame.columns]


def write_text_summary(
    path: Path,
    metrics: dict[str, object],
    components: pd.DataFrame,
    polling_movers: pd.DataFrame,
    passes: list[str],
    failures: list[str],
) -> None:
    lines: list[str] = []

    lines.extend(
        [
            "House Live Production Audit",
            "=" * 72,
            "",
            f"Generated: {datetime.now().isoformat(timespec='seconds')}",
            "",
            "Chamber summary",
            "-" * 72,
        ]
    )

    priority_keys = [
        "dem_control_probability",
        "democratic_control_probability",
        "expected_dem_seats",
        "median_dem_seats",
        "p10_dem_seats",
        "p90_dem_seats",
        "audit_expected_dem_seats_from_race_probabilities",
        "audit_analytic_fundamentals_expected_dem_seats",
        "audit_analytic_post_poll_expected_dem_seats",
        "audit_analytic_polling_expected_seat_effect",
        "audit_active_polling_districts",
        "audit_tossups_model_margin_within_2",
        "audit_competitive_model_margin_within_5",
    ]

    shown: set[str] = set()

    for key in priority_keys:
        if key in metrics:
            lines.append(f"{key}: {metrics[key]}")
            shown.add(key)

    for key, value in metrics.items():
        if key not in shown:
            lines.append(f"{key}: {value}")

    lines.extend(
        [
            "",
            "Component summary",
            "-" * 72,
        ]
    )

    if components.empty:
        lines.append("No recognized component columns found.")
    else:
        for row in components.itertuples(index=False):
            lines.append(
                f"{row.component}: "
                f"mean={fmt(row.mean)}, "
                f"mean_abs={fmt(row.mean_absolute_value)}, "
                f"min={fmt(row.minimum)}, "
                f"max={fmt(row.maximum)}, "
                f"nonzero={row.nonzero_rows}"
            )

    lines.extend(
        [
            "",
            "Largest polling movers",
            "-" * 72,
        ]
    )

    for row in polling_movers.head(20).itertuples(index=False):
        lines.append(
            f"{row.district_id}: "
            f"fundamentals={fmt(row.fundamentals_margin_dem)}, "
            f"polling={fmt(row.polling_margin_dem)}, "
            f"weight={fmt(row.audit_polling_weight)}, "
            f"model={fmt(row.stored_model_margin_dem)}, "
            f"margin_shift={fmt(row.polling_weighted_margin_shift_dem)}, "
            f"analytic_seat_effect={fmt(row.analytic_polling_seat_effect, 4)}"
        )

    lines.extend(
        [
            "",
            "Validation",
            "-" * 72,
        ]
    )

    lines.extend(f"PASS: {item}" for item in passes)
    lines.extend(f"FAIL: {item}" for item in failures)

    lines.extend(
        [
            "",
            "VALIDATION PASSED" if not failures else "VALIDATION FAILED",
            "",
        ]
    )

    path.write_text("\n".join(lines))


def compare_with_previous(
    current: pd.DataFrame,
    snapshot_root: Path,
    current_snapshot_name: str,
) -> pd.DataFrame:
    previous_files = sorted(
        path
        for path in snapshot_root.glob("*/live_district_audit.csv")
        if path.parent.name != current_snapshot_name
    )

    if not previous_files:
        return pd.DataFrame()

    previous_path = previous_files[-1]
    previous = pd.read_csv(previous_path, low_memory=False)

    keep = [
        "district_id",
        "fundamentals_margin_dem",
        "polling_margin_dem",
        "audit_polling_weight",
        "stored_model_margin_dem",
        "production_dem_win_probability",
    ]

    previous = previous[
        [column for column in keep if column in previous.columns]
    ].copy()

    current_keep = current[
        [column for column in keep if column in current.columns]
    ].copy()

    comparison = current_keep.merge(
        previous,
        on="district_id",
        how="outer",
        suffixes=("_current", "_previous"),
        validate="one_to_one",
    )

    for column in keep:
        if column == "district_id":
            continue

        current_col = f"{column}_current"
        previous_col = f"{column}_previous"

        if current_col in comparison.columns and previous_col in comparison.columns:
            comparison[f"{column}_change"] = (
                numeric(comparison[current_col])
                - numeric(comparison[previous_col])
            )

    if "production_dem_win_probability_change" in comparison.columns:
        comparison = comparison.sort_values(
            "production_dem_win_probability_change",
            key=lambda values: values.abs(),
            ascending=False,
        )
    elif "stored_model_margin_dem_change" in comparison.columns:
        comparison = comparison.sort_values(
            "stored_model_margin_dem_change",
            key=lambda values: values.abs(),
            ascending=False,
        )

    comparison.attrs["previous_snapshot"] = str(previous_path.parent)
    return comparison


def write_outputs(
    frame: pd.DataFrame,
    summary_metrics: dict[str, object],
    components: pd.DataFrame,
    passes: list[str],
    failures: list[str],
    output_root: Path,
    no_snapshot: bool,
) -> None:
    current_dir = output_root / "current"
    snapshot_root = output_root / "snapshots"
    current_dir.mkdir(parents=True, exist_ok=True)
    snapshot_root.mkdir(parents=True, exist_ok=True)

    polling_movers = frame.loc[
        frame["audit_polling_weight"].gt(NUMERIC_TOLERANCE)
    ].copy()

    polling_movers = polling_movers.sort_values(
        "polling_weighted_margin_shift_dem",
        key=lambda values: values.abs(),
        ascending=False,
    )

    seat_effects = frame.sort_values(
        "analytic_polling_seat_effect",
        key=lambda values: values.abs(),
        ascending=False,
    )

    tossups = frame.loc[
        frame["stored_model_margin_dem"].abs().le(5.0)
    ].sort_values("stored_model_margin_dem")

    reconciliation = frame.sort_values(
        "bayesian_margin_residual_dem",
        key=lambda values: values.abs(),
        ascending=False,
    )

    output_columns = select_output_columns(frame)
    live_audit = frame[output_columns].copy()

    live_audit.to_csv(current_dir / "live_district_audit.csv", index=False)
    components.to_csv(current_dir / "component_summary.csv", index=False)
    polling_movers[output_columns].to_csv(
        current_dir / "polling_movers.csv",
        index=False,
    )
    seat_effects[output_columns].to_csv(
        current_dir / "analytic_chamber_attribution.csv",
        index=False,
    )
    tossups[output_columns].to_csv(
        current_dir / "competitive_districts.csv",
        index=False,
    )
    reconciliation[output_columns].to_csv(
        current_dir / "arithmetic_reconciliation.csv",
        index=False,
    )

    pd.DataFrame(
        [{"metric": key, "value": value} for key, value in summary_metrics.items()]
    ).to_csv(current_dir / "audit_summary_metrics.csv", index=False)

    validation_lines = (
        [f"PASS: {item}" for item in passes]
        + [f"FAIL: {item}" for item in failures]
        + [""]
        + ["VALIDATION PASSED" if not failures else "VALIDATION FAILED"]
    )
    (current_dir / "validation.txt").write_text("\n".join(validation_lines) + "\n")

    write_text_summary(
        current_dir / "production_audit_summary.txt",
        summary_metrics,
        components,
        polling_movers,
        passes,
        failures,
    )

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_rows": len(frame),
        "validation_passed": not failures,
        "validation_failures": failures,
    }
    (current_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str)
    )

    if no_snapshot:
        return

    snapshot_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = snapshot_root / snapshot_name
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    previous_comparison = compare_with_previous(
        live_audit,
        snapshot_root,
        snapshot_name,
    )

    for path in current_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, snapshot_dir / path.name)

    if not previous_comparison.empty:
        previous_comparison.to_csv(
            current_dir / "change_from_previous_snapshot.csv",
            index=False,
        )
        previous_comparison.to_csv(
            snapshot_dir / "change_from_previous_snapshot.csv",
            index=False,
        )
        (current_dir / "previous_snapshot.txt").write_text(
            previous_comparison.attrs.get("previous_snapshot", "") + "\n"
        )
        shutil.copy2(
            current_dir / "previous_snapshot.txt",
            snapshot_dir / "previous_snapshot.txt",
        )


def main() -> None:
    args = parse_args()

    require_file(args.inputs)
    require_file(args.race_stats)
    require_file(args.summary)

    inputs = read_inputs(args.inputs)
    race_stats = read_race_stats(args.race_stats)
    summary = pd.read_csv(args.summary, low_memory=False)

    merged = merge_live_data(inputs, race_stats)
    audited = add_derived_fields(merged)
    components = build_component_summary(audited)
    summary_metrics = extract_summary_metrics(summary, audited)
    passes, failures = validate(audited)

    write_outputs(
        audited,
        summary_metrics,
        components,
        passes,
        failures,
        args.output_root,
        args.no_snapshot,
    )

    print("=" * 78)
    print("House Live Production Audit")
    print("=" * 78)
    print(f"District rows: {len(audited)}")
    print(
        "Expected Democratic seats from production race probabilities: "
        f"{summary_metrics['audit_expected_dem_seats_from_race_probabilities']:.3f}"
    )
    print(
        "Analytic fundamentals-only expected seats: "
        f"{summary_metrics['audit_analytic_fundamentals_expected_dem_seats']:.3f}"
    )
    print(
        "Analytic post-poll expected seats: "
        f"{summary_metrics['audit_analytic_post_poll_expected_dem_seats']:.3f}"
    )
    print(
        "Analytic polling expected-seat effect: "
        f"{summary_metrics['audit_analytic_polling_expected_seat_effect']:+.3f}"
    )
    print(
        "Active polling districts: "
        f"{summary_metrics['audit_active_polling_districts']}"
    )
    print()

    print("Component averages")
    print("-" * 78)
    if components.empty:
        print("No recognized components found.")
    else:
        print(
            components[
                [
                    "component",
                    "nonzero_rows",
                    "mean",
                    "mean_absolute_value",
                    "minimum",
                    "maximum",
                ]
            ].to_string(index=False)
        )

    print()
    for item in passes:
        print(f"PASS: {item}")
    for item in failures:
        print(f"FAIL: {item}")

    print()
    print("VALIDATION PASSED" if not failures else "VALIDATION FAILED")
    print()
    print(f"Wrote: {args.output_root / 'current'}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
