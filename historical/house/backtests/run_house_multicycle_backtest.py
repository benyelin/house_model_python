#!/usr/bin/env python3
"""
Run and validate the canonical multicycle House historical backtest.

This harness delegates each individual cycle to:

    historical/house/backtests/run_house_historical_backtest.py

It then combines and validates the cycle-level outputs for:

    2016, 2018, 2020, and 2022

The harness can also compare the consolidated summary and race-level
predictions against an explicitly saved regression benchmark.

Design principles:
    - reuse the canonical single-cycle scorer
    - fail on subprocess errors
    - deterministic output organization
    - strict cycle/race uniqueness
    - explicit benchmark updates only
    - tolerances tight enough to catch unintended model changes
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SINGLE_CYCLE_RUNNER = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "backtests"
    / "run_house_historical_backtest.py"
)

CANONICAL_INPUT_PATH = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "warehouse"
    / "house_historical_backtest_inputs_2016_2022.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "canonical"
)

DEFAULT_BENCHMARK_DIR = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "backtests"
    / "benchmarks"
    / "canonical_fundamentals_v1"
)

TARGET_CYCLES: tuple[int, ...] = (2016, 2018, 2020, 2022)
EXPECTED_ROWS_PER_CYCLE = 435

SUMMARY_FILENAME = "house_multicycle_backtest_summary.csv"
RESULTS_FILENAME = "house_multicycle_backtest_results.csv"
VALIDATION_FILENAME = "house_multicycle_backtest_validation.txt"
MANIFEST_FILENAME = "house_multicycle_backtest_manifest.csv"

BENCHMARK_SUMMARY_FILENAME = "benchmark_summary.csv"
BENCHMARK_RESULTS_FILENAME = "benchmark_results.csv"
BENCHMARK_METADATA_FILENAME = "benchmark_metadata.txt"

SUMMARY_KEY = ["cycle"]
RESULTS_KEY = ["cycle", "race_id"]

SUMMARY_METRIC_COLUMNS = [
    "all_races",
    "scored_major_party_races",
    "winner_accuracy",
    "mean_abs_margin_error",
    "median_abs_margin_error",
    "rmse_margin_error",
    "mean_margin_error_dem_bias",
    "brier_score",
    "log_loss",
    "actual_dem_seats",
    "predicted_dem_seats",
    "expected_dem_seats",
    "predicted_seat_error",
    "expected_seat_error",
]

RACE_REGRESSION_COLUMNS = [
    "cycle",
    "race_id",
    "model_margin_dem",
    "dem_win_probability",
    "include_in_scoring",
    "actual_dem_margin",
    "actual_winner",
]

EXACT_SUMMARY_COLUMNS = {
    "all_races",
    "scored_major_party_races",
    "actual_dem_seats",
    "predicted_dem_seats",
    "predicted_seat_error",
}

FLOAT_TOLERANCE = 1e-9


class ValidationError(RuntimeError):
    """Raised when the multicycle backtest violates its contract."""


@dataclass(frozen=True)
class CycleArtifacts:
    cycle: int
    output_dir: Path
    readiness_path: Path
    summary_path: Path
    results_path: Path
    calibration_path: Path


def log(message: str = "") -> None:
    print(message, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def require_file(path: Path, label: str) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def atomic_write_csv(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".tmp",
        prefix=f".{output_path.stem}.",
        dir=output_path.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)

    try:
        frame.to_csv(
            temporary_path,
            index=False,
            lineterminator="\n",
            float_format="%.17g",
        )
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_text(text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".tmp",
        prefix=f".{output_path.stem}.",
        dir=output_path.parent,
        delete=False,
    ) as temporary:
        temporary.write(text)
        temporary_path = Path(temporary.name)

    try:
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def cycle_artifacts(
    cycle: int,
    output_root: Path,
) -> CycleArtifacts:
    cycle_dir = output_root / str(cycle)

    return CycleArtifacts(
        cycle=cycle,
        output_dir=cycle_dir,
        readiness_path=(
            cycle_dir / f"house_{cycle}_backtest_readiness.txt"
        ),
        summary_path=(
            cycle_dir / f"house_{cycle}_backtest_summary.csv"
        ),
        results_path=(
            cycle_dir / f"house_{cycle}_backtest_results.csv"
        ),
        calibration_path=(
            cycle_dir / f"house_{cycle}_backtest_calibration.csv"
        ),
    )


def run_cycle(
    *,
    cycle: int,
    master_path: Path,
    output_root: Path,
    default_error_sd: float,
    python_executable: str,
    candidate_quality_weight: float,
    candidate_war_path: Path,
) -> CycleArtifacts:
    artifacts = cycle_artifacts(cycle, output_root)
    artifacts.output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        python_executable,
        str(SINGLE_CYCLE_RUNNER),
        "--master-path",
        str(master_path),
        "--cycle",
        str(cycle),
        "--output-dir",
        str(artifacts.output_dir),
        "--default-error-sd",
        str(default_error_sd),
        "--candidate-quality-weight",
        str(candidate_quality_weight),
        "--candidate-war-path",
        str(candidate_war_path),
    ]

    log()
    log("=" * 72)
    log(f"Running canonical House backtest: {cycle}")
    log("=" * 72)

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"Backtest subprocess failed for {cycle} with "
            f"exit code {completed.returncode}."
        )

    for path, label in (
        (artifacts.readiness_path, "readiness report"),
        (artifacts.summary_path, "summary output"),
        (artifacts.results_path, "race-level output"),
        (artifacts.calibration_path, "calibration output"),
    ):
        require_file(path, f"{cycle} {label}")

    return artifacts


def validate_cycle_outputs(
    artifacts: CycleArtifacts,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    summary = pd.read_csv(
        artifacts.summary_path,
        low_memory=False,
    )
    results = pd.read_csv(
        artifacts.results_path,
        low_memory=False,
    )

    if len(summary) != 1:
        raise ValidationError(
            f"{artifacts.cycle} summary must contain exactly one row; "
            f"found {len(summary)}."
        )

    if len(results) != EXPECTED_ROWS_PER_CYCLE:
        raise ValidationError(
            f"{artifacts.cycle} results expected "
            f"{EXPECTED_ROWS_PER_CYCLE} rows; found {len(results)}."
        )

    required_summary = {"cycle", *SUMMARY_METRIC_COLUMNS}
    missing_summary = sorted(
        required_summary - set(summary.columns)
    )
    if missing_summary:
        raise ValidationError(
            f"{artifacts.cycle} summary missing columns: "
            f"{missing_summary}"
        )

    required_results = set(RACE_REGRESSION_COLUMNS)
    missing_results = sorted(
        required_results - set(results.columns)
    )
    if missing_results:
        raise ValidationError(
            f"{artifacts.cycle} results missing columns: "
            f"{missing_results}"
        )

    summary_cycle = int(
        pd.to_numeric(
            summary["cycle"],
            errors="raise",
        ).iloc[0]
    )
    if summary_cycle != artifacts.cycle:
        raise ValidationError(
            f"Summary cycle {summary_cycle} does not match "
            f"expected cycle {artifacts.cycle}."
        )

    result_cycles = (
        pd.to_numeric(results["cycle"], errors="raise")
        .astype(int)
        .unique()
        .tolist()
    )
    if result_cycles != [artifacts.cycle]:
        raise ValidationError(
            f"{artifacts.cycle} result rows contain cycles: "
            f"{result_cycles}"
        )

    if results["race_id"].isna().any():
        raise ValidationError(
            f"{artifacts.cycle} results contain missing race IDs."
        )

    if results["race_id"].duplicated().any():
        duplicate_ids = (
            results.loc[
                results["race_id"].duplicated(False),
                "race_id",
            ]
            .astype(str)
            .tolist()
        )
        raise ValidationError(
            f"{artifacts.cycle} duplicate race IDs: {duplicate_ids}"
        )

    for column in (
        "model_margin_dem",
        "dem_win_probability",
        "actual_dem_margin",
    ):
        numeric = pd.to_numeric(
            results[column],
            errors="coerce",
        )

        if column == "model_margin_dem":
            eligible = parse_bool_series(
                results["include_in_scoring"]
            )
            if numeric.loc[eligible].isna().any():
                raise ValidationError(
                    f"{artifacts.cycle} has missing model margins "
                    "among scored races."
                )
        elif numeric.isna().any():
            raise ValidationError(
                f"{artifacts.cycle} has missing {column} values."
            )

        finite = numeric.dropna()
        if not np.isfinite(finite).all():
            raise ValidationError(
                f"{artifacts.cycle} has non-finite {column} values."
            )

    probabilities = pd.to_numeric(
        results["dem_win_probability"],
        errors="coerce",
    )
    invalid_probability = (
        probabilities.notna()
        & ~probabilities.between(0.0, 1.0)
    )
    if invalid_probability.any():
        raise ValidationError(
            f"{artifacts.cycle} contains probabilities outside [0, 1]."
        )

    for column in SUMMARY_METRIC_COLUMNS:
        numeric = pd.to_numeric(
            summary[column],
            errors="coerce",
        )

        if numeric.isna().any() or not np.isfinite(numeric).all():
            raise ValidationError(
                f"{artifacts.cycle} summary contains invalid "
                f"metric {column}."
            )

    included = int(
        parse_bool_series(results["include_in_scoring"]).sum()
    )
    summary_included = int(
        pd.to_numeric(
            summary["scored_major_party_races"],
            errors="raise",
        ).iloc[0]
    )

    if included != summary_included:
        raise ValidationError(
            f"{artifacts.cycle} scored-race count mismatch: "
            f"results={included}, summary={summary_included}."
        )

    manifest_row = {
        "cycle": artifacts.cycle,
        "rows": len(results),
        "unique_races": int(results["race_id"].nunique()),
        "scored_races": included,
        "missing_model_margin": int(
            pd.to_numeric(
                results["model_margin_dem"],
                errors="coerce",
            ).isna().sum()
        ),
        "missing_presidential_baseline": (
            int(
                pd.to_numeric(
                    results["district_pres_margin_dem"],
                    errors="coerce",
                ).isna().sum()
            )
            if "district_pres_margin_dem" in results.columns
            else np.nan
        ),
        "summary_sha256": sha256_file(artifacts.summary_path),
        "results_sha256": sha256_file(artifacts.results_path),
        "calibration_sha256": sha256_file(
            artifacts.calibration_path
        ),
        "readiness_sha256": sha256_file(
            artifacts.readiness_path
        ),
    }

    return summary, results, manifest_row


def validate_combined_outputs(
    summary: pd.DataFrame,
    results: pd.DataFrame,
) -> None:
    if len(summary) != len(TARGET_CYCLES):
        raise ValidationError(
            f"Combined summary expected {len(TARGET_CYCLES)} rows; "
            f"found {len(summary)}."
        )

    summary_cycles = sorted(
        pd.to_numeric(summary["cycle"], errors="raise")
        .astype(int)
        .tolist()
    )
    if summary_cycles != list(TARGET_CYCLES):
        raise ValidationError(
            f"Combined summary cycles are {summary_cycles}; "
            f"expected {list(TARGET_CYCLES)}."
        )

    expected_total_rows = (
        EXPECTED_ROWS_PER_CYCLE * len(TARGET_CYCLES)
    )
    if len(results) != expected_total_rows:
        raise ValidationError(
            f"Combined results expected {expected_total_rows} rows; "
            f"found {len(results)}."
        )

    if results.duplicated(RESULTS_KEY).any():
        examples = (
            results.loc[
                results.duplicated(RESULTS_KEY, keep=False),
                RESULTS_KEY,
            ]
            .head(20)
            .to_dict("records")
        )
        raise ValidationError(
            f"Combined results contain duplicate keys. "
            f"Examples: {examples}"
        )

    counts = (
        results.groupby("cycle")
        .size()
        .reindex(TARGET_CYCLES, fill_value=0)
    )
    if not counts.eq(EXPECTED_ROWS_PER_CYCLE).all():
        raise ValidationError(
            f"Combined cycle row counts are invalid: "
            f"{counts.to_dict()}"
        )


def normalize_regression_frame(
    frame: pd.DataFrame,
    *,
    keys: list[str],
    columns: Iterable[str],
) -> pd.DataFrame:
    selected = frame[list(columns)].copy()

    for key in keys:
        if key == "cycle":
            selected[key] = pd.to_numeric(
                selected[key],
                errors="raise",
            ).astype(int)
        else:
            selected[key] = selected[key].astype(str)

    selected = selected.sort_values(
        keys,
        kind="mergesort",
    ).reset_index(drop=True)

    return selected


def compare_exact_column(
    current: pd.Series,
    benchmark: pd.Series,
    *,
    label: str,
) -> list[str]:
    differences: list[str] = []

    current_numeric = pd.to_numeric(
        current,
        errors="coerce",
    )
    benchmark_numeric = pd.to_numeric(
        benchmark,
        errors="coerce",
    )

    mismatch = current_numeric.ne(benchmark_numeric)

    if mismatch.any():
        examples = pd.DataFrame(
            {
                "current": current_numeric.loc[mismatch],
                "benchmark": benchmark_numeric.loc[mismatch],
            }
        ).head(10)

        differences.append(
            f"{label} changed. Examples: "
            f"{examples.to_dict('records')}"
        )

    return differences


def compare_float_column(
    current: pd.Series,
    benchmark: pd.Series,
    *,
    label: str,
    tolerance: float,
) -> list[str]:
    differences: list[str] = []

    current_numeric = pd.to_numeric(
        current,
        errors="coerce",
    )
    benchmark_numeric = pd.to_numeric(
        benchmark,
        errors="coerce",
    )

    both_missing = current_numeric.isna() & benchmark_numeric.isna()
    one_missing = current_numeric.isna() ^ benchmark_numeric.isna()

    absolute_difference = (
        current_numeric - benchmark_numeric
    ).abs()

    mismatch = (
        one_missing
        | (
            ~both_missing
            & absolute_difference.gt(tolerance)
        )
    )

    if mismatch.any():
        examples = pd.DataFrame(
            {
                "current": current_numeric.loc[mismatch],
                "benchmark": benchmark_numeric.loc[mismatch],
                "absolute_difference": (
                    absolute_difference.loc[mismatch]
                ),
            }
        ).head(10)

        differences.append(
            f"{label} changed beyond tolerance {tolerance:.3g}. "
            f"Maximum difference: "
            f"{absolute_difference.loc[mismatch].max():.12g}. "
            f"Examples: {examples.to_dict('records')}"
        )

    return differences


def compare_to_benchmark(
    *,
    current_summary: pd.DataFrame,
    current_results: pd.DataFrame,
    benchmark_dir: Path,
    tolerance: float,
) -> list[str]:
    benchmark_summary_path = (
        benchmark_dir / BENCHMARK_SUMMARY_FILENAME
    )
    benchmark_results_path = (
        benchmark_dir / BENCHMARK_RESULTS_FILENAME
    )

    require_file(
        benchmark_summary_path,
        "benchmark summary",
    )
    require_file(
        benchmark_results_path,
        "benchmark results",
    )

    benchmark_summary = pd.read_csv(
        benchmark_summary_path,
        low_memory=False,
    )
    benchmark_results = pd.read_csv(
        benchmark_results_path,
        low_memory=False,
    )

    current_summary = normalize_regression_frame(
        current_summary,
        keys=SUMMARY_KEY,
        columns=["cycle", *SUMMARY_METRIC_COLUMNS],
    )
    benchmark_summary = normalize_regression_frame(
        benchmark_summary,
        keys=SUMMARY_KEY,
        columns=["cycle", *SUMMARY_METRIC_COLUMNS],
    )

    current_results = normalize_regression_frame(
        current_results,
        keys=RESULTS_KEY,
        columns=RACE_REGRESSION_COLUMNS,
    )
    benchmark_results = normalize_regression_frame(
        benchmark_results,
        keys=RESULTS_KEY,
        columns=RACE_REGRESSION_COLUMNS,
    )

    differences: list[str] = []

    if current_summary[SUMMARY_KEY].to_dict("records") != (
        benchmark_summary[SUMMARY_KEY].to_dict("records")
    ):
        differences.append(
            "Summary cycle keys differ from benchmark."
        )
        return differences

    if current_results[RESULTS_KEY].to_dict("records") != (
        benchmark_results[RESULTS_KEY].to_dict("records")
    ):
        current_keys = set(
            map(tuple, current_results[RESULTS_KEY].to_numpy())
        )
        benchmark_keys = set(
            map(tuple, benchmark_results[RESULTS_KEY].to_numpy())
        )

        differences.append(
            "Race-level keys differ from benchmark. "
            f"Only current: {sorted(current_keys - benchmark_keys)[:20]}; "
            f"only benchmark: "
            f"{sorted(benchmark_keys - current_keys)[:20]}"
        )
        return differences

    for column in SUMMARY_METRIC_COLUMNS:
        if column in EXACT_SUMMARY_COLUMNS:
            differences.extend(
                compare_exact_column(
                    current_summary[column],
                    benchmark_summary[column],
                    label=f"summary.{column}",
                )
            )
        else:
            differences.extend(
                compare_float_column(
                    current_summary[column],
                    benchmark_summary[column],
                    label=f"summary.{column}",
                    tolerance=tolerance,
                )
            )

    current_scoring = parse_bool_series(
        current_results["include_in_scoring"]
    )
    benchmark_scoring = parse_bool_series(
        benchmark_results["include_in_scoring"]
    )

    if not current_scoring.equals(benchmark_scoring):
        mismatch = current_scoring.ne(benchmark_scoring)
        examples = current_results.loc[
            mismatch,
            RESULTS_KEY,
        ].head(20)

        differences.append(
            "Race-level scoring eligibility changed. "
            f"Examples: {examples.to_dict('records')}"
        )

    actual_winner_changed = (
        current_results["actual_winner"]
        .fillna("")
        .astype(str)
        .ne(
            benchmark_results["actual_winner"]
            .fillna("")
            .astype(str)
        )
    )
    if actual_winner_changed.any():
        examples = current_results.loc[
            actual_winner_changed,
            RESULTS_KEY,
        ].head(20)

        differences.append(
            "Actual winner fields differ from benchmark. "
            f"Examples: {examples.to_dict('records')}"
        )

    for column in (
        "model_margin_dem",
        "dem_win_probability",
        "actual_dem_margin",
    ):
        differences.extend(
            compare_float_column(
                current_results[column],
                benchmark_results[column],
                label=f"results.{column}",
                tolerance=tolerance,
            )
        )

    return differences


def write_benchmark(
    *,
    summary: pd.DataFrame,
    results: pd.DataFrame,
    benchmark_dir: Path,
    master_path: Path,
    default_error_sd: float,
) -> None:
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    benchmark_summary = normalize_regression_frame(
        summary,
        keys=SUMMARY_KEY,
        columns=["cycle", *SUMMARY_METRIC_COLUMNS],
    )

    benchmark_results = normalize_regression_frame(
        results,
        keys=RESULTS_KEY,
        columns=RACE_REGRESSION_COLUMNS,
    )

    atomic_write_csv(
        benchmark_summary,
        benchmark_dir / BENCHMARK_SUMMARY_FILENAME,
    )
    atomic_write_csv(
        benchmark_results,
        benchmark_dir / BENCHMARK_RESULTS_FILENAME,
    )

    metadata = "\n".join(
        [
            "House canonical fundamentals regression benchmark",
            "=" * 49,
            "",
            f"Cycles: {', '.join(map(str, TARGET_CYCLES))}",
            f"Rows per cycle: {EXPECTED_ROWS_PER_CYCLE}",
            f"Canonical input: {master_path}",
            f"Canonical input SHA-256: {sha256_file(master_path)}",
            f"Default error SD: {default_error_sd:.12g}",
            (
                "Benchmark summary SHA-256: "
                f"{sha256_file(benchmark_dir / BENCHMARK_SUMMARY_FILENAME)}"
            ),
            (
                "Benchmark results SHA-256: "
                f"{sha256_file(benchmark_dir / BENCHMARK_RESULTS_FILENAME)}"
            ),
            "",
            (
                "This benchmark represents the cycle-safe fundamentals "
                "model before out-of-fold elasticity, District DNA, and "
                "candidate-quality components are added."
            ),
        ]
    )

    atomic_write_text(
        metadata + "\n",
        benchmark_dir / BENCHMARK_METADATA_FILENAME,
    )


def build_validation_report(
    *,
    summary: pd.DataFrame,
    manifest: pd.DataFrame,
    benchmark_status: str,
    benchmark_differences: list[str],
    master_path: Path,
    output_dir: Path,
) -> str:
    lines = [
        "House Canonical Multicycle Backtest Validation",
        "=" * 47,
        "",
        f"Canonical input: {master_path}",
        f"Canonical input SHA-256: {sha256_file(master_path)}",
        f"Output directory: {output_dir}",
        f"Cycles: {', '.join(map(str, TARGET_CYCLES))}",
        "",
        "Cycle manifest:",
        manifest.to_string(index=False),
        "",
        "Combined summary:",
        summary.to_string(index=False),
        "",
        f"Regression benchmark status: {benchmark_status}",
    ]

    if benchmark_differences:
        lines.extend(
            [
                "",
                "Regression differences:",
                *[
                    f"- {difference}"
                    for difference in benchmark_differences
                ],
            ]
        )

    lines.extend(
        [
            "",
            "Validation PASSED"
            if not benchmark_differences
            else "Validation FAILED",
        ]
    )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run and validate canonical House historical backtests "
            "for 2016, 2018, 2020, and 2022."
        )
    )

    parser.add_argument(
        "--master-path",
        type=Path,
        default=CANONICAL_INPUT_PATH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=DEFAULT_BENCHMARK_DIR,
    )
    parser.add_argument(
        "--default-error-sd",
        type=float,
        default=6.5,
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=FLOAT_TOLERANCE,
        help=(
            "Maximum permitted absolute difference from the saved "
            "regression benchmark. Default: 1e-9."
        ),
    )
    parser.add_argument(
        "--write-benchmark",
        action="store_true",
        help=(
            "Explicitly replace the saved benchmark with the current "
            "validated outputs."
        ),
    )
    parser.add_argument(
        "--skip-benchmark-check",
        action="store_true",
        help=(
            "Run structural validations without comparing against a "
            "saved benchmark."
        ),
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help=(
            "Delete the multicycle output directory before running."
        ),
    )

    parser.add_argument(
        "--candidate-quality-weight",
        type=float,
        default=0.0,
        help=(
            "Multiplier applied to leakage-safe historical "
            "candidate WAR. Default: 0.0."
        ),
    )

    parser.add_argument(
        "--candidate-war-path",
        type=Path,
        default=(
            PROJECT_ROOT
            / "historical"
            / "house"
            / "backtests"
            / "outputs"
            / "candidate_war"
            / "house_historical_candidate_war.csv"
        ),
        help=(
            "Path to the leakage-safe historical candidate WAR "
            "warehouse."
        ),
    )

    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def main() -> None:
    args = parse_args()

    master_path = resolve_project_path(args.master_path)
    output_dir = resolve_project_path(args.output_dir)
    benchmark_dir = resolve_project_path(args.benchmark_dir)

    require_file(master_path, "canonical historical input")
    require_file(SINGLE_CYCLE_RUNNER, "single-cycle runner")

    if args.default_error_sd <= 0:
        raise ValueError("--default-error-sd must be positive.")

    if args.tolerance < 0:
        raise ValueError("--tolerance cannot be negative.")

    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    cycle_artifact_list: list[CycleArtifacts] = []

    for cycle in TARGET_CYCLES:
        cycle_artifact_list.append(
            run_cycle(
                cycle=cycle,
                master_path=master_path,
                output_root=output_dir,
                default_error_sd=args.default_error_sd,
                python_executable=sys.executable,
                candidate_quality_weight=args.candidate_quality_weight,
                candidate_war_path=args.candidate_war_path,
            )
        )

    summaries: list[pd.DataFrame] = []
    results_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, object]] = []

    for artifacts in cycle_artifact_list:
        summary, results, manifest_row = validate_cycle_outputs(
            artifacts
        )
        summaries.append(summary)
        results_frames.append(results)
        manifest_rows.append(manifest_row)

    combined_summary = (
        pd.concat(summaries, ignore_index=True)
        .sort_values("cycle", kind="mergesort")
        .reset_index(drop=True)
    )

    combined_results = (
        pd.concat(results_frames, ignore_index=True)
        .sort_values(
            ["cycle", "race_id"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    manifest = (
        pd.DataFrame(manifest_rows)
        .sort_values("cycle", kind="mergesort")
        .reset_index(drop=True)
    )

    validate_combined_outputs(
        combined_summary,
        combined_results,
    )

    summary_output_path = output_dir / SUMMARY_FILENAME
    results_output_path = output_dir / RESULTS_FILENAME
    manifest_output_path = output_dir / MANIFEST_FILENAME
    validation_output_path = output_dir / VALIDATION_FILENAME

    atomic_write_csv(combined_summary, summary_output_path)
    atomic_write_csv(combined_results, results_output_path)
    atomic_write_csv(manifest, manifest_output_path)

    benchmark_differences: list[str] = []

    if args.write_benchmark:
        write_benchmark(
            summary=combined_summary,
            results=combined_results,
            benchmark_dir=benchmark_dir,
            master_path=master_path,
            default_error_sd=args.default_error_sd,
        )
        benchmark_status = (
            f"WRITTEN to {benchmark_dir.relative_to(PROJECT_ROOT)}"
        )

    elif args.skip_benchmark_check:
        benchmark_status = "SKIPPED by request"

    else:
        benchmark_summary_path = (
            benchmark_dir / BENCHMARK_SUMMARY_FILENAME
        )
        benchmark_results_path = (
            benchmark_dir / BENCHMARK_RESULTS_FILENAME
        )

        if (
            benchmark_summary_path.exists()
            and benchmark_results_path.exists()
        ):
            benchmark_differences = compare_to_benchmark(
                current_summary=combined_summary,
                current_results=combined_results,
                benchmark_dir=benchmark_dir,
                tolerance=args.tolerance,
            )

            benchmark_status = (
                "MATCHED"
                if not benchmark_differences
                else "DIFFERENCES DETECTED"
            )
        else:
            benchmark_status = (
                "NOT FOUND; rerun with --write-benchmark after "
                "reviewing current outputs"
            )

    validation_text = build_validation_report(
        summary=combined_summary,
        manifest=manifest,
        benchmark_status=benchmark_status,
        benchmark_differences=benchmark_differences,
        master_path=master_path,
        output_dir=output_dir,
    )

    atomic_write_text(
        validation_text + "\n",
        validation_output_path,
    )

    log()
    log("=" * 72)
    log("Canonical House Multicycle Backtest")
    log("=" * 72)
    log()
    log(combined_summary.to_string(index=False))
    log()
    log("Cycle manifest:")
    log(manifest.to_string(index=False))
    log()
    log(f"Regression benchmark: {benchmark_status}")
    log()
    log(f"Wrote: {summary_output_path}")
    log(f"Wrote: {results_output_path}")
    log(f"Wrote: {manifest_output_path}")
    log(f"Wrote: {validation_output_path}")

    if benchmark_differences:
        log()
        log("Regression differences:")
        for difference in benchmark_differences:
            log(f"- {difference}")

        raise ValidationError(
            "Current outputs do not match the saved regression benchmark."
        )

    log()
    log("Validation PASSED")


if __name__ == "__main__":
    main()
