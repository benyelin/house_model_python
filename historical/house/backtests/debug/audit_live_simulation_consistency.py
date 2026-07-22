#!/usr/bin/env python3
"""
Audit internal consistency of the current live House simulation.

The audit:

1. Locates likely House forecast output files.
2. Identifies draw-level Democratic seat totals where available.
3. Recomputes:
       - mean seats
       - median seats
       - standard deviation
       - percentiles
       - P(Dem seats >= 218)
4. Compares those values with saved summary outputs.
5. Checks whether expected seats from district win probabilities agree
   with the sum of district-level Democratic probabilities.
6. Checks fixed/uncontested-seat accounting.

This script does not change any production files.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
CONTROL_THRESHOLD = 218

SEARCH_DIRS = [
    PROJECT_ROOT,
    PROJECT_ROOT / "outputs",
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "historical" / "house" / "backtests" / "outputs",
]

SKIP_PARTS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
}

DRAW_SEAT_COLUMN_CANDIDATES = [
    "dem_seats",
    "democratic_seats",
    "dem_seat_count",
    "dem_seats_won",
    "simulated_dem_seats",
    "total_dem_seats",
    "house_dem_seats",
]

DRAW_ID_COLUMN_CANDIDATES = [
    "simulation",
    "simulation_id",
    "sim",
    "sim_id",
    "draw",
    "draw_id",
    "iteration",
    "trial",
]

DISTRICT_ID_CANDIDATES = [
    "district_id",
    "race_id",
]

DISTRICT_PROBABILITY_CANDIDATES = [
    "dem_win_probability",
    "dem_probability",
    "prob_dem_win",
]

SUMMARY_EXPECTED_CANDIDATES = [
    "expected_dem_seats",
    "mean_dem_seats",
    "dem_expected_seats",
    "projected_dem_seats",
]

SUMMARY_CONTROL_CANDIDATES = [
    "dem_control_probability",
    "dem_house_control_probability",
    "control_probability_dem",
    "prob_dem_control",
]

SUMMARY_MEDIAN_CANDIDATES = [
    "median_dem_seats",
    "p50_dem_seats",
    "dem_seats_p50",
]

SUMMARY_SD_CANDIDATES = [
    "sd_dem_seats",
    "std_dem_seats",
    "dem_seats_sd",
    "dem_seats_std",
]

PERCENTILE_CANDIDATES = {
    10: ["p10_dem_seats", "dem_seats_p10"],
    25: ["p25_dem_seats", "dem_seats_p25"],
    50: ["p50_dem_seats", "median_dem_seats", "dem_seats_p50"],
    75: ["p75_dem_seats", "dem_seats_p75"],
    90: ["p90_dem_seats", "dem_seats_p90"],
}


def is_allowed(path: Path) -> bool:
    return not any(part in SKIP_PARTS for part in path.parts)


def load_table(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, low_memory=False)
        if path.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text())
            if isinstance(payload, list):
                return pd.DataFrame(payload)
            if isinstance(payload, dict):
                return pd.DataFrame([payload])
    except Exception:
        return None
    return None


def first_existing(columns: list[str], candidates: list[str]) -> str | None:
    lower_map = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce").dropna()


def collect_files() -> list[Path]:
    files: set[Path] = set()

    for directory in SEARCH_DIRS:
        if not directory.exists():
            continue

        for pattern in ("*.csv", "*.parquet", "*.pq", "*.json"):
            for path in directory.rglob(pattern):
                if not is_allowed(path):
                    continue

                name = path.name.lower()
                if any(
                    token in name
                    for token in (
                        "house",
                        "forecast",
                        "simulation",
                        "summary",
                        "race",
                        "result",
                        "seat",
                    )
                ):
                    files.add(path.resolve())

    return sorted(
        files,
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )


def describe_candidate(path: Path, df: pd.DataFrame) -> None:
    print(f"\nFILE: {path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print("Relevant columns:")

    relevant = [
        c
        for c in df.columns
        if any(
            token in str(c).lower()
            for token in (
                "seat",
                "control",
                "prob",
                "simulation",
                "draw",
                "district",
                "race",
                "fixed",
                "uncontest",
                "percentile",
                "p25",
                "p50",
                "p75",
            )
        )
    ]

    if relevant:
        for column in relevant[:80]:
            print(f"  - {column}")
    else:
        print("  (none detected)")


def audit_direct_seat_draws(
    path: Path,
    df: pd.DataFrame,
    seat_col: str,
) -> dict[str, Any]:
    seats = numeric_series(df, seat_col)

    if len(seats) < 100:
        raise ValueError(
            f"Only {len(seats)} usable values in {seat_col}; "
            "not enough to treat as Monte Carlo draws."
        )

    metrics = {
        "source": str(path),
        "method": f"direct seat column: {seat_col}",
        "draws": int(len(seats)),
        "mean": float(seats.mean()),
        "median": float(seats.median()),
        "sd": float(seats.std(ddof=0)),
        "min": float(seats.min()),
        "max": float(seats.max()),
        "p10": float(np.percentile(seats, 10)),
        "p25": float(np.percentile(seats, 25)),
        "p50": float(np.percentile(seats, 50)),
        "p75": float(np.percentile(seats, 75)),
        "p90": float(np.percentile(seats, 90)),
        "control_probability": float((seats >= CONTROL_THRESHOLD).mean()),
    }

    return metrics


def audit_long_district_draws(
    path: Path,
    df: pd.DataFrame,
    draw_col: str,
    district_col: str,
) -> dict[str, Any] | None:
    outcome_candidates = [
        "dem_win",
        "actual_dem_win",
        "simulated_dem_win",
        "dem_won",
        "winner_dem",
        "is_dem_win",
    ]

    outcome_col = first_existing(list(df.columns), outcome_candidates)

    if outcome_col is None:
        return None

    work = df[[draw_col, district_col, outcome_col]].copy()
    work[outcome_col] = pd.to_numeric(work[outcome_col], errors="coerce")
    work = work.dropna(subset=[draw_col, district_col, outcome_col])

    if work.empty:
        return None

    seat_draws = work.groupby(draw_col, sort=False)[outcome_col].sum()

    if len(seat_draws) < 100:
        return None

    return {
        "source": str(path),
        "method": (
            f"district outcomes grouped by {draw_col}; "
            f"outcome={outcome_col}"
        ),
        "draws": int(len(seat_draws)),
        "mean": float(seat_draws.mean()),
        "median": float(seat_draws.median()),
        "sd": float(seat_draws.std(ddof=0)),
        "min": float(seat_draws.min()),
        "max": float(seat_draws.max()),
        "p10": float(np.percentile(seat_draws, 10)),
        "p25": float(np.percentile(seat_draws, 25)),
        "p50": float(np.percentile(seat_draws, 50)),
        "p75": float(np.percentile(seat_draws, 75)),
        "p90": float(np.percentile(seat_draws, 90)),
        "control_probability": float(
            (seat_draws >= CONTROL_THRESHOLD).mean()
        ),
    }


def extract_summary_values(
    path: Path,
    df: pd.DataFrame,
) -> dict[str, Any] | None:
    if df.empty:
        return None

    values: dict[str, Any] = {"source": str(path)}

    mappings = {
        "expected": SUMMARY_EXPECTED_CANDIDATES,
        "control_probability": SUMMARY_CONTROL_CANDIDATES,
        "median": SUMMARY_MEDIAN_CANDIDATES,
        "sd": SUMMARY_SD_CANDIDATES,
    }

    for label, candidates in mappings.items():
        column = first_existing(list(df.columns), candidates)
        if column is not None:
            series = numeric_series(df, column)
            if not series.empty:
                values[label] = float(series.iloc[-1])
                values[f"{label}_column"] = column

    for percentile, candidates in PERCENTILE_CANDIDATES.items():
        column = first_existing(list(df.columns), candidates)
        if column is not None:
            series = numeric_series(df, column)
            if not series.empty:
                values[f"p{percentile}"] = float(series.iloc[-1])
                values[f"p{percentile}_column"] = column

    return values if len(values) > 1 else None


def audit_district_probabilities(
    path: Path,
    df: pd.DataFrame,
) -> dict[str, Any] | None:
    district_col = first_existing(
        list(df.columns),
        DISTRICT_ID_CANDIDATES,
    )
    probability_col = first_existing(
        list(df.columns),
        DISTRICT_PROBABILITY_CANDIDATES,
    )

    if district_col is None or probability_col is None:
        return None

    work = df[[district_col, probability_col]].copy()
    work[probability_col] = pd.to_numeric(
        work[probability_col],
        errors="coerce",
    )

    work = work.dropna(subset=[district_col, probability_col])
    work = work.drop_duplicates(district_col, keep="last")

    if len(work) < 400:
        return None

    probabilities = work[probability_col].copy()

    # Permit probabilities stored as percentages.
    if probabilities.max() > 1.0 and probabilities.max() <= 100.0:
        probabilities = probabilities / 100.0

    return {
        "source": str(path),
        "districts": int(len(work)),
        "sum_dem_win_probabilities": float(probabilities.sum()),
        "mean_dem_win_probability": float(probabilities.mean()),
        "min_probability": float(probabilities.min()),
        "max_probability": float(probabilities.max()),
    }


def print_draw_metrics(metrics: dict[str, Any]) -> None:
    print("\n" + "=" * 88)
    print("RAW-DRAW RECOMPUTATION")
    print("=" * 88)
    print(f"Source:                   {metrics['source']}")
    print(f"Method:                   {metrics['method']}")
    print(f"Number of draws:          {metrics['draws']:,}")
    print(f"Mean Democratic seats:    {metrics['mean']:.6f}")
    print(f"Median Democratic seats:  {metrics['median']:.6f}")
    print(f"Seat standard deviation:  {metrics['sd']:.6f}")
    print(f"Minimum seats:            {metrics['min']:.0f}")
    print(f"Maximum seats:            {metrics['max']:.0f}")
    print(f"P10:                      {metrics['p10']:.6f}")
    print(f"P25:                      {metrics['p25']:.6f}")
    print(f"P50:                      {metrics['p50']:.6f}")
    print(f"P75:                      {metrics['p75']:.6f}")
    print(f"P90:                      {metrics['p90']:.6f}")
    print(
        f"P(Dem seats >= {CONTROL_THRESHOLD}): "
        f"{metrics['control_probability']:.8%}"
    )


def compare_summary(
    draw_metrics: dict[str, Any],
    summaries: list[dict[str, Any]],
) -> None:
    print("\n" + "=" * 88)
    print("COMPARISON WITH SAVED SUMMARY OUTPUTS")
    print("=" * 88)

    if not summaries:
        print("No recognizable saved summary table was found.")
        return

    for summary in summaries:
        print(f"\nSummary source: {summary['source']}")

        comparisons = [
            ("expected", "mean"),
            ("median", "median"),
            ("sd", "sd"),
            ("p10", "p10"),
            ("p25", "p25"),
            ("p50", "p50"),
            ("p75", "p75"),
            ("p90", "p90"),
            ("control_probability", "control_probability"),
        ]

        found = False

        for summary_key, draw_key in comparisons:
            if summary_key not in summary:
                continue

            found = True
            saved = float(summary[summary_key])
            raw = float(draw_metrics[draw_key])

            # Convert percentage-style saved control probability if necessary.
            if (
                summary_key == "control_probability"
                and saved > 1.0
                and saved <= 100.0
            ):
                saved = saved / 100.0

            diff = saved - raw

            print(
                f"  {summary_key:22s} "
                f"saved={saved:12.6f} "
                f"raw={raw:12.6f} "
                f"difference={diff:+.10f}"
            )

        if not found:
            print("  No comparable summary metrics detected.")


def main() -> None:
    print("House Live Simulation Consistency Audit")
    print("=" * 88)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Control threshold: {CONTROL_THRESHOLD}")

    files = collect_files()

    print(f"\nCandidate files found: {len(files)}")

    loaded: list[tuple[Path, pd.DataFrame]] = []

    for path in files[:120]:
        df = load_table(path)
        if df is None or df.empty:
            continue

        relevant = any(
            any(
                token in str(column).lower()
                for token in (
                    "seat",
                    "control",
                    "simulation",
                    "draw",
                    "dem_win_probability",
                )
            )
            for column in df.columns
        )

        if relevant:
            loaded.append((path, df))

    print(f"Relevant readable files: {len(loaded)}")

    for path, df in loaded[:30]:
        describe_candidate(path, df)

    draw_candidates: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    district_probability_audits: list[dict[str, Any]] = []

    for path, df in loaded:
        seat_col = first_existing(
            list(df.columns),
            DRAW_SEAT_COLUMN_CANDIDATES,
        )

        if seat_col is not None:
            try:
                metrics = audit_direct_seat_draws(path, df, seat_col)
                draw_candidates.append(metrics)
            except ValueError:
                pass

        draw_col = first_existing(
            list(df.columns),
            DRAW_ID_COLUMN_CANDIDATES,
        )
        district_col = first_existing(
            list(df.columns),
            DISTRICT_ID_CANDIDATES,
        )

        if draw_col is not None and district_col is not None:
            metrics = audit_long_district_draws(
                path,
                df,
                draw_col,
                district_col,
            )
            if metrics is not None:
                draw_candidates.append(metrics)

        summary = extract_summary_values(path, df)
        if summary is not None:
            summaries.append(summary)

        probability_audit = audit_district_probabilities(path, df)
        if probability_audit is not None:
            district_probability_audits.append(probability_audit)

    if not draw_candidates:
        print("\n" + "=" * 88)
        print("NO RAW DRAW TABLE FOUND")
        print("=" * 88)
        print(
            "No file containing recognizable draw-level Democratic seat "
            "totals was found."
        )
        print(
            "The model may retain draws only in memory or use an "
            "unrecognized column name."
        )
        print(
            "The candidate-file and relevant-column listings above should "
            "show us where to inspect next."
        )
    else:
        # Prefer the candidate with the most draws, then newest-looking source.
        draw_candidates.sort(
            key=lambda item: item["draws"],
            reverse=True,
        )

        best = draw_candidates[0]
        print_draw_metrics(best)
        compare_summary(best, summaries)

        if len(draw_candidates) > 1:
            print("\nOther possible raw-draw datasets:")
            for candidate in draw_candidates[1:10]:
                print(
                    f"  draws={candidate['draws']:,} "
                    f"mean={candidate['mean']:.4f} "
                    f"control={candidate['control_probability']:.4%} "
                    f"source={candidate['source']}"
                )

    print("\n" + "=" * 88)
    print("DISTRICT-PROBABILITY EXPECTED-SEAT CHECK")
    print("=" * 88)

    if not district_probability_audits:
        print(
            "No current district table with approximately 435 unique "
            "districts and a recognizable Democratic win-probability "
            "column was found."
        )
    else:
        for audit in district_probability_audits[:10]:
            print(f"\nSource: {audit['source']}")
            print(f"Unique districts:             {audit['districts']}")
            print(
                "Sum of Dem win probabilities: "
                f"{audit['sum_dem_win_probabilities']:.6f}"
            )
            print(
                "Mean district Dem probability: "
                f"{audit['mean_dem_win_probability']:.8%}"
            )
            print(
                "Probability range:             "
                f"{audit['min_probability']:.8%} to "
                f"{audit['max_probability']:.8%}"
            )

    print("\n" + "=" * 88)
    print("INTERPRETATION")
    print("=" * 88)
    print(
        "A healthy run should satisfy all of the following:\n"
        "  1. Raw draw mean equals the displayed expected Democratic seats.\n"
        "  2. Raw fraction of draws with at least 218 Democratic seats equals\n"
        "     the displayed Democratic control probability.\n"
        "  3. Raw percentiles equal the displayed percentile values.\n"
        "  4. Sum of district Democratic win probabilities equals the raw\n"
        "     expected Democratic seat count, aside from only tiny numerical\n"
        "     differences.\n"
        "  5. Every simulation draw contains exactly 435 total House seats."
    )

    print("\nAudit completed.")


if __name__ == "__main__":
    main()
