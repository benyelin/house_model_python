from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".")
OUTPUT = Path(
    "diagnostics/idempotency_20260714_224346/"
    "candidate_quality_backup_audit.csv"
)

patterns = [
    "*house_race_inputs*.csv",
    "*race_inputs*house*.csv",
]

paths: set[Path] = set()

for pattern in patterns:
    paths.update(ROOT.rglob(pattern))

rows = []

for path in sorted(paths):
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        rows.append(
            {
                "path": str(path),
                "readable": False,
                "error": str(exc),
            }
        )
        continue

    quality_col = None

    for candidate in [
        "candidate_quality_adjustment_dem",
        "objective_candidate_quality_adjustment_dem",
    ]:
        if candidate in df.columns:
            quality_col = candidate
            break

    if quality_col is None:
        rows.append(
            {
                "path": str(path),
                "readable": True,
                "rows": len(df),
                "quality_column": "",
                "error": "No candidate-quality column",
            }
        )
        continue

    quality = pd.to_numeric(df[quality_col], errors="coerce")

    before_war_col = None

    for candidate in [
        "candidate_quality_adjustment_dem_before_war",
        "objective_candidate_quality_adjustment_dem_before_war",
    ]:
        if candidate in df.columns:
            before_war_col = candidate
            break

    before_war = (
        pd.to_numeric(df[before_war_col], errors="coerce")
        if before_war_col
        else pd.Series(np.nan, index=df.index)
    )

    war = (
        pd.to_numeric(
            df["candidate_war_adjustment_dem"],
            errors="coerce",
        )
        if "candidate_war_adjustment_dem" in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    rows.append(
        {
            "path": str(path),
            "modified_time": path.stat().st_mtime,
            "readable": True,
            "rows": len(df),
            "columns": len(df.columns),
            "quality_column": quality_col,
            "before_war_column": before_war_col or "",
            "quality_nonzero": int(quality.fillna(0).ne(0).sum()),
            "quality_mean": quality.mean(),
            "quality_mean_abs": quality.abs().mean(),
            "quality_max_abs": quality.abs().max(),
            "quality_gt_3_abs": int(quality.abs().gt(3).sum()),
            "quality_gt_5_abs": int(quality.abs().gt(5).sum()),
            "quality_gt_10_abs": int(quality.abs().gt(10).sum()),
            "before_war_mean_abs": before_war.abs().mean(),
            "before_war_max_abs": before_war.abs().max(),
            "war_mean_abs": war.abs().mean(),
            "war_max_abs": war.abs().max(),
            "error": "",
        }
    )

audit = pd.DataFrame(rows)

if not audit.empty:
    audit = audit.sort_values(
        ["readable", "quality_max_abs", "path"],
        ascending=[False, True, True],
        na_position="last",
    )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
audit.to_csv(OUTPUT, index=False)

show = [
    "path",
    "rows",
    "quality_column",
    "before_war_column",
    "quality_nonzero",
    "quality_mean_abs",
    "quality_max_abs",
    "quality_gt_3_abs",
    "quality_gt_5_abs",
    "quality_gt_10_abs",
]

show = [column for column in show if column in audit.columns]

print(audit[show].to_string(index=False))
print()
print(f"Full audit written to: {OUTPUT}")
