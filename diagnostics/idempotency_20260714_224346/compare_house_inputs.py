from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def choose_key(columns: list[str]) -> str:
    candidates = [
        "district_id",
        "district",
        "district_code",
        "race_id",
        "state_district",
    ]

    for candidate in candidates:
        if candidate in columns:
            return candidate

    raise RuntimeError(
        "Could not identify a district key. Available columns:\n"
        + "\n".join(columns)
    )


if len(sys.argv) != 3:
    raise SystemExit(
        "Usage: python3 compare_house_inputs.py BEFORE.csv AFTER.csv"
    )

before_path = Path(sys.argv[1])
after_path = Path(sys.argv[2])

before = pd.read_csv(before_path)
after = pd.read_csv(after_path)

before.columns = before.columns.astype(str)
after.columns = after.columns.astype(str)

common_columns = [
    column for column in before.columns
    if column in after.columns
]

key = choose_key(common_columns)

if list(before.columns) != list(after.columns):
    print("COLUMN STRUCTURE CHANGED")
    print("Added:", sorted(set(after.columns) - set(before.columns)))
    print("Removed:", sorted(set(before.columns) - set(after.columns)))

before[key] = before[key].astype(str).str.strip()
after[key] = after[key].astype(str).str.strip()

before = before.set_index(key).sort_index()
after = after.set_index(key).sort_index()

added_rows = after.index.difference(before.index)
removed_rows = before.index.difference(after.index)

print(f"Comparison: {before_path} -> {after_path}")
print(f"Key column: {key}")
print(f"Rows before: {len(before)}")
print(f"Rows after:  {len(after)}")
print(f"Added districts: {list(added_rows)}")
print(f"Removed districts: {list(removed_rows)}")

common_index = before.index.intersection(after.index)

comparison_columns = [
    column for column in common_columns
    if column != key
    and column in before.columns
    and column in after.columns
]

before = before.loc[common_index, comparison_columns]
after = after.loc[common_index, comparison_columns]

results = []
detail_frames = []

for column in comparison_columns:
    left = before[column]
    right = after[column]

    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")

    numeric_coverage = max(
        left_num.notna().mean(),
        right_num.notna().mean(),
    )

    if numeric_coverage >= 0.5:
        same_numeric = np.isclose(
            left_num.fillna(0.0),
            right_num.fillna(0.0),
            rtol=0.0,
            atol=1e-12,
        )

        same_missingness = (
            left_num.isna().to_numpy()
            == right_num.isna().to_numpy()
        )

        changed = ~(same_numeric & same_missingness)

        if changed.any():
            difference = right_num - left_num

            results.append(
                {
                    "column": column,
                    "type": "numeric",
                    "changed_rows": int(changed.sum()),
                    "mean_before": left_num.mean(),
                    "mean_after": right_num.mean(),
                    "mean_change": difference.mean(),
                    "min_change": difference.min(),
                    "max_change": difference.max(),
                    "sum_change": difference.sum(),
                }
            )

            detail = pd.DataFrame(
                {
                    key: before.index,
                    "column": column,
                    "before": left_num.to_numpy(),
                    "after": right_num.to_numpy(),
                    "change": difference.to_numpy(),
                }
            )

            detail_frames.append(detail.loc[changed].copy())

    else:
        left_text = left.fillna("<NA>").astype(str)
        right_text = right.fillna("<NA>").astype(str)
        changed = left_text.ne(right_text).to_numpy()

        if changed.any():
            results.append(
                {
                    "column": column,
                    "type": "text",
                    "changed_rows": int(changed.sum()),
                    "mean_before": np.nan,
                    "mean_after": np.nan,
                    "mean_change": np.nan,
                    "min_change": np.nan,
                    "max_change": np.nan,
                    "sum_change": np.nan,
                }
            )

            detail = pd.DataFrame(
                {
                    key: before.index,
                    "column": column,
                    "before": left_text.to_numpy(),
                    "after": right_text.to_numpy(),
                    "change": "",
                }
            )

            detail_frames.append(detail.loc[changed].copy())

summary = pd.DataFrame(results)

if summary.empty:
    print("\nRESULT: No values changed.")
    raise SystemExit(0)

summary = summary.sort_values(
    ["changed_rows", "column"],
    ascending=[False, True],
)

print("\nCHANGED COLUMNS")
print(summary.to_string(index=False))

if detail_frames:
    detailed = pd.concat(detail_frames, ignore_index=True)

    numeric_change = pd.to_numeric(
        detailed["change"],
        errors="coerce",
    )

    detailed["_absolute_change"] = numeric_change.abs()
    detailed = detailed.sort_values(
        ["_absolute_change", "column"],
        ascending=[False, True],
        na_position="last",
    ).drop(columns="_absolute_change")

    output_path = after_path.parent / (
        after_path.stem + "_detailed_changes.csv"
    )

    detailed.to_csv(output_path, index=False)

    print("\nDetailed changes written to:")
    print(output_path)

    print("\nLargest 60 changes:")
    print(detailed.head(60).to_string(index=False))
