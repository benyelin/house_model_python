from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def choose_key(columns: list[str]) -> str:
    candidates = [
        "district",
        "district_id",
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

if list(before.columns) != list(after.columns):
    print("COLUMN STRUCTURE CHANGED")
    print("Added:", sorted(set(after.columns) - set(before.columns)))
    print("Removed:", sorted(set(before.columns) - set(after.columns)))

common_columns = [c for c in before.columns if c in after.columns]
key = choose_key(common_columns)

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
before = before.loc[common_index, common_columns[1:]]
after = after.loc[common_index, common_columns[1:]]

results = []

for col in before.columns:
    left = before[col]
    right = after[col]

    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")

    numeric_coverage = max(
        left_num.notna().mean(),
        right_num.notna().mean(),
    )

    if numeric_coverage >= 0.5:
        diff = right_num - left_num
        changed = ~(
            np.isclose(
                left_num.fillna(0),
                right_num.fillna(0),
                rtol=0,
                atol=1e-12,
            )
            & (left_num.isna() == right_num.isna())
        )

        if changed.any():
            results.append(
                {
                    "column": col,
                    "type": "numeric",
                    "changed_rows": int(changed.sum()),
                    "mean_before": left_num.mean(),
                    "mean_after": right_num.mean(),
                    "mean_change": diff.mean(),
                    "min_change": diff.min(),
                    "max_change": diff.max(),
                    "sum_change": diff.sum(),
                }
            )
    else:
        left_text = left.fillna("<NA>").astype(str)
        right_text = right.fillna("<NA>").astype(str)
        changed = left_text != right_text

        if changed.any():
            results.append(
                {
                    "column": col,
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

keywords = (
    "margin",
    "war",
    "candidate",
    "quality",
    "fundamental",
    "baseline",
    "adjust",
    "environment",
    "poll",
)

suspect_columns = [
    col for col in summary["column"]
    if any(keyword in col.lower() for keyword in keywords)
]

if suspect_columns:
    detailed = []
    for col in suspect_columns:
        left = before[col]
        right = after[col]

        left_num = pd.to_numeric(left, errors="coerce")
        right_num = pd.to_numeric(right, errors="coerce")

        if max(left_num.notna().mean(), right_num.notna().mean()) >= 0.5:
            changed = ~(
                np.isclose(
                    left_num.fillna(0),
                    right_num.fillna(0),
                    rtol=0,
                    atol=1e-12,
                )
                & (left_num.isna() == right_num.isna())
            )

            temp = pd.DataFrame(
                {
                    key: before.index,
                    "column": col,
                    "before": left_num.values,
                    "after": right_num.values,
                    "change": (right_num - left_num).values,
                }
            )
            temp = temp.loc[changed.values]
            detailed.append(temp)

    if detailed:
        detail = pd.concat(detailed, ignore_index=True)
        detail = detail.sort_values(
            "change",
            key=lambda s: s.abs(),
            ascending=False,
        )

        output_path = after_path.parent / (
            after_path.stem + "_detailed_changes.csv"
        )
        detail.to_csv(output_path, index=False)

        print(f"\nDetailed suspect-field changes written to:")
        print(output_path)

        print("\nLargest 40 suspect-field changes:")
        print(detail.head(40).to_string(index=False))
