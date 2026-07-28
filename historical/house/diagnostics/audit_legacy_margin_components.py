#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

INPUT = Path(
    "historical/house/backtests/outputs/"
    "full_production_replay/"
    "house_production_replay_predictions.csv"
)

OUTPUT = Path(
    "historical/house/backtests/outputs/"
    "legacy_margin_audit"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT, low_memory=False)

# Restrict to legacy vs production fundamentals
df = df[
    df["replay_spec"].isin(
        [
            "legacy_fixed_6_5",
            "production_fundamentals_fixed_6_5",
        ]
    )
].copy()

index_cols = [
    "cycle",
    "race_id",
    "district_id",
]

pivot = (
    df.set_index(index_cols + ["replay_spec"])
      .unstack("replay_spec")
)

def col(name, spec):
    return pivot[(name, spec)]

out = pd.DataFrame(index=pivot.index)

out["legacy_margin"] = col(
    "model_margin_dem",
    "legacy_fixed_6_5",
)

out["production_margin"] = col(
    "model_margin_dem",
    "production_fundamentals_fixed_6_5",
)

out["margin_difference"] = (
    out["production_margin"]
    - out["legacy_margin"]
)

for variable in [
    "district_partisan_baseline_dem",
    "district_environment_adjustment_dem",
    "incumbency_adjustment_dem",
    "generic_ballot_contribution_dem",
    "approval_contribution_dem",
    "midterm_contribution_dem",
    "candidate_quality_adjustment_dem",
    "special_adjustment_dem",
]:
    if (variable, "legacy_fixed_6_5") in pivot.columns:
        out[f"legacy_{variable}"] = col(
            variable,
            "legacy_fixed_6_5",
        )

    if (
        variable,
        "production_fundamentals_fixed_6_5",
    ) in pivot.columns:
        out[f"production_{variable}"] = col(
            variable,
            "production_fundamentals_fixed_6_5",
        )

summary = (
    out.reset_index()
       .groupby("cycle")
       .agg(
           mean_margin_difference=(
               "margin_difference",
               "mean",
           ),
           median_margin_difference=(
               "margin_difference",
               "median",
           ),
           min_difference=(
               "margin_difference",
               "min",
           ),
           max_difference=(
               "margin_difference",
               "max",
           ),
       )
)

out.reset_index().to_csv(
    OUTPUT / "district_margin_audit.csv",
    index=False,
)

summary.to_csv(
    OUTPUT / "cycle_summary.csv"
)

print()
print(summary)

print()
print("Largest Republican shifts:")
print(
    out.reset_index()
       .sort_values(
           "margin_difference"
       )
       .head(20)
       [
           [
               "cycle",
               "race_id",
               "margin_difference",
               "legacy_margin",
               "production_margin",
           ]
       ]
)

print()
print("Largest Democratic shifts:")
print(
    out.reset_index()
       .sort_values(
           "margin_difference",
           ascending=False,
       )
       .head(20)
       [
           [
               "cycle",
               "race_id",
               "margin_difference",
               "legacy_margin",
               "production_margin",
           ]
       ]
)
