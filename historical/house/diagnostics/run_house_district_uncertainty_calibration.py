#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

PREDICTIONS = (
    ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "full_production_replay"
    / "house_production_replay_predictions.csv"
)

OUTPUT = (
    ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "district_uncertainty_calibration"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PREDICTIONS)

df = df.loc[
    df["replay_spec"] == "production_election_day_v1"
].copy()

df = df.loc[
    df["include_in_scoring"] == True
].copy()

required = [
    "actual_dem_margin",
    "model_margin_dem",
    "total_error_sd",
]

df = df.dropna(subset=required).copy()

df["residual"] = (
    df["actual_dem_margin"]
    - df["model_margin_dem"]
)

df["z"] = (
    df["residual"]
    / df["total_error_sd"]
)

summary = []

summary.append(
    {
        "metric": "districts",
        "value": len(df),
    }
)

summary.append(
    {
        "metric": "mean_residual",
        "value": df["residual"].mean(),
    }
)

summary.append(
    {
        "metric": "rmse_margin",
        "value": np.sqrt(
            np.mean(
                df["residual"] ** 2
            )
        ),
    }
)

summary.append(
    {
        "metric": "mean_z",
        "value": df["z"].mean(),
    }
)

summary.append(
    {
        "metric": "sd_z",
        "value": df["z"].std(ddof=1),
    }
)

for cutoff, label in [
    (0.67449, "50"),
    (1.28155, "80"),
    (1.64485, "90"),
    (1.95996, "95"),
]:
    empirical = np.mean(
        np.abs(df["z"]) <= cutoff
    )

    summary.append(
        {
            "metric": f"coverage_{label}",
            "value": empirical,
        }
    )

summary = pd.DataFrame(summary)

summary.to_csv(
    OUTPUT / "district_uncertainty_summary.csv",
    index=False,
)

bins = [
    -np.inf,
    -3,
    -2,
    -1,
    0,
    1,
    2,
    3,
    np.inf,
]

hist = (
    pd.cut(df["z"], bins=bins)
    .value_counts(sort=False)
    .rename_axis("bin")
    .reset_index(name="count")
)

hist.to_csv(
    OUTPUT / "district_z_histogram.csv",
    index=False,
)

print(summary)

print()
print("Histogram")
print(hist)

print()
print("Wrote:")
print(OUTPUT / "district_uncertainty_summary.csv")
print(OUTPUT / "district_z_histogram.csv")

# ---------------------------------------------------------------------
# Time-to-election uncertainty audit
# ---------------------------------------------------------------------

time_columns = [
    "cycle",
    "race_id",
    "uncertainty_days_out",
    "total_error_sd",
    "national_error_sd",
    "region_error_sd",
    "demographic_error_sd",
    "district_error_sd",
    "effective_district_error_sd",
    "poll_count",
]

available_time_columns = [
    column for column in time_columns
    if column in df.columns
]

time_audit = df[available_time_columns].copy()

numeric_time_columns = [
    "uncertainty_days_out",
    "total_error_sd",
    "national_error_sd",
    "region_error_sd",
    "demographic_error_sd",
    "district_error_sd",
    "effective_district_error_sd",
    "poll_count",
]

for column in numeric_time_columns:
    if column in time_audit.columns:
        time_audit[column] = pd.to_numeric(
            time_audit[column],
            errors="coerce",
        )

print()
print("=" * 72)
print("TIME-TO-ELECTION UNCERTAINTY AUDIT")
print("=" * 72)

if (
    "uncertainty_days_out" not in time_audit.columns
    or time_audit["uncertainty_days_out"].notna().sum() == 0
):
    print("No usable uncertainty_days_out values were found.")
else:
    print("\nDays-out values:")
    print(
        time_audit["uncertainty_days_out"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    aggregation = {
        column: ["count", "mean", "min", "max"]
        for column in [
            "total_error_sd",
            "national_error_sd",
            "region_error_sd",
            "demographic_error_sd",
            "district_error_sd",
            "effective_district_error_sd",
        ]
        if column in time_audit.columns
    }

    by_days_out = (
        time_audit
        .groupby("uncertainty_days_out", dropna=False)
        .agg(aggregation)
        .reset_index()
    )

    by_days_out.columns = [
        (
            column
            if not statistic
            else f"{column}_{statistic}"
        )
        for column, statistic in by_days_out.columns
    ]

    by_days_out.to_csv(
        OUTPUT / "uncertainty_by_days_out.csv",
        index=False,
    )

    print("\nUncertainty by days out:")
    print(by_days_out.to_string(index=False))

    correlation_columns = [
        column
        for column in [
            "uncertainty_days_out",
            "total_error_sd",
            "national_error_sd",
            "region_error_sd",
            "demographic_error_sd",
            "district_error_sd",
            "effective_district_error_sd",
        ]
        if column in time_audit.columns
    ]

    correlations = (
        time_audit[correlation_columns]
        .corr(numeric_only=True)
        ["uncertainty_days_out"]
        .rename("correlation_with_days_out")
        .reset_index()
        .rename(columns={"index": "metric"})
    )

    correlations.to_csv(
        OUTPUT / "uncertainty_days_out_correlations.csv",
        index=False,
    )

    print("\nCorrelation with days out:")
    print(correlations.to_string(index=False))

    print()
    print("Interpretation:")
    print(
        "A positive correlation between days out and an SD means "
        "that uncertainty decreases as Election Day approaches."
    )

    print("\nWrote:")
    print(OUTPUT / "uncertainty_by_days_out.csv")
    print(OUTPUT / "uncertainty_days_out_correlations.csv")
