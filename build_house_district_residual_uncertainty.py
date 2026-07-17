#!/usr/bin/env python3
"""
Build production House district-specific residual uncertainty multipliers.

The selected parameters come from the leakage-safe multicycle sensitivity test:

    shrinkage strength: 4.0
    multiplier floor:  0.80
    multiplier ceiling: 1.20

The multiplier scales only the district-specific error component in the live
correlated simulation. It does not alter point-margin forecasts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

INPUT_PATH = (
    ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "canonical"
    / "house_multicycle_backtest_results.csv"
)

RACE_INPUTS_PATH = ROOT / "inputs" / "house_race_inputs.csv"

OUTPUT_PATH = (
    ROOT
    / "outputs"
    / "house_district_residual_uncertainty.csv"
)

SHRINKAGE_STRENGTH = 4.0
MULTIPLIER_FLOOR = 0.80
MULTIPLIER_CEILING = 1.20
EPSILON = 1e-9


def parse_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def normalize_district_id(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
    )


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing canonical historical results: {INPUT_PATH}"
        )

    if not RACE_INPUTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing current House race inputs: {RACE_INPUTS_PATH}"
        )

    historical = pd.read_csv(INPUT_PATH)

    required = [
        "district_id",
        "margin_error",
        "include_in_scoring",
    ]

    missing = [
        column for column in required
        if column not in historical.columns
    ]

    if missing:
        raise ValueError(
            f"Historical results missing required columns: {missing}"
        )

    historical = historical.loc[
        parse_bool(historical["include_in_scoring"])
    ].copy()

    historical["district_id"] = normalize_district_id(
        historical["district_id"]
    )

    historical["margin_error"] = pd.to_numeric(
        historical["margin_error"],
        errors="coerce",
    )

    historical = historical.dropna(
        subset=["district_id", "margin_error"]
    ).copy()

    pooled_variance = float(
        np.mean(
            np.square(
                historical["margin_error"].to_numpy(dtype=float)
            )
        )
    )

    pooled_rmse = float(np.sqrt(pooled_variance))

    district = (
        historical.groupby("district_id", as_index=False)
        .agg(
            residual_observations=("margin_error", "size"),
            raw_residual_mse=(
                "margin_error",
                lambda values: float(
                    np.mean(
                        np.square(
                            values.to_numpy(dtype=float)
                        )
                    )
                ),
            ),
            residual_mean=("margin_error", "mean"),
        )
    )

    district["raw_residual_rmse"] = np.sqrt(
        district["raw_residual_mse"]
    )

    district["reliability"] = (
        district["residual_observations"]
        / (
            district["residual_observations"]
            + SHRINKAGE_STRENGTH
        )
    )

    district["shrunk_residual_variance"] = (
        district["reliability"]
        * district["raw_residual_mse"]
        + (
            1.0 - district["reliability"]
        )
        * pooled_variance
    )

    district["shrunk_residual_rmse"] = np.sqrt(
        district["shrunk_residual_variance"]
    )

    district["raw_district_uncertainty_multiplier"] = (
        district["shrunk_residual_rmse"]
        / max(pooled_rmse, EPSILON)
    )

    district["district_uncertainty_multiplier"] = (
        district["raw_district_uncertainty_multiplier"]
        .clip(
            lower=MULTIPLIER_FLOOR,
            upper=MULTIPLIER_CEILING,
        )
    )

    current = pd.read_csv(RACE_INPUTS_PATH)

    if "district_id" not in current.columns:
        raise ValueError(
            "house_race_inputs.csv must include district_id."
        )

    current_districts = pd.DataFrame(
        {
            "district_id": normalize_district_id(
                current["district_id"]
            )
        }
    ).drop_duplicates()

    output = current_districts.merge(
        district,
        on="district_id",
        how="left",
    )

    output["residual_observations"] = (
        output["residual_observations"]
        .fillna(0)
        .astype(int)
    )

    output["raw_residual_mse"] = output[
        "raw_residual_mse"
    ].fillna(pooled_variance)

    output["raw_residual_rmse"] = output[
        "raw_residual_rmse"
    ].fillna(pooled_rmse)

    output["residual_mean"] = output[
        "residual_mean"
    ].fillna(0.0)

    output["reliability"] = output[
        "reliability"
    ].fillna(0.0)

    output["shrunk_residual_variance"] = output[
        "shrunk_residual_variance"
    ].fillna(pooled_variance)

    output["shrunk_residual_rmse"] = output[
        "shrunk_residual_rmse"
    ].fillna(pooled_rmse)

    output["raw_district_uncertainty_multiplier"] = output[
        "raw_district_uncertainty_multiplier"
    ].fillna(1.0)

    output["district_uncertainty_multiplier"] = output[
        "district_uncertainty_multiplier"
    ].fillna(1.0)

    output["pooled_residual_rmse"] = pooled_rmse
    output["shrinkage_strength"] = SHRINKAGE_STRENGTH
    output["multiplier_floor"] = MULTIPLIER_FLOOR
    output["multiplier_ceiling"] = MULTIPLIER_CEILING

    output["uncertainty_source"] = np.where(
        output["residual_observations"] > 0,
        "Historical canonical backtest residuals",
        "Pooled fallback",
    )

    output = output.sort_values(
        "district_id"
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "Built House district residual uncertainty:"
        f" {OUTPUT_PATH}"
    )
    print(f"Rows: {len(output)}")
    print(f"Pooled residual RMSE: {pooled_rmse:.6f}")
    print()

    print("Observation counts:")
    print(
        output["residual_observations"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nMultiplier summary:")
    print(
        output["district_uncertainty_multiplier"]
        .describe(
            percentiles=[
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
            ]
        )
        .to_string()
    )

    print("\nHighest uncertainty districts:")
    columns = [
        "district_id",
        "residual_observations",
        "raw_residual_rmse",
        "shrunk_residual_rmse",
        "district_uncertainty_multiplier",
    ]

    print(
        output.sort_values(
            "district_uncertainty_multiplier",
            ascending=False,
        )[columns]
        .head(20)
        .to_string(index=False)
    )

    print("\nLowest uncertainty districts:")
    print(
        output.sort_values(
            "district_uncertainty_multiplier",
            ascending=True,
        )[columns]
        .head(20)
        .to_string(index=False)
    )

    if len(output) != 435:
        raise SystemExit(
            f"FAILED: expected 435 districts, found {len(output)}."
        )

    if output[
        "district_uncertainty_multiplier"
    ].isna().any():
        raise SystemExit(
            "FAILED: missing district uncertainty multipliers."
        )

    if not output[
        "district_uncertainty_multiplier"
    ].between(
        MULTIPLIER_FLOOR,
        MULTIPLIER_CEILING,
    ).all():
        raise SystemExit(
            "FAILED: multiplier outside selected bounds."
        )

    print(
        "\nPASSED: production district uncertainty "
        "multipliers are complete and bounded."
    )


if __name__ == "__main__":
    main()
