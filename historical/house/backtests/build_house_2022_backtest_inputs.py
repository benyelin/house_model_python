from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MASTER_PATH = (
    PROJECT_ROOT
    / "historical/house/master/house_2022_master.csv"
)

BASELINE_PATH = (
    PROJECT_ROOT
    / "historical/warehouse/processed/races/"
    "house_2022_presidential_baseline.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "historical/house/backtests/outputs/"
    "house_2022_backtest_inputs_provisional.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "historical/house/backtests/outputs/"
    "house_2022_backtest_inputs_provisional_validation.txt"
)


def main() -> None:
    master = pd.read_csv(
        MASTER_PATH,
        dtype={"race_id": str},
    )

    baseline = pd.read_csv(
        BASELINE_PATH,
        dtype={"race_id": str},
    )

    required_baseline = [
        "race_id",
        "district_pres_margin_dem",
        "boundary_compatibility",
        "include_in_2022_backtest",
        "presidential_result_year",
        "boundary_cycle",
    ]

    missing = [
        column
        for column in required_baseline
        if column not in baseline.columns
    ]

    if missing:
        raise ValueError(
            "Baseline file is missing required columns: "
            + ", ".join(missing)
        )

    baseline_keep = baseline[
        required_baseline
    ].copy()

    baseline_keep = baseline_keep.rename(
        columns={
            "include_in_2022_backtest": (
                "presidential_baseline_available"
            ),
        }
    )

    combined = master.merge(
        baseline_keep,
        on="race_id",
        how="left",
        validate="one_to_one",
    )

    if len(combined) != 435:
        raise ValueError(
            f"Expected 435 rows after merge; found {len(combined)}."
        )

    if combined["race_id"].duplicated().any():
        raise ValueError(
            "Duplicate race IDs created by baseline merge."
        )

    combined["district_pres_margin_dem"] = pd.to_numeric(
        combined["district_pres_margin_dem"],
        errors="coerce",
    )

    combined["presidential_baseline_available"] = (
        combined["district_pres_margin_dem"].notna()
    )

    # This field will later require a historically sourced value.
    # It is intentionally left missing rather than populated using
    # the current 2026 national environment.
    combined["national_environment_margin_dem"] = pd.NA

    # Historical feature fields are also left missing until they are
    # reconstructed without hindsight.
    combined["environment_multiplier"] = pd.NA
    combined["district_elasticity"] = pd.NA
    combined["state_environment_adjustment_dem"] = pd.NA
    combined["incumbency_adjustment_dem"] = pd.NA
    combined["candidate_quality_adjustment_dem"] = pd.NA
    combined["special_adjustment_dem"] = pd.NA
    combined["polling_adjustment_dem"] = pd.NA

    combined["historical_input_status"] = (
        "missing_presidential_baseline"
    )

    combined.loc[
        combined["presidential_baseline_available"],
        "historical_input_status",
    ] = "presidential_baseline_ready"

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    usable = int(
        combined["presidential_baseline_available"].sum()
    )

    withheld = int(
        (~combined["presidential_baseline_available"]).sum()
    )

    scoring_eligible = (
        combined[
            "general_election_party_structure"
        ].eq("D_vs_R")
        & combined["presidential_baseline_available"]
    )

    validation_lines = [
        "2022 House Provisional Backtest Input Validation",
        "=" * 48,
        "",
        f"Rows: {len(combined)}",
        f"Unique race IDs: {combined['race_id'].nunique()}",
        f"Presidential baselines available: {usable}",
        f"Presidential baselines withheld: {withheld}",
        (
            "D-vs-R races with presidential baseline: "
            f"{int(scoring_eligible.sum())}"
        ),
        "",
        "National environment status:",
        "Not yet populated.",
        "",
        "Other historical feature status:",
        (
            "Elasticity, incumbency adjustment, candidate quality, "
            "special adjustments, and polling adjustments remain "
            "intentionally missing until historically reconstructed."
        ),
        "",
        "No-hindsight safeguard:",
        (
            "No 2024 presidential results or current-cycle national "
            "environment values are included."
        ),
    ]

    validation_text = "\n".join(validation_lines)

    VALIDATION_PATH.write_text(validation_text)

    print(validation_text)
    print()
    print(f"Wrote: {OUTPUT_PATH}")
    print(f"Wrote: {VALIDATION_PATH}")


if __name__ == "__main__":
    main()
