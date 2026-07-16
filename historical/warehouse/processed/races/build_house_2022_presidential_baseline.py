from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]

CURRENT_INPUTS_PATH = (
    PROJECT_ROOT / "inputs/house_race_inputs.csv"
)

HISTORICAL_MASTER_PATH = (
    PROJECT_ROOT
    / "historical/house/master/house_2022_master.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "historical/warehouse/processed/races/"
    "house_2022_presidential_baseline_provisional.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "historical/warehouse/validation/"
    "house_2022_presidential_baseline_provisional_validation.txt"
)


# These states used different congressional boundaries in 2024
# than in the 2022 House election.
REMAPPED_AFTER_2022 = {
    "AL",
    "GA",
    "LA",
    "NC",
    "NY",
}


def main() -> None:
    current = pd.read_csv(
        CURRENT_INPUTS_PATH,
        dtype={"district_id": str},
    )

    master = pd.read_csv(
        HISTORICAL_MASTER_PATH,
        dtype={"race_id": str},
    )

    required_current = [
        "district_id",
        "pres_2020_margin_dem",
    ]

    missing = [
        column
        for column in required_current
        if column not in current.columns
    ]

    if missing:
        raise ValueError(
            "Current House inputs are missing required columns: "
            + ", ".join(missing)
        )

    if len(current) != 435:
        raise ValueError(
            f"Expected 435 current input rows; found {len(current)}."
        )

    if len(master) != 435:
        raise ValueError(
            f"Expected 435 historical master rows; found {len(master)}."
        )

    baseline = current[
        [
            "district_id",
            "pres_2020_margin_dem",
        ]
    ].copy()

    baseline = baseline.rename(
        columns={
            "district_id": "race_id",
            "pres_2020_margin_dem": (
                "district_pres_margin_dem"
            ),
        }
    )

    baseline["state"] = (
        baseline["race_id"]
        .astype(str)
        .str.split("-")
        .str[0]
    )

    baseline["boundary_compatibility"] = (
        "same_as_2022_assumed"
    )

    baseline.loc[
        baseline["state"].isin(REMAPPED_AFTER_2022),
        "boundary_compatibility",
    ] = "different_from_2022_withheld"

    baseline["include_in_2022_backtest"] = ~(
        baseline["state"].isin(REMAPPED_AFTER_2022)
    )

    baseline["district_pres_margin_dem"] = pd.to_numeric(
        baseline["district_pres_margin_dem"],
        errors="coerce",
    )

    # Never silently pass current-boundary values into the 2022 backtest
    # for states whose maps changed after the 2022 election.
    baseline["district_pres_margin_dem_2022_ready"] = (
        baseline["district_pres_margin_dem"]
        .where(baseline["include_in_2022_backtest"])
    )

    baseline["presidential_result_year"] = 2020
    baseline["boundary_cycle"] = 2022

    baseline["source_type"] = (
        "Existing manually curated House production input"
    )

    baseline["source_local_path"] = (
        "inputs/house_race_inputs.csv"
    )

    baseline["methodology_note"] = (
        "2020 Democratic presidential margin copied from the "
        "existing House production input. Values are withheld "
        "for states that changed congressional boundaries after "
        "the 2022 election."
    )

    comparison = master[
        ["race_id"]
    ].merge(
        baseline,
        on="race_id",
        how="left",
        validate="one_to_one",
    )

    missing_join = int(
        comparison["state"].isna().sum()
    )

    usable = int(
        comparison[
            "district_pres_margin_dem_2022_ready"
        ].notna().sum()
    )

    withheld = int(
        comparison[
            "boundary_compatibility"
        ].eq("different_from_2022_withheld").sum()
    )

    unexpected_missing = comparison.loc[
        comparison["include_in_2022_backtest"].fillna(False)
        & comparison[
            "district_pres_margin_dem_2022_ready"
        ].isna(),
        ["race_id"],
    ]

    if missing_join:
        raise ValueError(
            f"{missing_join} master races failed to join "
            "the production baseline."
        )

    if not unexpected_missing.empty:
        raise ValueError(
            "Unexpected missing presidential baselines:\n"
            + unexpected_missing.to_string(index=False)
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    VALIDATION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    state_withheld = (
        comparison.loc[
            comparison["boundary_compatibility"].eq(
                "different_from_2022_withheld"
            ),
            "state",
        ]
        .value_counts()
        .sort_index()
    )

    validation_lines = [
        "2022 House Presidential Baseline Validation",
        "=" * 43,
        "",
        f"Rows: {len(comparison)}",
        f"Unique race IDs: {comparison['race_id'].nunique()}",
        f"Usable provisional 2022 baselines: {usable}",
        f"Withheld due to changed boundaries: {withheld}",
        f"Failed joins: {missing_join}",
        "",
        "Withheld districts by state:",
        state_withheld.to_string(),
        "",
        "Anti-hindsight rule:",
        (
            "Only the 2020 presidential margin is imported. "
            "The 2024 presidential result and the current blended "
            "partisan baseline are excluded from the 2022 backtest."
        ),
        "",
        "Next requirement:",
        (
            "Obtain 2020 presidential results calculated on the "
            "actual 2022 congressional boundaries for AL, GA, LA, "
            "NC, and NY."
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
