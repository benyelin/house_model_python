from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

INPUT = (
    ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "generic_ballot_rv_lv_selection_bakeoff"
    / "rv_lv_selection_predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "generic_ballot_rv_lv_blend_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MIDTERMS = [
    2018,
    2022,
]

BLENDS = {
    "rv_only": 0.00,
    "75rv_25lv": 0.25,
    "50rv_50lv": 0.50,
    "25rv_75lv": 0.75,
    "lv_only": 1.00,
}


def summarize(frame):
    rows = []

    for strategy, group in frame.groupby("strategy"):
        error = group["error"]

        rows.append(
            {
                "strategy": strategy,
                "observations": len(group),
                "cycles": group["cycle"].nunique(),
                "mae": error.abs().mean(),
                "rmse": np.sqrt(np.mean(error ** 2)),
                "bias_dem": error.mean(),
                "max_absolute_error": error.abs().max(),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("mae")
        .reset_index(drop=True)
    )


def summarize_horizons(frame):
    rows = []

    for (days_out, strategy), group in frame.groupby(
        [
            "days_out",
            "strategy",
        ]
    ):
        error = group["error"]

        rows.append(
            {
                "days_out": int(days_out),
                "strategy": strategy,
                "cycles": group["cycle"].nunique(),
                "mae": error.abs().mean(),
                "rmse": np.sqrt(np.mean(error ** 2)),
                "bias_dem": error.mean(),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "days_out",
                "mae",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def main():
    print("=" * 110)
    print(
        "HOUSE GENERIC BALLOT — "
        "SIMPLE RV/LV BLEND TEST"
    )
    print("=" * 110)

    source = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    print()
    print("INPUT COLUMNS")
    print("-" * 110)

    print(
        "\n".join(
            source.columns.tolist()
        )
    )

    required = {
        "cycle",
        "days_out",
        "population_rule",
        "environment_estimate",
        "actual_house_margin_dem",
    }

    missing = required - set(source.columns)

    if missing:
        raise RuntimeError(
            "Missing required input columns: "
            f"{sorted(missing)}"
        )

    needed = source.loc[
        source["population_rule"].isin(
            [
                "rv_only",
                "lv_only",
            ]
        )
    ].copy()

    pivot = needed.pivot(
        index=[
            "cycle",
            "days_out",
            "actual_house_margin_dem",
        ],
        columns="population_rule",
        values="environment_estimate",
    ).reset_index()

    if (
        pivot["rv_only"].isna().any()
        or pivot["lv_only"].isna().any()
    ):
        raise RuntimeError(
            "At least one cycle/horizon is "
            "missing an RV or LV estimate."
        )

    rows = []

    for _, row in pivot.iterrows():
        rv = float(row["rv_only"])
        lv = float(row["lv_only"])

        actual = float(
            row["actual_house_margin_dem"]
        )

        for strategy, lv_weight in BLENDS.items():
            estimate = (
                (1.0 - lv_weight) * rv
                + lv_weight * lv
            )

            rows.append(
                {
                    "cycle": int(row["cycle"]),
                    "days_out": int(row["days_out"]),
                    "strategy": strategy,
                    "lv_weight": lv_weight,
                    "rv_weight": 1.0 - lv_weight,
                    "rv_estimate": rv,
                    "lv_estimate": lv,
                    "blended_estimate": estimate,
                    "actual_house_margin_dem": actual,
                    "error": estimate - actual,
                    "absolute_error": abs(
                        estimate - actual
                    ),
                }
            )

    results = pd.DataFrame(rows)

    all_summary = summarize(
        results
    )

    midterm_results = results.loc[
        results["cycle"].isin(
            MIDTERMS
        )
    ].copy()

    midterm_summary = summarize(
        midterm_results
    )

    all_horizons = summarize_horizons(
        results
    )

    midterm_horizons = summarize_horizons(
        midterm_results
    )

    print()
    print("OVERALL — ALL FOUR CYCLES")
    print("-" * 110)

    print(
        all_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("OVERALL — MIDTERMS ONLY")
    print("-" * 110)

    print(
        midterm_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("BY HORIZON — ALL FOUR CYCLES")
    print("-" * 110)

    print(
        all_horizons.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("BY HORIZON — MIDTERMS ONLY")
    print("-" * 110)

    print(
        midterm_horizons.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    winners_all = (
        all_horizons
        .sort_values(
            [
                "days_out",
                "mae",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .groupby(
            "days_out",
            as_index=False,
        )
        .first()
    )

    winners_midterms = (
        midterm_horizons
        .sort_values(
            [
                "days_out",
                "mae",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .groupby(
            "days_out",
            as_index=False,
        )
        .first()
    )

    print()
    print("WINNER BY HORIZON — ALL FOUR CYCLES")
    print("-" * 110)

    print(
        winners_all[
            [
                "days_out",
                "strategy",
                "mae",
                "rmse",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("WINNER BY HORIZON — MIDTERMS ONLY")
    print("-" * 110)

    print(
        winners_midterms[
            [
                "days_out",
                "strategy",
                "mae",
                "rmse",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    validation = pd.DataFrame(
        [
            {
                "check": "four_cycles",
                "passed": results["cycle"].nunique() == 4,
            },
            {
                "check": "forty_cycle_horizons",
                "passed": pivot.shape[0] == 40,
            },
            {
                "check": "five_strategies",
                "passed": (
                    results["strategy"].nunique()
                    == 5
                ),
            },
            {
                "check": "200_predictions",
                "passed": len(results) == 200,
            },
            {
                "check": "all_finite",
                "passed": bool(
                    np.isfinite(
                        results["blended_estimate"]
                    ).all()
                ),
            },
        ]
    )

    print()
    print("VALIDATION")
    print("-" * 110)

    print(
        validation.to_string(
            index=False
        )
    )

    if not validation["passed"].all():
        raise RuntimeError(
            "RV/LV blend validation FAILED."
        )

    results.to_csv(
        OUTPUT_DIR
        / "rv_lv_blend_predictions.csv",
        index=False,
    )

    all_summary.to_csv(
        OUTPUT_DIR
        / "rv_lv_blend_all_cycles_summary.csv",
        index=False,
    )

    midterm_summary.to_csv(
        OUTPUT_DIR
        / "rv_lv_blend_midterm_summary.csv",
        index=False,
    )

    all_horizons.to_csv(
        OUTPUT_DIR
        / "rv_lv_blend_all_cycles_horizons.csv",
        index=False,
    )

    midterm_horizons.to_csv(
        OUTPUT_DIR
        / "rv_lv_blend_midterm_horizons.csv",
        index=False,
    )

    validation.to_csv(
        OUTPUT_DIR
        / "validation.csv",
        index=False,
    )

    print()
    print(
        "Simple RV/LV blend test "
        "validation PASSED."
    )


if __name__ == "__main__":
    main()
