#!/usr/bin/env python3
"""
House national-environment signal decomposition.

Purpose
-------
Explain why the production environment bakeoff prefers attenuation of the
shared national environment.

This script does not select or modify a production coefficient. It separates:

1. Election Day generic-ballot accuracy.
2. Normalized partisan baseline error.
3. District-elasticity transmission.
4. Full forecast behavior at representative multipliers.
5. Cycle-specific coefficient preferences.
6. Leave-one-cycle-out coefficient stability.

The analysis deliberately reuses the existing production-environment bakeoff
predictions so that the diagnostic evaluates exactly what the bakeoff scored.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

WAREHOUSE_PATH = (
    ROOT
    / "historical"
    / "house"
    / "warehouse"
    / "house_historical_backtest_inputs_2016_2022.csv"
)

BAKEOFF_DIR = (
    ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "production_environment_bakeoff"
)

PREDICTIONS_PATH = (
    BAKEOFF_DIR
    / "house_production_environment_bakeoff_predictions.csv"
)

BY_CYCLE_PATH = (
    BAKEOFF_DIR
    / "house_production_environment_bakeoff_by_cycle.csv"
)

OUTPUT_DIR = (
    ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "environment_signal_decomposition"
)

SHARED_GENERIC_BALLOT_COEFFICIENT = 0.90

REPRESENTATIVE_MULTIPLIERS = [
    0.00,
    0.50,
    0.70,
    0.73,
    0.80,
    0.85,
    0.90,
    1.00,
]


def require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    label: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]

    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}"
        )


def normalize_cycle(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["cycle"] = pd.to_numeric(
        result["cycle"],
        errors="raise",
    ).astype(int)
    return result


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in [
        WAREHOUSE_PATH,
        PREDICTIONS_PATH,
        BY_CYCLE_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    warehouse = normalize_cycle(pd.read_csv(WAREHOUSE_PATH))
    predictions = normalize_cycle(pd.read_csv(PREDICTIONS_PATH))
    by_cycle = normalize_cycle(pd.read_csv(BY_CYCLE_PATH))

    require_columns(
        warehouse,
        [
            "cycle",
            "district_id",
            "actual_dem_margin",
            "dem_vote_total",
            "gop_vote_total",
            "generic_ballot_margin_dem",
            "include_in_canonical_margin_backtest",
        ],
        "Historical warehouse",
    )

    require_columns(
        predictions,
        [
            "baseline_type",
            "cycle",
            "district_id",
            "selected_baseline_margin_dem",
            "district_elasticity",
            "composite_environment_margin_dem",
            "house_environment_multiplier",
            "district_environment_adjustment_dem",
            "forecast_margin_dem",
            "actual_dem_margin",
            "dem_win_probability",
        ],
        "Bakeoff predictions",
    )

    require_columns(
        by_cycle,
        [
            "baseline_type",
            "house_environment_multiplier",
            "cycle",
            "mean_absolute_error",
            "rmse",
            "brier_score",
            "winner_accuracy",
            "mean_margin_error_dem_bias",
        ],
        "Bakeoff by-cycle results",
    )

    return warehouse, predictions, by_cycle


def select_scoring_warehouse(
    warehouse: pd.DataFrame,
) -> pd.DataFrame:
    eligible = warehouse[
        warehouse["include_in_canonical_margin_backtest"]
        .fillna(False)
        .astype(bool)
    ].copy()

    eligible["dem_vote_total"] = pd.to_numeric(
        eligible["dem_vote_total"],
        errors="coerce",
    )
    eligible["gop_vote_total"] = pd.to_numeric(
        eligible["gop_vote_total"],
        errors="coerce",
    )
    eligible["actual_dem_margin"] = pd.to_numeric(
        eligible["actual_dem_margin"],
        errors="coerce",
    )
    eligible["generic_ballot_margin_dem"] = pd.to_numeric(
        eligible["generic_ballot_margin_dem"],
        errors="coerce",
    )

    return eligible


def select_national_vote_warehouse(
    warehouse: pd.DataFrame,
) -> pd.DataFrame:
    """
    Retain all processed House districts with usable major-party vote
    totals for the national popular-vote calculation.

    This population is intentionally broader than the canonical
    district-margin scoring population. Uncontested and nonstandard
    districts still contribute to the national House vote and therefore
    must not be dropped from the national aggregation.
    """
    national = warehouse.copy()

    numeric_columns = [
        "dem_vote_total",
        "gop_vote_total",
        "actual_dem_margin",
        "generic_ballot_margin_dem",
    ]

    for column in numeric_columns:
        national[column] = pd.to_numeric(
            national[column],
            errors="coerce",
        )

    national = national[
        national["dem_vote_total"].notna()
        & national["gop_vote_total"].notna()
        & (
            national["dem_vote_total"]
            + national["gop_vote_total"]
            > 0
        )
    ].copy()

    return national


def select_normalized_predictions(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    normalized = predictions[
        predictions["baseline_type"]
        == "normalized_partisan_baseline_dem"
    ].copy()

    numeric_columns = [
        "selected_baseline_margin_dem",
        "district_elasticity",
        "composite_environment_margin_dem",
        "house_environment_multiplier",
        "district_environment_adjustment_dem",
        "forecast_margin_dem",
        "actual_dem_margin",
        "dem_win_probability",
    ]

    for column in numeric_columns:
        normalized[column] = pd.to_numeric(
            normalized[column],
            errors="coerce",
        )

    normalized["actual_dem_win"] = (
        normalized["actual_dem_margin"] > 0
    ).astype(float)

    normalized["forecast_error_dem"] = (
        normalized["forecast_margin_dem"]
        - normalized["actual_dem_margin"]
    )

    normalized["baseline_error_dem"] = (
        normalized["selected_baseline_margin_dem"]
        - normalized["actual_dem_margin"]
    )

    normalized["actual_residual_from_baseline"] = (
        normalized["actual_dem_margin"]
        - normalized["selected_baseline_margin_dem"]
    )

    return normalized


def national_signal_table(
    warehouse: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    for cycle, group in warehouse.groupby("cycle", sort=True):
        dem_votes = group["dem_vote_total"].sum(min_count=1)
        gop_votes = group["gop_vote_total"].sum(min_count=1)
        major_party_votes = dem_votes + gop_votes

        if not np.isfinite(major_party_votes) or major_party_votes <= 0:
            actual_national_margin = np.nan
        else:
            actual_national_margin = (
                100.0 * (dem_votes - gop_votes) / major_party_votes
            )

        generic_values = (
            group["generic_ballot_margin_dem"]
            .dropna()
            .unique()
        )

        if len(generic_values) != 1:
            raise ValueError(
                f"Cycle {cycle} has {len(generic_values)} distinct "
                "generic-ballot values."
            )

        generic_ballot = float(generic_values[0])
        shared_environment = (
            generic_ballot
            * SHARED_GENERIC_BALLOT_COEFFICIENT
        )

        records.append(
            {
                "cycle": cycle,
                "districts_in_national_vote": len(group),
                "dem_major_party_votes": dem_votes,
                "gop_major_party_votes": gop_votes,
                "generic_ballot_margin_dem": generic_ballot,
                "actual_national_house_margin_dem": (
                    actual_national_margin
                ),
                "generic_ballot_error_dem": (
                    generic_ballot - actual_national_margin
                ),
                "shared_environment_margin_dem": (
                    shared_environment
                ),
                "shared_environment_error_dem": (
                    shared_environment - actual_national_margin
                ),
                "actual_to_generic_ratio": (
                    actual_national_margin / generic_ballot
                    if abs(generic_ballot) > 1e-9
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(records)


def baseline_table(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    # Baseline is identical across multiplier specifications. Use one row
    # per cycle/district to avoid counting it repeatedly.
    baseline = (
        predictions
        .sort_values("house_environment_multiplier")
        .drop_duplicates(
            subset=["cycle", "district_id"],
            keep="first",
        )
        .copy()
    )

    records = []

    for cycle, group in baseline.groupby("cycle", sort=True):
        actual_wins = int((group["actual_dem_margin"] > 0).sum())
        baseline_wins = int(
            (group["selected_baseline_margin_dem"] > 0).sum()
        )

        records.append(
            {
                "cycle": cycle,
                "districts": len(group),
                "mean_actual_margin_dem": (
                    group["actual_dem_margin"].mean()
                ),
                "mean_baseline_margin_dem": (
                    group["selected_baseline_margin_dem"].mean()
                ),
                "baseline_mean_error_dem": (
                    group["baseline_error_dem"].mean()
                ),
                "baseline_mae": (
                    group["baseline_error_dem"].abs().mean()
                ),
                "baseline_rmse": np.sqrt(
                    np.mean(group["baseline_error_dem"] ** 2)
                ),
                "actual_dem_district_wins": actual_wins,
                "baseline_dem_district_wins": baseline_wins,
                "baseline_win_count_error": (
                    baseline_wins - actual_wins
                ),
                "mean_actual_residual_from_baseline": (
                    group["actual_residual_from_baseline"].mean()
                ),
            }
        )

    return pd.DataFrame(records)


def nearest_available_multiplier(
    available: np.ndarray,
    target: float,
) -> float:
    index = np.argmin(np.abs(available - target))
    selected = float(available[index])

    if abs(selected - target) > 0.0051:
        raise ValueError(
            f"Requested multiplier {target:.2f} is unavailable. "
            f"Nearest value is {selected:.2f}."
        )

    return selected


def multiplier_transmission_table(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    available = np.sort(
        predictions["house_environment_multiplier"]
        .dropna()
        .unique()
    )

    selected_targets = []

    for target in REPRESENTATIVE_MULTIPLIERS:
        if target == 0.00 and not np.any(
            np.isclose(available, 0.00)
        ):
            # Construct baseline-only metrics separately below.
            continue

        selected_targets.append(
            nearest_available_multiplier(available, target)
        )

    selected_targets = sorted(set(selected_targets))
    records = []

    # Add a synthetic multiplier-zero specification using the stored
    # selected baseline. This is explanatory only.
    baseline = (
        predictions
        .sort_values("house_environment_multiplier")
        .drop_duplicates(
            subset=["cycle", "district_id"],
            keep="first",
        )
        .copy()
    )

    baseline["house_environment_multiplier"] = 0.0
    baseline["district_environment_adjustment_dem"] = 0.0
    baseline["forecast_margin_dem"] = (
        baseline["selected_baseline_margin_dem"]
    )
    baseline["dem_win_probability"] = (
        1.0
        / (
            1.0
            + np.exp(
                -baseline["forecast_margin_dem"] / 6.0
            )
        )
    )

    frames = [baseline]

    for multiplier in selected_targets:
        frames.append(
            predictions[
                np.isclose(
                    predictions["house_environment_multiplier"],
                    multiplier,
                )
            ].copy()
        )

    comparison = pd.concat(frames, ignore_index=True)

    comparison["forecast_error_dem"] = (
        comparison["forecast_margin_dem"]
        - comparison["actual_dem_margin"]
    )

    comparison["actual_dem_win"] = (
        comparison["actual_dem_margin"] > 0
    ).astype(float)

    comparison["brier_component"] = (
        comparison["dem_win_probability"]
        - comparison["actual_dem_win"]
    ) ** 2

    for (
        cycle,
        multiplier,
    ), group in comparison.groupby(
        ["cycle", "house_environment_multiplier"],
        sort=True,
    ):
        expected_dem_wins = group["dem_win_probability"].sum()
        actual_dem_wins = group["actual_dem_win"].sum()

        records.append(
            {
                "cycle": cycle,
                "house_environment_multiplier": multiplier,
                "effective_generic_ballot_coefficient": (
                    SHARED_GENERIC_BALLOT_COEFFICIENT
                    * multiplier
                ),
                "districts": len(group),
                "mean_district_elasticity": (
                    group["district_elasticity"].mean()
                ),
                "mean_environment_adjustment_dem": (
                    group[
                        "district_environment_adjustment_dem"
                    ].mean()
                ),
                "mean_actual_residual_from_baseline": (
                    (
                        group["actual_dem_margin"]
                        - group["selected_baseline_margin_dem"]
                    ).mean()
                ),
                "environment_residual_error_dem": (
                    (
                        group[
                            "district_environment_adjustment_dem"
                        ]
                        - (
                            group["actual_dem_margin"]
                            - group[
                                "selected_baseline_margin_dem"
                            ]
                        )
                    ).mean()
                ),
                "margin_mae": (
                    group["forecast_error_dem"].abs().mean()
                ),
                "margin_rmse": np.sqrt(
                    np.mean(group["forecast_error_dem"] ** 2)
                ),
                "margin_bias_dem": (
                    group["forecast_error_dem"].mean()
                ),
                "brier_score": (
                    group["brier_component"].mean()
                ),
                "winner_accuracy": (
                    (
                        (group["forecast_margin_dem"] > 0)
                        == (group["actual_dem_margin"] > 0)
                    ).mean()
                ),
                "expected_dem_wins": expected_dem_wins,
                "actual_dem_wins": actual_dem_wins,
                "expected_win_count_error": (
                    expected_dem_wins - actual_dem_wins
                ),
            }
        )

    return pd.DataFrame(records)


def component_overlap_table(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    baseline = (
        predictions
        .sort_values("house_environment_multiplier")
        .drop_duplicates(
            subset=["cycle", "district_id"],
            keep="first",
        )
        .copy()
    )

    columns = {
        "baseline_margin": baseline[
            "selected_baseline_margin_dem"
        ],
        "district_elasticity": baseline["district_elasticity"],
        "actual_margin": baseline["actual_dem_margin"],
        "actual_residual_from_baseline": (
            baseline["actual_dem_margin"]
            - baseline["selected_baseline_margin_dem"]
        ),
    }

    frame = pd.DataFrame(columns)

    correlation = frame.corr(numeric_only=True)

    records = []

    for left in correlation.columns:
        for right in correlation.columns:
            if left >= right:
                continue

            records.append(
                {
                    "component_a": left,
                    "component_b": right,
                    "correlation": correlation.loc[left, right],
                }
            )

    return (
        pd.DataFrame(records)
        .sort_values(
            "correlation",
            key=lambda values: values.abs(),
            ascending=False,
        )
        .reset_index(drop=True)
    )


def cycle_preference_table(
    by_cycle: pd.DataFrame,
) -> pd.DataFrame:
    normalized = by_cycle[
        by_cycle["baseline_type"]
        == "normalized_partisan_baseline_dem"
    ].copy()

    numeric_columns = [
        "house_environment_multiplier",
        "mean_absolute_error",
        "rmse",
        "brier_score",
        "winner_accuracy",
        "mean_margin_error_dem_bias",
    ]

    for column in numeric_columns:
        normalized[column] = pd.to_numeric(
            normalized[column],
            errors="coerce",
        )

    records = []

    for cycle, group in normalized.groupby("cycle", sort=True):
        best_mae = group.loc[
            group["mean_absolute_error"].idxmin()
        ]
        best_rmse = group.loc[
            group["rmse"].idxmin()
        ]
        best_brier = group.loc[
            group["brier_score"].idxmin()
        ]
        best_bias = group.loc[
            group["mean_margin_error_dem_bias"].abs().idxmin()
        ]

        records.append(
            {
                "cycle": cycle,
                "best_mae_multiplier": (
                    best_mae["house_environment_multiplier"]
                ),
                "best_mae": best_mae["mean_absolute_error"],
                "best_rmse_multiplier": (
                    best_rmse["house_environment_multiplier"]
                ),
                "best_rmse": best_rmse["rmse"],
                "best_brier_multiplier": (
                    best_brier["house_environment_multiplier"]
                ),
                "best_brier": best_brier["brier_score"],
                "lowest_abs_bias_multiplier": (
                    best_bias["house_environment_multiplier"]
                ),
                "lowest_abs_bias": abs(
                    best_bias["mean_margin_error_dem_bias"]
                ),
            }
        )

    return pd.DataFrame(records)


def leave_one_cycle_out_table(
    by_cycle: pd.DataFrame,
) -> pd.DataFrame:
    normalized = by_cycle[
        by_cycle["baseline_type"]
        == "normalized_partisan_baseline_dem"
    ].copy()

    cycles = sorted(normalized["cycle"].unique())
    records = []

    for holdout_cycle in cycles:
        training = normalized[
            normalized["cycle"] != holdout_cycle
        ].copy()

        # Give every cycle equal weight rather than allowing cycles with
        # slightly more scorable districts to dominate.
        training_summary = (
            training
            .groupby("house_environment_multiplier", as_index=False)
            .agg(
                mean_cycle_mae=("mean_absolute_error", "mean"),
                mean_cycle_rmse=("rmse", "mean"),
                mean_cycle_brier=("brier_score", "mean"),
                mean_abs_cycle_bias=(
                    "mean_margin_error_dem_bias",
                    lambda values: np.mean(np.abs(values)),
                ),
            )
        )

        for metric, selected_column in [
            ("mean_cycle_mae", "mae"),
            ("mean_cycle_rmse", "rmse"),
            ("mean_cycle_brier", "brier"),
            ("mean_abs_cycle_bias", "bias"),
        ]:
            selected = training_summary.loc[
                training_summary[metric].idxmin()
            ]

            multiplier = selected[
                "house_environment_multiplier"
            ]

            held_out = normalized[
                (normalized["cycle"] == holdout_cycle)
                & np.isclose(
                    normalized["house_environment_multiplier"],
                    multiplier,
                )
            ]

            if len(held_out) != 1:
                raise ValueError(
                    "Could not uniquely locate held-out result for "
                    f"cycle {holdout_cycle}, multiplier {multiplier}."
                )

            row = held_out.iloc[0]

            records.append(
                {
                    "holdout_cycle": holdout_cycle,
                    "selection_metric": selected_column,
                    "selected_multiplier": multiplier,
                    "heldout_mae": row["mean_absolute_error"],
                    "heldout_rmse": row["rmse"],
                    "heldout_brier": row["brier_score"],
                    "heldout_winner_accuracy": (
                        row["winner_accuracy"]
                    ),
                    "heldout_margin_bias_dem": (
                        row["mean_margin_error_dem_bias"]
                    ),
                }
            )

    return pd.DataFrame(records)


def print_table(title: str, frame: pd.DataFrame) -> None:
    print()
    print(title)
    print("=" * 120)
    print(frame.to_string(index=False))


def main() -> int:
    raw_warehouse, predictions, by_cycle = load_inputs()

    national_vote_warehouse = (
        select_national_vote_warehouse(raw_warehouse)
    )
    scoring_warehouse = select_scoring_warehouse(
        raw_warehouse
    )
    predictions = select_normalized_predictions(predictions)

    signal = national_signal_table(
        national_vote_warehouse
    )
    baseline = baseline_table(predictions)
    transmission = multiplier_transmission_table(predictions)
    overlap = component_overlap_table(predictions)
    preferences = cycle_preference_table(by_cycle)
    loo = leave_one_cycle_out_table(by_cycle)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "house_environment_national_signal.csv": signal,
        "house_environment_baseline_diagnostic.csv": baseline,
        "house_environment_multiplier_transmission.csv": transmission,
        "house_environment_component_overlap.csv": overlap,
        "house_environment_cycle_preferences.csv": preferences,
        "house_environment_leave_one_cycle_out.csv": loo,
    }

    for filename, frame in outputs.items():
        frame.to_csv(OUTPUT_DIR / filename, index=False)

    print()
    print("House environment signal decomposition")
    print("=" * 120)
    print(
        "National-vote warehouse rows: "
        f"{len(national_vote_warehouse):,}"
    )
    print(
        "District-margin scoring rows: "
        f"{len(scoring_warehouse):,}"
    )
    print(f"Normalized prediction rows: {len(predictions):,}")
    print(
        "Cycles: "
        f"{sorted(raw_warehouse['cycle'].unique())}"
    )
    print(f"Outputs: {OUTPUT_DIR}")

    print_table(
        "A. NATIONAL GENERIC-BALLOT REALITY CHECK",
        signal.round(4),
    )

    print_table(
        "B. NORMALIZED BASELINE-ONLY DIAGNOSTIC",
        baseline.round(4),
    )

    print_table(
        "C. ENVIRONMENT TRANSMISSION BY CYCLE AND MULTIPLIER",
        transmission.round(4),
    )

    print_table(
        "D. COMPONENT OVERLAP",
        overlap.round(4),
    )

    print_table(
        "E. CYCLE-SPECIFIC OPTIMAL MULTIPLIERS",
        preferences.round(4),
    )

    print_table(
        "F. LEAVE-ONE-CYCLE-OUT MULTIPLIER STABILITY",
        loo.round(4),
    )

    print()
    print("Validation checks")
    print("-" * 120)

    checks = {
        "Four historical cycles present": (
            signal["cycle"].nunique() == 4
        ),
        "National margins available": (
            signal["actual_national_house_margin_dem"]
            .notna()
            .all()
        ),
        "Baseline table covers all cycles": (
            baseline["cycle"].nunique() == 4
        ),
        "Cycle preferences cover all cycles": (
            preferences["cycle"].nunique() == 4
        ),
        "LOO covers four cycles and four metrics": (
            len(loo) == 16
        ),
        "No duplicate cycle/multiplier transmission rows": (
            not transmission.duplicated(
                ["cycle", "house_environment_multiplier"]
            ).any()
        ),
    }

    failed = []

    for label, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {label}")

        if not passed:
            failed.append(label)

    if failed:
        print()
        print("House environment signal decomposition: FAILED")
        return 1

    print()
    print("House environment signal decomposition: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
