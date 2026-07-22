#!/usr/bin/env python3
"""
Leave-one-cycle-out analysis of systematic House margin bias.

Purpose
-------
Determine whether the production replay's average Democratic margin
overprediction is:

1. stable across election cycles;
2. predictable from prior cycles; and
3. worth correcting in production.

For each held-out cycle, the script estimates the mean signed margin
error using the other cycles:

    bias_dem = mean(model_margin_dem - actual_dem_margin)

It then subtracts that training-only bias from the held-out model
margins and evaluates:

- MAE
- RMSE
- mean signed error
- winner accuracy
- Brier score
- log loss

Probabilities are evaluated at:

- current production uncertainty scale: 1.00
- compromise uncertainty scale: 1.75

This is deliberately leakage-free: the held-out cycle never contributes
to its own bias estimate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "production_replay_v1"
    / "house_production_replay_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "production_replay_v1"
)

PRODUCTION_SPEC = "production_election_day_v1"

UNCERTAINTY_SCALES = {
    "current_scale_1_00": 1.00,
    "compromise_scale_1_75": 1.75,
}

EPSILON = 1e-15


class ValidationError(RuntimeError):
    pass


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)

    normalized = (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return normalized.isin(
        {
            "true",
            "1",
            "yes",
            "y",
            "t",
        }
    )


def require_columns(
    frame: pd.DataFrame,
    columns: list[str],
) -> None:
    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]

    if missing:
        raise ValidationError(
            "Missing required columns: "
            + ", ".join(missing)
        )


def probability_from_margin(
    frame: pd.DataFrame,
    margin: pd.Series,
    uncertainty_scale: float,
) -> pd.Series:
    base_sd = pd.to_numeric(
        frame["total_error_sd_used"],
        errors="raise",
    )

    effective_sd = (
        base_sd
        * float(uncertainty_scale)
    )

    probability = pd.Series(
        ndtr(
            margin.to_numpy(dtype=float)
            / effective_sd.to_numpy(dtype=float)
        ),
        index=frame.index,
        dtype=float,
    )

    fixed = (
        frame["party_control_fixed"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    probability.loc[fixed.eq("D")] = 1.0
    probability.loc[fixed.eq("R")] = 0.0

    return probability.clip(
        EPSILON,
        1.0 - EPSILON,
    )


def margin_metrics(
    actual_margin: pd.Series,
    predicted_margin: pd.Series,
) -> dict[str, float]:
    error = (
        predicted_margin
        - actual_margin
    )

    actual_winner_dem = (
        actual_margin > 0
    )

    predicted_winner_dem = (
        predicted_margin > 0
    )

    return {
        "mean_error_dem": float(
            error.mean()
        ),
        "mae": float(
            error.abs().mean()
        ),
        "rmse": float(
            np.sqrt(
                np.mean(
                    np.square(error)
                )
            )
        ),
        "winner_accuracy": float(
            (
                actual_winner_dem
                == predicted_winner_dem
            ).mean()
        ),
    }


def probability_metrics(
    actual_dem_win: pd.Series,
    probability: pd.Series,
) -> dict[str, float]:
    actual = actual_dem_win.astype(float)

    brier = float(
        np.mean(
            np.square(
                probability
                - actual
            )
        )
    )

    log_loss = -float(
        np.mean(
            actual * np.log(probability)
            + (1.0 - actual)
            * np.log(
                1.0 - probability
            )
        )
    )

    return {
        "brier_score": brier,
        "log_loss": log_loss,
    }


def score_variant(
    frame: pd.DataFrame,
    predicted_margin: pd.Series,
    uncertainty_scale: float,
) -> dict[str, float]:
    margin_mask = as_bool(
        frame[
            "include_in_major_party_margin_scoring"
        ]
    )

    probability_mask = as_bool(
        frame["include_in_scoring"]
    )

    actual_margin = pd.to_numeric(
        frame["actual_dem_margin"],
        errors="coerce",
    )

    actual_dem_win = as_bool(
        frame["actual_dem_win"]
    )

    probability = probability_from_margin(
        frame=frame,
        margin=predicted_margin,
        uncertainty_scale=uncertainty_scale,
    )

    metrics = margin_metrics(
        actual_margin=actual_margin.loc[
            margin_mask
        ],
        predicted_margin=predicted_margin.loc[
            margin_mask
        ],
    )

    metrics.update(
        probability_metrics(
            actual_dem_win=actual_dem_win.loc[
                probability_mask
            ],
            probability=probability.loc[
                probability_mask
            ],
        )
    )

    expected_dem_seats = float(
        probability.sum()
    )

    actual_dem_seats = int(
        frame["actual_winner"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .eq("D")
        .sum()
    )

    metrics.update(
        {
            "expected_dem_seats": (
                expected_dem_seats
            ),
            "actual_dem_seats": (
                actual_dem_seats
            ),
            "expected_seat_error": (
                expected_dem_seats
                - actual_dem_seats
            ),
            "absolute_expected_seat_error": abs(
                expected_dem_seats
                - actual_dem_seats
            ),
        }
    )

    return metrics


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Replay predictions not found: "
            f"{INPUT_PATH}"
        )

    raw = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    require_columns(
        raw,
        [
            "replay_spec",
            "cycle",
            "model_margin_dem",
            "actual_dem_margin",
            "actual_dem_win",
            "actual_winner",
            "include_in_major_party_margin_scoring",
            "include_in_scoring",
            "party_control_fixed",
            "total_error_sd_used",
        ],
    )

    frame = raw.loc[
        raw["replay_spec"].eq(
            PRODUCTION_SPEC
        )
    ].copy()

    frame["cycle"] = pd.to_numeric(
        frame["cycle"],
        errors="raise",
    ).astype(int)

    frame["model_margin_dem"] = pd.to_numeric(
        frame["model_margin_dem"],
        errors="raise",
    )

    frame["actual_dem_margin"] = pd.to_numeric(
        frame["actual_dem_margin"],
        errors="coerce",
    )

    margin_mask = as_bool(
        frame[
            "include_in_major_party_margin_scoring"
        ]
    )

    cycles = sorted(
        frame["cycle"]
        .dropna()
        .unique()
        .tolist()
    )

    if len(cycles) < 3:
        raise ValidationError(
            f"Expected at least three cycles; "
            f"found {cycles}"
        )

    cycle_bias_rows = []

    for cycle in cycles:
        cycle_frame = frame.loc[
            frame["cycle"].eq(cycle)
            & margin_mask
        ].copy()

        error = (
            cycle_frame["model_margin_dem"]
            - cycle_frame["actual_dem_margin"]
        )

        cycle_bias_rows.append(
            {
                "cycle": cycle,
                "n_margin_races": int(
                    error.notna().sum()
                ),
                "mean_error_dem": float(
                    error.mean()
                ),
                "median_error_dem": float(
                    error.median()
                ),
                "mae": float(
                    error.abs().mean()
                ),
                "rmse": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                error.dropna()
                            )
                        )
                    )
                ),
            }
        )

    cycle_bias = pd.DataFrame(
        cycle_bias_rows
    )

    loo_rows = []
    race_rows = []

    for held_out_cycle in cycles:
        training_mask = (
            frame["cycle"].ne(
                held_out_cycle
            )
            & margin_mask
        )

        test = frame.loc[
            frame["cycle"].eq(
                held_out_cycle
            )
        ].copy()

        training_error = (
            frame.loc[
                training_mask,
                "model_margin_dem",
            ]
            - frame.loc[
                training_mask,
                "actual_dem_margin",
            ]
        )

        training_bias_dem = float(
            training_error.mean()
        )

        raw_margin = pd.to_numeric(
            test["model_margin_dem"],
            errors="raise",
        )

        corrected_margin = (
            raw_margin
            - training_bias_dem
        )

        for uncertainty_name, scale in (
            UNCERTAINTY_SCALES.items()
        ):
            for variant_name, margin in [
                (
                    "uncorrected",
                    raw_margin,
                ),
                (
                    "loo_bias_corrected",
                    corrected_margin,
                ),
            ]:
                metrics = score_variant(
                    frame=test,
                    predicted_margin=margin,
                    uncertainty_scale=scale,
                )

                loo_rows.append(
                    {
                        "held_out_cycle": (
                            held_out_cycle
                        ),
                        "variant": variant_name,
                        "uncertainty_spec": (
                            uncertainty_name
                        ),
                        "uncertainty_scale": (
                            scale
                        ),
                        "training_bias_dem": (
                            training_bias_dem
                        ),
                        **metrics,
                    }
                )

        test_output = test[
            [
                column
                for column in [
                    "cycle",
                    "race_id",
                    "district_id",
                    "state",
                    "district",
                    "actual_dem_margin",
                    "actual_winner",
                    "model_margin_dem",
                    "total_error_sd_used",
                    "party_control_fixed",
                ]
                if column in test.columns
            ]
        ].copy()

        test_output[
            "held_out_cycle"
        ] = held_out_cycle

        test_output[
            "training_bias_dem"
        ] = training_bias_dem

        test_output[
            "loo_corrected_margin_dem"
        ] = corrected_margin.to_numpy()

        test_output[
            "raw_margin_error"
        ] = (
            raw_margin
            - test["actual_dem_margin"]
        ).to_numpy()

        test_output[
            "corrected_margin_error"
        ] = (
            corrected_margin
            - test["actual_dem_margin"]
        ).to_numpy()

        race_rows.append(test_output)

    loo_results = pd.DataFrame(
        loo_rows
    )

    race_results = pd.concat(
        race_rows,
        ignore_index=True,
    )

    summary = (
        loo_results.groupby(
            [
                "variant",
                "uncertainty_spec",
                "uncertainty_scale",
            ],
            as_index=False,
        )
        .agg(
            mean_error_dem=(
                "mean_error_dem",
                "mean",
            ),
            mean_mae=(
                "mae",
                "mean",
            ),
            mean_rmse=(
                "rmse",
                "mean",
            ),
            mean_winner_accuracy=(
                "winner_accuracy",
                "mean",
            ),
            mean_brier_score=(
                "brier_score",
                "mean",
            ),
            mean_log_loss=(
                "log_loss",
                "mean",
            ),
            mean_abs_expected_seat_error=(
                "absolute_expected_seat_error",
                "mean",
            ),
            mean_expected_seat_error=(
                "expected_seat_error",
                "mean",
            ),
        )
        .sort_values(
            [
                "uncertainty_scale",
                "variant",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    training_biases = (
        loo_results[
            [
                "held_out_cycle",
                "training_bias_dem",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "held_out_cycle"
        )
        .reset_index(drop=True)
    )

    cycle_bias_path = (
        OUTPUT_DIR
        / "house_margin_bias_by_cycle.csv"
    )

    loo_path = (
        OUTPUT_DIR
        / "house_margin_bias_loo_results.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "house_margin_bias_loo_summary.csv"
    )

    race_path = (
        OUTPUT_DIR
        / "house_margin_bias_loo_races.csv"
    )

    cycle_bias.to_csv(
        cycle_bias_path,
        index=False,
    )

    loo_results.to_csv(
        loo_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    race_results.to_csv(
        race_path,
        index=False,
    )

    print(
        "House production replay margin-bias analysis"
    )
    print("=" * 76)

    print()
    print("Observed signed margin error by cycle:")
    print(
        cycle_bias.to_string(
            index=False
        )
    )

    print()
    print(
        "Training-only Democratic bias used "
        "for each held-out cycle:"
    )
    print(
        training_biases.to_string(
            index=False
        )
    )

    print()
    print("Held-out results:")
    print(
        loo_results[
            [
                "held_out_cycle",
                "variant",
                "uncertainty_scale",
                "training_bias_dem",
                "mean_error_dem",
                "mae",
                "rmse",
                "winner_accuracy",
                "brier_score",
                "log_loss",
                "absolute_expected_seat_error",
            ]
        ]
        .sort_values(
            [
                "held_out_cycle",
                "uncertainty_scale",
                "variant",
            ],
            kind="mergesort",
        )
        .to_string(index=False)
    )

    print()
    print("Overall leave-one-cycle-out summary:")
    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print("Wrote:")
    for path in [
        cycle_bias_path,
        loo_path,
        summary_path,
        race_path,
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
