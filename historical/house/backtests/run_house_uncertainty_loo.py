#!/usr/bin/env python3
"""
Leave-one-cycle-out calibration of House Election Day uncertainty.

For each held-out election cycle:

1. Select the uncertainty scale using the other three cycles.
2. Evaluate that selected scale on the held-out cycle.
3. Compare it with:
   - current production uncertainty
   - canonical fixed logistic-scale probabilities
   - pooled best scales

Race-level probabilities use the normal CDF because the production
uncertainty engine is normal.
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

SCALE_GRID = np.arange(
    0.80,
    4.51,
    0.05,
)

CURRENT_PRODUCTION_SCALE = 1.0
POOLED_BRIER_SCALE = 1.55
POOLED_LOG_LOSS_SCALE = 2.25


class ValidationError(RuntimeError):
    pass


def score_probabilities(
    probability: pd.Series,
    actual: pd.Series,
) -> tuple[float, float]:
    probability = probability.clip(
        1e-15,
        1.0 - 1e-15,
    )

    brier = float(
        (
            probability
            - actual
        ).pow(2).mean()
    )

    log_loss = -float(
        (
            actual
            * np.log(probability)
            + (1.0 - actual)
            * np.log(
                1.0 - probability
            )
        ).mean()
    )

    return brier, log_loss


def probabilities_for_scale(
    frame: pd.DataFrame,
    scale: float,
) -> pd.Series:
    margin = pd.to_numeric(
        frame["model_margin_dem"],
        errors="raise",
    )

    base_sd = pd.to_numeric(
        frame["total_error_sd_used"],
        errors="raise",
    )

    effective_sd = (
        base_sd
        * float(scale)
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

    probability.loc[
        fixed.eq("D")
    ] = 1.0

    probability.loc[
        fixed.eq("R")
    ] = 0.0

    return probability


def score_frame(
    frame: pd.DataFrame,
    scale: float,
) -> dict[str, float]:
    probability = probabilities_for_scale(
        frame,
        scale,
    )

    scoring_mask = (
        frame["include_in_scoring"]
        .fillna(False)
        .astype(bool)
    )

    actual = (
        frame["actual_dem_win"]
        .fillna(False)
        .astype(bool)
        .astype(float)
    )

    brier, log_loss = score_probabilities(
        probability.loc[scoring_mask],
        actual.loc[scoring_mask],
    )

    expected_dem_seats = float(
        probability.sum()
    )

    actual_dem_seats = int(
        frame["actual_winner"]
        .fillna("")
        .astype(str)
        .str.upper()
        .eq("D")
        .sum()
    )

    return {
        "brier_score": brier,
        "log_loss": log_loss,
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


def choose_training_scale(
    training: pd.DataFrame,
    metric: str,
) -> tuple[float, pd.DataFrame]:
    rows = []

    for scale in SCALE_GRID:
        metrics = score_frame(
            training,
            float(scale),
        )

        rows.append(
            {
                "uncertainty_scale": float(
                    scale
                ),
                **metrics,
            }
        )

    sweep = pd.DataFrame(rows)

    if metric not in sweep.columns:
        raise ValidationError(
            f"Unknown selection metric: {metric}"
        )

    best_index = sweep[metric].idxmin()

    return (
        float(
            sweep.loc[
                best_index,
                "uncertainty_scale",
            ]
        ),
        sweep,
    )


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

    production = raw.loc[
        raw["replay_spec"].eq(
            "production_election_day_v1"
        )
    ].copy()

    canonical = raw.loc[
        raw["replay_spec"].eq(
            "canonical_fixed_6_5"
        )
    ].copy()

    cycles = sorted(
        pd.to_numeric(
            production["cycle"],
            errors="raise",
        )
        .astype(int)
        .unique()
        .tolist()
    )

    if cycles != [
        2016,
        2018,
        2020,
        2022,
    ]:
        raise ValidationError(
            f"Unexpected cycles: {cycles}"
        )

    loo_rows = []
    training_sweep_rows = []

    for held_out_cycle in cycles:
        training = production.loc[
            production["cycle"].ne(
                held_out_cycle
            )
        ].copy()

        test = production.loc[
            production["cycle"].eq(
                held_out_cycle
            )
        ].copy()

        brier_scale, brier_sweep = (
            choose_training_scale(
                training,
                "brier_score",
            )
        )

        log_loss_scale, log_loss_sweep = (
            choose_training_scale(
                training,
                "log_loss",
            )
        )

        for metric_name, sweep in [
            ("brier_score", brier_sweep),
            ("log_loss", log_loss_sweep),
        ]:
            sweep = sweep.copy()
            sweep.insert(
                0,
                "held_out_cycle",
                held_out_cycle,
            )
            sweep.insert(
                1,
                "selection_metric",
                metric_name,
            )
            training_sweep_rows.append(
                sweep
            )

        specifications = [
            (
                "current_production",
                CURRENT_PRODUCTION_SCALE,
            ),
            (
                "loo_selected_brier",
                brier_scale,
            ),
            (
                "loo_selected_log_loss",
                log_loss_scale,
            ),
            (
                "pooled_best_brier",
                POOLED_BRIER_SCALE,
            ),
            (
                "pooled_best_log_loss",
                POOLED_LOG_LOSS_SCALE,
            ),
        ]

        for name, scale in specifications:
            metrics = score_frame(
                test,
                scale,
            )

            loo_rows.append(
                {
                    "held_out_cycle": (
                        held_out_cycle
                    ),
                    "specification": name,
                    "uncertainty_scale": (
                        float(scale)
                    ),
                    "marginal_normal_sd": (
                        float(
                            pd.to_numeric(
                                test[
                                    "total_error_sd_used"
                                ],
                                errors="raise",
                            ).median()
                            * scale
                        )
                    ),
                    "training_selected_brier_scale": (
                        brier_scale
                    ),
                    "training_selected_log_loss_scale": (
                        log_loss_scale
                    ),
                    **metrics,
                }
            )

        canonical_cycle = canonical.loc[
            canonical["cycle"].eq(
                held_out_cycle
            )
        ].copy()

        scoring_mask = (
            canonical_cycle[
                "include_in_scoring"
            ]
            .fillna(False)
            .astype(bool)
        )

        canonical_probability = (
            pd.to_numeric(
                canonical_cycle[
                    "dem_win_probability"
                ],
                errors="raise",
            )
        )

        canonical_actual = (
            canonical_cycle[
                "actual_dem_win"
            ]
            .fillna(False)
            .astype(bool)
            .astype(float)
        )

        canonical_brier, canonical_log_loss = (
            score_probabilities(
                canonical_probability.loc[
                    scoring_mask
                ],
                canonical_actual.loc[
                    scoring_mask
                ],
            )
        )

        expected_dem_seats = float(
            canonical_probability.sum()
        )

        actual_dem_seats = int(
            canonical_cycle[
                "actual_winner"
            ]
            .fillna("")
            .astype(str)
            .str.upper()
            .eq("D")
            .sum()
        )

        loo_rows.append(
            {
                "held_out_cycle": (
                    held_out_cycle
                ),
                "specification": (
                    "canonical_fixed_6_5"
                ),
                "uncertainty_scale": np.nan,
                "marginal_normal_sd": np.nan,
                "training_selected_brier_scale": (
                    brier_scale
                ),
                "training_selected_log_loss_scale": (
                    log_loss_scale
                ),
                "brier_score": (
                    canonical_brier
                ),
                "log_loss": (
                    canonical_log_loss
                ),
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

    loo = pd.DataFrame(loo_rows)

    training_sweeps = pd.concat(
        training_sweep_rows,
        ignore_index=True,
    )

    overall = (
        loo.groupby(
            "specification",
            as_index=False,
        )
        .agg(
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
                "mean_brier_score",
                "mean_log_loss",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    selected_scales = (
        loo[
            [
                "held_out_cycle",
                "training_selected_brier_scale",
                "training_selected_log_loss_scale",
            ]
        ]
        .drop_duplicates()
        .sort_values("held_out_cycle")
        .reset_index(drop=True)
    )

    loo_path = (
        OUTPUT_DIR
        / "house_uncertainty_loo_results.csv"
    )

    overall_path = (
        OUTPUT_DIR
        / "house_uncertainty_loo_summary.csv"
    )

    selected_path = (
        OUTPUT_DIR
        / "house_uncertainty_loo_selected_scales.csv"
    )

    sweeps_path = (
        OUTPUT_DIR
        / "house_uncertainty_loo_training_sweeps.csv"
    )

    loo.to_csv(
        loo_path,
        index=False,
    )

    overall.to_csv(
        overall_path,
        index=False,
    )

    selected_scales.to_csv(
        selected_path,
        index=False,
    )

    training_sweeps.to_csv(
        sweeps_path,
        index=False,
    )

    print(
        "House uncertainty leave-one-cycle-out calibration"
    )
    print("=" * 72)

    print()
    print("Selected training scales:")
    print(
        selected_scales.to_string(
            index=False
        )
    )

    print()
    print("Held-out cycle results:")
    print(
        loo[
            [
                "held_out_cycle",
                "specification",
                "uncertainty_scale",
                "marginal_normal_sd",
                "brier_score",
                "log_loss",
                "absolute_expected_seat_error",
            ]
        ]
        .sort_values(
            [
                "held_out_cycle",
                "specification",
            ]
        )
        .to_string(index=False)
    )

    print()
    print("Overall held-out summary:")
    print(
        overall.to_string(
            index=False
        )
    )

    print()
    print("Wrote:")
    for path in [
        loo_path,
        overall_path,
        selected_path,
        sweeps_path,
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
