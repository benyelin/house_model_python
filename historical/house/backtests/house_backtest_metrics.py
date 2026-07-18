#!/usr/bin/env python3
"""
Shared scoring utilities for House historical backtests.

This module centralizes the margin-to-probability transform and the principal
forecast-performance metrics used by House calibration and sensitivity tests.

It is intentionally independent of any particular model layer so that the
canonical backtest, component sweeps, and future calibration studies can use
the same definitions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_PROBABILITY_SCALE = 6.5


@dataclass(frozen=True)
class ForecastMetrics:
    """Aggregate scoring metrics for one set of race forecasts."""

    scored_races: int
    mean_margin_error_dem_bias: float
    mean_absolute_error: float
    median_absolute_error: float
    rmse: float
    winner_accuracy: float
    brier_score: float
    log_loss: float
    actual_dem_wins: int
    predicted_dem_wins: int
    expected_dem_wins: float
    predicted_win_count_error: int
    expected_win_count_error: float

    def as_dict(self) -> dict[str, object]:
        return {
            "scored_races": self.scored_races,
            "mean_margin_error_dem_bias": (
                self.mean_margin_error_dem_bias
            ),
            "mean_absolute_error": self.mean_absolute_error,
            "median_absolute_error": self.median_absolute_error,
            "rmse": self.rmse,
            "winner_accuracy": self.winner_accuracy,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "actual_dem_wins": self.actual_dem_wins,
            "predicted_dem_wins": self.predicted_dem_wins,
            "expected_dem_wins": self.expected_dem_wins,
            "predicted_win_count_error": (
                self.predicted_win_count_error
            ),
            "expected_win_count_error": (
                self.expected_win_count_error
            ),
        }


def logistic_probability(
    margin_dem: pd.Series | np.ndarray | list[float],
    error_sd: float = DEFAULT_PROBABILITY_SCALE,
) -> pd.Series:
    """
    Convert a Democratic forecast margin into Democratic win probability.

    The calculation matches the existing House layered backtest:

        p(D win) = 1 / (1 + exp(-margin / error_sd))

    Parameters
    ----------
    margin_dem:
        Forecast margin from the Democratic perspective.
    error_sd:
        Positive logistic scale parameter.
    """
    if not np.isfinite(error_sd) or error_sd <= 0:
        raise ValueError("error_sd must be a positive finite number.")

    if isinstance(margin_dem, pd.Series):
        index = margin_dem.index
    else:
        index = None

    margin = pd.Series(
        margin_dem,
        index=index,
        dtype="float64",
    )

    margin = pd.to_numeric(
        margin,
        errors="coerce",
    )

    z = (margin / float(error_sd)).clip(
        lower=-40.0,
        upper=40.0,
    )

    return 1.0 / (1.0 + np.exp(-z))


def safe_log_loss(
    actual_dem_win: pd.Series | np.ndarray | list[float],
    probability_dem: pd.Series | np.ndarray | list[float],
) -> float:
    """Calculate binary log loss while safely handling missing values."""
    actual = pd.Series(actual_dem_win, dtype="float64")
    probability = pd.Series(probability_dem, dtype="float64")

    if len(actual) != len(probability):
        raise ValueError(
            "actual_dem_win and probability_dem must have equal lengths."
        )

    valid = actual.notna() & probability.notna()

    if not valid.any():
        return math.nan

    actual = actual.loc[valid].astype(float)

    probability = (
        probability.loc[valid]
        .astype(float)
        .clip(lower=1e-9, upper=1.0 - 1e-9)
    )

    return float(
        -(
            actual * np.log(probability)
            + (1.0 - actual) * np.log(1.0 - probability)
        ).mean()
    )


def score_forecasts(
    actual_margin_dem: pd.Series | np.ndarray | list[float],
    forecast_margin_dem: pd.Series | np.ndarray | list[float],
    probability_dem: (
        pd.Series | np.ndarray | list[float] | None
    ) = None,
    error_sd: float = DEFAULT_PROBABILITY_SCALE,
) -> ForecastMetrics:
    """
    Score race-level margin forecasts using the canonical House definitions.

    Missing rows are excluded if either the actual or forecast margin is
    unavailable. Probabilities may be provided directly; otherwise they are
    generated from the forecast margin using ``logistic_probability``.
    """
    actual = pd.Series(
        actual_margin_dem,
        dtype="float64",
    )

    forecast = pd.Series(
        forecast_margin_dem,
        dtype="float64",
    )

    if len(actual) != len(forecast):
        raise ValueError(
            "actual_margin_dem and forecast_margin_dem "
            "must have equal lengths."
        )

    work = pd.DataFrame(
        {
            "actual_margin_dem": pd.to_numeric(
                actual,
                errors="coerce",
            ),
            "forecast_margin_dem": pd.to_numeric(
                forecast,
                errors="coerce",
            ),
        }
    )

    if probability_dem is None:
        work["probability_dem"] = logistic_probability(
            work["forecast_margin_dem"],
            error_sd=error_sd,
        )
    else:
        probability = pd.Series(
            probability_dem,
            dtype="float64",
        )

        if len(probability) != len(work):
            raise ValueError(
                "probability_dem must have the same length "
                "as the margin inputs."
            )

        work["probability_dem"] = pd.to_numeric(
            probability,
            errors="coerce",
        )

    valid = (
        work["actual_margin_dem"].notna()
        & work["forecast_margin_dem"].notna()
        & work["probability_dem"].notna()
    )

    work = work.loc[valid].copy()

    if work.empty:
        raise ValueError("No valid forecast rows were available to score.")

    if not np.isfinite(
        work[
            [
                "actual_margin_dem",
                "forecast_margin_dem",
                "probability_dem",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("Forecast scoring inputs contain non-finite values.")

    invalid_probability = (
        (work["probability_dem"] < 0.0)
        | (work["probability_dem"] > 1.0)
    )

    if invalid_probability.any():
        raise ValueError(
            "probability_dem contains values outside [0, 1]."
        )

    work["margin_error"] = (
        work["forecast_margin_dem"]
        - work["actual_margin_dem"]
    )

    work["absolute_margin_error"] = (
        work["margin_error"].abs()
    )

    work["squared_margin_error"] = (
        work["margin_error"] ** 2
    )

    work["actual_dem_win"] = (
        work["actual_margin_dem"] > 0.0
    )

    work["predicted_dem_win"] = (
        work["probability_dem"] >= 0.5
    )

    work["correct_winner"] = (
        work["predicted_dem_win"]
        == work["actual_dem_win"]
    )

    work["brier_score"] = (
        work["probability_dem"]
        - work["actual_dem_win"].astype(float)
    ) ** 2

    actual_dem_wins = int(
        work["actual_dem_win"].sum()
    )

    predicted_dem_wins = int(
        work["predicted_dem_win"].sum()
    )

    expected_dem_wins = float(
        work["probability_dem"].sum()
    )

    return ForecastMetrics(
        scored_races=int(len(work)),
        mean_margin_error_dem_bias=float(
            work["margin_error"].mean()
        ),
        mean_absolute_error=float(
            work["absolute_margin_error"].mean()
        ),
        median_absolute_error=float(
            work["absolute_margin_error"].median()
        ),
        rmse=float(
            np.sqrt(
                work["squared_margin_error"].mean()
            )
        ),
        winner_accuracy=float(
            work["correct_winner"].mean()
        ),
        brier_score=float(
            work["brier_score"].mean()
        ),
        log_loss=safe_log_loss(
            work["actual_dem_win"].astype(float),
            work["probability_dem"],
        ),
        actual_dem_wins=actual_dem_wins,
        predicted_dem_wins=predicted_dem_wins,
        expected_dem_wins=expected_dem_wins,
        predicted_win_count_error=(
            predicted_dem_wins - actual_dem_wins
        ),
        expected_win_count_error=(
            expected_dem_wins - actual_dem_wins
        ),
    )


def add_race_diagnostics(
    frame: pd.DataFrame,
    *,
    actual_margin_column: str,
    forecast_margin_column: str,
    probability_column: str | None = None,
    error_sd: float = DEFAULT_PROBABILITY_SCALE,
) -> pd.DataFrame:
    """Add canonical race-level error and outcome diagnostics."""
    required = {
        actual_margin_column,
        forecast_margin_column,
    }

    if probability_column is not None:
        required.add(probability_column)

    missing = sorted(required - set(frame.columns))

    if missing:
        raise KeyError(
            f"Missing required diagnostic columns: {missing}"
        )

    output = frame.copy()

    output["actual_margin_dem_scored"] = pd.to_numeric(
        output[actual_margin_column],
        errors="coerce",
    )

    output["forecast_margin_dem_scored"] = pd.to_numeric(
        output[forecast_margin_column],
        errors="coerce",
    )

    if probability_column is None:
        output["dem_win_probability_scored"] = (
            logistic_probability(
                output["forecast_margin_dem_scored"],
                error_sd=error_sd,
            )
        )
    else:
        output["dem_win_probability_scored"] = pd.to_numeric(
            output[probability_column],
            errors="coerce",
        )

    output["actual_dem_win"] = (
        output["actual_margin_dem_scored"] > 0.0
    )

    output["predicted_dem_win"] = (
        output["dem_win_probability_scored"] >= 0.5
    )

    output["correct_winner"] = (
        output["predicted_dem_win"]
        == output["actual_dem_win"]
    )

    output["margin_error"] = (
        output["forecast_margin_dem_scored"]
        - output["actual_margin_dem_scored"]
    )

    output["absolute_margin_error"] = (
        output["margin_error"].abs()
    )

    output["squared_margin_error"] = (
        output["margin_error"] ** 2
    )

    output["brier_score"] = (
        output["dem_win_probability_scored"]
        - output["actual_dem_win"].astype(float)
    ) ** 2

    return output


def build_calibration_table(
    actual_dem_win: pd.Series | np.ndarray | list[float],
    probability_dem: pd.Series | np.ndarray | list[float],
    *,
    model_name: str,
    bins: int = 10,
) -> pd.DataFrame:
    """Build a probability-calibration table."""
    if bins < 2:
        raise ValueError("bins must be at least 2.")

    work = pd.DataFrame(
        {
            "actual_dem_win": pd.to_numeric(
                pd.Series(actual_dem_win),
                errors="coerce",
            ),
            "probability_dem": pd.to_numeric(
                pd.Series(probability_dem),
                errors="coerce",
            ),
        }
    ).dropna()

    if work.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "probability_bucket",
                "races",
                "average_dem_probability",
                "actual_dem_win_rate",
                "calibration_error",
                "bucket_brier_score",
            ]
        )

    edges = np.linspace(0.0, 1.0, bins + 1)

    labels = [
        f"{int(edges[index] * 100)}–"
        f"{int(edges[index + 1] * 100)}%"
        for index in range(len(edges) - 1)
    ]

    work["probability_bucket"] = pd.cut(
        work["probability_dem"],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    rows: list[dict[str, object]] = []

    for bucket, group in work.groupby(
        "probability_bucket",
        observed=False,
    ):
        if group.empty:
            continue

        average_probability = float(
            group["probability_dem"].mean()
        )

        actual_win_rate = float(
            group["actual_dem_win"].mean()
        )

        rows.append(
            {
                "model_name": model_name,
                "probability_bucket": str(bucket),
                "races": int(len(group)),
                "average_dem_probability": (
                    average_probability
                ),
                "actual_dem_win_rate": actual_win_rate,
                "calibration_error": (
                    average_probability - actual_win_rate
                ),
                "bucket_brier_score": float(
                    (
                        group["probability_dem"]
                        - group["actual_dem_win"]
                    ).pow(2).mean()
                ),
            }
        )

    return pd.DataFrame(rows)
