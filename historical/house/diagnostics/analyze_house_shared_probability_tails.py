from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "full_production_replay"
    / "house_production_replay_predictions.csv"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "full_production_replay"
    / "house_production_replay_config.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "diagnostics"
    / "outputs"
    / "shared_probability_tail_audit"
)

SHARED_SPEC = "production_shared_uncertainty_v2"
EPSILON = 1e-15


class AuditError(RuntimeError):
    pass


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    numeric_values = pd.to_numeric(
        series,
        errors="coerce",
    )

    text_values = (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return (
        numeric_values.eq(1)
        | text_values.isin(
            {
                "true",
                "t",
                "yes",
                "y",
                "1",
                "d",
                "dem",
            }
        )
    )


def find_column(
    frame: pd.DataFrame,
    candidates: list[str],
    *,
    required: bool = True,
) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    if required:
        raise AuditError(
            "None of the expected columns were found: "
            + ", ".join(candidates)
        )

    return None


def row_log_loss(
    actual: pd.Series,
    probability: pd.Series,
) -> pd.Series:
    y = numeric(actual)
    p = numeric(probability).clip(
        EPSILON,
        1.0 - EPSILON,
    )

    return -(
        y * np.log(p)
        + (1.0 - y) * np.log(1.0 - p)
    )


def row_brier(
    actual: pd.Series,
    probability: pd.Series,
) -> pd.Series:
    y = numeric(actual)
    p = numeric(probability)

    return (p - y) ** 2


def infer_n_sims(
    probabilities: pd.Series,
) -> int | None:
    """
    Infer a likely simulation count from the probability grid.

    If probabilities come from k / N, then their nonzero differences
    should usually be integer multiples of 1 / N.
    """
    values = (
        numeric(probabilities)
        .dropna()
        .clip(0.0, 1.0)
        .unique()
    )

    values = np.sort(values)

    if len(values) < 2:
        return None

    differences = np.diff(values)
    differences = differences[
        differences > 1e-12
    ]

    if len(differences) == 0:
        return None

    candidates = [
        1000,
        2000,
        5000,
        10000,
        20000,
        50000,
        100000,
    ]

    best_n = None
    best_error = math.inf

    for n_sims in candidates:
        scaled = values * n_sims
        rounding_error = np.abs(
            scaled - np.round(scaled)
        )

        error = float(
            np.nanmedian(rounding_error)
        )

        if error < best_error:
            best_error = error
            best_n = n_sims

    if best_error > 1e-6:
        return None

    return best_n


def add_smoothed_probabilities(
    frame: pd.DataFrame,
    n_sims: int,
) -> pd.DataFrame:
    out = frame.copy()

    p = numeric(
        out["probability_shared"]
    ).clip(0.0, 1.0)

    wins_estimated = np.rint(
        p * n_sims
    ).astype("Int64")

    out["estimated_dem_sim_wins"] = (
        wins_estimated
    )

    out[
        "probability_add_half_smoothing"
    ] = (
        wins_estimated.astype(float) + 0.5
    ) / (n_sims + 1.0)

    out[
        "probability_laplace_smoothing"
    ] = (
        wins_estimated.astype(float) + 1.0
    ) / (n_sims + 2.0)

    return out


def summarize_variant(
    frame: pd.DataFrame,
    probability_column: str,
    label: str,
) -> dict[str, object]:
    probability = numeric(
        frame[probability_column]
    )

    return {
        "variant": label,
        "scored_races": int(len(frame)),
        "mean_probability": float(
            probability.mean()
        ),
        "expected_dem_seats": float(
            probability.sum()
        ),
        "log_loss": float(
            row_log_loss(
                frame["_actual_dem_win"],
                probability,
            ).mean()
        ),
        "brier_score": float(
            row_brier(
                frame["_actual_dem_win"],
                probability,
            ).mean()
        ),
        "winner_accuracy": float(
            (
                probability.ge(0.5)
                == frame[
                    "_actual_dem_win"
                ].astype(bool)
            ).mean()
        ),
        "exact_zero_count": int(
            probability.eq(0.0).sum()
        ),
        "exact_one_count": int(
            probability.eq(1.0).sum()
        ),
        "below_0_001_count": int(
            probability.lt(0.001).sum()
        ),
        "above_0_999_count": int(
            probability.gt(0.999).sum()
        ),
        "below_0_01_count": int(
            probability.lt(0.01).sum()
        ),
        "above_0_99_count": int(
            probability.gt(0.99).sum()
        ),
        "below_0_05_count": int(
            probability.lt(0.05).sum()
        ),
        "above_0_95_count": int(
            probability.gt(0.95).sum()
        ),
    }


def main() -> None:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            PREDICTIONS_PATH
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        low_memory=False,
    )

    required = {
        "replay_spec",
        "cycle",
    }

    missing = required - set(
        predictions.columns
    )

    if missing:
        raise AuditError(
            "Replay predictions are missing: "
            + ", ".join(sorted(missing))
        )

    race_id_column = find_column(
        predictions,
        [
            "district_id",
            "race_id",
        ],
    )

    scoring_column = find_column(
        predictions,
        [
            "include_in_scoring",
            "scorable",
        ],
        required=False,
    )

    actual_column = find_column(
        predictions,
        [
            "actual_dem_win",
            "actual_winner_dem",
            "dem_won",
            "actual_winner",
        ],
    )

    probability_column = find_column(
        predictions,
        [
            "simulated_dem_win_probability",
            "dem_win_probability",
        ],
    )

    shared = predictions.loc[
        predictions[
            "replay_spec"
        ].eq(SHARED_SPEC)
    ].copy()

    if scoring_column is not None:
        shared = shared.loc[
            parse_bool(
                shared[scoring_column]
            )
        ].copy()

    if shared.empty:
        raise AuditError(
            "No scored shared-simulator rows found."
        )

    if actual_column == "actual_winner":
        shared["_actual_dem_win"] = (
            shared[actual_column]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
            .eq("D")
            .astype(float)
        )
    else:
        shared["_actual_dem_win"] = (
            parse_bool(
                shared[actual_column]
            ).astype(float)
        )

    shared[
        "probability_shared"
    ] = numeric(
        shared[probability_column]
    )

    if shared[
        "probability_shared"
    ].isna().any():
        raise AuditError(
            "Missing shared probabilities found."
        )

    inferred_n_sims = infer_n_sims(
        shared["probability_shared"]
    )

    if inferred_n_sims is None:
        inferred_n_sims = 20000
        inference_note = (
            "Could not infer n_sims cleanly; "
            "used 20000 as replay default."
        )
    else:
        inference_note = (
            "n_sims inferred from probability grid."
        )

    shared = add_smoothed_probabilities(
        shared,
        inferred_n_sims,
    )

    shared[
        "raw_log_loss"
    ] = row_log_loss(
        shared["_actual_dem_win"],
        shared["probability_shared"],
    )

    shared[
        "add_half_log_loss"
    ] = row_log_loss(
        shared["_actual_dem_win"],
        shared[
            "probability_add_half_smoothing"
        ],
    )

    shared[
        "laplace_log_loss"
    ] = row_log_loss(
        shared["_actual_dem_win"],
        shared[
            "probability_laplace_smoothing"
        ],
    )

    shared[
        "raw_minus_add_half_log_loss"
    ] = (
        shared["raw_log_loss"]
        - shared["add_half_log_loss"]
    )

    shared[
        "raw_minus_laplace_log_loss"
    ] = (
        shared["raw_log_loss"]
        - shared["laplace_log_loss"]
    )

    shared[
        "is_exact_zero"
    ] = shared[
        "probability_shared"
    ].eq(0.0)

    shared[
        "is_exact_one"
    ] = shared[
        "probability_shared"
    ].eq(1.0)

    shared[
        "exact_tail_wrong"
    ] = (
        (
            shared["is_exact_zero"]
            & shared[
                "_actual_dem_win"
            ].eq(1.0)
        )
        | (
            shared["is_exact_one"]
            & shared[
                "_actual_dem_win"
            ].eq(0.0)
        )
    )

    tail_thresholds = [
        (
            "exact_zero",
            lambda p: p.eq(0.0),
        ),
        (
            "exact_one",
            lambda p: p.eq(1.0),
        ),
        (
            "below_0_001",
            lambda p: p.lt(0.001),
        ),
        (
            "above_0_999",
            lambda p: p.gt(0.999),
        ),
        (
            "below_0_01",
            lambda p: p.lt(0.01),
        ),
        (
            "above_0_99",
            lambda p: p.gt(0.99),
        ),
        (
            "below_0_05",
            lambda p: p.lt(0.05),
        ),
        (
            "above_0_95",
            lambda p: p.gt(0.95),
        ),
    ]

    tail_rows = []

    for cycle, frame in shared.groupby(
        "cycle",
        sort=True,
    ):
        probability = frame[
            "probability_shared"
        ]

        for label, selector in tail_thresholds:
            mask = selector(probability)

            wrong = (
                probability.ge(0.5)
                != frame[
                    "_actual_dem_win"
                ].astype(bool)
            )

            tail_rows.append(
                {
                    "cycle": int(cycle),
                    "tail_definition": label,
                    "race_count": int(
                        mask.sum()
                    ),
                    "wrong_winner_count": int(
                        (
                            mask & wrong
                        ).sum()
                    ),
                    "mean_raw_log_loss": float(
                        frame.loc[
                            mask,
                            "raw_log_loss",
                        ].mean()
                    )
                    if mask.any()
                    else math.nan,
                    "total_raw_log_loss": float(
                        frame.loc[
                            mask,
                            "raw_log_loss",
                        ].sum()
                    ),
                    "total_add_half_log_loss": float(
                        frame.loc[
                            mask,
                            "add_half_log_loss",
                        ].sum()
                    ),
                    "log_loss_reduction_add_half": float(
                        frame.loc[
                            mask,
                            "raw_minus_add_half_log_loss",
                        ].sum()
                    ),
                }
            )

    tail_summary = pd.DataFrame(
        tail_rows
    )

    variant_rows = []

    for cycle, frame in shared.groupby(
        "cycle",
        sort=True,
    ):
        for probability_column_name, label in [
            (
                "probability_shared",
                "raw_simulation_frequency",
            ),
            (
                "probability_add_half_smoothing",
                "add_half_smoothing",
            ),
            (
                "probability_laplace_smoothing",
                "laplace_smoothing",
            ),
        ]:
            row = summarize_variant(
                frame,
                probability_column_name,
                label,
            )
            row["cycle"] = int(cycle)
            variant_rows.append(row)

    for probability_column_name, label in [
        (
            "probability_shared",
            "raw_simulation_frequency",
        ),
        (
            "probability_add_half_smoothing",
            "add_half_smoothing",
        ),
        (
            "probability_laplace_smoothing",
            "laplace_smoothing",
        ),
    ]:
        row = summarize_variant(
            shared,
            probability_column_name,
            label,
        )
        row["cycle"] = "overall"
        variant_rows.append(row)

    variant_summary = pd.DataFrame(
        variant_rows
    )

    exact_tail_rows = shared.loc[
        shared[
            "is_exact_zero"
        ]
        | shared[
            "is_exact_one"
        ]
    ].copy()

    exact_tail_rows = exact_tail_rows.sort_values(
        [
            "exact_tail_wrong",
            "raw_log_loss",
        ],
        ascending=[
            False,
            False,
        ],
        kind="mergesort",
    )

    display_columns = [
        column
        for column in [
            "cycle",
            race_id_column,
            "state",
            "actual_dem_margin",
            "model_margin_dem",
            "_actual_dem_win",
            "probability_shared",
            "estimated_dem_sim_wins",
            "probability_add_half_smoothing",
            "raw_log_loss",
            "add_half_log_loss",
            "raw_minus_add_half_log_loss",
            "exact_tail_wrong",
        ]
        if column
        in exact_tail_rows.columns
    ]

    shared.to_csv(
        OUTPUT_DIR
        / "house_shared_probability_tail_race_audit.csv",
        index=False,
    )

    exact_tail_rows.to_csv(
        OUTPUT_DIR
        / "house_shared_exact_zero_one_races.csv",
        index=False,
    )

    tail_summary.to_csv(
        OUTPUT_DIR
        / "house_shared_probability_tail_counts.csv",
        index=False,
    )

    variant_summary.to_csv(
        OUTPUT_DIR
        / "house_shared_probability_smoothing_comparison.csv",
        index=False,
    )

    print(
        "HOUSE SHARED-SIMULATOR PROBABILITY-TAIL AUDIT"
    )
    print("=" * 100)
    print(f"Inferred simulations: {inferred_n_sims}")
    print(f"Inference note: {inference_note}")
    print()

    print("SCORING COMPARISON")
    print("-" * 100)
    print(
        variant_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.9f}"
            ),
        )
    )

    print()
    print("TAIL COUNTS BY CYCLE")
    print("-" * 100)
    print(
        tail_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print("EXACT ZERO/ONE RACES")
    print("-" * 100)
    print(
        exact_tail_rows[
            display_columns
        ]
        .head(100)
        .to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.9f}"
            ),
        )
    )

    raw_overall = variant_summary.loc[
        (
            variant_summary[
                "cycle"
            ].astype(str).eq("overall")
        )
        & variant_summary[
            "variant"
        ].eq(
            "raw_simulation_frequency"
        )
    ].iloc[0]

    half_overall = variant_summary.loc[
        (
            variant_summary[
                "cycle"
            ].astype(str).eq("overall")
        )
        & variant_summary[
            "variant"
        ].eq(
            "add_half_smoothing"
        )
    ].iloc[0]

    print()
    print("OVERALL EFFECT OF ADD-HALF SMOOTHING")
    print("-" * 100)
    print(
        "Log loss change: "
        f"{float(half_overall['log_loss']) - float(raw_overall['log_loss']):+.9f}"
    )
    print(
        "Brier change:    "
        f"{float(half_overall['brier_score']) - float(raw_overall['brier_score']):+.9f}"
    )
    print(
        "Expected seats:  "
        f"{float(raw_overall['expected_dem_seats']):.9f}"
        " -> "
        f"{float(half_overall['expected_dem_seats']):.9f}"
    )
    print(
        "Winner accuracy: "
        f"{float(raw_overall['winner_accuracy']):.9f}"
        " -> "
        f"{float(half_overall['winner_accuracy']):.9f}"
    )

    print()
    print("OUTPUTS")
    print("-" * 100)

    for output in sorted(
        OUTPUT_DIR.glob("*.csv")
    ):
        print(
            f"  {output.relative_to(PROJECT_ROOT)}"
        )

    print()
    print("VALIDATION PASSED")
    print(
        "Raw and smoothed probabilities were scored on the "
        "same historical race universe."
    )


if __name__ == "__main__":
    main()
