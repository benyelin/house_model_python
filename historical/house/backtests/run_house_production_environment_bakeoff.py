from __future__ import annotations

from pathlib import Path
import glob
import json

import numpy as np
import pandas as pd

import run_house_baseline_environment_bakeoff as baseline_bakeoff


INPUT_PATH = Path(
    "historical/house/warehouse/"
    "house_historical_backtest_inputs_2016_2022.csv"
)

PRESIDENTIAL_MARGIN_LOOKUP = Path(
    "historical/warehouse/raw/national_environment/"
    "house_presidential_baseline_national_margins.csv"
)

ENVIRONMENT_GLOB = (
    "historical/warehouse/processed/national_environment/"
    "house_*_election_day_national_environment.csv"
)

OUTPUT_DIR = Path(
    "historical/house/backtests/outputs/"
    "production_environment_bakeoff"
)

BASELINE_TYPES = [
    "raw_presidential_margin_dem",
    "normalized_partisan_baseline_dem",
]

MULTIPLIERS = np.round(np.arange(0.50, 1.201, 0.05), 2)

PROBABILITY_SCALE = 6.0

REQUIRED_INPUT_COLUMNS = {
    "cycle",
    "district_pres_margin_dem",
    "actual_dem_margin",
}

BASELINE_LABELS = {
    "raw_presidential_margin_dem": "Raw presidential margin",
    "normalized_partisan_baseline_dem": "Normalized partisan baseline",
}


def logistic_probability(margin: pd.Series) -> pd.Series:
    clipped = np.clip(
        pd.to_numeric(margin, errors="coerce"),
        -50.0,
        50.0,
    )
    return 1.0 / (1.0 + np.exp(-clipped / PROBABILITY_SCALE))


def load_inputs() -> pd.DataFrame:
    """
    Reuse the established baseline bakeoff loader so this experiment
    uses the identical validated race universe, national presidential
    margin lookup, filtering, and normalized baseline construction.
    """
    df = baseline_bakeoff.load_inputs().copy()

    required = {
        "cycle",
        "district_pres_margin_dem",
        "district_partisan_baseline_dem",
        "actual_dem_margin",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Established bakeoff inputs missing required columns: "
            f"{sorted(missing)}"
        )

    df["raw_presidential_margin_dem"] = pd.to_numeric(
        df["district_pres_margin_dem"],
        errors="raise",
    )

    df["normalized_partisan_baseline_dem"] = pd.to_numeric(
        df["district_partisan_baseline_dem"],
        errors="raise",
    )

    if "district_elasticity" not in df.columns:
        df["district_elasticity"] = 1.0
    else:
        df["district_elasticity"] = pd.to_numeric(
            df["district_elasticity"],
            errors="coerce",
        ).fillna(1.0)

    return df


def load_composite_environments() -> pd.DataFrame:
    rows = []

    for filename in sorted(glob.glob(ENVIRONMENT_GLOB)):
        frame = pd.read_csv(filename)

        required = {
            "cycle",
            "national_environment_margin_dem",
        }
        missing = required - set(frame.columns)

        if missing:
            raise ValueError(
                f"{filename} missing columns: {sorted(missing)}"
            )

        row = frame.iloc[-1].copy()

        rows.append(
            {
                "cycle": int(row["cycle"]),
                "composite_environment_margin_dem": float(
                    row["national_environment_margin_dem"]
                ),
                "generic_ballot_margin_dem": float(
                    row.get("generic_ballot_margin_dem", np.nan)
                ),
                "approval_adjustment_dem": float(
                    row.get("approval_adjustment_dem", np.nan)
                ),
                "midterm_adjustment_dem": float(
                    row.get("midterm_adjustment_dem", np.nan)
                ),
                "environment_source_file": filename,
            }
        )

    if not rows:
        raise FileNotFoundError(
            f"No historical environment files matched {ENVIRONMENT_GLOB}"
        )

    environments = pd.DataFrame(rows)

    if environments["cycle"].duplicated().any():
        duplicates = environments.loc[
            environments["cycle"].duplicated(False),
            "cycle",
        ].tolist()
        raise ValueError(
            f"Duplicate historical environment cycles: {duplicates}"
        )

    return environments


def calculate_metrics(frame: pd.DataFrame) -> dict[str, float]:
    actual = frame["actual_dem_margin"].to_numpy(dtype=float)
    forecast = frame["forecast_margin_dem"].to_numpy(dtype=float)
    probability = frame["dem_win_probability"].to_numpy(dtype=float)

    error = forecast - actual

    actual_dem_win = (actual > 0.0).astype(float)
    predicted_dem_win = (forecast > 0.0).astype(float)

    actual_dem_seats = float(actual_dem_win.sum())
    expected_dem_seats = float(probability.sum())

    return {
        "observations": int(len(frame)),
        "mean_absolute_error": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "brier_score": float(
            np.mean(np.square(probability - actual_dem_win))
        ),
        "winner_accuracy": float(
            np.mean(predicted_dem_win == actual_dem_win)
        ),
        "mean_margin_error_dem_bias": float(np.mean(error)),
        "absolute_margin_bias": float(abs(np.mean(error))),
        "actual_dem_seats": actual_dem_seats,
        "expected_dem_seats": expected_dem_seats,
        "expected_win_count_error": float(
            abs(expected_dem_seats - actual_dem_seats)
        ),
    }


def percentile_rank(series: pd.Series) -> pd.Series:
    if series.nunique(dropna=True) <= 1:
        return pd.Series(0.0, index=series.index)

    return series.rank(
        method="average",
        pct=True,
        ascending=True,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs()
    environments = load_composite_environments()

    df = inputs.merge(
        environments,
        on="cycle",
        how="left",
        validate="many_to_one",
    )

    if df["composite_environment_margin_dem"].isna().any():
        missing_cycles = sorted(
            df.loc[
                df["composite_environment_margin_dem"].isna(),
                "cycle",
            ].unique()
        )
        raise ValueError(
            "Missing composite environments for cycles: "
            f"{missing_cycles}"
        )

    summary_rows = []
    cycle_rows = []
    prediction_frames = []

    for baseline_type in BASELINE_TYPES:
        baseline_col = baseline_type

        for multiplier in MULTIPLIERS:
            specification_id = (
                f"{baseline_type}"
                f"__production_composite"
                f"__multiplier_{multiplier:.2f}"
            )

            calculated = df.copy()

            calculated["selected_baseline_margin_dem"] = (
                calculated[baseline_col]
            )

            calculated["house_environment_multiplier"] = (
                float(multiplier)
            )

            calculated["district_environment_adjustment_dem"] = (
                calculated["composite_environment_margin_dem"]
                * calculated["house_environment_multiplier"]
                * calculated["district_elasticity"]
            )

            calculated["forecast_margin_dem"] = (
                calculated["selected_baseline_margin_dem"]
                + calculated["district_environment_adjustment_dem"]
            )

            calculated["dem_win_probability"] = (
                logistic_probability(
                    calculated["forecast_margin_dem"]
                )
            )

            overall_metrics = calculate_metrics(calculated)

            summary_rows.append(
                {
                    "specification_id": specification_id,
                    "baseline_type": baseline_type,
                    "baseline_label": BASELINE_LABELS[
                        baseline_type
                    ],
                    "environment_family": "production_composite",
                    "house_environment_multiplier": float(
                        multiplier
                    ),
                    **overall_metrics,
                }
            )

            for cycle, cycle_frame in calculated.groupby(
                "cycle",
                sort=True,
            ):
                cycle_metrics = calculate_metrics(cycle_frame)

                cycle_rows.append(
                    {
                        "specification_id": specification_id,
                        "baseline_type": baseline_type,
                        "baseline_label": BASELINE_LABELS[
                            baseline_type
                        ],
                        "environment_family": (
                            "production_composite"
                        ),
                        "house_environment_multiplier": float(
                            multiplier
                        ),
                        "cycle": int(cycle),
                        **cycle_metrics,
                    }
                )

            prediction_keep = [
                "cycle",
                "district_pres_margin_dem",
                "national_pres_margin_dem",
                "raw_presidential_margin_dem",
                "normalized_partisan_baseline_dem",
                "selected_baseline_margin_dem",
                "district_elasticity",
                "composite_environment_margin_dem",
                "house_environment_multiplier",
                "district_environment_adjustment_dem",
                "forecast_margin_dem",
                "actual_dem_margin",
                "dem_win_probability",
            ]

            for optional in [
                "district",
                "district_id",
                "presidential_result_year",
            ]:
                if optional in calculated.columns:
                    prediction_keep.insert(1, optional)

            prediction_output = calculated[
                prediction_keep
            ].copy()

            prediction_output.insert(
                0,
                "specification_id",
                specification_id,
            )
            prediction_output.insert(
                1,
                "baseline_type",
                baseline_type,
            )

            prediction_frames.append(prediction_output)

    summary = pd.DataFrame(summary_rows)
    by_cycle = pd.DataFrame(cycle_rows)
    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    cycle_robustness = (
        by_cycle.groupby(
            [
                "specification_id",
                "baseline_type",
                "baseline_label",
                "house_environment_multiplier",
            ],
            as_index=False,
        )
        .agg(
            max_cycle_mae=("mean_absolute_error", "max"),
            mean_cycle_mae=("mean_absolute_error", "mean"),
            max_cycle_rmse=("rmse", "max"),
            max_cycle_absolute_bias=("absolute_margin_bias", "max"),
            mean_cycle_seat_error=(
                "expected_win_count_error",
                "mean",
            ),
            max_cycle_seat_error=(
                "expected_win_count_error",
                "max",
            ),
        )
    )

    summary = summary.merge(
        cycle_robustness,
        on=[
            "specification_id",
            "baseline_type",
            "baseline_label",
            "house_environment_multiplier",
        ],
        how="left",
        validate="one_to_one",
    )

    ranking_metrics = [
        "mean_absolute_error",
        "rmse",
        "brier_score",
        "absolute_margin_bias",
        "expected_win_count_error",
        "max_cycle_mae",
        "max_cycle_absolute_bias",
    ]

    for metric in ranking_metrics:
        summary[f"{metric}_rank_pct"] = percentile_rank(
            summary[metric]
        )

    summary["joint_performance_score"] = summary[
        [f"{metric}_rank_pct" for metric in ranking_metrics]
    ].mean(axis=1)

    summary = summary.sort_values(
        [
            "joint_performance_score",
            "mean_absolute_error",
            "rmse",
            "brier_score",
            "absolute_margin_bias",
        ]
    ).reset_index(drop=True)

    best_by_baseline = (
        summary.sort_values(
            [
                "baseline_type",
                "joint_performance_score",
                "mean_absolute_error",
                "rmse",
            ]
        )
        .groupby("baseline_type", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    normalized = summary.loc[
        summary["baseline_type"].eq(
            "normalized_partisan_baseline_dem"
        )
    ].copy()

    best_normalized_score = normalized[
        "joint_performance_score"
    ].min()

    normalized_plateau = normalized.loc[
        normalized["joint_performance_score"]
        <= best_normalized_score + 0.05
    ].copy()

    normalized_plateau = normalized_plateau.sort_values(
        [
            "joint_performance_score",
            "absolute_margin_bias",
            "mean_absolute_error",
        ]
    )

    recommendation = normalized_plateau.iloc[[0]].copy()

    summary.to_csv(
        OUTPUT_DIR
        / "house_production_environment_bakeoff_summary.csv",
        index=False,
    )

    by_cycle.to_csv(
        OUTPUT_DIR
        / "house_production_environment_bakeoff_by_cycle.csv",
        index=False,
    )

    predictions.to_csv(
        OUTPUT_DIR
        / "house_production_environment_bakeoff_predictions.csv",
        index=False,
    )

    best_by_baseline.to_csv(
        OUTPUT_DIR
        / "house_production_environment_bakeoff_best_by_baseline.csv",
        index=False,
    )

    normalized_plateau.to_csv(
        OUTPUT_DIR
        / "house_production_environment_bakeoff_normalized_plateau.csv",
        index=False,
    )

    recommendation.to_csv(
        OUTPUT_DIR
        / "house_production_environment_bakeoff_recommendation.csv",
        index=False,
    )

    config = {
        "input_path": str(INPUT_PATH),
        "presidential_margin_lookup": str(
            PRESIDENTIAL_MARGIN_LOOKUP
        ),
        "environment_glob": ENVIRONMENT_GLOB,
        "baseline_types": BASELINE_TYPES,
        "multipliers": [float(x) for x in MULTIPLIERS],
        "probability_scale": PROBABILITY_SCALE,
        "forecast_formula": (
            "baseline + composite_environment * "
            "house_environment_multiplier * district_elasticity"
        ),
    }

    with open(
        OUTPUT_DIR
        / "house_production_environment_bakeoff_config.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(config, handle, indent=2)

    validation_lines = [
        "House production environment bakeoff: PASSED",
        f"Input observations: {len(df):,}",
        f"Cycles: {sorted(df['cycle'].unique().tolist())}",
        f"Specifications: {len(summary):,}",
        (
            "Recommended normalized multiplier: "
            f"{recommendation.iloc[0]['house_environment_multiplier']:.2f}"
        ),
        (
            "Recommended normalized MAE: "
            f"{recommendation.iloc[0]['mean_absolute_error']:.6f}"
        ),
        (
            "Recommended normalized RMSE: "
            f"{recommendation.iloc[0]['rmse']:.6f}"
        ),
        (
            "Recommended normalized Brier: "
            f"{recommendation.iloc[0]['brier_score']:.6f}"
        ),
    ]

    (
        OUTPUT_DIR
        / "house_production_environment_bakeoff_validation.txt"
    ).write_text(
        "\n".join(validation_lines) + "\n",
        encoding="utf-8",
    )

    display_cols = [
        "baseline_label",
        "house_environment_multiplier",
        "mean_absolute_error",
        "rmse",
        "brier_score",
        "winner_accuracy",
        "mean_margin_error_dem_bias",
        "expected_win_count_error",
        "max_cycle_mae",
        "joint_performance_score",
    ]

    print()
    print("House production environment bakeoff")
    print("=" * 110)
    print(f"Observations: {len(df):,}")
    print(f"Cycles: {sorted(df['cycle'].unique().tolist())}")
    print(f"Specifications: {len(summary):,}")
    print()
    print("Historical composite environments")
    print("=" * 110)
    print(
        environments[
            [
                "cycle",
                "generic_ballot_margin_dem",
                "approval_adjustment_dem",
                "midterm_adjustment_dem",
                "composite_environment_margin_dem",
            ]
        ].to_string(index=False)
    )
    print()
    print("Best specification by baseline")
    print("=" * 110)
    print(
        best_by_baseline[display_cols].to_string(
            index=False
        )
    )
    print()
    print("Recommended normalized specification")
    print("=" * 110)
    print(
        recommendation[display_cols].to_string(
            index=False
        )
    )
    print()
    print("Normalized joint-performance plateau")
    print("=" * 110)
    print(
        normalized_plateau[display_cols].to_string(
            index=False
        )
    )
    print()
    print(
        "House production environment bakeoff validation: "
        "PASSED"
    )


if __name__ == "__main__":
    main()
