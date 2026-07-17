from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_MASTER_PATH = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "warehouse"
    / "house_historical_backtest_inputs_2016_2022.csv"
)

SUPPORTED_CYCLES = (2016, 2018, 2020, 2022)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical/house/backtests/outputs"
)


COMPONENT_DEFAULTS = {
    "district_pres_margin_dem": 0.0,
    "national_environment_margin_dem": 0.0,
    "environment_multiplier": 1.0,
    "district_elasticity": 1.0,
    "state_environment_adjustment_dem": 0.0,
    "incumbency_adjustment_dem": 0.0,
    "candidate_quality_adjustment_dem": 0.0,
    "special_adjustment_dem": 0.0,
    "polling_adjustment_dem": 0.0,
}


CORE_RESULT_COLUMNS = [
    "cycle",
    "race_id",
    "actual_dem_margin",
    "actual_winner",
    "general_election_party_structure",
]


def parse_bool_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


def logistic_probability(
    margin: pd.Series,
    error_scale: pd.Series,
) -> pd.Series:
    scale = pd.to_numeric(
        error_scale,
        errors="coerce",
    ).fillna(6.5).clip(lower=0.5)

    values = pd.to_numeric(
        margin,
        errors="coerce",
    )

    z = (values / scale).clip(lower=-40, upper=40)

    return 1.0 / (1.0 + np.exp(-z))


def safe_log_loss(
    actual: pd.Series,
    probability: pd.Series,
) -> float:
    y = pd.to_numeric(actual, errors="coerce")
    p = pd.to_numeric(probability, errors="coerce")

    valid = y.notna() & p.notna()

    if not valid.any():
        return math.nan

    y = y.loc[valid].astype(float)
    p = p.loc[valid].clip(1e-9, 1.0 - 1e-9)

    return float(
        -(
            y * np.log(p)
            + (1.0 - y) * np.log(1.0 - p)
        ).mean()
    )


def build_model_margin(
    df: pd.DataFrame,
) -> tuple[pd.Series | None, str]:
    if "model_margin_dem" in df.columns:
        margin = pd.to_numeric(
            df["model_margin_dem"],
            errors="coerce",
        )

        return margin, "precomputed_model_margin_dem"

    required_components = {
        "district_pres_margin_dem",
        "national_environment_margin_dem",
    }

    if not required_components.issubset(df.columns):
        return None, "missing_required_forecast_components"

    required_numeric = {
        column: pd.to_numeric(
            df[column],
            errors="coerce",
        )
        for column in required_components
    }

    required_coverage = {
        column: int(series.notna().sum())
        for column, series in required_numeric.items()
    }

    if any(count == 0 for count in required_coverage.values()):
        return None, "required_forecast_components_empty"

    component_data = {}

    for column, default in COMPONENT_DEFAULTS.items():
        if column in df.columns:
            component_data[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            ).fillna(default)
        else:
            component_data[column] = pd.Series(
                default,
                index=df.index,
                dtype=float,
            )

    environment_adjustment = (
        component_data["national_environment_margin_dem"]
        * component_data["environment_multiplier"]
        * component_data["district_elasticity"]
    )

    margin = (
        component_data["district_pres_margin_dem"]
        + environment_adjustment
        + component_data["state_environment_adjustment_dem"]
        + component_data["incumbency_adjustment_dem"]
        + component_data["candidate_quality_adjustment_dem"]
        + component_data["special_adjustment_dem"]
        + component_data["polling_adjustment_dem"]
    )

    return margin, "calculated_from_components"


def build_scoring_mask(df: pd.DataFrame) -> pd.Series:
    """
    Return the cycle-safe canonical scoring mask when available.

    The canonical flag incorporates ordinary major-party scoring,
    required observed inputs, and documented historical exceptions such
    as the unavailable leakage-free presidential baseline for Florida
    in 2016.
    """
    if "include_in_canonical_margin_backtest" in df.columns:
        return parse_bool_series(
            df["include_in_canonical_margin_backtest"]
        )

    if "include_in_major_party_margin_scoring" in df.columns:
        return parse_bool_series(
            df["include_in_major_party_margin_scoring"]
        )

    if "general_election_party_structure" in df.columns:
        return (
            df["general_election_party_structure"]
            .fillna("")
            .eq("D_vs_R")
        )

    return pd.Series(
        True,
        index=df.index,
        dtype=bool,
    )


def make_calibration_table(
    scored: pd.DataFrame,
) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame(
            columns=[
                "probability_bucket",
                "bucket_lower",
                "bucket_upper",
                "races",
                "avg_dem_probability",
                "actual_dem_win_rate",
                "calibration_error",
                "brier_score",
            ]
        )

    bins = np.linspace(0.0, 1.0, 11)

    labels = [
        f"{int(bins[i] * 100)}–{int(bins[i + 1] * 100)}%"
        for i in range(len(bins) - 1)
    ]

    temp = scored.copy()

    temp["probability_bucket"] = pd.cut(
        temp["dem_win_probability"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    rows = []

    for bucket, group in temp.groupby(
        "probability_bucket",
        observed=False,
    ):
        if group.empty:
            continue

        bucket_index = labels.index(str(bucket))

        avg_probability = float(
            group["dem_win_probability"].mean()
        )

        actual_rate = float(
            group["actual_dem_win"].mean()
        )

        rows.append(
            {
                "probability_bucket": str(bucket),
                "bucket_lower": bins[bucket_index],
                "bucket_upper": bins[bucket_index + 1],
                "races": len(group),
                "avg_dem_probability": avg_probability,
                "actual_dem_win_rate": actual_rate,
                "calibration_error": (
                    avg_probability - actual_rate
                ),
                "brier_score": float(
                    group["brier_score"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def build_readiness_report(
    df: pd.DataFrame,
    forecast_source: str,
) -> str:
    lines = [
        "House Historical Backtest Data Readiness",
        "=" * 40,
        "",
        f"Rows loaded: {len(df)}",
        f"Unique race IDs: {df['race_id'].nunique()}",
        f"Forecast source: {forecast_source}",
        "",
        "Core result fields:",
    ]

    for column in CORE_RESULT_COLUMNS:
        if column not in df.columns:
            lines.append(f"- {column}: MISSING")
            continue

        missing = int(df[column].isna().sum())
        lines.append(
            f"- {column}: present; missing values={missing}"
        )

    lines.extend(
        [
            "",
            "Forecast component fields:",
        ]
    )

    for column in COMPONENT_DEFAULTS:
        if column in df.columns:
            nonmissing = int(
                pd.to_numeric(
                    df[column],
                    errors="coerce",
                ).notna().sum()
            )

            lines.append(
                f"- {column}: present; "
                f"numeric rows={nonmissing}/{len(df)}"
            )
        else:
            required_label = (
                " REQUIRED"
                if column in {
                    "district_pres_margin_dem",
                    "national_environment_margin_dem",
                }
                else ""
            )

            lines.append(
                f"- {column}: missing{required_label}"
            )

    if "model_margin_dem" in df.columns:
        valid_model_margins = int(
            pd.to_numeric(
                df["model_margin_dem"],
                errors="coerce",
            ).notna().sum()
        )

        lines.extend(
            [
                "",
                (
                    "Precomputed model margins: "
                    f"{valid_model_margins}/{len(df)}"
                ),
            ]
        )

    scoring_mask = build_scoring_mask(df)

    lines.extend(
        [
            "",
            (
                "Ordinary D-vs-R margin-scoring races: "
                f"{int(scoring_mask.sum())}"
            ),
            (
                "Excluded nonstandard races: "
                f"{int((~scoring_mask).sum())}"
            ),
        ]
    )

    return "\n".join(lines)


def score_backtest(
    df: pd.DataFrame,
    model_margin: pd.Series,
    default_error_sd: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = df.copy()

    results["model_margin_dem"] = pd.to_numeric(
        model_margin,
        errors="coerce",
    )

    if "total_error_sd" in results.columns:
        error_sd = pd.to_numeric(
            results["total_error_sd"],
            errors="coerce",
        ).fillna(default_error_sd)
    else:
        error_sd = pd.Series(
            default_error_sd,
            index=results.index,
            dtype=float,
        )

    results["total_error_sd_used"] = error_sd

    if "dem_win_probability" in results.columns:
        existing_probability = pd.to_numeric(
            results["dem_win_probability"],
            errors="coerce",
        )

        calculated_probability = logistic_probability(
            results["model_margin_dem"],
            error_sd,
        )

        results["dem_win_probability"] = (
            existing_probability
            .where(
                existing_probability.notna(),
                calculated_probability,
            )
            .clip(0.0, 1.0)
        )
    else:
        results["dem_win_probability"] = (
            logistic_probability(
                results["model_margin_dem"],
                error_sd,
            )
        )

    results["actual_dem_margin"] = pd.to_numeric(
        results["actual_dem_margin"],
        errors="coerce",
    )

    results["actual_dem_win"] = (
        results["actual_dem_margin"] > 0
    )

    results["predicted_dem_win"] = (
        results["dem_win_probability"] >= 0.5
    )

    results["correct_winner"] = (
        results["predicted_dem_win"]
        == results["actual_dem_win"]
    )

    results["margin_error"] = (
        results["model_margin_dem"]
        - results["actual_dem_margin"]
    )

    results["abs_margin_error"] = (
        results["margin_error"].abs()
    )

    results["squared_margin_error"] = (
        results["margin_error"] ** 2
    )

    results["brier_score"] = (
        results["dem_win_probability"]
        - results["actual_dem_win"].astype(float)
    ) ** 2

    results["include_in_scoring"] = (
        build_scoring_mask(results)
        & results["model_margin_dem"].notna()
        & results["actual_dem_margin"].notna()
    )

    scored = results.loc[
        results["include_in_scoring"]
    ].copy()

    if scored.empty:
        raise RuntimeError(
            "No races are eligible for scoring."
        )

    actual_dem_seats = int(
        results["actual_winner"]
        .fillna("")
        .eq("D")
        .sum()
    )

    predicted_dem_seats = int(
        results["predicted_dem_win"].sum()
    )

    expected_dem_seats = float(
        results["dem_win_probability"].sum()
    )

    summary = pd.DataFrame(
        [
            {
                "cycle": int(
                    pd.to_numeric(
                        results["cycle"],
                        errors="coerce",
                    ).dropna().iloc[0]
                ),
                "all_races": len(results),
                "scored_major_party_races": len(scored),
                "winner_accuracy": float(
                    scored["correct_winner"].mean()
                ),
                "mean_abs_margin_error": float(
                    scored["abs_margin_error"].mean()
                ),
                "median_abs_margin_error": float(
                    scored["abs_margin_error"].median()
                ),
                "rmse_margin_error": float(
                    np.sqrt(
                        scored[
                            "squared_margin_error"
                        ].mean()
                    )
                ),
                "mean_margin_error_dem_bias": float(
                    scored["margin_error"].mean()
                ),
                "brier_score": float(
                    scored["brier_score"].mean()
                ),
                "log_loss": safe_log_loss(
                    scored["actual_dem_win"],
                    scored["dem_win_probability"],
                ),
                "actual_dem_seats": actual_dem_seats,
                "predicted_dem_seats": predicted_dem_seats,
                "expected_dem_seats": expected_dem_seats,
                "predicted_seat_error": (
                    predicted_dem_seats
                    - actual_dem_seats
                ),
                "expected_seat_error": (
                    expected_dem_seats
                    - actual_dem_seats
                ),
            }
        ]
    )

    calibration = make_calibration_table(scored)

    return results, summary, calibration


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--master-path",
        type=Path,
        default=DEFAULT_MASTER_PATH,
        help=(
            "Canonical historical backtest warehouse or a compatible "
            "single-cycle input file."
        ),
    )

    parser.add_argument(
        "--cycle",
        type=int,
        choices=SUPPORTED_CYCLES,
        default=2022,
        help=(
            "Historical election cycle to score from the canonical "
            "multicycle warehouse. Default: 2022."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--default-error-sd",
        type=float,
        default=6.5,
    )

    parser.add_argument(
        "--readiness-only",
        action="store_true",
        help=(
            "Validate available data without requiring "
            "forecast inputs."
        ),
    )

    args = parser.parse_args()

    if not args.master_path.exists():
        raise FileNotFoundError(
            f"Historical master not found: "
            f"{args.master_path}"
        )

    loaded = pd.read_csv(
        args.master_path,
        low_memory=False,
    )

    # The canonical warehouse contains four cycles. Compatible legacy
    # single-cycle inputs remain supported.
    if "forecast_cycle" in loaded.columns:
        forecast_cycle = pd.to_numeric(
            loaded["forecast_cycle"],
            errors="coerce",
        )

        df = loaded.loc[
            forecast_cycle.eq(args.cycle)
        ].copy()

        if df.empty:
            available_cycles = sorted(
                forecast_cycle.dropna()
                .astype(int)
                .unique()
                .tolist()
            )

            raise ValueError(
                f"No rows found for cycle {args.cycle}. "
                f"Available cycles: {available_cycles}"
            )

        if "cycle" in df.columns:
            existing_cycle = pd.to_numeric(
                df["cycle"],
                errors="coerce",
            )

            inconsistent = (
                existing_cycle.notna()
                & existing_cycle.ne(args.cycle)
            )

            if inconsistent.any():
                examples = df.loc[
                    inconsistent,
                    ["forecast_cycle", "cycle", "race_id"],
                ].head(20)

                raise ValueError(
                    "Canonical forecast_cycle and cycle values are "
                    "inconsistent. Examples: "
                    f"{examples.to_dict('records')}"
                )
        else:
            df["cycle"] = args.cycle

    else:
        df = loaded.copy()

        if "cycle" not in df.columns:
            raise ValueError(
                "Input must contain forecast_cycle or cycle."
            )

        cycle_values = pd.to_numeric(
            df["cycle"],
            errors="coerce",
        ).dropna().unique()

        if len(cycle_values) != 1:
            raise ValueError(
                "Legacy input must contain exactly one cycle."
            )

        input_cycle = int(cycle_values[0])

        if input_cycle != args.cycle:
            raise ValueError(
                f"Requested cycle {args.cycle}, but legacy input "
                f"contains cycle {input_cycle}."
            )

    missing_core = [
        column
        for column in CORE_RESULT_COLUMNS
        if column not in df.columns
    ]

    if missing_core:
        raise ValueError(
            "Historical input is missing core fields: "
            + ", ".join(missing_core)
        )

    if len(df) != 435:
        raise ValueError(
            f"Expected 435 House races for {args.cycle}; "
            f"found {len(df)}."
        )

    if df["race_id"].duplicated().any():
        duplicates = (
            df.loc[
                df["race_id"].duplicated(False),
                "race_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Duplicate race IDs: "
            + ", ".join(duplicates)
        )

    # Canonical geography and ordering should be deterministic.
    sort_columns = [
        column
        for column in ("state", "district", "race_id")
        if column in df.columns
    ]

    if sort_columns:
        df = df.sort_values(
            sort_columns,
            kind="mergesort",
        ).reset_index(drop=True)
    else:
        df = df.sort_values(
            ["race_id"],
            kind="mergesort",
        ).reset_index(drop=True)

    model_margin, forecast_source = build_model_margin(df)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cycle_values = pd.to_numeric(
        df["cycle"],
        errors="coerce",
    ).dropna().unique()

    if len(cycle_values) != 1:
        raise ValueError(
            "Selected historical input must contain exactly one cycle."
        )

    cycle = int(cycle_values[0])

    if cycle != args.cycle:
        raise ValueError(
            f"Selected data cycle {cycle} does not match requested "
            f"cycle {args.cycle}."
        )

    readiness_text = build_readiness_report(
        df,
        forecast_source,
    )

    readiness_path = (
        args.output_dir
        / f"house_{cycle}_backtest_readiness.txt"
    )

    readiness_path.write_text(readiness_text)

    print(readiness_text)
    print()
    print(f"Wrote: {readiness_path}")

    if args.readiness_only:
        return

    if model_margin is None:
        raise RuntimeError(
            "\nForecast scoring cannot run yet because the "
            "historical master lacks either:\n"
            "1. a precomputed model_margin_dem column, or\n"
            "2. district_pres_margin_dem and "
            "national_environment_margin_dem.\n\n"
            "Run with --readiness-only for a data audit."
        )

    results, summary, calibration = score_backtest(
        df=df,
        model_margin=model_margin,
        default_error_sd=args.default_error_sd,
    )

    results_path = (
        args.output_dir
        / f"house_{cycle}_backtest_results.csv"
    )

    summary_path = (
        args.output_dir
        / f"house_{cycle}_backtest_summary.csv"
    )

    calibration_path = (
        args.output_dir
        / f"house_{cycle}_backtest_calibration.csv"
    )

    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    calibration.to_csv(calibration_path, index=False)

    print()
    print("House historical backtest summary")
    print("---------------------------------")
    print(summary.to_string(index=False))

    print()
    print("Probability calibration")
    print("-----------------------")
    print(calibration.to_string(index=False))

    print()
    print("Worst margin misses")
    print("-------------------")

    show_columns = [
        "race_id",
        "model_margin_dem",
        "actual_dem_margin",
        "margin_error",
        "dem_win_probability",
        "actual_winner",
        "correct_winner",
    ]

    print(
        results.loc[
            results["include_in_scoring"]
        ]
        .sort_values(
            "abs_margin_error",
            ascending=False,
        )
        .head(25)[show_columns]
        .to_string(index=False)
    )

    print()
    print(f"Wrote: {results_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {calibration_path}")


if __name__ == "__main__":
    main()
