from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )


from historical.house.backtests.run_house_full_production_replay import (
    DEFAULT_MASTER_PATH,
    DEFAULT_WAR_PATH,
    build_model_margin,
    build_production_fundamentals,
    prepare_cycle,
    run_production_shared_spec,
)
from historical.house.polling.load_house_historical_polling_snapshot import (
    SUPPORTED_CYCLES,
    SUPPORTED_DAYS_OUT,
    load_house_historical_polling_snapshot,
)


FUNDAMENTALS_SPEC = (
    "historical_fundamentals_same_days_out"
)

POLLING_SPEC = (
    "historical_production_bayesian_polling"
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "historical_polling_experiment"
)

POLLING_COLUMNS = [
    "snapshot_date",
    "days_out",
    "polling_margin_dem",
    "poll_count",
    "polling_active",
    "latest_poll_end_date",
    "avg_poll_age_days",
    "total_poll_weight",
    "effective_poll_count",
    "largest_pollster_weight_share",
    "only_partisan_or_internal_polls",
    "polling_notes",
]

SUMMARY_METRICS = [
    "brier_score",
    "log_loss",
    "expected_dem_seats",
    "actual_dem_seats",
    "expected_seat_error",
    "expected_dem_seats_from_simulation",
    "simulation_dem_seat_sd",
    "median_dem_seats",
    "dem_seats_p25",
    "dem_seats_p50",
    "dem_seats_p75",
    "dem_control_probability",
    "average_polling_weight",
    "districts_with_polling",
]

DISTRICT_METRICS = [
    "model_margin_dem",
    "dem_win_probability",
    "raw_simulated_dem_win_probability",
    "simulated_dem_win_probability",
    "avg_simulated_margin_dem",
    "margin_p25_dem",
    "margin_p50_dem",
    "margin_p75_dem",
    "district_posterior_sd",
    "bayesian_polling_weight",
    "bayesian_fundamentals_weight",
    "fundamentals_margin_dem",
    "polling_margin_dem",
    "poll_count",
    "effective_poll_count",
    "poll_quality_count",
]


class HistoricalPollingExperimentError(
    RuntimeError
):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run matched fundamentals-only and "
            "production-Bayesian-polling House replays "
            "at historical polling snapshots."
        )
    )

    parser.add_argument(
        "--master-path",
        type=Path,
        default=DEFAULT_MASTER_PATH,
    )

    parser.add_argument(
        "--candidate-war-path",
        type=Path,
        default=DEFAULT_WAR_PATH,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--sims",
        type=int,
        default=20000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260804,
    )

    parser.add_argument(
        "--fixed-error-sd",
        type=float,
        default=6.5,
    )

    parser.add_argument(
        "--clean-output",
        action="store_true",
    )

    return parser.parse_args()


def normalize_bool(
    value: object,
) -> bool:
    if value is None or pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    return (
        str(value)
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
            "y",
        }
    )


def neutralize_polling(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a no-polling arm that still passes through the live
    prepare_house_table() function at the historical days-out value.
    """
    out = frame.copy()

    out["polling_margin_dem"] = np.nan
    out["poll_count"] = 0.0
    out["polling_active"] = False
    out["latest_poll_end_date"] = ""
    out["avg_poll_age_days"] = np.nan
    out["total_poll_weight"] = 0.0
    out["effective_poll_count"] = 0.0
    out["largest_pollster_weight_share"] = 0.0
    out[
        "only_partisan_or_internal_polls"
    ] = False
    out["polling_notes"] = ""

    return out


def attach_polling_snapshot(
    frame: pd.DataFrame,
    snapshot: pd.DataFrame,
) -> pd.DataFrame:
    out = frame.copy()

    if "district_id" not in out.columns:
        raise HistoricalPollingExperimentError(
            "Historical replay dataframe lacks district_id."
        )

    stale_columns = [
        column
        for column in POLLING_COLUMNS
        if column in out.columns
    ]

    out = out.drop(
        columns=stale_columns,
        errors="ignore",
    )

    merge_columns = [
        "district_id",
        *POLLING_COLUMNS,
    ]

    rows_before = len(out)

    out = out.merge(
        snapshot[
            merge_columns
        ],
        on="district_id",
        how="left",
        validate="one_to_one",
    )

    if len(out) != rows_before:
        raise HistoricalPollingExperimentError(
            "Historical polling merge changed the race-row count."
        )

    missing = out[
        "snapshot_date"
    ].isna()

    if missing.any():
        district_ids = (
            out.loc[
                missing,
                "district_id",
            ]
            .astype(str)
            .head(20)
            .tolist()
        )

        raise HistoricalPollingExperimentError(
            "Historical polling snapshot did not map to: "
            + ", ".join(district_ids)
        )

    return out


def require_unique_keys(
    frame: pd.DataFrame,
    keys: list[str],
    *,
    label: str,
) -> None:
    missing = [
        key
        for key in keys
        if key not in frame.columns
    ]

    if missing:
        raise HistoricalPollingExperimentError(
            f"{label} is missing keys: "
            + ", ".join(missing)
        )

    duplicate_mask = frame.duplicated(
        keys,
        keep=False,
    )

    if duplicate_mask.any():
        examples = (
            frame.loc[
                duplicate_mask,
                keys,
            ]
            .head(20)
            .to_string(index=False)
        )

        raise HistoricalPollingExperimentError(
            f"{label} contains duplicate keys:\n"
            + examples
        )


def add_experiment_metadata(
    frame: pd.DataFrame,
    *,
    cycle: int,
    days_out: int,
    snapshot_date: str,
    arm: str,
) -> pd.DataFrame:
    out = frame.copy()

    out["experiment_arm"] = arm
    out["historical_days_out"] = int(
        days_out
    )
    out[
        "historical_snapshot_date"
    ] = snapshot_date

    if "cycle" not in out.columns:
        out["cycle"] = int(cycle)

    return out


def extract_snapshot_date(
    snapshot: pd.DataFrame,
) -> str:
    values = (
        snapshot["snapshot_date"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    if len(values) != 1:
        raise HistoricalPollingExperimentError(
            "Historical polling snapshot expected one "
            f"snapshot_date; found {values}."
        )

    return values[0]


def build_district_comparison(
    fundamentals: pd.DataFrame,
    polling: pd.DataFrame,
    *,
    cycle: int,
    days_out: int,
    snapshot_date: str,
) -> pd.DataFrame:
    keys = [
        "cycle",
        "race_id",
    ]

    require_unique_keys(
        fundamentals,
        keys,
        label="Fundamentals results",
    )

    require_unique_keys(
        polling,
        keys,
        label="Polling results",
    )

    fundamentals_columns = [
        *keys,
        *[
            column
            for column in DISTRICT_METRICS
            if column
            in fundamentals.columns
        ],
    ]

    polling_columns = [
        *keys,
        *[
            column
            for column in DISTRICT_METRICS
            if column
            in polling.columns
        ],
    ]

    merged = fundamentals[
        fundamentals_columns
    ].merge(
        polling[
            polling_columns
        ],
        on=keys,
        how="outer",
        suffixes=(
            "_fundamentals",
            "_polling",
        ),
        indicator=True,
        validate="one_to_one",
    )

    if not merged[
        "_merge"
    ].eq("both").all():
        raise HistoricalPollingExperimentError(
            "Fundamentals and polling district keys differ."
        )

    merged = merged.drop(
        columns="_merge"
    )

    merged[
        "historical_days_out"
    ] = int(days_out)

    merged[
        "historical_snapshot_date"
    ] = snapshot_date

    for metric in DISTRICT_METRICS:
        fundamentals_column = (
            f"{metric}_fundamentals"
        )

        polling_column = (
            f"{metric}_polling"
        )

        if (
            fundamentals_column
            not in merged.columns
            or polling_column
            not in merged.columns
        ):
            continue

        left = pd.to_numeric(
            merged[
                fundamentals_column
            ],
            errors="coerce",
        )

        right = pd.to_numeric(
            merged[
                polling_column
            ],
            errors="coerce",
        )

        merged[
            f"{metric}_polling_minus_fundamentals"
        ] = (
            right - left
        )

    return merged


def build_summary_comparison(
    fundamentals: pd.DataFrame,
    polling: pd.DataFrame,
    *,
    cycle: int,
    days_out: int,
    snapshot_date: str,
) -> pd.DataFrame:
    if len(fundamentals) != 1:
        raise HistoricalPollingExperimentError(
            "Fundamentals cycle summary must contain one row."
        )

    if len(polling) != 1:
        raise HistoricalPollingExperimentError(
            "Polling cycle summary must contain one row."
        )

    row: dict[str, Any] = {
        "cycle": int(cycle),
        "historical_days_out": int(
            days_out
        ),
        "historical_snapshot_date":
            snapshot_date,
    }

    for metric in SUMMARY_METRICS:
        fundamentals_value = (
            fundamentals.iloc[0].get(
                metric,
                np.nan,
            )
        )

        polling_value = (
            polling.iloc[0].get(
                metric,
                np.nan,
            )
        )

        fundamentals_numeric = (
            pd.to_numeric(
                pd.Series(
                    [fundamentals_value]
                ),
                errors="coerce",
            ).iloc[0]
        )

        polling_numeric = (
            pd.to_numeric(
                pd.Series(
                    [polling_value]
                ),
                errors="coerce",
            ).iloc[0]
        )

        row[
            f"{metric}_fundamentals"
        ] = fundamentals_numeric

        row[
            f"{metric}_polling"
        ] = polling_numeric

        if (
            pd.notna(
                fundamentals_numeric
            )
            and pd.notna(
                polling_numeric
            )
        ):
            row[
                f"{metric}_polling_minus_fundamentals"
            ] = (
                float(polling_numeric)
                - float(
                    fundamentals_numeric
                )
            )
        else:
            row[
                f"{metric}_polling_minus_fundamentals"
            ] = np.nan

    return pd.DataFrame(
        [row]
    )


def summarize_overall(
    comparisons: pd.DataFrame,
) -> pd.DataFrame:
    metric_bases = [
        "brier_score",
        "log_loss",
        "expected_seat_error",
        "dem_control_probability",
        "average_polling_weight",
        "districts_with_polling",
    ]

    rows = []

    for days_out in sorted(
        comparisons[
            "historical_days_out"
        ].unique(),
        reverse=True,
    ):
        subset = comparisons.loc[
            comparisons[
                "historical_days_out"
            ].eq(days_out)
        ]

        row: dict[str, Any] = {
            "historical_days_out":
                int(days_out),
            "cycles":
                int(
                    subset[
                        "cycle"
                    ].nunique()
                ),
        }

        for metric in metric_bases:
            for suffix in [
                "fundamentals",
                "polling",
                "polling_minus_fundamentals",
            ]:
                column = (
                    f"{metric}_{suffix}"
                )

                if column not in subset.columns:
                    continue

                values = pd.to_numeric(
                    subset[column],
                    errors="coerce",
                )

                row[
                    f"mean_{column}"
                ] = float(
                    values.mean()
                )

        fundamentals_seat_error = (
            pd.to_numeric(
                subset[
                    "expected_seat_error_fundamentals"
                ],
                errors="coerce",
            )
        )

        polling_seat_error = (
            pd.to_numeric(
                subset[
                    "expected_seat_error_polling"
                ],
                errors="coerce",
            )
        )

        row[
            "mean_abs_expected_seat_error_fundamentals"
        ] = float(
            fundamentals_seat_error
            .abs()
            .mean()
        )

        row[
            "mean_abs_expected_seat_error_polling"
        ] = float(
            polling_seat_error
            .abs()
            .mean()
        )

        row[
            "mean_abs_expected_seat_error_improvement"
        ] = (
            row[
                "mean_abs_expected_seat_error_fundamentals"
            ]
            - row[
                "mean_abs_expected_seat_error_polling"
            ]
        )

        rows.append(row)

    return pd.DataFrame(rows)


def validate_experiment(
    predictions: pd.DataFrame,
    summaries: pd.DataFrame,
    comparisons: pd.DataFrame,
    district_comparisons: pd.DataFrame,
) -> list[str]:
    checks = []

    expected_snapshot_count = (
        len(SUPPORTED_CYCLES)
        * len(SUPPORTED_DAYS_OUT)
    )

    expected_prediction_rows = (
        expected_snapshot_count
        * 2
        * 435
    )

    if len(predictions) != (
        expected_prediction_rows
    ):
        raise HistoricalPollingExperimentError(
            "Prediction row count should be "
            f"{expected_prediction_rows:,}; "
            f"found {len(predictions):,}."
        )

    checks.append(
        "PASS: prediction rows = "
        f"{expected_prediction_rows:,}"
    )

    require_unique_keys(
        predictions,
        [
            "experiment_arm",
            "cycle",
            "historical_days_out",
            "race_id",
        ],
        label="Experiment predictions",
    )

    checks.append(
        "PASS: arm/cycle/snapshot/race keys "
        "are unique"
    )

    expected_summary_rows = (
        expected_snapshot_count * 2
    )

    if len(summaries) != (
        expected_summary_rows
    ):
        raise HistoricalPollingExperimentError(
            "Summary row count should be "
            f"{expected_summary_rows}; "
            f"found {len(summaries)}."
        )

    checks.append(
        "PASS: arm-level summary rows = "
        f"{expected_summary_rows}"
    )

    if len(comparisons) != (
        expected_snapshot_count
    ):
        raise HistoricalPollingExperimentError(
            "Matched comparison row count should be "
            f"{expected_snapshot_count}; "
            f"found {len(comparisons)}."
        )

    checks.append(
        "PASS: matched snapshot comparisons = "
        f"{expected_snapshot_count}"
    )

    expected_district_comparison_rows = (
        expected_snapshot_count * 435
    )

    if len(
        district_comparisons
    ) != (
        expected_district_comparison_rows
    ):
        raise HistoricalPollingExperimentError(
            "District comparison row count should be "
            f"{expected_district_comparison_rows:,}; "
            f"found {len(district_comparisons):,}."
        )

    checks.append(
        "PASS: matched district comparisons = "
        f"{expected_district_comparison_rows:,}"
    )

    probabilities = pd.to_numeric(
        predictions[
            "dem_win_probability"
        ],
        errors="coerce",
    )

    if (
        probabilities.isna().any()
        or not probabilities.between(
            0.0,
            1.0,
        ).all()
    ):
        raise HistoricalPollingExperimentError(
            "Experiment contains invalid probabilities."
        )

    checks.append(
        "PASS: all probabilities are finite "
        "and within [0, 1]"
    )

    fundamentals = predictions.loc[
        predictions[
            "experiment_arm"
        ].eq(FUNDAMENTALS_SPEC)
    ]

    fundamentals_weights = (
        pd.to_numeric(
            fundamentals[
                "bayesian_polling_weight"
            ],
            errors="coerce",
        ).fillna(0.0)
    )

    if not fundamentals_weights.eq(
        0.0
    ).all():
        raise HistoricalPollingExperimentError(
            "Fundamentals arm contains nonzero "
            "polling weights."
        )

    checks.append(
        "PASS: fundamentals arm has zero "
        "polling weight"
    )

    polling = predictions.loc[
        predictions[
            "experiment_arm"
        ].eq(POLLING_SPEC)
    ]

    polling_count = pd.to_numeric(
        polling[
            "poll_count"
        ],
        errors="coerce",
    ).fillna(0.0)

    polling_weights = pd.to_numeric(
        polling[
            "bayesian_polling_weight"
        ],
        errors="coerce",
    ).fillna(0.0)

    if not polling.loc[
        polling_count.le(0.0)
    ].empty:
        unpolled_weight = (
            polling_weights.loc[
                polling_count.le(0.0)
            ]
        )

        if not unpolled_weight.eq(
            0.0
        ).all():
            raise HistoricalPollingExperimentError(
                "Unpolled districts received nonzero "
                "Bayesian polling weight."
            )

    checks.append(
        "PASS: unpolled districts receive zero "
        "polling weight"
    )

    active_snapshot_rows = (
        comparisons[
            "districts_with_polling_polling"
        ]
    )

    if pd.to_numeric(
        active_snapshot_rows,
        errors="coerce",
    ).fillna(0).le(0).any():
        raise HistoricalPollingExperimentError(
            "At least one snapshot activated no polling."
        )

    checks.append(
        "PASS: every snapshot activates polling "
        "in at least one district"
    )

    return checks


def main() -> None:
    args = parse_args()

    if args.sims <= 0:
        raise ValueError(
            "--sims must be positive."
        )

    output_dir = args.output_dir

    if (
        args.clean_output
        and output_dir.exists()
    ):
        for path in output_dir.iterdir():
            if path.is_file():
                path.unlink()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    master = pd.read_csv(
        args.master_path,
        low_memory=False,
    )

    prediction_frames = []
    summary_frames = []
    calibration_frames = []
    comparison_frames = []
    district_comparison_frames = []

    for cycle in SUPPORTED_CYCLES:
        print()
        print("=" * 88)
        print(
            f"HISTORICAL POLLING EXPERIMENT: {cycle}"
        )
        print("=" * 88)

        prepared_df, national_environment = (
            prepare_cycle(
                master=master,
                cycle=int(cycle),
                candidate_quality_weight=0.0,
                candidate_war_path=(
                    args.candidate_war_path
                ),
            )
        )

        legacy_margin, legacy_source = (
            build_model_margin(
                prepared_df.copy()
            )
        )

        if legacy_margin is None:
            raise HistoricalPollingExperimentError(
                f"Cycle {cycle} legacy margin unavailable: "
                f"{legacy_source}"
            )

        (
            production_df,
            model_margin,
            production_source,
        ) = build_production_fundamentals(
            df=prepared_df,
            cycle=int(cycle),
            national_environment=(
                national_environment
            ),
        )

        for days_out in SUPPORTED_DAYS_OUT:
            snapshot = (
                load_house_historical_polling_snapshot(
                    cycle=int(cycle),
                    days_out=int(days_out),
                )
            )

            snapshot_date = (
                extract_snapshot_date(
                    snapshot
                )
            )

            print()
            print(
                f"{cycle} — {days_out:>3} days out "
                f"({snapshot_date})"
            )
            print("-" * 88)

            fundamentals_df = (
                neutralize_polling(
                    production_df
                )
            )

            polling_df = (
                attach_polling_snapshot(
                    production_df,
                    snapshot,
                )
            )

            # Identical seed for the matched arms.
            matched_seed = (
                int(args.seed)
                + int(cycle) * 1000
                + int(days_out)
            )

            (
                fundamentals_results,
                fundamentals_summary,
                fundamentals_calibration,
            ) = run_production_shared_spec(
                df=fundamentals_df,
                model_margin=model_margin,
                fallback_margin=legacy_margin,
                n_sims=int(args.sims),
                seed=matched_seed,
                fixed_error_sd=float(
                    args.fixed_error_sd
                ),
                days_out=int(days_out),
                enable_polling=True,
                replay_spec=FUNDAMENTALS_SPEC,
            )

            (
                polling_results,
                polling_summary,
                polling_calibration,
            ) = run_production_shared_spec(
                df=polling_df,
                model_margin=model_margin,
                fallback_margin=legacy_margin,
                n_sims=int(args.sims),
                seed=matched_seed,
                fixed_error_sd=float(
                    args.fixed_error_sd
                ),
                days_out=int(days_out),
                enable_polling=True,
                replay_spec=POLLING_SPEC,
            )

            fundamentals_results = (
                add_experiment_metadata(
                    fundamentals_results,
                    cycle=int(cycle),
                    days_out=int(days_out),
                    snapshot_date=(
                        snapshot_date
                    ),
                    arm=FUNDAMENTALS_SPEC,
                )
            )

            polling_results = (
                add_experiment_metadata(
                    polling_results,
                    cycle=int(cycle),
                    days_out=int(days_out),
                    snapshot_date=(
                        snapshot_date
                    ),
                    arm=POLLING_SPEC,
                )
            )

            fundamentals_summary = (
                add_experiment_metadata(
                    fundamentals_summary,
                    cycle=int(cycle),
                    days_out=int(days_out),
                    snapshot_date=(
                        snapshot_date
                    ),
                    arm=FUNDAMENTALS_SPEC,
                )
            )

            polling_summary = (
                add_experiment_metadata(
                    polling_summary,
                    cycle=int(cycle),
                    days_out=int(days_out),
                    snapshot_date=(
                        snapshot_date
                    ),
                    arm=POLLING_SPEC,
                )
            )

            fundamentals_calibration = (
                add_experiment_metadata(
                    fundamentals_calibration,
                    cycle=int(cycle),
                    days_out=int(days_out),
                    snapshot_date=(
                        snapshot_date
                    ),
                    arm=FUNDAMENTALS_SPEC,
                )
            )

            polling_calibration = (
                add_experiment_metadata(
                    polling_calibration,
                    cycle=int(cycle),
                    days_out=int(days_out),
                    snapshot_date=(
                        snapshot_date
                    ),
                    arm=POLLING_SPEC,
                )
            )

            summary_comparison = (
                build_summary_comparison(
                    fundamentals_summary,
                    polling_summary,
                    cycle=int(cycle),
                    days_out=int(days_out),
                    snapshot_date=(
                        snapshot_date
                    ),
                )
            )

            district_comparison = (
                build_district_comparison(
                    fundamentals_results,
                    polling_results,
                    cycle=int(cycle),
                    days_out=int(days_out),
                    snapshot_date=(
                        snapshot_date
                    ),
                )
            )

            prediction_frames.extend(
                [
                    fundamentals_results,
                    polling_results,
                ]
            )

            summary_frames.extend(
                [
                    fundamentals_summary,
                    polling_summary,
                ]
            )

            calibration_frames.extend(
                [
                    fundamentals_calibration,
                    polling_calibration,
                ]
            )

            comparison_frames.append(
                summary_comparison
            )

            district_comparison_frames.append(
                district_comparison
            )

            brier_delta = float(
                summary_comparison.iloc[0][
                    "brier_score_polling_minus_fundamentals"
                ]
            )

            log_loss_delta = float(
                summary_comparison.iloc[0][
                    "log_loss_polling_minus_fundamentals"
                ]
            )

            fundamentals_seat_error = float(
                summary_comparison.iloc[0][
                    "expected_seat_error_fundamentals"
                ]
            )

            polling_seat_error = float(
                summary_comparison.iloc[0][
                    "expected_seat_error_polling"
                ]
            )

            average_weight = float(
                summary_comparison.iloc[0][
                    "average_polling_weight_polling"
                ]
            )

            districts_with_polling = int(
                summary_comparison.iloc[0][
                    "districts_with_polling_polling"
                ]
            )

            print(
                "Districts with active polling: "
                f"{districts_with_polling}"
            )

            print(
                "Average polling weight:        "
                f"{average_weight:.6f}"
            )

            print(
                "Brier delta, poll - fund.:     "
                f"{brier_delta:+.6f}"
            )

            print(
                "Log-loss delta, poll - fund.:  "
                f"{log_loss_delta:+.6f}"
            )

            print(
                "Expected-seat error:            "
                f"{fundamentals_seat_error:+.3f} "
                f"-> {polling_seat_error:+.3f}"
            )

        print()
        print(
            f"Completed cycle {cycle}; "
            f"production source: {production_source}"
        )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    summaries = pd.concat(
        summary_frames,
        ignore_index=True,
    )

    calibrations = pd.concat(
        calibration_frames,
        ignore_index=True,
    )

    comparisons = pd.concat(
        comparison_frames,
        ignore_index=True,
    )

    district_comparisons = pd.concat(
        district_comparison_frames,
        ignore_index=True,
    )

    overall = summarize_overall(
        comparisons
    )

    validation_checks = (
        validate_experiment(
            predictions,
            summaries,
            comparisons,
            district_comparisons,
        )
    )

    predictions_path = (
        output_dir
        / "house_historical_polling_predictions.csv"
    )

    summaries_path = (
        output_dir
        / "house_historical_polling_by_snapshot.csv"
    )

    comparisons_path = (
        output_dir
        / "house_historical_polling_matched_comparison.csv"
    )

    overall_path = (
        output_dir
        / "house_historical_polling_overall_by_days_out.csv"
    )

    district_comparisons_path = (
        output_dir
        / "house_historical_polling_district_comparison.csv"
    )

    calibration_path = (
        output_dir
        / "house_historical_polling_calibration.csv"
    )

    config_path = (
        output_dir
        / "house_historical_polling_config.json"
    )

    validation_path = (
        output_dir
        / "house_historical_polling_validation.txt"
    )

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    summaries.to_csv(
        summaries_path,
        index=False,
    )

    comparisons.to_csv(
        comparisons_path,
        index=False,
    )

    overall.to_csv(
        overall_path,
        index=False,
    )

    district_comparisons.to_csv(
        district_comparisons_path,
        index=False,
    )

    calibrations.to_csv(
        calibration_path,
        index=False,
    )

    config = {
        "cycles": [
            int(cycle)
            for cycle in SUPPORTED_CYCLES
        ],
        "days_out": [
            int(days_out)
            for days_out in SUPPORTED_DAYS_OUT
        ],
        "n_sims": int(args.sims),
        "base_seed": int(args.seed),
        "fixed_error_sd": float(
            args.fixed_error_sd
        ),
        "fundamentals_spec":
            FUNDAMENTALS_SPEC,
        "polling_spec":
            POLLING_SPEC,
        "master_path":
            str(args.master_path),
        "candidate_war_path":
            str(
                args.candidate_war_path
            ),
        "comparison_design": (
            "Matched historical days-out and random seed. "
            "Both arms use live prepare_house_table(); "
            "the fundamentals arm neutralizes polling fields."
        ),
    }

    config_path.write_text(
        json.dumps(
            config,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    validation_text = "\n".join(
        [
            "House Historical Bayesian Polling Experiment",
            "=" * 52,
            "",
            *validation_checks,
            "",
            "VALIDATION PASSED",
            "",
        ]
    )

    validation_path.write_text(
        validation_text
    )

    display_columns = [
        "historical_days_out",
        "cycles",
        "mean_brier_score_fundamentals",
        "mean_brier_score_polling",
        "mean_brier_score_polling_minus_fundamentals",
        "mean_log_loss_fundamentals",
        "mean_log_loss_polling",
        "mean_log_loss_polling_minus_fundamentals",
        "mean_abs_expected_seat_error_fundamentals",
        "mean_abs_expected_seat_error_polling",
        "mean_abs_expected_seat_error_improvement",
        "mean_average_polling_weight_polling",
        "mean_districts_with_polling_polling",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in overall.columns
    ]

    print()
    print("=" * 110)
    print(
        "HISTORICAL POLLING PERFORMANCE "
        "BY DAYS OUT"
    )
    print("=" * 110)

    print(
        overall[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(validation_text)

    print("Wrote:")
    for path in [
        predictions_path,
        summaries_path,
        comparisons_path,
        overall_path,
        district_comparisons_path,
        calibration_path,
        config_path,
        validation_path,
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
