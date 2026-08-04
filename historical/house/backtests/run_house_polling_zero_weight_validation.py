from __future__ import annotations

from pathlib import Path
import sys

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
    load_house_historical_polling_snapshot,
)
from run_house_dynamic_uncertainty import (
    read_settings,
)


SUPPORTED_CYCLES = (
    2018,
    2020,
    2022,
)

CONTROL_DAYS_OUT = 1

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "polling_zero_weight_validation"
)

POLLING_MERGE_COLUMNS = [
    "district_id",
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


class PollingZeroWeightValidationError(
    RuntimeError
):
    pass


def attach_polling_snapshot(
    frame: pd.DataFrame,
    snapshot: pd.DataFrame,
) -> pd.DataFrame:
    out = frame.copy()

    if "district_id" not in out.columns:
        raise PollingZeroWeightValidationError(
            "Replay dataframe lacks district_id."
        )

    stale_polling_columns = [
        column
        for column in POLLING_MERGE_COLUMNS
        if (
            column != "district_id"
            and column in out.columns
        )
    ]

    out = out.drop(
        columns=stale_polling_columns,
        errors="ignore",
    )

    rows_before = len(out)

    out = out.merge(
        snapshot[
            POLLING_MERGE_COLUMNS
        ],
        on="district_id",
        how="left",
        validate="one_to_one",
    )

    if len(out) != rows_before:
        raise PollingZeroWeightValidationError(
            "Polling snapshot merge changed "
            "the historical race-row count."
        )

    missing_snapshot_rows = out[
        "snapshot_date"
    ].isna()

    if missing_snapshot_rows.any():
        districts = (
            out.loc[
                missing_snapshot_rows,
                "district_id",
            ]
            .astype(str)
            .head(20)
            .tolist()
        )

        raise PollingZeroWeightValidationError(
            "Historical polling snapshot failed "
            "to match districts: "
            + ", ".join(districts)
        )

    return out


def compare_numeric_columns(
    baseline: pd.DataFrame,
    control: pd.DataFrame,
    *,
    key_columns: list[str],
    numeric_columns: list[str],
    tolerance: float = 0.0,
) -> dict[str, float]:
    baseline_sorted = (
        baseline.sort_values(
            key_columns,
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    control_sorted = (
        control.sort_values(
            key_columns,
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    if len(baseline_sorted) != len(
        control_sorted
    ):
        raise PollingZeroWeightValidationError(
            "Baseline/control row counts differ."
        )

    for column in key_columns:
        if not baseline_sorted[
            column
        ].astype(str).equals(
            control_sorted[
                column
            ].astype(str)
        ):
            raise PollingZeroWeightValidationError(
                f"Baseline/control keys differ: {column}"
            )

    differences = {}

    for column in numeric_columns:
        if (
            column not in baseline_sorted.columns
            or column not in control_sorted.columns
        ):
            continue

        left = pd.to_numeric(
            baseline_sorted[column],
            errors="coerce",
        )

        right = pd.to_numeric(
            control_sorted[column],
            errors="coerce",
        )

        difference = (
            left - right
        ).abs()

        maximum = (
            float(
                difference
                .dropna()
                .max()
            )
            if difference.notna().any()
            else 0.0
        )

        differences[column] = maximum

        if not np.allclose(
            left,
            right,
            atol=tolerance,
            rtol=0.0,
            equal_nan=True,
        ):
            raise PollingZeroWeightValidationError(
                f"Merge-only polling control changed {column}. "
                f"Maximum difference: {maximum:.18g}"
            )

    return differences


def main() -> None:
    output_dir = DEFAULT_OUTPUT_DIR

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    master = pd.read_csv(
        DEFAULT_MASTER_PATH,
        low_memory=False,
    )

    settings = read_settings()

    audit_rows = []
    baseline_prediction_frames = []
    control_prediction_frames = []
    baseline_summary_frames = []
    control_summary_frames = []

    for cycle in SUPPORTED_CYCLES:
        print()
        print("=" * 80)
        print(
            "HOUSE POLLING ZERO-WEIGHT CONTROL: "
            f"{cycle}"
        )
        print("=" * 80)

        prepared_df, national_environment = (
            prepare_cycle(
                master=master,
                cycle=cycle,
                candidate_quality_weight=0.0,
                candidate_war_path=DEFAULT_WAR_PATH,
            )
        )

        legacy_margin, legacy_source = (
            build_model_margin(
                prepared_df.copy()
            )
        )

        if legacy_margin is None:
            raise PollingZeroWeightValidationError(
                f"Cycle {cycle} legacy fallback "
                f"margin unavailable: {legacy_source}"
            )

        production_df, model_margin, source = (
            build_production_fundamentals(
                df=prepared_df,
                cycle=cycle,
                national_environment=(
                    national_environment
                ),
            )
        )

        snapshot = (
            load_house_historical_polling_snapshot(
                cycle=cycle,
                days_out=CONTROL_DAYS_OUT,
            )
        )

        merged_df = attach_polling_snapshot(
            production_df,
            snapshot,
        )

        baseline_results, baseline_summary, _ = (
            run_production_shared_spec(
                df=production_df,
                model_margin=model_margin,
                fallback_margin=legacy_margin,
                n_sims=20000,
                seed=20260719 + cycle,
                fixed_error_sd=6.5,
            )
        )

        control_results, control_summary, _ = (
            run_production_shared_spec(
                df=merged_df,
                model_margin=model_margin,
                fallback_margin=legacy_margin,
                n_sims=20000,
                seed=20260719 + cycle,
                fixed_error_sd=6.5,
            )
        )

        result_differences = (
            compare_numeric_columns(
                baseline_results,
                control_results,
                key_columns=[
                    "cycle",
                    "race_id",
                ],
                numeric_columns=[
                    "model_margin",
                    "model_margin_dem",
                    "dem_win_probability",
                    "raw_simulated_dem_win_probability",
                    "simulated_dem_win_probability",
                    "avg_simulated_margin_dem",
                    "margin_p25_dem",
                    "margin_p50_dem",
                    "margin_p75_dem",
                    "brier_score",
                    "margin_error",
                ],
                tolerance=0.0,
            )
        )

        summary_differences = (
            compare_numeric_columns(
                baseline_summary,
                control_summary,
                key_columns=[
                    "cycle",
                ],
                numeric_columns=[
                    "brier_score",
                    "log_loss",
                    "expected_dem_seats",
                    "actual_dem_seats",
                    "expected_seat_error",
                    "expected_dem_seats_from_simulation",
                    "dem_control_probability",
                    "dem_seats_p25",
                    "dem_seats_p50",
                    "dem_seats_p75",
                ],
                tolerance=0.0,
            )
        )

        if "replay_spec" in baseline_results.columns:
            baseline_results = (
                baseline_results.copy()
            )
            baseline_results[
                "replay_spec"
            ] = (
                "polling_control_baseline"
            )

        if "replay_spec" in control_results.columns:
            control_results = (
                control_results.copy()
            )
            control_results[
                "replay_spec"
            ] = (
                "polling_merged_zero_weight"
            )

        if "replay_spec" in baseline_summary.columns:
            baseline_summary = (
                baseline_summary.copy()
            )
            baseline_summary[
                "replay_spec"
            ] = (
                "polling_control_baseline"
            )

        if "replay_spec" in control_summary.columns:
            control_summary = (
                control_summary.copy()
            )
            control_summary[
                "replay_spec"
            ] = (
                "polling_merged_zero_weight"
            )

        baseline_prediction_frames.append(
            baseline_results
        )
        control_prediction_frames.append(
            control_results
        )
        baseline_summary_frames.append(
            baseline_summary
        )
        control_summary_frames.append(
            control_summary
        )

        districts_with_polling = int(
            pd.to_numeric(
                snapshot["poll_count"],
                errors="coerce",
            )
            .fillna(0)
            .gt(0)
            .sum()
        )

        max_result_difference = max(
            result_differences.values(),
            default=0.0,
        )

        max_summary_difference = max(
            summary_differences.values(),
            default=0.0,
        )

        audit_rows.append(
            {
                "cycle": cycle,
                "days_out": CONTROL_DAYS_OUT,
                "forecast_source": source,
                "districts_with_polling_in_snapshot":
                    districts_with_polling,
                "maximum_result_difference":
                    max_result_difference,
                "maximum_summary_difference":
                    max_summary_difference,
                "validation_status": "PASSED",
            }
        )

        print(
            "Districts with polling attached: "
            f"{districts_with_polling}"
        )
        print(
            "Maximum district-result difference: "
            f"{max_result_difference:.18g}"
        )
        print(
            "Maximum summary difference: "
            f"{max_summary_difference:.18g}"
        )
        print("CYCLE VALIDATION PASSED")

    predictions = pd.concat(
        [
            *baseline_prediction_frames,
            *control_prediction_frames,
        ],
        ignore_index=True,
    )

    summaries = pd.concat(
        [
            *baseline_summary_frames,
            *control_summary_frames,
        ],
        ignore_index=True,
    )

    audit = pd.DataFrame(
        audit_rows
    )

    predictions_path = (
        output_dir
        / "house_polling_zero_weight_predictions.csv"
    )

    summaries_path = (
        output_dir
        / "house_polling_zero_weight_by_cycle.csv"
    )

    audit_path = (
        output_dir
        / "house_polling_zero_weight_audit.csv"
    )

    validation_path = (
        output_dir
        / "house_polling_zero_weight_validation.txt"
    )

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    summaries.to_csv(
        summaries_path,
        index=False,
    )

    audit.to_csv(
        audit_path,
        index=False,
    )

    validation_lines = [
        "House Historical Polling Zero-Weight Validation",
        "=" * 52,
        "",
        (
            "Historical polling snapshot days out: "
            f"{CONTROL_DAYS_OUT}"
        ),
        (
            "Cycles: "
            + ", ".join(
                str(cycle)
                for cycle in SUPPORTED_CYCLES
            )
        ),
        "",
        audit.to_string(index=False),
        "",
        "VALIDATION PASSED",
        (
            "Attaching historical polling fields without "
            "activating the Bayesian blend produced exactly "
            "identical shared-v2 replay results."
        ),
        "",
    ]

    validation_text = "\n".join(
        validation_lines
    )

    validation_path.write_text(
        validation_text
    )

    print()
    print(validation_text)

    print("Wrote:")
    for path in [
        predictions_path,
        summaries_path,
        audit_path,
        validation_path,
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
