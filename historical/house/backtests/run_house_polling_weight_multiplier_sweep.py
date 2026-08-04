from __future__ import annotations

import argparse
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
from historical.house.backtests.run_house_historical_polling_experiment import (
    attach_polling_snapshot,
    extract_snapshot_date,
)
from historical.house.polling.load_house_historical_polling_snapshot import (
    SUPPORTED_CYCLES,
    SUPPORTED_DAYS_OUT,
    load_house_historical_polling_snapshot,
)


MULTIPLIERS = (
    0.00,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "polling_weight_multiplier_sweep"
)


class SweepValidationError(
    RuntimeError
):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep House historical polling-weight multipliers "
            "while retaining the production time-based cap."
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


def add_metadata(
    frame: pd.DataFrame,
    *,
    multiplier: float,
    days_out: int,
    snapshot_date: str,
) -> pd.DataFrame:
    out = frame.copy()

    out[
        "polling_weight_multiplier"
    ] = float(multiplier)

    out[
        "historical_days_out"
    ] = int(days_out)

    out[
        "historical_snapshot_date"
    ] = snapshot_date

    return out


def build_overall_summary(
    snapshot_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for multiplier, group in (
        snapshot_summary.groupby(
            "polling_weight_multiplier",
            sort=True,
        )
    ):
        brier = pd.to_numeric(
            group["brier_score"],
            errors="coerce",
        )

        log_loss = pd.to_numeric(
            group["log_loss"],
            errors="coerce",
        )

        seat_error = pd.to_numeric(
            group["expected_seat_error"],
            errors="coerce",
        )

        mean_abs_margin_error = pd.to_numeric(
            group.get(
                "mean_abs_margin_error",
                pd.Series(
                    np.nan,
                    index=group.index,
                ),
            ),
            errors="coerce",
        )

        average_polling_weight = pd.to_numeric(
            group["average_polling_weight"],
            errors="coerce",
        )

        districts_with_polling = pd.to_numeric(
            group["districts_with_polling"],
            errors="coerce",
        )

        rows.append(
            {
                "polling_weight_multiplier":
                    float(multiplier),
                "snapshots":
                    int(len(group)),
                "mean_brier_score":
                    float(brier.mean()),
                "mean_log_loss":
                    float(log_loss.mean()),
                "mean_abs_expected_seat_error":
                    float(
                        seat_error.abs().mean()
                    ),
                "mean_signed_expected_seat_error":
                    float(
                        seat_error.mean()
                    ),
                "mean_abs_margin_error":
                    float(
                        mean_abs_margin_error.mean()
                    ),
                "mean_average_polling_weight":
                    float(
                        average_polling_weight.mean()
                    ),
                "mean_districts_with_polling":
                    float(
                        districts_with_polling.mean()
                    ),
                "cycles_improved_brier_vs_zero":
                    0,
                "snapshots_improved_brier_vs_zero":
                    0,
                "snapshots_improved_log_loss_vs_zero":
                    0,
            }
        )

    overall = pd.DataFrame(rows)

    zero = snapshot_summary.loc[
        snapshot_summary[
            "polling_weight_multiplier"
        ].eq(0.0)
    ][
        [
            "cycle",
            "historical_days_out",
            "brier_score",
            "log_loss",
        ]
    ].rename(
        columns={
            "brier_score":
                "zero_brier_score",
            "log_loss":
                "zero_log_loss",
        }
    )

    joined = snapshot_summary.merge(
        zero,
        on=[
            "cycle",
            "historical_days_out",
        ],
        how="left",
        validate="many_to_one",
    )

    joined["brier_improved_vs_zero"] = (
        pd.to_numeric(
            joined["brier_score"],
            errors="coerce",
        )
        < pd.to_numeric(
            joined["zero_brier_score"],
            errors="coerce",
        )
    )

    joined["log_loss_improved_vs_zero"] = (
        pd.to_numeric(
            joined["log_loss"],
            errors="coerce",
        )
        < pd.to_numeric(
            joined["zero_log_loss"],
            errors="coerce",
        )
    )

    for multiplier in MULTIPLIERS:
        mask = joined[
            "polling_weight_multiplier"
        ].eq(multiplier)

        overall.loc[
            overall[
                "polling_weight_multiplier"
            ].eq(multiplier),
            "snapshots_improved_brier_vs_zero",
        ] = int(
            joined.loc[
                mask,
                "brier_improved_vs_zero",
            ].sum()
        )

        overall.loc[
            overall[
                "polling_weight_multiplier"
            ].eq(multiplier),
            "snapshots_improved_log_loss_vs_zero",
        ] = int(
            joined.loc[
                mask,
                "log_loss_improved_vs_zero",
            ].sum()
        )

        cycle_means = (
            joined.loc[mask]
            .groupby("cycle")
            .agg(
                multiplier_brier=(
                    "brier_score",
                    "mean",
                ),
                zero_brier=(
                    "zero_brier_score",
                    "mean",
                ),
            )
        )

        overall.loc[
            overall[
                "polling_weight_multiplier"
            ].eq(multiplier),
            "cycles_improved_brier_vs_zero",
        ] = int(
            cycle_means[
                "multiplier_brier"
            ].lt(
                cycle_means[
                    "zero_brier"
                ]
            ).sum()
        )

    for metric in [
        "mean_brier_score",
        "mean_log_loss",
        "mean_abs_expected_seat_error",
        "mean_abs_margin_error",
    ]:
        if (
            metric not in overall.columns
            or overall[metric].isna().all()
        ):
            continue

        overall[
            f"{metric}_rank"
        ] = overall[
            metric
        ].rank(
            method="min",
            ascending=True,
        )

    rank_columns = [
        column
        for column in overall.columns
        if column.endswith("_rank")
    ]

    overall[
        "mean_primary_metric_rank"
    ] = overall[
        rank_columns
    ].mean(axis=1)

    overall[
        "production_multiplier"
    ] = overall[
        "polling_weight_multiplier"
    ].eq(1.0)

    return overall.sort_values(
        [
            "mean_primary_metric_rank",
            "polling_weight_multiplier",
        ],
        ascending=[
            True,
            True,
        ],
    ).reset_index(drop=True)


def build_days_out_summary(
    snapshot_summary: pd.DataFrame,
) -> pd.DataFrame:
    return (
        snapshot_summary.groupby(
            [
                "polling_weight_multiplier",
                "historical_days_out",
            ],
            as_index=False,
        )
        .agg(
            cycles=(
                "cycle",
                "nunique",
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
            mean_average_polling_weight=(
                "average_polling_weight",
                "mean",
            ),
            mean_districts_with_polling=(
                "districts_with_polling",
                "mean",
            ),
        )
        .sort_values(
            [
                "historical_days_out",
                "polling_weight_multiplier",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )


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

    for cycle in SUPPORTED_CYCLES:
        print()
        print("=" * 96)
        print(
            "HOUSE POLLING-WEIGHT MULTIPLIER "
            f"SWEEP: {cycle}"
        )
        print("=" * 96)

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
            raise SweepValidationError(
                f"Legacy margin unavailable for {cycle}: "
                f"{legacy_source}"
            )

        (
            production_df,
            model_margin,
            _,
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

            polling_df = (
                attach_polling_snapshot(
                    production_df,
                    snapshot,
                )
            )

            matched_seed = (
                int(args.seed)
                + int(cycle) * 1000
                + int(days_out)
            )

            print()
            print(
                f"{cycle} — {days_out:>3} days out "
                f"({snapshot_date})"
            )
            print("-" * 96)

            for multiplier in MULTIPLIERS:
                replay_spec = (
                    "historical_polling_weight_"
                    + str(multiplier)
                    .replace(".", "_")
                )

                (
                    results,
                    summary,
                    calibration,
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
                    polling_weight_multiplier=(
                        float(multiplier)
                    ),
                    replay_spec=replay_spec,
                )

                results = add_metadata(
                    results,
                    multiplier=float(multiplier),
                    days_out=int(days_out),
                    snapshot_date=snapshot_date,
                )

                summary = add_metadata(
                    summary,
                    multiplier=float(multiplier),
                    days_out=int(days_out),
                    snapshot_date=snapshot_date,
                )

                calibration = add_metadata(
                    calibration,
                    multiplier=float(multiplier),
                    days_out=int(days_out),
                    snapshot_date=snapshot_date,
                )

                summary[
                    "absolute_expected_seat_error"
                ] = pd.to_numeric(
                    summary[
                        "expected_seat_error"
                    ],
                    errors="coerce",
                ).abs()

                prediction_frames.append(
                    results
                )

                summary_frames.append(
                    summary
                )

                calibration_frames.append(
                    calibration
                )

                print(
                    f"{multiplier:>4.2f}×  "
                    f"Brier={float(summary.iloc[0]['brier_score']):.6f}  "
                    f"LogLoss={float(summary.iloc[0]['log_loss']):.6f}  "
                    f"SeatErr={float(summary.iloc[0]['expected_seat_error']):+.3f}  "
                    f"AvgWeight={float(summary.iloc[0]['average_polling_weight']):.4f}"
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

    expected_summary_rows = (
        len(SUPPORTED_CYCLES)
        * len(SUPPORTED_DAYS_OUT)
        * len(MULTIPLIERS)
    )

    if len(summaries) != expected_summary_rows:
        raise SweepValidationError(
            "Unexpected summary row count: "
            f"{len(summaries)}; expected "
            f"{expected_summary_rows}."
        )

    expected_prediction_rows = (
        expected_summary_rows
        * 435
    )

    if len(predictions) != expected_prediction_rows:
        raise SweepValidationError(
            "Unexpected prediction row count: "
            f"{len(predictions)}; expected "
            f"{expected_prediction_rows}."
        )

    duplicate_summary = summaries.duplicated(
        [
            "cycle",
            "historical_days_out",
            "polling_weight_multiplier",
        ],
        keep=False,
    )

    if duplicate_summary.any():
        raise SweepValidationError(
            "Duplicate cycle/snapshot/multiplier summaries."
        )

    zero_weights = pd.to_numeric(
        predictions.loc[
            predictions[
                "polling_weight_multiplier"
            ].eq(0.0),
            "bayesian_polling_weight",
        ],
        errors="coerce",
    ).fillna(0.0)

    if not zero_weights.eq(0.0).all():
        raise SweepValidationError(
            "Zero-multiplier arm contains nonzero polling weight."
        )

    summaries[
        "absolute_expected_seat_error"
    ] = pd.to_numeric(
        summaries[
            "expected_seat_error"
        ],
        errors="coerce",
    ).abs()

    overall = build_overall_summary(
        summaries
    )

    by_days_out = build_days_out_summary(
        summaries
    )

    predictions_path = (
        output_dir
        / "house_polling_weight_sweep_predictions.csv"
    )

    summaries_path = (
        output_dir
        / "house_polling_weight_sweep_by_snapshot.csv"
    )

    overall_path = (
        output_dir
        / "house_polling_weight_sweep_overall.csv"
    )

    days_out_path = (
        output_dir
        / "house_polling_weight_sweep_by_days_out.csv"
    )

    calibration_path = (
        output_dir
        / "house_polling_weight_sweep_calibration.csv"
    )

    validation_path = (
        output_dir
        / "house_polling_weight_sweep_validation.txt"
    )

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    summaries.to_csv(
        summaries_path,
        index=False,
    )

    overall.to_csv(
        overall_path,
        index=False,
    )

    by_days_out.to_csv(
        days_out_path,
        index=False,
    )

    calibrations.to_csv(
        calibration_path,
        index=False,
    )

    validation_text = "\n".join(
        [
            "House Polling-Weight Multiplier Sweep",
            "=" * 44,
            "",
            (
                "PASS: summary rows = "
                f"{len(summaries):,}"
            ),
            (
                "PASS: prediction rows = "
                f"{len(predictions):,}"
            ),
            (
                "PASS: zero-multiplier polling weights "
                "are exactly zero"
            ),
            (
                "PASS: existing production caps were "
                "retained for all multipliers"
            ),
            "",
            "VALIDATION PASSED",
            "",
        ]
    )

    validation_path.write_text(
        validation_text
    )

    display_columns = [
        "polling_weight_multiplier",
        "mean_brier_score",
        "mean_log_loss",
        "mean_abs_expected_seat_error",
        "mean_abs_margin_error",
        "mean_average_polling_weight",
        "snapshots_improved_brier_vs_zero",
        "snapshots_improved_log_loss_vs_zero",
        "cycles_improved_brier_vs_zero",
        "mean_primary_metric_rank",
        "production_multiplier",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in overall.columns
    ]

    print()
    print("=" * 120)
    print(
        "HOUSE POLLING-WEIGHT MULTIPLIER "
        "SWEEP — OVERALL"
    )
    print("=" * 120)

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
        overall_path,
        days_out_path,
        calibration_path,
        validation_path,
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
