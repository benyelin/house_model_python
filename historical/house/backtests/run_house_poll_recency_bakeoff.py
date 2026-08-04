from __future__ import annotations

import argparse
from pathlib import Path
import sys

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


RECENCY_SCHEMES = (
    "none",
    "gentle",
    "moderate",
    "half_life_180",
    "half_life_90",
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "poll_recency_bakeoff"
)


class RecencyBakeoffError(
    RuntimeError
):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare replay-only poll-age adjustments "
            "against the current House polling structure."
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


def main() -> None:
    args = parse_args()

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
            f"HOUSE POLL-RECENCY BAKEOFF: {cycle}"
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
            raise RecencyBakeoffError(
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

            snapshot_date = extract_snapshot_date(
                snapshot
            )

            polling_df = attach_polling_snapshot(
                production_df,
                snapshot,
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

            for scheme in RECENCY_SCHEMES:
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
                    polling_weight_multiplier=1.0,
                    polling_recency_scheme=scheme,
                    replay_spec=(
                        "historical_poll_recency_"
                        + scheme
                    ),
                )

                for frame in [
                    results,
                    summary,
                    calibration,
                ]:
                    frame[
                        "polling_recency_scheme"
                    ] = scheme

                    frame[
                        "historical_days_out"
                    ] = int(days_out)

                    frame[
                        "historical_snapshot_date"
                    ] = snapshot_date

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
                    f"{scheme:<14} "
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
        * len(RECENCY_SCHEMES)
    )

    if len(summaries) != expected_summary_rows:
        raise RecencyBakeoffError(
            "Unexpected summary row count: "
            f"{len(summaries)}; expected "
            f"{expected_summary_rows}."
        )

    expected_prediction_rows = (
        expected_summary_rows * 435
    )

    if len(predictions) != expected_prediction_rows:
        raise RecencyBakeoffError(
            "Unexpected prediction row count: "
            f"{len(predictions)}; expected "
            f"{expected_prediction_rows}."
        )

    duplicate_mask = summaries.duplicated(
        [
            "cycle",
            "historical_days_out",
            "polling_recency_scheme",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        raise RecencyBakeoffError(
            "Duplicate cycle/snapshot/scheme summaries."
        )

    overall = (
        summaries.groupby(
            "polling_recency_scheme",
            as_index=False,
        )
        .agg(
            snapshots=(
                "historical_days_out",
                "size",
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
        )
    )

    control = summaries.loc[
        summaries[
            "polling_recency_scheme"
        ].eq("none"),
        [
            "cycle",
            "historical_days_out",
            "brier_score",
            "log_loss",
            "absolute_expected_seat_error",
        ],
    ].rename(
        columns={
            "brier_score":
                "control_brier_score",
            "log_loss":
                "control_log_loss",
            "absolute_expected_seat_error":
                "control_abs_seat_error",
        }
    )

    matched = summaries.merge(
        control,
        on=[
            "cycle",
            "historical_days_out",
        ],
        how="left",
        validate="many_to_one",
    )

    matched[
        "brier_improved_vs_control"
    ] = (
        pd.to_numeric(
            matched["brier_score"],
            errors="coerce",
        )
        < pd.to_numeric(
            matched[
                "control_brier_score"
            ],
            errors="coerce",
        )
    )

    matched[
        "log_loss_improved_vs_control"
    ] = (
        pd.to_numeric(
            matched["log_loss"],
            errors="coerce",
        )
        < pd.to_numeric(
            matched[
                "control_log_loss"
            ],
            errors="coerce",
        )
    )

    matched[
        "seat_error_improved_vs_control"
    ] = (
        pd.to_numeric(
            matched[
                "absolute_expected_seat_error"
            ],
            errors="coerce",
        )
        < pd.to_numeric(
            matched[
                "control_abs_seat_error"
            ],
            errors="coerce",
        )
    )

    wins = (
        matched.groupby(
            "polling_recency_scheme",
            as_index=False,
        )
        .agg(
            snapshots_improved_brier=(
                "brier_improved_vs_control",
                "sum",
            ),
            snapshots_improved_log_loss=(
                "log_loss_improved_vs_control",
                "sum",
            ),
            snapshots_improved_seat_error=(
                "seat_error_improved_vs_control",
                "sum",
            ),
        )
    )

    overall = overall.merge(
        wins,
        on="polling_recency_scheme",
        how="left",
        validate="one_to_one",
    )

    for metric in [
        "mean_brier_score",
        "mean_log_loss",
        "mean_abs_expected_seat_error",
    ]:
        overall[
            f"{metric}_rank"
        ] = overall[
            metric
        ].rank(
            method="min",
            ascending=True,
        )

    overall[
        "mean_primary_metric_rank"
    ] = overall[
        [
            "mean_brier_score_rank",
            "mean_log_loss_rank",
            "mean_abs_expected_seat_error_rank",
        ]
    ].mean(axis=1)

    overall[
        "production_control"
    ] = overall[
        "polling_recency_scheme"
    ].eq("none")

    overall = overall.sort_values(
        [
            "mean_primary_metric_rank",
            "mean_brier_score",
        ],
        ascending=True,
    ).reset_index(drop=True)

    by_days_out = (
        summaries.groupby(
            [
                "polling_recency_scheme",
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
        )
        .sort_values(
            [
                "historical_days_out",
                "polling_recency_scheme",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    predictions_path = (
        output_dir
        / "house_poll_recency_predictions.csv"
    )

    summaries_path = (
        output_dir
        / "house_poll_recency_by_snapshot.csv"
    )

    overall_path = (
        output_dir
        / "house_poll_recency_overall.csv"
    )

    days_out_path = (
        output_dir
        / "house_poll_recency_by_days_out.csv"
    )

    calibration_path = (
        output_dir
        / "house_poll_recency_calibration.csv"
    )

    validation_path = (
        output_dir
        / "house_poll_recency_validation.txt"
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
            "House Poll-Recency Bakeoff",
            "=" * 36,
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
                "PASS: all five recency schemes "
                "completed all 21 snapshots"
            ),
            "",
            "VALIDATION PASSED",
            "",
        ]
    )

    validation_path.write_text(
        validation_text
    )

    print()
    print("=" * 120)
    print(
        "HOUSE POLL-RECENCY BAKEOFF — OVERALL"
    )
    print("=" * 120)

    print(
        overall.to_string(
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
