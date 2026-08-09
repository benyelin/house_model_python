from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]

INPUT_DIR = (
    REPO_ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "historical_polling_experiment"
)

OUTPUT_DIR = (
    INPUT_DIR
    / "days_out_analysis"
)

DISTRICT_PATH = (
    INPUT_DIR
    / "house_historical_polling_district_comparison.csv"
)

SNAPSHOT_PATH = (
    INPUT_DIR
    / "house_historical_polling_matched_comparison.csv"
)


def numeric(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(
            np.nan,
            index=frame.index,
            dtype=float,
        )

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    )


def find_first_column(
    frame: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column

    return None


def main() -> None:
    if not DISTRICT_PATH.exists():
        raise FileNotFoundError(
            DISTRICT_PATH
        )

    if not SNAPSHOT_PATH.exists():
        raise FileNotFoundError(
            SNAPSHOT_PATH
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    districts = pd.read_csv(
        DISTRICT_PATH,
        low_memory=False,
    )

    snapshots = pd.read_csv(
        SNAPSHOT_PATH,
        low_memory=False,
    )

    required = {
        "cycle",
        "historical_days_out",
    }

    missing = sorted(
        required - set(districts.columns)
    )

    if missing:
        raise RuntimeError(
            "District comparison file is missing: "
            + ", ".join(missing)
        )

    # ----------------------------------------------------------
    # Locate the realized Democratic margin / winner.
    # ----------------------------------------------------------

    actual_margin_column = (
        find_first_column(
            districts,
            [
                "actual_margin_dem_fundamentals",
                "actual_margin_dem_polling",
                "actual_dem_margin_fundamentals",
                "actual_dem_margin_polling",
                "dem_margin_actual_fundamentals",
                "dem_margin_actual_polling",
                "actual_margin_dem",
                "actual_dem_margin",
            ],
        )
    )

    actual_win_column = (
        find_first_column(
            districts,
            [
                "actual_dem_win_fundamentals",
                "actual_dem_win_polling",
                "dem_win_actual_fundamentals",
                "dem_win_actual_polling",
                "actual_dem_win",
            ],
        )
    )

    fund_margin_column = (
        find_first_column(
            districts,
            [
                "model_margin_dem_fundamentals",
                "bayesian_model_margin_dem_fundamentals",
            ],
        )
    )

    poll_margin_column = (
        find_first_column(
            districts,
            [
                "model_margin_dem_polling",
                "bayesian_model_margin_dem_polling",
            ],
        )
    )

    fund_prob_column = (
        find_first_column(
            districts,
            [
                "dem_win_probability_fundamentals",
                "simulated_dem_win_probability_fundamentals",
                "raw_simulated_dem_win_probability_fundamentals",
            ],
        )
    )

    poll_prob_column = (
        find_first_column(
            districts,
            [
                "dem_win_probability_polling",
                "simulated_dem_win_probability_polling",
                "raw_simulated_dem_win_probability_polling",
            ],
        )
    )

    poll_count_column = (
        find_first_column(
            districts,
            [
                "poll_count_polling",
                "poll_quality_count_polling",
            ],
        )
    )

    polling_weight_column = (
        find_first_column(
            districts,
            [
                "bayesian_polling_weight_polling",
            ],
        )
    )

    if fund_margin_column is None:
        raise RuntimeError(
            "Could not identify fundamentals model-margin column."
        )

    if poll_margin_column is None:
        raise RuntimeError(
            "Could not identify polling model-margin column."
        )

    if fund_prob_column is None:
        raise RuntimeError(
            "Could not identify fundamentals probability column."
        )

    if poll_prob_column is None:
        raise RuntimeError(
            "Could not identify polling probability column."
        )

    fund_margin = numeric(
        districts,
        fund_margin_column,
    )

    poll_margin = numeric(
        districts,
        poll_margin_column,
    )

    fund_prob = numeric(
        districts,
        fund_prob_column,
    )

    poll_prob = numeric(
        districts,
        poll_prob_column,
    )

    if poll_count_column is not None:
        poll_count = numeric(
            districts,
            poll_count_column,
        ).fillna(0.0)
    else:
        poll_count = pd.Series(
            0.0,
            index=districts.index,
        )

    if polling_weight_column is not None:
        polling_weight = numeric(
            districts,
            polling_weight_column,
        ).fillna(0.0)
    else:
        polling_weight = pd.Series(
            0.0,
            index=districts.index,
        )

    districts[
        "_poll_count"
    ] = poll_count

    districts[
        "_polling_weight"
    ] = polling_weight

    districts[
        "_polling_active"
    ] = poll_count.gt(0.0)

    districts[
        "_margin_change"
    ] = (
        poll_margin - fund_margin
    )

    districts[
        "_probability_change"
    ] = (
        poll_prob - fund_prob
    )

    districts[
        "_absolute_probability_change"
    ] = (
        districts[
            "_probability_change"
        ].abs()
    )

    districts[
        "_winner_call_fundamentals"
    ] = fund_prob.ge(0.5)

    districts[
        "_winner_call_polling"
    ] = poll_prob.ge(0.5)

    districts[
        "_winner_call_changed"
    ] = (
        districts[
            "_winner_call_fundamentals"
        ]
        != districts[
            "_winner_call_polling"
        ]
    )

    # ----------------------------------------------------------
    # Actual-result metrics if realized margins are available.
    # ----------------------------------------------------------

    if actual_margin_column is not None:
        actual_margin = numeric(
            districts,
            actual_margin_column,
        )

        districts[
            "_fund_margin_abs_error"
        ] = (
            fund_margin
            - actual_margin
        ).abs()

        districts[
            "_poll_margin_abs_error"
        ] = (
            poll_margin
            - actual_margin
        ).abs()

        districts[
            "_margin_mae_improvement"
        ] = (
            districts[
                "_fund_margin_abs_error"
            ]
            - districts[
                "_poll_margin_abs_error"
            ]
        )

    if actual_win_column is not None:
        actual_win = numeric(
            districts,
            actual_win_column,
        ).eq(1.0)

    elif actual_margin_column is not None:
        actual_win = numeric(
            districts,
            actual_margin_column,
        ).gt(0.0)

    else:
        actual_win = None

    if actual_win is not None:
        districts[
            "_fund_winner_correct"
        ] = (
            districts[
                "_winner_call_fundamentals"
            ]
            == actual_win
        )

        districts[
            "_poll_winner_correct"
        ] = (
            districts[
                "_winner_call_polling"
            ]
            == actual_win
        )

    # ----------------------------------------------------------
    # Days-out summary.
    # ----------------------------------------------------------

    rows = []

    for days_out, group in districts.groupby(
        "historical_days_out",
        sort=True,
    ):
        polled = group.loc[
            group[
                "_polling_active"
            ]
        ]

        row = {
            "days_out":
                int(days_out),

            "district_rows":
                int(len(group)),

            "polled_district_rows":
                int(len(polled)),

            "polling_coverage":
                float(
                    group[
                        "_polling_active"
                    ].mean()
                ),

            "mean_polling_weight_all":
                float(
                    group[
                        "_polling_weight"
                    ].mean()
                ),

            "mean_polling_weight_polled":
                float(
                    polled[
                        "_polling_weight"
                    ].mean()
                )
                if len(polled)
                else 0.0,

            "median_polling_weight_polled":
                float(
                    polled[
                        "_polling_weight"
                    ].median()
                )
                if len(polled)
                else 0.0,

            "mean_abs_probability_change":
                float(
                    group[
                        "_absolute_probability_change"
                    ].mean()
                ),

            "mean_abs_probability_change_polled":
                float(
                    polled[
                        "_absolute_probability_change"
                    ].mean()
                )
                if len(polled)
                else 0.0,

            "districts_probability_move_2pt":
                int(
                    group[
                        "_absolute_probability_change"
                    ].ge(0.02).sum()
                ),

            "districts_probability_move_5pt":
                int(
                    group[
                        "_absolute_probability_change"
                    ].ge(0.05).sum()
                ),

            "districts_probability_move_10pt":
                int(
                    group[
                        "_absolute_probability_change"
                    ].ge(0.10).sum()
                ),

            "winner_calls_changed":
                int(
                    group[
                        "_winner_call_changed"
                    ].sum()
                ),
        }

        if (
            "_fund_margin_abs_error"
            in group.columns
        ):
            row[
                "fundamentals_margin_mae"
            ] = float(
                group[
                    "_fund_margin_abs_error"
                ].mean()
            )

            row[
                "polling_margin_mae"
            ] = float(
                group[
                    "_poll_margin_abs_error"
                ].mean()
            )

            row[
                "margin_mae_improvement"
            ] = (
                row[
                    "fundamentals_margin_mae"
                ]
                - row[
                    "polling_margin_mae"
                ]
            )

        if (
            "_fund_winner_correct"
            in group.columns
        ):
            row[
                "fundamentals_winner_accuracy"
            ] = float(
                group[
                    "_fund_winner_correct"
                ].mean()
            )

            row[
                "polling_winner_accuracy"
            ] = float(
                group[
                    "_poll_winner_correct"
                ].mean()
            )

            row[
                "winner_accuracy_improvement"
            ] = (
                row[
                    "polling_winner_accuracy"
                ]
                - row[
                    "fundamentals_winner_accuracy"
                ]
            )

        rows.append(row)

    district_summary = (
        pd.DataFrame(rows)
        .sort_values(
            "days_out",
            ascending=False,
        )
    )

    # ----------------------------------------------------------
    # Bring in the already-computed Brier/log-loss/seat metrics.
    # ----------------------------------------------------------

    snapshot_days = (
        snapshots.groupby(
            "historical_days_out",
            as_index=False,
        )
        .agg(
            cycles=(
                "cycle",
                "nunique",
            ),

            fundamentals_brier=(
                "brier_score_fundamentals",
                "mean",
            ),

            polling_brier=(
                "brier_score_polling",
                "mean",
            ),

            fundamentals_log_loss=(
                "log_loss_fundamentals",
                "mean",
            ),

            polling_log_loss=(
                "log_loss_polling",
                "mean",
            ),

            fundamentals_abs_seat_error=(
                "expected_seat_error_fundamentals",
                lambda x: (
                    pd.to_numeric(
                        x,
                        errors="coerce",
                    )
                    .abs()
                    .mean()
                ),
            ),

            polling_abs_seat_error=(
                "expected_seat_error_polling",
                lambda x: (
                    pd.to_numeric(
                        x,
                        errors="coerce",
                    )
                    .abs()
                    .mean()
                ),
            ),
        )
    )

    snapshot_days[
        "brier_improvement"
    ] = (
        snapshot_days[
            "fundamentals_brier"
        ]
        - snapshot_days[
            "polling_brier"
        ]
    )

    snapshot_days[
        "log_loss_improvement"
    ] = (
        snapshot_days[
            "fundamentals_log_loss"
        ]
        - snapshot_days[
            "polling_log_loss"
        ]
    )

    snapshot_days[
        "seat_error_improvement"
    ] = (
        snapshot_days[
            "fundamentals_abs_seat_error"
        ]
        - snapshot_days[
            "polling_abs_seat_error"
        ]
    )

    combined = district_summary.merge(
        snapshot_days,
        left_on="days_out",
        right_on="historical_days_out",
        how="left",
        validate="one_to_one",
    )

    combined = combined.drop(
        columns=[
            "historical_days_out"
        ],
        errors="ignore",
    )

    # ----------------------------------------------------------
    # Snapshot win/loss table.
    # ----------------------------------------------------------

    snapshot_detail = snapshots.copy()

    snapshot_detail[
        "polling_improves_brier"
    ] = (
        numeric(
            snapshot_detail,
            "brier_score_polling",
        )
        < numeric(
            snapshot_detail,
            "brier_score_fundamentals",
        )
    )

    snapshot_detail[
        "polling_improves_log_loss"
    ] = (
        numeric(
            snapshot_detail,
            "log_loss_polling",
        )
        < numeric(
            snapshot_detail,
            "log_loss_fundamentals",
        )
    )

    snapshot_detail[
        "polling_improves_abs_seat_error"
    ] = (
        numeric(
            snapshot_detail,
            "expected_seat_error_polling",
        ).abs()
        < numeric(
            snapshot_detail,
            "expected_seat_error_fundamentals",
        ).abs()
    )

    # ----------------------------------------------------------
    # Save.
    # ----------------------------------------------------------

    combined_path = (
        OUTPUT_DIR
        / "house_polling_days_out_summary.csv"
    )

    snapshot_detail_path = (
        OUTPUT_DIR
        / "house_polling_snapshot_decisions.csv"
    )

    movers_path = (
        OUTPUT_DIR
        / "house_polling_largest_probability_moves.csv"
    )

    combined.to_csv(
        combined_path,
        index=False,
    )

    snapshot_detail.to_csv(
        snapshot_detail_path,
        index=False,
    )

    movers = (
        districts.sort_values(
            "_absolute_probability_change",
            ascending=False,
        )
        .head(200)
    )

    movers.to_csv(
        movers_path,
        index=False,
    )

    print(
        "HOUSE HISTORICAL POLLING — "
        "DAYS-OUT REPLAY"
    )
    print("=" * 120)

    display = [
        "days_out",
        "polling_coverage",
        "mean_polling_weight_polled",
        "mean_abs_probability_change_polled",
        "districts_probability_move_5pt",
        "winner_calls_changed",
        "fundamentals_brier",
        "polling_brier",
        "brier_improvement",
        "fundamentals_log_loss",
        "polling_log_loss",
        "log_loss_improvement",
        "fundamentals_abs_seat_error",
        "polling_abs_seat_error",
        "seat_error_improvement",
    ]

    display = [
        column
        for column in display
        if column in combined.columns
    ]

    print(
        combined[
            display
        ].to_string(
            index=False
        )
    )

    print()
    print("SNAPSHOT WIN COUNTS")
    print("-" * 120)

    print(
        "Brier:    "
        f"{int(snapshot_detail['polling_improves_brier'].sum())}"
        f" / {len(snapshot_detail)}"
    )

    print(
        "Log loss: "
        f"{int(snapshot_detail['polling_improves_log_loss'].sum())}"
        f" / {len(snapshot_detail)}"
    )

    print(
        "Seat err: "
        f"{int(snapshot_detail['polling_improves_abs_seat_error'].sum())}"
        f" / {len(snapshot_detail)}"
    )

    print()
    print("Wrote:")
    print(f"  {combined_path}")
    print(f"  {snapshot_detail_path}")
    print(f"  {movers_path}")


if __name__ == "__main__":
    main()
