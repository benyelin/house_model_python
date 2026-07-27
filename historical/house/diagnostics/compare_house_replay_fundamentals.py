#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from historical.house.backtests.run_house_full_production_replay import (
    SUPPORTED_CYCLES,
    build_model_margin,
    build_production_fundamentals,
    prepare_cycle,
    read_master,
)


def summarize_difference(
    label: str,
    legacy: pd.Series,
    production: pd.Series,
) -> pd.DataFrame:
    legacy_numeric = pd.to_numeric(
        legacy,
        errors="coerce",
    )
    production_numeric = pd.to_numeric(
        production,
        errors="coerce",
    )

    comparable_mask = (
        legacy_numeric.notna()
        & production_numeric.notna()
    )

    legacy_comparable = legacy_numeric.loc[comparable_mask]
    production_comparable = production_numeric.loc[
        comparable_mask
    ]

    difference = (
        production_comparable
        - legacy_comparable
    )

    if difference.empty:
        return pd.DataFrame(
            [
                {
                    "component": label,
                    "rows_compared": 0,
                    "legacy_mean": float("nan"),
                    "production_mean": float("nan"),
                    "mean_delta": float("nan"),
                    "median_delta": float("nan"),
                    "p95_abs_delta": float("nan"),
                    "max_abs_delta": float("nan"),
                }
            ]
        )

    return pd.DataFrame(
        [
            {
                "component": label,
                "rows_compared": int(len(difference)),
                "legacy_mean": float(
                    legacy_comparable.mean()
                ),
                "production_mean": float(
                    production_comparable.mean()
                ),
                "mean_delta": float(difference.mean()),
                "median_delta": float(
                    difference.median()
                ),
                "p95_abs_delta": float(
                    difference.abs().quantile(0.95)
                ),
                "max_abs_delta": float(
                    difference.abs().max()
                ),
            }
        ]
    )


def main() -> None:
    master = read_master()

    all_summaries: list[pd.DataFrame] = []

    for cycle in SUPPORTED_CYCLES:
        print()
        print("=" * 72)
        print(f"House fundamentals comparison: {cycle}")
        print("=" * 72)

        prepared_df, national_environment = prepare_cycle(
            master=master,
            cycle=cycle,
            candidate_quality_weight=1.0,
            candidate_war_path=None,
        )

        legacy_margin, legacy_source = (
            build_model_margin(
                prepared_df.copy()
            )
        )

        production_df, production_margin, production_source = (
            build_production_fundamentals(
                df=prepared_df.copy(),
                cycle=cycle,
                national_environment=(
                    national_environment
                ),
            )
        )

        print(
            f"Legacy margin source:     {legacy_source}"
        )
        print(
            f"Production margin source: "
            f"{production_source}"
        )
        print(
            f"National environment:     "
            f"{national_environment:+.3f}"
        )

        cycle_frames: list[pd.DataFrame] = []

        required_baseline_columns = {
            "district_pres_margin_dem",
            "district_partisan_baseline_dem",
        }

        if required_baseline_columns.issubset(
            set(prepared_df.columns)
            | set(production_df.columns)
        ):
            cycle_frames.append(
                summarize_difference(
                    label="partisan_baseline",
                    legacy=prepared_df[
                        "district_pres_margin_dem"
                    ],
                    production=production_df[
                        "district_partisan_baseline_dem"
                    ],
                )
            )
        else:
            missing = sorted(
                column
                for column in required_baseline_columns
                if (
                    column not in prepared_df.columns
                    and column
                    not in production_df.columns
                )
            )
            print(
                "Baseline comparison unavailable; "
                f"missing columns: {missing}"
            )

        if legacy_margin is not None:
            cycle_frames.append(
                summarize_difference(
                    label="model_margin",
                    legacy=legacy_margin,
                    production=production_margin,
                )
            )
        else:
            print(
                "Legacy model margin unavailable: "
                f"{legacy_source}"
            )

        if not cycle_frames:
            continue

        cycle_summary = pd.concat(
            cycle_frames,
            ignore_index=True,
        )
        cycle_summary.insert(0, "cycle", cycle)

        print()
        print(
            cycle_summary.to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.6f}"
                ),
            )
        )

        all_summaries.append(cycle_summary)

    if not all_summaries:
        raise RuntimeError(
            "No fundamentals comparisons were produced."
        )

    combined = pd.concat(
        all_summaries,
        ignore_index=True,
    )

    output_dir = (
        REPO_ROOT
        / "historical"
        / "house"
        / "diagnostics"
        / "outputs"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "house_replay_fundamentals_comparison.csv"
    )
    combined.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 72)
    print("Combined comparison")
    print("=" * 72)
    print(
        combined.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )
    print()
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
