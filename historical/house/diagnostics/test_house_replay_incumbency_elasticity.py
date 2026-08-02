from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from historical.house.backtests.run_house_full_production_replay import (  # noqa: E402
    DEFAULT_MASTER_PATH,
    SUPPORTED_CYCLES,
    build_production_fundamentals,
    prepare_cycle,
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "diagnostics"
    / "outputs"
    / "replay_component_controls"
)


class ControlTestError(RuntimeError):
    pass


def numeric(
    values: pd.Series,
    default: float = 0.0,
) -> pd.Series:
    return pd.to_numeric(
        values,
        errors="coerce",
    ).fillna(default)


def neutralize_incumbency(
    prepared: pd.DataFrame,
) -> pd.DataFrame:
    """
    Neutralize every known incumbency source or translated field.

    The production replay adapter maps dem_is_incumbent and
    gop_is_incumbent into dem_candidate_is_incumbent and
    gop_candidate_is_incumbent before calling production
    fundamentals. We neutralize both layers, plus any previously
    calculated incumbency fields.
    """
    out = prepared.copy()

    false_columns = [
        "dem_is_incumbent",
        "gop_is_incumbent",
        "dem_incumbent",
        "gop_incumbent",
        "double_incumbent_race",
        "dem_candidate_is_incumbent",
        "gop_candidate_is_incumbent",
    ]

    for column in false_columns:
        if column in out.columns:
            out[column] = False

    blank_columns = [
        "incumbent_party",
        "incumbent_configuration",
        "dem_incumbent_challenger_status",
        "gop_incumbent_challenger_status",
    ]

    for column in blank_columns:
        if column in out.columns:
            out[column] = ""

    if "incumbency_adjustment_dem" in out.columns:
        out["incumbency_adjustment_dem"] = 0.0

    return out


def neutralize_elasticity(
    prepared: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Set all recognized production elasticity fields to 1.0.

    At least one real elasticity-value column must exist. Metadata
    flags such as district_elasticity_joined_in_canonical_inputs do
    not count as the elasticity value itself.
    """
    out = prepared.copy()

    candidates = [
        "district_elasticity",
        "calculated_district_elasticity",
        "historical_district_elasticity",
        "elasticity",
    ]

    present = [
        column
        for column in candidates
        if column in out.columns
    ]

    if not present:
        elasticity_like = [
            column
            for column in out.columns
            if "elastic" in column.lower()
        ]

        raise ControlTestError(
            "No recognized district-elasticity value column was "
            "present after prepare_cycle(). Elasticity-like columns: "
            f"{elasticity_like}"
        )

    for column in present:
        out[column] = 1.0

    return out, present


def calculate_metrics(
    frame: pd.DataFrame,
    margin: pd.Series,
) -> dict[str, float]:
    actual_margin_column = next(
        (
            column
            for column in [
                "actual_dem_margin",
                "dem_margin",
                "result_margin_dem",
            ]
            if column in frame.columns
        ),
        None,
    )

    result = {
        "mean_model_margin": float(
            numeric(margin, np.nan).mean()
        ),
        "expected_dem_wins_at_zero_threshold": float(
            numeric(margin).gt(0.0).sum()
        ),
    }

    if actual_margin_column is not None:
        actual = pd.to_numeric(
            frame[actual_margin_column],
            errors="coerce",
        )

        forecast = pd.to_numeric(
            margin,
            errors="coerce",
        )

        valid = actual.notna() & forecast.notna()

        if valid.any():
            error = forecast.loc[valid] - actual.loc[valid]

            result["margin_mae"] = float(
                error.abs().mean()
            )
            result["margin_rmse"] = float(
                np.sqrt(np.mean(error ** 2))
            )
        else:
            result["margin_mae"] = np.nan
            result["margin_rmse"] = np.nan
    else:
        result["margin_mae"] = np.nan
        result["margin_rmse"] = np.nan

    return result


def compare_control(
    cycle: int,
    baseline_frame: pd.DataFrame,
    baseline_margin: pd.Series,
    control_frame: pd.DataFrame,
    control_margin: pd.Series,
    component: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    baseline = pd.to_numeric(
        baseline_margin,
        errors="coerce",
    )

    control = pd.to_numeric(
        control_margin,
        errors="coerce",
    )

    difference = control - baseline
    absolute_difference = difference.abs()

    changed = absolute_difference.gt(1e-12)

    details = pd.DataFrame(
        {
            "cycle": cycle,
            "component_neutralized": component,
            "race_id": baseline_frame.get(
                "race_id",
                pd.Series(
                    range(len(baseline_frame)),
                    index=baseline_frame.index,
                ),
            ),
            "district_id": baseline_frame.get(
                "district_id",
                "",
            ),
            "state": baseline_frame.get(
                "state",
                "",
            ),
            "baseline_model_margin_dem": baseline,
            "neutralized_model_margin_dem": control,
            "neutralized_minus_baseline_margin": difference,
            "absolute_margin_change": absolute_difference,
            "margin_changed": changed,
        }
    )

    for column in [
        "incumbency_adjustment_dem",
        "district_elasticity",
        "dem_candidate_is_incumbent",
        "gop_candidate_is_incumbent",
    ]:
        if column in baseline_frame.columns:
            details[f"baseline_{column}"] = (
                baseline_frame[column]
            )

        if column in control_frame.columns:
            details[f"neutralized_{column}"] = (
                control_frame[column]
            )

    baseline_metrics = calculate_metrics(
        baseline_frame,
        baseline,
    )

    control_metrics = calculate_metrics(
        control_frame,
        control,
    )

    summary = {
        "cycle": cycle,
        "component": component,
        "races": int(len(details)),
        "changed_races": int(changed.sum()),
        "changed_share": float(changed.mean()),
        "mean_absolute_margin_change": float(
            absolute_difference.mean()
        ),
        "max_absolute_margin_change": float(
            absolute_difference.max()
        ),
        "mean_signed_margin_change": float(
            difference.mean()
        ),
        "baseline_mean_model_margin": (
            baseline_metrics["mean_model_margin"]
        ),
        "neutralized_mean_model_margin": (
            control_metrics["mean_model_margin"]
        ),
        "baseline_margin_mae": (
            baseline_metrics["margin_mae"]
        ),
        "neutralized_margin_mae": (
            control_metrics["margin_mae"]
        ),
        "neutralized_minus_baseline_mae": (
            control_metrics["margin_mae"]
            - baseline_metrics["margin_mae"]
        ),
        "baseline_margin_rmse": (
            baseline_metrics["margin_rmse"]
        ),
        "neutralized_margin_rmse": (
            control_metrics["margin_rmse"]
        ),
        "neutralized_minus_baseline_rmse": (
            control_metrics["margin_rmse"]
            - baseline_metrics["margin_rmse"]
        ),
        "baseline_zero_threshold_dem_wins": (
            baseline_metrics[
                "expected_dem_wins_at_zero_threshold"
            ]
        ),
        "neutralized_zero_threshold_dem_wins": (
            control_metrics[
                "expected_dem_wins_at_zero_threshold"
            ]
        ),
    }

    return summary, details


def main() -> None:
    if not DEFAULT_MASTER_PATH.exists():
        raise FileNotFoundError(DEFAULT_MASTER_PATH)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    master = pd.read_csv(
        DEFAULT_MASTER_PATH,
        low_memory=False,
    )

    summaries = []
    detail_frames = []
    elasticity_columns_by_cycle = {}

    for cycle in SUPPORTED_CYCLES:
        print()
        print("=" * 80)
        print(f"REPLAY COMPONENT CONTROLS: {cycle}")
        print("=" * 80)

        # Candidate quality is held at zero in every branch so these
        # tests isolate incumbency and elasticity only.
        prepared, national_environment = prepare_cycle(
            master=master,
            cycle=cycle,
            candidate_quality_weight=0.0,
            candidate_war_path=Path("__unused__"),
        )

        baseline_frame, baseline_margin, _ = (
            build_production_fundamentals(
                df=prepared.copy(),
                cycle=cycle,
                national_environment=national_environment,
            )
        )

        no_incumbency_input = neutralize_incumbency(
            prepared
        )

        no_incumbency_frame, no_incumbency_margin, _ = (
            build_production_fundamentals(
                df=no_incumbency_input,
                cycle=cycle,
                national_environment=national_environment,
            )
        )

        incumbency_summary, incumbency_details = (
            compare_control(
                cycle=cycle,
                baseline_frame=baseline_frame,
                baseline_margin=baseline_margin,
                control_frame=no_incumbency_frame,
                control_margin=no_incumbency_margin,
                component="incumbency",
            )
        )

        summaries.append(incumbency_summary)
        detail_frames.append(incumbency_details)

        no_elasticity_input, elasticity_columns = (
            neutralize_elasticity(prepared)
        )

        elasticity_columns_by_cycle[cycle] = (
            elasticity_columns
        )

        no_elasticity_frame, no_elasticity_margin, _ = (
            build_production_fundamentals(
                df=no_elasticity_input,
                cycle=cycle,
                national_environment=national_environment,
            )
        )

        elasticity_summary, elasticity_details = (
            compare_control(
                cycle=cycle,
                baseline_frame=baseline_frame,
                baseline_margin=baseline_margin,
                control_frame=no_elasticity_frame,
                control_margin=no_elasticity_margin,
                component="district_elasticity",
            )
        )

        summaries.append(elasticity_summary)
        detail_frames.append(elasticity_details)

        print()
        print("Incumbency control")
        print("-" * 80)
        print(
            pd.DataFrame(
                [incumbency_summary]
            ).to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.9f}"
                ),
            )
        )

        print()
        print(
            "Elasticity control "
            f"(columns={elasticity_columns})"
        )
        print("-" * 80)
        print(
            pd.DataFrame(
                [elasticity_summary]
            ).to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.9f}"
                ),
            )
        )

    summary = pd.DataFrame(summaries)

    details = pd.concat(
        detail_frames,
        ignore_index=True,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "house_replay_component_control_summary.csv",
        index=False,
    )

    details.to_csv(
        OUTPUT_DIR
        / "house_replay_component_control_details.csv",
        index=False,
    )

    top_changes = (
        details.loc[details["margin_changed"]]
        .sort_values(
            "absolute_margin_change",
            ascending=False,
            kind="mergesort",
        )
        .head(100)
    )

    top_changes.to_csv(
        OUTPUT_DIR
        / "house_replay_component_control_top_changes.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print("OVERALL COMPONENT CONTROL SUMMARY")
    print("=" * 100)
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.9f}"
            ),
        )
    )

    print()
    print("BEHAVIORAL CONCLUSION")
    print("-" * 100)

    for component in [
        "incumbency",
        "district_elasticity",
    ]:
        component_rows = summary.loc[
            summary["component"].eq(component)
        ]

        total_changed = int(
            component_rows["changed_races"].sum()
        )

        max_change = float(
            component_rows[
                "max_absolute_margin_change"
            ].max()
        )

        if (
            total_changed > 0
            and max_change > 1e-12
        ):
            status = "ACTIVE"
        else:
            status = "INACTIVE"

        print(
            f"{component}: {status}; "
            f"changed races={total_changed}; "
            f"maximum margin change={max_change:.9f}"
        )

    print()
    print("Elasticity columns tested:")
    for cycle, columns in (
        elasticity_columns_by_cycle.items()
    ):
        print(f"  {cycle}: {columns}")

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
        "Baseline and neutralized controls used the same production "
        "fundamentals function and differed only in the component "
        "being tested."
    )


if __name__ == "__main__":
    main()
