from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from historical.house.build_house_district_elasticity import (  # noqa: E402
    build_elasticity_table,
)


SWING_PATH = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "elasticity"
    / "house_district_swing_observations_2012_2020.csv"
)

MASTER_PATH = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "warehouse"
    / "house_historical_backtest_inputs_2016_2022.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "elasticity"
    / "house_historical_district_elasticity_2016_2022.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "elasticity"
    / "house_historical_district_elasticity_summary.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "elasticity"
    / "house_historical_district_elasticity_validation.txt"
)


FORECAST_TRAINING_CUTOFFS = {
    2016: 2014,
    2018: 2016,
    2020: 2018,
    2022: 2020,
}

SHRINKAGE_STRENGTH = 0.50

# Match the operational elasticity limits used by the live House model.
ELASTICITY_FLOOR = 0.55
ELASTICITY_CEILING = 1.25
NEUTRAL_FALLBACK = 1.00

EXPECTED_ROWS_PER_CYCLE = 435
EXPECTED_TOTAL_ROWS = (
    EXPECTED_ROWS_PER_CYCLE
    * len(FORECAST_TRAINING_CUTOFFS)
)


class HistoricalElasticityError(RuntimeError):
    pass


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def build_cycle_universe(
    master: pd.DataFrame,
    forecast_cycle: int,
) -> pd.DataFrame:
    cycle_values = pd.to_numeric(
        master["forecast_cycle"],
        errors="coerce",
    )

    universe = master.loc[
        cycle_values.eq(forecast_cycle),
        [
            "forecast_cycle",
            "race_id",
            "state",
            "district",
        ],
    ].copy()

    universe = universe.drop_duplicates(
        ["forecast_cycle", "race_id"]
    )

    if len(universe) != EXPECTED_ROWS_PER_CYCLE:
        raise HistoricalElasticityError(
            f"{forecast_cycle}: expected "
            f"{EXPECTED_ROWS_PER_CYCLE} district rows; "
            f"found {len(universe)}."
        )

    if universe["race_id"].duplicated().any():
        raise HistoricalElasticityError(
            f"{forecast_cycle}: duplicate race_id values "
            "in the historical replay universe."
        )

    return universe


def estimate_cycle(
    observations: pd.DataFrame,
    universe: pd.DataFrame,
    forecast_cycle: int,
    training_end_cycle: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    cycle_to = pd.to_numeric(
        observations["cycle_to"],
        errors="coerce",
    )

    if cycle_to.isna().any():
        raise HistoricalElasticityError(
            "Swing observations contain missing or "
            "nonnumeric cycle_to values."
        )

    training = observations.loc[
        cycle_to.le(training_end_cycle)
    ].copy()

    if training.empty:
        raise HistoricalElasticityError(
            f"{forecast_cycle}: no observations remain through "
            f"training cycle {training_end_cycle}."
        )

    if pd.to_numeric(
        training["cycle_to"],
        errors="coerce",
    ).ge(forecast_cycle).any():
        raise HistoricalElasticityError(
            f"{forecast_cycle}: training data contain a transition "
            "ending in or after the forecast cycle."
        )

    estimates, _ = build_elasticity_table(
        observations=training,
        shrinkage_strength=SHRINKAGE_STRENGTH,
    )

    estimates = estimates.rename(
        columns={
            "shrunk_elasticity": (
                "historical_shrunk_elasticity_unbounded"
            ),
            "raw_elasticity": (
                "historical_raw_elasticity"
            ),
            "shrink_target": (
                "historical_elasticity_shrink_target"
            ),
            "shrinkage_strength": (
                "historical_elasticity_shrinkage_strength"
            ),
            "has_elasticity_estimate": (
                "historical_elasticity_estimate_available"
            ),
            "low_information_estimate": (
                "historical_elasticity_low_information"
            ),
        }
    )

    estimate_columns = [
        "race_id",
        "observation_count",
        "transition_count_available",
        "first_cycle_from",
        "last_cycle_to",
        "transitions_used",
        "historical_raw_elasticity",
        "historical_elasticity_shrink_target",
        "historical_elasticity_shrinkage_strength",
        "historical_shrunk_elasticity_unbounded",
        "raw_deviation_from_target",
        "shrinkage_adjustment",
        "district_swing_mean",
        "district_swing_sd",
        "residual_mean",
        "residual_rmse",
        "residual_sd",
        "national_swing_sum_squares",
        "historical_elasticity_estimate_available",
        "historical_elasticity_low_information",
        "estimation_method",
    ]

    joined = universe.merge(
        estimates[estimate_columns],
        on="race_id",
        how="left",
        validate="one_to_one",
    )

    estimate_available = as_bool(
        joined[
            "historical_elasticity_estimate_available"
        ]
    )

    unbounded = pd.to_numeric(
        joined[
            "historical_shrunk_elasticity_unbounded"
        ],
        errors="coerce",
    )

    estimate_available = (
        estimate_available
        & unbounded.notna()
        & np.isfinite(unbounded)
    )

    joined[
        "historical_elasticity_estimate_available"
    ] = estimate_available

    joined["district_elasticity_before_bounds"] = (
        unbounded.where(
            estimate_available,
            NEUTRAL_FALLBACK,
        )
    )

    joined["district_elasticity"] = (
        joined["district_elasticity_before_bounds"]
        .clip(
            lower=ELASTICITY_FLOOR,
            upper=ELASTICITY_CEILING,
        )
    )

    joined[
        "historical_elasticity_used_neutral_fallback"
    ] = ~estimate_available

    joined[
        "historical_elasticity_clipped_to_floor"
    ] = (
        estimate_available
        & joined[
            "district_elasticity_before_bounds"
        ].lt(ELASTICITY_FLOOR)
    )

    joined[
        "historical_elasticity_clipped_to_ceiling"
    ] = (
        estimate_available
        & joined[
            "district_elasticity_before_bounds"
        ].gt(ELASTICITY_CEILING)
    )

    joined["training_end_cycle"] = int(
        training_end_cycle
    )

    joined["elasticity_floor"] = float(
        ELASTICITY_FLOOR
    )
    joined["elasticity_ceiling"] = float(
        ELASTICITY_CEILING
    )
    joined["neutral_elasticity_fallback"] = float(
        NEUTRAL_FALLBACK
    )

    joined["historical_elasticity_method"] = (
        "expanding-window district OLS through origin; "
        "50% shrinkage toward information-weighted target; "
        "production bounds applied after estimation"
    )

    if len(joined) != EXPECTED_ROWS_PER_CYCLE:
        raise HistoricalElasticityError(
            f"{forecast_cycle}: row count changed after "
            f"elasticity merge: {len(joined)}."
        )

    if joined[
        ["forecast_cycle", "race_id"]
    ].duplicated().any():
        raise HistoricalElasticityError(
            f"{forecast_cycle}: duplicate output keys."
        )

    if not pd.to_numeric(
        joined["district_elasticity"],
        errors="coerce",
    ).between(
        ELASTICITY_FLOOR,
        ELASTICITY_CEILING,
        inclusive="both",
    ).all():
        raise HistoricalElasticityError(
            f"{forecast_cycle}: bounded district elasticity "
            "fell outside production limits."
        )

    last_cycle = pd.to_numeric(
        joined["last_cycle_to"],
        errors="coerce",
    )

    if last_cycle.dropna().gt(
        training_end_cycle
    ).any():
        raise HistoricalElasticityError(
            f"{forecast_cycle}: future elasticity observation "
            "survived the training cutoff."
        )

    summary = {
        "forecast_cycle": int(forecast_cycle),
        "training_end_cycle": int(
            training_end_cycle
        ),
        "district_rows": int(len(joined)),
        "training_observations": int(
            len(training)
        ),
        "districts_with_estimate": int(
            estimate_available.sum()
        ),
        "districts_using_neutral_fallback": int(
            (
                ~estimate_available
            ).sum()
        ),
        "districts_clipped_to_floor": int(
            joined[
                "historical_elasticity_clipped_to_floor"
            ].sum()
        ),
        "districts_clipped_to_ceiling": int(
            joined[
                "historical_elasticity_clipped_to_ceiling"
            ].sum()
        ),
        "mean_bounded_elasticity": float(
            joined["district_elasticity"].mean()
        ),
        "median_bounded_elasticity": float(
            joined["district_elasticity"].median()
        ),
        "min_bounded_elasticity": float(
            joined["district_elasticity"].min()
        ),
        "max_bounded_elasticity": float(
            joined["district_elasticity"].max()
        ),
        "latest_transition_used": int(
            last_cycle.dropna().max()
        ),
    }

    return joined, summary


def main() -> None:
    if not SWING_PATH.exists():
        raise FileNotFoundError(SWING_PATH)

    if not MASTER_PATH.exists():
        raise FileNotFoundError(MASTER_PATH)

    observations = pd.read_csv(
        SWING_PATH,
        dtype={
            "race_id": str,
            "state": str,
            "district": str,
        },
        low_memory=False,
    )

    master = pd.read_csv(
        MASTER_PATH,
        dtype={
            "race_id": str,
            "state": str,
            "district": str,
        },
        low_memory=False,
    )

    required_observation_columns = {
        "race_id",
        "state",
        "district",
        "cycle_from",
        "cycle_to",
        "transition",
        "district_swing_dem",
        "national_swing_dem",
        "eligible_for_elasticity_estimation",
    }

    missing_observation_columns = (
        required_observation_columns
        - set(observations.columns)
    )

    if missing_observation_columns:
        raise HistoricalElasticityError(
            "Swing observations are missing columns: "
            + ", ".join(
                sorted(
                    missing_observation_columns
                )
            )
        )

    required_master_columns = {
        "forecast_cycle",
        "race_id",
        "state",
        "district",
    }

    missing_master_columns = (
        required_master_columns
        - set(master.columns)
    )

    if missing_master_columns:
        raise HistoricalElasticityError(
            "Historical replay master is missing columns: "
            + ", ".join(
                sorted(missing_master_columns)
            )
        )

    cycle_frames = []
    summary_rows = []

    for (
        forecast_cycle,
        training_end_cycle,
    ) in FORECAST_TRAINING_CUTOFFS.items():
        universe = build_cycle_universe(
            master,
            forecast_cycle,
        )

        cycle_table, cycle_summary = estimate_cycle(
            observations=observations,
            universe=universe,
            forecast_cycle=forecast_cycle,
            training_end_cycle=training_end_cycle,
        )

        cycle_frames.append(cycle_table)
        summary_rows.append(cycle_summary)

    warehouse = pd.concat(
        cycle_frames,
        ignore_index=True,
    )

    summary = pd.DataFrame(summary_rows)

    if len(warehouse) != EXPECTED_TOTAL_ROWS:
        raise HistoricalElasticityError(
            f"Expected {EXPECTED_TOTAL_ROWS} total rows; "
            f"found {len(warehouse)}."
        )

    if warehouse[
        ["forecast_cycle", "race_id"]
    ].duplicated().any():
        raise HistoricalElasticityError(
            "Historical elasticity warehouse contains "
            "duplicate forecast-cycle/race keys."
        )

    if (
        pd.to_numeric(
            warehouse["training_end_cycle"],
            errors="coerce",
        )
        .ge(
            pd.to_numeric(
                warehouse["forecast_cycle"],
                errors="coerce",
            )
        )
        .any()
    ):
        raise HistoricalElasticityError(
            "At least one row violates the leakage-safe "
            "training cutoff."
        )

    warehouse = warehouse.sort_values(
        [
            "forecast_cycle",
            "state",
            "district",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    warehouse.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    validation_lines = [
        "Historical House District Elasticity Validation",
        "=" * 48,
        "",
        f"Output rows: {len(warehouse)}",
        (
            "Unique forecast-cycle/race keys: "
            f"{warehouse[['forecast_cycle', 'race_id']].drop_duplicates().shape[0]}"
        ),
        (
            "Production bounds: "
            f"{ELASTICITY_FLOOR:.2f} to "
            f"{ELASTICITY_CEILING:.2f}"
        ),
        (
            "Neutral fallback: "
            f"{NEUTRAL_FALLBACK:.2f}"
        ),
        "",
        "Cycle summary:",
        summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        ),
        "",
        "Validation status:",
        "PASSED",
    ]

    VALIDATION_PATH.write_text(
        "\n".join(validation_lines)
    )

    print(
        "HISTORICAL HOUSE DISTRICT ELASTICITY"
    )
    print("=" * 96)
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.6f}"
            ),
        )
    )

    print()
    print(f"Wrote: {OUTPUT_PATH}")
    print(f"Wrote: {SUMMARY_PATH}")
    print(f"Wrote: {VALIDATION_PATH}")

    print()
    print("VALIDATION PASSED")
    print(
        "Every historical elasticity uses only transitions "
        "ending before its forecast cycle."
    )


if __name__ == "__main__":
    main()
