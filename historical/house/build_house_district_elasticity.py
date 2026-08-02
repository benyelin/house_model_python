from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT_PATH = Path(
    "historical/house/elasticity/"
    "house_district_swing_observations_2012_2020.csv"
)

DEFAULT_OUTPUT_PATH = Path(
    "historical/house/elasticity/"
    "house_district_elasticity.csv"
)

DEFAULT_VALIDATION_PATH = Path(
    "historical/house/elasticity/"
    "house_district_elasticity_validation.txt"
)


def estimate_origin_slope(group: pd.DataFrame) -> float:
    """
    Estimate district_swing = elasticity * national_swing.

    The regression is constrained through the origin because a zero
    national swing should imply a zero systematic district swing.
    """
    x = pd.to_numeric(
        group["national_swing_dem"],
        errors="coerce",
    ).to_numpy(dtype=float)

    y = pd.to_numeric(
        group["district_swing_dem"],
        errors="coerce",
    ).to_numpy(dtype=float)

    denominator = float(np.dot(x, x))

    if denominator <= 0:
        return np.nan

    return float(np.dot(x, y) / denominator)


def estimate_district(group: pd.DataFrame) -> dict[str, object]:
    """Estimate raw elasticity and residual diagnostics for one district."""
    work = group.loc[
        group["eligible_for_elasticity_estimation"].eq(True)
    ].copy()

    observation_count = len(work)

    if observation_count == 0:
        return {
            "observation_count": 0,
            "raw_elasticity": np.nan,
            "district_swing_mean": np.nan,
            "district_swing_sd": np.nan,
            "residual_mean": np.nan,
            "residual_rmse": np.nan,
            "residual_sd": np.nan,
            "national_swing_sum_squares": np.nan,
        }

    elasticity = estimate_origin_slope(work)

    x = pd.to_numeric(
        work["national_swing_dem"],
        errors="coerce",
    ).to_numpy(dtype=float)

    y = pd.to_numeric(
        work["district_swing_dem"],
        errors="coerce",
    ).to_numpy(dtype=float)

    fitted = elasticity * x
    residuals = y - fitted

    residual_rmse = float(
        np.sqrt(np.mean(np.square(residuals)))
    )

    residual_sd = (
        float(np.std(residuals, ddof=1))
        if observation_count >= 2
        else np.nan
    )

    district_swing_sd = (
        float(np.std(y, ddof=1))
        if observation_count >= 2
        else np.nan
    )

    return {
        "observation_count": observation_count,
        "raw_elasticity": elasticity,
        "district_swing_mean": float(np.mean(y)),
        "district_swing_sd": district_swing_sd,
        "residual_mean": float(np.mean(residuals)),
        "residual_rmse": residual_rmse,
        "residual_sd": residual_sd,
        "national_swing_sum_squares": float(np.dot(x, x)),
    }


def build_elasticity_table(
    observations: pd.DataFrame,
    shrinkage_strength: float,
) -> tuple[pd.DataFrame, str]:
    """
    Estimate district elasticities and apply configurable shrinkage.

    shrinkage_strength:
        0.0 = no shrinkage
        1.0 = complete shrinkage to the empirical national target
    """
    required_columns = {
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

    missing = sorted(
        required_columns - set(observations.columns)
    )

    if missing:
        raise ValueError(
            "Swing dataset is missing required columns: "
            + ", ".join(missing)
        )

    if not 0.0 <= shrinkage_strength <= 1.0:
        raise ValueError(
            "shrinkage_strength must be between 0.0 and 1.0."
        )

    if observations.duplicated(
        ["cycle_from", "cycle_to", "race_id"]
    ).any():
        raise ValueError(
            "Duplicate transition/district observations found."
        )

    metadata = (
        observations[
            ["race_id", "state", "district"]
        ]
        .drop_duplicates()
        .copy()
    )

    if metadata["race_id"].duplicated().any():
        raise ValueError(
            "A race_id maps to multiple state/district combinations."
        )

    estimate_rows: list[dict[str, object]] = []

    for race_id, group in observations.groupby(
        "race_id",
        sort=True,
    ):
        estimate = estimate_district(group)
        estimate["race_id"] = race_id
        estimate["first_cycle_from"] = int(
            group["cycle_from"].min()
        )
        estimate["last_cycle_to"] = int(
            group["cycle_to"].max()
        )
        estimate["transition_count_available"] = len(group)
        estimate["transitions_used"] = " | ".join(
            sorted(
                group.loc[
                    group[
                        "eligible_for_elasticity_estimation"
                    ].eq(True),
                    "transition",
                ].astype(str)
            )
        )
        estimate_rows.append(estimate)

    estimates = pd.DataFrame(estimate_rows)

    table = metadata.merge(
        estimates,
        on="race_id",
        how="left",
        validate="one_to_one",
    )

    valid_raw = table["raw_elasticity"].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if valid_raw.empty:
        raise RuntimeError(
            "No valid district elasticity estimates were produced."
        )

    # Weight the empirical shrinkage target by the amount of national
    # swing information available to each district. Districts observed
    # during larger national swings contain more slope information.
    target_weights = pd.to_numeric(
        table["national_swing_sum_squares"],
        errors="coerce",
    ).fillna(0.0)

    target_mask = (
        table["raw_elasticity"].notna()
        & target_weights.gt(0)
    )

    shrink_target = float(
        np.average(
            table.loc[target_mask, "raw_elasticity"],
            weights=target_weights.loc[target_mask],
        )
    )

    table["shrink_target"] = shrink_target
    table["shrinkage_strength"] = shrinkage_strength
    table["raw_deviation_from_target"] = (
        table["raw_elasticity"] - shrink_target
    )

    table["shrunk_elasticity"] = (
        shrink_target
        + (1.0 - shrinkage_strength)
        * table["raw_deviation_from_target"]
    )

    table["shrinkage_adjustment"] = (
        table["shrunk_elasticity"]
        - table["raw_elasticity"]
    )

    table["has_elasticity_estimate"] = (
        table["raw_elasticity"].notna()
    )

    table["low_information_estimate"] = (
        table["observation_count"].fillna(0).lt(3)
    )

    table["estimation_method"] = (
        "district OLS through origin; "
        "district_swing_dem ~ elasticity * national_swing_dem"
    )

    output_columns = [
        "race_id",
        "state",
        "district",
        "observation_count",
        "transition_count_available",
        "first_cycle_from",
        "last_cycle_to",
        "transitions_used",
        "raw_elasticity",
        "shrink_target",
        "shrinkage_strength",
        "shrunk_elasticity",
        "raw_deviation_from_target",
        "shrinkage_adjustment",
        "district_swing_mean",
        "district_swing_sd",
        "residual_mean",
        "residual_rmse",
        "residual_sd",
        "national_swing_sum_squares",
        "has_elasticity_estimate",
        "low_information_estimate",
        "estimation_method",
    ]

    table = table[output_columns].sort_values(
        ["state", "district"],
        key=lambda series: series.map(
            lambda value: (
                0
                if str(value) == "AL"
                else int(value)
                if str(value).isdigit()
                else 999
            )
        )
        if series.name == "district"
        else series,
    ).reset_index(drop=True)

    duplicate_districts = int(
        table["race_id"].duplicated().sum()
    )
    missing_raw = int(
        table["raw_elasticity"].isna().sum()
    )
    nonfinite_raw = int(
        (
            table["raw_elasticity"].notna()
            & ~np.isfinite(table["raw_elasticity"])
        ).sum()
    )
    nonfinite_shrunk = int(
        (
            table["shrunk_elasticity"].notna()
            & ~np.isfinite(table["shrunk_elasticity"])
        ).sum()
    )

    failures: list[str] = []

    if duplicate_districts:
        failures.append(
            f"Found {duplicate_districts} duplicate district rows."
        )

    if nonfinite_raw:
        failures.append(
            f"Found {nonfinite_raw} nonfinite raw elasticities."
        )

    if nonfinite_shrunk:
        failures.append(
            f"Found {nonfinite_shrunk} nonfinite shrunk elasticities."
        )

    if len(table) != observations["race_id"].nunique():
        failures.append(
            "Elasticity row count does not match unique districts "
            "in the swing dataset."
        )

    raw_quantiles = table["raw_elasticity"].quantile(
        [0.00, 0.05, 0.25, 0.50, 0.75, 0.95, 1.00]
    )

    shrunk_quantiles = table["shrunk_elasticity"].quantile(
        [0.00, 0.05, 0.25, 0.50, 0.75, 0.95, 1.00]
    )

    report_lines = [
        "House District Elasticity Validation",
        "=" * 36,
        "",
        f"District rows: {len(table)}",
        (
            "Districts with estimates: "
            f"{int(table['has_elasticity_estimate'].sum())}"
        ),
        f"Districts missing estimates: {missing_raw}",
        (
            "Districts with fewer than 3 observations: "
            f"{int(table['low_information_estimate'].sum())}"
        ),
        f"Duplicate district rows: {duplicate_districts}",
        f"Nonfinite raw elasticities: {nonfinite_raw}",
        f"Nonfinite shrunk elasticities: {nonfinite_shrunk}",
        "",
        "Estimator:",
        (
            "Constrained OLS through the origin using consecutive "
            "district and national two-party swings."
        ),
        "",
        f"Empirical shrinkage target: {shrink_target:.6f}",
        f"Shrinkage strength: {shrinkage_strength:.4f}",
        "",
        "Observation counts:",
        table["observation_count"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string(),
        "",
        "Raw elasticity quantiles:",
        raw_quantiles.to_string(
            float_format=lambda value: f"{value:.6f}"
        ),
        "",
        "Shrunk elasticity quantiles:",
        shrunk_quantiles.to_string(
            float_format=lambda value: f"{value:.6f}"
        ),
        "",
        "Largest raw elasticities:",
        table.nlargest(
            10,
            "raw_elasticity",
        )[
            [
                "race_id",
                "observation_count",
                "raw_elasticity",
                "shrunk_elasticity",
                "residual_rmse",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        ),
        "",
        "Smallest raw elasticities:",
        table.nsmallest(
            10,
            "raw_elasticity",
        )[
            [
                "race_id",
                "observation_count",
                "raw_elasticity",
                "shrunk_elasticity",
                "residual_rmse",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        ),
        "",
        "Validation status:",
    ]

    if failures:
        report_lines.append("FAILED")
        report_lines.extend(
            f"- {failure}"
            for failure in failures
        )
    else:
        report_lines.append("PASSED")

    report = "\n".join(report_lines)

    if failures:
        raise RuntimeError(report)

    return table, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate historical district-level House elasticities."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
    )
    parser.add_argument(
        "--max-cycle-to",
        type=int,
        default=None,
        help=(
            "Optional leakage-safe training cutoff. When supplied, "
            "use only swing transitions whose cycle_to is less than "
            "or equal to this value. Omitting the option preserves "
            "the current all-transition production behavior."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument(
        "--validation-path",
        type=Path,
        default=DEFAULT_VALIDATION_PATH,
    )
    parser.add_argument(
        "--shrinkage-strength",
        type=float,
        default=0.50,
        help=(
            "Share of each district's raw deviation pulled toward "
            "the empirical target. Range: 0.0 to 1.0."
        ),
    )
    args = parser.parse_args()

    if not args.input_path.exists():
        raise FileNotFoundError(
            f"Missing swing dataset: {args.input_path}"
        )

    observations = pd.read_csv(
        args.input_path,
        dtype={"race_id": str},
    )

    if args.max_cycle_to is not None:
        cycle_to = pd.to_numeric(
            observations["cycle_to"],
            errors="coerce",
        )

        if cycle_to.isna().any():
            raise ValueError(
                "Swing observations contain missing or nonnumeric "
                "cycle_to values."
            )

        observations = observations.loc[
            cycle_to.le(args.max_cycle_to)
        ].copy()

        if observations.empty:
            raise ValueError(
                "No swing observations remain after applying "
                f"--max-cycle-to {args.max_cycle_to}."
            )

        retained_cycle_to = pd.to_numeric(
            observations["cycle_to"],
            errors="coerce",
        )

        if retained_cycle_to.gt(
            args.max_cycle_to
        ).any():
            raise RuntimeError(
                "Training cutoff failed: observations after the "
                "requested maximum cycle remain."
            )

        print(
            "Applied elasticity training cutoff: "
            f"cycle_to <= {args.max_cycle_to}"
        )
        print(
            "Retained swing observations: "
            f"{len(observations)}"
        )
        print()

    table, report = build_elasticity_table(
        observations=observations,
        shrinkage_strength=args.shrinkage_strength,
    )

    args.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.validation_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table.to_csv(
        args.output_path,
        index=False,
    )
    args.validation_path.write_text(report)

    print(report)
    print()
    print(f"Wrote: {args.output_path}")
    print(f"Wrote: {args.validation_path}")


if __name__ == "__main__":
    main()
