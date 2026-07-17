from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CHARACTERISTICS_PATH = (
    PROJECT_ROOT
    / "historical/house/warehouse/"
    "house_district_characteristics.csv"
)

DEFAULT_ELASTICITY_PATH = (
    PROJECT_ROOT
    / "historical/house/elasticity/"
    "house_district_elasticity.csv"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "historical/house/elasticity/"
    "house_characteristics_elasticity_training.csv"
)

DEFAULT_VALIDATION_PATH = (
    PROJECT_ROOT
    / "historical/house/elasticity/"
    "house_characteristics_elasticity_training_validation.txt"
)


CATEGORICAL_FEATURES = [
    "region",
    "district_type",
    "college_share_tier",
    "white_share_tier",
    "black_share_tier",
    "hispanic_share_tier",
]

NUMERIC_FEATURES = [
    "pres_2020_margin_dem",
]


def clean_category(series: pd.Series) -> pd.Series:
    """Normalize categorical values and retain missingness explicitly."""
    cleaned = (
        series.fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    return cleaned.mask(
        cleaned.eq(""),
        "Unknown",
    )


def build_training_dataset(
    characteristics: pd.DataFrame,
    elasticity: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    required_characteristics = {
        "race_id",
        *CATEGORICAL_FEATURES,
        *NUMERIC_FEATURES,
    }

    missing_characteristics = sorted(
        required_characteristics
        - set(characteristics.columns)
    )

    if missing_characteristics:
        raise ValueError(
            "Characteristics table is missing required columns: "
            + ", ".join(missing_characteristics)
        )

    required_elasticity = {
        "race_id",
        "raw_elasticity",
        "observation_count",
        "national_swing_sum_squares",
        "residual_rmse",
        "low_information_estimate",
    }

    missing_elasticity = sorted(
        required_elasticity
        - set(elasticity.columns)
    )

    if missing_elasticity:
        raise ValueError(
            "Elasticity table is missing required columns: "
            + ", ".join(missing_elasticity)
        )

    if len(characteristics) != 435:
        raise ValueError(
            "Expected 435 district-characteristic rows; "
            f"found {len(characteristics)}."
        )

    if characteristics["race_id"].duplicated().any():
        raise ValueError(
            "Duplicate race IDs found in characteristics table."
        )

    if elasticity["race_id"].duplicated().any():
        raise ValueError(
            "Duplicate race IDs found in elasticity table."
        )

    elasticity_keep = elasticity[
        [
            "race_id",
            "raw_elasticity",
            "observation_count",
            "national_swing_sum_squares",
            "residual_rmse",
            "low_information_estimate",
        ]
    ].copy()

    training = characteristics[
        [
            "race_id",
            *CATEGORICAL_FEATURES,
            *NUMERIC_FEATURES,
        ]
    ].merge(
        elasticity_keep,
        on="race_id",
        how="inner",
        validate="one_to_one",
    )

    for column in CATEGORICAL_FEATURES:
        training[column] = clean_category(
            training[column]
        )

    numeric_columns = [
        *NUMERIC_FEATURES,
        "raw_elasticity",
        "observation_count",
        "national_swing_sum_squares",
        "residual_rmse",
    ]

    for column in numeric_columns:
        training[column] = pd.to_numeric(
            training[column],
            errors="coerce",
        )

    training["eligible_for_characteristics_model"] = (
        training["raw_elasticity"].notna()
        & np.isfinite(training["raw_elasticity"])
        & training["pres_2020_margin_dem"].notna()
        & training["observation_count"].gt(0)
        & training["national_swing_sum_squares"].gt(0)
    )

    # The historical slope's information content is proportional to
    # the squared national swings observed for that district.
    training["model_sample_weight"] = (
        training["national_swing_sum_squares"]
    )

    # A bounded alternative is retained for sensitivity testing so that
    # a few districts cannot dominate solely because of coverage.
    median_information = float(
        training.loc[
            training["eligible_for_characteristics_model"],
            "national_swing_sum_squares",
        ].median()
    )

    training["bounded_model_sample_weight"] = (
        training["national_swing_sum_squares"]
        .clip(
            lower=0.25 * median_information,
            upper=2.00 * median_information,
        )
    )

    training["target_definition"] = (
        "2012-2020 district elasticity estimated by constrained OLS "
        "through the origin"
    )

    training["characteristics_boundary_warning"] = (
        "Characteristics describe post-2020 districts; historical "
        "elasticity targets use 2012-2020 district-label matches."
    )

    training = training.sort_values(
        "race_id"
    ).reset_index(drop=True)

    duplicate_rows = int(
        training["race_id"].duplicated().sum()
    )
    eligible_rows = int(
        training[
            "eligible_for_characteristics_model"
        ].sum()
    )
    nonfinite_targets = int(
        (
            training["raw_elasticity"].notna()
            & ~np.isfinite(training["raw_elasticity"])
        ).sum()
    )
    missing_weights = int(
        training["model_sample_weight"].isna().sum()
    )

    failures: list[str] = []

    if duplicate_rows:
        failures.append(
            f"Found {duplicate_rows} duplicate race IDs."
        )

    if nonfinite_targets:
        failures.append(
            f"Found {nonfinite_targets} nonfinite target values."
        )

    if missing_weights:
        failures.append(
            f"Found {missing_weights} missing sample weights."
        )

    if eligible_rows == 0:
        failures.append(
            "No rows are eligible for characteristics modeling."
        )

    category_report: list[str] = []

    for column in CATEGORICAL_FEATURES:
        category_report.extend(
            [
                "",
                f"{column}:",
                training[column]
                .value_counts(dropna=False)
                .to_string(),
            ]
        )

    report_lines = [
        "House Characteristics-Elasticity Training Validation",
        "=" * 52,
        "",
        f"Characteristic rows available: {len(characteristics)}",
        f"Historical elasticity rows available: {len(elasticity)}",
        f"Joined training rows: {len(training)}",
        f"Eligible modeling rows: {eligible_rows}",
        f"Unique race IDs: {training['race_id'].nunique()}",
        f"Duplicate race IDs: {duplicate_rows}",
        f"Nonfinite elasticity targets: {nonfinite_targets}",
        f"Missing model weights: {missing_weights}",
        "",
        "Observation counts:",
        training["observation_count"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string(),
        "",
        "Elasticity target summary:",
        training["raw_elasticity"]
        .describe()
        .to_string(
            float_format=lambda value: f"{value:.6f}"
        ),
        "",
        "Weight summary:",
        training["model_sample_weight"]
        .describe()
        .to_string(
            float_format=lambda value: f"{value:.6f}"
        ),
        "",
        "Categorical feature coverage:",
        *category_report,
        "",
        "Important limitation:",
        (
            "The predictors describe post-2020 congressional districts, "
            "while the historical elasticity targets were estimated from "
            "2012-2020 district-label matches. This training dataset is "
            "appropriate for an exploratory out-of-sample test, not for "
            "claiming exact geographic continuity."
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

    return training, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a reusable House characteristics-elasticity "
            "training dataset."
        )
    )

    parser.add_argument(
        "--characteristics-path",
        type=Path,
        default=DEFAULT_CHARACTERISTICS_PATH,
    )

    parser.add_argument(
        "--elasticity-path",
        type=Path,
        default=DEFAULT_ELASTICITY_PATH,
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

    args = parser.parse_args()

    if not args.characteristics_path.exists():
        raise FileNotFoundError(
            "Missing characteristics warehouse: "
            f"{args.characteristics_path}"
        )

    if not args.elasticity_path.exists():
        raise FileNotFoundError(
            "Missing historical elasticity table: "
            f"{args.elasticity_path}"
        )

    characteristics = pd.read_csv(
        args.characteristics_path,
        dtype={"race_id": str},
    )

    elasticity = pd.read_csv(
        args.elasticity_path,
        dtype={"race_id": str},
    )

    training, report = build_training_dataset(
        characteristics=characteristics,
        elasticity=elasticity,
    )

    args.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.validation_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    training.to_csv(
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
