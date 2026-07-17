from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CHARACTERISTICS_PATH = (
    PROJECT_ROOT
    / "historical/house/warehouse/"
    "house_enriched_district_characteristics.csv"
)

DEFAULT_ELASTICITY_PATH = (
    PROJECT_ROOT
    / "historical/house/elasticity/"
    "house_district_elasticity.csv"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "historical/house/elasticity/"
    "house_continuous_elasticity_training.csv"
)

DEFAULT_VALIDATION_PATH = (
    PROJECT_ROOT
    / "historical/house/elasticity/"
    "house_continuous_elasticity_training_validation.txt"
)


CATEGORICAL_FEATURES = [
    "region",
    "district_type",
    "dra_population_dataset",
]

CONTINUOUS_FEATURES = [
    "pres_2020_margin_dem",
    "pres_2024_margin_dem",
    "presidential_swing_2020_to_2024_dem",
    "dra_composite_margin_dem",
    "dra_composite_minus_2020_margin_dem",
    "dra_composite_minus_2024_margin_dem",
    "white_vap_share",
    "black_vap_share",
    "hispanic_vap_share",
    "asian_vap_share",
    "native_vap_share",
    "pacific_vap_share",
]


def clean_category(series: pd.Series) -> pd.Series:
    return (
        series.fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )


def build_training_dataset(
    characteristics: pd.DataFrame,
    elasticity: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    required_characteristics = {
        "race_id",
        "dra_data_available",
        *CATEGORICAL_FEATURES,
        *CONTINUOUS_FEATURES,
    }

    missing_characteristics = sorted(
        required_characteristics
        - set(characteristics.columns)
    )

    if missing_characteristics:
        raise ValueError(
            "Enriched characteristics table is missing columns: "
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
            "Elasticity table is missing columns: "
            + ", ".join(missing_elasticity)
        )

    if len(characteristics) != 435:
        raise ValueError(
            "Expected 435 enriched characteristic rows; "
            f"found {len(characteristics)}."
        )

    if characteristics["race_id"].duplicated().any():
        raise ValueError(
            "Duplicate race IDs found in enriched characteristics."
        )

    if elasticity["race_id"].duplicated().any():
        raise ValueError(
            "Duplicate race IDs found in historical elasticity table."
        )

    characteristics_keep = characteristics[
        [
            "race_id",
            "dra_data_available",
            *CATEGORICAL_FEATURES,
            *CONTINUOUS_FEATURES,
        ]
    ].copy()

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

    training = characteristics_keep.merge(
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
        *CONTINUOUS_FEATURES,
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

    complete_continuous_features = training[
        CONTINUOUS_FEATURES
    ].notna().all(axis=1)

    finite_continuous_features = pd.Series(
        np.isfinite(
            training[CONTINUOUS_FEATURES]
            .to_numpy(dtype=float)
        ).all(axis=1),
        index=training.index,
    )

    training[
        "eligible_for_continuous_elasticity_model"
    ] = (
        training["dra_data_available"].eq(True)
        & training["raw_elasticity"].notna()
        & np.isfinite(training["raw_elasticity"])
        & training["observation_count"].gt(0)
        & training["national_swing_sum_squares"].gt(0)
        & complete_continuous_features
        & finite_continuous_features
    )

    training["model_sample_weight"] = (
        training["national_swing_sum_squares"]
    )

    eligible_weights = training.loc[
        training[
            "eligible_for_continuous_elasticity_model"
        ],
        "national_swing_sum_squares",
    ]

    if eligible_weights.empty:
        raise RuntimeError(
            "No rows are eligible for the continuous elasticity model."
        )

    median_information = float(
        eligible_weights.median()
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

    training["feature_definition"] = (
        "Continuous post-2020 DRA demographic, VAP, presidential, "
        "and composite-election characteristics"
    )

    training["geographic_limitation"] = (
        "Historical targets use 2012-2020 district-label histories; "
        "predictors describe post-2020 district geography."
    )

    training = training.sort_values(
        "race_id"
    ).reset_index(drop=True)

    eligible_rows = int(
        training[
            "eligible_for_continuous_elasticity_model"
        ].sum()
    )

    duplicate_rows = int(
        training["race_id"].duplicated().sum()
    )

    failures: list[str] = []

    if duplicate_rows:
        failures.append(
            f"Found {duplicate_rows} duplicate race IDs."
        )

    if eligible_rows == 0:
        failures.append(
            "No eligible continuous-feature training rows."
        )

    if len(training) != 398:
        failures.append(
            "Expected 398 joined historical-target rows; "
            f"found {len(training)}."
        )

    missing_feature_report = (
        training[CONTINUOUS_FEATURES]
        .isna()
        .sum()
        .sort_values(ascending=False)
    )

    report_lines = [
        "House Continuous Elasticity Training Validation",
        "=" * 47,
        "",
        f"Enriched characteristic rows available: {len(characteristics)}",
        f"Historical elasticity rows available: {len(elasticity)}",
        f"Joined rows: {len(training)}",
        f"Eligible modeling rows: {eligible_rows}",
        f"Unique race IDs: {training['race_id'].nunique()}",
        f"Duplicate race IDs: {duplicate_rows}",
        "",
        "Ineligible-row reasons:",
        (
            "Missing DRA coverage: "
            f"{int((~training['dra_data_available'].eq(True)).sum())}"
        ),
        (
            "Incomplete continuous features: "
            f"{int((~complete_continuous_features).sum())}"
        ),
        "",
        "Missing values by continuous feature:",
        missing_feature_report.to_string(),
        "",
        "Observation counts among eligible rows:",
        training.loc[
            training[
                "eligible_for_continuous_elasticity_model"
            ],
            "observation_count",
        ]
        .value_counts()
        .sort_index()
        .to_string(),
        "",
        "Elasticity target summary among eligible rows:",
        training.loc[
            training[
                "eligible_for_continuous_elasticity_model"
            ],
            "raw_elasticity",
        ]
        .describe()
        .to_string(
            float_format=lambda value: f"{value:.6f}"
        ),
        "",
        "Continuous feature summary among eligible rows:",
        training.loc[
            training[
                "eligible_for_continuous_elasticity_model"
            ],
            CONTINUOUS_FEATURES,
        ]
        .describe()
        .transpose()
        .to_string(
            float_format=lambda value: f"{value:.4f}"
        ),
        "",
        "Important limitation:",
        (
            "The model will use current district characteristics to "
            "predict elasticity targets derived from older district-label "
            "histories. Results remain exploratory until geographic "
            "crosswalks or multiple election cycles are available."
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
            "Build a House elasticity training dataset using "
            "continuous DRA district characteristics."
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
            "Missing enriched characteristics table: "
            f"{args.characteristics_path}"
        )

    if not args.elasticity_path.exists():
        raise FileNotFoundError(
            "Missing historical elasticity table: "
            f"{args.elasticity_path}"
        )

    characteristics = pd.read_csv(
        args.characteristics_path,
        dtype={
            "race_id": str,
            "state": str,
            "district": str,
        },
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

    args.validation_path.write_text(
        report
    )

    print(report)
    print()
    print(f"Wrote: {args.output_path}")
    print(f"Wrote: {args.validation_path}")


if __name__ == "__main__":
    main()
