from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MASTER_PATH = (
    PROJECT_ROOT
    / "historical/house/warehouse/"
    "house_district_master_features.csv"
)

DEFAULT_CHARACTERISTICS_PREDICTION_PATH = (
    PROJECT_ROOT
    / "historical/house/elasticity/"
    "house_characteristics_predicted_elasticity.csv"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "historical/house/elasticity/"
    "house_district_dna_predicted_elasticity.csv"
)

DEFAULT_COEFFICIENT_PATH = (
    PROJECT_ROOT
    / "historical/house/elasticity/"
    "house_district_dna_elasticity_coefficients.csv"
)

DEFAULT_VALIDATION_PATH = (
    PROJECT_ROOT
    / "historical/house/elasticity/"
    "house_district_dna_elasticity_validation.txt"
)


MODEL_ALPHA = 1000.0

CATEGORICAL_FEATURES = [
    "region",
    "district_type",
    "selected_regime_reliability",
]

NUMERIC_FEATURES = [
    "pres_2020_margin_dem",
    "selected_regime_scorable_elections",
    "selected_regime_margin_std",
    "selected_regime_margin_mad",
    "selected_regime_mean_absolute_swing",
    "selected_regime_swing_std",
    "selected_regime_largest_absolute_swing",
    "selected_regime_party_switch_count",
    "selected_regime_competitive_within_5_rate",
    "selected_regime_competitive_within_10_rate",
    "selected_regime_competitive_within_15_rate",
    "selected_regime_trend_slope_points_per_election",
    "selected_regime_trend_residual_rmse",
    "behavior_reliability_score",
]


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def clean_category(series: pd.Series) -> pd.Series:
    return (
        series.fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )


def make_pipeline(alpha: float) -> Pipeline:
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Unknown",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=False,
                ),
            ),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                Ridge(
                    alpha=alpha,
                    fit_intercept=True,
                ),
            ),
        ]
    )


def find_characteristics_prediction_column(
    frame: pd.DataFrame,
) -> str:
    candidates = [
        "characteristics_elasticity",
        "predicted_elasticity_normalized",
        "normalized_predicted_elasticity",
        "predicted_elasticity",
    ]

    for column in candidates:
        if column in frame.columns:
            return column

    raise ValueError(
        "Could not identify the characteristics-prediction column. "
        f"Available columns: {frame.columns.tolist()}"
    )


def build_predictions(
    master: pd.DataFrame,
    characteristics: pd.DataFrame,
    lower_bound: float,
    upper_bound: float,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    required_master = {
        "race_id",
        "state",
        "district",
        "historical_raw_elasticity",
        "historical_elasticity_information",
        "historical_elasticity_available",
        "behavior_multi_cycle_available",
        *CATEGORICAL_FEATURES,
        *NUMERIC_FEATURES,
    }

    missing_master = sorted(
        required_master - set(master.columns)
    )

    if missing_master:
        raise ValueError(
            "Master feature table is missing columns: "
            + ", ".join(missing_master)
        )

    if len(master) != 435:
        raise ValueError(
            f"Expected 435 master rows; found {len(master)}."
        )

    if master["race_id"].duplicated().any():
        raise ValueError(
            "Duplicate race IDs found in master feature table."
        )

    if characteristics["race_id"].duplicated().any():
        raise ValueError(
            "Duplicate race IDs found in characteristics predictions."
        )

    work = master.copy()

    for column in CATEGORICAL_FEATURES:
        work[column] = clean_category(
            work[column]
        )

    for column in [
        *NUMERIC_FEATURES,
        "historical_raw_elasticity",
        "historical_elasticity_information",
    ]:
        work[column] = pd.to_numeric(
            work[column],
            errors="coerce",
        )

    work["historical_elasticity_available"] = parse_bool(
        work["historical_elasticity_available"]
    )

    work["behavior_multi_cycle_available"] = parse_bool(
        work["behavior_multi_cycle_available"]
    )

    training_mask = (
        work["historical_elasticity_available"]
        & work["behavior_multi_cycle_available"]
        & work["historical_raw_elasticity"].notna()
        & work["historical_elasticity_information"].notna()
        & work["historical_elasticity_information"].gt(0)
    )

    training = work.loc[
        training_mask
    ].copy()

    if len(training) < 350:
        raise RuntimeError(
            "Unexpectedly small District DNA training sample: "
            f"{len(training)} rows."
        )

    median_information = float(
        training[
            "historical_elasticity_information"
        ].median()
    )

    training_weights = (
        training[
            "historical_elasticity_information"
        ]
        .clip(
            lower=0.25 * median_information,
            upper=2.00 * median_information,
        )
        .to_numpy(dtype=float)
    )

    pipeline = make_pipeline(
        MODEL_ALPHA
    )

    feature_columns = [
        *CATEGORICAL_FEATURES,
        *NUMERIC_FEATURES,
    ]

    pipeline.fit(
        training[feature_columns],
        training[
            "historical_raw_elasticity"
        ].to_numpy(dtype=float),
        model__sample_weight=training_weights,
    )

    work["district_dna_raw_prediction"] = (
        pipeline.predict(
            work[feature_columns]
        )
    )

    characteristics_column = (
        find_characteristics_prediction_column(
            characteristics
        )
    )

    characteristics_keep = characteristics[
        [
            "race_id",
            characteristics_column,
        ]
    ].copy()

    characteristics_keep = characteristics_keep.rename(
        columns={
            characteristics_column:
                "characteristics_elasticity_fallback"
        }
    )

    characteristics_keep[
        "characteristics_elasticity_fallback"
    ] = pd.to_numeric(
        characteristics_keep[
            "characteristics_elasticity_fallback"
        ],
        errors="coerce",
    )

    work = work.merge(
        characteristics_keep,
        on="race_id",
        how="left",
        validate="one_to_one",
    )

    use_dna = (
        work["behavior_multi_cycle_available"]
        & work[
            "district_dna_raw_prediction"
        ].notna()
    )

    use_characteristics = (
        ~use_dna
        & work[
            "characteristics_elasticity_fallback"
        ].notna()
    )

    work["elasticity_source"] = np.select(
        [
            use_dna,
            use_characteristics,
        ],
        [
            "district_dna_behavior_model",
            "characteristics_model_sparse_history_fallback",
        ],
        default="national_mean_fallback",
    )

    work["district_dna_candidate_before_normalization"] = np.select(
        [
            use_dna,
            use_characteristics,
        ],
        [
            work["district_dna_raw_prediction"],
            work["characteristics_elasticity_fallback"],
        ],
        default=1.0,
    ).astype(float)

    pre_normalization_mean = float(
        work[
            "district_dna_candidate_before_normalization"
        ].mean()
    )

    if not np.isfinite(pre_normalization_mean):
        raise RuntimeError(
            "Candidate elasticity mean is nonfinite."
        )

    if pre_normalization_mean <= 0:
        raise RuntimeError(
            "Candidate elasticity mean must be positive."
        )

    work["district_dna_elasticity_normalized"] = (
        work[
            "district_dna_candidate_before_normalization"
        ]
        / pre_normalization_mean
    )

    work["district_dna_elasticity_bounded"] = (
        work[
            "district_dna_elasticity_normalized"
        ]
        .clip(
            lower=lower_bound,
            upper=upper_bound,
        )
    )

    bounded_mean = float(
        work[
            "district_dna_elasticity_bounded"
        ].mean()
    )

    work["district_dna_elasticity_bounded_normalized"] = (
        work[
            "district_dna_elasticity_bounded"
        ]
        / bounded_mean
    )

    work["district_dna_model_alpha"] = MODEL_ALPHA

    work["district_dna_training_sample_match"] = (
        training_mask
    )

    work["district_dna_prediction_version"] = "1.0"

    work["district_dna_prediction_notes"] = (
        "Behavior-only ridge model selected by fixed-fold screening. "
        "Sparse-history districts use characteristics-model fallback. "
        "Final values are normalized to a national mean of 1.0. "
        "Production adoption remains contingent on the 2022 election "
        "backtest."
    )

    output_columns = [
        "race_id",
        "state",
        "district",
        "selected_boundary_regime_id",
        "selected_regime_reliability",
        "selected_regime_scorable_elections",
        "behavior_reliability_score",
        "district_dna_training_sample_match",
        "district_dna_raw_prediction",
        "characteristics_elasticity_fallback",
        "elasticity_source",
        "district_dna_candidate_before_normalization",
        "district_dna_elasticity_normalized",
        "district_dna_elasticity_bounded",
        "district_dna_elasticity_bounded_normalized",
        "district_dna_model_alpha",
        "district_dna_prediction_version",
        "district_dna_prediction_notes",
    ]

    output = work[
        output_columns
    ].copy()

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    ridge = pipeline.named_steps[
        "model"
    ]

    coefficients = pd.DataFrame(
        {
            "feature": (
                preprocessor.get_feature_names_out()
            ),
            "coefficient": ridge.coef_,
        }
    )

    coefficients["absolute_coefficient"] = (
        coefficients["coefficient"].abs()
    )

    intercept = pd.DataFrame(
        {
            "feature": ["<INTERCEPT>"],
            "coefficient": [
                float(ridge.intercept_)
            ],
            "absolute_coefficient": [
                abs(float(ridge.intercept_))
            ],
        }
    )

    coefficients = pd.concat(
        [
            intercept,
            coefficients.sort_values(
                "absolute_coefficient",
                ascending=False,
            ),
        ],
        ignore_index=True,
    )

    source_counts = (
        output["elasticity_source"]
        .value_counts()
    )

    failures: list[str] = []

    if len(output) != 435:
        failures.append(
            f"Expected 435 prediction rows; found {len(output)}."
        )

    if output["race_id"].nunique() != 435:
        failures.append(
            "Expected 435 unique race IDs."
        )

    if output[
        "district_dna_elasticity_bounded_normalized"
    ].isna().any():
        failures.append(
            "Final candidate elasticity contains missing values."
        )

    if not np.isclose(
        output[
            "district_dna_elasticity_bounded_normalized"
        ].mean(),
        1.0,
        atol=1e-12,
    ):
        failures.append(
            "Final candidate elasticity does not average exactly 1.0."
        )

    if (
        output[
            "district_dna_elasticity_bounded"
        ].lt(lower_bound).any()
        or output[
            "district_dna_elasticity_bounded"
        ].gt(upper_bound).any()
    ):
        failures.append(
            "Bounded candidate values fall outside configured limits."
        )

    report_lines = [
        "House District DNA Elasticity Candidate Validation",
        "=" * 50,
        "",
        f"Master rows: {len(master)}",
        f"Training rows: {len(training)}",
        f"Prediction rows: {len(output)}",
        f"Model alpha: {MODEL_ALPHA:.4f}",
        "",
        "Prediction sources:",
        source_counts.to_string(),
        "",
        (
            "Mean before normalization: "
            f"{pre_normalization_mean:.6f}"
        ),
        (
            "Mean after initial normalization: "
            f"{output['district_dna_elasticity_normalized'].mean():.6f}"
        ),
        (
            "Mean after bounding and renormalization: "
            f"{output['district_dna_elasticity_bounded_normalized'].mean():.6f}"
        ),
        f"Configured lower bound: {lower_bound:.4f}",
        f"Configured upper bound: {upper_bound:.4f}",
        "",
        "Final candidate summary:",
        output[
            "district_dna_elasticity_bounded_normalized"
        ]
        .describe()
        .to_string(
            float_format=lambda value: f"{value:.6f}"
        ),
        "",
        "Important methodological caution:",
        (
            "Behavior predictors and the historical elasticity target "
            "are derived from overlapping elections. The strong screening "
            "result may therefore overstate transferable predictive value. "
            "The candidate must improve the downstream 2022 election "
            "backtest before adoption."
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

    report = "\n".join(
        report_lines
    )

    if failures:
        raise RuntimeError(
            report
        )

    return output, coefficients, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the winning behavior-only District DNA elasticity "
            "candidate and generate predictions for all 435 districts."
        )
    )

    parser.add_argument(
        "--master-path",
        type=Path,
        default=DEFAULT_MASTER_PATH,
    )

    parser.add_argument(
        "--characteristics-prediction-path",
        type=Path,
        default=DEFAULT_CHARACTERISTICS_PREDICTION_PATH,
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )

    parser.add_argument(
        "--coefficient-path",
        type=Path,
        default=DEFAULT_COEFFICIENT_PATH,
    )

    parser.add_argument(
        "--validation-path",
        type=Path,
        default=DEFAULT_VALIDATION_PATH,
    )

    parser.add_argument(
        "--lower-bound",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--upper-bound",
        type=float,
        default=1.75,
    )

    args = parser.parse_args()

    if args.lower_bound <= 0:
        raise ValueError(
            "lower-bound must be positive."
        )

    if args.upper_bound <= args.lower_bound:
        raise ValueError(
            "upper-bound must exceed lower-bound."
        )

    for path in [
        args.master_path,
        args.characteristics_prediction_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required input: {path}"
            )

    master = pd.read_csv(
        args.master_path,
        dtype={
            "race_id": str,
            "state": str,
            "district": str,
        },
    )

    characteristics = pd.read_csv(
        args.characteristics_prediction_path,
        dtype={"race_id": str},
    )

    output, coefficients, report = build_predictions(
        master=master,
        characteristics=characteristics,
        lower_bound=args.lower_bound,
        upper_bound=args.upper_bound,
    )

    args.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        args.output_path,
        index=False,
    )

    coefficients.to_csv(
        args.coefficient_path,
        index=False,
    )

    args.validation_path.write_text(
        report
    )

    print(report)
    print()
    print(f"Wrote: {args.output_path}")
    print(f"Wrote: {args.coefficient_path}")
    print(f"Wrote: {args.validation_path}")


if __name__ == "__main__":
    main()
