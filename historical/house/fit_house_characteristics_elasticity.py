from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TRAINING_PATH = (
    PROJECT_ROOT
    / "historical/house/elasticity/"
    "house_characteristics_elasticity_training.csv"
)

DEFAULT_CHARACTERISTICS_PATH = (
    PROJECT_ROOT
    / "historical/house/warehouse/"
    "house_district_characteristics.csv"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "historical/house/elasticity/"
    "house_characteristics_predicted_elasticity.csv"
)

DEFAULT_COEFFICIENT_PATH = (
    PROJECT_ROOT
    / "historical/house/elasticity/"
    "house_characteristics_elasticity_coefficients.csv"
)

DEFAULT_CV_PATH = (
    PROJECT_ROOT
    / "historical/house/elasticity/"
    "house_characteristics_elasticity_cv.csv"
)

DEFAULT_VALIDATION_PATH = (
    PROJECT_ROOT
    / "historical/house/elasticity/"
    "house_characteristics_elasticity_model_validation.txt"
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

DEFAULT_ALPHAS = (
    0.01,
    0.03,
    0.10,
    0.30,
    1.00,
    3.00,
    10.00,
    30.00,
    100.00,
    300.00,
)


def clean_categories(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()

    for column in CATEGORICAL_FEATURES:
        output[column] = (
            output[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )

    return output


def make_pipeline(alpha: float) -> Pipeline:
    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Unknown",
                ),
            ),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                    sparse_output=False,
                ),
            ),
        ]
    )

    numeric_transformer = Pipeline(
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
                categorical_transformer,
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                numeric_transformer,
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


def evaluate_alpha(
    modeling: pd.DataFrame,
    alpha: float,
    folds: int,
    random_seed: int,
) -> dict[str, float]:
    features = modeling[
        CATEGORICAL_FEATURES + NUMERIC_FEATURES
    ]

    target = modeling["raw_elasticity"].to_numpy(dtype=float)

    weights = modeling[
        "bounded_model_sample_weight"
    ].to_numpy(dtype=float)

    splitter = KFold(
        n_splits=folds,
        shuffle=True,
        random_state=random_seed,
    )

    predictions = np.full(
        len(modeling),
        np.nan,
        dtype=float,
    )

    for train_index, validation_index in splitter.split(features):
        pipeline = make_pipeline(alpha)

        pipeline.fit(
            features.iloc[train_index],
            target[train_index],
            model__sample_weight=weights[train_index],
        )

        predictions[validation_index] = pipeline.predict(
            features.iloc[validation_index]
        )

    if np.isnan(predictions).any():
        raise RuntimeError(
            f"Missing cross-validation predictions for alpha={alpha}."
        )

    errors = predictions - target

    weighted_mae = float(
        np.average(
            np.abs(errors),
            weights=weights,
        )
    )

    weighted_rmse = float(
        np.sqrt(
            np.average(
                np.square(errors),
                weights=weights,
            )
        )
    )

    return {
        "alpha": alpha,
        "cv_mae": float(
            mean_absolute_error(
                target,
                predictions,
            )
        ),
        "cv_rmse": float(
            np.sqrt(
                mean_squared_error(
                    target,
                    predictions,
                )
            )
        ),
        "weighted_cv_mae": weighted_mae,
        "weighted_cv_rmse": weighted_rmse,
        "prediction_mean": float(
            np.mean(predictions)
        ),
        "prediction_sd": float(
            np.std(predictions, ddof=0)
        ),
        "prediction_min": float(
            np.min(predictions)
        ),
        "prediction_max": float(
            np.max(predictions)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit an interpretable ridge model predicting House "
            "district elasticity from district characteristics."
        )
    )

    parser.add_argument(
        "--training-path",
        type=Path,
        default=DEFAULT_TRAINING_PATH,
    )

    parser.add_argument(
        "--characteristics-path",
        type=Path,
        default=DEFAULT_CHARACTERISTICS_PATH,
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
        "--cv-path",
        type=Path,
        default=DEFAULT_CV_PATH,
    )

    parser.add_argument(
        "--validation-path",
        type=Path,
        default=DEFAULT_VALIDATION_PATH,
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=20260715,
    )

    args = parser.parse_args()

    if args.folds < 2:
        raise ValueError("folds must be at least 2.")

    if not args.training_path.exists():
        raise FileNotFoundError(
            f"Missing training data: {args.training_path}"
        )

    if not args.characteristics_path.exists():
        raise FileNotFoundError(
            f"Missing characteristics data: {args.characteristics_path}"
        )

    training = pd.read_csv(
        args.training_path,
        dtype={"race_id": str},
    )

    characteristics = pd.read_csv(
        args.characteristics_path,
        dtype={"race_id": str},
    )

    training = clean_categories(training)
    characteristics = clean_categories(characteristics)

    modeling = training.loc[
        training[
            "eligible_for_characteristics_model"
        ].eq(True)
    ].copy()

    required_training_columns = {
        "race_id",
        "raw_elasticity",
        "bounded_model_sample_weight",
        *CATEGORICAL_FEATURES,
        *NUMERIC_FEATURES,
    }

    missing_training = sorted(
        required_training_columns - set(modeling.columns)
    )

    if missing_training:
        raise ValueError(
            "Training data is missing required columns: "
            + ", ".join(missing_training)
        )

    required_characteristic_columns = {
        "race_id",
        "state",
        "district",
        *CATEGORICAL_FEATURES,
        *NUMERIC_FEATURES,
    }

    missing_characteristics = sorted(
        required_characteristic_columns
        - set(characteristics.columns)
    )

    if missing_characteristics:
        raise ValueError(
            "Characteristics data is missing required columns: "
            + ", ".join(missing_characteristics)
        )

    if len(modeling) < args.folds:
        raise ValueError(
            "Fewer eligible modeling rows than cross-validation folds."
        )

    cv_rows = [
        evaluate_alpha(
            modeling=modeling,
            alpha=alpha,
            folds=args.folds,
            random_seed=args.random_seed,
        )
        for alpha in DEFAULT_ALPHAS
    ]

    cv = pd.DataFrame(cv_rows)

    cv["weighted_rmse_rank"] = cv[
        "weighted_cv_rmse"
    ].rank(method="min")

    cv["weighted_mae_rank"] = cv[
        "weighted_cv_mae"
    ].rank(method="min")

    cv["combined_rank"] = (
        cv["weighted_rmse_rank"]
        + cv["weighted_mae_rank"]
    )

    cv = cv.sort_values(
        [
            "combined_rank",
            "weighted_cv_rmse",
            "alpha",
        ]
    ).reset_index(drop=True)

    best_alpha = float(
        cv.iloc[0]["alpha"]
    )

    final_pipeline = make_pipeline(best_alpha)

    features = modeling[
        CATEGORICAL_FEATURES + NUMERIC_FEATURES
    ]

    target = modeling[
        "raw_elasticity"
    ].to_numpy(dtype=float)

    weights = modeling[
        "bounded_model_sample_weight"
    ].to_numpy(dtype=float)

    final_pipeline.fit(
        features,
        target,
        model__sample_weight=weights,
    )

    all_features = characteristics[
        CATEGORICAL_FEATURES + NUMERIC_FEATURES
    ]

    predictions = final_pipeline.predict(all_features)

    prediction_mean = float(
        np.mean(predictions)
    )

    if not np.isfinite(prediction_mean) or prediction_mean == 0:
        raise RuntimeError(
            "Predicted elasticity mean must be finite and nonzero."
        )

    normalized_predictions = predictions / prediction_mean

    output = characteristics[
        [
            "race_id",
            "state",
            "district",
            *CATEGORICAL_FEATURES,
            *NUMERIC_FEATURES,
        ]
    ].copy()

    output[
        "characteristics_elasticity_raw_prediction"
    ] = predictions

    output["characteristics_elasticity"] = (
        normalized_predictions
    )

    output[
        "characteristics_elasticity_model_alpha"
    ] = best_alpha

    output[
        "characteristics_elasticity_prediction_mean_before_normalization"
    ] = prediction_mean

    output["characteristics_model_training_match"] = (
        output["race_id"].isin(
            set(modeling["race_id"])
        )
    )

    output["characteristics_elasticity_method"] = (
        "weighted ridge regression with one-hot categorical "
        "district characteristics and standardized 2020 "
        "presidential margin"
    )

    output["characteristics_elasticity_limitations"] = (
        "Exploratory model: post-2020 characteristics predict "
        "2012-2020 district-label elasticity estimates."
    )

    preprocessor = final_pipeline.named_steps[
        "preprocessor"
    ]

    ridge = final_pipeline.named_steps[
        "model"
    ]

    feature_names = preprocessor.get_feature_names_out()

    coefficients = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": ridge.coef_,
            "absolute_coefficient": np.abs(
                ridge.coef_
            ),
        }
    ).sort_values(
        "absolute_coefficient",
        ascending=False,
    ).reset_index(drop=True)

    intercept_row = pd.DataFrame(
        {
            "feature": ["<INTERCEPT>"],
            "coefficient": [float(ridge.intercept_)],
            "absolute_coefficient": [
                abs(float(ridge.intercept_))
            ],
        }
    )

    coefficients = pd.concat(
        [intercept_row, coefficients],
        ignore_index=True,
    )

    failures: list[str] = []

    if len(output) != 435:
        failures.append(
            f"Expected 435 predictions; found {len(output)}."
        )

    if output["race_id"].duplicated().any():
        failures.append(
            "Duplicate race IDs found in prediction output."
        )

    missing_predictions = int(
        output["characteristics_elasticity"].isna().sum()
    )

    nonfinite_predictions = int(
        (
            output["characteristics_elasticity"].notna()
            & ~np.isfinite(
                output["characteristics_elasticity"]
            )
        ).sum()
    )

    if missing_predictions:
        failures.append(
            f"Found {missing_predictions} missing predictions."
        )

    if nonfinite_predictions:
        failures.append(
            f"Found {nonfinite_predictions} nonfinite predictions."
        )

    best_cv = cv.iloc[0]

    report_lines = [
        "House Characteristics Elasticity Model Validation",
        "=" * 47,
        "",
        f"Eligible training rows: {len(modeling)}",
        f"Prediction rows: {len(output)}",
        f"Cross-validation folds: {args.folds}",
        f"Random seed: {args.random_seed}",
        f"Selected ridge alpha: {best_alpha:.6f}",
        "",
        "Selected model cross-validation:",
        f"Unweighted MAE: {best_cv['cv_mae']:.6f}",
        f"Unweighted RMSE: {best_cv['cv_rmse']:.6f}",
        f"Weighted MAE: {best_cv['weighted_cv_mae']:.6f}",
        f"Weighted RMSE: {best_cv['weighted_cv_rmse']:.6f}",
        "",
        "Prediction distribution before normalization:",
        pd.Series(predictions)
        .describe()
        .to_string(
            float_format=lambda value: f"{value:.6f}"
        ),
        "",
        "Prediction distribution after normalization:",
        output["characteristics_elasticity"]
        .describe()
        .to_string(
            float_format=lambda value: f"{value:.6f}"
        ),
        "",
        (
            "Training-label matches among predicted districts: "
            f"{int(output['characteristics_model_training_match'].sum())}"
        ),
        (
            "Districts predicted without a historical target match: "
            f"{int((~output['characteristics_model_training_match']).sum())}"
        ),
        "",
        "Largest absolute coefficients:",
        coefficients.head(25).to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        ),
        "",
        "Important limitation:",
        (
            "Predictors describe post-2020 districts, while elasticity "
            "targets were estimated from 2012-2020 district-label "
            "histories."
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

    for path in [
        args.output_path,
        args.coefficient_path,
        args.cv_path,
        args.validation_path,
    ]:
        path.parent.mkdir(
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

    cv.to_csv(
        args.cv_path,
        index=False,
    )

    args.validation_path.write_text(report)

    print(report)
    print()
    print(f"Wrote: {args.output_path}")
    print(f"Wrote: {args.coefficient_path}")
    print(f"Wrote: {args.cv_path}")
    print(f"Wrote: {args.validation_path}")


if __name__ == "__main__":
    main()
