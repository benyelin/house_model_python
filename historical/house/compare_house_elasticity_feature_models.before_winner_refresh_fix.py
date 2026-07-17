from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TRAINING_PATH = (
    PROJECT_ROOT
    / "historical/house/elasticity/"
    "house_continuous_elasticity_training.csv"
)

DEFAULT_CHARACTERISTICS_PATH = (
    PROJECT_ROOT
    / "historical/house/warehouse/"
    "house_enriched_district_characteristics.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical/house/elasticity/model_comparison"
)

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
    1000.00,
)


@dataclass(frozen=True)
class ModelSpecification:
    model_id: str
    description: str
    categorical_features: tuple[str, ...]
    numeric_features: tuple[str, ...]


MODEL_SPECIFICATIONS = (
    ModelSpecification(
        model_id="A_legacy_tiers",
        description=(
            "Legacy tiered demographic characteristics, region, "
            "district type, and 2020 presidential margin."
        ),
        categorical_features=(
            "region",
            "district_type",
            "college_share_tier",
            "white_share_tier",
            "black_share_tier",
            "hispanic_share_tier",
        ),
        numeric_features=(
            "pres_2020_margin_dem",
        ),
    ),
    ModelSpecification(
        model_id="B_continuous_demographics",
        description=(
            "Continuous voting-age demographic shares, region, "
            "district type, and 2020 presidential margin."
        ),
        categorical_features=(
            "region",
            "district_type",
            "dra_population_dataset",
        ),
        numeric_features=(
            "pres_2020_margin_dem",
            "white_vap_share",
            "black_vap_share",
            "hispanic_vap_share",
            "asian_vap_share",
            "native_vap_share",
            "pacific_vap_share",
        ),
    ),
    ModelSpecification(
        model_id="C_continuous_full",
        description=(
            "Continuous demographic and political characteristics, "
            "including presidential movement and DRA composite margins."
        ),
        categorical_features=(
            "region",
            "district_type",
            "dra_population_dataset",
        ),
        numeric_features=(
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
        ),
    ),
    ModelSpecification(
        model_id="D_hybrid_all_features",
        description=(
            "All legacy tiered characteristics and all continuous "
            "demographic and political characteristics."
        ),
        categorical_features=(
            "region",
            "district_type",
            "college_share_tier",
            "white_share_tier",
            "black_share_tier",
            "hispanic_share_tier",
            "dra_population_dataset",
        ),
        numeric_features=(
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
        ),
    ),
)


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


def build_modeling_table(
    training: pd.DataFrame,
    characteristics: pd.DataFrame,
) -> pd.DataFrame:
    required_training = {
        "race_id",
        "raw_elasticity",
        "bounded_model_sample_weight",
        "eligible_for_continuous_elasticity_model",
    }

    missing_training = sorted(
        required_training - set(training.columns)
    )

    if missing_training:
        raise ValueError(
            "Training table is missing required columns: "
            + ", ".join(missing_training)
        )

    all_features = {
        feature
        for specification in MODEL_SPECIFICATIONS
        for feature in (
            *specification.categorical_features,
            *specification.numeric_features,
        )
    }

    required_characteristics = {
        "race_id",
        *all_features,
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

    if training["race_id"].duplicated().any():
        raise ValueError(
            "Duplicate race IDs found in training table."
        )

    if characteristics["race_id"].duplicated().any():
        raise ValueError(
            "Duplicate race IDs found in characteristics table."
        )

    targets = training[
        [
            "race_id",
            "raw_elasticity",
            "bounded_model_sample_weight",
            "eligible_for_continuous_elasticity_model",
            "observation_count",
            "residual_rmse",
        ]
    ].copy()

    feature_table = characteristics[
        [
            "race_id",
            *sorted(all_features),
        ]
    ].copy()

    modeling = targets.merge(
        feature_table,
        on="race_id",
        how="left",
        validate="one_to_one",
    )

    eligible = parse_bool(
        modeling[
            "eligible_for_continuous_elasticity_model"
        ]
    )

    modeling = modeling.loc[
        eligible
    ].copy()

    modeling["raw_elasticity"] = pd.to_numeric(
        modeling["raw_elasticity"],
        errors="coerce",
    )

    modeling["bounded_model_sample_weight"] = pd.to_numeric(
        modeling["bounded_model_sample_weight"],
        errors="coerce",
    )

    if modeling["raw_elasticity"].isna().any():
        raise ValueError(
            "Eligible modeling rows contain missing targets."
        )

    if (
        modeling["bounded_model_sample_weight"].isna().any()
        or modeling["bounded_model_sample_weight"].le(0).any()
    ):
        raise ValueError(
            "Eligible modeling rows contain invalid weights."
        )

    for specification in MODEL_SPECIFICATIONS:
        for column in specification.categorical_features:
            modeling[column] = clean_category(
                modeling[column]
            )

        for column in specification.numeric_features:
            modeling[column] = pd.to_numeric(
                modeling[column],
                errors="coerce",
            )

    required_numeric = sorted(
        {
            feature
            for specification in MODEL_SPECIFICATIONS
            for feature in specification.numeric_features
        }
    )

    if modeling[required_numeric].isna().any().any():
        missing = (
            modeling[required_numeric]
            .isna()
            .sum()
        )

        missing = missing.loc[
            missing.gt(0)
        ]

        raise ValueError(
            "Eligible modeling rows contain missing numeric features:\n"
            + missing.to_string()
        )

    if len(modeling) != 393:
        raise ValueError(
            f"Expected 393 common modeling rows; found {len(modeling)}."
        )

    return modeling.sort_values(
        "race_id"
    ).reset_index(drop=True)


def make_pipeline(
    specification: ModelSpecification,
    alpha: float,
) -> Pipeline:
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

    transformers = []

    if specification.categorical_features:
        transformers.append(
            (
                "categorical",
                categorical_transformer,
                list(specification.categorical_features),
            )
        )

    if specification.numeric_features:
        transformers.append(
            (
                "numeric",
                numeric_transformer,
                list(specification.numeric_features),
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
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


def weighted_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    errors = prediction - target

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

    weighted_target_mean = float(
        np.average(
            target,
            weights=weights,
        )
    )

    weighted_sse = float(
        np.sum(
            weights
            * np.square(errors)
        )
    )

    weighted_sst = float(
        np.sum(
            weights
            * np.square(
                target - weighted_target_mean
            )
        )
    )

    weighted_r2 = (
        1.0 - weighted_sse / weighted_sst
        if weighted_sst > 0
        else np.nan
    )

    return {
        "weighted_mae": weighted_mae,
        "weighted_rmse": weighted_rmse,
        "weighted_r2": float(weighted_r2),
    }


def evaluate_specification_alpha(
    modeling: pd.DataFrame,
    specification: ModelSpecification,
    alpha: float,
    fold_assignments: np.ndarray,
) -> tuple[dict[str, object], pd.DataFrame]:
    feature_columns = [
        *specification.categorical_features,
        *specification.numeric_features,
    ]

    features = modeling[
        feature_columns
    ]

    target = modeling[
        "raw_elasticity"
    ].to_numpy(dtype=float)

    weights = modeling[
        "bounded_model_sample_weight"
    ].to_numpy(dtype=float)

    predictions = np.full(
        len(modeling),
        np.nan,
        dtype=float,
    )

    fold_rows: list[dict[str, object]] = []

    for fold in sorted(
        np.unique(fold_assignments)
    ):
        validation_mask = (
            fold_assignments == fold
        )
        training_mask = ~validation_mask

        pipeline = make_pipeline(
            specification=specification,
            alpha=alpha,
        )

        pipeline.fit(
            features.loc[training_mask],
            target[training_mask],
            model__sample_weight=weights[training_mask],
        )

        fold_prediction = pipeline.predict(
            features.loc[validation_mask]
        )

        predictions[validation_mask] = fold_prediction

        fold_target = target[validation_mask]
        fold_weights = weights[validation_mask]

        fold_weighted = weighted_metrics(
            target=fold_target,
            prediction=fold_prediction,
            weights=fold_weights,
        )

        fold_rows.append(
            {
                "model_id": specification.model_id,
                "alpha": alpha,
                "fold": int(fold),
                "validation_rows": int(
                    validation_mask.sum()
                ),
                "fold_mae": float(
                    mean_absolute_error(
                        fold_target,
                        fold_prediction,
                    )
                ),
                "fold_rmse": float(
                    np.sqrt(
                        mean_squared_error(
                            fold_target,
                            fold_prediction,
                        )
                    )
                ),
                "fold_r2": float(
                    r2_score(
                        fold_target,
                        fold_prediction,
                    )
                ),
                **{
                    f"fold_{key}": value
                    for key, value in fold_weighted.items()
                },
            }
        )

    if np.isnan(predictions).any():
        raise RuntimeError(
            f"Missing predictions for {specification.model_id}, "
            f"alpha={alpha}."
        )

    weighted = weighted_metrics(
        target=target,
        prediction=predictions,
        weights=weights,
    )

    correlation = float(
        np.corrcoef(
            target,
            predictions,
        )[0, 1]
    )

    result = {
        "model_id": specification.model_id,
        "model_description": specification.description,
        "categorical_feature_count": len(
            specification.categorical_features
        ),
        "numeric_feature_count": len(
            specification.numeric_features
        ),
        "alpha": alpha,
        "cv_rows": len(modeling),
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
        "cv_r2": float(
            r2_score(
                target,
                predictions,
            )
        ),
        "cv_correlation": correlation,
        **weighted,
        "prediction_mean": float(
            np.mean(predictions)
        ),
        "prediction_sd": float(
            np.std(
                predictions,
                ddof=0,
            )
        ),
        "prediction_min": float(
            np.min(predictions)
        ),
        "prediction_max": float(
            np.max(predictions)
        ),
    }

    prediction_table = pd.DataFrame(
        {
            "model_id": specification.model_id,
            "alpha": alpha,
            "race_id": modeling["race_id"],
            "fold": fold_assignments,
            "actual_raw_elasticity": target,
            "predicted_raw_elasticity": predictions,
            "prediction_error": predictions - target,
            "absolute_prediction_error": np.abs(
                predictions - target
            ),
            "bounded_model_sample_weight": weights,
        }
    )

    return result, pd.DataFrame(fold_rows), prediction_table


def fit_final_model(
    modeling: pd.DataFrame,
    specification: ModelSpecification,
    alpha: float,
) -> tuple[Pipeline, pd.DataFrame]:
    feature_columns = [
        *specification.categorical_features,
        *specification.numeric_features,
    ]

    pipeline = make_pipeline(
        specification=specification,
        alpha=alpha,
    )

    pipeline.fit(
        modeling[feature_columns],
        modeling["raw_elasticity"].to_numpy(dtype=float),
        model__sample_weight=(
            modeling[
                "bounded_model_sample_weight"
            ].to_numpy(dtype=float)
        ),
    )

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    ridge = pipeline.named_steps[
        "model"
    ]

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    coefficients = pd.DataFrame(
        {
            "model_id": specification.model_id,
            "alpha": alpha,
            "feature": feature_names,
            "coefficient": ridge.coef_,
            "absolute_coefficient": np.abs(
                ridge.coef_
            ),
        }
    ).sort_values(
        "absolute_coefficient",
        ascending=False,
    )

    intercept = pd.DataFrame(
        {
            "model_id": [specification.model_id],
            "alpha": [alpha],
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
            coefficients,
        ],
        ignore_index=True,
    )

    return pipeline, coefficients


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare tiered, continuous, and hybrid ridge models "
            "for House district elasticity."
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
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=20260716,
    )

    args = parser.parse_args()

    if args.folds < 2:
        raise ValueError(
            "folds must be at least 2."
        )

    if not args.training_path.exists():
        raise FileNotFoundError(
            f"Missing training file: {args.training_path}"
        )

    if not args.characteristics_path.exists():
        raise FileNotFoundError(
            "Missing enriched characteristics file: "
            f"{args.characteristics_path}"
        )

    training = pd.read_csv(
        args.training_path,
        dtype={"race_id": str},
    )

    characteristics = pd.read_csv(
        args.characteristics_path,
        dtype={"race_id": str},
    )

    modeling = build_modeling_table(
        training=training,
        characteristics=characteristics,
    )

    splitter = KFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.random_seed,
    )

    fold_assignments = np.full(
        len(modeling),
        -1,
        dtype=int,
    )

    for fold, (_, validation_index) in enumerate(
        splitter.split(modeling)
    ):
        fold_assignments[
            validation_index
        ] = fold

    if (fold_assignments < 0).any():
        raise RuntimeError(
            "Failed to assign every district to a fold."
        )

    grid_rows: list[dict[str, object]] = []
    fold_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []

    for specification in MODEL_SPECIFICATIONS:
        for alpha in DEFAULT_ALPHAS:
            result, fold_results, predictions = (
                evaluate_specification_alpha(
                    modeling=modeling,
                    specification=specification,
                    alpha=alpha,
                    fold_assignments=fold_assignments,
                )
            )

            grid_rows.append(result)
            fold_frames.append(fold_results)
            prediction_frames.append(predictions)

    grid = pd.DataFrame(grid_rows)

    grid["weighted_rmse_rank_within_model"] = (
        grid.groupby("model_id")[
            "weighted_rmse"
        ].rank(
            method="min",
            ascending=True,
        )
    )

    grid["weighted_mae_rank_within_model"] = (
        grid.groupby("model_id")[
            "weighted_mae"
        ].rank(
            method="min",
            ascending=True,
        )
    )

    grid["selection_rank_within_model"] = (
        grid[
            "weighted_rmse_rank_within_model"
        ]
        + grid[
            "weighted_mae_rank_within_model"
        ]
    )

    selected_rows = []

    for specification in MODEL_SPECIFICATIONS:
        candidates = grid.loc[
            grid["model_id"].eq(
                specification.model_id
            )
        ].sort_values(
            [
                "selection_rank_within_model",
                "weighted_rmse",
                "weighted_mae",
                "alpha",
            ]
        )

        selected_rows.append(
            candidates.iloc[0]
        )

    comparison = pd.DataFrame(
        selected_rows
    ).reset_index(drop=True)

    comparison["weighted_rmse_rank"] = (
        comparison["weighted_rmse"]
        .rank(
            method="min",
            ascending=True,
        )
    )

    comparison["weighted_mae_rank"] = (
        comparison["weighted_mae"]
        .rank(
            method="min",
            ascending=True,
        )
    )

    comparison["cv_rmse_rank"] = (
        comparison["cv_rmse"]
        .rank(
            method="min",
            ascending=True,
        )
    )

    comparison["cv_mae_rank"] = (
        comparison["cv_mae"]
        .rank(
            method="min",
            ascending=True,
        )
    )

    comparison["combined_comparison_rank"] = (
        comparison["weighted_rmse_rank"]
        + comparison["weighted_mae_rank"]
        + comparison["cv_rmse_rank"]
        + comparison["cv_mae_rank"]
    )

    comparison = comparison.sort_values(
        [
            "combined_comparison_rank",
            "weighted_rmse",
            "weighted_mae",
        ]
    ).reset_index(drop=True)

    winner_model_id = str(
        comparison.iloc[0]["model_id"]
    )

    selected_alpha_by_model = {
        str(row["model_id"]): float(
            row["alpha"]
        )
        for _, row in comparison.iterrows()
    }

    all_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    selected_predictions = all_predictions.merge(
        comparison[
            [
                "model_id",
                "alpha",
            ]
        ],
        on=[
            "model_id",
            "alpha",
        ],
        how="inner",
        validate="many_to_one",
    )

    all_folds = pd.concat(
        fold_frames,
        ignore_index=True,
    )

    selected_folds = all_folds.merge(
        comparison[
            [
                "model_id",
                "alpha",
            ]
        ],
        on=[
            "model_id",
            "alpha",
        ],
        how="inner",
        validate="many_to_one",
    )

    coefficient_frames: list[pd.DataFrame] = []

    for specification in MODEL_SPECIFICATIONS:
        selected_alpha = (
            selected_alpha_by_model[
                specification.model_id
            ]
        )

        _, coefficients = fit_final_model(
            modeling=modeling,
            specification=specification,
            alpha=selected_alpha,
        )

        coefficient_frames.append(
            coefficients
        )

    coefficients = pd.concat(
        coefficient_frames,
        ignore_index=True,
    )

    failures: list[str] = []

    if len(comparison) != len(
        MODEL_SPECIFICATIONS
    ):
        failures.append(
            "Did not select exactly one alpha per model."
        )

    if selected_predictions["model_id"].nunique() != 4:
        failures.append(
            "Selected prediction output does not contain four models."
        )

    if len(selected_predictions) != (
        len(modeling)
        * len(MODEL_SPECIFICATIONS)
    ):
        failures.append(
            "Selected prediction row count is incorrect."
        )

    legacy = comparison.loc[
        comparison["model_id"].eq(
            "A_legacy_tiers"
        )
    ].iloc[0]

    winner = comparison.iloc[0]

    comparison["weighted_rmse_change_vs_legacy"] = (
        comparison["weighted_rmse"]
        - float(legacy["weighted_rmse"])
    )

    comparison[
        "weighted_rmse_percent_change_vs_legacy"
    ] = (
        100.0
        * comparison[
            "weighted_rmse_change_vs_legacy"
        ]
        / float(legacy["weighted_rmse"])
    )

    comparison["weighted_mae_change_vs_legacy"] = (
        comparison["weighted_mae"]
        - float(legacy["weighted_mae"])
    )

    comparison[
        "weighted_mae_percent_change_vs_legacy"
    ] = (
        100.0
        * comparison[
            "weighted_mae_change_vs_legacy"
        ]
        / float(legacy["weighted_mae"])
    )

    report_columns = [
        "model_id",
        "alpha",
        "cv_mae",
        "cv_rmse",
        "weighted_mae",
        "weighted_rmse",
        "cv_r2",
        "weighted_r2",
        "cv_correlation",
        "prediction_sd",
        "weighted_mae_percent_change_vs_legacy",
        "weighted_rmse_percent_change_vs_legacy",
        "combined_comparison_rank",
    ]

    report_lines = [
        "House Elasticity Feature Model Comparison",
        "=" * 41,
        "",
        f"Common modeling rows: {len(modeling)}",
        f"Cross-validation folds: {args.folds}",
        f"Random seed: {args.random_seed}",
        (
            "Alpha grid: "
            + ", ".join(
                f"{alpha:g}"
                for alpha in DEFAULT_ALPHAS
            )
        ),
        "",
        "Selected specification results:",
        comparison[
            report_columns
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        ),
        "",
        "Top-ranked specification:",
        f"Model: {winner_model_id}",
        f"Alpha: {float(winner['alpha']):.6f}",
        (
            "Weighted MAE: "
            f"{float(winner['weighted_mae']):.6f}"
        ),
        (
            "Weighted RMSE: "
            f"{float(winner['weighted_rmse']):.6f}"
        ),
        (
            "Weighted MAE change versus legacy: "
            f"{float(winner['weighted_mae_percent_change_vs_legacy']):+.3f}%"
        ),
        (
            "Weighted RMSE change versus legacy: "
            f"{float(winner['weighted_rmse_percent_change_vs_legacy']):+.3f}%"
        ),
        (
            "Cross-validated correlation: "
            f"{float(winner['cv_correlation']):.6f}"
        ),
        (
            "Cross-validated R²: "
            f"{float(winner['cv_r2']):.6f}"
        ),
        "",
        "Interpretation:",
        (
            "Lower MAE and RMSE are better. Positive R² means the "
            "specification explains more target variation than a "
            "constant-mean prediction. Negative R² means the noisy "
            "historical targets remain harder to predict than the mean."
        ),
        "",
        "Methodological limitation:",
        (
            "Alpha selection and specification comparison use the same "
            "fixed out-of-fold prediction grid. This is appropriate for "
            "screening models, but final adoption still depends on the "
            "downstream 2022 election backtest."
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

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    grid_path = (
        args.output_dir
        / "house_elasticity_model_alpha_grid.csv"
    )

    comparison_path = (
        args.output_dir
        / "house_elasticity_model_comparison.csv"
    )

    predictions_path = (
        args.output_dir
        / "house_elasticity_model_selected_oof_predictions.csv"
    )

    folds_path = (
        args.output_dir
        / "house_elasticity_model_selected_fold_metrics.csv"
    )

    coefficients_path = (
        args.output_dir
        / "house_elasticity_model_selected_coefficients.csv"
    )

    validation_path = (
        args.output_dir
        / "house_elasticity_model_comparison_validation.txt"
    )

    grid.to_csv(
        grid_path,
        index=False,
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    selected_predictions.to_csv(
        predictions_path,
        index=False,
    )

    selected_folds.to_csv(
        folds_path,
        index=False,
    )

    coefficients.to_csv(
        coefficients_path,
        index=False,
    )

    validation_path.write_text(
        report
    )

    print(report)
    print()
    print(f"Wrote: {grid_path}")
    print(f"Wrote: {comparison_path}")
    print(f"Wrote: {predictions_path}")
    print(f"Wrote: {folds_path}")
    print(f"Wrote: {coefficients_path}")
    print(f"Wrote: {validation_path}")


if __name__ == "__main__":
    main()
