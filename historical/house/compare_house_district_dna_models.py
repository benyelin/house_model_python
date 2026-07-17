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

DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "historical/house/warehouse/"
    "house_district_master_features.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical/house/elasticity/"
    "district_dna_model_comparison"
)

ALPHA_GRID = (
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
    3000.00,
)


@dataclass(frozen=True)
class ModelSpecification:
    model_id: str
    description: str
    categorical_features: tuple[str, ...]
    numeric_features: tuple[str, ...]


BASE_CATEGORICAL = (
    "region",
    "district_type",
    "dra_population_dataset",
)

BASE_DEMOGRAPHIC_NUMERIC = (
    "pres_2020_margin_dem",
    "white_vap_share",
    "black_vap_share",
    "hispanic_vap_share",
    "asian_vap_share",
    "native_vap_share",
    "pacific_vap_share",
)

BEHAVIOR_CATEGORICAL = (
    "selected_regime_reliability",
)

BEHAVIOR_NUMERIC = (
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
)

FULL_STRUCTURAL_NUMERIC = (
    "pres_2024_margin_dem",
    "presidential_swing_2020_to_2024_dem",
    "dra_composite_margin_dem",
    "dra_composite_minus_presidential_average_dem",
)


MODEL_SPECIFICATIONS = (
    ModelSpecification(
        model_id="A_continuous_demographics",
        description=(
            "Current best continuous-demographics specification."
        ),
        categorical_features=BASE_CATEGORICAL,
        numeric_features=BASE_DEMOGRAPHIC_NUMERIC,
    ),
    ModelSpecification(
        model_id="B_behavior_only",
        description=(
            "Boundary-regime-aware historical behavior without "
            "direct historical mean House margin."
        ),
        categorical_features=(
            "region",
            "district_type",
            *BEHAVIOR_CATEGORICAL,
        ),
        numeric_features=(
            "pres_2020_margin_dem",
            *BEHAVIOR_NUMERIC,
        ),
    ),
    ModelSpecification(
        model_id="C_district_dna",
        description=(
            "Continuous demographics plus regime-aware behavior."
        ),
        categorical_features=(
            *BASE_CATEGORICAL,
            *BEHAVIOR_CATEGORICAL,
        ),
        numeric_features=(
            *BASE_DEMOGRAPHIC_NUMERIC,
            *BEHAVIOR_NUMERIC,
        ),
    ),
    ModelSpecification(
        model_id="D_full_structural_dna",
        description=(
            "District DNA plus recent presidential movement and "
            "DRA composite political structure."
        ),
        categorical_features=(
            *BASE_CATEGORICAL,
            *BEHAVIOR_CATEGORICAL,
        ),
        numeric_features=(
            *BASE_DEMOGRAPHIC_NUMERIC,
            *BEHAVIOR_NUMERIC,
            *FULL_STRUCTURAL_NUMERIC,
        ),
    ),
)


def clean_category(series: pd.Series) -> pd.Series:
    return (
        series.fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
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


def build_modeling_table(master: pd.DataFrame) -> pd.DataFrame:
    all_categorical = sorted(
        {
            feature
            for specification in MODEL_SPECIFICATIONS
            for feature in specification.categorical_features
        }
    )

    all_numeric = sorted(
        {
            feature
            for specification in MODEL_SPECIFICATIONS
            for feature in specification.numeric_features
        }
    )

    required_columns = {
        "race_id",
        "complete_master_feature_row",
        "historical_raw_elasticity",
        "historical_elasticity_information",
        *all_categorical,
        *all_numeric,
    }

    missing = sorted(
        required_columns - set(master.columns)
    )

    if missing:
        raise ValueError(
            "Master feature table is missing columns: "
            + ", ".join(missing)
        )

    if len(master) != 435:
        raise ValueError(
            f"Expected 435 master rows; found {len(master)}."
        )

    if master["race_id"].duplicated().any():
        raise ValueError(
            "Duplicate race IDs found in master feature table."
        )

    eligible = parse_bool(
        master["complete_master_feature_row"]
    )

    modeling = master.loc[
        eligible,
        [
            "race_id",
            "historical_raw_elasticity",
            "historical_elasticity_information",
            *all_categorical,
            *all_numeric,
        ],
    ].copy()

    if len(modeling) != 393:
        raise ValueError(
            f"Expected 393 complete modeling rows; found {len(modeling)}."
        )

    for column in all_categorical:
        modeling[column] = clean_category(
            modeling[column]
        )

    for column in [
        "historical_raw_elasticity",
        "historical_elasticity_information",
        *all_numeric,
    ]:
        modeling[column] = pd.to_numeric(
            modeling[column],
            errors="coerce",
        )

    if modeling["historical_raw_elasticity"].isna().any():
        raise ValueError(
            "Complete rows contain missing elasticity targets."
        )

    if (
        modeling["historical_elasticity_information"].isna().any()
        or modeling["historical_elasticity_information"].le(0).any()
    ):
        raise ValueError(
            "Complete rows contain invalid elasticity information."
        )

    missing_numeric = (
        modeling[all_numeric]
        .isna()
        .sum()
    )

    missing_numeric = missing_numeric.loc[
        missing_numeric.gt(0)
    ]

    if not missing_numeric.empty:
        raise ValueError(
            "Complete rows contain missing numeric predictors:\n"
            + missing_numeric.to_string()
        )

    median_information = float(
        modeling[
            "historical_elasticity_information"
        ].median()
    )

    modeling["model_sample_weight"] = (
        modeling[
            "historical_elasticity_information"
        ].clip(
            lower=0.25 * median_information,
            upper=2.00 * median_information,
        )
    )

    return modeling.sort_values(
        "race_id"
    ).reset_index(drop=True)


def make_pipeline(
    specification: ModelSpecification,
    alpha: float,
) -> Pipeline:
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
                list(specification.categorical_features),
            ),
            (
                "numeric",
                numeric_pipeline,
                list(specification.numeric_features),
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

    target_mean = float(
        np.average(
            target,
            weights=weights,
        )
    )

    weighted_sse = float(
        np.sum(
            weights * np.square(errors)
        )
    )

    weighted_sst = float(
        np.sum(
            weights
            * np.square(
                target - target_mean
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


def evaluate_model(
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
        "historical_raw_elasticity"
    ].to_numpy(dtype=float)

    weights = modeling[
        "model_sample_weight"
    ].to_numpy(dtype=float)

    predictions = np.full(
        len(modeling),
        np.nan,
        dtype=float,
    )

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

        predictions[validation_mask] = (
            pipeline.predict(
                features.loc[validation_mask]
            )
        )

    if np.isnan(predictions).any():
        raise RuntimeError(
            f"Missing OOF predictions for {specification.model_id}."
        )

    weighted = weighted_metrics(
        target=target,
        prediction=predictions,
        weights=weights,
    )

    result = {
        "model_id": specification.model_id,
        "model_description": specification.description,
        "alpha": alpha,
        "categorical_feature_count": len(
            specification.categorical_features
        ),
        "numeric_feature_count": len(
            specification.numeric_features
        ),
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
        "cv_correlation": float(
            np.corrcoef(
                target,
                predictions,
            )[0, 1]
        ),
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
            "model_sample_weight": weights,
        }
    )

    return result, prediction_table


def fit_coefficients(
    modeling: pd.DataFrame,
    specification: ModelSpecification,
    alpha: float,
) -> pd.DataFrame:
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
        modeling[
            "historical_raw_elasticity"
        ].to_numpy(dtype=float),
        model__sample_weight=(
            modeling[
                "model_sample_weight"
            ].to_numpy(dtype=float)
        ),
    )

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    ridge = pipeline.named_steps[
        "model"
    ]

    coefficients = pd.DataFrame(
        {
            "model_id": specification.model_id,
            "alpha": alpha,
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
            "model_id": [
                specification.model_id
            ],
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

    return pd.concat(
        [
            intercept,
            coefficients.sort_values(
                "absolute_coefficient",
                ascending=False,
            ),
        ],
        ignore_index=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare House District DNA elasticity specifications."
        )
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
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

    if not args.input_path.exists():
        raise FileNotFoundError(
            f"Missing master feature table: {args.input_path}"
        )

    master = pd.read_csv(
        args.input_path,
        dtype={
            "race_id": str,
            "state": str,
            "district": str,
        },
    )

    modeling = build_modeling_table(
        master
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

    grid_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for specification in MODEL_SPECIFICATIONS:
        for alpha in ALPHA_GRID:
            result, predictions = evaluate_model(
                modeling=modeling,
                specification=specification,
                alpha=alpha,
                fold_assignments=fold_assignments,
            )

            grid_rows.append(result)
            prediction_frames.append(
                predictions
            )

    grid = pd.DataFrame(
        grid_rows
    )

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

    baseline = comparison.loc[
        comparison["model_id"].eq(
            "A_continuous_demographics"
        )
    ].iloc[0]

    comparison["weighted_mae_change_vs_baseline"] = (
        comparison["weighted_mae"]
        - float(
            baseline["weighted_mae"]
        )
    )

    comparison[
        "weighted_mae_percent_change_vs_baseline"
    ] = (
        100.0
        * comparison[
            "weighted_mae_change_vs_baseline"
        ]
        / float(
            baseline["weighted_mae"]
        )
    )

    comparison["weighted_rmse_change_vs_baseline"] = (
        comparison["weighted_rmse"]
        - float(
            baseline["weighted_rmse"]
        )
    )

    comparison[
        "weighted_rmse_percent_change_vs_baseline"
    ] = (
        100.0
        * comparison[
            "weighted_rmse_change_vs_baseline"
        ]
        / float(
            baseline["weighted_rmse"]
        )
    )

    for metric in [
        "weighted_mae",
        "weighted_rmse",
        "cv_mae",
        "cv_rmse",
    ]:
        comparison[
            f"{metric}_rank"
        ] = comparison[metric].rank(
            method="min",
            ascending=True,
        )

    comparison["combined_rank"] = (
        comparison[
            [
                "weighted_mae_rank",
                "weighted_rmse_rank",
                "cv_mae_rank",
                "cv_rmse_rank",
            ]
        ].sum(axis=1)
    )

    comparison = comparison.sort_values(
        [
            "combined_rank",
            "weighted_rmse",
            "weighted_mae",
        ]
    ).reset_index(drop=True)

    selected_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    ).merge(
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

    coefficient_frames = []

    for specification in MODEL_SPECIFICATIONS:
        selected_alpha = float(
            comparison.loc[
                comparison["model_id"].eq(
                    specification.model_id
                ),
                "alpha",
            ].iloc[0]
        )

        coefficient_frames.append(
            fit_coefficients(
                modeling=modeling,
                specification=specification,
                alpha=selected_alpha,
            )
        )

    coefficients = pd.concat(
        coefficient_frames,
        ignore_index=True,
    )

    winner = comparison.iloc[0]

    failures: list[str] = []

    if len(comparison) != 4:
        failures.append(
            "Did not select exactly four model specifications."
        )

    if len(selected_predictions) != (
        4 * len(modeling)
    ):
        failures.append(
            "Selected OOF prediction row count is incorrect."
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
        "weighted_mae_percent_change_vs_baseline",
        "weighted_rmse_percent_change_vs_baseline",
        "combined_rank",
    ]

    report_lines = [
        "House District DNA Model Comparison",
        "=" * 35,
        "",
        f"Common modeling rows: {len(modeling)}",
        f"Cross-validation folds: {args.folds}",
        f"Random seed: {args.random_seed}",
        (
            "Alpha grid: "
            + ", ".join(
                f"{alpha:g}"
                for alpha in ALPHA_GRID
            )
        ),
        "",
        "Selected model results:",
        comparison[
            report_columns
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        ),
        "",
        "Top-ranked specification:",
        f"Model: {winner['model_id']}",
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
            "Weighted MAE change versus demographics baseline: "
            f"{float(winner['weighted_mae_percent_change_vs_baseline']):+.3f}%"
        ),
        (
            "Weighted RMSE change versus demographics baseline: "
            f"{float(winner['weighted_rmse_percent_change_vs_baseline']):+.3f}%"
        ),
        (
            "Cross-validated R²: "
            f"{float(winner['cv_r2']):.6f}"
        ),
        (
            "Cross-validated correlation: "
            f"{float(winner['cv_correlation']):.6f}"
        ),
        "",
        "Leakage control:",
        (
            "Direct historical mean House margin and derived historical "
            "partisan-baseline indexes are excluded. Behavior features "
            "focus on volatility, swing magnitude, competitiveness, "
            "party switching, trend residuals, and reliability."
        ),
        "",
        "Adoption rule:",
        (
            "This screening comparison does not by itself authorize "
            "production use. Any winning District DNA specification must "
            "next improve the downstream 2022 Layer 3 election backtest."
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

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    grid_path = (
        args.output_dir
        / "house_district_dna_alpha_grid.csv"
    )

    comparison_path = (
        args.output_dir
        / "house_district_dna_model_comparison.csv"
    )

    predictions_path = (
        args.output_dir
        / "house_district_dna_selected_oof_predictions.csv"
    )

    coefficients_path = (
        args.output_dir
        / "house_district_dna_selected_coefficients.csv"
    )

    validation_path = (
        args.output_dir
        / "house_district_dna_model_comparison_validation.txt"
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
    print(f"Wrote: {coefficients_path}")
    print(f"Wrote: {validation_path}")


if __name__ == "__main__":
    main()
