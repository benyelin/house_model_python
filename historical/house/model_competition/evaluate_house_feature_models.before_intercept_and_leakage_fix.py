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
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_BACKTEST_INPUT_PATH = (
    PROJECT_ROOT
    / "historical/house/backtests/outputs/"
    "house_2022_backtest_inputs_characteristics_elasticity.csv"
)

DEFAULT_MASTER_PATH = (
    PROJECT_ROOT
    / "historical/house/warehouse/"
    "house_district_master_features.csv"
)

DEFAULT_DNA_PATH = (
    PROJECT_ROOT
    / "historical/house/elasticity/"
    "house_district_dna_predicted_elasticity.csv"
)

DEFAULT_ENVIRONMENT_PATH = (
    PROJECT_ROOT
    / "historical/warehouse/processed/national_environment/"
    "house_2022_election_day_national_environment.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical/house/model_competition/outputs"
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
    10000.00,
)


@dataclass(frozen=True)
class FeatureSpecification:
    model_id: str
    description: str
    categorical_features: tuple[str, ...]
    numeric_features: tuple[str, ...]


BASE_CATEGORICAL = (
    "region",
    "district_type",
)

ELASTICITY_FEATURES = (
    "district_dna_elasticity_bounded_normalized",
)

COMPETITIVENESS_FEATURES = (
    "selected_regime_competitive_within_5_rate",
    "selected_regime_competitive_within_10_rate",
    "selected_regime_competitive_within_15_rate",
    "selected_regime_mean_absolute_margin",
)

VOLATILITY_FEATURES = (
    "selected_regime_margin_std",
    "selected_regime_margin_mad",
)

SWING_PREDICTABILITY_FEATURES = (
    "selected_regime_mean_absolute_swing",
    "selected_regime_swing_std",
    "selected_regime_largest_absolute_swing",
    "selected_regime_trend_slope_points_per_election",
    "selected_regime_trend_residual_rmse",
)

PARTY_BEHAVIOR_FEATURES = (
    "selected_regime_party_switch_count",
    "selected_regime_democratic_win_rate",
)

RELIABILITY_FEATURES = (
    "selected_regime_scorable_elections",
    "behavior_reliability_score",
)

DEMOGRAPHIC_FEATURES = (
    "white_vap_share",
    "black_vap_share",
    "hispanic_vap_share",
    "asian_vap_share",
    "native_vap_share",
    "pacific_vap_share",
)

PRESIDENTIAL_STRUCTURE_FEATURES = (
    "pres_2020_margin_dem",
    "pres_2024_margin_dem",
    "presidential_swing_2020_to_2024_dem",
    "dra_composite_margin_dem",
    "dra_composite_minus_presidential_average_dem",
)


FEATURE_SPECIFICATIONS = (
    FeatureSpecification(
        model_id="A_elasticity_only",
        description=(
            "District DNA elasticity candidate as the only residual feature."
        ),
        categorical_features=(),
        numeric_features=ELASTICITY_FEATURES,
    ),
    FeatureSpecification(
        model_id="B_competitiveness_only",
        description=(
            "Historical competitiveness measures only."
        ),
        categorical_features=(),
        numeric_features=COMPETITIVENESS_FEATURES,
    ),
    FeatureSpecification(
        model_id="C_volatility_only",
        description=(
            "Historical within-regime volatility measures only."
        ),
        categorical_features=(),
        numeric_features=VOLATILITY_FEATURES,
    ),
    FeatureSpecification(
        model_id="D_swing_predictability",
        description=(
            "Historical swing, trend, and predictability measures."
        ),
        categorical_features=(),
        numeric_features=SWING_PREDICTABILITY_FEATURES,
    ),
    FeatureSpecification(
        model_id="E_party_behavior",
        description=(
            "Party switching and historical Democratic win rate."
        ),
        categorical_features=(),
        numeric_features=PARTY_BEHAVIOR_FEATURES,
    ),
    FeatureSpecification(
        model_id="F_demographics",
        description=(
            "Continuous DRA voting-age demographic shares."
        ),
        categorical_features=BASE_CATEGORICAL,
        numeric_features=DEMOGRAPHIC_FEATURES,
    ),
    FeatureSpecification(
        model_id="G_combined_behavior",
        description=(
            "All regime-aware behavior measures plus reliability."
        ),
        categorical_features=(
            "selected_regime_reliability",
        ),
        numeric_features=(
            *COMPETITIVENESS_FEATURES,
            *VOLATILITY_FEATURES,
            *SWING_PREDICTABILITY_FEATURES,
            *PARTY_BEHAVIOR_FEATURES,
            *RELIABILITY_FEATURES,
        ),
    ),
    FeatureSpecification(
        model_id="H_full_district_dna",
        description=(
            "Elasticity, behavior, demographics, presidential structure, "
            "region, district type, and reliability."
        ),
        categorical_features=(
            *BASE_CATEGORICAL,
            "selected_regime_reliability",
        ),
        numeric_features=(
            *ELASTICITY_FEATURES,
            *COMPETITIVENESS_FEATURES,
            *VOLATILITY_FEATURES,
            *SWING_PREDICTABILITY_FEATURES,
            *PARTY_BEHAVIOR_FEATURES,
            *RELIABILITY_FEATURES,
            *DEMOGRAPHIC_FEATURES,
            *PRESIDENTIAL_STRUCTURE_FEATURES,
        ),
    ),
)


def parse_bool_series(series: pd.Series) -> pd.Series:
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


def logistic_probability(
    margin_dem: np.ndarray,
    error_sd: float,
) -> np.ndarray:
    scale = error_sd * np.sqrt(3.0) / np.pi

    values = np.clip(
        margin_dem / scale,
        -50.0,
        50.0,
    )

    return 1.0 / (
        1.0 + np.exp(-values)
    )


def parse_incumbent(series: pd.Series) -> pd.Series:
    return parse_bool_series(series)


def build_layer_2_margin(
    races: pd.DataFrame,
    national_environment: float,
    incumbency_bonus: float,
) -> pd.Series:
    baseline = pd.to_numeric(
        races["district_pres_margin_dem"],
        errors="coerce",
    )

    dem_incumbent = parse_incumbent(
        races["dem_is_incumbent"]
    )

    gop_incumbent = parse_incumbent(
        races["gop_is_incumbent"]
    )

    incumbency = pd.Series(
        0.0,
        index=races.index,
        dtype=float,
    )

    incumbency.loc[
        dem_incumbent & ~gop_incumbent
    ] = incumbency_bonus

    incumbency.loc[
        gop_incumbent & ~dem_incumbent
    ] = -incumbency_bonus

    incumbency.loc[
        dem_incumbent & gop_incumbent
    ] = 0.0

    return (
        baseline
        + float(national_environment)
        + incumbency
    )


def score_predictions(
    actual_margin: np.ndarray,
    predicted_margin: np.ndarray,
    error_sd: float,
) -> dict[str, float]:
    actual_winner_dem = (
        actual_margin > 0
    ).astype(int)

    predicted_winner_dem = (
        predicted_margin > 0
    ).astype(int)

    probabilities = logistic_probability(
        predicted_margin,
        error_sd,
    )

    probabilities = np.clip(
        probabilities,
        1e-12,
        1.0 - 1e-12,
    )

    return {
        "mean_absolute_error": float(
            mean_absolute_error(
                actual_margin,
                predicted_margin,
            )
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    actual_margin,
                    predicted_margin,
                )
            )
        ),
        "winner_accuracy": float(
            accuracy_score(
                actual_winner_dem,
                predicted_winner_dem,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                actual_winner_dem,
                probabilities,
            )
        ),
        "log_loss": float(
            log_loss(
                actual_winner_dem,
                probabilities,
                labels=[0, 1],
            )
        ),
        "predicted_dem_wins": float(
            predicted_winner_dem.sum()
        ),
        "expected_dem_wins": float(
            probabilities.sum()
        ),
        "actual_dem_wins": float(
            actual_winner_dem.sum()
        ),
        "mean_margin_error_dem_bias": float(
            np.mean(
                predicted_margin
                - actual_margin
            )
        ),
    }


def make_pipeline(
    specification: FeatureSpecification,
    alpha: float,
) -> Pipeline:
    transformers = []

    if specification.categorical_features:
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

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                list(
                    specification.categorical_features
                ),
            )
        )

    if specification.numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                list(
                    specification.numeric_features
                ),
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                Ridge(
                    alpha=alpha,
                    fit_intercept=True,
                ),
            ),
        ]
    )


def evaluate_specification(
    modeling: pd.DataFrame,
    specification: FeatureSpecification,
    alpha: float,
    fold_assignments: np.ndarray,
    error_sd: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    feature_columns = [
        *specification.categorical_features,
        *specification.numeric_features,
    ]

    features = modeling[
        feature_columns
    ]

    target_residual = modeling[
        "layer_2_residual_dem"
    ].to_numpy(dtype=float)

    layer_2_margin = modeling[
        "layer_2_margin_dem"
    ].to_numpy(dtype=float)

    actual_margin = modeling[
        "actual_dem_margin"
    ].to_numpy(dtype=float)

    predicted_residual = np.full(
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

        training_mask = (
            ~validation_mask
        )

        pipeline = make_pipeline(
            specification=specification,
            alpha=alpha,
        )

        pipeline.fit(
            features.loc[training_mask],
            target_residual[training_mask],
        )

        predicted_residual[
            validation_mask
        ] = pipeline.predict(
            features.loc[validation_mask]
        )

    if np.isnan(
        predicted_residual
    ).any():
        raise RuntimeError(
            "Missing out-of-fold residual predictions for "
            f"{specification.model_id}."
        )

    corrected_margin = (
        layer_2_margin
        + predicted_residual
    )

    scores = score_predictions(
        actual_margin=actual_margin,
        predicted_margin=corrected_margin,
        error_sd=error_sd,
    )

    result: dict[str, object] = {
        "model_id": specification.model_id,
        "model_description": specification.description,
        "alpha": alpha,
        "rows": len(modeling),
        "categorical_feature_count": len(
            specification.categorical_features
        ),
        "numeric_feature_count": len(
            specification.numeric_features
        ),
        "residual_prediction_mean": float(
            np.mean(predicted_residual)
        ),
        "residual_prediction_sd": float(
            np.std(
                predicted_residual,
                ddof=0,
            )
        ),
        "residual_prediction_min": float(
            np.min(predicted_residual)
        ),
        "residual_prediction_max": float(
            np.max(predicted_residual)
        ),
        **scores,
    }

    prediction_table = pd.DataFrame(
        {
            "model_id": specification.model_id,
            "alpha": alpha,
            "race_id": modeling["race_id"],
            "fold": fold_assignments,
            "layer_2_margin_dem": layer_2_margin,
            "predicted_residual_dem": predicted_residual,
            "corrected_margin_dem": corrected_margin,
            "actual_dem_margin": actual_margin,
            "layer_2_absolute_error": np.abs(
                layer_2_margin
                - actual_margin
            ),
            "corrected_absolute_error": np.abs(
                corrected_margin
                - actual_margin
            ),
        }
    )

    prediction_table[
        "absolute_error_change"
    ] = (
        prediction_table[
            "corrected_absolute_error"
        ]
        - prediction_table[
            "layer_2_absolute_error"
        ]
    )

    return result, prediction_table


def fit_coefficients(
    modeling: pd.DataFrame,
    specification: FeatureSpecification,
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
            "layer_2_residual_dem"
        ].to_numpy(dtype=float),
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

    coefficients[
        "absolute_coefficient"
    ] = coefficients[
        "coefficient"
    ].abs()

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
            "Evaluate candidate House feature blocks as out-of-fold "
            "corrections to the established 2022 Layer 2 forecast."
        )
    )

    parser.add_argument(
        "--backtest-input-path",
        type=Path,
        default=DEFAULT_BACKTEST_INPUT_PATH,
    )

    parser.add_argument(
        "--master-path",
        type=Path,
        default=DEFAULT_MASTER_PATH,
    )

    parser.add_argument(
        "--dna-path",
        type=Path,
        default=DEFAULT_DNA_PATH,
    )

    parser.add_argument(
        "--environment-path",
        type=Path,
        default=DEFAULT_ENVIRONMENT_PATH,
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

    parser.add_argument(
        "--incumbency-bonus",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--error-sd",
        type=float,
        default=6.5,
    )

    args = parser.parse_args()

    for path in [
        args.backtest_input_path,
        args.master_path,
        args.dna_path,
        args.environment_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required input: {path}"
            )

    races = pd.read_csv(
        args.backtest_input_path,
        dtype={"race_id": str},
    )

    master = pd.read_csv(
        args.master_path,
        dtype={"race_id": str},
    )

    dna = pd.read_csv(
        args.dna_path,
        dtype={"race_id": str},
    )

    environment = pd.read_csv(
        args.environment_path,
    )

    if len(races) != 435:
        raise ValueError(
            f"Expected 435 backtest rows; found {len(races)}."
        )

    if len(master) != 435:
        raise ValueError(
            f"Expected 435 master rows; found {len(master)}."
        )

    if len(dna) != 435:
        raise ValueError(
            f"Expected 435 DNA rows; found {len(dna)}."
        )

    for name, frame in [
        ("backtest", races),
        ("master", master),
        ("dna", dna),
    ]:
        if frame["race_id"].duplicated().any():
            raise ValueError(
                f"Duplicate race IDs found in {name} table."
            )

    if len(environment) != 1:
        raise ValueError(
            "National-environment file must contain one row."
        )

    national_environment = pd.to_numeric(
        environment.iloc[0][
            "national_environment_margin_dem"
        ],
        errors="coerce",
    )

    if pd.isna(national_environment):
        raise ValueError(
            "National environment is blank or nonnumeric."
        )

    dna_keep = dna[
        [
            "race_id",
            "district_dna_elasticity_bounded_normalized",
        ]
    ].copy()

    feature_columns = sorted(
        {
            feature
            for specification in FEATURE_SPECIFICATIONS
            for feature in (
                *specification.categorical_features,
                *specification.numeric_features,
            )
            if feature
            != "district_dna_elasticity_bounded_normalized"
        }
    )

    missing_master_features = sorted(
        set(feature_columns)
        - set(master.columns)
    )

    if missing_master_features:
        raise ValueError(
            "Master table is missing required feature columns: "
            + ", ".join(
                missing_master_features
            )
        )

    master_keep = master[
        [
            "race_id",
            *feature_columns,
        ]
    ].copy()

    modeling = races.merge(
        master_keep,
        on="race_id",
        how="left",
        validate="one_to_one",
        suffixes=(
            "",
            "_master",
        ),
    )

    modeling = modeling.merge(
        dna_keep,
        on="race_id",
        how="left",
        validate="one_to_one",
    )

    required_race_columns = {
        "race_id",
        "district_pres_margin_dem",
        "actual_dem_margin",
        "general_election_party_structure",
        "dem_is_incumbent",
        "gop_is_incumbent",
    }

    missing_race_columns = sorted(
        required_race_columns
        - set(modeling.columns)
    )

    if missing_race_columns:
        raise ValueError(
            "Backtest table is missing columns: "
            + ", ".join(
                missing_race_columns
            )
        )

    modeling[
        "district_pres_margin_dem"
    ] = pd.to_numeric(
        modeling[
            "district_pres_margin_dem"
        ],
        errors="coerce",
    )

    modeling[
        "actual_dem_margin"
    ] = pd.to_numeric(
        modeling[
            "actual_dem_margin"
        ],
        errors="coerce",
    )

    modeling[
        "layer_2_margin_dem"
    ] = build_layer_2_margin(
        races=modeling,
        national_environment=float(
            national_environment
        ),
        incumbency_bonus=args.incumbency_bonus,
    )

    scoring_mask = (
        modeling[
            "general_election_party_structure"
        ]
        .fillna("")
        .eq("D_vs_R")
        & modeling[
            "district_pres_margin_dem"
        ].notna()
        & modeling[
            "actual_dem_margin"
        ].notna()
    )

    modeling = modeling.loc[
        scoring_mask
    ].copy()

    if len(modeling) != 400:
        raise ValueError(
            f"Expected 400 scored races; found {len(modeling)}."
        )

    modeling[
        "layer_2_residual_dem"
    ] = (
        modeling[
            "actual_dem_margin"
        ]
        - modeling[
            "layer_2_margin_dem"
        ]
    )

    all_categorical = sorted(
        {
            feature
            for specification in FEATURE_SPECIFICATIONS
            for feature in specification.categorical_features
        }
    )

    all_numeric = sorted(
        {
            feature
            for specification in FEATURE_SPECIFICATIONS
            for feature in specification.numeric_features
        }
    )

    for column in all_categorical:
        modeling[column] = clean_category(
            modeling[column]
        )

    for column in all_numeric:
        modeling[column] = pd.to_numeric(
            modeling[column],
            errors="coerce",
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

    actual_margin = modeling[
        "actual_dem_margin"
    ].to_numpy(dtype=float)

    layer_2_margin = modeling[
        "layer_2_margin_dem"
    ].to_numpy(dtype=float)

    baseline_scores = score_predictions(
        actual_margin=actual_margin,
        predicted_margin=layer_2_margin,
        error_sd=args.error_sd,
    )

    grid_rows: list[
        dict[str, object]
    ] = []

    prediction_frames: list[
        pd.DataFrame
    ] = []

    for specification in FEATURE_SPECIFICATIONS:
        for alpha in ALPHA_GRID:
            result, predictions = evaluate_specification(
                modeling=modeling,
                specification=specification,
                alpha=alpha,
                fold_assignments=fold_assignments,
                error_sd=args.error_sd,
            )

            grid_rows.append(
                result
            )

            prediction_frames.append(
                predictions
            )

    grid = pd.DataFrame(
        grid_rows
    )

    for metric in [
        "mean_absolute_error",
        "rmse",
        "brier_score",
        "log_loss",
    ]:
        grid[
            f"{metric}_rank_within_model"
        ] = grid.groupby(
            "model_id"
        )[metric].rank(
            method="min",
            ascending=True,
        )

    grid[
        "selection_rank_within_model"
    ] = grid[
        [
            "mean_absolute_error_rank_within_model",
            "rmse_rank_within_model",
            "brier_score_rank_within_model",
            "log_loss_rank_within_model",
        ]
    ].sum(axis=1)

    selected_rows = []

    for specification in FEATURE_SPECIFICATIONS:
        selected = grid.loc[
            grid["model_id"].eq(
                specification.model_id
            )
        ].sort_values(
            [
                "selection_rank_within_model",
                "mean_absolute_error",
                "rmse",
                "alpha",
            ]
        ).iloc[0]

        selected_rows.append(
            selected
        )

    comparison = pd.DataFrame(
        selected_rows
    ).reset_index(drop=True)

    for metric in [
        "mean_absolute_error",
        "rmse",
        "winner_accuracy",
        "brier_score",
        "log_loss",
        "predicted_dem_wins",
        "expected_dem_wins",
    ]:
        comparison[
            f"layer_2_{metric}"
        ] = baseline_scores[metric]

        comparison[
            f"change_vs_layer_2_{metric}"
        ] = (
            comparison[metric]
            - baseline_scores[metric]
        )

    comparison[
        "predicted_win_count_error"
    ] = (
        comparison[
            "predicted_dem_wins"
        ]
        - comparison[
            "actual_dem_wins"
        ]
    ).abs()

    comparison[
        "expected_win_count_error"
    ] = (
        comparison[
            "expected_dem_wins"
        ]
        - comparison[
            "actual_dem_wins"
        ]
    ).abs()

    comparison[
        "layer_2_predicted_win_count_error"
    ] = abs(
        baseline_scores[
            "predicted_dem_wins"
        ]
        - baseline_scores[
            "actual_dem_wins"
        ]
    )

    comparison[
        "layer_2_expected_win_count_error"
    ] = abs(
        baseline_scores[
            "expected_dem_wins"
        ]
        - baseline_scores[
            "actual_dem_wins"
        ]
    )

    for metric in [
        "mean_absolute_error",
        "rmse",
        "brier_score",
        "log_loss",
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
                "mean_absolute_error_rank",
                "rmse_rank",
                "brier_score_rank",
                "log_loss_rank",
            ]
        ].sum(axis=1)
    )

    comparison = comparison.sort_values(
        [
            "combined_rank",
            "mean_absolute_error",
            "rmse",
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

    for specification in FEATURE_SPECIFICATIONS:
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

    failures: list[str] = []

    if len(comparison) != len(
        FEATURE_SPECIFICATIONS
    ):
        failures.append(
            "Not all feature specifications produced a selected model."
        )

    if len(selected_predictions) != (
        len(FEATURE_SPECIFICATIONS)
        * len(modeling)
    ):
        failures.append(
            "Selected prediction row count is incorrect."
        )

    winner = comparison.iloc[0]

    report_columns = [
        "model_id",
        "alpha",
        "mean_absolute_error",
        "change_vs_layer_2_mean_absolute_error",
        "rmse",
        "change_vs_layer_2_rmse",
        "winner_accuracy",
        "change_vs_layer_2_winner_accuracy",
        "brier_score",
        "change_vs_layer_2_brier_score",
        "log_loss",
        "change_vs_layer_2_log_loss",
        "predicted_win_count_error",
        "expected_win_count_error",
        "residual_prediction_sd",
        "combined_rank",
    ]

    report_lines = [
        "2022 House Feature Model Competition",
        "=" * 36,
        "",
        f"Scored races: {len(modeling)}",
        f"Cross-validation folds: {args.folds}",
        f"Random seed: {args.random_seed}",
        (
            "National environment Dem: "
            f"{float(national_environment):+.6f}"
        ),
        f"Incumbency bonus: {args.incumbency_bonus:.4f}",
        f"Probability error scale: {args.error_sd:.4f}",
        "",
        "Layer 2 benchmark:",
        (
            "MAE: "
            f"{baseline_scores['mean_absolute_error']:.6f}"
        ),
        (
            "RMSE: "
            f"{baseline_scores['rmse']:.6f}"
        ),
        (
            "Winner accuracy: "
            f"{baseline_scores['winner_accuracy']:.6f}"
        ),
        (
            "Brier: "
            f"{baseline_scores['brier_score']:.6f}"
        ),
        (
            "Log loss: "
            f"{baseline_scores['log_loss']:.6f}"
        ),
        "",
        "Selected feature models:",
        comparison[
            report_columns
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        ),
        "",
        "Top-ranked feature specification:",
        f"Model: {winner['model_id']}",
        f"Alpha: {float(winner['alpha']):.6f}",
        (
            "MAE change versus Layer 2: "
            f"{float(winner['change_vs_layer_2_mean_absolute_error']):+.6f}"
        ),
        (
            "RMSE change versus Layer 2: "
            f"{float(winner['change_vs_layer_2_rmse']):+.6f}"
        ),
        (
            "Brier change versus Layer 2: "
            f"{float(winner['change_vs_layer_2_brier_score']):+.6f}"
        ),
        (
            "Log-loss change versus Layer 2: "
            f"{float(winner['change_vs_layer_2_log_loss']):+.6f}"
        ),
        "",
        "Methodological limitation:",
        (
            "This is a held-out-district 2022 residual screen, not a "
            "held-out-election-cycle test. Features that win here must "
            "still pass pseudo-out-of-sample testing across earlier cycles."
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
        / "house_feature_model_alpha_grid.csv"
    )

    comparison_path = (
        args.output_dir
        / "house_feature_model_comparison.csv"
    )

    predictions_path = (
        args.output_dir
        / "house_feature_model_selected_oof_predictions.csv"
    )

    coefficients_path = (
        args.output_dir
        / "house_feature_model_selected_coefficients.csv"
    )

    folds_path = (
        args.output_dir
        / "house_feature_model_fold_assignments.csv"
    )

    validation_path = (
        args.output_dir
        / "house_feature_model_competition_validation.txt"
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

    pd.DataFrame(
        {
            "race_id": modeling["race_id"],
            "fold": fold_assignments,
        }
    ).to_csv(
        folds_path,
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
    print(f"Wrote: {folds_path}")
    print(f"Wrote: {validation_path}")


if __name__ == "__main__":
    main()
