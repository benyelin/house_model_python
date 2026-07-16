from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


ComponentFunction = Callable[
    [pd.DataFrame, "BacktestParameters"],
    pd.Series,
]


@dataclass(frozen=True)
class BacktestParameters:
    national_environment_margin_dem: float
    incumbency_bonus: float = 1.5
    elasticity_default: float = 1.0


@dataclass(frozen=True)
class ForecastComponent:
    name: str
    function: ComponentFunction
    description: str


def numeric_series(
    df: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> pd.Series:
    if column not in df.columns:
        return pd.Series(
            default,
            index=df.index,
            dtype=float,
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).fillna(default)


def parse_bool_series(
    series: pd.Series,
) -> pd.Series:
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


def presidential_baseline_component(
    df: pd.DataFrame,
    parameters: BacktestParameters,
) -> pd.Series:
    del parameters

    if "district_pres_margin_dem" not in df.columns:
        raise ValueError(
            "Missing district_pres_margin_dem."
        )

    return pd.to_numeric(
        df["district_pres_margin_dem"],
        errors="coerce",
    )


def national_environment_component(
    df: pd.DataFrame,
    parameters: BacktestParameters,
) -> pd.Series:
    return pd.Series(
        parameters.national_environment_margin_dem,
        index=df.index,
        dtype=float,
    )


def elasticity_environment_component(
    df: pd.DataFrame,
    parameters: BacktestParameters,
) -> pd.Series:
    """
    Replacement for the unscaled national-environment component.

    This component is not yet used by the official Layer 1–2 benchmark.
    It will become the Layer 3 environment contribution.
    """
    elasticity = numeric_series(
        df,
        "district_elasticity",
        parameters.elasticity_default,
    )

    return (
        parameters.national_environment_margin_dem
        * elasticity
    )


def incumbency_component(
    df: pd.DataFrame,
    parameters: BacktestParameters,
) -> pd.Series:
    if "dem_is_incumbent" not in df.columns:
        raise ValueError(
            "Missing dem_is_incumbent."
        )

    if "gop_is_incumbent" not in df.columns:
        raise ValueError(
            "Missing gop_is_incumbent."
        )

    dem_incumbent = parse_bool_series(
        df["dem_is_incumbent"]
    )

    gop_incumbent = parse_bool_series(
        df["gop_is_incumbent"]
    )

    adjustment = pd.Series(
        0.0,
        index=df.index,
        dtype=float,
    )

    adjustment.loc[
        dem_incumbent & ~gop_incumbent
    ] = parameters.incumbency_bonus

    adjustment.loc[
        gop_incumbent & ~dem_incumbent
    ] = -parameters.incumbency_bonus

    # Member-versus-member races receive no net party adjustment.
    adjustment.loc[
        dem_incumbent & gop_incumbent
    ] = 0.0

    return adjustment


def state_environment_component(
    df: pd.DataFrame,
    parameters: BacktestParameters,
) -> pd.Series:
    del parameters

    return numeric_series(
        df,
        "state_environment_adjustment_dem",
        0.0,
    )


def candidate_quality_component(
    df: pd.DataFrame,
    parameters: BacktestParameters,
) -> pd.Series:
    del parameters

    return numeric_series(
        df,
        "candidate_quality_adjustment_dem",
        0.0,
    )


def special_adjustment_component(
    df: pd.DataFrame,
    parameters: BacktestParameters,
) -> pd.Series:
    del parameters

    return numeric_series(
        df,
        "special_adjustment_dem",
        0.0,
    )


def polling_component(
    df: pd.DataFrame,
    parameters: BacktestParameters,
) -> pd.Series:
    del parameters

    return numeric_series(
        df,
        "polling_adjustment_dem",
        0.0,
    )


COMPONENTS: dict[str, ForecastComponent] = {
    "presidential_baseline": ForecastComponent(
        name="presidential_baseline",
        function=presidential_baseline_component,
        description=(
            "2020 Democratic presidential margin on the "
            "district boundaries used in the election."
        ),
    ),
    "national_environment": ForecastComponent(
        name="national_environment",
        function=national_environment_component,
        description=(
            "Uniform national-environment adjustment."
        ),
    ),
    "elasticity_environment": ForecastComponent(
        name="elasticity_environment",
        function=elasticity_environment_component,
        description=(
            "National environment multiplied by district elasticity."
        ),
    ),
    "incumbency": ForecastComponent(
        name="incumbency",
        function=incumbency_component,
        description=(
            "Symmetric Democratic or Republican incumbency adjustment."
        ),
    ),
    "state_environment": ForecastComponent(
        name="state_environment",
        function=state_environment_component,
        description="State-specific environment adjustment.",
    ),
    "candidate_quality": ForecastComponent(
        name="candidate_quality",
        function=candidate_quality_component,
        description="Candidate-quality adjustment.",
    ),
    "special_adjustment": ForecastComponent(
        name="special_adjustment",
        function=special_adjustment_component,
        description="Manual or race-specific special adjustment.",
    ),
    "polling": ForecastComponent(
        name="polling",
        function=polling_component,
        description="Polling-derived adjustment.",
    ),
}


def calculate_component(
    df: pd.DataFrame,
    component_name: str,
    parameters: BacktestParameters,
) -> pd.Series:
    if component_name not in COMPONENTS:
        raise KeyError(
            f"Unknown component: {component_name}"
        )

    values = COMPONENTS[
        component_name
    ].function(
        df,
        parameters,
    )

    if not values.index.equals(df.index):
        raise ValueError(
            f"Component {component_name} returned "
            "a misaligned index."
        )

    return pd.to_numeric(
        values,
        errors="coerce",
    )


def calculate_forecast(
    df: pd.DataFrame,
    component_names: list[str],
    parameters: BacktestParameters,
) -> tuple[pd.Series, pd.DataFrame]:
    if not component_names:
        raise ValueError(
            "At least one forecast component is required."
        )

    detail = pd.DataFrame(
        index=df.index,
    )

    for component_name in component_names:
        detail[component_name] = calculate_component(
            df=df,
            component_name=component_name,
            parameters=parameters,
        )

    forecast = detail.sum(
        axis=1,
        min_count=1,
    )

    return forecast, detail
