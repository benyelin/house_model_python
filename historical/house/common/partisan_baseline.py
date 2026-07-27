from __future__ import annotations

import pandas as pd


def add_normalized_partisan_baseline(
    df: pd.DataFrame,
    national_margins: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add the historical normalized district partisan baseline:

        district presidential margin
        minus national presidential margin for the same election year

    This function performs only baseline normalization. It contains no
    forecasting, environment, incumbency, candidate-quality, polling,
    or simulation logic.
    """
    required_input_columns = {
        "district_pres_margin_dem",
        "presidential_result_year",
    }

    missing_inputs = required_input_columns - set(df.columns)

    if missing_inputs:
        raise ValueError(
            "Cannot construct normalized partisan baseline; "
            f"missing input columns: {sorted(missing_inputs)}"
        )

    required_lookup_columns = {
        "presidential_result_year",
        "national_pres_margin_dem",
    }

    missing_lookup = required_lookup_columns - set(
        national_margins.columns
    )

    if missing_lookup:
        raise ValueError(
            "National presidential-margin lookup is missing columns: "
            f"{sorted(missing_lookup)}"
        )

    out = df.copy()

    lookup = national_margins[
        [
            "presidential_result_year",
            "national_pres_margin_dem",
        ]
    ].copy()

    lookup["presidential_result_year"] = pd.to_numeric(
        lookup["presidential_result_year"],
        errors="raise",
    ).astype(int)

    lookup["national_pres_margin_dem"] = pd.to_numeric(
        lookup["national_pres_margin_dem"],
        errors="raise",
    )

    conflicting_years = (
        lookup.groupby("presidential_result_year")[
            "national_pres_margin_dem"
        ]
        .nunique(dropna=False)
    )

    conflicting_years = conflicting_years[
        conflicting_years.gt(1)
    ]

    if not conflicting_years.empty:
        raise ValueError(
            "Conflicting national presidential margins for years: "
            f"{conflicting_years.index.tolist()}"
        )

    margin_by_year = (
        lookup.drop_duplicates(
            subset=["presidential_result_year"]
        )
        .set_index("presidential_result_year")[
            "national_pres_margin_dem"
        ]
    )

    result_year = pd.to_numeric(
        out["presidential_result_year"],
        errors="coerce",
    )

    out["national_pres_margin_dem"] = (
        result_year.map(margin_by_year)
    )

    missing_year_mask = (
        result_year.notna()
        & out["national_pres_margin_dem"].isna()
    )

    if missing_year_mask.any():
        missing_years = sorted(
            result_year.loc[missing_year_mask]
            .astype(int)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Missing national presidential margins for years: "
            f"{missing_years}"
        )

    district_margin = pd.to_numeric(
        out["district_pres_margin_dem"],
        errors="coerce",
    )

    out["district_partisan_baseline_dem"] = (
        district_margin
        - out["national_pres_margin_dem"]
    )

    return out
