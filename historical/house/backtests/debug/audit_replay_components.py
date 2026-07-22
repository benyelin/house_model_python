#!/usr/bin/env python3
"""
Audit which forecasting components are actually present in the House
production replay predictions.

This script does not modify any project files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]

REPLAY_PATH = (
    ROOT
    / "historical/house/backtests/outputs/production_replay_v1"
    / "house_production_replay_predictions.csv"
)

CYCLE = 2016
SPEC = "production_election_day_v1"


def heading(title: str) -> None:
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)


def boolean_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def describe_residual(
    frame: pd.DataFrame,
    label: str,
    expected: pd.Series,
) -> None:
    residual = frame["model_margin_dem_num"] - expected

    print()
    print(label)
    print("-" * len(label))
    print(f"Rows tested:             {int(residual.notna().sum())}")
    print(f"Mean residual:           {residual.mean():+.8f}")
    print(f"Median residual:         {residual.median():+.8f}")
    print(f"Maximum absolute error:  {residual.abs().max():.8f}")
    print(
        "Rows matching exactly:  "
        f"{int(np.isclose(residual, 0.0, atol=1e-8).sum())}"
        f" / {int(residual.notna().sum())}"
    )


def main() -> None:
    if not REPLAY_PATH.exists():
        raise SystemExit(
            f"Replay file not found:\n{REPLAY_PATH}"
        )

    frame = pd.read_csv(REPLAY_PATH, low_memory=False)

    print(f"Input: {REPLAY_PATH}")
    print(f"Rows in file: {len(frame)}")
    print(f"Columns in file: {len(frame.columns)}")

    heading("All replay columns")

    for column in frame.columns:
        print(column)

    data = frame.loc[
        pd.to_numeric(
            frame["cycle"],
            errors="coerce",
        ).eq(CYCLE)
    ].copy()

    if "replay_spec" in data.columns:
        data = data.loc[
            data["replay_spec"].astype(str).eq(SPEC)
        ].copy()

    if "include_in_scoring" in data.columns:
        data = data.loc[
            boolean_mask(data["include_in_scoring"])
        ].copy()

    if data.empty:
        raise SystemExit(
            f"No scorable rows found for {CYCLE}, spec={SPEC!r}."
        )

    required = [
        "model_margin_dem",
        "district_pres_margin_dem",
        "national_environment_margin_dem",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise SystemExit(
            "Missing required columns:\n  - "
            + "\n  - ".join(missing)
        )

    data["model_margin_dem_num"] = numeric(
        data,
        "model_margin_dem",
    )

    presidential = numeric(
        data,
        "district_pres_margin_dem",
    )

    environment = numeric(
        data,
        "national_environment_margin_dem",
    )

    heading("Core additive identity tests")

    describe_residual(
        data,
        "Presidential baseline only",
        presidential,
    )

    describe_residual(
        data,
        "Presidential baseline + national environment",
        presidential + environment,
    )

    optional_components = [
        "district_environment_adjustment_dem",
        "incumbency_adjustment_dem",
        "candidate_quality_adjustment_dem",
        "state_environment_adjustment_dem",
        "elasticity_adjustment_dem",
        "district_elasticity_adjustment_dem",
    ]

    running_expected = presidential + environment
    included_components: list[str] = []

    for column in optional_components:
        if column not in data.columns:
            continue

        values = numeric(data, column)

        print()
        print(
            f"{column}: "
            f"nonmissing={int(values.notna().sum())}, "
            f"nonzero={int(values.fillna(0).ne(0).sum())}, "
            f"mean={values.mean():+.6f}, "
            f"min={values.min():+.6f}, "
            f"max={values.max():+.6f}"
        )

        running_expected = running_expected + values.fillna(0)
        included_components.append(column)

        describe_residual(
            data,
            "Baseline + environment + "
            + " + ".join(included_components),
            running_expected,
        )

    heading("Potential production-component columns")

    keywords = [
        "incumb",
        "elastic",
        "candidate",
        "quality",
        "environment",
        "baseline",
        "fundamental",
        "adjustment",
        "margin",
    ]

    component_columns = [
        column
        for column in data.columns
        if any(
            keyword in column.lower()
            for keyword in keywords
        )
    ]

    for column in component_columns:
        values = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        if values.notna().any():
            print(
                f"{column}: "
                f"nonmissing={int(values.notna().sum())}, "
                f"nonzero={int(values.fillna(0).ne(0).sum())}, "
                f"mean={values.mean():+.6f}, "
                f"min={values.min():+.6f}, "
                f"max={values.max():+.6f}"
            )
        else:
            unique = (
                data[column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            print(
                f"{column}: categorical values="
                f"{unique[:15]}"
            )

    heading("Sample Republican-incumbent rows")

    incumbency_column = next(
        (
            column
            for column in [
                "incumbent_configuration",
                "incumbency_configuration",
                "incumbency_status",
            ]
            if column in data.columns
        ),
        None,
    )

    sample_columns = [
        "race_id",
        "state",
        incumbency_column,
        "district_pres_margin_dem",
        "national_environment_margin_dem",
        "model_margin_dem",
        "actual_dem_margin",
    ]

    sample_columns.extend(
        column
        for column in optional_components
        if column in data.columns
    )

    sample_columns = [
        column
        for column in dict.fromkeys(sample_columns)
        if column is not None and column in data.columns
    ]

    if incumbency_column is not None:
        republicans = data.loc[
            data[incumbency_column]
            .astype(str)
            .str.upper()
            .eq("R")
        ].copy()

        republicans["audit_margin_error"] = (
            numeric(republicans, "model_margin_dem")
            - numeric(republicans, "actual_dem_margin")
        )

        print(
            republicans.sort_values(
                "audit_margin_error",
                ascending=False,
            )[
                sample_columns
                + ["audit_margin_error"]
            ]
            .head(25)
            .to_string(
                index=False,
                float_format=lambda value: f"{value:.3f}",
            )
        )

    heading("Replay source-code search")

    search_terms = [
        "production_election_day_v1",
        "model_margin_dem",
        "incumbency_adjustment",
        "district_elasticity",
        "candidate_quality_adjustment",
    ]

    candidate_files = sorted(
        (
            ROOT
            / "historical/house/backtests"
        ).rglob("*.py")
    )

    for term in search_terms:
        print()
        print(f"TERM: {term}")
        print("-" * 92)

        matches = 0

        for path in candidate_files:
            try:
                lines = path.read_text(
                    encoding="utf-8"
                ).splitlines()
            except UnicodeDecodeError:
                continue

            for line_number, line in enumerate(
                lines,
                start=1,
            ):
                if term.lower() not in line.lower():
                    continue

                relative = path.relative_to(ROOT)

                print(
                    f"{relative}:{line_number}: "
                    f"{line.strip()}"
                )

                matches += 1

        if matches == 0:
            print("No matches found.")

    print()
    print("Replay component audit completed successfully.")


if __name__ == "__main__":
    main()
