#!/usr/bin/env python3
"""
Build the canonical historical House race-level backtest input warehouse.

The builder combines validated, cycle-safe historical input
warehouses for:
    2016, 2018, 2020, and 2022

Learned District DNA and elasticity features are intentionally excluded
from this canonical file unless they are generated out of fold for the
target forecast cycle.

Canonical grain:
    one row per forecast_cycle × race_id

Authoritative base:
    historical/house/warehouse/house_historical_results_2012_2022.csv

Design principles:
    - validation first
    - deterministic, idempotent output
    - no silent row gain or loss
    - no many-to-many joins
    - boundary-aware feature selection
    - atomic output writes
    - preserve nonscorable races while explicitly flagging them
"""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


TARGET_CYCLES: tuple[int, ...] = (2016, 2018, 2020, 2022)

# Selected by the validated House incumbency sensitivity sweep:
# best combined rank, best RMSE, and near-best MAE.
INCUMBENCY_BONUS = 2.25

EXPECTED_ROWS_PER_CYCLE = 435
EXPECTED_TOTAL_ROWS = EXPECTED_ROWS_PER_CYCLE * len(TARGET_CYCLES)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_PATH = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "warehouse"
    / "house_historical_results_2012_2022.csv"
)

PRESIDENTIAL_BASELINE_PATH = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "warehouse"
    / "house_historical_presidential_baselines.csv"
)

NATIONAL_ENVIRONMENT_DIR = (
    PROJECT_ROOT
    / "historical"
    / "warehouse"
    / "processed"
    / "national_environment"
)

CANDIDATE_REGISTRY_DIR = (
    PROJECT_ROOT
    / "historical"
    / "warehouse"
    / "processed"
    / "candidates"
)

HOUSE_WAREHOUSE_DIR = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "warehouse"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "warehouse"
    / "house_historical_backtest_inputs_2016_2022.csv"
)

HISTORICAL_ELASTICITY_PATH = (
    PROJECT_ROOT
    / "historical"
    / "house"
    / "elasticity"
    / "house_historical_district_elasticity_2016_2022.csv"
)


class ValidationError(RuntimeError):
    """Raised when a warehouse or canonical merge violates its contract."""


@dataclass(frozen=True)
class WarehouseCandidate:
    path: Path
    columns: frozenset[str]


def log(message: str = "") -> None:
    print(message, flush=True)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {path}")


def read_csv(path: Path, label: str) -> pd.DataFrame:
    require_file(path, label)

    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise RuntimeError(f"Could not read {label} at {path}: {exc}") from exc

    if frame.empty:
        raise ValidationError(f"{label} is empty: {path}")

    return frame


def require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    label: str,
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValidationError(
            f"{label} is missing required columns: {missing}"
        )


def normalize_state(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper()


def normalize_district_value(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip().upper()

    if text in {"AL", "AT LARGE", "AT-LARGE", "00", "0.0"}:
        return "AL"

    try:
        numeric = float(text)
    except ValueError:
        return text

    if numeric.is_integer():
        integer = int(numeric)
        return "AL" if integer == 0 else str(integer)

    return text


def normalize_district(series: pd.Series) -> pd.Series:
    return series.map(normalize_district_value).astype("string")


def make_race_id(state: pd.Series, district: pd.Series) -> pd.Series:
    return normalize_state(state) + "-" + normalize_district(district)


def normalize_race_keys(
    frame: pd.DataFrame,
    label: str,
    *,
    cycle_column: str | None = None,
) -> pd.DataFrame:
    out = frame.copy()

    if "state" in out.columns:
        out["state"] = normalize_state(out["state"])

    if "district" in out.columns:
        out["district"] = normalize_district(out["district"])

    if "race_id" in out.columns:
        out["race_id"] = (
            out["race_id"]
            .astype("string")
            .str.strip()
            .str.upper()
            .str.replace(r"\.0$", "", regex=True)
        )
    elif {"state", "district"}.issubset(out.columns):
        out["race_id"] = make_race_id(out["state"], out["district"])
    else:
        raise ValidationError(
            f"{label} must contain race_id or both state and district."
        )

    if cycle_column is not None:
        if cycle_column not in out.columns:
            raise ValidationError(
                f"{label} does not contain cycle column {cycle_column!r}."
            )

        out[cycle_column] = pd.to_numeric(
            out[cycle_column],
            errors="raise",
        ).astype(int)

    return out


def assert_unique(
    frame: pd.DataFrame,
    keys: Sequence[str],
    label: str,
) -> None:
    duplicated = frame.duplicated(list(keys), keep=False)

    if duplicated.any():
        examples = (
            frame.loc[duplicated, list(keys)]
            .sort_values(list(keys))
            .head(20)
            .to_dict("records")
        )
        raise ValidationError(
            f"{label} has duplicate keys {list(keys)}. "
            f"Examples: {examples}"
        )


def assert_no_null_keys(
    frame: pd.DataFrame,
    keys: Sequence[str],
    label: str,
) -> None:
    null_mask = frame[list(keys)].isna().any(axis=1)

    for key in keys:
        if pd.api.types.is_string_dtype(frame[key]):
            null_mask |= frame[key].astype("string").str.strip().eq("")

    if null_mask.any():
        examples = frame.loc[null_mask, list(keys)].head(20).to_dict("records")
        raise ValidationError(
            f"{label} has null or blank join keys. Examples: {examples}"
        )


def assert_expected_cycle_counts(
    frame: pd.DataFrame,
    cycle_column: str,
    label: str,
) -> None:
    counts = (
        frame.groupby(cycle_column, dropna=False)
        .size()
        .reindex(TARGET_CYCLES, fill_value=0)
    )

    invalid = counts[counts != EXPECTED_ROWS_PER_CYCLE]

    if not invalid.empty:
        raise ValidationError(
            f"{label} must have {EXPECTED_ROWS_PER_CYCLE} rows per cycle. "
            f"Observed counts: {counts.to_dict()}"
        )

    unexpected = sorted(
        set(pd.to_numeric(frame[cycle_column], errors="coerce").dropna().astype(int))
        - set(TARGET_CYCLES)
    )

    if unexpected:
        raise ValidationError(
            f"{label} contains unexpected cycles: {unexpected}"
        )


def validated_left_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on: Sequence[str],
    label: str,
    require_complete: bool = True,
) -> pd.DataFrame:
    before_rows = len(left)

    assert_unique(right, on, label)
    assert_no_null_keys(right, on, label)

    indicator = f"__merge_{label.lower().replace(' ', '_')}"

    merged = left.merge(
        right,
        how="left",
        on=list(on),
        validate="one_to_one",
        indicator=indicator,
        sort=False,
    )

    after_rows = len(merged)

    log(
        f"{label}: rows before={before_rows:,}; "
        f"rows after={after_rows:,}; difference={after_rows - before_rows:+,}"
    )

    if after_rows != before_rows:
        raise ValidationError(
            f"{label} changed row count from {before_rows} to {after_rows}."
        )

    unmatched = merged[indicator].ne("both")

    if require_complete and unmatched.any():
        examples = (
            merged.loc[unmatched, list(on)]
            .head(20)
            .to_dict("records")
        )
        raise ValidationError(
            f"{label} failed to match {int(unmatched.sum())} canonical rows. "
            f"Examples: {examples}"
        )

    merged.drop(columns=[indicator], inplace=True)
    return merged


def load_results() -> pd.DataFrame:
    results = read_csv(RESULTS_PATH, "historical House results")

    required = {
        "cycle",
        "chamber",
        "race_id",
        "state",
        "district",
        "actual_dem_margin",
        "actual_winner",
        "major_party_contested",
        "general_election_party_structure",
        "include_in_major_party_margin_scoring",
    }
    require_columns(results, required, "historical House results")

    results = normalize_race_keys(
        results,
        "historical House results",
        cycle_column="cycle",
    )

    results = results.loc[results["cycle"].isin(TARGET_CYCLES)].copy()

    chamber_values = (
        results["chamber"]
        .astype("string")
        .str.strip()
        .str.lower()
        .dropna()
        .unique()
        .tolist()
    )

    if chamber_values != ["house"]:
        raise ValidationError(
            f"Historical results contain unexpected chamber values: "
            f"{chamber_values}"
        )

    assert_no_null_keys(results, ["cycle", "race_id"], "historical results")
    assert_unique(results, ["cycle", "race_id"], "historical results")
    assert_expected_cycle_counts(results, "cycle", "historical results")

    if len(results) != EXPECTED_TOTAL_ROWS:
        raise ValidationError(
            f"Historical results expected {EXPECTED_TOTAL_ROWS} rows; "
            f"found {len(results)}."
        )

    reconstructed = make_race_id(results["state"], results["district"])
    mismatched = results["race_id"].ne(reconstructed)

    if mismatched.any():
        examples = results.loc[
            mismatched,
            ["cycle", "race_id", "state", "district"],
        ].head(20)
        raise ValidationError(
            "Historical results contain race_id values inconsistent with "
            f"state/district. Examples: {examples.to_dict('records')}"
        )

    results["forecast_cycle"] = results["cycle"].astype(int)

    return results


def load_presidential_baselines() -> pd.DataFrame:
    historical = read_csv(
        PRESIDENTIAL_BASELINE_PATH,
        "historical presidential baseline warehouse",
    )

    required = {
        "forecast_cycle",
        "boundary_cycle",
        "presidential_result_year",
        "race_id",
        "state",
        "district",
        "district_pres_margin_dem",
        "presidential_result_available",
        "presidential_baseline_available",
        "boundary_compatibility",
        "baseline_selection_method",
    }
    require_columns(
        historical,
        required,
        "historical presidential baseline warehouse",
    )

    historical = normalize_race_keys(
        historical,
        "historical presidential baseline warehouse",
        cycle_column="forecast_cycle",
    )

    historical = historical.loc[
        historical["forecast_cycle"].isin((2016, 2018, 2020))
    ].copy()

    authoritative_2022_path = (
        PROJECT_ROOT
        / "historical"
        / "warehouse"
        / "processed"
        / "races"
        / "house_2022_presidential_baseline.csv"
    )

    authoritative_2022 = read_csv(
        authoritative_2022_path,
        "authoritative 2022 presidential baseline",
    )

    required_2022 = {
        "race_id",
        "district_pres_margin_dem",
        "presidential_result_year",
        "boundary_cycle",
        "boundary_compatibility",
        "include_in_2022_backtest",
        "source_organization",
        "source_local_path",
        "source_description",
    }
    require_columns(
        authoritative_2022,
        required_2022,
        "authoritative 2022 presidential baseline",
    )

    authoritative_2022 = normalize_race_keys(
        authoritative_2022,
        "authoritative 2022 presidential baseline",
    )

    if "state" not in authoritative_2022.columns:
        authoritative_2022["state"] = (
            authoritative_2022["race_id"]
            .astype("string")
            .str.split("-", n=1)
            .str[0]
        )

    if "district" not in authoritative_2022.columns:
        authoritative_2022["district"] = (
            authoritative_2022["race_id"]
            .astype("string")
            .str.split("-", n=1)
            .str[1]
        )

    authoritative_2022["forecast_cycle"] = 2022
    authoritative_2022["presidential_result_available"] = (
        authoritative_2022["district_pres_margin_dem"].notna()
    )
    authoritative_2022["presidential_baseline_available"] = (
        authoritative_2022["include_in_2022_backtest"]
        .fillna(False)
        .astype(bool)
        & authoritative_2022["district_pres_margin_dem"].notna()
    )
    authoritative_2022["baseline_selection_method"] = (
        "most_recent_completed_presidential_election_"
        "on_authoritative_2022_house_boundaries"
    )

    authoritative_2022["dem_presidential_candidate"] = "Joe Biden"
    authoritative_2022["gop_presidential_candidate"] = "Donald Trump"

    if "biden_vote_share" in authoritative_2022.columns:
        authoritative_2022["dem_presidential_share"] = (
            authoritative_2022["biden_vote_share"]
        )
    else:
        authoritative_2022["dem_presidential_share"] = pd.NA

    if "trump_vote_share" in authoritative_2022.columns:
        authoritative_2022["gop_presidential_share"] = (
            authoritative_2022["trump_vote_share"]
        )
    else:
        authoritative_2022["gop_presidential_share"] = pd.NA

    authoritative_2022["major_party_share_total"] = (
        pd.to_numeric(
            authoritative_2022["dem_presidential_share"],
            errors="coerce",
        )
        + pd.to_numeric(
            authoritative_2022["gop_presidential_share"],
            errors="coerce",
        )
    )

    authoritative_2022["source_name"] = (
        authoritative_2022["source_organization"]
    )
    authoritative_2022["source_file"] = (
        authoritative_2022["source_local_path"]
    )
    authoritative_2022["source_boundary_description"] = (
        authoritative_2022["source_description"]
    )
    authoritative_2022["warehouse_version"] = 1.0

    common_columns = [
        "forecast_cycle",
        "boundary_cycle",
        "presidential_result_year",
        "race_id",
        "state",
        "district",
        "dem_presidential_candidate",
        "gop_presidential_candidate",
        "dem_presidential_share",
        "gop_presidential_share",
        "district_pres_margin_dem",
        "major_party_share_total",
        "source_name",
        "source_file",
        "source_boundary_description",
        "warehouse_version",
        "presidential_result_available",
        "boundary_compatibility",
        "presidential_baseline_available",
        "baseline_selection_method",
    ]

    for column in common_columns:
        if column not in historical.columns:
            historical[column] = pd.NA
        if column not in authoritative_2022.columns:
            authoritative_2022[column] = pd.NA

    baseline = pd.concat(
        [
            historical[common_columns],
            authoritative_2022[common_columns],
        ],
        ignore_index=True,
    )

    baseline["forecast_cycle"] = pd.to_numeric(
        baseline["forecast_cycle"],
        errors="raise",
    ).astype(int)

    baseline["boundary_cycle"] = pd.to_numeric(
        baseline["boundary_cycle"],
        errors="raise",
    ).astype(int)

    baseline["presidential_result_year"] = pd.to_numeric(
        baseline["presidential_result_year"],
        errors="raise",
    ).astype(int)

    assert_unique(
        baseline,
        ["forecast_cycle", "race_id"],
        "combined historical presidential baseline warehouse",
    )
    assert_expected_cycle_counts(
        baseline,
        "forecast_cycle",
        "combined historical presidential baseline warehouse",
    )

    availability = (
        baseline["presidential_baseline_available"]
        .fillna(False)
        .astype(bool)
    )

    unavailable = baseline.loc[~availability].copy()

    if not unavailable.empty:
        expected_exception = (
            unavailable["forecast_cycle"].eq(2016)
            & unavailable["state"].eq("FL")
            & unavailable["district_pres_margin_dem"].isna()
            & unavailable["presidential_result_available"]
                .fillna(False)
                .eq(False)
        )

        if not expected_exception.all():
            unexpected = unavailable.loc[
                ~expected_exception,
                [
                    "forecast_cycle",
                    "race_id",
                    "state",
                    "district",
                    "boundary_compatibility",
                    "district_pres_margin_dem",
                    "presidential_result_available",
                    "baseline_selection_method",
                ],
            ].head(20)

            raise ValidationError(
                "Combined presidential baseline warehouse contains unexpected "
                "unavailable target-cycle baselines. "
                f"Examples: {unexpected.to_dict('records')}"
            )

        if len(unavailable) != 27:
            raise ValidationError(
                "Expected exactly 27 legitimate unavailable presidential "
                "baselines for Florida in 2016; "
                f"found {len(unavailable)}."
            )

        log(
            "Presidential baseline warehouse contains 27 documented "
            "Florida 2016 rows without a leakage-free pre-election baseline."
        )

    margin_missing = baseline["district_pres_margin_dem"].isna()

    if not margin_missing.equals(~availability):
        examples = baseline.loc[
            margin_missing.ne(~availability),
            [
                "forecast_cycle",
                "race_id",
                "district_pres_margin_dem",
                "presidential_baseline_available",
            ],
        ].head(20)

        raise ValidationError(
            "Presidential baseline availability flags are inconsistent with "
            f"margin presence. Examples: {examples.to_dict('records')}"
        )

    renamed = baseline[common_columns].rename(
        columns={
            "source_name": "presidential_baseline_source_name",
            "source_file": "presidential_baseline_source_file",
            "source_boundary_description":
                "presidential_baseline_source_boundary_description",
            "warehouse_version":
                "presidential_baseline_warehouse_version",
        }
    )

    # State and district are retained above for internal validation, but the
    # election-results base remains authoritative for canonical geography.
    renamed = renamed.drop(columns=["state", "district"])

    return renamed


def load_national_environment() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for cycle in TARGET_CYCLES:
        path = (
            NATIONAL_ENVIRONMENT_DIR
            / f"house_{cycle}_election_day_national_environment.csv"
        )
        frame = read_csv(path, f"{cycle} national environment snapshot")
        frame["__source_path"] = str(path.relative_to(PROJECT_ROOT))
        frames.append(frame)

    environment = pd.concat(frames, ignore_index=True)

    required = {
        "cycle",
        "snapshot_label",
        "as_of_date",
        "generic_ballot_margin_dem",
        "presidential_approval",
        "presidential_disapproval",
        "presidential_net_approval",
        "president_party",
        "national_environment_margin_dem",
        "formula_version",
    }
    require_columns(
        environment,
        required,
        "historical national environment snapshots",
    )

    environment["cycle"] = pd.to_numeric(
        environment["cycle"],
        errors="raise",
    ).astype(int)

    environment = environment.loc[
        environment["cycle"].isin(TARGET_CYCLES)
    ].copy()

    if len(environment) != len(TARGET_CYCLES):
        raise ValidationError(
            "National environment warehouse must contain exactly one row "
            f"for each target cycle. Found {len(environment)} rows."
        )

    assert_unique(
        environment,
        ["cycle"],
        "historical national environment snapshots",
    )

    labels = (
        environment["snapshot_label"]
        .astype("string")
        .str.strip()
        .str.lower()
    )
    if not labels.eq("election_day").all():
        raise ValidationError(
            "All historical national environment records must be "
            "election_day snapshots."
        )

    required_non_null = [
        "generic_ballot_margin_dem",
        "presidential_approval",
        "presidential_disapproval",
        "presidential_net_approval",
        "president_party",
        "national_environment_margin_dem",
    ]

    if environment[required_non_null].isna().any().any():
        missing = environment.loc[
            environment[required_non_null].isna().any(axis=1),
            ["cycle", *required_non_null],
        ]
        raise ValidationError(
            "National environment contains missing required values: "
            f"{missing.to_dict('records')}"
        )

    environment = environment.rename(
        columns={
            "cycle": "forecast_cycle",
            "snapshot_label": "environment_snapshot_label",
            "as_of_date": "environment_as_of_date",
            "formula_version": "environment_formula_version",
            "notes": "environment_notes",
            "__source_path": "environment_source_path",
        }
    )

    return environment


def candidate_registry_paths() -> list[Path]:
    search_roots = [
        PROJECT_ROOT / "historical" / "warehouse",
        PROJECT_ROOT / "historical" / "house",
    ]

    required_columns = {
        "cycle",
        "race_id",
        "party_code",
        "candidate_name_canonical",
        "is_incumbent",
    }

    discovered: list[tuple[Path, set[int]]] = []

    for root in search_roots:
        if not root.exists():
            continue

        for path in sorted(root.rglob("*.csv")):
            lowered = path.name.lower()

            if (
                ".before_" in lowered
                or "backup" in lowered
                or "validation" in lowered
                or "summary" in lowered
            ):
                continue

            try:
                columns = set(pd.read_csv(path, nrows=0).columns)
            except Exception:
                continue

            if not required_columns.issubset(columns):
                continue

            try:
                cycle_values = pd.read_csv(
                    path,
                    usecols=["cycle"],
                    low_memory=False,
                )["cycle"]

                cycles = set(
                    pd.to_numeric(cycle_values, errors="coerce")
                    .dropna()
                    .astype(int)
                    .tolist()
                )
            except Exception:
                continue

            relevant_cycles = cycles.intersection(TARGET_CYCLES)

            if relevant_cycles:
                discovered.append((path, relevant_cycles))

    if not discovered:
        raise FileNotFoundError(
            "Could not identify any candidate registry CSVs by schema."
        )

    coverage: dict[int, list[Path]] = {
        cycle: [] for cycle in TARGET_CYCLES
    }

    for registry_path, cycles in discovered:
        for cycle in cycles:
            coverage[cycle].append(registry_path)

    missing_cycles = [
        cycle
        for cycle, paths in coverage.items()
        if not paths
    ]

    if missing_cycles:
        inventory = {
            str(registry_path.relative_to(PROJECT_ROOT)):
                sorted(cycles)
            for registry_path, cycles in discovered
        }

        raise FileNotFoundError(
            "Candidate registry discovery did not cover all target cycles. "
            f"Missing cycles: {missing_cycles}. "
            f"Discovered registry inventory: {inventory}"
        )

    selected_paths: list[Path] = []

    for cycle in TARGET_CYCLES:
        cycle_paths = coverage[cycle]

        exact_cycle_paths = [
            registry_path
            for registry_path in cycle_paths
            if str(cycle) in registry_path.name
        ]

        if len(exact_cycle_paths) == 1:
            chosen = exact_cycle_paths[0]

        elif len(cycle_paths) == 1:
            chosen = cycle_paths[0]

        else:
            comparison_frames = []

            for registry_path in cycle_paths:
                frame = pd.read_csv(registry_path, low_memory=False)

                frame = frame.sort_values(
                    [
                        column
                        for column in (
                            "cycle",
                            "race_id",
                            "party_code",
                            "candidate_uid",
                            "candidate_name_canonical",
                        )
                        if column in frame.columns
                    ],
                    kind="mergesort",
                ).reset_index(drop=True)

                frame = frame.reindex(sorted(frame.columns), axis=1)
                comparison_frames.append((registry_path, frame))

            reference_path, reference_frame = comparison_frames[0]
            conflicting_paths = []

            for other_path, other_frame in comparison_frames[1:]:
                try:
                    pd.testing.assert_frame_equal(
                        reference_frame,
                        other_frame,
                        check_dtype=False,
                        check_like=True,
                    )
                except AssertionError:
                    conflicting_paths.append(other_path)

            if conflicting_paths:
                details = [
                    str(registry_path.relative_to(PROJECT_ROOT))
                    for registry_path in cycle_paths
                ]

                raise ValidationError(
                    f"Conflicting candidate registries discovered for "
                    f"{cycle}. Files are not equivalent: {details}"
                )

            preferred = [
                registry_path
                for registry_path in cycle_paths
                if (
                    PROJECT_ROOT
                    / "historical"
                    / "house"
                    / "processed"
                ) in registry_path.parents
            ]

            chosen = (
                preferred[0]
                if len(preferred) == 1
                else sorted(cycle_paths, key=lambda p: str(p))[0]
            )

            log(
                f"Collapsed {len(cycle_paths)} equivalent candidate "
                f"registries for {cycle}; using "
                f"{chosen.relative_to(PROJECT_ROOT)}"
            )

        selected_paths.append(chosen)

    unique_paths = list(dict.fromkeys(selected_paths))

    log("Discovered candidate registry files:")
    for registry_path in unique_paths:
        log(f"  {registry_path.relative_to(PROJECT_ROOT)}")

    return unique_paths


def select_party_candidate(group: pd.DataFrame) -> pd.Series:
    ranked = group.copy()

    for column, default in (
        ("is_incumbent", False),
        ("candidate_vote_total", 0),
        ("match_score", -np.inf),
    ):
        if column not in ranked.columns:
            ranked[column] = default

    ranked["is_incumbent"] = ranked["is_incumbent"].fillna(False).astype(bool)
    ranked["candidate_vote_total"] = pd.to_numeric(
        ranked["candidate_vote_total"],
        errors="coerce",
    ).fillna(0)
    ranked["match_score"] = pd.to_numeric(
        ranked["match_score"],
        errors="coerce",
    ).fillna(-np.inf)

    ranked = ranked.sort_values(
        ["candidate_vote_total", "is_incumbent", "match_score"],
        ascending=[False, False, False],
        kind="mergesort",
    )

    return ranked.iloc[0]


def aggregate_candidate_registries() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for path in candidate_registry_paths():
        frame = read_csv(path, f"candidate registry {path.name}")
        frame["candidate_registry_source_file"] = str(
            path.relative_to(PROJECT_ROOT)
        )
        frames.append(frame)

    candidates = pd.concat(frames, ignore_index=True)

    required = {
        "cycle",
        "race_id",
        "state",
        "district",
        "party_code",
        "candidate_uid",
        "candidate_name_canonical",
        "incumbent_challenger_status",
        "is_incumbent",
        "is_challenger",
        "is_open_seat_candidate",
    }
    require_columns(candidates, required, "candidate registries")

    candidates = normalize_race_keys(
        candidates,
        "candidate registries",
        cycle_column="cycle",
    )

    candidates = candidates.loc[
        candidates["cycle"].isin(TARGET_CYCLES)
    ].copy()

    candidates["party_code"] = (
        candidates["party_code"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    candidates = candidates.loc[
        candidates["party_code"].isin(["D", "R"])
    ].copy()

    if candidates.empty:
        raise ValidationError(
            "Candidate registries contain no Democratic or Republican rows."
        )

    selected_rows: list[pd.Series] = []

    for _, group in candidates.groupby(
        ["cycle", "race_id", "party_code"],
        sort=True,
        dropna=False,
    ):
        selected_rows.append(select_party_candidate(group))

    selected = pd.DataFrame(selected_rows).reset_index(drop=True)

    assert_unique(
        selected,
        ["cycle", "race_id", "party_code"],
        "selected major-party candidate registry",
    )

    value_columns = [
        "candidate_uid",
        "candidate_name_canonical",
        "candidate_name_mit",
        "candidate_name_fec",
        "fec_candidate_id",
        "principal_campaign_committee_id",
        "incumbent_challenger_status",
        "is_incumbent",
        "is_challenger",
        "is_open_seat_candidate",
        "fec_candidate_status",
        "candidate_vote_total",
        "candidate_vote_share",
        "match_method",
        "match_score",
        "match_reason",
        "candidate_registry_source_file",
    ]
    value_columns = [
        column for column in value_columns if column in selected.columns
    ]

    party_frames: list[pd.DataFrame] = []

    for party_code, prefix in (("D", "dem"), ("R", "gop")):
        party = selected.loc[
            selected["party_code"].eq(party_code),
            ["cycle", "race_id", *value_columns],
        ].copy()

        party = party.rename(
            columns={
                "cycle": "forecast_cycle",
                **{
                    column: f"{prefix}_{column}"
                    for column in value_columns
                },
            }
        )

        party_frames.append(party)

    race_level = party_frames[0].merge(
        party_frames[1],
        how="outer",
        on=["forecast_cycle", "race_id"],
        validate="one_to_one",
    )

    race_level["dem_incumbent"] = (
        race_level.get("dem_is_incumbent", False)
        .fillna(False)
        .astype(bool)
    )
    race_level["gop_incumbent"] = (
        race_level.get("gop_is_incumbent", False)
        .fillna(False)
        .astype(bool)
    )

    # Orient the validated incumbency advantage toward the Democratic margin.
    #
    # Democratic incumbent only -> +INCUMBENCY_BONUS
    # Republican incumbent only -> -INCUMBENCY_BONUS
    # Open seat or double-incumbent race -> 0
    race_level["incumbency_adjustment_dem"] = np.select(
        [
            race_level["dem_incumbent"]
            & ~race_level["gop_incumbent"],
            race_level["gop_incumbent"]
            & ~race_level["dem_incumbent"],
        ],
        [
            INCUMBENCY_BONUS,
            -INCUMBENCY_BONUS,
        ],
        default=0.0,
    ).astype(float)

    race_level["double_incumbent_race"] = (
        race_level["dem_incumbent"]
        & race_level["gop_incumbent"]
    )

    race_level["incumbent_configuration"] = np.select(
        [
            race_level["double_incumbent_race"],
            race_level["dem_incumbent"],
            race_level["gop_incumbent"],
        ],
        [
            "D_AND_R",
            "D",
            "R",
        ],
        default="OPEN",
    )

    # Retain incumbent_party as a compatibility field for existing
    # single-incumbent backtest code. Double-incumbent races must not be
    # mislabeled as belonging to only one incumbent party.
    race_level["incumbent_party"] = np.select(
        [
            race_level["double_incumbent_race"],
            race_level["dem_incumbent"],
            race_level["gop_incumbent"],
        ],
        [
            "BOTH",
            "D",
            "R",
        ],
        default="OPEN",
    )

    race_level["open_seat"] = (
        ~race_level["dem_incumbent"]
        & ~race_level["gop_incumbent"]
    )

    double_incumbent_count = int(
        race_level["double_incumbent_race"].sum()
    )

    if double_incumbent_count:
        examples = race_level.loc[
            race_level["double_incumbent_race"],
            [
                "forecast_cycle",
                "race_id",
                "dem_candidate_name_canonical",
                "gop_candidate_name_canonical",
            ],
        ].sort_values(
            ["forecast_cycle", "race_id"]
        )

        log(
            "Candidate aggregation identified "
            f"{double_incumbent_count} valid redistricting-driven "
            "double-incumbent races:"
        )

        for row in examples.itertuples(index=False):
            log(
                f"  {row.forecast_cycle} {row.race_id}: "
                f"{row.dem_candidate_name_canonical} (D) vs "
                f"{row.gop_candidate_name_canonical} (R)"
            )

    race_level["candidate_registry_available"] = True

    assert_unique(
        race_level,
        ["forecast_cycle", "race_id"],
        "race-level candidate aggregation",
    )

    return race_level


def inspect_csv_candidates(
    directory: Path,
    *,
    excluded_names: set[str] | None = None,
) -> list[WarehouseCandidate]:
    excluded_names = excluded_names or set()
    candidates: list[WarehouseCandidate] = []

    for path in sorted(directory.glob("*.csv")):
        if path.name in excluded_names:
            continue

        lowered = path.name.lower()

        if (
            ".before_" in lowered
            or "backup" in lowered
            or "provisional" in lowered
            or "validation" in lowered
            or "summary" in lowered
        ):
            continue

        try:
            columns = frozenset(pd.read_csv(path, nrows=0).columns)
        except Exception:
            continue

        candidates.append(WarehouseCandidate(path=path, columns=columns))

    return candidates


def choose_schema_file(
    *,
    label: str,
    required_any: Sequence[set[str]],
    preferred_name_terms: Sequence[str],
    excluded_names: set[str] | None = None,
) -> WarehouseCandidate:
    candidates = inspect_csv_candidates(
        HOUSE_WAREHOUSE_DIR,
        excluded_names=excluded_names,
    )

    matches = [
        candidate
        for candidate in candidates
        if any(required.issubset(candidate.columns) for required in required_any)
    ]

    if not matches:
        detail = [
            f"{candidate.path.name}: {sorted(candidate.columns)}"
            for candidate in candidates
        ]
        raise FileNotFoundError(
            f"Could not identify {label} by schema in {HOUSE_WAREHOUSE_DIR}. "
            f"Inspected files: {detail}"
        )

    def score(candidate: WarehouseCandidate) -> tuple[int, str]:
        name = candidate.path.name.lower()
        term_score = sum(
            1 for term in preferred_name_terms if term.lower() in name
        )
        return (-term_score, name)

    matches = sorted(matches, key=score)
    best = matches[0]
    best_score = -score(best)[0]

    equally_ranked = [
        candidate
        for candidate in matches
        if -score(candidate)[0] == best_score
    ]

    if len(equally_ranked) > 1:
        raise ValidationError(
            f"Ambiguous {label} discovery. Equally ranked files: "
            f"{[str(candidate.path) for candidate in equally_ranked]}"
        )

    log(f"Discovered {label}: {best.path.relative_to(PROJECT_ROOT)}")
    return best


def prepare_cycle_aware_feature(
    frame: pd.DataFrame,
    *,
    label: str,
    prefix: str,
) -> pd.DataFrame:
    out = frame.copy()

    cycle_source: str | None = None
    for candidate in ("forecast_cycle", "cycle", "boundary_cycle"):
        if candidate in out.columns:
            cycle_source = candidate
            break

    if cycle_source is None:
        raise ValidationError(
            f"{label} must contain forecast_cycle, cycle, or boundary_cycle."
        )

    out = normalize_race_keys(
        out,
        label,
        cycle_column=cycle_source,
    )

    if cycle_source != "forecast_cycle":
        out = out.rename(columns={cycle_source: "forecast_cycle"})

    out = out.loc[out["forecast_cycle"].isin(TARGET_CYCLES)].copy()

    assert_unique(out, ["forecast_cycle", "race_id"], label)
    assert_expected_cycle_counts(out, "forecast_cycle", label)

    protected = {"forecast_cycle", "race_id"}
    rename_map = {
        column: f"{prefix}_{column}"
        for column in out.columns
        if column not in protected
        and not column.startswith(f"{prefix}_")
    }

    return out.rename(columns=rename_map)


def load_district_dna() -> pd.DataFrame:
    candidate = choose_schema_file(
        label="historical District DNA warehouse",
        required_any=(
            {
                "race_id",
                "forecast_cycle",
            },
            {
                "race_id",
                "cycle",
            },
            {
                "race_id",
                "boundary_cycle",
                "region",
                "district_type",
            },
        ),
        preferred_name_terms=(
            "historical",
            "district",
            "dna",
            "character",
            "behavior",
        ),
        excluded_names={
            RESULTS_PATH.name,
            PRESIDENTIAL_BASELINE_PATH.name,
            OUTPUT_PATH.name,
            "house_enriched_district_characteristics.csv",
        },
    )

    dna = read_csv(candidate.path, "historical District DNA warehouse")

    return prepare_cycle_aware_feature(
        dna,
        label="historical District DNA warehouse",
        prefix="dna",
    )


def find_elasticity_column(columns: Iterable[str]) -> str | None:
    ordered = (
        "shrunk_elasticity",
        "district_elasticity",
        "elasticity_shrunk",
        "estimated_elasticity",
        "elasticity",
    )

    column_set = set(columns)

    for column in ordered:
        if column in column_set:
            return column

    return None


def load_district_elasticity() -> pd.DataFrame:
    candidates = inspect_csv_candidates(
        HOUSE_WAREHOUSE_DIR,
        excluded_names={
            RESULTS_PATH.name,
            PRESIDENTIAL_BASELINE_PATH.name,
            OUTPUT_PATH.name,
        },
    )

    matches = [
        candidate
        for candidate in candidates
        if "race_id" in candidate.columns
        and find_elasticity_column(candidate.columns) is not None
    ]

    if not matches:
        raise FileNotFoundError(
            "Could not identify a district elasticity warehouse by schema "
            f"in {HOUSE_WAREHOUSE_DIR}."
        )

    def score(candidate: WarehouseCandidate) -> tuple[int, str]:
        name = candidate.path.name.lower()
        score_value = (
            int("historical" in name)
            + int("district" in name)
            + int("elastic" in name)
            + int("shrunk" in name)
        )
        return (-score_value, name)

    matches = sorted(matches, key=score)

    if len(matches) > 1 and score(matches[0])[0] == score(matches[1])[0]:
        raise ValidationError(
            "Ambiguous district elasticity warehouse discovery. "
            f"Candidates: {[str(candidate.path) for candidate in matches]}"
        )

    chosen = matches[0]
    log(
        "Discovered district elasticity warehouse: "
        f"{chosen.path.relative_to(PROJECT_ROOT)}"
    )

    elasticity = read_csv(
        chosen.path,
        "historical district elasticity warehouse",
    )

    return prepare_cycle_aware_feature(
        elasticity,
        label="historical district elasticity warehouse",
        prefix="elasticity",
    )


def merge_historical_district_elasticity(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge leakage-safe, expanding-window district elasticities by
    forecast_cycle and race_id.

    The source table guarantees that every training_end_cycle precedes
    its forecast_cycle. Missing historical estimates use the explicit
    neutral elasticity fallback of 1.0.
    """
    require_file(
        HISTORICAL_ELASTICITY_PATH,
        "historical district elasticity warehouse",
    )

    elasticity = read_csv(
        HISTORICAL_ELASTICITY_PATH,
        "historical district elasticity warehouse",
    )

    required = {
        "forecast_cycle",
        "race_id",
        "district_elasticity",
        "training_end_cycle",
        "historical_elasticity_estimate_available",
        "historical_elasticity_used_neutral_fallback",
    }

    require_columns(
        elasticity,
        required,
        "historical district elasticity warehouse",
    )

    elasticity["forecast_cycle"] = pd.to_numeric(
        elasticity["forecast_cycle"],
        errors="coerce",
    )

    elasticity["training_end_cycle"] = pd.to_numeric(
        elasticity["training_end_cycle"],
        errors="coerce",
    )

    elasticity["district_elasticity"] = pd.to_numeric(
        elasticity["district_elasticity"],
        errors="coerce",
    )

    if elasticity[
        ["forecast_cycle", "race_id"]
    ].duplicated().any():
        raise ValidationError(
            "Historical district elasticity contains duplicate "
            "forecast_cycle/race_id keys."
        )

    if elasticity["training_end_cycle"].ge(
        elasticity["forecast_cycle"]
    ).any():
        raise ValidationError(
            "Historical district elasticity contains future "
            "training information."
        )

    if not elasticity["district_elasticity"].between(
        0.55,
        1.25,
        inclusive="both",
    ).all():
        raise ValidationError(
            "Historical district elasticity falls outside "
            "the production bounds of 0.55 to 1.25."
        )

    merge_columns = [
        column
        for column in elasticity.columns
        if column not in {"state", "district"}
    ]

    before_rows = len(frame)

    out = frame.merge(
        elasticity[merge_columns],
        on=["forecast_cycle", "race_id"],
        how="left",
        validate="one_to_one",
    )

    if len(out) != before_rows:
        raise ValidationError(
            "Historical elasticity merge changed the canonical row count."
        )

    if out["district_elasticity"].isna().any():
        missing = (
            out.loc[
                out["district_elasticity"].isna(),
                ["forecast_cycle", "race_id"],
            ]
            .head(20)
            .to_dict("records")
        )

        raise ValidationError(
            "Historical elasticity merge left missing district values. "
            f"Examples: {missing}"
        )

    out["district_elasticity_joined_in_canonical_inputs"] = True

    return out


def add_feature_availability_flags(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add canonical input-availability and eligibility flags.

    District DNA remains excluded unless generated out of fold.
    District elasticity is supplied by the dedicated leakage-safe,
    cycle-aware historical elasticity warehouse.
    """
    out = frame.copy()

    out["results_available"] = out["actual_winner"].notna()

    out["presidential_baseline_feature_available"] = (
        out["district_pres_margin_dem"].notna()
        & out["presidential_baseline_available"]
            .fillna(False)
            .astype(bool)
    )

    out["national_environment_feature_available"] = (
        out["national_environment_margin_dem"].notna()
    )

    out["candidate_registry_feature_available"] = (
        out["candidate_registry_available"]
        .fillna(False)
        .astype(bool)
    )

    out["district_dna_joined_in_canonical_inputs"] = False

    if "district_elasticity_joined_in_canonical_inputs" not in out.columns:
        out["district_elasticity_joined_in_canonical_inputs"] = False

    out["district_elasticity_joined_in_canonical_inputs"] = (
        out["district_elasticity_joined_in_canonical_inputs"]
        .fillna(False)
        .astype(bool)
    )

    out["learned_district_features_required_oof"] = (
        ~out["district_elasticity_joined_in_canonical_inputs"]
    )

    out["canonical_feature_set_complete"] = (
        out["results_available"]
        & out["national_environment_feature_available"]
        & out["candidate_registry_feature_available"]
    )

    out["include_in_presidential_baseline_backtest"] = (
        out["include_in_major_party_margin_scoring"]
        .fillna(False)
        .astype(bool)
        & out["presidential_baseline_feature_available"]
    )

    out["include_in_environment_backtest"] = (
        out["include_in_major_party_margin_scoring"]
        .fillna(False)
        .astype(bool)
        & out["national_environment_feature_available"]
    )

    out["include_in_incumbency_backtest"] = (
        out["include_in_major_party_margin_scoring"]
        .fillna(False)
        .astype(bool)
        & out["candidate_registry_feature_available"]
    )

    out["include_in_canonical_margin_backtest"] = (
        out["include_in_major_party_margin_scoring"]
        .fillna(False)
        .astype(bool)
        & out["canonical_feature_set_complete"]
        & out["presidential_baseline_feature_available"]
    )

    return out


def validate_numeric_ranges(frame: pd.DataFrame) -> None:
    margin_columns = [
        "actual_dem_margin",
        "district_pres_margin_dem",
        "generic_ballot_margin_dem",
        "national_environment_margin_dem",
    ]

    for column in margin_columns:
        if column not in frame.columns:
            continue

        numeric = pd.to_numeric(frame[column], errors="coerce")
        invalid = numeric.notna() & ~numeric.between(-100.0, 100.0)

        if invalid.any():
            examples = frame.loc[
                invalid,
                ["forecast_cycle", "race_id", column],
            ].head(20)
            raise ValidationError(
                f"{column} contains values outside [-100, 100]. "
                f"Examples: {examples.to_dict('records')}"
            )

    share_columns = [
        column
        for column in frame.columns
        if (
            column.endswith("_share")
            or column.endswith("_vote_share")
        )
    ]

    for column in share_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        non_null = numeric.dropna()

        if non_null.empty:
            continue

        # Historical warehouses use either proportions or percentages.
        upper_bound = 1.0 if non_null.max() <= 1.000001 else 100.0
        invalid = numeric.notna() & ~numeric.between(0.0, upper_bound)

        if invalid.any():
            examples = frame.loc[
                invalid,
                ["forecast_cycle", "race_id", column],
            ].head(20)
            raise ValidationError(
                f"{column} contains invalid share values. "
                f"Examples: {examples.to_dict('records')}"
            )


def validate_final(frame: pd.DataFrame) -> None:
    if len(frame) != EXPECTED_TOTAL_ROWS:
        raise ValidationError(
            f"Canonical output expected {EXPECTED_TOTAL_ROWS} rows; "
            f"found {len(frame)}."
        )

    assert_no_null_keys(
        frame,
        ["forecast_cycle", "race_id"],
        "canonical historical backtest inputs",
    )
    assert_unique(
        frame,
        ["forecast_cycle", "race_id"],
        "canonical historical backtest inputs",
    )
    assert_expected_cycle_counts(
        frame,
        "forecast_cycle",
        "canonical historical backtest inputs",
    )

    required_non_null = [
        "state",
        "district",
        "actual_winner",
        "general_election_party_structure",
        "include_in_major_party_margin_scoring",
        "national_environment_margin_dem",
        "environment_as_of_date",
    ]

    missing_required = [
        column for column in required_non_null if column not in frame.columns
    ]
    if missing_required:
        raise ValidationError(
            f"Canonical output is missing required columns: {missing_required}"
        )

    null_rows = frame[required_non_null].isna().any(axis=1)
    if null_rows.any():
        examples = frame.loc[
            null_rows,
            ["forecast_cycle", "race_id", *required_non_null],
        ].head(20)
        raise ValidationError(
            "Canonical output contains unexpected missing required fields. "
            f"Examples: {examples.to_dict('records')}"
        )

    winner_values = set(
        frame["actual_winner"]
        .astype("string")
        .str.strip()
        .str.upper()
        .dropna()
    )
    unexpected_winners = sorted(winner_values - {"D", "R", "TIE", "OTHER"})
    if unexpected_winners:
        raise ValidationError(
            f"Unexpected actual_winner values: {unexpected_winners}"
        )

    unavailable_presidential = frame.loc[
        ~frame["presidential_baseline_feature_available"]
    ].copy()

    if not unavailable_presidential.empty:
        expected_exception = (
            unavailable_presidential["forecast_cycle"].eq(2016)
            & unavailable_presidential["state"].eq("FL")
            & unavailable_presidential["district_pres_margin_dem"].isna()
            & unavailable_presidential["presidential_baseline_available"]
                .fillna(False)
                .eq(False)
        )

        if (
            len(unavailable_presidential) != 27
            or not expected_exception.all()
        ):
            examples = unavailable_presidential.loc[
                ~expected_exception,
                [
                    "forecast_cycle",
                    "race_id",
                    "state",
                    "district",
                    "district_pres_margin_dem",
                    "presidential_baseline_available",
                ],
            ].head(20)

            raise ValidationError(
                "Canonical output contains unexpected missing presidential "
                f"baselines. Examples: {examples.to_dict('records')}"
            )

    if not frame["national_environment_feature_available"].all():
        raise ValidationError(
            "National environment coverage is not complete."
        )

    if not frame["candidate_registry_feature_available"].all():
        missing = frame.loc[
            ~frame["candidate_registry_feature_available"],
            ["forecast_cycle", "race_id"],
        ].head(20)
        raise ValidationError(
            "Candidate registry coverage is not complete. "
            f"Examples: {missing.to_dict('records')}"
        )

    if not frame["canonical_feature_set_complete"].all():
        raise ValidationError(
            "At least one canonical row lacks a required feature warehouse."
        )

    validate_numeric_ranges(frame)


def order_columns(frame: pd.DataFrame) -> pd.DataFrame:
    leading = [
        "forecast_cycle",
        "cycle",
        "race_id",
        "district_id",
        "state",
        "district",
        "chamber",
        "election_date",
        "election_type",
        "actual_dem_margin",
        "actual_winner",
        "include_in_major_party_margin_scoring",
        "major_party_contested",
        "general_election_party_structure",
        "uncontested",
        "uncontested_dem",
        "uncontested_gop",
        "dem_candidate",
        "gop_candidate",
        "dem_vote_total",
        "gop_vote_total",
        "other_vote_total",
        "total_vote",
        "dem_vote_share",
        "gop_vote_share",
        "district_pres_margin_dem",
        "presidential_result_year",
        "boundary_cycle",
        "boundary_compatibility",
        "baseline_selection_method",
        "generic_ballot_margin_dem",
        "presidential_approval",
        "presidential_disapproval",
        "presidential_net_approval",
        "president_party",
        "national_environment_margin_dem",
        "environment_as_of_date",
        "environment_formula_version",
        "incumbent_configuration",
        "incumbent_party",
        "incumbency_adjustment_dem",
        "dem_incumbent",
        "gop_incumbent",
        "double_incumbent_race",
        "open_seat",
        "results_available",
        "presidential_baseline_feature_available",
        "national_environment_feature_available",
        "candidate_registry_feature_available",
        "district_dna_joined_in_canonical_inputs",
        "district_elasticity_joined_in_canonical_inputs",
        "learned_district_features_required_oof",
        "canonical_feature_set_complete",
        "include_in_presidential_baseline_backtest",
        "include_in_environment_backtest",
        "include_in_incumbency_backtest",
        "include_in_canonical_margin_backtest",
    ]

    leading = [column for column in leading if column in frame.columns]
    remaining = sorted(column for column in frame.columns if column not in leading)

    return frame[leading + remaining]


def atomic_write_csv(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(file_descriptor)

    temporary_path = Path(temporary_name)

    try:
        frame.to_csv(
            temporary_path,
            index=False,
            lineterminator="\n",
            float_format="%.10g",
        )
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def print_summary(frame: pd.DataFrame, output_path: Path) -> None:
    counts = (
        frame.groupby("forecast_cycle")
        .size()
        .sort_index()
        .to_dict()
    )

    scorable_counts = (
        frame.loc[
            frame["include_in_major_party_margin_scoring"]
            .fillna(False)
            .astype(bool)
        ]
        .groupby("forecast_cycle")
        .size()
        .reindex(TARGET_CYCLES, fill_value=0)
        .to_dict()
    )

    log()
    log("=" * 72)
    log("House Historical Backtest Inputs")
    log("=" * 72)
    log(f"Output: {output_path.relative_to(PROJECT_ROOT)}")
    log(f"Rows: {len(frame):,}")
    log(f"Columns: {len(frame.columns):,}")
    log()
    log("Rows by cycle:")
    for cycle in TARGET_CYCLES:
        log(f"  {cycle}: {counts.get(cycle, 0):,}")
    log()
    log("Scorable major-party races by cycle:")
    for cycle in TARGET_CYCLES:
        log(f"  {cycle}: {scorable_counts.get(cycle, 0):,}")
    log()
    presidential_available = int(
        frame["presidential_baseline_feature_available"].sum()
    )
    presidential_unavailable = len(frame) - presidential_available

    log(
        "Presidential baseline availability: "
        f"{presidential_available:,} / {len(frame):,} "
        f"({presidential_unavailable:,} documented unavailable)"
    )
    log(
        "National environment coverage: "
        f"{int(frame['national_environment_feature_available'].sum()):,}"
        f" / {len(frame):,}"
    )
    log(
        "District DNA in canonical inputs: 0 / "
        f"{len(frame):,} "
        "(intentionally deferred to leakage-safe OOF backtests)"
    )
    elasticity_joined = int(
        frame[
            "district_elasticity_joined_in_canonical_inputs"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    )
    log(
        "District elasticity in canonical inputs: "
        f"{elasticity_joined:,} / {len(frame):,} "
        "(merged from leakage-safe cycle-aware historical estimates)"
    )
    log(
        "Candidate registry coverage: "
        f"{int(frame['candidate_registry_feature_available'].sum()):,}"
        f" / {len(frame):,}"
    )
    log()
    log("Duplicate canonical keys: 0")
    log("Validation PASSED")
    log("=" * 72)


def build(output_path: Path) -> pd.DataFrame:
    log("Loading authoritative historical election results...")
    canonical = load_results()

    log(f"Canonical base rows: {len(canonical):,}")
    log()

    log("Loading historical presidential baselines...")
    presidential = load_presidential_baselines()
    canonical = validated_left_merge(
        canonical,
        presidential,
        on=["forecast_cycle", "race_id"],
        label="Presidential baseline merge",
    )
    log()

    log("Loading historical national environment snapshots...")
    environment = load_national_environment()
    canonical = canonical.merge(
        environment,
        how="left",
        on=["forecast_cycle"],
        validate="many_to_one",
        indicator="__environment_merge",
        sort=False,
    )

    unmatched_environment = canonical["__environment_merge"].ne("both")
    if unmatched_environment.any():
        examples = canonical.loc[
            unmatched_environment,
            ["forecast_cycle", "race_id"],
        ].head(20)
        raise ValidationError(
            "National environment failed to match canonical rows. "
            f"Examples: {examples.to_dict('records')}"
        )
    canonical.drop(columns=["__environment_merge"], inplace=True)
    log(
        "National environment merge: "
        f"rows before={EXPECTED_TOTAL_ROWS:,}; "
        f"rows after={len(canonical):,}; difference=+0"
    )
    log()

    log("Loading and aggregating historical candidate registries...")
    candidates = aggregate_candidate_registries()
    canonical = validated_left_merge(
        canonical,
        candidates,
        on=["forecast_cycle", "race_id"],
        label="Candidate registry merge",
    )
    log()

    log(
        "Deferring District DNA to leakage-safe out-of-fold backtests."
    )
    log(
        "Merging leakage-safe cycle-aware historical district elasticity."
    )
    log()

    canonical = merge_historical_district_elasticity(canonical)
    canonical = add_feature_availability_flags(canonical)
    canonical = canonical.sort_values(
        ["forecast_cycle", "state", "district", "race_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    canonical = order_columns(canonical)

    log("Running final canonical validations...")
    validate_final(canonical)

    log("Writing canonical warehouse atomically...")
    atomic_write_csv(canonical, output_path)

    print_summary(canonical, output_path)
    return canonical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the canonical historical House backtest input warehouse "
            "for 2016, 2018, 2020, and 2022."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output CSV path. Default: {OUTPUT_PATH}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_path = args.output
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    build(output_path.resolve())


if __name__ == "__main__":
    main()
