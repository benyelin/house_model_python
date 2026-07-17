from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

import build_house_candidate_war as live_war


DEFAULT_RESULTS_PATH = (
    PROJECT_ROOT
    / "historical/house/warehouse/"
    "house_historical_results_2012_2022.csv"
)

DEFAULT_REGISTRY_DIR = (
    PROJECT_ROOT
    / "historical/house/processed"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical/house/backtests/outputs/"
    "candidate_war"
)

DEFAULT_FORECAST_CYCLES = (
    2016,
    2018,
    2020,
    2022,
)

US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE",
    "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS",
    "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY",
    "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY",
}


def parse_cycles(
    value: str,
) -> tuple[int, ...]:
    cycles: list[int] = []

    for item in value.split(","):
        item = item.strip()

        if not item:
            continue

        try:
            cycle = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid cycle: {item!r}"
            ) from exc

        cycles.append(cycle)

    if not cycles:
        raise argparse.ArgumentTypeError(
            "At least one forecast cycle is required."
        )

    return tuple(
        sorted(set(cycles))
    )


def parse_bool_series(
    series: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            {
                "true",
                "1",
                "yes",
                "y",
            }
        )
    )


def standardize_registry_party(
    value: object,
) -> str:
    text = str(value).strip().upper()

    if text in {
        "D",
        "DEM",
        "DEMOCRAT",
        "DEMOCRATIC",
    }:
        return "D"

    if text in {
        "R",
        "REP",
        "REPUBLICAN",
        "GOP",
    }:
        return "R"

    return text


def rebuild_cycle_weights(
    war: pd.DataFrame,
) -> pd.DataFrame:
    """
    Recompute the live model's configured cycle weights after
    applying the forecast-cycle cutoff.
    """
    out = war.copy()

    out[
        "war_cycle_weight"
    ] = (
        out["war_cycle"]
        .map(live_war.CYCLE_WEIGHTS)
        .fillna(0.05)
    )

    out[
        "war_weighted_score"
    ] = (
        out["war_score_raw"]
        * out["war_cycle_weight"]
    )

    return out


def aggregate_candidate_war(
    war: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate only leakage-safe prior observations.

    This reproduces the live recency-weighted candidate proxy while
    preserving the district histories needed by the matcher.
    """
    if war.empty:
        return pd.DataFrame(
            columns=[
                "war_candidate_norm",
                "war_party",
                "war_candidate_name",
                "war_score_weighted_sum",
                "war_weight_sum",
                "war_cycles",
                "war_observations",
                "war_latest_cycle",
                "war_latest_score",
                "war_district_ids",
                "candidate_war_recency_weighted",
            ]
        )

    ordered = war.sort_values(
        [
            "war_candidate_norm",
            "war_party",
            "war_cycle",
        ]
    ).copy()

    grouped = (
        ordered.groupby(
            [
                "war_candidate_norm",
                "war_party",
            ],
            as_index=False,
        )
        .agg(
            war_candidate_name=(
                "war_candidate_name",
                "last",
            ),
            war_score_weighted_sum=(
                "war_weighted_score",
                "sum",
            ),
            war_weight_sum=(
                "war_cycle_weight",
                "sum",
            ),
            war_cycles=(
                "war_cycle",
                lambda values: ",".join(
                    str(int(value))
                    for value in sorted(
                        set(values),
                        reverse=True,
                    )
                ),
            ),
            war_observations=(
                "war_score_raw",
                "count",
            ),
            war_latest_cycle=(
                "war_cycle",
                "max",
            ),
            war_latest_score=(
                "war_score_raw",
                "last",
            ),
            war_district_ids=(
                "war_district_id",
                lambda values: ";".join(
                    sorted(
                        {
                            str(value).strip()
                            for value in values
                            if str(value).strip()
                        }
                    )
                ),
            ),
        )
    )

    grouped[
        "candidate_war_recency_weighted"
    ] = (
        grouped[
            "war_score_weighted_sum"
        ]
        / grouped[
            "war_weight_sum"
        ].replace(
            0,
            np.nan,
        )
    ).fillna(0.0)

    return grouped


def observation_multiplier(
    observations: int,
    prior_strength: float,
) -> float:
    """
    Optional empirical-Bayes-style reliability shrinkage.

    prior_strength = 0 reproduces the current formula.
    Larger values pull candidates with little history toward zero.
    """
    observations = max(
        int(observations),
        0,
    )

    if prior_strength <= 0:
        return 1.0

    return float(
        observations
        / (
            observations
            + prior_strength
        )
    )


def load_cycle_races(
    results: pd.DataFrame,
    cycle: int,
) -> pd.DataFrame:
    races = results.loc[
        results["cycle"].eq(cycle)
        & results["state"].isin(US_STATES)
    ].copy()

    if len(races) != 435:
        raise ValueError(
            f"Expected 435 canonical races for {cycle}; "
            f"found {len(races)}."
        )

    if races["race_id"].duplicated().any():
        duplicates = (
            races.loc[
                races["race_id"].duplicated(
                    keep=False
                ),
                "race_id",
            ]
            .tolist()
        )

        raise ValueError(
            f"Duplicate race IDs in {cycle}: "
            + ", ".join(duplicates)
        )

    return races


def load_cycle_registry(
    registry_dir: Path,
    cycle: int,
) -> pd.DataFrame:
    path = (
        registry_dir
        / f"house_candidate_registry_{cycle}.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing candidate registry for {cycle}: {path}"
        )

    registry = pd.read_csv(
        path,
        dtype={
            "race_id": str,
            "party_code": str,
        },
    )

    required = {
        "race_id",
        "party_code",
        "candidate_name_mit",
        "incumbent_challenger_status",
        "match_method",
    }

    missing = sorted(
        required
        - set(registry.columns)
    )

    if missing:
        raise ValueError(
            f"{cycle} registry is missing columns: "
            + ", ".join(missing)
        )

    registry[
        "party_standardized"
    ] = registry[
        "party_code"
    ].apply(
        standardize_registry_party
    )

    registry = registry.loc[
        registry[
            "party_standardized"
        ].isin(
            {
                "D",
                "R",
            }
        )
    ].copy()

    return registry


def registry_party_record(
    registry: pd.DataFrame,
    race_id: str,
    party: str,
    expected_candidate: object,
) -> pd.Series | None:
    pool = registry.loc[
        registry["race_id"].eq(race_id)
        & registry[
            "party_standardized"
        ].eq(party)
    ].copy()

    if pool.empty:
        return None

    expected_norm = live_war.normalize_name(
        expected_candidate
    )

    pool[
        "expected_name_match"
    ] = pool[
        "candidate_name_mit"
    ].apply(
        lambda value: live_war.name_similarity(
            expected_norm,
            live_war.normalize_name(value),
        )
        if expected_norm
        else 0.0
    )

    method_priority = {
        "manual_override": 5,
        "high_confidence_automatic": 4,
        "high_confidence_state_fallback": 3,
        "probable_automatic": 2,
        "probable_state_fallback": 1,
        "review_required": 0,
        "unmatched": -1,
    }

    pool[
        "method_priority"
    ] = (
        pool["match_method"]
        .map(method_priority)
        .fillna(-2)
    )

    pool = pool.sort_values(
        [
            "expected_name_match",
            "method_priority",
        ],
        ascending=[
            False,
            False,
        ],
    )

    return pool.iloc[0]


def derive_incumbent_party(
    race: pd.Series,
    registry: pd.DataFrame,
) -> tuple[
    str,
    str,
    str,
]:
    dem_record = registry_party_record(
        registry=registry,
        race_id=str(race["race_id"]),
        party="D",
        expected_candidate=race.get(
            "dem_candidate",
            "",
        ),
    )

    gop_record = registry_party_record(
        registry=registry,
        race_id=str(race["race_id"]),
        party="R",
        expected_candidate=race.get(
            "gop_candidate",
            "",
        ),
    )

    dem_status = (
        str(
            dem_record.get(
                "incumbent_challenger_status",
                "",
            )
        ).strip().upper()
        if dem_record is not None
        else ""
    )

    gop_status = (
        str(
            gop_record.get(
                "incumbent_challenger_status",
                "",
            )
        ).strip().upper()
        if gop_record is not None
        else ""
    )

    if dem_status == "I" and gop_status != "I":
        incumbent_party = "D"
    elif gop_status == "I" and dem_status != "I":
        incumbent_party = "R"
    elif dem_status == "I" and gop_status == "I":
        incumbent_party = "BOTH"
    else:
        incumbent_party = ""

    return (
        incumbent_party,
        dem_status,
        gop_status,
    )


def build_cycle_audit(
    races: pd.DataFrame,
    registry: pd.DataFrame,
    war_source: pd.DataFrame,
    aliases: pd.DataFrame,
    forecast_cycle: int,
    shrinkage: float,
    cap: float,
    incumbent_discount: float,
    one_sided_multiplier: float,
    min_name_score: float,
    observation_prior_strength: float,
) -> tuple[
    pd.DataFrame,
    dict[str, object],
]:
    eligible_war = war_source.loc[
        war_source[
            "war_cycle"
        ].lt(forecast_cycle)
    ].copy()

    future_rows_excluded = int(
        war_source[
            "war_cycle"
        ].ge(forecast_cycle)
        .sum()
    )

    eligible_war = rebuild_cycle_weights(
        eligible_war
    )

    war_agg = aggregate_candidate_war(
        eligible_war
    )

    audit_rows: list[
        dict[str, object]
    ] = []

    for _, race in races.iterrows():
        district_id = (
            live_war.normalize_district_id(
                race.get(
                    "race_id",
                    "",
                )
            )
        )

        dem_candidate = race.get(
            "dem_candidate",
            "",
        )

        gop_candidate = race.get(
            "gop_candidate",
            "",
        )

        (
            incumbent_party,
            dem_registry_status,
            gop_registry_status,
        ) = derive_incumbent_party(
            race=race,
            registry=registry,
        )

        dem_match = live_war.match_candidate(
            dem_candidate,
            "D",
            war_agg,
            min_name_score,
            district_id,
            aliases,
        )

        gop_match = live_war.match_candidate(
            gop_candidate,
            "R",
            war_agg,
            min_name_score,
            district_id,
            aliases,
        )

        dem_war = (
            float(
                dem_match[
                    "candidate_war_recency_weighted"
                ]
            )
            if dem_match is not None
            else 0.0
        )

        gop_war = (
            float(
                gop_match[
                    "candidate_war_recency_weighted"
                ]
            )
            if gop_match is not None
            else 0.0
        )

        dem_observations = (
            int(
                dem_match[
                    "war_observations"
                ]
            )
            if dem_match is not None
            else 0
        )

        gop_observations = (
            int(
                gop_match[
                    "war_observations"
                ]
            )
            if gop_match is not None
            else 0
        )

        dem_observation_multiplier = (
            observation_multiplier(
                dem_observations,
                observation_prior_strength,
            )
        )

        gop_observation_multiplier = (
            observation_multiplier(
                gop_observations,
                observation_prior_strength,
            )
        )

        dem_war_after_observation_shrinkage = (
            dem_war
            * dem_observation_multiplier
        )

        gop_war_after_observation_shrinkage = (
            gop_war
            * gop_observation_multiplier
        )

        dem_incumbent_multiplier = (
            incumbent_discount
            if incumbent_party == "D"
            else 1.0
        )

        gop_incumbent_multiplier = (
            incumbent_discount
            if incumbent_party == "R"
            else 1.0
        )

        dem_effective_war = (
            dem_war_after_observation_shrinkage
            * dem_incumbent_multiplier
        )

        gop_effective_war = (
            gop_war_after_observation_shrinkage
            * gop_incumbent_multiplier
        )

        raw_net_dem = (
            dem_effective_war
            - gop_effective_war
        )

        shrunk_net_dem = (
            raw_net_dem
            * shrinkage
        )

        capped_before_match_quality = float(
            np.clip(
                shrunk_net_dem,
                -cap,
                cap,
            )
        )

        if (
            dem_match is not None
            and gop_match is not None
        ):
            match_status = "Both matched"
            match_quality_multiplier = 1.0
        elif dem_match is not None:
            match_status = "Only D matched"
            match_quality_multiplier = (
                one_sided_multiplier
            )
        elif gop_match is not None:
            match_status = "Only R matched"
            match_quality_multiplier = (
                one_sided_multiplier
            )
        else:
            match_status = "Neither matched"
            match_quality_multiplier = 0.0

        final_adjustment = (
            capped_before_match_quality
            * match_quality_multiplier
        )

        audit_rows.append(
            {
                "forecast_cycle": forecast_cycle,
                "district_id": district_id,
                "race_id": race["race_id"],
                "state": race["state"],
                "district": race["district"],
                "dem_candidate": dem_candidate,
                "gop_candidate": gop_candidate,
                "incumbent_party": incumbent_party,
                "dem_registry_status": (
                    dem_registry_status
                ),
                "gop_registry_status": (
                    gop_registry_status
                ),
                "dem_war_matched": (
                    dem_match is not None
                ),
                "gop_war_matched": (
                    gop_match is not None
                ),
                "dem_war_name": (
                    dem_match[
                        "war_candidate_name"
                    ]
                    if dem_match is not None
                    else ""
                ),
                "gop_war_name": (
                    gop_match[
                        "war_candidate_name"
                    ]
                    if gop_match is not None
                    else ""
                ),
                "dem_name_match_score": (
                    float(
                        dem_match[
                            "name_match_score"
                        ]
                    )
                    if dem_match is not None
                    else 0.0
                ),
                "gop_name_match_score": (
                    float(
                        gop_match[
                            "name_match_score"
                        ]
                    )
                    if gop_match is not None
                    else 0.0
                ),
                "dem_war_cycles": (
                    dem_match["war_cycles"]
                    if dem_match is not None
                    else ""
                ),
                "gop_war_cycles": (
                    gop_match["war_cycles"]
                    if gop_match is not None
                    else ""
                ),
                "dem_war_observations": (
                    dem_observations
                ),
                "gop_war_observations": (
                    gop_observations
                ),
                "dem_candidate_war": dem_war,
                "gop_candidate_war": gop_war,
                (
                    "dem_observation_"
                    "reliability_multiplier"
                ): dem_observation_multiplier,
                (
                    "gop_observation_"
                    "reliability_multiplier"
                ): gop_observation_multiplier,
                (
                    "dem_war_after_"
                    "observation_shrinkage"
                ): (
                    dem_war_after_observation_shrinkage
                ),
                (
                    "gop_war_after_"
                    "observation_shrinkage"
                ): (
                    gop_war_after_observation_shrinkage
                ),
                "dem_incumbent_multiplier": (
                    dem_incumbent_multiplier
                ),
                "gop_incumbent_multiplier": (
                    gop_incumbent_multiplier
                ),
                "dem_effective_war": (
                    dem_effective_war
                ),
                "gop_effective_war": (
                    gop_effective_war
                ),
                "candidate_war_net_dem": (
                    raw_net_dem
                ),
                "house_candidate_war_shrinkage": (
                    shrinkage
                ),
                "house_candidate_war_cap": cap,
                (
                    "house_candidate_war_"
                    "incumbent_discount"
                ): incumbent_discount,
                (
                    "house_candidate_war_"
                    "one_sided_multiplier"
                ): one_sided_multiplier,
                (
                    "house_candidate_war_"
                    "observation_prior_strength"
                ): observation_prior_strength,
                (
                    "candidate_war_adjustment_"
                    "dem_before_match_quality"
                ): capped_before_match_quality,
                (
                    "candidate_war_match_"
                    "quality_multiplier"
                ): match_quality_multiplier,
                "candidate_war_adjustment_dem": (
                    final_adjustment
                ),
                "war_match_status": (
                    match_status
                ),
                "maximum_war_cycle_used": (
                    int(
                        max(
                            [
                                value
                                for value in [
                                    (
                                        dem_match[
                                            "war_latest_cycle"
                                        ]
                                        if dem_match
                                        is not None
                                        else np.nan
                                    ),
                                    (
                                        gop_match[
                                            "war_latest_cycle"
                                        ]
                                        if gop_match
                                        is not None
                                        else np.nan
                                    ),
                                ]
                                if pd.notna(value)
                            ],
                            default=0,
                        )
                    )
                    or np.nan
                ),
                "future_war_used": False,
                "target_actual_dem_margin": (
                    race.get(
                        "actual_dem_margin",
                        np.nan,
                    )
                ),
                (
                    "target_include_in_major_"
                    "party_margin_scoring"
                ): (
                    race.get(
                        (
                            "include_in_major_"
                            "party_margin_scoring"
                        ),
                        False,
                    )
                ),
            }
        )

    audit = pd.DataFrame(
        audit_rows
    )

    summary = {
        "forecast_cycle": forecast_cycle,
        "source_war_rows": len(war_source),
        "eligible_prior_war_rows": len(
            eligible_war
        ),
        "future_or_same_cycle_rows_excluded": (
            future_rows_excluded
        ),
        "aggregated_candidate_records": len(
            war_agg
        ),
        "race_rows": len(audit),
        "both_matched": int(
            audit[
                "war_match_status"
            ].eq("Both matched").sum()
        ),
        "only_dem_matched": int(
            audit[
                "war_match_status"
            ].eq("Only D matched").sum()
        ),
        "only_gop_matched": int(
            audit[
                "war_match_status"
            ].eq("Only R matched").sum()
        ),
        "neither_matched": int(
            audit[
                "war_match_status"
            ].eq("Neither matched").sum()
        ),
        "nonzero_adjustments": int(
            audit[
                "candidate_war_adjustment_dem"
            ].abs().gt(0).sum()
        ),
        "mean_absolute_adjustment": float(
            audit[
                "candidate_war_adjustment_dem"
            ].abs().mean()
        ),
        "maximum_absolute_adjustment": float(
            audit[
                "candidate_war_adjustment_dem"
            ].abs().max()
        ),
        "future_war_rows_used": int(
            audit["future_war_used"].sum()
        ),
    }

    return audit, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-safe historical House candidate WAR "
            "audits using only observations prior to each "
            "forecast cycle."
        )
    )

    parser.add_argument(
        "--forecast-cycles",
        type=parse_cycles,
        default=DEFAULT_FORECAST_CYCLES,
    )

    parser.add_argument(
        "--results-path",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
    )

    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=DEFAULT_REGISTRY_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--shrinkage",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--cap",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--incumbent-discount",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--one-sided-multiplier",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--min-name-score",
        type=float,
        default=0.90,
    )

    parser.add_argument(
        "--observation-prior-strength",
        type=float,
        default=0.0,
        help=(
            "Additional shrinkage for candidates with few WAR "
            "observations. Zero reproduces the current formula."
        ),
    )

    args = parser.parse_args()

    if not args.results_path.exists():
        raise FileNotFoundError(
            f"Missing historical results: "
            f"{args.results_path}"
        )

    results = pd.read_csv(
        args.results_path,
        dtype={
            "race_id": str,
            "state": str,
            "district": str,
        },
    )

    results["cycle"] = pd.to_numeric(
        results["cycle"],
        errors="raise",
    ).astype(int)

    results[
        "include_in_major_party_margin_scoring"
    ] = parse_bool_series(
        results[
            "include_in_major_party_margin_scoring"
        ]
    )

    war_source = live_war.load_war()

    war_source[
        "war_cycle"
    ] = pd.to_numeric(
        war_source["war_cycle"],
        errors="raise",
    ).astype(int)

    aliases = live_war.load_candidate_aliases()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_frames: list[
        pd.DataFrame
    ] = []

    summaries: list[
        dict[str, object]
    ] = []

    failures: list[str] = []

    for forecast_cycle in args.forecast_cycles:
        races = load_cycle_races(
            results=results,
            cycle=forecast_cycle,
        )

        registry = load_cycle_registry(
            registry_dir=args.registry_dir,
            cycle=forecast_cycle,
        )

        audit, summary = build_cycle_audit(
            races=races,
            registry=registry,
            war_source=war_source,
            aliases=aliases,
            forecast_cycle=forecast_cycle,
            shrinkage=args.shrinkage,
            cap=args.cap,
            incumbent_discount=(
                args.incumbent_discount
            ),
            one_sided_multiplier=(
                args.one_sided_multiplier
            ),
            min_name_score=args.min_name_score,
            observation_prior_strength=(
                args.observation_prior_strength
            ),
        )

        if len(audit) != 435:
            failures.append(
                f"{forecast_cycle}: expected 435 rows; "
                f"found {len(audit)}."
            )

        leakage = audit.loc[
            audit[
                "maximum_war_cycle_used"
            ].notna()
            & audit[
                "maximum_war_cycle_used"
            ].ge(forecast_cycle)
        ]

        if not leakage.empty:
            failures.append(
                f"{forecast_cycle}: found "
                f"{len(leakage)} races using same-cycle "
                "or future WAR."
            )

        cycle_path = (
            args.output_dir
            / (
                f"house_{forecast_cycle}_"
                "candidate_war_audit.csv"
            )
        )

        audit.to_csv(
            cycle_path,
            index=False,
        )

        audit_frames.append(
            audit
        )

        summaries.append(
            summary
        )

        print()
        print(
            f"=== {forecast_cycle} HISTORICAL WAR ==="
        )
        print(
            audit[
                "war_match_status"
            ]
            .value_counts()
            .to_string()
        )
        print(
            "Nonzero adjustments:",
            summary[
                "nonzero_adjustments"
            ],
        )
        print(
            "Mean absolute adjustment:",
            f"{summary['mean_absolute_adjustment']:.6f}",
        )
        print(
            "Maximum absolute adjustment:",
            f"{summary['maximum_absolute_adjustment']:.6f}",
        )
        print(
            f"Wrote: {cycle_path}"
        )

    warehouse = pd.concat(
        audit_frames,
        ignore_index=True,
    )

    summary_frame = pd.DataFrame(
        summaries
    )

    warehouse_path = (
        args.output_dir
        / "house_historical_candidate_war.csv"
    )

    summary_path = (
        args.output_dir
        / "house_historical_candidate_war_summary.csv"
    )

    validation_path = (
        args.output_dir
        / "house_historical_candidate_war_validation.txt"
    )

    warehouse.to_csv(
        warehouse_path,
        index=False,
    )

    summary_frame.to_csv(
        summary_path,
        index=False,
    )

    report_lines = [
        "House Historical Candidate WAR Validation",
        "=" * 41,
        "",
        (
            "Forecast cycles: "
            + ", ".join(
                str(cycle)
                for cycle in args.forecast_cycles
            )
        ),
        f"Warehouse rows: {len(warehouse)}",
        (
            "Unique forecast-cycle/race rows: "
            f"{warehouse[['forecast_cycle', 'race_id']].drop_duplicates().shape[0]}"
        ),
        (
            "Rows using same-cycle or future WAR: "
            f"{int(warehouse['future_war_used'].sum())}"
        ),
        "",
        "Configured historical formula:",
        f"Shrinkage: {args.shrinkage:.4f}",
        f"Cap: {args.cap:.4f}",
        (
            "Incumbent discount: "
            f"{args.incumbent_discount:.4f}"
        ),
        (
            "One-sided multiplier: "
            f"{args.one_sided_multiplier:.4f}"
        ),
        (
            "Observation prior strength: "
            f"{args.observation_prior_strength:.4f}"
        ),
        "",
        "Cycle summary:",
        summary_frame.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        ),
        "",
        "Temporal rule:",
        (
            "Only WAR observations with war_cycle strictly "
            "less than forecast_cycle are eligible."
        ),
        "",
        "Interpretation note:",
        (
            "Candidate scores are proxies derived by assigning "
            "one-half of each published race-level WAR margin "
            "to each major-party candidate with opposite signs."
        ),
        "",
        "Validation status:",
    ]

    if failures:
        report_lines.append(
            "FAILED"
        )

        report_lines.extend(
            f"- {failure}"
            for failure in failures
        )
    else:
        report_lines.append(
            "PASSED"
        )

    report = "\n".join(
        report_lines
    )

    validation_path.write_text(
        report
    )

    if failures:
        raise RuntimeError(
            report
        )

    print()
    print(report)
    print()
    print(f"Wrote: {warehouse_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {validation_path}")


if __name__ == "__main__":
    main()
