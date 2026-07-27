from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = (
    ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
    / "dynamic_uncertainty_audit"
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_house_dynamic_uncertainty as dynamic
import run_house_model as production


DAYS_OUT_VALUES = [
    365,
    240,
    180,
    150,
    120,
    90,
    60,
    30,
    29,
    14,
    7,
    0,
]

VALIDATED_ELECTION_DAY_TOTAL_SD = 7.994627
TOLERANCE = 0.05


def find_production_config():
    """
    Locate the no-argument production configuration class without assuming
    its exact class name.
    """
    candidates = []

    for name, obj in vars(production).items():
        if not inspect.isclass(obj):
            continue

        try:
            instance = obj()
        except Exception:
            continue

        required = [
            "total_error_sd",
            "national_error_share",
        ]

        if all(hasattr(instance, attr) for attr in required):
            candidates.append((name, instance))

    if not candidates:
        raise RuntimeError(
            "Could not locate the production model configuration class."
        )

    if len(candidates) > 1:
        print(
            "WARNING: Multiple production configuration candidates found:",
            [name for name, _ in candidates],
        )

    return candidates[0]


def production_total_sd(base_total_sd: float, days_out: int) -> float:
    """
    Reproduce the current run_house_model.py time adjustment exactly.
    """
    total_sd = float(base_total_sd)

    if days_out > 180:
        total_sd *= 1.15
    elif days_out > 120:
        total_sd *= 1.05
    elif days_out < 30:
        total_sd *= 0.80

    return total_sd


def is_non_decreasing_with_days_out(series: pd.Series) -> bool:
    """
    Uncertainty should be equal or larger farther from Election Day.

    DAYS_OUT_VALUES are later sorted ascending before this check.
    """
    differences = series.diff().dropna()
    return bool((differences >= -1e-12).all())


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    config_name, config = find_production_config()
    settings = dynamic.read_settings()

    base_total_sd = float(config.total_error_sd)
    national_share = float(config.national_error_share)

    rows = []

    for days_out in DAYS_OUT_VALUES:
        (
            dynamic_national_sd,
            dynamic_region_sd,
            dynamic_demographic_sd,
            dynamic_district_sd,
        ) = dynamic.get_dynamic_sds(settings, days_out)

        dynamic_component_total_sd = math.sqrt(
            dynamic_national_sd ** 2
            + dynamic_region_sd ** 2
            + dynamic_demographic_sd ** 2
            + dynamic_district_sd ** 2
        )

        current_production_total_sd = production_total_sd(
            base_total_sd,
            days_out,
        )

        current_production_national_sd = (
            current_production_total_sd * national_share
        )

        current_production_non_national_sd = math.sqrt(
            max(
                0.0,
                current_production_total_sd ** 2
                - current_production_national_sd ** 2,
            )
        )

        rows.append(
            {
                "days_out": days_out,
                "production_base_total_sd": base_total_sd,
                "production_total_sd": current_production_total_sd,
                "production_national_error_share": national_share,
                "production_national_sd": current_production_national_sd,
                "production_implied_non_national_sd": (
                    current_production_non_national_sd
                ),
                "dynamic_national_sd": dynamic_national_sd,
                "dynamic_region_sd": dynamic_region_sd,
                "dynamic_demographic_sd": dynamic_demographic_sd,
                "dynamic_district_sd": dynamic_district_sd,
                "dynamic_component_total_sd": dynamic_component_total_sd,
                "dynamic_minus_production_total_sd": (
                    dynamic_component_total_sd
                    - current_production_total_sd
                ),
            }
        )

    audit = pd.DataFrame(rows).sort_values(
        "days_out",
        ascending=False,
    )

    audit.to_csv(
        OUTPUT / "house_dynamic_uncertainty_schedule.csv",
        index=False,
    )

    ascending = audit.sort_values("days_out").reset_index(drop=True)

    monotonic_checks = []

    for column in [
        "production_total_sd",
        "dynamic_national_sd",
        "dynamic_region_sd",
        "dynamic_demographic_sd",
        "dynamic_district_sd",
        "dynamic_component_total_sd",
    ]:
        monotonic_checks.append(
            {
                "metric": column,
                "uncertainty_non_decreasing_with_days_out": (
                    is_non_decreasing_with_days_out(ascending[column])
                ),
                "minimum": float(ascending[column].min()),
                "maximum": float(ascending[column].max()),
            }
        )

    monotonic = pd.DataFrame(monotonic_checks)

    monotonic.to_csv(
        OUTPUT / "house_dynamic_uncertainty_monotonicity.csv",
        index=False,
    )

    election_day = audit.loc[audit["days_out"] == 0].iloc[0]

    dynamic_election_day_total = float(
        election_day["dynamic_component_total_sd"]
    )

    production_election_day_total = float(
        election_day["production_total_sd"]
    )

    summary = pd.DataFrame(
        [
            {
                "metric": "production_config_class",
                "value": config_name,
            },
            {
                "metric": "production_base_total_sd",
                "value": base_total_sd,
            },
            {
                "metric": "production_national_error_share",
                "value": national_share,
            },
            {
                "metric": "validated_election_day_total_sd",
                "value": VALIDATED_ELECTION_DAY_TOTAL_SD,
            },
            {
                "metric": "production_election_day_total_sd",
                "value": production_election_day_total,
            },
            {
                "metric": "production_election_day_difference",
                "value": (
                    production_election_day_total
                    - VALIDATED_ELECTION_DAY_TOTAL_SD
                ),
            },
            {
                "metric": "dynamic_election_day_component_total_sd",
                "value": dynamic_election_day_total,
            },
            {
                "metric": "dynamic_election_day_difference",
                "value": (
                    dynamic_election_day_total
                    - VALIDATED_ELECTION_DAY_TOTAL_SD
                ),
            },
            {
                "metric": "production_matches_validated_total",
                "value": (
                    abs(
                        production_election_day_total
                        - VALIDATED_ELECTION_DAY_TOTAL_SD
                    )
                    <= TOLERANCE
                ),
            },
            {
                "metric": "dynamic_matches_validated_total",
                "value": (
                    abs(
                        dynamic_election_day_total
                        - VALIDATED_ELECTION_DAY_TOTAL_SD
                    )
                    <= TOLERANCE
                ),
            },
        ]
    )

    summary.to_csv(
        OUTPUT / "house_dynamic_uncertainty_summary.csv",
        index=False,
    )

    print("=" * 88)
    print("HOUSE DYNAMIC UNCERTAINTY SCHEDULE AUDIT")
    print("=" * 88)

    display_columns = [
        "days_out",
        "production_total_sd",
        "production_national_sd",
        "dynamic_national_sd",
        "dynamic_region_sd",
        "dynamic_demographic_sd",
        "dynamic_district_sd",
        "dynamic_component_total_sd",
        "dynamic_minus_production_total_sd",
    ]

    print(
        audit[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:0.4f}",
        )
    )

    print("\nMonotonicity checks:")
    print(monotonic.to_string(index=False))

    print("\nElection Day comparison:")
    print(
        f"Validated replay total SD:    "
        f"{VALIDATED_ELECTION_DAY_TOTAL_SD:0.6f}"
    )
    print(
        f"Current production total SD: "
        f"{production_election_day_total:0.6f}"
    )
    print(
        f"Dynamic component total SD:  "
        f"{dynamic_election_day_total:0.6f}"
    )

    print("\nImportant interpretation:")
    print(
        "The dynamic component total is calculated as the square root "
        "of the summed component variances."
    )
    print(
        "A mismatch does not automatically mean the dynamic schedule is "
        "wrong. It may mean that its components were designed as a "
        "correlation allocation rather than independent additive errors."
    )
    print(
        "Do not promote the dynamic schedule into production until that "
        "variance-allocation interpretation is confirmed."
    )

    print("\nWrote:")
    print(OUTPUT / "house_dynamic_uncertainty_schedule.csv")
    print(OUTPUT / "house_dynamic_uncertainty_monotonicity.csv")
    print(OUTPUT / "house_dynamic_uncertainty_summary.csv")


if __name__ == "__main__":
    main()
