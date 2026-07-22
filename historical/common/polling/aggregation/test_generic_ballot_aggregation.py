"""Smoke tests for the reusable generic-ballot aggregation engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from generic_ballot_aggregation import (
    GenericBallotAggregationSpec,
    aggregate_generic_ballot_snapshot,
    select_snapshot,
)


ROOT = Path(__file__).resolve().parents[4]

SNAPSHOT_PATH = (
    ROOT
    / "historical"
    / "common"
    / "polling"
    / "snapshots"
    / "generic_ballot_polling_snapshots.csv"
)


def main() -> None:
    print("Generic Ballot Aggregation Engine Smoke Test")
    print("=" * 64)

    snapshots = pd.read_csv(
        SNAPSHOT_PATH,
        low_memory=False,
    )

    specifications = {
        "all_equal_question": (
            GenericBallotAggregationSpec()
        ),
        "30d_equal_question": (
            GenericBallotAggregationSpec(
                lookback_days=30,
            )
        ),
        "30d_half_life_14_poll": (
            GenericBallotAggregationSpec(
                lookback_days=30,
                recency_mode="half_life",
                recency_half_life_days=14,
                duplicate_mode="poll",
            )
        ),
        "60d_weighted_quality": (
            GenericBallotAggregationSpec(
                lookback_days=60,
                recency_mode="half_life",
                recency_half_life_days=21,
                population_mode="weighted",
                sample_size_weighting=True,
                pollster_quality_weighting=True,
                partisan_mode="downweight",
                partisan_weight=0.50,
                duplicate_mode="poll",
            )
        ),
        "30d_lv_preferred": (
            GenericBallotAggregationSpec(
                lookback_days=30,
                population_mode="lv_preferred",
                duplicate_mode="poll",
            )
        ),
    }

    output_rows = []

    for cycle in [
        2018,
        2020,
        2022,
        2024,
    ]:
        snapshot = select_snapshot(
            snapshots,
            cycle=cycle,
            snapshot_days_before_election=0,
        )

        raw_mean = pd.to_numeric(
            snapshot["two_party_margin_dem"],
            errors="coerce",
        ).mean()

        equal_result = (
            aggregate_generic_ballot_snapshot(
                snapshot,
                specifications[
                    "all_equal_question"
                ],
            )
        )

        if not np.isclose(
            raw_mean,
            equal_result.estimate_margin_dem,
            atol=1e-12,
        ):
            raise AssertionError(
                "Equal-question aggregation does not "
                f"match raw mean for {cycle}: "
                f"{equal_result.estimate_margin_dem} "
                f"versus {raw_mean}"
            )

        for name, specification in (
            specifications.items()
        ):
            result = (
                aggregate_generic_ballot_snapshot(
                    snapshot,
                    specification,
                )
            )

            if not np.isfinite(
                result.estimate_margin_dem
            ):
                raise AssertionError(
                    f"Nonfinite estimate for "
                    f"{cycle}, {name}."
                )

            if result.total_weight <= 0:
                raise AssertionError(
                    f"Nonpositive weight for "
                    f"{cycle}, {name}."
                )

            if result.effective_sample_size <= 0:
                raise AssertionError(
                    f"Nonpositive effective sample "
                    f"size for {cycle}, {name}."
                )

            output_rows.append(
                {
                    "cycle": cycle,
                    "specification": name,
                    "estimate_margin_dem": (
                        result.estimate_margin_dem
                    ),
                    "input_questions": (
                        result.input_question_rows
                    ),
                    "retained_questions": (
                        result.retained_question_rows
                    ),
                    "aggregated_rows": (
                        result.aggregated_rows
                    ),
                    "unique_polls": (
                        result.unique_polls
                    ),
                    "unique_pollsters": (
                        result.unique_pollsters
                    ),
                    "effective_sample_size": (
                        result.effective_sample_size
                    ),
                    "weighted_mean_age": (
                        result
                        .weighted_mean_poll_age_days
                    ),
                }
            )

    output = pd.DataFrame(output_rows)

    print()
    print("Election Day estimates")
    print("-" * 64)
    print(
        output.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print("Validation")
    print("-" * 64)
    print(
        "Equal-question estimates match raw "
        "snapshot means: PASSED"
    )
    print(
        "All estimates and weights finite: PASSED"
    )
    print(
        "All effective sample sizes positive: PASSED"
    )
    print()
    print("Aggregation engine smoke test PASSED.")


if __name__ == "__main__":
    main()
