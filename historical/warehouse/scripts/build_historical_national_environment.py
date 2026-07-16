from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "historical/warehouse/processed/national_environment"
)

VALIDATION_DIR = (
    PROJECT_ROOT
    / "historical/warehouse/validation"
)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def approval_adjustment_dem(
    approval: float,
    president_party: str,
) -> float:
    republican_president_adjustment = clamp(
        (45.0 - approval) / 3.0,
        -3.0,
        3.0,
    )

    if president_party == "R":
        return republican_president_adjustment

    if president_party == "D":
        return -republican_president_adjustment

    raise ValueError(
        "president_party must be D or R."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a historical national-environment snapshot "
            "using the production reduced-double-count formula."
        )
    )

    parser.add_argument(
        "--cycle",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--as-of-date",
        required=True,
    )

    parser.add_argument(
        "--generic-ballot-dem-margin",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--approval",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--disapproval",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--president-party",
        choices=["D", "R"],
        required=True,
    )

    parser.add_argument(
        "--midterm-adjustment-dem",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--generic-source",
        required=True,
    )

    parser.add_argument(
        "--approval-source",
        required=True,
    )

    parser.add_argument(
        "--snapshot-label",
        default="election_day",
    )

    parser.add_argument(
        "--notes",
        default="",
    )

    args = parser.parse_args()

    if not 0.0 <= args.approval <= 100.0:
        raise ValueError(
            "Approval must be between 0 and 100."
        )

    if not 0.0 <= args.disapproval <= 100.0:
        raise ValueError(
            "Disapproval must be between 0 and 100."
        )

    approval_adjustment = approval_adjustment_dem(
        approval=args.approval,
        president_party=args.president_party,
    )

    generic_contribution = (
        0.85 * args.generic_ballot_dem_margin
    )

    approval_contribution = (
        0.50 * approval_adjustment
    )

    midterm_contribution = (
        0.50 * args.midterm_adjustment_dem
    )

    national_environment = (
        generic_contribution
        + approval_contribution
        + midterm_contribution
    )

    net_approval = (
        args.approval - args.disapproval
    )

    row = {
        "cycle": args.cycle,
        "snapshot_label": args.snapshot_label,
        "as_of_date": args.as_of_date,
        "generic_ballot_margin_dem": (
            args.generic_ballot_dem_margin
        ),
        "presidential_approval": args.approval,
        "presidential_disapproval": args.disapproval,
        "presidential_net_approval": net_approval,
        "president_party": args.president_party,
        "midterm_adjustment_dem": (
            args.midterm_adjustment_dem
        ),
        "approval_adjustment_dem": (
            approval_adjustment
        ),
        "generic_ballot_coefficient": 0.85,
        "approval_coefficient": 0.50,
        "midterm_coefficient": 0.50,
        "generic_ballot_contribution_dem": (
            generic_contribution
        ),
        "approval_contribution_dem": (
            approval_contribution
        ),
        "midterm_contribution_dem": (
            midterm_contribution
        ),
        "national_environment_margin_dem": (
            national_environment
        ),
        "generic_ballot_source": args.generic_source,
        "approval_source": args.approval_source,
        "formula_version": (
            "reduced_double_count_v1"
        ),
        "notes": args.notes,
    }

    output = pd.DataFrame([row])

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = (
        f"house_{args.cycle}_"
        f"{args.snapshot_label}_national_environment"
    )

    output_path = OUTPUT_DIR / f"{stem}.csv"
    validation_path = (
        VALIDATION_DIR / f"{stem}_validation.txt"
    )

    output.to_csv(
        output_path,
        index=False,
    )

    validation_lines = [
        (
            f"{args.cycle} Historical National "
            "Environment Validation"
        ),
        "=" * 47,
        "",
        (
            "Generic ballot Dem margin: "
            f"{args.generic_ballot_dem_margin:+.4f}"
        ),
        (
            "Presidential approval: "
            f"{args.approval:.4f}"
        ),
        (
            "Presidential disapproval: "
            f"{args.disapproval:.4f}"
        ),
        (
            "Presidential net approval: "
            f"{net_approval:+.4f}"
        ),
        (
            "President party: "
            f"{args.president_party}"
        ),
        (
            "Approval adjustment Dem: "
            f"{approval_adjustment:+.4f}"
        ),
        (
            "Midterm adjustment Dem: "
            f"{args.midterm_adjustment_dem:+.4f}"
        ),
        "",
        "Formula:",
        (
            "0.85*generic ballot "
            "+ 0.50*approval adjustment "
            "+ 0.50*midterm adjustment"
        ),
        "",
        (
            "Generic contribution: "
            f"{generic_contribution:+.4f}"
        ),
        (
            "Approval contribution: "
            f"{approval_contribution:+.4f}"
        ),
        (
            "Midterm contribution: "
            f"{midterm_contribution:+.4f}"
        ),
        (
            "National environment Dem: "
            f"{national_environment:+.4f}"
        ),
        "",
        (
            "This is a forecast input snapshot, "
            "not the actual House popular-vote outcome."
        ),
    ]

    validation_text = "\n".join(
        validation_lines
    )

    validation_path.write_text(
        validation_text
    )

    print(validation_text)
    print()
    print(f"Wrote: {output_path}")
    print(f"Wrote: {validation_path}")


if __name__ == "__main__":
    main()
