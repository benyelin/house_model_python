"""
House uncertainty calibration audit.

Summarizes historical production-replay seat errors. This is a
preliminary diagnostic before adding simulation-based interval coverage.
"""

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]

REPLAY_OUTPUT_DIR = (
    REPO_ROOT
    / "historical"
    / "house"
    / "backtests"
    / "outputs"
)

OUTPUT_DIR = (
    REPO_ROOT
    / "historical"
    / "house"
    / "diagnostics"
    / "outputs"
)

PRODUCTION_SPEC = "production_election_day_v1"


def locate_replay_summary() -> Path:
    candidates = sorted(
        REPLAY_OUTPUT_DIR.glob("*replay*summary*.csv")
    )

    if not candidates:
        candidates = sorted(
            REPLAY_OUTPUT_DIR.glob("*production*replay*.csv")
        )

    valid_candidates: list[Path] = []

    for candidate in candidates:
        try:
            columns = set(
                pd.read_csv(candidate, nrows=2).columns
            )
        except Exception:
            continue

        required = {
            "replay_spec",
            "cycle",
            "expected_dem_seats",
            "actual_dem_seats",
        }

        if required.issubset(columns):
            valid_candidates.append(candidate)

    if len(valid_candidates) == 1:
        return valid_candidates[0]

    if not valid_candidates:
        raise FileNotFoundError(
            "Could not locate a replay summary CSV containing "
            "replay_spec, cycle, expected_dem_seats, and "
            "actual_dem_seats in:\n"
            f"{REPLAY_OUTPUT_DIR}"
        )

    raise RuntimeError(
        "Multiple possible replay summary files found:\n"
        + "\n".join(
            f"  - {path}"
            for path in valid_candidates
        )
    )


def main() -> None:
    replay_summary = locate_replay_summary()

    summary = pd.read_csv(replay_summary)

    production = summary.loc[
        summary["replay_spec"].eq(PRODUCTION_SPEC)
    ].copy()

    if production.empty:
        raise RuntimeError(
            f"No rows found for replay spec {PRODUCTION_SPEC!r}."
        )

    production["cycle"] = pd.to_numeric(
        production["cycle"],
        errors="raise",
    ).astype(int)

    for column in [
        "expected_dem_seats",
        "actual_dem_seats",
    ]:
        production[column] = pd.to_numeric(
            production[column],
            errors="raise",
        )

    production = production.sort_values("cycle").reset_index(
        drop=True
    )

    production["seat_error_expected_minus_actual"] = (
        production["expected_dem_seats"]
        - production["actual_dem_seats"]
    )

    production["absolute_seat_error"] = (
        production[
            "seat_error_expected_minus_actual"
        ].abs()
    )

    errors = production[
        "seat_error_expected_minus_actual"
    ]

    mean_error = float(errors.mean())
    mae = float(errors.abs().mean())
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    sample_std = float(errors.std(ddof=1))

    if np.isfinite(sample_std) and sample_std > 0:
        production["seat_error_z_score"] = (
            errors - mean_error
        ) / sample_std
    else:
        production["seat_error_z_score"] = np.nan

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = (
        OUTPUT_DIR
        / "house_uncertainty_calibration.csv"
    )

    text_path = (
        OUTPUT_DIR
        / "house_uncertainty_summary.txt"
    )

    production.to_csv(csv_path, index=False)

    lines = [
        "HOUSE UNCERTAINTY CALIBRATION",
        "=" * 40,
        "",
        f"Replay source: {replay_summary}",
        f"Replay spec:   {PRODUCTION_SPEC}",
        f"Cycles:        {len(production)}",
        "",
        f"Mean seat error:       {mean_error:+.3f}",
        f"Mean absolute error:    {mae:.3f}",
        f"RMSE:                   {rmse:.3f}",
        f"Sample standard dev.:   {sample_std:.3f}",
        "",
        "Per-cycle central forecast errors",
        "-" * 40,
    ]

    for row in production.itertuples(index=False):
        lines.append(
            f"{row.cycle}: "
            f"expected={row.expected_dem_seats:.3f}, "
            f"actual={row.actual_dem_seats:.0f}, "
            f"error={row.seat_error_expected_minus_actual:+.3f}"
        )

    lines.extend(
        [
            "",
            "NOTE:",
            "These statistics evaluate central seat forecasts only.",
            "They do not yet measure 50%, 80%, or 95% interval coverage.",
            "",
            f"Wrote: {csv_path}",
            f"Wrote: {text_path}",
        ]
    )

    output = "\n".join(lines) + "\n"
    text_path.write_text(output)

    print(output, end="")


if __name__ == "__main__":
    main()
