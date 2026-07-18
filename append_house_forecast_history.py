from pathlib import Path
from datetime import datetime
import pandas as pd

OUTPUTS = Path("outputs")

SUMMARY_PATH = OUTPUTS / "house_forecast_summary.csv"
HISTORY_PATH = OUTPUTS / "house_forecast_history.csv"
RACE_STATS_PATH = OUTPUTS / "house_race_stats.csv"


def first_available(row, names, default=None):
    for name in names:
        if name in row.index:
            value = row.get(name)
            if pd.notna(value):
                return value
    return default


def main():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError("outputs/house_forecast_summary.csv not found. Run the House pipeline first.")

    summary = pd.read_csv(SUMMARY_PATH)

    if summary.empty:
        raise ValueError("outputs/house_forecast_summary.csv is empty.")

    srow = summary.iloc[-1]

    race_environment = {}
    if RACE_STATS_PATH.exists():
        race_environment_df = pd.read_csv(RACE_STATS_PATH)

        for output_name, source_columns in {
            "imported_national_environment": [
                "imported_national_environment_margin_dem",
            ],
            "house_environment_multiplier": [
                "house_environment_multiplier",
            ],
            "house_adjusted_national_environment": [
                "house_national_environment_used_dem",
                "national_environment_margin_dem",
            ],
        }.items():
            race_environment[output_name] = None

            for source_column in source_columns:
                if source_column not in race_environment_df.columns:
                    continue

                values = pd.to_numeric(
                    race_environment_df[source_column],
                    errors="coerce",
                ).dropna()

                if not values.empty:
                    race_environment[output_name] = float(
                        values.iloc[0]
                    )
                    break

    timestamp = datetime.now().isoformat(timespec="seconds")

    snapshot = {
        "timestamp": timestamp,
        "run_date": datetime.now().date().isoformat(),
        "n_sims": first_available(srow, ["n_sims"]),
        "days_out": first_available(srow, ["days_out"]),
        "expected_dem_seats": first_available(srow, ["expected_dem_seats"]),
        "median_dem_seats": first_available(srow, ["median_dem_seats"]),
        "dem_control_probability": first_available(
            srow,
            ["dem_control_probability", "dem_majority_probability"],
        ),
        "dem_control_threshold": first_available(srow, ["dem_control_threshold"], 218),
        "national_environment": first_available(
            srow,
            [
                "house_adjusted_national_environment",
                "national_environment_margin",
                "national_environment",
                "national_environment_margin_dem",
            ],
            race_environment.get(
                "house_adjusted_national_environment"
            ),
        ),
        "imported_national_environment": first_available(
            srow,
            [
                "imported_national_environment_margin",
                "imported_national_environment",
            ],
            race_environment.get(
                "imported_national_environment"
            ),
        ),
        "house_environment_multiplier": first_available(
            srow,
            ["house_environment_multiplier"],
            race_environment.get(
                "house_environment_multiplier"
            ),
        ),
        "house_adjusted_national_environment": first_available(
            srow,
            [
                "house_adjusted_national_environment",
                "national_environment_margin",
            ],
            race_environment.get(
                "house_adjusted_national_environment"
            ),
        ),
        "national_error_sd": first_available(srow, ["national_error_sd"]),
        "region_error_sd": first_available(srow, ["region_error_sd"]),
        "demographic_error_sd": first_available(
            srow,
            ["demographic_error_sd", "education_race_error_sd"],
        ),
        "district_error_sd": first_available(srow, ["district_error_sd", "race_error_sd"]),
        "total_error_sd": first_available(srow, ["total_error_sd"]),
        "implied_correlation": first_available(srow, ["implied_correlation"]),
        "uncertainty_engine": first_available(srow, ["uncertainty_engine"]),
    }

    # Add rating-count snapshot if race stats exist.
    if RACE_STATS_PATH.exists():
        race = pd.read_csv(RACE_STATS_PATH)
        if "rating" in race.columns:
            counts = race["rating"].value_counts(dropna=False).to_dict()
            for rating in [
                "Safe D",
                "Likely D",
                "Lean D",
                "Tilt D",
                "Toss-Up",
                "Tilt R",
                "Lean R",
                "Likely R",
                "Safe R",
            ]:
                key = "rating_" + rating.lower().replace(" ", "_").replace("-", "")
                snapshot[key] = counts.get(rating, 0)

    new_row = pd.DataFrame([snapshot])

    if HISTORY_PATH.exists():
        history = pd.read_csv(HISTORY_PATH)
        history = pd.concat([history, new_row], ignore_index=True)
    else:
        history = new_row

    # Avoid accidental duplicate rows from rerunning the append script within the same second.
    history = history.drop_duplicates(subset=["timestamp"], keep="last")

    history.to_csv(HISTORY_PATH, index=False)

    print(f"Appended House forecast snapshot to {HISTORY_PATH}")
    print(new_row.to_string(index=False))


if __name__ == "__main__":
    main()
