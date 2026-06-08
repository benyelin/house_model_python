from pathlib import Path
import subprocess
import sys
import pandas as pd

SETTINGS_PATH = Path("inputs/house_calibration_settings.csv")
OUTPUTS = Path("outputs")

COMPARE_PATH = OUTPUTS / "house_poll_spillover_toggle_comparison.csv"
SUMMARY_COMPARE_PATH = OUTPUTS / "house_poll_spillover_summary_comparison.csv"


def set_setting(name, value):
    settings = pd.read_csv(SETTINGS_PATH)
    mask = settings["setting"].astype(str).str.strip().eq(name)

    if mask.any():
        settings.loc[mask, "value"] = value
    else:
        settings = pd.concat(
            [
                settings,
                pd.DataFrame(
                    [
                        {
                            "setting": name,
                            "value": value,
                            "notes": "Set by poll spillover toggle comparison.",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    settings.to_csv(SETTINGS_PATH, index=False)


def run_pipeline(label):
    print()
    print("=" * 72)
    print(f"Running House pipeline with poll spillover {label}")
    print("=" * 72)

    subprocess.run(
        [sys.executable, "run_house_full_pipeline.py", "--import-seed"],
        check=True,
    )

    race = pd.read_csv(OUTPUTS / "house_race_stats.csv")
    summary = pd.read_csv(OUTPUTS / "house_forecast_summary.csv")

    race["poll_spillover_toggle"] = label
    summary["poll_spillover_toggle"] = label

    return race, summary


def main():
    if not SETTINGS_PATH.exists():
        raise FileNotFoundError("inputs/house_calibration_settings.csv not found.")

    original = pd.read_csv(SETTINGS_PATH)
    mask = original["setting"].astype(str).str.strip().eq("use_house_poll_spillover_adjustments")
    original_value = original.loc[mask, "value"].iloc[0] if mask.any() else 0

    try:
        set_setting("use_house_poll_spillover_adjustments", 0)
        race_off, summary_off = run_pipeline("off")

        set_setting("use_house_poll_spillover_adjustments", 1)
        race_on, summary_on = run_pipeline("on")

    finally:
        set_setting("use_house_poll_spillover_adjustments", original_value)

    key = "district_id"

    keep = [
        key,
        "rating",
        "dem_win_probability",
        "model_margin_dem",
        "fundamentals_margin_dem",
        "poll_spillover_adjustment_dem",
        "poll_spillover_source_count",
        "dem_candidate",
        "gop_candidate",
    ]

    off_cols = [c for c in keep if c in race_off.columns]
    on_cols = [c for c in keep if c in race_on.columns]

    off = race_off[off_cols].copy().add_suffix("_off")
    on = race_on[on_cols].copy().add_suffix("_on")

    merged = off.merge(
        on,
        left_on=f"{key}_off",
        right_on=f"{key}_on",
        how="inner",
    )

    merged["district_id"] = merged[f"{key}_off"]

    for col in ["dem_win_probability", "model_margin_dem", "fundamentals_margin_dem"]:
        off_col = f"{col}_off"
        on_col = f"{col}_on"

        if off_col in merged.columns and on_col in merged.columns:
            merged[f"{col}_change"] = (
                pd.to_numeric(merged[on_col], errors="coerce")
                - pd.to_numeric(merged[off_col], errors="coerce")
            )

    if "model_margin_dem_change" in merged.columns:
        merged = merged.sort_values(
            "model_margin_dem_change",
            key=lambda s: s.abs(),
            ascending=False,
        )

    OUTPUTS.mkdir(exist_ok=True)
    merged.to_csv(COMPARE_PATH, index=False)

    summary_compare = pd.concat([summary_off, summary_on], ignore_index=True)
    summary_compare.to_csv(SUMMARY_COMPARE_PATH, index=False)

    print()
    print("Summary comparison")
    print("------------------")
    summary_cols = [
        "poll_spillover_toggle",
        "expected_dem_seats",
        "median_dem_seats",
        "dem_control_probability",
        "total_error_sd",
        "days_out",
    ]
    summary_cols = [c for c in summary_cols if c in summary_compare.columns]
    print(summary_compare[summary_cols].to_string(index=False))

    print()
    print("Largest district-level movements")
    print("--------------------------------")
    display_cols = [
        "district_id",
        "dem_candidate_on",
        "gop_candidate_on",
        "rating_off",
        "rating_on",
        "model_margin_dem_off",
        "model_margin_dem_on",
        "model_margin_dem_change",
        "dem_win_probability_off",
        "dem_win_probability_on",
        "dem_win_probability_change",
        "poll_spillover_adjustment_dem_on",
        "poll_spillover_source_count_on",
    ]
    display_cols = [c for c in display_cols if c in merged.columns]
    print(merged[display_cols].head(60).to_string(index=False))

    print()
    print(f"Wrote {COMPARE_PATH}")
    print(f"Wrote {SUMMARY_COMPARE_PATH}")


if __name__ == "__main__":
    main()
