from pathlib import Path
import subprocess
import sys
import pandas as pd

SETTINGS_PATH = Path("inputs/house_calibration_settings.csv")
AUDIT_PATH = Path("outputs/house_calibration_audit.csv")
OUTPUT_PATH = Path("outputs/house_environment_multiplier_comparison.csv")
DETAIL_OUTPUT_PATH = Path("outputs/house_environment_multiplier_district_details.csv")

MULTIPLIERS = [0.80, 0.85, 0.90, 1.00]

RATING_ORDER = [
    "Safe D",
    "Likely D",
    "Lean D",
    "Tilt D",
    "Toss-Up",
    "Tilt R",
    "Lean R",
    "Likely R",
    "Safe R",
]


def run(cmd):
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def read_settings():
    if SETTINGS_PATH.exists():
        return pd.read_csv(SETTINGS_PATH)

    return pd.DataFrame(
        [
            {
                "setting": "house_environment_multiplier",
                "value": 0.85,
                "notes": "Multiplier applied to imported Senate/shared national environment before House district elasticity.",
            }
        ]
    )


def write_multiplier(value):
    settings = read_settings()

    if "setting" not in settings.columns:
        settings["setting"] = ""

    if "value" not in settings.columns:
        settings["value"] = ""

    if "notes" not in settings.columns:
        settings["notes"] = ""

    mask = settings["setting"].astype(str).str.strip() == "house_environment_multiplier"

    if mask.any():
        settings.loc[mask, "value"] = value
    else:
        settings = pd.concat(
            [
                settings,
                pd.DataFrame(
                    [
                        {
                            "setting": "house_environment_multiplier",
                            "value": value,
                            "notes": "Multiplier applied to imported Senate/shared national environment before House district elasticity.",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    settings.to_csv(SETTINGS_PATH, index=False)


def summarize_audit(multiplier):
    audit = pd.read_csv(AUDIT_PATH)

    rating_counts = audit["rating"].value_counts(dropna=False).to_dict()

    dem_favored = int(
        audit[audit["dem_win_probability"] > 0.5].shape[0]
    )

    gop_favored = int(
        audit[audit["dem_win_probability"] < 0.5].shape[0]
    )

    tossup_prob_band = int(
        audit[
            audit["dem_win_probability"].between(0.45, 0.55, inclusive="both")
        ].shape[0]
    )

    competitive_prob_band = int(
        audit[
            audit["dem_win_probability"].between(0.35, 0.65, inclusive="both")
        ].shape[0]
    )

    likely_or_safer_d = int(
        audit[audit["dem_win_probability"] >= 0.85].shape[0]
    )

    likely_or_safer_r = int(
        audit[audit["dem_win_probability"] <= 0.15].shape[0]
    )

    summary = {
        "house_environment_multiplier": multiplier,
        "dem_favored_seats": dem_favored,
        "gop_favored_seats": gop_favored,
        "tossup_probability_band_45_55": tossup_prob_band,
        "competitive_probability_band_35_65": competitive_prob_band,
        "likely_or_safer_dem": likely_or_safer_d,
        "likely_or_safer_gop": likely_or_safer_r,
        "median_dem_win_probability": audit["dem_win_probability"].median(),
        "mean_model_margin_dem": audit["model_margin_dem"].mean(),
    }

    for rating in RATING_ORDER:
        summary[f"rating_{rating.replace(' ', '_').replace('-', '').lower()}"] = int(
            rating_counts.get(rating, 0)
        )

    details_cols = [
        "district_id",
        "rating",
        "model_margin_label",
        "model_margin_dem",
        "dem_win_probability",
        "district_partisan_baseline_dem",
        "district_elasticity",
        "imported_national_environment_margin_dem",
        "house_environment_multiplier",
        "house_national_environment_used_dem",
        "district_environment_adjustment_dem",
        "incumbency_adjustment_dem",
        "candidate_quality_adjustment_dem",
        "region",
        "district_type",
    ]

    details_cols = [c for c in details_cols if c in audit.columns]

    details = audit[details_cols].copy()
    details["tested_multiplier"] = multiplier

    return summary, details


def main():
    original_settings = read_settings()

    summaries = []
    details_list = []

    try:
        for multiplier in MULTIPLIERS:
            print()
            print("=" * 72)
            print(f"Testing house_environment_multiplier = {multiplier:.2f}")
            print("=" * 72)

            write_multiplier(multiplier)

            run([sys.executable, "run_house_full_pipeline.py", "--import-seed"])
            run([sys.executable, "build_house_calibration_audit.py"])

            summary, details = summarize_audit(multiplier)
            summaries.append(summary)
            details_list.append(details)

        comparison = pd.DataFrame(summaries)
        comparison.to_csv(OUTPUT_PATH, index=False)

        all_details = pd.concat(details_list, ignore_index=True)
        all_details.to_csv(DETAIL_OUTPUT_PATH, index=False)

        print()
        print("Wrote:")
        print(f"  {OUTPUT_PATH}")
        print(f"  {DETAIL_OUTPUT_PATH}")

        display_cols = [
            "house_environment_multiplier",
            "dem_favored_seats",
            "gop_favored_seats",
            "tossup_probability_band_45_55",
            "competitive_probability_band_35_65",
            "likely_or_safer_dem",
            "likely_or_safer_gop",
            "mean_model_margin_dem",
            "rating_safe_d",
            "rating_likely_d",
            "rating_lean_d",
            "rating_tilt_d",
            "rating_tossup",
            "rating_tilt_r",
            "rating_lean_r",
            "rating_likely_r",
            "rating_safe_r",
        ]

        display_cols = [c for c in display_cols if c in comparison.columns]

        print()
        print("Multiplier comparison:")
        print(comparison[display_cols].to_string(index=False))

    finally:
        original_settings.to_csv(SETTINGS_PATH, index=False)
        print()
        print("Restored original house_calibration_settings.csv.")


if __name__ == "__main__":
    main()
