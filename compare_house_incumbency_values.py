from pathlib import Path
import subprocess
import sys
import pandas as pd

SETTINGS_PATH = Path("inputs/house_calibration_settings.csv")
AUDIT_PATH = Path("outputs/house_calibration_audit.csv")
OUTPUT_PATH = Path("outputs/house_incumbency_value_comparison.csv")
DETAIL_OUTPUT_PATH = Path("outputs/house_incumbency_value_district_details.csv")

VALUES = [1.25, 1.50, 2.00]

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

    return pd.DataFrame([
        {
            "setting": "house_environment_multiplier",
            "value": 0.85,
            "notes": "Multiplier applied to imported Senate/shared national environment before House district elasticity.",
        },
        {
            "setting": "generic_house_incumbency_points",
            "value": 2.0,
            "notes": "Generic House incumbency adjustment.",
        },
    ])


def write_setting(setting_name, value, notes):
    settings = read_settings()

    for col in ["setting", "value", "notes"]:
        if col not in settings.columns:
            settings[col] = ""

    mask = settings["setting"].astype(str).str.strip() == setting_name

    if mask.any():
        settings.loc[mask, "value"] = value
        settings.loc[mask, "notes"] = notes
    else:
        settings = pd.concat(
            [
                settings,
                pd.DataFrame([
                    {
                        "setting": setting_name,
                        "value": value,
                        "notes": notes,
                    }
                ]),
            ],
            ignore_index=True,
        )

    settings.to_csv(SETTINGS_PATH, index=False)


def summarize(value):
    audit = pd.read_csv(AUDIT_PATH)

    rating_counts = audit["rating"].value_counts(dropna=False).to_dict()

    summary = {
        "generic_house_incumbency_points": value,
        "dem_favored_seats": int((audit["dem_win_probability"] > 0.5).sum()),
        "gop_favored_seats": int((audit["dem_win_probability"] < 0.5).sum()),
        "tossup_probability_band_45_55": int(audit["dem_win_probability"].between(0.45, 0.55, inclusive="both").sum()),
        "competitive_probability_band_35_65": int(audit["dem_win_probability"].between(0.35, 0.65, inclusive="both").sum()),
        "likely_or_safer_dem": int((audit["dem_win_probability"] >= 0.85).sum()),
        "likely_or_safer_gop": int((audit["dem_win_probability"] <= 0.15).sum()),
        "mean_model_margin_dem": audit["model_margin_dem"].mean(),
    }

    for rating in RATING_ORDER:
        summary[f"rating_{rating.replace(' ', '_').replace('-', '').lower()}"] = int(
            rating_counts.get(rating, 0)
        )

    detail_cols = [
        "district_id",
        "rating",
        "model_margin_label",
        "model_margin_dem",
        "dem_win_probability",
        "incumbent_party",
        "inferred_incumbent_party",
        "incumbency_adjustment_dem",
        "generic_house_incumbency_points",
        "district_partisan_baseline_dem",
        "district_elasticity",
        "region",
        "district_type",
    ]

    detail_cols = [c for c in detail_cols if c in audit.columns]

    details = audit[detail_cols].copy()
    details["tested_incumbency_points"] = value

    return summary, details


def main():
    original_settings = read_settings()

    summaries = []
    details = []

    try:
        for value in VALUES:
            print()
            print("=" * 72)
            print(f"Testing generic_house_incumbency_points = {value:.2f}")
            print("=" * 72)

            write_setting(
                "generic_house_incumbency_points",
                value,
                "Generic House incumbency adjustment in points; positive for Dem incumbents and negative for GOP incumbents.",
            )

            run([sys.executable, "run_house_full_pipeline.py", "--import-seed"])
            run([sys.executable, "build_house_calibration_audit.py"])

            summary, detail = summarize(value)
            summaries.append(summary)
            details.append(detail)

        comparison = pd.DataFrame(summaries)
        comparison.to_csv(OUTPUT_PATH, index=False)

        all_details = pd.concat(details, ignore_index=True)
        all_details.to_csv(DETAIL_OUTPUT_PATH, index=False)

        display_cols = [
            "generic_house_incumbency_points",
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
        print("Incumbency comparison:")
        print(comparison[display_cols].to_string(index=False))

        print()
        print(f"Wrote {OUTPUT_PATH}")
        print(f"Wrote {DETAIL_OUTPUT_PATH}")

    finally:
        original_settings.to_csv(SETTINGS_PATH, index=False)
        print()
        print("Restored original house_calibration_settings.csv.")


if __name__ == "__main__":
    main()
