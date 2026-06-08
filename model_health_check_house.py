from pathlib import Path
import sys
import pandas as pd
import numpy as np

INPUTS = Path("inputs")
OUTPUTS = Path("outputs")

REQUIRED_FILES = [
    INPUTS / "house_race_inputs.csv",
    INPUTS / "house_calibration_settings.csv",
    OUTPUTS / "house_race_stats.csv",
    OUTPUTS / "house_forecast_summary.csv",
    OUTPUTS / "house_forecast_history.csv",
    OUTPUTS / "house_calibration_audit.csv",
    OUTPUTS / "house_local_context_audit.csv",
]

OPTIONAL_BUT_EXPECTED_FILES = [
    OUTPUTS / "house_candidate_war_audit.csv",
    OUTPUTS / "house_candidate_war_unmatched_candidates.csv",
    INPUTS / "house_manual_polls.csv",
    INPUTS / "house_polling_averages_generated.csv",
]


def read_csv_safe(path):
    try:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Could not read {path}: {exc}")


def print_messages(title, messages):
    print(title)
    print("-" * len(title))

    if not messages:
        print("None")
    else:
        for msg in messages:
            print(f"- {msg}")

    print()


def get_setting(settings, name, default=None):
    if settings.empty or "setting" not in settings.columns or "value" not in settings.columns:
        return default

    mask = settings["setting"].astype(str).str.strip().eq(name)

    if not mask.any():
        return default

    try:
        return settings.loc[mask, "value"].iloc[0]
    except Exception:
        return default


def as_float(value, default=np.nan):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def rating_from_prob(p):
    p = as_float(p)

    if pd.isna(p):
        return "Unknown"

    if p >= 0.95:
        return "Safe D"
    if p >= 0.80:
        return "Likely D"
    if p >= 0.65:
        return "Lean D"
    if p >= 0.55:
        return "Tilt D"
    if p > 0.45:
        return "Toss-Up"
    if p > 0.35:
        return "Tilt R"
    if p > 0.20:
        return "Lean R"
    if p > 0.05:
        return "Likely R"
    return "Safe R"


def normalize_district_id(x):
    return str(x).strip().upper()


def main():
    errors = []
    warnings = []
    info = []

    print("2026 House Model Health Check")
    print("=============================")

    # ------------------------------------------------------------
    # File existence
    # ------------------------------------------------------------
    for path in REQUIRED_FILES:
        if path.exists():
            info.append(f"Found {path}")
        else:
            errors.append(f"Missing required file: {path}")

    for path in OPTIONAL_BUT_EXPECTED_FILES:
        if path.exists():
            info.append(f"Found optional file: {path}")
        else:
            warnings.append(f"Optional/expected file not found: {path}")

    race_inputs = read_csv_safe(INPUTS / "house_race_inputs.csv")
    race_stats = read_csv_safe(OUTPUTS / "house_race_stats.csv")
    summary = read_csv_safe(OUTPUTS / "house_forecast_summary.csv")
    history = read_csv_safe(OUTPUTS / "house_forecast_history.csv")
    calibration = read_csv_safe(OUTPUTS / "house_calibration_audit.csv")
    local_context = read_csv_safe(OUTPUTS / "house_local_context_audit.csv")
    settings = read_csv_safe(INPUTS / "house_calibration_settings.csv")
    war_audit = read_csv_safe(OUTPUTS / "house_candidate_war_audit.csv")

    # ------------------------------------------------------------
    # Basic data shape
    # ------------------------------------------------------------
    if not race_inputs.empty:
        if len(race_inputs) != 435:
            errors.append(f"house_race_inputs.csv has {len(race_inputs)} rows, expected 435.")
        else:
            info.append("house_race_inputs.csv contains 435 districts.")

        if "district_id" not in race_inputs.columns:
            errors.append("house_race_inputs.csv missing district_id.")
        else:
            duplicated = race_inputs["district_id"].astype(str).str.upper().duplicated().sum()
            if duplicated:
                errors.append(f"house_race_inputs.csv has {duplicated} duplicated district_id rows.")
            else:
                info.append("No duplicate district_id rows in house_race_inputs.csv.")

    if not race_stats.empty:
        if len(race_stats) != 435:
            errors.append(f"house_race_stats.csv has {len(race_stats)} rows, expected 435.")
        else:
            info.append("house_race_stats.csv contains 435 districts.")

        if "district_id" not in race_stats.columns:
            errors.append("house_race_stats.csv missing district_id.")
        else:
            duplicated = race_stats["district_id"].astype(str).str.upper().duplicated().sum()
            if duplicated:
                errors.append(f"house_race_stats.csv has {duplicated} duplicated district_id rows.")
            else:
                info.append("No duplicate district_id rows in house_race_stats.csv.")

    # ------------------------------------------------------------
    # Required columns
    # ------------------------------------------------------------
    required_race_input_cols = [
        "district_id",
        "district_partisan_baseline_dem",
        "district_environment_adjustment_dem",
        "state_environment_adjustment_dem",
        "incumbency_adjustment_dem",
        "candidate_quality_adjustment_dem",
        "special_adjustment_dem",
        "fundamentals_margin_dem",
    ]

    for col in required_race_input_cols:
        if col not in race_inputs.columns:
            errors.append(f"house_race_inputs.csv missing required column: {col}")

    required_race_stats_cols = [
        "district_id",
        "model_margin_dem",
        "dem_win_probability",
        "rating",
    ]

    for col in required_race_stats_cols:
        if col not in race_stats.columns:
            errors.append(f"house_race_stats.csv missing required column: {col}")

    # ------------------------------------------------------------
    # Baselines and numeric sanity
    # ------------------------------------------------------------
    if "district_partisan_baseline_dem" in race_inputs.columns:
        baseline = pd.to_numeric(race_inputs["district_partisan_baseline_dem"], errors="coerce")
        missing = baseline.isna().sum()

        if missing:
            errors.append(f"{missing} districts have missing/non-numeric district_partisan_baseline_dem.")
        else:
            info.append("All districts have numeric district_partisan_baseline_dem.")

    if "district_elasticity" in race_inputs.columns:
        elasticity = pd.to_numeric(race_inputs["district_elasticity"], errors="coerce")
        bad = elasticity.isna().sum()

        if bad:
            errors.append(f"{bad} districts have missing/non-numeric district_elasticity.")

        if (elasticity <= 0).any():
            errors.append("At least one district has district_elasticity <= 0.")
        else:
            info.append("District elasticity values are positive.")

    # ------------------------------------------------------------
    # Probabilities and margins
    # ------------------------------------------------------------
    if not race_stats.empty and "dem_win_probability" in race_stats.columns:
        probs = pd.to_numeric(race_stats["dem_win_probability"], errors="coerce")

        if probs.isna().sum():
            errors.append(f"{probs.isna().sum()} races have missing/non-numeric dem_win_probability.")

        out_of_range = ((probs < 0) | (probs > 1)).sum()
        if out_of_range:
            errors.append(f"{out_of_range} races have dem_win_probability outside [0, 1].")
        else:
            info.append("All race probabilities are within [0, 1].")

    if not race_stats.empty and {"model_margin_dem", "dem_win_probability"}.issubset(race_stats.columns):
        margins = pd.to_numeric(race_stats["model_margin_dem"], errors="coerce")
        probs = pd.to_numeric(race_stats["dem_win_probability"], errors="coerce")

        direction_mismatch = (
            ((margins > 0.25) & (probs < 0.5))
            | ((margins < -0.25) & (probs > 0.5))
        ).sum()

        if direction_mismatch:
            errors.append(f"{direction_mismatch} races have margin/probability direction mismatches.")
        else:
            info.append("Race margin/probability directions look consistent.")

    if not race_stats.empty and {"rating", "dem_win_probability"}.issubset(race_stats.columns):
        expected = race_stats["dem_win_probability"].apply(rating_from_prob)
        actual = race_stats["rating"].astype(str)

        mismatches = (expected.astype(str) != actual.astype(str)).sum()

        if mismatches:
            warnings.append(f"{mismatches} races have ratings that differ from probability-derived rating bands.")
        else:
            info.append("Race ratings match probability-derived rating bands.")

    # ------------------------------------------------------------
    # Fundamentals component integrity
    # ------------------------------------------------------------
    component_cols = [
        "district_partisan_baseline_dem",
        "district_environment_adjustment_dem",
        "state_environment_adjustment_dem",
        "incumbency_adjustment_dem",
        "candidate_quality_adjustment_dem",
        "special_adjustment_dem",
    ]

    # Optional active components that are added after the core fundamentals formula.
    optional_component_cols = []

    use_poll_spillover = as_float(
        get_setting(settings, "use_house_poll_spillover_adjustments", 0),
        0,
    )

    if use_poll_spillover >= 0.5 and "poll_spillover_adjustment_dem" in race_inputs.columns:
        optional_component_cols.append("poll_spillover_adjustment_dem")
        info.append("Poll spillover adjustment is active and included in fundamentals component check.")
    elif "poll_spillover_adjustment_dem" in race_inputs.columns:
        info.append("Poll spillover adjustment column present but inactive.")

    all_component_cols = component_cols + optional_component_cols

    if not race_inputs.empty and all(c in race_inputs.columns for c in component_cols + ["fundamentals_margin_dem"]):
        component_sum = sum(
            pd.to_numeric(race_inputs[c], errors="coerce").fillna(0.0)
            for c in all_component_cols
            if c in race_inputs.columns
        )

        actual_fund = pd.to_numeric(race_inputs["fundamentals_margin_dem"], errors="coerce")
        diff = (actual_fund - component_sum).abs()

        bad = diff.gt(0.01).sum()

        if bad:
            worst = race_inputs.loc[
                diff.sort_values(ascending=False).head(5).index,
                ["district_id", "fundamentals_margin_dem"]
            ].copy()
            worst["component_sum"] = component_sum.loc[worst.index]
            worst["difference"] = diff.loc[worst.index]
            errors.append(
                "Fundamentals margins do not match active component sums for "
                f"{bad} districts. Worst examples: {worst.to_dict(orient='records')}"
            )
        else:
            info.append("Fundamentals margins match active component sums.")

    # ------------------------------------------------------------
    # Forecast summary
    # ------------------------------------------------------------
    if not summary.empty:
        row = summary.iloc[-1]

        for col in ["expected_dem_seats", "median_dem_seats", "dem_control_probability"]:
            if col not in summary.columns:
                errors.append(f"house_forecast_summary.csv missing {col}")

        if "dem_control_probability" in summary.columns:
            p = as_float(row.get("dem_control_probability"))

            if pd.isna(p) or p < 0 or p > 1:
                errors.append("Forecast summary dem_control_probability is missing or outside [0, 1].")
            else:
                info.append(f"Dem control probability: {p:.1%}")

        if "expected_dem_seats" in summary.columns:
            expected_seats = as_float(row.get("expected_dem_seats"))

            if pd.isna(expected_seats) or expected_seats < 0 or expected_seats > 435:
                errors.append("Expected Dem seats is missing or outside [0, 435].")
            else:
                info.append(f"Expected Dem seats: {expected_seats:.2f}")

        if "uncertainty_engine" in summary.columns:
            engine = str(row.get("uncertainty_engine", ""))
            if "dynamic" in engine.lower():
                info.append(f"Dynamic uncertainty engine detected: {engine}")
            else:
                warnings.append(f"Forecast summary uncertainty_engine is not dynamic-looking: {engine}")

        for col in ["national_error_sd", "region_error_sd", "demographic_error_sd", "district_error_sd", "total_error_sd"]:
            if col in summary.columns:
                val = as_float(row.get(col))
                if pd.isna(val) or val <= 0:
                    errors.append(f"{col} is missing or non-positive in forecast summary.")

    # ------------------------------------------------------------
    # Forecast history
    # ------------------------------------------------------------
    if not history.empty:
        if len(history) < 1:
            warnings.append("Forecast history exists but has no rows.")
        else:
            info.append(f"Forecast history contains {len(history)} snapshot row(s).")

        if "dem_control_probability" in history.columns:
            probs = pd.to_numeric(history["dem_control_probability"], errors="coerce")
            bad = ((probs < 0) | (probs > 1)).sum()
            if bad:
                errors.append(f"{bad} forecast history rows have dem_control_probability outside [0, 1].")

    # ------------------------------------------------------------
    # Candidate WAR checks
    # ------------------------------------------------------------
    use_war = as_float(get_setting(settings, "use_candidate_war_adjustments", 0), 0)
    war_cap = as_float(get_setting(settings, "house_candidate_war_cap", 3.0), 3.0)

    if use_war >= 0.5:
        info.append("Candidate WAR adjustments are active.")

        if war_audit.empty:
            errors.append("Candidate WAR is active but outputs/house_candidate_war_audit.csv is missing or empty.")
        else:
            if "candidate_war_adjustment_dem" not in war_audit.columns:
                errors.append("WAR audit missing candidate_war_adjustment_dem.")
            else:
                war_adj = pd.to_numeric(war_audit["candidate_war_adjustment_dem"], errors="coerce").fillna(0.0)
                over_cap = war_adj.abs().gt(war_cap + 0.001).sum()

                if over_cap:
                    errors.append(f"{over_cap} WAR audit rows exceed configured cap ±{war_cap}.")
                else:
                    info.append(f"Candidate WAR adjustments are within configured cap ±{war_cap}.")

            if "war_match_status" in war_audit.columns:
                counts = war_audit["war_match_status"].fillna("Unknown").value_counts().to_dict()
                info.append(f"Candidate WAR match coverage: {counts}")

        if not race_stats.empty:
            if "candidate_war_adjustment_dem" not in race_stats.columns:
                warnings.append("Candidate WAR is active but house_race_stats.csv lacks candidate_war_adjustment_dem.")
            else:
                nonzero = pd.to_numeric(race_stats["candidate_war_adjustment_dem"], errors="coerce").fillna(0.0).abs().gt(0).sum()
                info.append(f"Race stats include nonzero candidate WAR adjustments in {nonzero} districts.")
    else:
        info.append("Candidate WAR adjustments are inactive.")

    # ------------------------------------------------------------
    # Calibration audit
    # ------------------------------------------------------------
    if not calibration.empty:
        flagged_col = None
        for candidate in ["flagged", "is_flagged", "has_issue", "audit_flag"]:
            if candidate in calibration.columns:
                flagged_col = candidate
                break

        if flagged_col:
            flagged = calibration[flagged_col].astype(str).str.lower().isin(["true", "1", "yes"]).sum()
            if flagged:
                warnings.append(f"Calibration audit has {flagged} flagged row(s).")
            else:
                info.append("Calibration audit has no flagged rows.")
        else:
            info.append("Calibration audit found; no explicit flagged column detected.")

    # ------------------------------------------------------------
    # Local context audit
    # ------------------------------------------------------------
    if not local_context.empty:
        info.append(f"Local context audit contains {len(local_context)} rows.")

        if "audit_priority" in local_context.columns:
            counts = local_context["audit_priority"].fillna("Unknown").value_counts().to_dict()
            info.append(f"Local context audit priority counts: {counts}")

            high = counts.get("High", 0)
            if high:
                warnings.append(f"Local context audit has {high} high-priority review target(s).")

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------
    print()
    print_messages("Errors", errors)
    print_messages("Warnings", warnings)
    print_messages("Info", info)

    if errors:
        print("Health check result: FAIL")
        sys.exit(1)

    if warnings:
        print("Health check result: PASS WITH WARNINGS")
        sys.exit(0)

    print("Health check result: PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
