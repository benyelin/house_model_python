from pathlib import Path
import pandas as pd
import numpy as np

INPUTS = Path("inputs")
OUTPUTS = Path("outputs")

HOUSE_INPUTS = INPUTS / "house_race_inputs.csv"
HOUSE_NATIONAL_ENV_AUDIT = INPUTS / "house_national_environment_audit.csv"
HOUSE_MANUAL_POLLS = INPUTS / "house_manual_polls.csv"

HOUSE_RACE_STATS = OUTPUTS / "house_race_stats.csv"
HOUSE_SEAT_DISTRIBUTION = OUTPUTS / "house_seat_distribution.csv"
HOUSE_FORECAST_SUMMARY = OUTPUTS / "house_forecast_summary.csv"

EXPECTED_DISTRICTS = 435
AT_LARGE_STATES = {"AK", "DE", "ND", "SD", "VT", "WY"}

VALID_ELECTION_SYSTEMS = {"standard", "top_two", "top_four_rcv"}
VALID_PARTY_STRUCTURES = {
    "unresolved",
    "D_vs_R",
    "D_vs_D",
    "R_vs_R",
    "D_vs_R_vs_Other",
    "Other",
}
VALID_INCUMBENT_PARTIES = {"D", "R", "Vacant", "I", ""}
VALID_PARTY_OVERRIDES = {"", "D", "R"}
VALID_DISTRICT_TYPES = {"Urban", "Suburban", "Exurban", "Rural", "Mixed"}
VALID_TIERS = {"Low", "Medium", "High", "Very High", "Unknown", ""}

EXPECTED_REGIONS = {
    "Northeast",
    "Mid-Atlantic",
    "Deep South",
    "Middle South",
    "Urban South",
    "Appalachia",
    "Midwest",
    "Great Plains",
    "Mountain West",
    "Pacific",
    "Northwest",
    "Unknown Region",
}


def read_csv(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def as_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def fmt_margin(x):
    x = as_float(x)
    if pd.isna(x):
        return "NA"
    if x > 0:
        return f"D+{x:.2f}"
    if x < 0:
        return f"R+{abs(x):.2f}"
    return "Even"


def normalize_blank_series(s):
    return s.fillna("").astype(str).str.strip()


def normalize_bool_series(s):
    return s.fillna(False).astype(str).str.lower().isin(["true", "1", "yes", "y"])


def print_section(title, rows):
    print()
    print(title)
    print("-" * len(title))

    if not rows:
        print("None")
        return

    for row in rows:
        print(f"- {row}")


def add_missing_col_errors(df, required, errors, file_name):
    for col in required:
        if col not in df.columns:
            errors.append(f"{file_name} missing required column: {col}")


def check_files(errors, warnings, info):
    required_files = [
        HOUSE_INPUTS,
        HOUSE_NATIONAL_ENV_AUDIT,
        HOUSE_RACE_STATS,
        HOUSE_SEAT_DISTRIBUTION,
        HOUSE_FORECAST_SUMMARY,
    ]

    optional_files = [
        HOUSE_MANUAL_POLLS,
    ]

    for path in required_files:
        if path.exists():
            info.append(f"Found {path}")
        else:
            errors.append(f"Missing required file: {path}")

    for path in optional_files:
        if path.exists():
            info.append(f"Found optional file: {path}")
        else:
            info.append(f"Optional file not present yet: {path}")


def check_house_inputs(df, errors, warnings, info):
    if df.empty:
        errors.append("inputs/house_race_inputs.csv is missing or empty.")
        return

    if len(df) != EXPECTED_DISTRICTS:
        errors.append(f"Expected {EXPECTED_DISTRICTS} districts, found {len(df)}.")
    else:
        info.append("House input has 435 districts.")

    required = [
        "state",
        "district",
        "district_id",
        "incumbent_party",
        "inferred_incumbent_party",
        "dem_candidate",
        "gop_candidate",
        "dem_candidate_is_incumbent",
        "gop_candidate_is_incumbent",
        "pres_2024_margin_dem",
        "pres_2020_margin_dem",
        "district_partisan_baseline_dem",
        "region",
        "district_type",
        "state_environment_adjustment_dem",
        "college_share_tier",
        "white_share_tier",
        "black_share_tier",
        "hispanic_share_tier",
        "median_income_tier",
        "education_race_error_group",
        "demographic_error_group",
        "election_system",
        "general_election_party_structure",
        "party_control_override",
        "fundamentals_margin_dem",
        "model_margin_dem",
        "dem_win_probability",
    ]

    add_missing_col_errors(df, required, errors, "house_race_inputs.csv")

    if errors:
        # Continue checking what we can, but avoid key errors below.
        pass

    if "state" in df.columns:
        df["state"] = normalize_blank_series(df["state"]).str.upper()

    if "district_id" in df.columns:
        df["district_id"] = normalize_blank_series(df["district_id"])

        missing_ids = df[df["district_id"].eq("")]
        if not missing_ids.empty:
            errors.append(f"{len(missing_ids)} rows have blank district_id.")

        duplicated = df[df["district_id"].duplicated(keep=False)]
        if not duplicated.empty:
            errors.append(
                "Duplicate district_id values: "
                + ", ".join(sorted(duplicated["district_id"].unique().tolist())[:20])
            )

    # At-large normalization
    if all(c in df.columns for c in ["state", "district", "district_id"]):
        bad_at_large = []
        for _, row in df.iterrows():
            state = str(row["state"]).strip().upper()
            district = str(row["district"]).strip().upper()
            district_id = str(row["district_id"]).strip().upper()

            if state in AT_LARGE_STATES:
                if district not in ["AL", "1", "01"]:
                    bad_at_large.append(f"{district_id}: district={district}")
                if district_id not in [f"{state}-AL", f"{state}-1", f"{state}-01"]:
                    bad_at_large.append(f"{district_id}: unexpected at-large id")

        if bad_at_large:
            warnings.append(
                "At-large district normalization issue: "
                + "; ".join(bad_at_large[:20])
            )
        else:
            info.append("At-large districts look normalized.")

    # Presidential margins / baselines
    for col in ["pres_2024_margin_dem", "pres_2020_margin_dem", "district_partisan_baseline_dem"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            missing = vals.isna().sum()
            if missing:
                warnings.append(f"{missing} districts missing {col}.")
            else:
                info.append(f"No missing {col} values.")

    # Region / district type
    if "region" in df.columns:
        region = normalize_blank_series(df["region"])
        missing = region.eq("").sum()
        unknown = region.eq("Unknown Region").sum()
        invalid = sorted(set(region.unique()) - EXPECTED_REGIONS - {""})
        if missing:
            warnings.append(f"{missing} districts have blank region.")
        if unknown:
            warnings.append(f"{unknown} districts have Unknown Region.")
        if invalid:
            warnings.append(f"Unexpected region values: {invalid}")
        if not missing and not unknown and not invalid:
            info.append("Region values look complete.")

    if "district_type" in df.columns:
        district_type = normalize_blank_series(df["district_type"])
        invalid = sorted(set(district_type.unique()) - VALID_DISTRICT_TYPES - {""})
        missing = district_type.eq("").sum()
        if missing:
            warnings.append(f"{missing} districts have blank district_type.")
        if invalid:
            warnings.append(f"Unexpected district_type values: {invalid}")
        if not missing and not invalid:
            info.append("District type values look complete.")

    # Tiers
    tier_cols = [
        "college_share_tier",
        "white_share_tier",
        "black_share_tier",
        "hispanic_share_tier",
        "median_income_tier",
    ]

    for col in tier_cols:
        if col not in df.columns:
            continue

        s = normalize_blank_series(df[col])
        unknown = s.isin(["", "Unknown"]).sum()
        invalid = sorted(set(s.unique()) - VALID_TIERS)

        if unknown:
            warnings.append(f"{unknown} districts have missing/Unknown {col}.")

        if invalid:
            warnings.append(f"Unexpected values in {col}: {invalid}")

        if not unknown and not invalid:
            info.append(f"{col} is fully populated.")

    # Incumbency
    if "incumbent_party" in df.columns:
        s = normalize_blank_series(df["incumbent_party"])
        invalid = sorted(set(s.unique()) - VALID_INCUMBENT_PARTIES)
        missing = s.eq("").sum()

        if missing:
            warnings.append(f"{missing} districts have blank incumbent_party.")
        if invalid:
            warnings.append(f"Unexpected incumbent_party values: {invalid}")
        if not missing and not invalid:
            info.append("Incumbent party values look complete.")

    if all(c in df.columns for c in ["dem_candidate_is_incumbent", "gop_candidate_is_incumbent", "district_id"]):
        dem_inc = normalize_bool_series(df["dem_candidate_is_incumbent"])
        gop_inc = normalize_bool_series(df["gop_candidate_is_incumbent"])
        both = df[dem_inc & gop_inc]
        if not both.empty:
            errors.append(
                "Both candidates marked incumbent in: "
                + ", ".join(both["district_id"].head(20).tolist())
            )
        else:
            info.append("No districts have both candidates marked incumbent.")

    if all(c in df.columns for c in ["incumbent_party", "dem_candidate_is_incumbent", "gop_candidate_is_incumbent", "district_id"]):
        party = normalize_blank_series(df["incumbent_party"])
        dem_inc = normalize_bool_series(df["dem_candidate_is_incumbent"])
        gop_inc = normalize_bool_series(df["gop_candidate_is_incumbent"])

        conflicts = df[
            ((party == "D") & gop_inc)
            | ((party == "R") & dem_inc)
        ]

        if not conflicts.empty:
            warnings.append(
                "Incumbent-party conflicts with italic incumbent detection: "
                + ", ".join(conflicts["district_id"].head(20).tolist())
            )
        else:
            info.append("Incumbent-party fields do not conflict with incumbent candidate flags.")

    # Election systems
    if "election_system" in df.columns:
        s = normalize_blank_series(df["election_system"])
        invalid = sorted(set(s.unique()) - VALID_ELECTION_SYSTEMS - {""})
        if invalid:
            warnings.append(f"Unexpected election_system values: {invalid}")

        if "state" in df.columns and "district_id" in df.columns:
            ca_wa_wrong = df[df["state"].isin(["CA", "WA"]) & (s != "top_two")]
            ak_wrong = df[(df["state"] == "AK") & (s != "top_four_rcv")]
            other_top_two = df[~df["state"].isin(["CA", "WA"]) & (s == "top_two")]

            if not ca_wa_wrong.empty:
                warnings.append(
                    "CA/WA districts not marked top_two: "
                    + ", ".join(ca_wa_wrong["district_id"].head(20).tolist())
                )
            if not ak_wrong.empty:
                warnings.append("AK district not marked top_four_rcv.")
            if not other_top_two.empty:
                warnings.append(
                    "Non-CA/WA districts marked top_two: "
                    + ", ".join(other_top_two["district_id"].head(20).tolist())
                )

    if "general_election_party_structure" in df.columns:
        s = normalize_blank_series(df["general_election_party_structure"])
        invalid = sorted(set(s.unique()) - VALID_PARTY_STRUCTURES - {""})
        if invalid:
            warnings.append(f"Unexpected general_election_party_structure values: {invalid}")

    if "party_control_override" in df.columns:
        s = normalize_blank_series(df["party_control_override"]).str.upper()
        invalid = sorted(set(s.unique()) - VALID_PARTY_OVERRIDES)
        if invalid:
            warnings.append(f"Unexpected party_control_override values: {invalid}")

    if all(c in df.columns for c in ["general_election_party_structure", "party_control_override", "district_id"]):
        structure = normalize_blank_series(df["general_election_party_structure"])
        override = normalize_blank_series(df["party_control_override"]).str.upper()

        d_vs_d_missing = df[(structure == "D_vs_D") & (override != "D")]
        r_vs_r_missing = df[(structure == "R_vs_R") & (override != "R")]

        if not d_vs_d_missing.empty:
            warnings.append(
                "D_vs_D races missing D party-control override: "
                + ", ".join(d_vs_d_missing["district_id"].head(20).tolist())
            )

        if not r_vs_r_missing.empty:
            warnings.append(
                "R_vs_R races missing R party-control override: "
                + ", ".join(r_vs_r_missing["district_id"].head(20).tolist())
            )

    # Margin/probability sanity
    if all(c in df.columns for c in ["model_margin_dem", "dem_win_probability", "district_id"]):
        margin = pd.to_numeric(df["model_margin_dem"], errors="coerce")
        prob = pd.to_numeric(df["dem_win_probability"], errors="coerce")

        impossible = df[(prob < -0.001) | (prob > 1.001)]
        if not impossible.empty:
            errors.append(
                "Win probabilities outside 0-1 in: "
                + ", ".join(impossible["district_id"].head(20).tolist())
            )

        weird = df[((margin > 5) & (prob < 0.45)) | ((margin < -5) & (prob > 0.55))]
        if not weird.empty:
            warnings.append(
                "Margin/probability direction looks odd in: "
                + ", ".join(weird["district_id"].head(20).tolist())
            )
        else:
            info.append("Margin/probability directions look plausible.")


def check_national_env(env, errors, warnings, info):
    if env.empty:
        errors.append("house_national_environment_audit.csv is missing or empty.")
        return

    required = [
        "national_environment_margin_dem",
        "national_environment_source_path",
    ]

    add_missing_col_errors(env, required, errors, "house_national_environment_audit.csv")

    if "national_environment_margin_dem" in env.columns:
        val = as_float(env.iloc[-1]["national_environment_margin_dem"])
        if pd.isna(val):
            errors.append("house_national_environment_audit.csv has invalid national_environment_margin_dem.")
        else:
            info.append(f"House national environment imported as {fmt_margin(val)}.")

    if "national_environment_source_path" in env.columns:
        source = str(env.iloc[-1]["national_environment_source_path"])
        if "senate_model" not in source and "national_environment.csv" not in source:
            warnings.append(f"National environment source path looks unusual: {source}")
        else:
            info.append(f"National environment source path: {source}")


def check_outputs(summary, seat_dist, race_stats, errors, warnings, info):
    if summary.empty:
        errors.append("outputs/house_forecast_summary.csv is missing or empty.")
    else:
        required = [
            "expected_dem_seats",
            "median_dem_seats",
            "dem_majority_probability",
            "national_error_sd",
            "state_error_sd",
            "region_error_sd",
            "district_type_error_sd",
            "education_race_error_sd",
        ]
        add_missing_col_errors(summary, required, errors, "house_forecast_summary.csv")

        if "dem_majority_probability" in summary.columns:
            p = as_float(summary.iloc[-1]["dem_majority_probability"])
            if pd.isna(p) or p < 0 or p > 1:
                errors.append("dem_majority_probability is invalid.")
            else:
                info.append(f"Dem majority probability: {p:.1%}")

        if "expected_dem_seats" in summary.columns:
            seats = as_float(summary.iloc[-1]["expected_dem_seats"])
            if pd.isna(seats) or seats < 0 or seats > 435:
                errors.append("expected_dem_seats is invalid.")
            else:
                info.append(f"Expected Dem seats: {seats:.2f}")

    if seat_dist.empty:
        errors.append("outputs/house_seat_distribution.csv is missing or empty.")
    else:
        if "probability" not in seat_dist.columns:
            errors.append("house_seat_distribution.csv missing probability column.")
        else:
            total_prob = pd.to_numeric(seat_dist["probability"], errors="coerce").sum()
            if abs(total_prob - 1.0) > 0.01:
                warnings.append(f"Seat distribution probabilities sum to {total_prob:.3f}, not 1.0.")
            else:
                info.append("Seat distribution probabilities sum to approximately 1.")

    if race_stats.empty:
        errors.append("outputs/house_race_stats.csv is missing or empty.")
    else:
        if len(race_stats) != EXPECTED_DISTRICTS:
            errors.append(f"house_race_stats.csv expected 435 rows, found {len(race_stats)}.")
        else:
            info.append("house_race_stats.csv has 435 rows.")


def check_manual_polls(polls, house, errors, warnings, info):
    if polls.empty:
        info.append("No House manual polls entered yet.")
        return

    required = [
        "state",
        "district",
        "district_id",
        "pollster",
        "dem_pct",
        "gop_pct",
        "end_date",
        "sample_size",
    ]

    add_missing_col_errors(polls, required, errors, "house_manual_polls.csv")

    if "district_id" in polls.columns and "district_id" in house.columns:
        valid_ids = set(normalize_blank_series(house["district_id"]))
        poll_ids = normalize_blank_series(polls["district_id"])
        unmatched = sorted(set(poll_ids) - valid_ids - {""})

        if unmatched:
            warnings.append(
                "Manual polls with district_id not found in house_race_inputs.csv: "
                + ", ".join(unmatched[:20])
            )
        else:
            info.append("Manual poll district IDs match House race inputs.")

    for col in ["dem_pct", "gop_pct"]:
        if col in polls.columns:
            vals = pd.to_numeric(polls[col], errors="coerce")
            bad = vals.isna() | (vals < 0) | (vals > 100)
            if bad.any():
                warnings.append(f"{bad.sum()} manual polls have invalid {col}.")


def main():
    errors = []
    warnings = []
    info = []

    print("House Model Health Check")
    print("========================")

    check_files(errors, warnings, info)

    house = read_csv(HOUSE_INPUTS)
    env = read_csv(HOUSE_NATIONAL_ENV_AUDIT)
    summary = read_csv(HOUSE_FORECAST_SUMMARY)
    seat_dist = read_csv(HOUSE_SEAT_DISTRIBUTION)
    race_stats = read_csv(HOUSE_RACE_STATS)
    polls = read_csv(HOUSE_MANUAL_POLLS)

    check_house_inputs(house, errors, warnings, info)
    check_national_env(env, errors, warnings, info)
    check_outputs(summary, seat_dist, race_stats, errors, warnings, info)
    check_manual_polls(polls, house, errors, warnings, info)

    print_section("Errors", errors)
    print_section("Warnings", warnings)
    print_section("Info", info)

    print()

    if errors:
        print("Health check result: FAIL")
        raise SystemExit(1)

    if warnings:
        print("Health check result: PASS WITH WARNINGS")
        raise SystemExit(0)

    print("Health check result: PASS")


if __name__ == "__main__":
    main()
