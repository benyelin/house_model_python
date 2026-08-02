from pathlib import Path
import pandas as pd
import numpy as np

INPUTS = Path("inputs")
OUTPUTS = Path("outputs") if "OUTPUTS" not in globals() else OUTPUTS
HOUSE_CANDIDATE_WAR_AUDIT = OUTPUTS / "house_candidate_war_audit.csv"
HOUSE_INPUT_PATH = INPUTS / "house_race_inputs.csv"

# The Senate model's shared national-environment file is the single
# production source of truth for both Senate and House forecasts.
NATIONAL_ENVIRONMENT_PATH = Path(
    "/Users/benyelin/Desktop/Desktop - Ben’s MacBook Air/"
    "senate_model_python_Q1_auto_calendar_candidate_refresh/"
    "inputs/national_environment.csv"
)

# House district baseline weights.
# Because of redistricting, use 2024 and 2020 only.
WEIGHT_2024 = 0.70
WEIGHT_2020 = 0.30

# First-pass House assumptions.
DEFAULT_DISTRICT_ELASTICITY = 0.90

# House incumbency is real but somewhat weaker than old-school House models.
DEM_INCUMBENCY_ADJUSTMENT = 2.0
GOP_INCUMBENCY_ADJUSTMENT = -2.0
OPEN_SEAT_INCUMBENCY_ADJUSTMENT = 0.0



def read_house_calibration_setting(setting_name, default):
    """
    Read House calibration settings from inputs/house_calibration_settings.csv.
    Returns default if the file or setting is missing.
    """
    from pathlib import Path
    import pandas as pd

    settings_path = Path("inputs/house_calibration_settings.csv")

    if not settings_path.exists():
        return default

    try:
        settings = pd.read_csv(settings_path)
    except Exception:
        return default

    if settings.empty or "setting" not in settings.columns or "value" not in settings.columns:
        return default

    rows = settings[settings["setting"].astype(str).str.strip() == setting_name]

    if rows.empty:
        return default

    try:
        return float(rows.iloc[0]["value"])
    except Exception:
        return default


def find_national_environment_path():
    if not NATIONAL_ENVIRONMENT_PATH.exists():
        raise FileNotFoundError(
            "Shared Senate national-environment file not found:\n"
            f"  {NATIONAL_ENVIRONMENT_PATH}"
        )

    return NATIONAL_ENVIRONMENT_PATH


def read_national_environment():
    path = find_national_environment_path()
    env = pd.read_csv(path)

    if env.empty:
        raise ValueError(f"{path} is empty")

    if "national_environment_margin_dem" not in env.columns:
        raise ValueError(f"{path} missing national_environment_margin_dem")

    row = env.iloc[-1]

    national_environment = pd.to_numeric(
        row["national_environment_margin_dem"],
        errors="coerce"
    )

    if pd.isna(national_environment):
        raise ValueError(f"national_environment_margin_dem is blank or invalid in {path}")

    metadata = {
        "national_environment_source_path": str(path),
        "national_environment_margin_dem": float(national_environment),
        "as_of_date": row.get("as_of_date", ""),
        "generic_ballot_margin_dem": row.get("generic_ballot_margin_dem", np.nan),
        "presidential_approval": row.get("presidential_approval", np.nan),
        "presidential_disapproval": row.get("presidential_disapproval", np.nan),
        "presidential_net_approval": row.get("presidential_net_approval", np.nan),
        "approval_adjustment_dem": row.get("approval_adjustment_dem", np.nan),
        "midterm_adjustment_dem": row.get("midterm_adjustment_dem", np.nan),
        "source_notes": row.get("source_notes", ""),
    }

    return float(national_environment), metadata


def parse_bool(x):
    if pd.isna(x):
        return False

    if isinstance(x, bool):
        return x

    return str(x).strip().lower() in ["true", "1", "yes", "y"]


def ensure_numeric(df, col, default=np.nan):
    if col not in df.columns:
        df[col] = default

    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def ensure_column(df, col, default):
    if col not in df.columns:
        df[col] = default

    return df




def read_house_calibration_setting_value(setting_name, default=0.0):
    """
    Small local settings reader used by optional candidate WAR integration.
    """
    from pathlib import Path
    import pandas as pd

    settings_path = Path("inputs/house_calibration_settings.csv")

    if not settings_path.exists():
        return default

    try:
        settings = pd.read_csv(settings_path)
    except Exception:
        return default

    if settings.empty or "setting" not in settings.columns or "value" not in settings.columns:
        return default

    mask = settings["setting"].astype(str).str.strip().eq(setting_name)

    if not mask.any():
        return default

    try:
        return float(settings.loc[mask, "value"].iloc[0])
    except Exception:
        return default


def merge_candidate_war_adjustments(df):
    """
    Merge candidate_war_adjustment_dem into race fundamentals data.

    This is audit-safe:
    - If use_candidate_war_adjustments is 0, all WAR adjustments are set to 0.
    - If the WAR audit file is missing, all WAR adjustments are set to 0.
    - If active, candidate_war_adjustment_dem is added to
      objective_candidate_quality_adjustment_dem.
    """
    import pandas as pd
    from pathlib import Path

    out = df.copy()

    use_war = read_house_calibration_setting_value("use_candidate_war_adjustments", 0.0) >= 0.5

    out["use_candidate_war_adjustments"] = int(use_war)
    out["candidate_war_adjustment_dem"] = 0.0
    out["candidate_war_match_status"] = "WAR inactive"

    war_path = Path("outputs/house_candidate_war_audit.csv")

    if not use_war:
        return out

    if not war_path.exists():
        out["candidate_war_match_status"] = "WAR active but audit missing"
        return out

    war = pd.read_csv(war_path)

    if war.empty or "district_id" not in war.columns or "candidate_war_adjustment_dem" not in war.columns:
        out["candidate_war_match_status"] = "WAR active but malformed audit"
        return out

    war_keep = [
        "district_id",
        "candidate_war_adjustment_dem",
        "war_match_status",
        "dem_war_name",
        "gop_war_name",
        "dem_candidate_war",
        "gop_candidate_war",
    ]

    war_keep = [c for c in war_keep if c in war.columns]
    war = war[war_keep].copy()

    war["district_id"] = war["district_id"].astype(str).str.strip().str.upper()
    out["district_id"] = out["district_id"].astype(str).str.strip().str.upper()

    out = out.merge(war, on="district_id", how="left", suffixes=("", "_war"))

    out["candidate_war_adjustment_dem"] = pd.to_numeric(
        out["candidate_war_adjustment_dem"],
        errors="coerce",
    ).fillna(0.0)

    if "war_match_status" in out.columns:
        out["candidate_war_match_status"] = out["war_match_status"].fillna("No WAR match")
    else:
        out["candidate_war_match_status"] = "No WAR match"

    if "objective_candidate_quality_adjustment_dem_before_war" in df.columns:
        base_objective_quality = pd.to_numeric(
            df["objective_candidate_quality_adjustment_dem_before_war"],
            errors="coerce",
        )
    elif "objective_candidate_quality_adjustment_dem" in df.columns:
        base_objective_quality = pd.to_numeric(
            df["objective_candidate_quality_adjustment_dem"],
            errors="coerce",
        )
    else:
        base_objective_quality = pd.Series(
            0.0,
            index=out.index,
            dtype=float,
        )

    base_objective_quality = base_objective_quality.fillna(0.0)

    out["objective_candidate_quality_adjustment_dem_before_war"] = (
        base_objective_quality.to_numpy()
    )

    out["objective_candidate_quality_adjustment_dem"] = (
        out["objective_candidate_quality_adjustment_dem_before_war"]
        + out["candidate_war_adjustment_dem"]
    )

    return out


def calculate_incumbency_adjustment(row, incumbency_points=None):
    if incumbency_points is None:
        incumbency_points = abs(DEM_INCUMBENCY_ADJUSTMENT)

    dem_inc = parse_bool(row.get("dem_candidate_is_incumbent", False))
    gop_inc = parse_bool(row.get("gop_candidate_is_incumbent", False))

    if dem_inc and not gop_inc:
        return incumbency_points

    if gop_inc and not dem_inc:
        return -incumbency_points

    return OPEN_SEAT_INCUMBENCY_ADJUSTMENT




def apply_candidate_war_to_house_fundamentals(df):
    """
    Optional candidate WAR integration.

    If use_candidate_war_adjustments = 1 in inputs/house_calibration_settings.csv,
    merge outputs/house_candidate_war_audit.csv and add candidate_war_adjustment_dem
    to candidate_quality_adjustment_dem.

    If off/missing, leaves candidate_quality_adjustment_dem unchanged and sets
    candidate_war_adjustment_dem to 0.
    """
    import pandas as pd
    from pathlib import Path

    out = df.copy()

    settings_path = Path("inputs/house_calibration_settings.csv")
    war_path = Path("outputs/house_candidate_war_audit.csv")

    use_war = 0.0

    if settings_path.exists():
        try:
            settings = pd.read_csv(settings_path)
            mask = settings["setting"].astype(str).str.strip().eq("use_candidate_war_adjustments")
            if mask.any():
                use_war = float(settings.loc[mask, "value"].iloc[0])
        except Exception:
            use_war = 0.0

    out["use_candidate_war_adjustments"] = int(use_war >= 0.5)
    out["candidate_war_adjustment_dem"] = 0.0
    out["candidate_war_match_status"] = "WAR inactive"

    if use_war < 0.5:
        return out

    if not war_path.exists():
        out["candidate_war_match_status"] = "WAR active but audit missing"
        return out

    try:
        war = pd.read_csv(war_path)
    except Exception:
        out["candidate_war_match_status"] = "WAR active but unreadable audit"
        return out

    if war.empty or "district_id" not in war.columns or "candidate_war_adjustment_dem" not in war.columns:
        out["candidate_war_match_status"] = "WAR active but malformed audit"
        return out

    keep = [
        "district_id",
        "candidate_war_adjustment_dem",
        "war_match_status",
        "dem_war_name",
        "gop_war_name",
        "dem_candidate_war",
        "gop_candidate_war",
    ]
    keep = [c for c in keep if c in war.columns]

    war = war[keep].copy()
    war["district_id"] = war["district_id"].astype(str).str.strip().str.upper()
    out["district_id"] = out["district_id"].astype(str).str.strip().str.upper()

    # Make this merge idempotent. The pipeline may run this script after
    # candidate WAR columns already exist from a previous pass/import.
    stale_war_cols = [
        "candidate_war_adjustment_dem",
        "candidate_war_adjustment_dem_from_war",
        "war_match_status",
        "dem_war_name",
        "gop_war_name",
        "dem_candidate_war",
        "gop_candidate_war",
        "candidate_war_match_status",
        "candidate_quality_adjustment_dem_before_war",
    ]

    out = out.drop(
        columns=[c for c in stale_war_cols if c in out.columns],
        errors="ignore",
    )

    out = out.merge(war, on="district_id", how="left")

    # If merge created a duplicate candidate_war_adjustment_dem_from_war, prefer that.
    if "candidate_war_adjustment_dem_from_war" in out.columns:
        out["candidate_war_adjustment_dem"] = pd.to_numeric(
            out["candidate_war_adjustment_dem_from_war"],
            errors="coerce",
        ).fillna(0.0)
    else:
        out["candidate_war_adjustment_dem"] = pd.to_numeric(
            out["candidate_war_adjustment_dem"],
            errors="coerce",
        ).fillna(0.0)

    if "war_match_status" in out.columns:
        out["candidate_war_match_status"] = out["war_match_status"].fillna("No WAR match")
    else:
        out["candidate_war_match_status"] = "No WAR match"

    # Recover the underlying candidate-quality value before WAR.
    #
    # On the first clean run, candidate_quality_adjustment_dem_before_war may
    # not exist, so candidate_quality_adjustment_dem is used as the source.
    #
    # On later runs, candidate_quality_adjustment_dem already includes WAR.
    # Reusing that field would compound WAR repeatedly, so prefer the saved
    # pre-WAR value from the incoming dataframe.
    if "candidate_quality_adjustment_dem_before_war" in df.columns:
        base_candidate_quality = pd.to_numeric(
            df["candidate_quality_adjustment_dem_before_war"],
            errors="coerce",
        )
    elif "candidate_quality_adjustment_dem" in df.columns:
        base_candidate_quality = pd.to_numeric(
            df["candidate_quality_adjustment_dem"],
            errors="coerce",
        )
    else:
        base_candidate_quality = pd.Series(
            0.0,
            index=out.index,
            dtype=float,
        )

    base_candidate_quality = base_candidate_quality.fillna(0.0)

    out["candidate_quality_adjustment_dem_before_war"] = (
        base_candidate_quality.to_numpy()
    )

    out["candidate_quality_adjustment_dem"] = (
        out["candidate_quality_adjustment_dem_before_war"]
        + out["candidate_war_adjustment_dem"]
    )

    return out




def apply_house_poll_spillover_adjustments(df):
    """
    Optional House poll spillover/context signal integration.

    If use_house_poll_spillover_adjustments = 1 in inputs/house_calibration_settings.csv,
    merge outputs/house_poll_spillover_signal.csv and add poll_spillover_adjustment_dem
    to fundamentals_margin_dem.

    If off/missing, keep the column visible but set the active adjustment to 0.
    """
    import pandas as pd
    from pathlib import Path

    out = df.copy()

    settings_path = Path("inputs/house_calibration_settings.csv")
    signal_path = Path("outputs/house_poll_spillover_signal.csv")

    use_spillover = 0.0

    if settings_path.exists():
        try:
            settings = pd.read_csv(settings_path)
            mask = settings["setting"].astype(str).str.strip().eq("use_house_poll_spillover_adjustments")
            if mask.any():
                use_spillover = float(settings.loc[mask, "value"].iloc[0])
        except Exception:
            use_spillover = 0.0

    out["use_house_poll_spillover_adjustments"] = int(use_spillover >= 0.5)
    out["poll_spillover_adjustment_dem"] = 0.0
    out["poll_spillover_raw_adjustment_dem"] = 0.0
    out["poll_spillover_source_count"] = 0
    out["poll_spillover_notes"] = "Poll spillover inactive"

    if use_spillover < 0.5:
        return out

    if not signal_path.exists():
        out["poll_spillover_notes"] = "Poll spillover active but signal file missing"
        return out

    try:
        signal = pd.read_csv(signal_path)
    except Exception:
        out["poll_spillover_notes"] = "Poll spillover active but signal file unreadable"
        return out

    if signal.empty or "district_id" not in signal.columns or "poll_spillover_adjustment_dem" not in signal.columns:
        out["poll_spillover_notes"] = "Poll spillover active but signal file malformed"
        return out

    keep = [
        "district_id",
        "poll_spillover_adjustment_dem",
        "poll_spillover_raw_adjustment_dem",
        "poll_spillover_source_count",
        "poll_spillover_abs_signal",
        "poll_spillover_cap",
        "poll_spillover_days_out",
        "poll_spillover_time_weight",
        "poll_spillover_target_has_polling",
        "poll_spillover_notes",
    ]

    keep = [c for c in keep if c in signal.columns]

    signal = signal[keep].copy()
    signal["district_id"] = signal["district_id"].astype(str).str.strip().str.upper()
    out["district_id"] = out["district_id"].astype(str).str.strip().str.upper()

    # Drop stale signal-merge columns from prior runs so this step is idempotent.
    stale_signal_cols = [
        c for c in out.columns
        if c.endswith("_from_signal")
        or c in {
            "poll_spillover_adjustment_dem_from_signal",
            "poll_spillover_raw_adjustment_dem_from_signal",
            "poll_spillover_source_count_from_signal",
            "poll_spillover_notes_from_signal",
            "poll_spillover_abs_signal_from_signal",
            "poll_spillover_cap_from_signal",
            "poll_spillover_days_out_from_signal",
            "poll_spillover_time_weight_from_signal",
            "poll_spillover_target_has_polling_from_signal",
        }
    ]
    if stale_signal_cols:
        out = out.drop(columns=stale_signal_cols, errors="ignore")

    out = out.merge(signal, on="district_id", how="left", suffixes=("", "_from_signal"))

    # Prefer merged signal values if suffixes were created.
    for col in [
        "poll_spillover_adjustment_dem",
        "poll_spillover_raw_adjustment_dem",
        "poll_spillover_source_count",
        "poll_spillover_abs_signal",
        "poll_spillover_cap",
        "poll_spillover_days_out",
        "poll_spillover_time_weight",
        "poll_spillover_target_has_polling",
        "poll_spillover_notes",
    ]:
        from_signal = f"{col}_from_signal"
        if from_signal in out.columns:
            out[col] = out[from_signal]

    out["poll_spillover_adjustment_dem"] = pd.to_numeric(
        out["poll_spillover_adjustment_dem"],
        errors="coerce",
    ).fillna(0.0)

    out["poll_spillover_raw_adjustment_dem"] = pd.to_numeric(
        out.get("poll_spillover_raw_adjustment_dem", 0.0),
        errors="coerce",
    ).fillna(0.0)

    out["poll_spillover_source_count"] = pd.to_numeric(
        out.get("poll_spillover_source_count", 0),
        errors="coerce",
    ).fillna(0).astype(int)

    if "poll_spillover_notes" in out.columns:
        out["poll_spillover_notes"] = out["poll_spillover_notes"].fillna("No spillover signal")
    else:
        out["poll_spillover_notes"] = "No spillover signal"

    if "fundamentals_margin_dem" in out.columns:
        out["fundamentals_margin_dem_before_poll_spillover"] = pd.to_numeric(
            out["fundamentals_margin_dem"],
            errors="coerce",
        ).fillna(0.0)

        out["fundamentals_margin_dem"] = (
            out["fundamentals_margin_dem_before_poll_spillover"]
            + out["poll_spillover_adjustment_dem"]
        )

    return out


def recalculate_house_fundamentals_dataframe(
    df,
    national_environment=None,
    env_metadata=None,
):
    """
    Recalculate House fundamentals using the production calculation pipeline.

    This function performs no file writes. It accepts an input DataFrame and
    returns a recalculated copy.

    When national_environment and env_metadata are omitted, the production
    shared national-environment file is loaded through read_national_environment().
    Historical replay callers may instead supply a cycle-specific environment
    and metadata directly.
    """
    df = df.copy()

    if df.empty:
        raise ValueError("House fundamentals input DataFrame is empty")

    if national_environment is None and env_metadata is None:
        national_environment, env_metadata = read_national_environment()
    elif national_environment is None or env_metadata is None:
        raise ValueError(
            "national_environment and env_metadata must either both be "
            "provided or both be omitted."
        )
    else:
        national_environment = float(national_environment)
        env_metadata = dict(env_metadata)

    env_metadata.setdefault(
        "national_environment_source_path",
        "provided directly",
    )
    env_metadata.setdefault(
        "national_environment_margin_dem",
        float(national_environment),
    )

    required = [
        "state",
        "district",
        "district_id",
        "pres_2024_margin_dem",
        "pres_2020_margin_dem",
        "dem_candidate_is_incumbent",
        "gop_candidate_is_incumbent",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"{HOUSE_INPUT_PATH} missing required columns: {missing}")

    df["state"] = df["state"].astype(str).str.strip().str.upper()
    df["district"] = df["district"].astype(str).str.strip()
    df["district_id"] = df["district_id"].astype(str).str.strip()

    for col in [
        "pres_2024_margin_dem",
        "pres_2020_margin_dem",
        "district_partisan_baseline_dem",
        "district_elasticity",
        "national_environment_margin_dem",
        "district_environment_adjustment_dem",
        "state_environment_adjustment_dem",
        "incumbency_adjustment_dem",
        "candidate_quality_adjustment_dem",
        "special_adjustment_dem",
        "fundamentals_margin_dem",
        "polling_margin_dem",
        "model_margin_dem",
        "dem_win_probability",
    ]:
        df = ensure_numeric(df, col)

    for col in [
        "state_environment_adjustment_dem",
        "candidate_quality_adjustment_dem",
        "special_adjustment_dem",
    ]:
        df[col] = df[col].fillna(0.0)

    # Preserve the supplied district-specific elasticity for auditing,
    # but use a neutral elasticity in production. Leakage-safe historical
    # replay found that district-specific values worsened margin MAE,
    # RMSE, Brier score, and log loss relative to uniform elasticity 1.0.
    df["district_elasticity_input"] = (
        df["district_elasticity"].fillna(
            DEFAULT_DISTRICT_ELASTICITY
        )
    )
    df["district_elasticity"] = 1.0

    # District partisan baseline from presidential margins.
    #
    # Normalize each district result against its election year's
    # national two-party presidential margin. This isolates district
    # partisanship from the national political environment, which is
    # imported separately from the Senate/shared environment model.
    national_pres_2024_margin_dem = read_house_calibration_setting(
        "national_pres_2024_margin_dem",
        -1.50,
    )
    national_pres_2020_margin_dem = read_house_calibration_setting(
        "national_pres_2020_margin_dem",
        4.52,
    )

    has_pres = (
        df["pres_2024_margin_dem"].notna()
        & df["pres_2020_margin_dem"].notna()
    )

    df["national_pres_2024_margin_dem"] = (
        national_pres_2024_margin_dem
    )
    df["national_pres_2020_margin_dem"] = (
        national_pres_2020_margin_dem
    )

    df["pres_2024_relative_to_national_dem"] = (
        df["pres_2024_margin_dem"]
        - df["national_pres_2024_margin_dem"]
    )
    df["pres_2020_relative_to_national_dem"] = (
        df["pres_2020_margin_dem"]
        - df["national_pres_2020_margin_dem"]
    )

    df.loc[
        has_pres,
        "district_partisan_baseline_dem"
    ] = (
        WEIGHT_2024
        * df.loc[
            has_pres,
            "pres_2024_relative_to_national_dem"
        ]
        + WEIGHT_2020
        * df.loc[
            has_pres,
            "pres_2020_relative_to_national_dem"
        ]
    )

    # If somehow missing presidential margins, keep existing baseline if present.
    missing_baseline = df["district_partisan_baseline_dem"].isna()

    if missing_baseline.any():
        print(
            "WARNING: Some districts are missing presidential margins/baseline: "
            + ", ".join(df.loc[missing_baseline, "district_id"].head(20).tolist())
            + (" ..." if missing_baseline.sum() > 20 else "")
        )

    for col, default in [
        ("region", "Unknown Region"),
        ("district_type", "Mixed"),
        ("state_error_group", None),
        ("region_error_group", None),
        ("district_type_error_group", None),
    ]:
        if col not in df.columns:
            df[col] = default

    df["region"] = df["region"].fillna("Unknown Region").astype(str).str.strip()
    df["district_type"] = df["district_type"].fillna("Mixed").astype(str).str.strip()

    df["state_error_group"] = df["state_error_group"].fillna(df["state"]).astype(str).str.strip().str.upper()
    df["region_error_group"] = df["region_error_group"].fillna(df["region"]).astype(str).str.strip()
    df["district_type_error_group"] = df["district_type_error_group"].fillna(df["district_type"]).astype(str).str.strip()

    # Imported environment comes from the Senate/shared national environment.
    # For House races, apply a transparent multiplier before district elasticity.
    house_environment_multiplier = read_house_calibration_setting(
        "house_environment_multiplier",
        0.85,
    )

    df["imported_national_environment_margin_dem"] = national_environment
    df["house_environment_multiplier"] = house_environment_multiplier
    # The imported shared national environment is already calibrated
    # from the raw generic ballot. Do not apply the House coefficient twice.
    df["house_national_environment_used_dem"] = (
        df["imported_national_environment_margin_dem"]
    )

    df["national_environment_margin_dem"] = df["house_national_environment_used_dem"]

    df["district_environment_adjustment_dem"] = (
        df["house_national_environment_used_dem"]
        * df["district_elasticity"]
    )

    # Incumbency from italic detection/importer flags.
    # Generic House incumbency is configurable because House incumbency
    # is weaker than it used to be and should be calibrated.
    generic_house_incumbency_points = read_house_calibration_setting(
        "generic_house_incumbency_points",
        abs(DEM_INCUMBENCY_ADJUSTMENT),
    )

    df["generic_house_incumbency_points"] = generic_house_incumbency_points

    df["incumbency_adjustment_dem"] = df.apply(
        lambda row: calculate_incumbency_adjustment(
            row,
            generic_house_incumbency_points,
        ),
        axis=1,
    )

    df = apply_candidate_war_to_house_fundamentals(df)

    df["fundamentals_margin_dem"] = (
        df["district_partisan_baseline_dem"]
        + df["district_environment_adjustment_dem"]
        + df["state_environment_adjustment_dem"]
        + df["incumbency_adjustment_dem"]
        + df["candidate_quality_adjustment_dem"]
        + df["special_adjustment_dem"]
    )

    df = apply_house_poll_spillover_adjustments(df)

    # For now, model margin equals fundamentals unless polling is later added.
    df["model_margin_dem"] = df["fundamentals_margin_dem"]

    # Simple first-pass win probability conversion.
    # This is only a placeholder until we build the simulation engine.
    probability_scale = 6.0
    df["dem_win_probability"] = 1 / (
        1 + np.exp(-df["model_margin_dem"] / probability_scale)
    )

    df["fundamentals_notes"] = (
        "House fundamentals calculated as "
        f"{WEIGHT_2024:.0%}*2024 presidential margin + "
        f"{WEIGHT_2020:.0%}*2020 presidential margin + "
        "national environment * district elasticity + state environment adjustment + incumbency + candidate quality + special adjustment."
    )

    df["national_environment_source_path"] = env_metadata["national_environment_source_path"]

    return df


def main():
    if not HOUSE_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {HOUSE_INPUT_PATH}. "
            "Run import_house_model_seed.py first."
        )

    input_df = pd.read_csv(HOUSE_INPUT_PATH)

    if input_df.empty:
        raise ValueError(f"{HOUSE_INPUT_PATH} is empty")

    national_environment, env_metadata = read_national_environment()

    df = recalculate_house_fundamentals_dataframe(
        input_df,
        national_environment=national_environment,
        env_metadata=env_metadata,
    )

    df.to_csv(HOUSE_INPUT_PATH, index=False)

    # Write national environment audit file locally for the House project.
    audit_path = INPUTS / "house_national_environment_audit.csv"
    pd.DataFrame([env_metadata]).to_csv(audit_path, index=False)

    print(f"Updated House fundamentals in {HOUSE_INPUT_PATH}")
    print(f"Read national environment from: {env_metadata['national_environment_source_path']}")
    print(f"National environment used: {national_environment:+.2f}")
    print(f"Wrote audit file: {audit_path}")

    print()
    print("Topline counts:")
    print(f"Districts: {len(df)}")
    print(f"Missing district baseline: {df['district_partisan_baseline_dem'].isna().sum()}")
    print(f"Dem incumbents detected: {df['dem_candidate_is_incumbent'].apply(parse_bool).sum()}")
    print(f"GOP incumbents detected: {df['gop_candidate_is_incumbent'].apply(parse_bool).sum()}")
    print(f"Open/no incumbent candidate listed: {(df['incumbency_adjustment_dem'] == 0).sum()}")

    print()
    print("Most competitive first-pass districts:")
    preview = df.copy()
    preview["distance_to_50"] = (preview["dem_win_probability"] - 0.5).abs()
    preview = preview.sort_values("distance_to_50").head(20)

    cols = [
        "district_id",
        "dem_candidate",
        "gop_candidate",
        "district_partisan_baseline_dem",
        "district_environment_adjustment_dem",
        "state_environment_adjustment_dem",
        "incumbency_adjustment_dem",
        "fundamentals_margin_dem",
        "dem_win_probability",
    ]

    print(preview[cols].to_string(index=False))


if __name__ == "__main__":
    main()
