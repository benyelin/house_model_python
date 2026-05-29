from pathlib import Path
import pandas as pd
import numpy as np

INPUTS = Path("inputs")
HOUSE_INPUT_PATH = INPUTS / "house_race_inputs.csv"

# Prefer local file if one exists, but default to the Senate model's shared national environment.
NATIONAL_ENVIRONMENT_CANDIDATES = [
    Path("inputs/national_environment.csv"),
    Path("../senate_model_python_Q1_auto_calendar_candidate_refresh/inputs/national_environment.csv"),
]

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


def find_national_environment_path():
    for path in NATIONAL_ENVIRONMENT_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find national_environment.csv. Expected either:\n"
        "  inputs/national_environment.csv\n"
        "or\n"
        "  ../senate_model_python_Q1_auto_calendar_candidate_refresh/inputs/national_environment.csv"
    )


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


def calculate_incumbency_adjustment(row):
    dem_inc = parse_bool(row.get("dem_candidate_is_incumbent", False))
    gop_inc = parse_bool(row.get("gop_candidate_is_incumbent", False))

    if dem_inc and not gop_inc:
        return DEM_INCUMBENCY_ADJUSTMENT

    if gop_inc and not dem_inc:
        return GOP_INCUMBENCY_ADJUSTMENT

    return OPEN_SEAT_INCUMBENCY_ADJUSTMENT


def main():
    if not HOUSE_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {HOUSE_INPUT_PATH}. Run import_house_model_seed.py first."
        )

    df = pd.read_csv(HOUSE_INPUT_PATH)

    if df.empty:
        raise ValueError(f"{HOUSE_INPUT_PATH} is empty")

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
        "candidate_quality_adjustment_dem",
        "special_adjustment_dem",
    ]:
        df[col] = df[col].fillna(0.0)

    df["district_elasticity"] = df["district_elasticity"].fillna(DEFAULT_DISTRICT_ELASTICITY)

    national_environment, env_metadata = read_national_environment()

    # District partisan baseline from presidential margins.
    has_pres = (
        df["pres_2024_margin_dem"].notna()
        & df["pres_2020_margin_dem"].notna()
    )

    df.loc[
        has_pres,
        "district_partisan_baseline_dem"
    ] = (
        WEIGHT_2024 * df.loc[has_pres, "pres_2024_margin_dem"]
        + WEIGHT_2020 * df.loc[has_pres, "pres_2020_margin_dem"]
    )

    # If somehow missing presidential margins, keep existing baseline if present.
    missing_baseline = df["district_partisan_baseline_dem"].isna()

    if missing_baseline.any():
        print(
            "WARNING: Some districts are missing presidential margins/baseline: "
            + ", ".join(df.loc[missing_baseline, "district_id"].head(20).tolist())
            + (" ..." if missing_baseline.sum() > 20 else "")
        )

    df["national_environment_margin_dem"] = national_environment

    df["district_environment_adjustment_dem"] = (
        df["national_environment_margin_dem"]
        * df["district_elasticity"]
    )

    # Incumbency from italic detection/importer flags.
    df["incumbency_adjustment_dem"] = df.apply(
        calculate_incumbency_adjustment,
        axis=1,
    )

    df["fundamentals_margin_dem"] = (
        df["district_partisan_baseline_dem"]
        + df["district_environment_adjustment_dem"]
        + df["incumbency_adjustment_dem"]
        + df["candidate_quality_adjustment_dem"]
        + df["special_adjustment_dem"]
    )

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
        "national environment * district elasticity + incumbency + candidate quality + special adjustment."
    )

    df["national_environment_source_path"] = env_metadata["national_environment_source_path"]

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
        "incumbency_adjustment_dem",
        "fundamentals_margin_dem",
        "dem_win_probability",
    ]

    print(preview[cols].to_string(index=False))


if __name__ == "__main__":
    main()
