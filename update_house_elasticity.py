from pathlib import Path
import pandas as pd
import numpy as np

INPUT_PATH = Path("inputs/house_race_inputs.csv")

# Base elasticity by district type.
# Interpretation: if national environment moves 1 point, how much does this district move?
DISTRICT_TYPE_BASE = {
    "Urban": 0.70,
    "Suburban": 1.10,
    "Exurban": 1.00,
    "Rural": 0.80,
    "Mixed": 0.95,
}

# Modest regional adjustments.
REGION_ADJUSTMENTS = {
    "Northeast": -0.05,
    "Mid-Atlantic": 0.00,
    "Deep South": -0.05,
    "Middle South": 0.00,
    "Urban South": 0.05,
    "Appalachia": -0.05,
    "Midwest": 0.05,
    "Great Plains": -0.05,
    "Mountain West": 0.00,
    "Pacific": -0.05,
    "Northwest": 0.00,
    "Unknown Region": 0.00,
}

MIN_ELASTICITY = 0.55
MAX_ELASTICITY = 1.25


def normalize_text(x, default):
    if pd.isna(x):
        return default

    s = str(x).strip()

    if s == "":
        return default

    return s


def normalize_district_type(x):
    s = normalize_text(x, "Mixed").lower()

    aliases = {
        "urban": "Urban",
        "suburban": "Suburban",
        "exurban": "Exurban",
        "rural": "Rural",
        "mixed": "Mixed",
    }

    return aliases.get(s, normalize_text(x, "Mixed"))


def normalize_region(x):
    s = normalize_text(x, "Unknown Region").lower()

    aliases = {
        "northeast": "Northeast",
        "mid-atlantic": "Mid-Atlantic",
        "mid atlantic": "Mid-Atlantic",
        "deep south": "Deep South",
        "middle south": "Middle South",
        "urban south": "Urban South",
        "appalachia": "Appalachia",
        "midwest": "Midwest",
        "great plains": "Great Plains",
        "mountain west": "Mountain West",
        "pacific": "Pacific",
        "northwest": "Northwest",
        "unknown region": "Unknown Region",
    }

    return aliases.get(s, normalize_text(x, "Unknown Region"))


def baseline_adjustment(abs_baseline):
    """
    Competitive districts are more elastic.
    Very lopsided districts are less elastic.
    """
    if pd.isna(abs_baseline):
        return 0.00

    if abs_baseline <= 5:
        return 0.10

    if abs_baseline <= 15:
        return 0.00

    if abs_baseline <= 25:
        return -0.10

    return -0.20


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Could not find {INPUT_PATH}. Run import_house_model_seed.py first.")

    df = pd.read_csv(INPUT_PATH)

    if df.empty:
        raise ValueError(f"{INPUT_PATH} is empty.")

    required = ["district_id", "pres_2024_margin_dem", "pres_2020_margin_dem"]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"{INPUT_PATH} missing required columns: {missing}")

    if "district_type" not in df.columns:
        df["district_type"] = "Mixed"

    if "region" not in df.columns:
        df["region"] = "Unknown Region"

    if "manual_elasticity_override" not in df.columns:
        df["manual_elasticity_override"] = np.nan

    df["district_type"] = df["district_type"].apply(normalize_district_type)
    df["region"] = df["region"].apply(normalize_region)

    df["pres_2024_margin_dem"] = pd.to_numeric(
        df["pres_2024_margin_dem"],
        errors="coerce"
    )

    df["pres_2020_margin_dem"] = pd.to_numeric(
        df["pres_2020_margin_dem"],
        errors="coerce"
    )

    # Use existing district baseline if present; otherwise calculate the same
    # 70/30 baseline used by recalculate_house_fundamentals.py.
    if "district_partisan_baseline_dem" in df.columns:
        baseline = pd.to_numeric(
            df["district_partisan_baseline_dem"],
            errors="coerce"
        )
    else:
        baseline = pd.Series(np.nan, index=df.index)

    fallback_baseline = (
        0.70 * df["pres_2024_margin_dem"]
        + 0.30 * df["pres_2020_margin_dem"]
    )

    df["elasticity_baseline_margin_dem"] = baseline.fillna(fallback_baseline)

    df["district_type_elasticity_base"] = (
        df["district_type"]
        .map(DISTRICT_TYPE_BASE)
        .fillna(0.95)
    )

    df["partisan_baseline_elasticity_adjustment"] = (
        df["elasticity_baseline_margin_dem"]
        .abs()
        .apply(baseline_adjustment)
    )

    df["region_elasticity_adjustment"] = (
        df["region"]
        .map(REGION_ADJUSTMENTS)
        .fillna(0.00)
    )

    calculated = (
        df["district_type_elasticity_base"]
        + df["partisan_baseline_elasticity_adjustment"]
        + df["region_elasticity_adjustment"]
    )

    calculated = calculated.clip(
        lower=MIN_ELASTICITY,
        upper=MAX_ELASTICITY,
    )

    df["calculated_district_elasticity"] = calculated

    df["manual_elasticity_override"] = pd.to_numeric(
        df["manual_elasticity_override"],
        errors="coerce"
    )

    df["district_elasticity"] = df["manual_elasticity_override"].fillna(
        df["calculated_district_elasticity"]
    )

    df["district_elasticity"] = df["district_elasticity"].clip(
        lower=MIN_ELASTICITY,
        upper=MAX_ELASTICITY,
    )

    notes = []

    for _, row in df.iterrows():
        parts = [
            f"District type base: {row['district_type']}={row['district_type_elasticity_base']:.2f}.",
            f"Partisan baseline adjustment: {row['partisan_baseline_elasticity_adjustment']:+.2f}.",
            f"Region adjustment: {row['region']}={row['region_elasticity_adjustment']:+.2f}.",
        ]

        if pd.notna(row["manual_elasticity_override"]):
            parts.append(f"Manual override used: {row['manual_elasticity_override']:.2f}.")
        else:
            parts.append("No manual override.")

        parts.append(f"Final elasticity: {row['district_elasticity']:.2f}.")

        notes.append(" ".join(parts))

    df["elasticity_notes"] = notes

    df.to_csv(INPUT_PATH, index=False)

    print(f"Updated House district elasticity in {INPUT_PATH}")

    print()
    print("Elasticity summary:")
    print(df["district_elasticity"].describe().to_string())

    print()
    print("Elasticity by district type:")
    print(
        df.groupby("district_type")["district_elasticity"]
        .agg(["count", "mean", "min", "max"])
        .round(3)
        .to_string()
    )

    print()
    print("Elasticity by region:")
    print(
        df.groupby("region")["district_elasticity"]
        .agg(["count", "mean", "min", "max"])
        .round(3)
        .to_string()
    )

    print()
    print("Most elastic districts:")
    cols = [
        "district_id",
        "region",
        "district_type",
        "elasticity_baseline_margin_dem",
        "district_elasticity",
        "elasticity_notes",
    ]
    print(df.sort_values("district_elasticity", ascending=False)[cols].head(20).to_string(index=False))

    print()
    print("Least elastic districts:")
    print(df.sort_values("district_elasticity", ascending=True)[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
