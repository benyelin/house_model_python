from pathlib import Path
import pandas as pd

EXCEL_PATH = Path("House Model Data.xlsx")
CSV_PATH = Path("inputs/house_race_inputs.csv")
BACKUP_PATH = Path("inputs/house_race_inputs.before_excel_import.csv")
AUDIT_PATH = Path("outputs/house_excel_import_audit.csv")
MISMATCH_PATH = Path("outputs/house_excel_import_unmatched_rows.csv")

SHEET_NAME = "House Model Data"

# Excel column -> model CSV column
COLUMN_MAP = {
    "State": "state",
    "District": "district",
    "Region": "region",
    "District Type": "district_type",
    "State Environment Adjustment": "state_environment_adjustment_dem",
    "College Share Tier": "college_share_tier",
    "White Share Tier": "white_share_tier",
    "Black Share Tier": "black_share_tier",
    "Hispanic Share Tier": "hispanic_share_tier",
    "Median Income Tier": "median_income_tier",
    "Election System": "election_system",
    "General Election Party Structure": "general_election_party_structure",
    "Party Control Override": "party_control_override",
    "Election System Notes": "election_system_notes",
    "Incumbent": "incumbent_raw",
    "Incumbent Party": "incumbent_party_raw",
    "Dem Candidate": "dem_candidate",
    "GOP Candidate": "gop_candidate",
    "Other": "other_candidate",
    "2024 Margin": "pres_2024_margin_dem",
    "2020 Margin": "pres_2020_margin_dem",
    "GenBallot Adjusted Margin": "genballot_adjusted_margin_dem",
}

def clean_text_value(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return pd.NA
    return s

def norm_state(x):
    return clean_text_value(x).upper() if not pd.isna(clean_text_value(x)) else ""

def norm_district(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    if s in {"AT-LARGE", "AT LARGE", "AL"}:
        return "AL"
    try:
        return str(int(float(s)))
    except Exception:
        return s

def main():
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Missing {EXCEL_PATH}")
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing {CSV_PATH}")

    xl = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
    model = pd.read_csv(CSV_PATH)

    xl.columns = [str(c).strip() for c in xl.columns]
    model.columns = [str(c).strip() for c in model.columns]

    CSV_PATH.parent.mkdir(exist_ok=True)
    AUDIT_PATH.parent.mkdir(exist_ok=True)

    model.to_csv(BACKUP_PATH, index=False)

    # Determine which states are at-large in the existing model CSV.
    model["state_key"] = model["state"].apply(norm_state)
    model["district_key"] = model["district"].apply(norm_district)

    at_large_states = set(
        model.loc[model["district_key"].eq("AL"), "state_key"]
        .dropna()
        .astype(str)
        .str.upper()
    )

    xl["state_key"] = xl["State"].apply(norm_state)
    xl["district_key_raw"] = xl["District"].apply(norm_district)

    # If model uses AL for a state and Excel uses 1, treat Excel 1 as AL.
    xl["district_key"] = xl.apply(
        lambda r: "AL"
        if r["state_key"] in at_large_states and r["district_key_raw"] == "1"
        else r["district_key_raw"],
        axis=1,
    )

    # Prepare Excel update frame.
    usable_cols = [c for c in COLUMN_MAP if c in xl.columns and COLUMN_MAP[c] in model.columns]
    update = xl[["state_key", "district_key"] + usable_cols].copy()
    update = update.rename(columns={c: COLUMN_MAP[c] for c in usable_cols})

    # Do not overwrite model state/district display columns from Excel.
    update = update.drop(columns=["state", "district"], errors="ignore")

    # Merge updates.
    merged = model.merge(
        update,
        on=["state_key", "district_key"],
        how="left",
        suffixes=("", "_excel"),
        indicator=True,
    )

    audit_rows = []

    for excel_col, model_col in COLUMN_MAP.items():
        if model_col in {"state", "district"}:
            continue
        excel_update_col = f"{model_col}_excel"
        if excel_update_col not in merged.columns:
            continue

        before = merged[model_col].copy()
        newvals = merged[excel_update_col]

        # Update only where Excel has a nonblank value.
        has_value = newvals.notna() & newvals.astype(str).str.strip().ne("")
        merged.loc[has_value, model_col] = newvals.loc[has_value]

        changed = before.fillna("").astype(str).str.strip().ne(
            merged[model_col].fillna("").astype(str).str.strip()
        )

        for _, r in merged.loc[changed, ["state", "district", model_col]].iterrows():
            audit_rows.append({
                "state": r["state"],
                "district": r["district"],
                "field": model_col,
                "new_value": r[model_col],
            })

    # Keep incumbent_party synced to incumbent_party_raw if present.
    if "incumbent_party" in merged.columns and "incumbent_party_raw" in merged.columns:
        merged["incumbent_party"] = merged["incumbent_party_raw"]

    # Recompute district_id if present.
    if "district_id" in merged.columns:
        merged["district_id"] = merged["state"].astype(str).str.upper() + "-" + merged["district"].astype(str)

    # Drop helper/update columns.
    drop_cols = [
        c for c in merged.columns
        if c.endswith("_excel") or c in {"state_key", "district_key", "_merge"}
    ]
    out = merged.drop(columns=drop_cols, errors="ignore")

    out.to_csv(CSV_PATH, index=False)

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(AUDIT_PATH, index=False)

    # Excel rows that did not match model rows.
    matched_keys = set(zip(model["state_key"], model["district_key"]))
    unmatched = xl[~xl.apply(lambda r: (r["state_key"], r["district_key"]) in matched_keys, axis=1)].copy()
    unmatched.to_csv(MISMATCH_PATH, index=False)

    print(f"Imported source fields from {EXCEL_PATH} -> {CSV_PATH}")
    print(f"Backup written to {BACKUP_PATH}")
    print(f"Audit written to {AUDIT_PATH} ({len(audit)} changed fields)")
    print(f"Unmatched Excel rows written to {MISMATCH_PATH} ({len(unmatched)} rows)")

    preview_cols = [
        "state", "district", "dem_candidate", "gop_candidate",
        "incumbent_raw", "incumbent_party", "general_election_party_structure"
    ]
    preview_cols = [c for c in preview_cols if c in out.columns]

    print("\nPreview of updated rows with known prior mismatches:")
    mask = (
        (out["state"].astype(str).str.upper().eq("AL") & out["district"].astype(str).eq("5"))
        | (out["state"].astype(str).str.upper().eq("GA") & out["district"].astype(str).eq("1"))
        | (out["state"].astype(str).str.upper().eq("OK") & out["district"].astype(str).eq("1"))
        | (out["state"].astype(str).str.upper().eq("SD") & out["district"].astype(str).str.upper().eq("AL"))
        | (out["state"].astype(str).str.upper().eq("VT") & out["district"].astype(str).str.upper().eq("AL"))
    )
    print(out.loc[mask, preview_cols].to_string(index=False))

if __name__ == "__main__":
    main()
