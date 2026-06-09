from pathlib import Path
import pandas as pd
import numpy as np

HOUSE_DIR = Path("~/Desktop/house_model_python").expanduser()
SENATE_DIR = Path("~/Desktop/senate_model_python_Q1_auto_calendar_candidate_refresh").expanduser()

OUT = HOUSE_DIR / "outputs" / "model_adjustment_review.csv"


def read(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def add_rows(rows, model, df, id_col, columns):
    if df.empty or id_col not in df.columns:
        return

    for col, category, recommended_range, note in columns:
        if col not in df.columns:
            continue

        vals = pd.to_numeric(df[col], errors="coerce").fillna(0)

        for idx, val in vals.items():
            if abs(val) < 0.001:
                continue

            row = df.loc[idx]

            risk = "Low"
            if abs(val) > 3:
                risk = "High"
            elif abs(val) > 1.5:
                risk = "Medium"

            rows.append({
                "model": model,
                "race_id": row.get(id_col),
                "state": row.get("state", ""),
                "rating": row.get("rating", ""),
                "dem_win_probability": row.get("dem_win_probability", ""),
                "model_margin_dem": row.get("model_margin_dem", ""),
                "adjustment_column": col,
                "adjustment_category": category,
                "adjustment_value_dem": val,
                "risk_level": risk,
                "recommended_range": recommended_range,
                "review_note": note,
                "dem_candidate": row.get("dem_candidate", ""),
                "gop_candidate": row.get("gop_candidate", ""),
            })


def main():
    rows = []

    house = read(HOUSE_DIR / "inputs" / "house_race_inputs.csv")
    house_stats = read(HOUSE_DIR / "outputs" / "house_race_stats.csv")

    if not house.empty and not house_stats.empty and "district_id" in house.columns and "district_id" in house_stats.columns:
        keep = [c for c in ["district_id", "rating", "dem_win_probability", "model_margin_dem"] if c in house_stats.columns]
        house = house.merge(house_stats[keep], on="district_id", how="left")

    senate = read(SENATE_DIR / "inputs" / "race_inputs.csv")
    senate_stats = read(SENATE_DIR / "outputs" / "race_stats.csv")

    if not senate.empty and not senate_stats.empty:
        possible_ids = ["race_id", "state"]
        id_col = next((c for c in possible_ids if c in senate.columns and c in senate_stats.columns), None)
        if id_col:
            keep = [c for c in [id_col, "rating", "dem_win_probability", "model_margin_dem"] if c in senate_stats.columns]
            senate = senate.merge(senate_stats[keep], on=id_col, how="left")

    house_cols = [
        ("incumbency_adjustment_dem", "Incumbency", "-2 to +2 House points", "House incumbency should generally stay around +/-1.5 unless backtests justify more."),
        ("candidate_quality_adjustment_dem", "Candidate quality", "-3 to +3, usually smaller", "Candidate quality should be shrunk/capped and reviewed in competitive races."),
        ("candidate_war_adjustment_dem", "Candidate WAR", "-3 to +3 cap", "Review large or one-sided WAR matches."),
        ("special_adjustment_dem", "Special/manual", "Normally 0; +/-1 to +/-2 with rationale", "Manual adjustments require written rationale."),
        # state_environment_adjustment_dem is excluded here because in this model it is
        # the derived national-environment-through-elasticity component, not a manual state adjustment.
        ("poll_spillover_adjustment_dem", "Poll spillover", "Near zero now; under 1 late", "Should remain tiny and auditable."),
    ]

    senate_cols = [
        ("incumbency_adjustment_dem", "Incumbency", "-3 to +3 generic Senate points", "Review whether generic incumbency is double-counted with overperformance."),
        ("overperformance_adjustment_dem", "Overperformance", "-4 to +4 after shrinkage", "Strong candidate/race priors should have written rationale."),
        ("candidate_liability_adjustment_dem", "Candidate liability", "-3 to +3 usually", "Large liabilities should be rare and documented."),
        ("candidate_quality_adjustment_dem", "Candidate quality", "-3 to +3 usually", "Candidate quality should be reviewed race-by-race."),
        ("special_adjustment_dem", "Special/manual", "Normally 0; +/-1 to +/-2 with rationale", "Manual adjustments require written rationale."),
        # state_environment_adjustment_dem is excluded here because in this model it is
        # the derived national-environment-through-elasticity component, not a manual state adjustment.
    ]

    add_rows(rows, "House", house, "district_id", house_cols)

    senate_id = "race_id" if "race_id" in senate.columns else "state"
    add_rows(rows, "Senate", senate, senate_id, senate_cols)

    out = pd.DataFrame(rows)

    if not out.empty:
        out["abs_adjustment"] = pd.to_numeric(out["adjustment_value_dem"], errors="coerce").abs()
        out = out.sort_values(["risk_level", "abs_adjustment"], ascending=[True, False])
        risk_order = {"High": 0, "Medium": 1, "Low": 2}
        out["_risk_sort"] = out["risk_level"].map(risk_order).fillna(9)
        out = out.sort_values(["_risk_sort", "abs_adjustment"], ascending=[True, False]).drop(columns=["_risk_sort"])

    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"Wrote {OUT}")
    print()
    if out.empty:
        print("No nonzero qualitative adjustments found.")
    else:
        show_cols = [
            "model",
            "race_id",
            "rating",
            "model_margin_dem",
            "adjustment_column",
            "adjustment_value_dem",
            "risk_level",
            "recommended_range",
            "review_note",
            "dem_candidate",
            "gop_candidate",
        ]
        show_cols = [c for c in show_cols if c in out.columns]
        print(out[show_cols].head(80).to_string(index=False))


if __name__ == "__main__":
    main()
