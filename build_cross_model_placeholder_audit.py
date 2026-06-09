from pathlib import Path
import pandas as pd
import numpy as np

HOUSE_DIR = Path("~/Desktop/house_model_python").expanduser()
SENATE_DIR = Path("~/Desktop/senate_model_python_Q1_auto_calendar_candidate_refresh").expanduser()

OUT = HOUSE_DIR / "outputs" / "model_placeholder_summary.csv"


def read(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def norm_unknown(s):
    return (
        s.isna()
        | s.astype(str).str.strip().eq("")
        | s.astype(str).str.strip().str.lower().isin(["unknown", "unresolved", "nan", "none"])
    )


def summarize_col(rows, model, df, col, label=None):
    if df.empty or col not in df.columns:
        rows.append({
            "model": model,
            "check": label or col,
            "count": None,
            "total": len(df) if not df.empty else 0,
            "share": None,
            "severity": "Missing column",
            "notes": f"{col} not found",
        })
        return

    flag = norm_unknown(df[col])
    count = int(flag.sum())
    total = int(len(df))
    share = count / total if total else 0

    if share >= 0.75:
        severity = "High"
    elif share >= 0.25:
        severity = "Medium"
    elif count > 0:
        severity = "Low"
    else:
        severity = "OK"

    rows.append({
        "model": model,
        "check": label or col,
        "count": count,
        "total": total,
        "share": round(share, 4),
        "severity": severity,
        "notes": "",
    })


def summarize_bool(rows, model, df, col, label=None, high_threshold=0.25):
    if df.empty or col not in df.columns:
        rows.append({
            "model": model,
            "check": label or col,
            "count": None,
            "total": len(df) if not df.empty else 0,
            "share": None,
            "severity": "Missing column",
            "notes": f"{col} not found",
        })
        return

    vals = df[col].fillna(False)
    if vals.dtype == object:
        vals = vals.astype(str).str.lower().isin(["true", "1", "yes"])
    count = int(vals.sum())
    total = int(len(df))
    share = count / total if total else 0

    if share >= high_threshold:
        severity = "High"
    elif count > 0:
        severity = "Medium"
    else:
        severity = "OK"

    rows.append({
        "model": model,
        "check": label or col,
        "count": count,
        "total": total,
        "share": round(share, 4),
        "severity": severity,
        "notes": "",
    })


def main():
    rows = []

    house_inputs = read(HOUSE_DIR / "inputs" / "house_race_inputs.csv")
    house_stats = read(HOUSE_DIR / "outputs" / "house_race_stats.csv")
    house_local = read(HOUSE_DIR / "outputs" / "house_local_context_audit.csv")
    house_war = read(HOUSE_DIR / "outputs" / "house_candidate_war_audit.csv")

    senate_inputs = read(SENATE_DIR / "inputs" / "race_inputs.csv")
    senate_stats = read(SENATE_DIR / "outputs" / "race_stats.csv")
    senate_summary = read(SENATE_DIR / "outputs" / "forecast_summary.csv")

    for col in [
        "dem_candidate",
        "gop_candidate",
        "median_income_tier",
        "college_share_tier",
        "white_share_tier",
        "black_share_tier",
        "hispanic_share_tier",
        "education_race_error_group",
        "general_election_party_structure",
        "party_control_override",
    ]:
        summarize_col(rows, "House", house_inputs, col)

    if not house_inputs.empty:
        if "poll_count" in house_inputs.columns:
            poll_count = pd.to_numeric(house_inputs["poll_count"], errors="coerce").fillna(0)
            rows.append({
                "model": "House",
                "check": "districts_with_no_polling",
                "count": int((poll_count == 0).sum()),
                "total": int(len(house_inputs)),
                "share": round(float((poll_count == 0).mean()), 4),
                "severity": "Expected early-cycle" if (poll_count == 0).mean() > 0.75 else "Medium",
                "notes": "Sparse polling is expected this far out but important for competitive races.",
            })

    summarize_bool(rows, "House", house_local, "mostly_fundamentals_only", "mostly_fundamentals_only")
    if not house_local.empty and "audit_priority" in house_local.columns:
        high = house_local["audit_priority"].astype(str).str.lower().eq("high")
        rows.append({
            "model": "House",
            "check": "high_priority_local_context_review",
            "count": int(high.sum()),
            "total": int(len(house_local)),
            "share": round(float(high.mean()), 4),
            "severity": "Medium" if high.sum() else "OK",
            "notes": "Review these first.",
        })

    if not house_war.empty and "war_match_status" in house_war.columns:
        incomplete = ~house_war["war_match_status"].astype(str).str.strip().eq("Both matched")
        rows.append({
            "model": "House",
            "check": "candidate_war_not_both_matched",
            "count": int(incomplete.sum()),
            "total": int(len(house_war)),
            "share": round(float(incomplete.mean()), 4),
            "severity": "Medium",
            "notes": "Not all districts need WAR, but competitive one-sided/large adjustments should be reviewed.",
        })

    for col in [
        "dem_candidate",
        "gop_candidate",
        "incumbent_party",
        "race_type",
        "state",
        "fundamentals_margin_dem",
        "polling_margin_dem",
        "special_adjustment_dem",
        "candidate_quality_adjustment_dem",
        "overperformance_adjustment_dem",
        "candidate_liability_adjustment_dem",
    ]:
        summarize_col(rows, "Senate", senate_inputs, col)

    if not senate_inputs.empty:
        if "poll_count" in senate_inputs.columns:
            poll_count = pd.to_numeric(senate_inputs["poll_count"], errors="coerce").fillna(0)
            rows.append({
                "model": "Senate",
                "check": "races_with_no_polling",
                "count": int((poll_count == 0).sum()),
                "total": int(len(senate_inputs)),
                "share": round(float((poll_count == 0).mean()), 4),
                "severity": "Medium",
                "notes": "Senate polling scarcity matters most for competitive states.",
            })

        manual_adjustment_cols = [
            c for c in [
                "special_adjustment_dem",
                "candidate_quality_adjustment_dem",
                "overperformance_adjustment_dem",
                "candidate_liability_adjustment_dem",
            ]
            if c in senate_inputs.columns
        ]

        for c in manual_adjustment_cols:
            vals = pd.to_numeric(senate_inputs[c], errors="coerce").fillna(0)
            nonzero = vals.abs().gt(0.001)
            rows.append({
                "model": "Senate",
                "check": f"nonzero_{c}",
                "count": int(nonzero.sum()),
                "total": int(len(senate_inputs)),
                "share": round(float(nonzero.mean()), 4),
                "severity": "Review",
                "notes": "Nonzero qualitative adjustments should have written rationale.",
            })

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"Wrote {OUT}")
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
