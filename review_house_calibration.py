from pathlib import Path
import pandas as pd
import numpy as np

AUDIT_PATH = Path("outputs/house_calibration_audit.csv")
REPORT_PATH = Path("outputs/house_calibration_review.txt")


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


def fmt_pct(x):
    try:
        return f"{float(x):.1%}"
    except Exception:
        return "NA"


def fmt_margin(x):
    try:
        x = float(x)
    except Exception:
        return "NA"

    if pd.isna(x):
        return "NA"
    if x > 0:
        return f"D+{x:.1f}"
    if x < 0:
        return f"R+{abs(x):.1f}"
    return "Even"


def line(title=""):
    if title:
        return "\n" + title + "\n" + "-" * len(title) + "\n"
    return "\n"


def table(df, cols, n=25):
    cols = [c for c in cols if c in df.columns]
    if df.empty:
        return "None\n"
    return df[cols].head(n).to_string(index=False) + "\n"


def add_abs_col(df, col, name):
    out = df.copy()
    if col in out.columns:
        out[name] = pd.to_numeric(out[col], errors="coerce").abs()
    else:
        out[name] = np.nan
    return out


def main():
    if not AUDIT_PATH.exists():
        raise FileNotFoundError("outputs/house_calibration_audit.csv not found. Run build_house_calibration_audit.py first.")

    audit = pd.read_csv(AUDIT_PATH)

    if audit.empty:
        raise ValueError("house_calibration_audit.csv is empty.")

    # Normalize important numeric columns.
    for col in [
        "dem_win_probability",
        "model_margin_dem",
        "fundamentals_margin_dem",
        "district_partisan_baseline_dem",
        "state_environment_adjustment_dem",
        "audit_elasticity",
        "incumbency_adjustment_dem",
        "candidate_quality_adjustment_dem",
        "poll_count",
        "bayesian_polling_weight",
        "audit_final_vs_fundamentals_gap",
        "audit_poll_vs_fundamentals_gap",
        "audit_fundamentals_gap",
    ]:
        if col in audit.columns:
            audit[col] = pd.to_numeric(audit[col], errors="coerce")

    chunks = []

    chunks.append("House Calibration Review\n========================\n")
    chunks.append(f"Rows: {len(audit)}\n")

    # 1. Rating distribution.
    chunks.append(line("1. Rating distribution"))

    if "rating" in audit.columns:
        counts = audit["rating"].value_counts(dropna=False)
        ordered = pd.DataFrame({
            "Rating": RATING_ORDER,
            "Districts": [int(counts.get(r, 0)) for r in RATING_ORDER],
        })

        other_ratings = [
            r for r in counts.index.tolist()
            if r not in RATING_ORDER and pd.notna(r)
        ]

        if other_ratings:
            extra = pd.DataFrame({
                "Rating": other_ratings,
                "Districts": [int(counts.get(r, 0)) for r in other_ratings],
            })
            ordered = pd.concat([ordered, extra], ignore_index=True)

        chunks.append(ordered.to_string(index=False) + "\n")
    else:
        chunks.append("No rating column found.\n")

    # 2. Audit flags.
    chunks.append(line("2. Audit flags"))

    if "audit_flags" in audit.columns:
        flagged = audit[audit["audit_flags"].fillna("").astype(str).str.strip().ne("")]
        chunks.append(f"Flagged rows: {len(flagged)}\n")

        if not flagged.empty:
            flag_counts = (
                flagged["audit_flags"]
                .str.split("; ")
                .explode()
                .value_counts()
                .reset_index()
            )
            flag_counts.columns = ["Flag", "Count"]
            chunks.append("\nFlag counts:\n")
            chunks.append(flag_counts.to_string(index=False) + "\n")

            chunks.append("\nFlagged sample:\n")
            chunks.append(table(
                flagged,
                [
                    "district_id",
                    "rating",
                    "model_margin_label",
                    "dem_win_probability",
                    "audit_flags",
                ],
                n=30,
            ))
    else:
        chunks.append("No audit_flags column found.\n")

    # 3. Safe districts that look too competitive.
    chunks.append(line("3. Safe districts that look too competitive"))

    if all(c in audit.columns for c in ["district_partisan_baseline_dem", "dem_win_probability"]):
        safe_comp = audit[
            (
                (audit["district_partisan_baseline_dem"] >= 15)
                & (audit["dem_win_probability"] < 0.85)
            )
            |
            (
                (audit["district_partisan_baseline_dem"] <= -15)
                & (audit["dem_win_probability"] > 0.15)
            )
        ].copy()

        safe_comp["baseline_abs"] = safe_comp["district_partisan_baseline_dem"].abs()
        safe_comp = safe_comp.sort_values(["baseline_abs"], ascending=False)

        chunks.append(table(
            safe_comp,
            [
                "district_id",
                "baseline_label",
                "fundamentals_margin_label",
                "model_margin_label",
                "dem_win_probability",
                "rating",
                "region",
                "district_type",
                "audit_flags",
            ],
            n=40,
        ))
    else:
        chunks.append("Missing baseline or probability columns.\n")

    # 4. Competitive districts that look too safe.
    chunks.append(line("4. Competitive districts that look too safe"))

    if all(c in audit.columns for c in ["district_partisan_baseline_dem", "dem_win_probability"]):
        comp_safe = audit[
            (audit["district_partisan_baseline_dem"].abs() <= 5)
            & (
                (audit["dem_win_probability"] >= 0.85)
                | (audit["dem_win_probability"] <= 0.15)
            )
        ].copy()

        comp_safe = comp_safe.sort_values("dem_win_probability")

        chunks.append(table(
            comp_safe,
            [
                "district_id",
                "baseline_label",
                "fundamentals_margin_label",
                "model_margin_label",
                "dem_win_probability",
                "rating",
                "region",
                "district_type",
                "audit_flags",
            ],
            n=40,
        ))
    else:
        chunks.append("Missing baseline or probability columns.\n")

    # 5. Environment and elasticity by district type.
    chunks.append(line("5. Environment and elasticity by district type"))

    if "district_type" in audit.columns:
        group_cols = ["district_type"]
        agg = audit.groupby(group_cols, dropna=False).agg(
            districts=("district_id", "count"),
            avg_elasticity=("audit_elasticity", "mean"),
            avg_env_adj=("state_environment_adjustment_dem", "mean"),
            avg_baseline=("district_partisan_baseline_dem", "mean"),
            avg_model_margin=("model_margin_dem", "mean"),
            avg_dem_prob=("dem_win_probability", "mean"),
        ).reset_index()

        chunks.append(agg.to_string(index=False, float_format=lambda x: f"{x:.3f}") + "\n")
    else:
        chunks.append("No district_type column found.\n")

    chunks.append(line("6. Environment and elasticity by region"))

    if "region" in audit.columns:
        agg = audit.groupby("region", dropna=False).agg(
            districts=("district_id", "count"),
            avg_elasticity=("audit_elasticity", "mean"),
            avg_env_adj=("state_environment_adjustment_dem", "mean"),
            avg_baseline=("district_partisan_baseline_dem", "mean"),
            avg_model_margin=("model_margin_dem", "mean"),
            avg_dem_prob=("dem_win_probability", "mean"),
        ).reset_index()

        chunks.append(agg.to_string(index=False, float_format=lambda x: f"{x:.3f}") + "\n")
    else:
        chunks.append("No region column found.\n")

    # 7. Incumbency adjustments.
    chunks.append(line("7. Incumbency adjustments"))

    if "incumbency_adjustment_dem" in audit.columns:
        inc = audit.copy()
        inc["inc_abs"] = inc["incumbency_adjustment_dem"].abs()
        inc_nonzero = inc[inc["inc_abs"] > 0.001].sort_values("inc_abs", ascending=False)

        chunks.append(f"Nonzero incumbency adjustment rows: {len(inc_nonzero)}\n")
        chunks.append(table(
            inc_nonzero,
            [
                "district_id",
                "incumbent_party",
                "inferred_incumbent_party",
                "dem_candidate",
                "gop_candidate",
                "incumbency_adjustment_dem",
                "model_margin_label",
                "dem_win_probability",
                "rating",
            ],
            n=50,
        ))
    else:
        chunks.append("No incumbency_adjustment_dem column found.\n")

    # 8. Candidate quality adjustments.
    chunks.append(line("8. Candidate quality adjustments"))

    if "candidate_quality_adjustment_dem" in audit.columns:
        cq = audit.copy()
        cq["cq_abs"] = cq["candidate_quality_adjustment_dem"].abs()
        cq_nonzero = cq[cq["cq_abs"] > 0.001].sort_values("cq_abs", ascending=False)

        chunks.append(f"Nonzero candidate-quality adjustment rows: {len(cq_nonzero)}\n")
        chunks.append(table(
            cq_nonzero,
            [
                "district_id",
                "dem_candidate",
                "gop_candidate",
                "candidate_quality_adjustment_dem",
                "model_margin_label",
                "dem_win_probability",
                "rating",
            ],
            n=50,
        ))
    else:
        chunks.append("No candidate_quality_adjustment_dem column found.\n")

    # 9. Largest movement from fundamentals to final.
    chunks.append(line("9. Largest movement from fundamentals to final model margin"))

    if "audit_final_vs_fundamentals_gap" in audit.columns:
        move = add_abs_col(audit, "audit_final_vs_fundamentals_gap", "abs_gap")
        move = move.sort_values("abs_gap", ascending=False)

        chunks.append(table(
            move,
            [
                "district_id",
                "fundamentals_margin_label",
                "polling_margin_label",
                "poll_count",
                "bayesian_polling_weight",
                "model_margin_label",
                "audit_final_vs_fundamentals_gap",
                "rating",
            ],
            n=40,
        ))
    else:
        chunks.append("No audit_final_vs_fundamentals_gap column found.\n")

    # 10. Election structure / fixed control.
    chunks.append(line("10. Election structure / fixed control checks"))

    structure_cols = [
        "district_id",
        "dem_candidate",
        "gop_candidate",
        "other_candidate",
        "general_election_party_structure",
        "party_control_override",
        "party_control_fixed",
        "dem_win_probability",
        "rating",
        "audit_flags",
    ]

    if "general_election_party_structure" in audit.columns:
        structures = audit["general_election_party_structure"].value_counts(dropna=False).reset_index()
        structures.columns = ["Party structure", "Districts"]
        chunks.append("Party structure counts:\n")
        chunks.append(structures.to_string(index=False) + "\n")

        fixed = audit[
            audit.get("party_control_fixed", "").fillna("").astype(str).str.strip().ne("")
        ] if "party_control_fixed" in audit.columns else pd.DataFrame()

        chunks.append("\nFixed-control rows:\n")
        chunks.append(table(fixed, structure_cols, n=60))
    else:
        chunks.append("No general_election_party_structure column found.\n")

    # 11. Demographic group summaries.
    chunks.append(line("11. Demographic summaries"))

    for group_col in [
        "college_share_tier",
        "white_share_tier",
        "black_share_tier",
        "hispanic_share_tier",
        "median_income_tier",
    ]:
        if group_col in audit.columns:
            agg = audit.groupby(group_col, dropna=False).agg(
                districts=("district_id", "count"),
                avg_baseline=("district_partisan_baseline_dem", "mean"),
                avg_model_margin=("model_margin_dem", "mean"),
                avg_dem_prob=("dem_win_probability", "mean"),
            ).reset_index()

            chunks.append(f"\n{group_col}:\n")
            chunks.append(agg.to_string(index=False, float_format=lambda x: f"{x:.3f}") + "\n")

    report = "".join(chunks)

    REPORT_PATH.write_text(report)

    print(report)
    print(f"\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
