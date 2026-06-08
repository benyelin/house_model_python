from pathlib import Path
import numpy as np
import pandas as pd

INPUTS = Path("inputs")
OUTPUTS = Path("outputs")

RACE_INPUTS_PATH = INPUTS / "house_race_inputs.csv"
POLLING_AVERAGES_PATH = INPUTS / "house_polling_averages_generated.csv"
SETTINGS_PATH = INPUTS / "house_calibration_settings.csv"

SIGNAL_OUTPUT = OUTPUTS / "house_poll_spillover_signal.csv"
SOURCE_OUTPUT = OUTPUTS / "house_poll_spillover_sources.csv"


def read_csv_safe(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_settings():
    settings = read_csv_safe(SETTINGS_PATH)

    if settings.empty or "setting" not in settings.columns or "value" not in settings.columns:
        return {}

    out = {}
    for _, row in settings.iterrows():
        key = str(row.get("setting", "")).strip()
        try:
            out[key] = float(row.get("value"))
        except Exception:
            continue

    return out


def setting(settings, key, default):
    return float(settings.get(key, default))


def normalize_district_id(x):
    return str(x).strip().upper()


def as_num(series, default=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def get_days_out(races):
    for col in ["days_out", "model_days_out"]:
        if col in races.columns:
            vals = pd.to_numeric(races[col], errors="coerce").dropna()
            if not vals.empty:
                return float(vals.iloc[0])

    # Fallback if race inputs do not carry days_out.
    summary_path = OUTPUTS / "house_forecast_summary.csv"
    if summary_path.exists():
        try:
            summary = pd.read_csv(summary_path)
            if "days_out" in summary.columns and not summary.empty:
                return float(pd.to_numeric(summary["days_out"], errors="coerce").dropna().iloc[-1])
        except Exception:
            pass

    return 150.0


def time_weight_from_days_out(days_out):
    """
    Very small early; grows as Election Day approaches.
    At ~150 days: roughly 0.25
    At ~60 days: roughly 0.55
    At ~14 days: roughly 0.90
    """
    try:
        days_out = float(days_out)
    except Exception:
        days_out = 150.0

    x = max(0.0, min(1.0, (180.0 - days_out) / 180.0))
    return 0.15 + 0.85 * (x ** 1.35)


def current_spillover_cap(days_out, cap_now, cap_final):
    tw = time_weight_from_days_out(days_out)
    # Scale smoothly between now-cap and final-cap.
    return cap_now + (cap_final - cap_now) * tw


def baseline_similarity(source_baseline, target_baseline):
    try:
        diff = abs(float(source_baseline) - float(target_baseline))
    except Exception:
        return 0.0

    # Full credit if near-identical; fades to zero by 20 points apart.
    return max(0.0, 1.0 - diff / 20.0)


def compute_similarity(source, target, settings):
    score = 0.0
    notes = []

    if str(source.get("state", "")).strip().upper() == str(target.get("state", "")).strip().upper():
        bonus = setting(settings, "house_poll_spillover_same_state_bonus", 0.35)
        score += bonus
        notes.append("same state")

    if str(source.get("region", "")).strip() and str(source.get("region", "")).strip() == str(target.get("region", "")).strip():
        bonus = setting(settings, "house_poll_spillover_same_region_bonus", 0.25)
        score += bonus
        notes.append("same region")

    if str(source.get("district_type", "")).strip() and str(source.get("district_type", "")).strip() == str(target.get("district_type", "")).strip():
        bonus = setting(settings, "house_poll_spillover_same_district_type_bonus", 0.15)
        score += bonus
        notes.append("same district type")

    demo_col = "education_race_error_group"
    if demo_col in source.index and demo_col in target.index:
        if str(source.get(demo_col, "")).strip() and str(source.get(demo_col, "")).strip() == str(target.get(demo_col, "")).strip():
            bonus = setting(settings, "house_poll_spillover_same_demo_group_bonus", 0.25)
            score += bonus
            notes.append("same education/race group")

    base_col = "district_partisan_baseline_dem"
    if base_col in source.index and base_col in target.index:
        bsim = baseline_similarity(source.get(base_col), target.get(base_col))
        bonus = setting(settings, "house_poll_spillover_baseline_similarity_bonus", 0.20) * bsim
        if bonus > 0:
            score += bonus
            notes.append(f"baseline similarity {bsim:.2f}")

    return min(score, 1.0), "; ".join(notes)


def main():
    settings = read_settings()

    races = read_csv_safe(RACE_INPUTS_PATH)
    polls = read_csv_safe(POLLING_AVERAGES_PATH)

    if races.empty:
        raise FileNotFoundError("inputs/house_race_inputs.csv missing or empty.")

    if "district_id" not in races.columns:
        raise ValueError("house_race_inputs.csv must contain district_id.")

    races = races.copy()
    races["district_id"] = races["district_id"].apply(normalize_district_id)

    if "state" not in races.columns:
        races["state"] = races["district_id"].str.extract(r"^([A-Z]{2})", expand=False)

    # Ensure required numeric columns exist.
    for col in [
        "fundamentals_margin_dem",
        "district_partisan_baseline_dem",
        "poll_count",
        "polling_margin_dem",
        "total_poll_weight",
    ]:
        if col not in races.columns:
            races[col] = np.nan

    # If polling averages file exists, merge the latest polling average into races.
    if not polls.empty and "district_id" in polls.columns:
        polls = polls.copy()
        polls["district_id"] = polls["district_id"].apply(normalize_district_id)

        keep = [
            "district_id",
            "polling_margin_dem",
            "poll_count",
            "avg_poll_age_days",
            "total_poll_weight",
            "polling_notes",
        ]
        keep = [c for c in keep if c in polls.columns]

        races = races.drop(
            columns=[c for c in keep if c != "district_id" and c in races.columns],
            errors="ignore",
        ).merge(
            polls[keep],
            on="district_id",
            how="left",
        )

    for col in ["fundamentals_margin_dem", "polling_margin_dem", "poll_count", "total_poll_weight", "avg_poll_age_days"]:
        if col in races.columns:
            races[col] = pd.to_numeric(races[col], errors="coerce")

    days_out = get_days_out(races)
    time_weight = time_weight_from_days_out(days_out)

    base_weight = setting(settings, "house_poll_spillover_base_weight", 0.06)
    cap_now = setting(settings, "house_poll_spillover_max_adjustment_now", 0.20)
    cap_final = setting(settings, "house_poll_spillover_max_adjustment_final", 0.75)
    min_similarity = setting(settings, "house_poll_spillover_min_similarity", 0.25)
    polled_discount = setting(settings, "house_poll_spillover_polled_district_discount", 0.35)

    adjustment_cap = current_spillover_cap(days_out, cap_now, cap_final)

    source_mask = (
        races["polling_margin_dem"].notna()
        & races["fundamentals_margin_dem"].notna()
        & races["poll_count"].fillna(0).gt(0)
    )

    sources = races[source_mask].copy()

    source_rows = []
    signal_rows = []

    # No polls yet: emit zero adjustment file.
    if sources.empty:
        out = races[["district_id"]].copy()
        out["poll_spillover_adjustment_dem"] = 0.0
        out["poll_spillover_source_count"] = 0
        out["poll_spillover_abs_signal"] = 0.0
        out["poll_spillover_notes"] = "No House polling sources available"
        OUTPUTS.mkdir(exist_ok=True)
        out.to_csv(SIGNAL_OUTPUT, index=False)
        pd.DataFrame().to_csv(SOURCE_OUTPUT, index=False)
        print("No polling sources found. Wrote zero spillover signal.")
        return

    for _, src in sources.iterrows():
        residual = float(src["polling_margin_dem"] - src["fundamentals_margin_dem"])

        # Weight source by poll weight and recency, but keep bounded.
        poll_weight = src.get("total_poll_weight", 1.0)
        try:
            poll_weight = float(poll_weight)
        except Exception:
            poll_weight = 1.0
        poll_weight_factor = max(0.25, min(1.25, poll_weight))

        age = src.get("avg_poll_age_days", 30.0)
        try:
            age = float(age)
        except Exception:
            age = 30.0
        recency_factor = max(0.25, min(1.0, 1.0 - age / 180.0))

        source_strength = base_weight * time_weight * poll_weight_factor * recency_factor

        source_rows.append(
            {
                "source_district_id": src["district_id"],
                "source_polling_margin_dem": src["polling_margin_dem"],
                "source_fundamentals_margin_dem": src["fundamentals_margin_dem"],
                "source_poll_residual_dem": residual,
                "source_poll_count": src.get("poll_count", np.nan),
                "source_total_poll_weight": src.get("total_poll_weight", np.nan),
                "source_avg_poll_age_days": src.get("avg_poll_age_days", np.nan),
                "source_strength": source_strength,
                "days_out": days_out,
                "time_weight": time_weight,
            }
        )

    for _, target in races.iterrows():
        target_id = target["district_id"]
        contributions = []
        notes = []

        target_has_polling = bool(pd.notna(target.get("polling_margin_dem")) and float(target.get("poll_count", 0) or 0) > 0)

        for _, src in sources.iterrows():
            source_id = src["district_id"]

            # Do not spill a district to itself; direct polling already handles it.
            if source_id == target_id:
                continue

            similarity, sim_notes = compute_similarity(src, target, settings)

            if similarity < min_similarity:
                continue

            residual = float(src["polling_margin_dem"] - src["fundamentals_margin_dem"])

            poll_weight = src.get("total_poll_weight", 1.0)
            try:
                poll_weight = float(poll_weight)
            except Exception:
                poll_weight = 1.0
            poll_weight_factor = max(0.25, min(1.25, poll_weight))

            age = src.get("avg_poll_age_days", 30.0)
            try:
                age = float(age)
            except Exception:
                age = 30.0
            recency_factor = max(0.25, min(1.0, 1.0 - age / 180.0))

            contribution = residual * base_weight * time_weight * similarity * poll_weight_factor * recency_factor

            if target_has_polling:
                contribution *= polled_discount

            contributions.append(contribution)

            notes.append(
                f"{source_id}: residual={residual:+.2f}, sim={similarity:.2f}, contrib={contribution:+.3f} ({sim_notes})"
            )

        raw_adjustment = float(np.sum(contributions)) if contributions else 0.0
        capped_adjustment = float(np.clip(raw_adjustment, -adjustment_cap, adjustment_cap))

        signal_rows.append(
            {
                "district_id": target_id,
                "poll_spillover_adjustment_dem": capped_adjustment,
                "poll_spillover_raw_adjustment_dem": raw_adjustment,
                "poll_spillover_source_count": len(contributions),
                "poll_spillover_abs_signal": abs(capped_adjustment),
                "poll_spillover_cap": adjustment_cap,
                "poll_spillover_days_out": days_out,
                "poll_spillover_time_weight": time_weight,
                "poll_spillover_target_has_polling": target_has_polling,
                "poll_spillover_notes": " | ".join(notes[:5]),
            }
        )

    signal = pd.DataFrame(signal_rows)
    source_audit = pd.DataFrame(source_rows)

    OUTPUTS.mkdir(exist_ok=True)
    signal.to_csv(SIGNAL_OUTPUT, index=False)
    source_audit.to_csv(SOURCE_OUTPUT, index=False)

    print(f"Wrote {SIGNAL_OUTPUT}")
    print(f"Wrote {SOURCE_OUTPUT}")
    print()
    print("Poll spillover source districts")
    print("-------------------------------")
    print(source_audit.to_string(index=False))
    print()
    print("Largest spillover adjustments")
    print("-----------------------------")
    show_cols = [
        "district_id",
        "poll_spillover_adjustment_dem",
        "poll_spillover_source_count",
        "poll_spillover_cap",
        "poll_spillover_days_out",
        "poll_spillover_time_weight",
        "poll_spillover_target_has_polling",
        "poll_spillover_notes",
    ]
    print(
        signal.reindex(signal["poll_spillover_adjustment_dem"].abs().sort_values(ascending=False).index)
        [show_cols]
        .head(40)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
