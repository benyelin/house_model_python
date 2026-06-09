from pathlib import Path
import pandas as pd
import numpy as np

INPUT = Path("inputs/house_manual_polls.csv")
OUTPUT = Path("inputs/house_manual_polls_adjusted.csv")
SETTINGS = Path("inputs/house_calibration_settings.csv")


def read_settings():
    if not SETTINGS.exists():
        return {}

    s = pd.read_csv(SETTINGS)
    out = {}

    if "setting" not in s.columns or "value" not in s.columns:
        return out

    for _, row in s.iterrows():
        key = str(row.get("setting", "")).strip()
        try:
            out[key] = float(row.get("value"))
        except Exception:
            pass

    return out


def setting(settings, key, default):
    return float(settings.get(key, default))


def norm(x):
    return str(x).strip().lower()


def boolish(x):
    return norm(x) in ["1", "true", "yes", "y", "internal", "campaign", "campaign internal"]


def infer_party(row):
    for col in ["partisan_sponsor_party", "sponsor_party", "pollster_partisan_affiliation"]:
        if col in row.index:
            v = norm(row.get(col, ""))
            if v in ["d", "dem", "democratic", "democrat"]:
                return "D"
            if v in ["r", "rep", "republican", "gop"]:
                return "R"
    return ""


def infer_internal(row):
    if "is_internal_poll" in row.index and boolish(row.get("is_internal_poll")):
        return True

    sponsor_type = norm(row.get("poll_sponsor_type", ""))
    return sponsor_type in [
        "internal",
        "campaign",
        "campaign internal",
        "candidate",
        "party",
        "party committee",
    ]


def find_margin_col(df):
    candidates = [
        "poll_margin_dem",
        "margin_dem",
        "polling_margin_dem",
        "dem_margin",
        "margin",
    ]
    return next((c for c in candidates if c in df.columns), None)


def find_vote_share_pair(df):
    pairs = [
        ("dem_pct", "gop_pct"),
        ("dem_pct", "rep_pct"),
        ("dem_pct", "republican_pct"),
        ("dem_share", "gop_share"),
        ("dem_share", "rep_share"),
        ("dem_share", "republican_share"),
        ("dem", "gop"),
        ("dem", "rep"),
        ("dem", "republican"),
        ("democratic", "republican"),
        ("democratic_pct", "republican_pct"),
        ("democratic_share", "republican_share"),
    ]

    lower_map = {c.lower(): c for c in df.columns}

    for d_col, r_col in pairs:
        if d_col in lower_map and r_col in lower_map:
            return lower_map[d_col], lower_map[r_col]

    return None, None


def find_weight_col(df):
    candidates = [
        "poll_weight",
        "weight",
        "manual_poll_weight",
    ]
    return next((c for c in candidates if c in df.columns), None)


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"{INPUT} not found.")

    settings = read_settings()

    use_adjustments = setting(settings, "use_partisan_pollster_adjustments", 1) >= 0.5
    default_adj = setting(settings, "partisan_pollster_default_adjustment", 2.0)
    internal_adj = setting(settings, "partisan_pollster_internal_adjustment", 2.5)
    max_adj = setting(settings, "partisan_pollster_max_adjustment", 3.0)
    weight_multiplier = setting(settings, "partisan_pollster_weight_multiplier", 0.50)

    df = pd.read_csv(INPUT)

    required_metadata = {
        "poll_sponsor_type": "",
        "pollster_partisan_affiliation": "",
        "partisan_sponsor_party": "",
        "is_internal_poll": False,
        "partisan_pollster_review_notes": "",
    }

    for col, default in required_metadata.items():
        if col not in df.columns:
            df[col] = default

    margin_col = find_margin_col(df)
    weight_col = find_weight_col(df)

    if margin_col is None:
        dem_col, gop_col = find_vote_share_pair(df)

        if dem_col is None or gop_col is None:
            raise ValueError(
                "Could not find a poll margin column or Democratic/GOP vote-share pair. "
                "Expected a margin column like margin_dem/polling_margin_dem, or vote-share "
                "columns like dem_pct and gop_pct. Found columns: "
                + ", ".join(df.columns)
            )

        margin_col = "margin_dem"
        df[margin_col] = (
            pd.to_numeric(df[dem_col], errors="coerce")
            - pd.to_numeric(df[gop_col], errors="coerce")
        )
        print(f"Computed {margin_col} from {dem_col} - {gop_col}")

    df["polling_margin_dem_original"] = pd.to_numeric(df[margin_col], errors="coerce")
    df["partisan_pollster_adjustment_dem"] = 0.0
    df["partisan_pollster_weight_multiplier"] = 1.0
    df["partisan_pollster_adjusted"] = False
    df["partisan_pollster_notes"] = "No partisan pollster adjustment"

    if weight_col:
        df[f"{weight_col}_original"] = pd.to_numeric(df[weight_col], errors="coerce").fillna(1.0)

    if use_adjustments:
        for idx, row in df.iterrows():
            party = infer_party(row)
            internal = infer_internal(row)

            if party not in ["D", "R"]:
                continue

            raw_adj = internal_adj if internal else default_adj
            raw_adj = min(abs(raw_adj), max_adj)

            # Adjust against the sponsoring party.
            # D sponsor: reduce Dem margin.
            # R sponsor: increase Dem margin.
            adj = -raw_adj if party == "D" else raw_adj

            df.loc[idx, "partisan_pollster_adjustment_dem"] = adj
            df.loc[idx, "partisan_pollster_weight_multiplier"] = weight_multiplier
            df.loc[idx, "partisan_pollster_adjusted"] = True
            df.loc[idx, "partisan_pollster_notes"] = (
                f"{party}-sponsored"
                f"{' internal/campaign' if internal else ''} poll: "
                f"applied {adj:+.1f} point Dem-margin adjustment and "
                f"{weight_multiplier:.2f} weight multiplier"
            )

    df[margin_col] = df["polling_margin_dem_original"] + df["partisan_pollster_adjustment_dem"]

    if weight_col:
        df[weight_col] = df[f"{weight_col}_original"] * df["partisan_pollster_weight_multiplier"]

    df.to_csv(OUTPUT, index=False)

    print(f"Wrote {OUTPUT}")
    print()
    print("Partisan pollster adjustment summary")
    print("------------------------------------")
    print(df["partisan_pollster_adjusted"].value_counts(dropna=False).to_string())
    print()
    show_cols = [
        "district_id",
        "pollster",
        "poll_sponsor_type",
        "partisan_sponsor_party",
        "is_internal_poll",
        "polling_margin_dem_original",
        margin_col,
        "partisan_pollster_adjustment_dem",
        "partisan_pollster_weight_multiplier",
        "partisan_pollster_notes",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    print(df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
