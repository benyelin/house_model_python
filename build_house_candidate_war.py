from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd

INPUTS = Path("inputs")
OUTPUTS = Path("outputs")

RACE_INPUTS_PATH = INPUTS / "house_race_inputs.csv"
SETTINGS_PATH = INPUTS / "house_calibration_settings.csv"

AUDIT_OUTPUT = OUTPUTS / "house_candidate_war_audit.csv"
WAR_MATCH_OUTPUT = OUTPUTS / "house_candidate_war_matches.csv"
ALIASES_PATH = INPUTS / "candidate_name_aliases.csv"


# -----------------------------
# Config defaults
# -----------------------------
DEFAULT_SHRINKAGE = 0.70
DEFAULT_CAP = 4.0
DEFAULT_INCUMBENT_DISCOUNT = 1.00
DEFAULT_MIN_NAME_SCORE = 0.90
DEFAULT_ONE_SIDED_MULTIPLIER = 0.75

# More recent cycles count most.
# These are intentionally conservative and can be tuned.
CYCLE_WEIGHTS = {
    2024: 1.00,
    2022: 0.65,
    2020: 0.40,
    2018: 0.25,
    2016: 0.15,
}


# -----------------------------
# Helpers
# -----------------------------
def read_settings():
    if not SETTINGS_PATH.exists():
        return {}

    df = pd.read_csv(SETTINGS_PATH)

    if df.empty or "setting" not in df.columns or "value" not in df.columns:
        return {}

    out = {}
    for _, row in df.iterrows():
        key = str(row.get("setting", "")).strip()
        try:
            out[key] = float(row.get("value"))
        except Exception:
            continue

    return out


def setting(settings, key, default):
    return float(settings.get(key, default))


def normalize_name(name):
    if pd.isna(name):
        return ""

    s = str(name).strip()

    # Handle common race-input format: "Last, First Middle"
    # Convert to "First Middle Last" before punctuation removal.
    if "," in s:
        parts = [p.strip() for p in s.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            s = f"{parts[1]} {parts[0]}"

    # Remove accents.
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    s = s.upper()

    # Common nicknames / abbreviations that can affect matching.
    nickname_map = {
        "MIKE": "MICHAEL",
        "DAVE": "DAVID",
        "DAN": "DANIEL",
        "BOB": "ROBERT",
        "ROB": "ROBERT",
        "BILL": "WILLIAM",
        "WILL": "WILLIAM",
        "CHUCK": "CHARLES",
        "TOM": "THOMAS",
        "JIM": "JAMES",
        "JIMMY": "JAMES",
        "JOE": "JOSEPH",
        "PAT": "PATRICK",
        "CHRIS": "CHRISTOPHER",
        "MATT": "MATTHEW",
        "BEN": "BENJAMIN",
        "SAM": "SAMUEL",
        "KATE": "KATHERINE",
        "KATIE": "KATHERINE",
        "KATHY": "KATHERINE",
        "KIM": "KIMBERLY",
        "LIZ": "ELIZABETH",
    }

    # Remove punctuation and extra spaces.
    s = re.sub(r"[^A-Z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Remove common suffixes.
    tokens = [
        t for t in s.split()
        if t not in {"JR", "SR", "II", "III", "IV", "V"}
    ]

    tokens = [nickname_map.get(t, t) for t in tokens]

    return " ".join(tokens)




def normalize_district_id(value):
    """
    Normalize district IDs so NC-7 and NC-07 match.
    Also preserves at-large labels like AK-AL.
    """
    if pd.isna(value):
        return ""

    s = str(value).strip().upper()
    s = s.replace(" ", "")

    import re

    m = re.match(r"^([A-Z]{2})[-_]?0*([0-9]+)$", s)
    if m:
        return f"{m.group(1)}-{int(m.group(2))}"

    m = re.match(r"^([A-Z]{2})[-_]?AL$", s)
    if m:
        return f"{m.group(1)}-AL"

    return s


def name_parts(norm):
    if not norm:
        return []

    # Drop very common particles.
    parts = [p for p in norm.split() if p not in {"THE", "DE", "DA", "DEL", "VAN", "VON"}]
    return parts


def name_similarity(a, b):
    """
    Conservative name similarity:
    - exact normalized match = 1.0
    - first + last match = high
    - last name match alone = moderate, not enough by itself
    """
    a_norm = normalize_name(a)
    b_norm = normalize_name(b)

    if not a_norm or not b_norm:
        return 0.0

    if a_norm == b_norm:
        return 1.0

    a_parts = name_parts(a_norm)
    b_parts = name_parts(b_norm)

    if not a_parts or not b_parts:
        return 0.0

    a_first, a_last = a_parts[0], a_parts[-1]
    b_first, b_last = b_parts[0], b_parts[-1]

    if a_first == b_first and a_last == b_last:
        return 0.97

    # Middle initials / shortened names can still match if first initial + last.
    if a_first[:1] == b_first[:1] and a_last == b_last:
        return 0.90

    # Last name plus one shared token.
    if a_last == b_last and len(set(a_parts).intersection(set(b_parts))) >= 2:
        return 0.88

    # Last name only is suggestive but below default threshold.
    if a_last == b_last:
        return 0.75

    return 0.0


def find_war_file():
    candidates = []

    for path in INPUTS.glob("*.csv"):
        name = path.name.lower()

        # Ignore backups and temporary files.
        if (
            ".before_" in name
            or ".backup" in name
            or name.endswith(".bak")
        ):
            continue

        if "war" in name or "wins" in name or "candidate_value" in name:
            candidates.append(path)

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        # Prefer explicit names.
        for path in candidates:
            if "candidate" in path.name.lower() and "war" in path.name.lower():
                return path
        return candidates[0]

    raise FileNotFoundError(
        "Could not auto-detect WAR CSV in inputs/. Rename it to include 'war', "
        "for example inputs/candidate_war.csv."
    )


def find_col(columns, options, required=False):
    lower_map = {str(c).strip().lower(): c for c in columns}

    for opt in options:
        key = opt.lower()
        if key in lower_map:
            return lower_map[key]

    # Fuzzy contains fallback.
    for c in columns:
        cl = str(c).strip().lower()
        for opt in options:
            if opt.lower() in cl:
                return c

    if required:
        raise ValueError(f"Could not find required column. Tried: {options}. Available: {list(columns)}")

    return None


def standardize_party(x):
    s = str(x).strip().upper()

    if s in {"D", "DEM", "DEMOCRAT", "DEMOCRATIC", "DFL"}:
        return "D"

    if s in {"R", "REP", "REPUBLICAN", "GOP"}:
        return "R"

    return s




def read_csv_with_fallback(path):
    """
    Read CSV-like files exported from web/Excel sources.

    Handles:
    - non-UTF-8 encodings
    - comma / tab / semicolon / pipe delimiters
    - files with metadata lines before the real header
    - irregular quoted rows
    """
    encodings = [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin1",
        "iso-8859-1",
    ]

    delimiters = [
        ",",
        "\t",
        ";",
        "|",
    ]

    last_error = None

    for encoding in encodings:
        for sep in delimiters:
            for skiprows in range(0, 20):
                try:
                    df = pd.read_csv(
                        path,
                        encoding=encoding,
                        sep=sep,
                        skiprows=skiprows,
                        engine="python",
                        on_bad_lines="skip",
                    )

                    # Require at least a few columns and at least one plausible candidate/name/year/WAR-ish column.
                    cols = [str(c).strip().lower() for c in df.columns]

                    plausible_col_tokens = [
                        "candidate",
                        "name",
                        "party",
                        "year",
                        "cycle",
                        "war",
                        "wins",
                        "replacement",
                        "margin",
                    ]

                    plausible = any(
                        any(token in col for token in plausible_col_tokens)
                        for col in cols
                    )

                    if df.shape[1] >= 3 and plausible:
                        print(
                            f"Read {path} using encoding={encoding}, sep={repr(sep)}, skiprows={skiprows}"
                        )
                        return df

                except Exception as exc:
                    last_error = exc
                    continue

    # Final attempt: let pandas sniff delimiter.
    for encoding in encodings:
        try:
            df = pd.read_csv(
                path,
                encoding=encoding,
                sep=None,
                engine="python",
                on_bad_lines="skip",
            )
            print(f"Read {path} using encoding={encoding}, sep=None")
            return df
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(f"Could not read {path}. Last error: {last_error}")




def parse_war_margin_to_dem_net(value, sortable=None):
    """
    Convert race-level WAR notation into a Democratic net overperformance.

    Examples:
      D+1.3 -> +1.3
      R+0.2 -> -0.2

    If WAR text is missing, fall back to Sortable if available. In the current
    export, Sortable appears to use the opposite sign convention:
      D+1.3 -> -1.3
      R+0.2 -> +0.2
    """
    s = str(value).strip().upper()

    if s and s not in {"NAN", "NONE", ""}:
        import re
        m = re.match(r"^([DR])\s*\+\s*([0-9.]+)", s)
        if m:
            side = m.group(1)
            amount = float(m.group(2))
            return amount if side == "D" else -amount

        m = re.match(r"^([DR])\s*-\s*([0-9.]+)", s)
        if m:
            side = m.group(1)
            amount = float(m.group(2))
            return -amount if side == "D" else amount

        try:
            return float(s)
        except Exception:
            pass

    try:
        # In this export, Sortable seems opposite-signed from Dem net.
        return -float(sortable)
    except Exception:
        return 0.0


def load_wide_house_war_export(war):
    """
    Handle WAR files with columns:
      Year, Chamber, Geography, Democrat, Republican, WAR, Sortable

    Converts each race row to two candidate rows:
      Democrat gets +net/2
      Republican gets -net/2

    This preserves the race-level net effect while keeping individual candidate
    values conservative.
    """
    required = {"Year", "Chamber", "Geography", "Democrat", "Republican", "WAR"}
    if not required.issubset(set(war.columns)):
        return None

    sortable_col = "Sortable" if "Sortable" in war.columns else None

    rows = []

    for _, row in war.iterrows():
        chamber = str(row.get("Chamber", "")).strip().upper()

        # Keep House rows for the House model.
        if chamber and chamber != "HOUSE":
            continue

        year = pd.to_numeric(row.get("Year"), errors="coerce")
        if pd.isna(year):
            continue

        district_id = normalize_district_id(row.get("Geography", ""))

        dem_name = row.get("Democrat", "")
        rep_name = row.get("Republican", "")

        sortable = row.get(sortable_col) if sortable_col else None
        dem_net = parse_war_margin_to_dem_net(row.get("WAR"), sortable)

        # Symmetric candidate-level allocation.
        dem_score = dem_net / 2.0
        rep_score = -dem_net / 2.0

        if str(dem_name).strip():
            rows.append(
                {
                    "war_candidate_name": dem_name,
                    "war_candidate_norm": normalize_name(dem_name),
                    "war_party": "D",
                    "war_cycle": int(year),
                    "war_score_raw": dem_score,
                    "war_district_id": district_id,
                    "war_state": district_id[:2],
                }
            )

        if str(rep_name).strip():
            rows.append(
                {
                    "war_candidate_name": rep_name,
                    "war_candidate_norm": normalize_name(rep_name),
                    "war_party": "R",
                    "war_cycle": int(year),
                    "war_score_raw": rep_score,
                    "war_district_id": district_id,
                    "war_state": district_id[:2],
                }
            )

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    out["war_cycle_weight"] = out["war_cycle"].map(CYCLE_WEIGHTS).fillna(0.05)
    out["war_weighted_score"] = out["war_score_raw"] * out["war_cycle_weight"]

    return out


def load_war():
    war_path = find_war_file()
    war = read_csv_with_fallback(war_path)

    print(f"Using WAR file: {war_path}")
    print(f"WAR columns: {list(war.columns)}")

    wide_house = load_wide_house_war_export(war)

    if wide_house is not None:
        print("Detected wide House WAR export format: Year, Chamber, Geography, Democrat, Republican, WAR.")
        print(f"Converted to {len(wide_house)} candidate-level WAR rows.")
        return wide_house

    candidate_col = find_col(
        war.columns,
        ["candidate_name", "candidate", "name", "person", "cand"],
        required=True,
    )

    party_col = find_col(
        war.columns,
        ["party", "candidate_party", "party_code"],
        required=True,
    )

    cycle_col = find_col(
        war.columns,
        ["cycle", "year", "election_year"],
        required=True,
    )

    war_col = find_col(
        war.columns,
        ["war_margin", "war", "wins_above_replacement", "wins above replacement", "candidate_war", "value"],
        required=True,
    )

    district_col = find_col(
        war.columns,
        ["district_id", "district", "cd", "seat", "race"],
        required=False,
    )

    state_col = find_col(
        war.columns,
        ["state", "state_code"],
        required=False,
    )

    out = pd.DataFrame()
    out["war_candidate_name"] = war[candidate_col]
    out["war_candidate_norm"] = war[candidate_col].apply(normalize_name)
    out["war_party"] = war[party_col].apply(standardize_party)
    out["war_cycle"] = pd.to_numeric(war[cycle_col], errors="coerce")
    out["war_score_raw"] = pd.to_numeric(war[war_col], errors="coerce")

    if district_col:
        out["war_district_id"] = war[district_col].apply(normalize_district_id)
    else:
        out["war_district_id"] = ""

    if state_col:
        out["war_state"] = war[state_col].fillna("").astype(str).str.strip().str.upper()
    else:
        # Try to infer from district like CA-13.
        out["war_state"] = out["war_district_id"].str.extract(r"^([A-Z]{2})", expand=False).fillna("")

    out = out.dropna(subset=["war_cycle", "war_score_raw"])
    out["war_cycle"] = out["war_cycle"].astype(int)
    out["war_cycle_weight"] = out["war_cycle"].map(CYCLE_WEIGHTS).fillna(0.05)
    out["war_weighted_score"] = out["war_score_raw"] * out["war_cycle_weight"]

    return out


def aggregate_candidate_war(war):
    grouped = (
        war.groupby(["war_candidate_norm", "war_party"], as_index=False)
        .agg(
            war_candidate_name=("war_candidate_name", "last"),
            war_score_weighted_sum=("war_weighted_score", "sum"),
            war_weight_sum=("war_cycle_weight", "sum"),
            war_cycles=("war_cycle", lambda x: ",".join(str(int(v)) for v in sorted(set(x), reverse=True))),
            war_observations=("war_score_raw", "count"),
            war_latest_cycle=("war_cycle", "max"),
            war_latest_score=("war_score_raw", lambda x: x.iloc[-1]),
        )
    )

    grouped["candidate_war_recency_weighted"] = (
        grouped["war_score_weighted_sum"] / grouped["war_weight_sum"].replace(0, np.nan)
    )

    grouped["candidate_war_recency_weighted"] = grouped["candidate_war_recency_weighted"].fillna(0.0)

    return grouped




def load_candidate_aliases():
    if not ALIASES_PATH.exists():
        return pd.DataFrame(columns=[
            "district_id",
            "party",
            "model_name",
            "war_name",
            "notes",
            "district_id_norm",
            "model_name_norm",
            "war_name_norm",
        ])

    aliases = pd.read_csv(ALIASES_PATH)

    for col in ["district_id", "party", "model_name", "war_name", "notes"]:
        if col not in aliases.columns:
            aliases[col] = ""

    aliases["district_id_norm"] = aliases["district_id"].apply(normalize_district_id)
    aliases["party"] = aliases["party"].astype(str).str.strip().str.upper()
    aliases["model_name_norm"] = aliases["model_name"].apply(normalize_name)
    aliases["war_name_norm"] = aliases["war_name"].apply(normalize_name)

    aliases = aliases[
        aliases["model_name_norm"].astype(str).str.len().gt(0)
        & aliases["war_name_norm"].astype(str).str.len().gt(0)
    ].copy()

    return aliases


def alias_war_name_for_candidate(candidate_name, party, district_id, aliases):
    if aliases.empty:
        return None

    candidate_norm = normalize_name(candidate_name)
    district_norm = normalize_district_id(district_id)
    party_norm = str(party).strip().upper()

    if not candidate_norm:
        return None

    matches = aliases[
        aliases["model_name_norm"].eq(candidate_norm)
        & aliases["party"].eq(party_norm)
        & (
            aliases["district_id_norm"].eq(district_norm)
            | aliases["district_id_norm"].eq("")
        )
    ]

    if matches.empty:
        return None

    return matches.iloc[0]["war_name_norm"]


def match_candidate(candidate_name, party, war_agg, min_score, district_id=None, aliases=None):
    cand_norm = normalize_name(candidate_name)

    if not cand_norm:
        return None

    pool = war_agg[war_agg["war_party"].eq(party)].copy()

    if pool.empty:
        return None

    norm_district = normalize_district_id(district_id) if district_id else ""

    # Alias override: if model name maps to a known WAR name, use that normalized WAR name.
    alias_norm = alias_war_name_for_candidate(candidate_name, party, district_id, aliases) if aliases is not None else None

    if alias_norm:
        pool["name_match_score"] = pool["war_candidate_norm"].apply(
            lambda x: 1.0 if x == alias_norm else 0.0
        )
        pool["alias_match"] = pool["name_match_score"].eq(1.0)
    else:
        pool["name_match_score"] = pool["war_candidate_norm"].apply(
            lambda x: name_similarity(cand_norm, x)
        )
        pool["alias_match"] = False

    # Prefer same-district matches when district data exists.
    if norm_district and "war_district_ids" in pool.columns:
        pool["same_district"] = pool["war_district_ids"].astype(str).apply(
            lambda ids: norm_district in {normalize_district_id(x) for x in ids.split(";") if x}
        )
    else:
        pool["same_district"] = False

    # Safety rule:
    # - Alias matches are allowed.
    # - Exact/near-exact full-name matches can match across districts.
    # - First-initial + last-name matches are allowed only in same district.
    pool["usable_match"] = (
        pool["alias_match"]
        | (pool["name_match_score"] >= 0.97)
        | ((pool["same_district"]) & (pool["name_match_score"] >= min_score - 0.04))
    )

    pool = pool[pool["usable_match"]].copy()

    if pool.empty:
        return None

    pool = pool.sort_values(
        ["alias_match", "same_district", "name_match_score", "war_latest_cycle", "war_observations"],
        ascending=[False, False, False, False, False],
    )

    return pool.iloc[0]


def boolish(x):
    s = str(x).strip().lower()
    return s in {"true", "1", "yes", "y"}




def suggest_war_aliases_for_unmatched(model_name, party, district_id, war_agg, max_suggestions=5):
    """
    Suggest possible WAR names for an unmatched model candidate.

    Priority:
    1. Same district and same party
    2. Same party with strong-ish name similarity
    """
    model_norm = normalize_name(model_name)
    district_norm = normalize_district_id(district_id)
    party_norm = str(party).strip().upper()

    if not model_norm:
        return ""

    pool = war_agg[war_agg["war_party"].eq(party_norm)].copy()

    if pool.empty:
        return ""

    if "war_district_ids" in pool.columns:
        pool["same_district"] = pool["war_district_ids"].astype(str).apply(
            lambda ids: district_norm in {normalize_district_id(x) for x in ids.split(";") if x}
        )
    else:
        pool["same_district"] = False

    pool["name_match_score"] = pool["war_candidate_norm"].apply(
        lambda x: name_similarity(model_norm, x)
    )

    # For suggestions, include all same-district options and any high-ish name match.
    pool = pool[
        pool["same_district"]
        | (pool["name_match_score"] >= 0.75)
    ].copy()

    if pool.empty:
        return ""

    pool = pool.sort_values(
        ["same_district", "name_match_score", "war_latest_cycle", "war_observations"],
        ascending=[False, False, False, False],
    )

    suggestions = []

    for _, row in pool.head(max_suggestions).iterrows():
        suggestions.append(
            f'{row.get("war_candidate_name", "")} '
            f'[{row.get("war_cycles", "")}; score={row.get("candidate_war_recency_weighted", 0):+.2f}; '
            f'match={row.get("name_match_score", 0):.2f}; same_district={bool(row.get("same_district", False))}]'
        )

    return " | ".join(suggestions)


def main():
    settings = read_settings()

    shrinkage = setting(settings, "house_candidate_war_shrinkage", DEFAULT_SHRINKAGE)
    cap = setting(settings, "house_candidate_war_cap", DEFAULT_CAP)
    incumbent_discount = setting(settings, "house_candidate_war_incumbent_discount", DEFAULT_INCUMBENT_DISCOUNT)
    min_name_score = setting(settings, "house_candidate_war_min_name_score", DEFAULT_MIN_NAME_SCORE)
    one_sided_multiplier = setting(
        settings,
        "house_candidate_war_one_sided_multiplier",
        DEFAULT_ONE_SIDED_MULTIPLIER,
    )

    if not RACE_INPUTS_PATH.exists():
        raise FileNotFoundError("inputs/house_race_inputs.csv not found.")

    races = pd.read_csv(RACE_INPUTS_PATH)

    if "district_id" not in races.columns:
        raise ValueError("house_race_inputs.csv must contain district_id.")

    for col in ["dem_candidate", "gop_candidate", "incumbent_party"]:
        if col not in races.columns:
            races[col] = ""

    war = load_war()
    war_agg = aggregate_candidate_war(war)
    aliases = load_candidate_aliases()
    print(f"Loaded {len(aliases)} candidate name aliases.")

    audit_rows = []

    for _, row in races.iterrows():
        district_id = normalize_district_id(row.get("district_id", ""))
        dem_candidate = row.get("dem_candidate", "")
        gop_candidate = row.get("gop_candidate", "")

        incumbent_party = str(row.get("incumbent_party", "")).strip().upper()

        dem_match = match_candidate(dem_candidate, "D", war_agg, min_name_score, district_id, aliases)
        gop_match = match_candidate(gop_candidate, "R", war_agg, min_name_score, district_id, aliases)

        dem_war = float(dem_match["candidate_war_recency_weighted"]) if dem_match is not None else 0.0
        gop_war = float(gop_match["candidate_war_recency_weighted"]) if gop_match is not None else 0.0

        dem_incumbent_discount = incumbent_discount if incumbent_party == "D" else 1.0
        gop_incumbent_discount = incumbent_discount if incumbent_party in {"R", "GOP"} else 1.0

        dem_effective_war = dem_war * dem_incumbent_discount
        gop_effective_war = gop_war * gop_incumbent_discount

        raw_net_dem = dem_effective_war - gop_effective_war
        shrunk = raw_net_dem * shrinkage
        capped_before_match_quality = float(np.clip(shrunk, -cap, cap))

        if dem_match is not None and gop_match is not None:
            war_match_status = "Both matched"
            match_quality_multiplier = 1.0
        elif dem_match is not None:
            war_match_status = "Only D matched"
            match_quality_multiplier = one_sided_multiplier
        elif gop_match is not None:
            war_match_status = "Only R matched"
            match_quality_multiplier = one_sided_multiplier
        else:
            war_match_status = "Neither matched"
            match_quality_multiplier = 0.0

        capped = capped_before_match_quality * match_quality_multiplier

        audit_rows.append(
            {
                "district_id": district_id,
                "dem_candidate": dem_candidate,
                "gop_candidate": gop_candidate,
                "incumbent_party": incumbent_party,
                "dem_war_matched": dem_match is not None,
                "gop_war_matched": gop_match is not None,
                "dem_war_name": dem_match["war_candidate_name"] if dem_match is not None else "",
                "gop_war_name": gop_match["war_candidate_name"] if gop_match is not None else "",
                "dem_name_match_score": float(dem_match["name_match_score"]) if dem_match is not None else 0.0,
                "gop_name_match_score": float(gop_match["name_match_score"]) if gop_match is not None else 0.0,
                "dem_war_cycles": dem_match["war_cycles"] if dem_match is not None else "",
                "gop_war_cycles": gop_match["war_cycles"] if gop_match is not None else "",
                "dem_war_observations": int(dem_match["war_observations"]) if dem_match is not None else 0,
                "gop_war_observations": int(gop_match["war_observations"]) if gop_match is not None else 0,
                "dem_candidate_war": dem_war,
                "gop_candidate_war": gop_war,
                "dem_effective_war": dem_effective_war,
                "gop_effective_war": gop_effective_war,
                "candidate_war_net_dem": raw_net_dem,
                "house_candidate_war_shrinkage": shrinkage,
                "house_candidate_war_cap": cap,
                "house_candidate_war_one_sided_multiplier": one_sided_multiplier,
                "candidate_war_adjustment_dem_before_match_quality": capped_before_match_quality,
                "candidate_war_match_quality_multiplier": match_quality_multiplier,
                "candidate_war_adjustment_dem": capped,
                "war_match_status": war_match_status,
            }
        )

    audit = pd.DataFrame(audit_rows)

    OUTPUTS.mkdir(exist_ok=True)
    audit.to_csv(AUDIT_OUTPUT, index=False)
    audit.to_csv(WAR_MATCH_OUTPUT, index=False)

    unmatched_rows = []

    for _, row in audit.iterrows():
        dem_name = row.get("dem_candidate", "")
        gop_name = row.get("gop_candidate", "")
        district_id = row.get("district_id", "")

        if pd.notna(dem_name) and str(dem_name).strip() and not bool(row.get("dem_war_matched", False)):
            unmatched_rows.append({
                "district_id": district_id,
                "party": "D",
                "model_name": dem_name,
                "suggested_alias_war_name": "",
                "possible_war_matches": suggest_war_aliases_for_unmatched(
                    dem_name,
                    "D",
                    district_id,
                    war_agg,
                ),
                "notes": "",
            })

        if pd.notna(gop_name) and str(gop_name).strip() and not bool(row.get("gop_war_matched", False)):
            unmatched_rows.append({
                "district_id": district_id,
                "party": "R",
                "model_name": gop_name,
                "suggested_alias_war_name": "",
                "possible_war_matches": suggest_war_aliases_for_unmatched(
                    gop_name,
                    "R",
                    district_id,
                    war_agg,
                ),
                "notes": "",
            })

    unmatched = pd.DataFrame(unmatched_rows)

    if not unmatched.empty:
        unmatched = unmatched.sort_values(["district_id", "party", "model_name"])
    unmatched_path = OUTPUTS / "house_candidate_war_unmatched_candidates.csv"
    unmatched.to_csv(unmatched_path, index=False)

    print(f"Wrote {AUDIT_OUTPUT}")
    print(f"Wrote {unmatched_path}")
    print()
    print("WAR match status")
    print("----------------")
    print(audit["war_match_status"].value_counts(dropna=False).to_string())

    print()
    print("Largest proposed WAR adjustments")
    print("--------------------------------")
    show_cols = [
        "district_id",
        "dem_candidate",
        "gop_candidate",
        "war_match_status",
        "dem_candidate_war",
        "gop_candidate_war",
        "candidate_war_net_dem",
        "candidate_war_adjustment_dem",
        "dem_war_cycles",
        "gop_war_cycles",
    ]
    print(
        audit.reindex(audit["candidate_war_adjustment_dem"].abs().sort_values(ascending=False).index)
        .head(30)[show_cols]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
