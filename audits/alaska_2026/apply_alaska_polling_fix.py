from pathlib import Path
from datetime import datetime
import shutil
import pandas as pd

ROOT = Path.cwd()
INPUTS = ROOT / "inputs"
SCRIPT = ROOT / "ingest_house_polls.py"
RAW = INPUTS / "house_manual_polls.csv"

BACKUP = (
    ROOT
    / "audits"
    / "alaska_2026"
    / (
        "pre_fix_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
)

BACKUP.mkdir(parents=True, exist_ok=False)

files = [
    SCRIPT,
    RAW,
    INPUTS / "house_manual_polls_clean.csv",
    INPUTS / "house_polling_averages_generated.csv",
    INPUTS / "house_race_inputs.csv",
    ROOT / "outputs" / "house_race_stats.csv",
]

for path in files:
    if path.exists():
        shutil.copy2(path, BACKUP / path.name)

print(f"Backup directory: {BACKUP}")

module = ROOT / "alaska_house_polling_override.py"

module.write_text('''
import pandas as pd


FINAL_ROUND_POLLSTERS = {
    "Fabrizio/Impact Research",
    "The Bullfinch Group",
    "NY Times/Siena",
    "Data for Progress",
}


def apply_alaska_house_polling_override(polls):
    out = polls.copy()

    if "district_id" not in out.columns:
        raise RuntimeError("Missing district_id.")

    alaska = (
        out["district_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("AK-AL")
    )

    # Exclude Alaska first-choice polling from the
    # final-round two-candidate margin calculation.
    first_choice = (
        alaska
        & out["pollster"].eq("Alaska Survey Research")
    )

    out = out.loc[~first_choice].copy()

    alaska = (
        out["district_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("AK-AL")
    )

    final_round = (
        alaska
        & out["pollster"].isin(
            FINAL_ROUND_POLLSTERS
        )
    )

    if final_round.sum() != 4:
        raise RuntimeError(
            "Expected exactly four Alaska final-round polls."
        )

    for index in out.index[final_round]:
        pollster = out.at[index, "pollster"]

        if pollster == "Fabrizio/Impact Research":
            if float(out.at[index, "dem_pct"]) != 43.0:
                raise RuntimeError(
                    "Unexpected Fabrizio polling data."
                )

            out.at[index, "sample_size"] = 800
            out.at[index, "start_date"] = "2026-09-08"
            out.at[index, "end_date"] = "2026-09-11"

        else:
            independent = float(
                out.at[index, "ind_pct"]
            )

            if independent <= 0:
                raise RuntimeError(
                    "Missing independent support: "
                    + pollster
                )

            out.at[index, "dem_pct"] = independent

        # This is a modeling-side alignment only.
        # The source CSV retains independent affiliation.

    return out
'''.lstrip())

text = SCRIPT.read_text()

import_line = (
    "from alaska_house_polling_override "
    "import apply_alaska_house_polling_override"
)

if import_line in text:
    raise RuntimeError(
        "Alaska override already installed."
    )

anchor = "    polls, unmatched, dropped = ("

if text.count(anchor) != 1:
    raise RuntimeError(
        "Could not uniquely locate ingestion call."
    )

text = text.replace(
    anchor,
    (
        "    # Alaska 2026: use verified R-vs-I "
        "final-round polling.\\n"
        "    polls = apply_alaska_house_polling_override("
        "polls)\\n\\n"
        + anchor
    ),
    1,
)

import_anchor = "import numpy as np"

if text.count(import_anchor) != 1:
    raise RuntimeError(
        "Could not uniquely locate import block."
    )

text = text.replace(
    import_anchor,
    import_anchor + "\\n" + import_line,
    1,
)

SCRIPT.write_text(text)

print("Installed Alaska-only preprocessing override.")
print("Raw polling CSV unchanged.")
print("Shared polling engine unchanged.")
