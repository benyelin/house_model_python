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

    if final_round.sum() < 4:
        raise RuntimeError(
            "Expected at least four Alaska final-round polls."
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
