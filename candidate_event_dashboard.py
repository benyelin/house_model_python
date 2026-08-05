from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import hashlib
import subprocess
import sys

import numpy as np
import pandas as pd
import streamlit as st

from candidate_event_registry import (
    REQUIRED_COLUMNS,
    VALID_CHAMBERS,
    VALID_CREDIBILITY_LEVELS,
    VALID_EVENT_SCOPES,
    VALID_EVENT_STATUSES,
    VALID_EVENT_TYPES,
    VALID_MEDIA_SALIENCE,
    VALID_POLLING_SUPERSESSION_MODES,
    VALID_SEVERITY_CATEGORIES,
    VALID_CONFIDENCE_LEVELS,
    load_candidate_event_registry,
    validate_candidate_event_registry,
)


TEXT_DEFAULTS = {
    "event_id": "",
    "chamber": "",
    "race_id": "",
    "candidate_name": "",
    "candidate_party": "",
    "event_type": "",
    "event_scope": "",
    "event_status": "",
    "severity_category": "",
    "credibility_level": "",
    "media_salience": "",
    "reported_date": "",
    "effective_date": "",
    "review_date": "",
    "expiration_date": "",
    "confidence": "",
    "source_summary": "",
    "source_url": "",
    "candidate_response": "",
    "analyst_rationale": "",
    "polling_supersession_mode": "manual_review",
}

NUMERIC_DEFAULTS = {
    "cycle": 2026,
    "baseline_event_adjustment_dem": 0.0,
    "analyst_modifier_dem": 0.0,
    "candidate_event_adjustment_dem": 0.0,
}

PARTY_OPTIONS = [
    "",
    "D",
    "R",
    "I",
    "OTHER",
    "UNKNOWN",
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            path,
            low_memory=False,
        )
    except Exception:
        return pd.DataFrame()


def _clean_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .replace(
            {
                "nan": "",
                "NaN": "",
                "None": "",
                "<NA>": "",
            }
        )
        .str.strip()
    )


def _normalize_bool(series: pd.Series) -> pd.Series:
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            {
                "true",
                "1",
                "yes",
                "y",
            }
        )
    )


def _ensure_schema(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    out = frame.copy()

    for column in REQUIRED_COLUMNS:
        if column in out.columns:
            continue

        if column in NUMERIC_DEFAULTS:
            out[column] = NUMERIC_DEFAULTS[column]
        elif column == "active":
            out[column] = False
        else:
            out[column] = TEXT_DEFAULTS.get(
                column,
                "",
            )

    out = out[REQUIRED_COLUMNS].copy()

    for column in NUMERIC_DEFAULTS:
        out[column] = pd.to_numeric(
            out[column],
            errors="coerce",
        ).fillna(NUMERIC_DEFAULTS[column])

    for column in TEXT_DEFAULTS:
        out[column] = _clean_text(
            out[column]
        )

    out["active"] = _normalize_bool(
        out["active"]
    )

    return out


def _drop_blank_rows(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    out = frame.copy()

    meaningful_columns = [
        "event_id",
        "race_id",
        "candidate_name",
        "event_type",
        "source_summary",
        "analyst_rationale",
    ]

    blank = pd.Series(
        True,
        index=out.index,
    )

    for column in meaningful_columns:
        blank &= _clean_text(
            out[column]
        ).eq("")

    blank &= (
        pd.to_numeric(
            out["baseline_event_adjustment_dem"],
            errors="coerce",
        )
        .fillna(0.0)
        .eq(0.0)
    )

    blank &= (
        pd.to_numeric(
            out["analyst_modifier_dem"],
            errors="coerce",
        )
        .fillna(0.0)
        .eq(0.0)
    )

    return out.loc[~blank].copy()


def _make_event_id(
    row: pd.Series,
    row_number: int,
) -> str:
    existing = str(
        row.get("event_id", "")
    ).strip()

    if existing:
        return existing

    raw = "|".join(
        [
            str(row.get("cycle", "2026")),
            str(row.get("chamber", "")),
            str(row.get("race_id", "")),
            str(row.get("candidate_name", "")),
            str(row.get("event_type", "")),
            str(row.get("reported_date", "")),
            str(row_number),
        ]
    )

    digest = hashlib.sha1(
        raw.encode("utf-8")
    ).hexdigest()[:8]

    chamber = (
        str(row.get("chamber", "event"))
        .strip()
        .lower()
        or "event"
    )

    race = (
        str(row.get("race_id", "unknown"))
        .strip()
        .lower()
        .replace(" ", "-")
        or "unknown"
    )

    return (
        f"2026-{chamber}-{race}-{digest}"
    )


def _prepare_for_validation(
    edited: pd.DataFrame,
) -> pd.DataFrame:
    out = _ensure_schema(
        edited
    )

    out = _drop_blank_rows(
        out
    ).reset_index(drop=True)

    out[
        "candidate_event_adjustment_dem"
    ] = (
        pd.to_numeric(
            out[
                "baseline_event_adjustment_dem"
            ],
            errors="coerce",
        ).fillna(0.0)
        + pd.to_numeric(
            out["analyst_modifier_dem"],
            errors="coerce",
        ).fillna(0.0)
    )

    out["event_id"] = [
        _make_event_id(
            row,
            row_number,
        )
        for row_number, (_, row)
        in enumerate(
            out.iterrows(),
            start=1,
        )
    ]

    for column in [
        "reported_date",
        "effective_date",
        "review_date",
        "expiration_date",
    ]:
        values = pd.to_datetime(
            out[column],
            errors="coerce",
        )

        out[column] = values.dt.strftime(
            "%Y-%m-%d"
        ).fillna("")

    out["cycle"] = pd.to_numeric(
        out["cycle"],
        errors="coerce",
    ).fillna(2026).astype(int)

    out["active"] = _normalize_bool(
        out["active"]
    )

    return out[REQUIRED_COLUMNS]


def _race_reference(
    house_race_path: Path,
    senate_race_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    house = _read_csv(
        house_race_path
    )

    if not house.empty:
        for _, row in house.iterrows():
            race_id = str(
                row.get(
                    "district_id",
                    row.get("race_id", ""),
                )
            ).strip()

            if not race_id:
                continue

            rows.append(
                {
                    "chamber": "house",
                    "race_id": race_id,
                    "dem_candidate": str(
                        row.get(
                            "dem_candidate",
                            "",
                        )
                    ).strip(),
                    "gop_candidate": str(
                        row.get(
                            "gop_candidate",
                            "",
                        )
                    ).strip(),
                }
            )

    senate = _read_csv(
        senate_race_path
    )

    if not senate.empty:
        for _, row in senate.iterrows():
            state = str(
                row.get("state", "")
            ).strip().upper()

            if not state:
                continue

            rows.append(
                {
                    "chamber": "senate",
                    "race_id": state,
                    "dem_candidate": str(
                        row.get(
                            "dem_candidate",
                            "",
                        )
                    ).strip(),
                    "gop_candidate": str(
                        row.get(
                            "gop_candidate",
                            "",
                        )
                    ).strip(),
                }
            )

    return pd.DataFrame(rows)


def _run_pipeline(
    command: list[str],
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def render_candidate_event_registry_editor(
    *,
    default_chamber: str,
    registry_path: Path,
    house_race_path: Path,
    senate_race_path: Path,
    house_root: Path,
    senate_root: Path,
    key_prefix: str,
) -> None:
    st.subheader(
        "Candidate Event Registry"
    )

    st.caption(
        "Maintain documented candidate-event adjustments "
        "shared by the House and Senate models. Positive "
        "values help Democrats; negative values help "
        "Republicans. Final adjustment equals baseline "
        "severity plus analyst modifier."
    )

    st.info(
        "Saving the registry does not automatically rerun "
        "either model. Use the explicit pipeline buttons "
        "after reviewing and applying changes."
    )

    if not registry_path.exists():
        st.error(
            "Shared candidate-event registry not found: "
            f"{registry_path}"
        )
        return

    try:
        registry = load_candidate_event_registry(
            registry_path,
            as_of=date.today(),
        )
    except Exception as error:
        st.error(
            "The current candidate-event registry is invalid: "
            f"{error}"
        )
        return

    registry = _ensure_schema(
        registry
    )

    reference = _race_reference(
        house_race_path,
        senate_race_path,
    )

    filter_col1, filter_col2 = st.columns(
        [1, 2]
    )

    with filter_col1:
        chamber_filter = st.selectbox(
            "Display chamber",
            options=[
                "house",
                "senate",
                "all",
            ],
            index=(
                0
                if default_chamber == "house"
                else 1
            ),
            key=(
                f"{key_prefix}_"
                "candidate_event_chamber_filter"
            ),
        )

    with filter_col2:
        active_filter = st.selectbox(
            "Display status",
            options=[
                "all",
                "active",
                "inactive",
            ],
            index=0,
            key=(
                f"{key_prefix}_"
                "candidate_event_active_filter"
            ),
        )

    display = registry.copy()

    if chamber_filter != "all":
        display = display.loc[
            display["chamber"].eq(
                chamber_filter
            )
        ].copy()

    if active_filter == "active":
        display = display.loc[
            display["active"]
        ].copy()
    elif active_filter == "inactive":
        display = display.loc[
            ~display["active"]
        ].copy()

    st.markdown(
        "#### Race and candidate reference"
    )

    reference_display = reference.copy()

    if chamber_filter != "all":
        reference_display = (
            reference_display.loc[
                reference_display[
                    "chamber"
                ].eq(chamber_filter)
            ]
        )

    st.dataframe(
        reference_display,
        use_container_width=True,
        hide_index=True,
        height=220,
    )

    editor_columns = [
        "event_id",
        "cycle",
        "chamber",
        "race_id",
        "candidate_name",
        "candidate_party",
        "event_type",
        "event_scope",
        "event_status",
        "severity_category",
        "credibility_level",
        "media_salience",
        "reported_date",
        "effective_date",
        "review_date",
        "expiration_date",
        "baseline_event_adjustment_dem",
        "analyst_modifier_dem",
        "candidate_event_adjustment_dem",
        "confidence",
        "source_summary",
        "source_url",
        "candidate_response",
        "analyst_rationale",
        "polling_supersession_mode",
        "active",
    ]

    st.markdown(
        "#### Edit candidate events"
    )

    edited = st.data_editor(
        display[editor_columns],
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        key=(
            f"{key_prefix}_"
            "candidate_event_registry_editor"
        ),
        column_config={
            "event_id": (
                st.column_config.TextColumn(
                    "Event ID",
                    help=(
                        "May be left blank for new rows; "
                        "a stable ID is generated on save."
                    ),
                )
            ),
            "cycle": (
                st.column_config.NumberColumn(
                    "Cycle",
                    min_value=2026,
                    max_value=2100,
                    step=2,
                    format="%d",
                )
            ),
            "chamber": (
                st.column_config.SelectboxColumn(
                    "Chamber",
                    options=sorted(
                        VALID_CHAMBERS
                    ),
                )
            ),
            "candidate_party": (
                st.column_config.SelectboxColumn(
                    "Party",
                    options=PARTY_OPTIONS,
                )
            ),
            "event_type": (
                st.column_config.SelectboxColumn(
                    "Event type",
                    options=[
                        "",
                        *sorted(
                            VALID_EVENT_TYPES
                        ),
                    ],
                )
            ),
            "event_scope": (
                st.column_config.SelectboxColumn(
                    "Scope",
                    options=[
                        "",
                        *sorted(
                            VALID_EVENT_SCOPES
                        ),
                    ],
                )
            ),
            "event_status": (
                st.column_config.SelectboxColumn(
                    "Status",
                    options=[
                        "",
                        *sorted(
                            VALID_EVENT_STATUSES
                        ),
                    ],
                )
            ),
            "severity_category": (
                st.column_config.SelectboxColumn(
                    "Severity",
                    options=[
                        "",
                        *sorted(
                            VALID_SEVERITY_CATEGORIES
                        ),
                    ],
                )
            ),
            "credibility_level": (
                st.column_config.SelectboxColumn(
                    "Credibility",
                    options=[
                        "",
                        *sorted(
                            VALID_CREDIBILITY_LEVELS
                        ),
                    ],
                )
            ),
            "media_salience": (
                st.column_config.SelectboxColumn(
                    "Media salience",
                    options=[
                        "",
                        *sorted(
                            VALID_MEDIA_SALIENCE
                        ),
                    ],
                )
            ),
            "baseline_event_adjustment_dem": (
                st.column_config.NumberColumn(
                    "Baseline",
                    min_value=-3.0,
                    max_value=3.0,
                    step=0.25,
                    format="%.2f",
                )
            ),
            "analyst_modifier_dem": (
                st.column_config.NumberColumn(
                    "Modifier",
                    min_value=-3.0,
                    max_value=3.0,
                    step=0.25,
                    format="%.2f",
                )
            ),
            "candidate_event_adjustment_dem": (
                st.column_config.NumberColumn(
                    "Final adjustment",
                    disabled=True,
                    format="%.2f",
                    help=(
                        "Computed on Preview or Apply as "
                        "baseline plus analyst modifier."
                    ),
                )
            ),
            "confidence": (
                st.column_config.SelectboxColumn(
                    "Confidence",
                    options=[
                        "",
                        *sorted(
                            VALID_CONFIDENCE_LEVELS
                        ),
                    ],
                )
            ),
            "polling_supersession_mode": (
                st.column_config.SelectboxColumn(
                    "Polling treatment",
                    options=sorted(
                        VALID_POLLING_SUPERSESSION_MODES
                    ),
                )
            ),
            "active": (
                st.column_config.CheckboxColumn(
                    "Active",
                    default=False,
                )
            ),
        },
    )

    button1, button2, _, button4, button5 = (
        st.columns(
            [1, 1, 0.3, 1, 1]
        )
    )

    preview_clicked = button1.button(
        "Preview Changes",
        key=(
            f"{key_prefix}_"
            "preview_candidate_events"
        ),
    )

    apply_clicked = button2.button(
        "Apply Registry",
        type="primary",
        key=(
            f"{key_prefix}_"
            "apply_candidate_events"
        ),
    )

    house_pipeline_clicked = button4.button(
        "Run House Pipeline",
        key=(
            f"{key_prefix}_"
            "run_house_candidate_events"
        ),
    )

    senate_pipeline_clicked = button5.button(
        "Run Senate Pipeline",
        key=(
            f"{key_prefix}_"
            "run_senate_candidate_events"
        ),
    )

    if preview_clicked or apply_clicked:
        try:
            proposed = _prepare_for_validation(
                edited
            )

            result = (
                validate_candidate_event_registry(
                    proposed,
                    as_of=date.today(),
                )
            )

            st.success(
                "Candidate-event registry validation passed."
            )

            metric1, metric2, metric3 = st.columns(
                3
            )

            metric1.metric(
                "Registry rows",
                result.rows,
            )
            metric2.metric(
                "Active events",
                result.active_rows,
            )
            metric3.metric(
                "Active nonzero events",
                result.nonzero_active_rows,
            )

            st.dataframe(
                proposed,
                use_container_width=True,
                hide_index=True,
            )

            if apply_clicked:
                timestamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )

                backup = registry_path.with_name(
                    f"{registry_path.stem}."
                    f"before_dashboard_apply_"
                    f"{timestamp}"
                    f"{registry_path.suffix}"
                )

                registry_path.replace(
                    backup
                )

                try:
                    proposed.to_csv(
                        registry_path,
                        index=False,
                    )

                    load_candidate_event_registry(
                        registry_path,
                        as_of=date.today(),
                    )
                except Exception:
                    if registry_path.exists():
                        registry_path.unlink()

                    backup.replace(
                        registry_path
                    )

                    raise

                st.success(
                    "Shared candidate-event registry saved. "
                    f"Backup: {backup}"
                )

                st.rerun()

        except Exception as error:
            st.error(
                "Candidate-event registry validation failed: "
                f"{error}"
            )

    if house_pipeline_clicked:
        with st.spinner(
            "Running House pipeline..."
        ):
            result = _run_pipeline(
                [
                    sys.executable,
                    "run_house_full_pipeline.py",
                ],
                house_root,
            )

        if result.returncode == 0:
            st.success(
                "House pipeline completed successfully."
            )
        else:
            st.error(
                "House pipeline failed."
            )

        with st.expander(
            "House pipeline output",
            expanded=(
                result.returncode != 0
            ),
        ):
            st.code(
                result.stdout
                + "\n"
                + result.stderr
            )

    if senate_pipeline_clicked:
        with st.spinner(
            "Running Senate pipeline..."
        ):
            result = _run_pipeline(
                [
                    sys.executable,
                    "run_full_pipeline.py",
                ],
                senate_root,
            )

        if result.returncode == 0:
            st.success(
                "Senate pipeline completed successfully."
            )
        else:
            st.error(
                "Senate pipeline failed."
            )

        with st.expander(
            "Senate pipeline output",
            expanded=(
                result.returncode != 0
            ),
        ):
            st.code(
                result.stdout
                + "\n"
                + result.stderr
            )
