#!/usr/bin/env python3

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]

PRODUCTION_FILES = [
    REPO_ROOT / "run_house_model.py",
    REPO_ROOT / "run_house_full_pipeline.py",
    REPO_ROOT / "run_house_full_pipeline_core.py",
]

REPLAY_CANDIDATES = [
    REPO_ROOT
    / "historical"
    / "house"
    / "backtests"
    / "run_house_full_production_replay.py",
    REPO_ROOT
    / "historical"
    / "house"
    / "backtests"
    / "run_house_production_replay.py",
]

OUTPUT_DIR = (
    REPO_ROOT
    / "historical"
    / "house"
    / "diagnostics"
    / "outputs"
)

COMPONENTS: list[dict[str, Any]] = [
    {
        "component": "partisan_baseline",
        "production_terms": [
            "district_partisan_baseline_dem",
            "presidential",
            "partisan_baseline",
        ],
        "replay_terms": [
            "district_partisan_baseline_dem",
            "build_production_fundamentals",
        ],
    },
    {
        "component": "national_environment",
        "production_terms": [
            "national_environment",
            "environment_multiplier",
        ],
        "replay_terms": [
            "national_environment",
            "build_production_fundamentals",
        ],
    },
    {
        "component": "incumbency",
        "production_terms": [
            "incumbency",
            "incumbency_bonus",
        ],
        "replay_terms": [
            "incumbency",
            "incumbency_bonus",
        ],
    },
    {
        "component": "district_elasticity",
        "production_terms": [
            "district_elasticity",
            "elasticity",
        ],
        "replay_terms": [
            "district_elasticity",
            "elasticity",
        ],
    },
    {
        "component": "candidate_quality",
        "production_terms": [
            "candidate_quality",
            "candidate_war",
            "war",
        ],
        "replay_terms": [
            "candidate_quality_weight",
            "candidate_war_path",
            "candidate_war",
        ],
    },
    {
        "component": "district_polling",
        "production_terms": [
            "polling_margin",
            "bayesian",
            "polling_weight",
        ],
        "replay_terms": [
            "historical_polling",
            "polling_margin",
            "bayesian",
        ],
    },
    {
        "component": "polling_variance_reduction",
        "production_terms": [
            "polling_variance",
            "posterior_sd",
            "polling_confidence",
        ],
        "replay_terms": [
            "polling_variance_reduction",
            "posterior_sd",
            "polling_confidence",
        ],
    },
    {
        "component": "national_correlated_error",
        "production_terms": [
            "national_error",
            "national_sd",
        ],
        "replay_terms": [
            "national_error",
            "national_sd",
        ],
    },
    {
        "component": "region_correlated_error",
        "production_terms": [
            "region_error",
            "regional_error",
            "region_groups",
        ],
        "replay_terms": [
            "region_error",
            "regional_error",
            "region_error_groups",
        ],
    },
    {
        "component": "demographic_correlated_error",
        "production_terms": [
            "demographic_error",
            "education",
            "race_group",
            "grouped_variance",
        ],
        "replay_terms": [
            "demographic_error",
            "demographic_error_groups",
            "education",
            "race_group",
        ],
    },
    {
        "component": "district_residual_error",
        "production_terms": [
            "district_error",
            "district_residual",
            "remaining_sd",
        ],
        "replay_terms": [
            "district_error",
            "district_residual",
            "remaining_sd",
        ],
    },
    {
        "component": "variance_preserving_uncertainty",
        "production_terms": [
            "variance_preserving",
            "uncertainty_mode",
            "grouped_variance",
        ],
        "replay_terms": [
            "variance_preserving",
            "uncertainty_mode",
            "grouped_variance",
        ],
    },
    {
        "component": "fixed_party_control",
        "production_terms": [
            "party_control_fixed",
            "fixed_control",
        ],
        "replay_terms": [
            "party_control_fixed",
            "fixed_control",
        ],
    },
    {
        "component": "simulation_seed_control",
        "production_terms": [
            "seed",
            "random_seed",
            "default_rng",
        ],
        "replay_terms": [
            "seed",
            "default_rng",
        ],
    },
    {
        "component": "majority_probability",
        "production_terms": [
            "dem_control_probability",
            "HOUSE_CONTROL_THRESHOLD",
            "majority",
        ],
        "replay_terms": [
            "dem_control_probability",
            "HOUSE_CONTROL_THRESHOLD",
        ],
    },
]


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="replace")


def locate_replay_file() -> Path:
    for path in REPLAY_CANDIDATES:
        if path.exists():
            return path

    matches = sorted(
        (
            REPO_ROOT
            / "historical"
            / "house"
            / "backtests"
        ).glob("*production*replay*.py")
    )

    if not matches:
        raise FileNotFoundError(
            "No House production replay script found."
        )

    return matches[0]


def contains_any(text: str, terms: list[str]) -> list[str]:
    lowered = text.lower()
    return [
        term
        for term in terms
        if term.lower() in lowered
    ]


def extract_deferred_components(
    replay_text: str,
) -> list[str]:
    try:
        tree = ast.parse(replay_text)
    except SyntaxError:
        return []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue

        for key_node, value_node in zip(
            node.keys,
            node.values,
        ):
            if not (
                isinstance(key_node, ast.Constant)
                and key_node.value
                == "production_components_deferred"
            ):
                continue

            if not isinstance(
                value_node,
                (ast.List, ast.Tuple),
            ):
                continue

            values: list[str] = []

            for element in value_node.elts:
                if (
                    isinstance(element, ast.Constant)
                    and isinstance(element.value, str)
                ):
                    values.append(element.value)

            return values

    return []


def extract_included_components(
    replay_text: str,
) -> list[str]:
    try:
        tree = ast.parse(replay_text)
    except SyntaxError:
        return []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue

        for key_node, value_node in zip(
            node.keys,
            node.values,
        ):
            if not (
                isinstance(key_node, ast.Constant)
                and key_node.value
                == "production_components_included"
            ):
                continue

            if not isinstance(
                value_node,
                (ast.List, ast.Tuple),
            ):
                continue

            values: list[str] = []

            for element in value_node.elts:
                if (
                    isinstance(element, ast.Constant)
                    and isinstance(element.value, str)
                ):
                    values.append(element.value)

            return values

    return []


def normalize_component_name(value: str) -> str:
    normalized = re.sub(
        r"[^a-z0-9]+",
        "_",
        value.lower(),
    ).strip("_")

    aliases = {
        "national_error": "national_correlated_error",
        "district_error": "district_residual_error",
        "region_error_groups": "region_correlated_error",
        "demographic_error_groups": (
            "demographic_correlated_error"
        ),
        "historical_polling": "district_polling",
        "polling_variance_reduction": (
            "polling_variance_reduction"
        ),
    }

    return aliases.get(normalized, normalized)


def main() -> None:
    replay_path = locate_replay_file()

    production_text = "\n".join(
        read_text(path)
        for path in PRODUCTION_FILES
    )
    replay_text = read_text(replay_path)

    if not production_text.strip():
        raise RuntimeError(
            "No production runner source was available."
        )

    if not replay_text.strip():
        raise RuntimeError(
            f"Replay file was empty: {replay_path}"
        )

    included_raw = extract_included_components(
        replay_text
    )
    deferred_raw = extract_deferred_components(
        replay_text
    )

    included = {
        normalize_component_name(value)
        for value in included_raw
    }
    deferred = {
        normalize_component_name(value)
        for value in deferred_raw
    }

    rows: list[dict[str, Any]] = []

    for definition in COMPONENTS:
        component = definition["component"]

        production_matches = contains_any(
            production_text,
            definition["production_terms"],
        )
        replay_matches = contains_any(
            replay_text,
            definition["replay_terms"],
        )

        production_detected = bool(
            production_matches
        )
        replay_detected = bool(replay_matches)

        if component in deferred:
            status = "DEFERRED"
        elif component in included:
            status = "INCLUDED_DECLARED"
        elif production_detected and replay_detected:
            status = "LIKELY_INCLUDED"
        elif production_detected and not replay_detected:
            status = "MISSING_FROM_REPLAY"
        elif not production_detected and replay_detected:
            status = "REPLAY_ONLY_OR_ALIAS"
        else:
            status = "NOT_DETECTED"

        rows.append(
            {
                "component": component,
                "status": status,
                "production_detected": (
                    production_detected
                ),
                "replay_detected": replay_detected,
                "production_matches": "; ".join(
                    production_matches
                ),
                "replay_matches": "; ".join(
                    replay_matches
                ),
                "declared_included": (
                    component in included
                ),
                "declared_deferred": (
                    component in deferred
                ),
            }
        )

    audit = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        OUTPUT_DIR
        / "house_production_replay_component_audit.csv"
    )
    json_path = (
        OUTPUT_DIR
        / "house_production_replay_component_audit.json"
    )
    txt_path = (
        OUTPUT_DIR
        / "house_production_replay_component_audit.txt"
    )

    audit.to_csv(csv_path, index=False)

    payload = {
        "replay_path": str(
            replay_path.relative_to(REPO_ROOT)
        ),
        "production_files": [
            str(path.relative_to(REPO_ROOT))
            for path in PRODUCTION_FILES
            if path.exists()
        ],
        "declared_included_raw": included_raw,
        "declared_deferred_raw": deferred_raw,
        "status_counts": (
            audit["status"]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
        "components": audit.to_dict(
            orient="records"
        ),
    }

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    display_columns = [
        "component",
        "status",
        "production_detected",
        "replay_detected",
    ]

    report_lines = [
        "HOUSE PRODUCTION REPLAY COMPONENT AUDIT",
        "=" * 78,
        "",
        f"Replay file: {replay_path.relative_to(REPO_ROOT)}",
        "",
        "Declared included:",
        *(
            [f"  - {value}" for value in included_raw]
            or ["  - none"]
        ),
        "",
        "Declared deferred:",
        *(
            [f"  - {value}" for value in deferred_raw]
            or ["  - none"]
        ),
        "",
        audit[display_columns].to_string(
            index=False
        ),
        "",
        "STATUS COUNTS",
        "-" * 78,
        audit["status"]
        .value_counts()
        .sort_index()
        .to_string(),
        "",
    ]

    txt_path.write_text(
        "\n".join(report_lines)
    )

    print("\n".join(report_lines))

    print("Wrote:")
    for path in [
        csv_path,
        json_path,
        txt_path,
    ]:
        print(f"  {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
