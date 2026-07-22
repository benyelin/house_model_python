from pathlib import Path
import subprocess
import sys

PYTHON = sys.executable


def run_script(label, script, extra_args=None):
    script_path = Path(script)

    if not script_path.exists():
        print(f"Skipping missing step: {label} ({script})")
        return

    cmd = [PYTHON, script]

    if extra_args:
        cmd.extend(extra_args)

    print()
    print("=" * 72)
    print(label)
    print("=" * 72)
    print("$ " + " ".join(cmd))

    subprocess.run(cmd, check=True)


def strip_import_seed(args):
    """The second core run should not re-import stale seed data."""
    return [a for a in args if a != "--import-seed"]


def main():
    args = sys.argv[1:]

    # First let the core pipeline perform any requested seed import.
    # This may overwrite inputs/house_race_inputs.csv, so we refresh Excel after it.
    if "--import-seed" in args:
        run_script("Run core House seed import pass", "run_house_full_pipeline_core.py", args)

    # Now import the authoritative Excel candidate/source fields.
    run_script("Import House Model Data Excel", "import_house_model_data_excel.py")

    # Build/update House candidate WAR after candidates are refreshed.
    run_script("Build/update House candidate WAR", "build_house_candidate_war.py")

    # Final forecast run. Do not pass --import-seed again, or it may wipe out Excel updates.
    final_args = strip_import_seed(args)
    run_script("Run core House full pipeline", "run_house_full_pipeline_core.py", final_args)

    # Post-forecast WAR diagnostics.
    run_script("Build House candidate WAR rankings", "build_house_candidate_war_rankings.py")

    print()
    print("House WAR-on full pipeline complete.")
    print("Main forecast uses House Model Data Excel candidate fields and candidate WAR.")
    print("For a separate WAR-on/WAR-off diagnostic comparison, run:")
    print("  python3 compare_house_candidate_war_toggle.py")


if __name__ == "__main__":
    main()

    print()
    print("Validating synchronized House forecast outputs...")
    subprocess.run(
        [
            sys.executable,
            "validation/validate_house_live_outputs.py",
        ],
        check=True,
    )
