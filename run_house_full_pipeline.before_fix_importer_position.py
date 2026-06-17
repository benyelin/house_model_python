from pathlib import Path
import subprocess
import sys

PYTHON = sys.executable
run([py, "import_house_model_data_excel.py"])


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


def main():
    args = sys.argv[1:]

    # Build/update House candidate WAR before core pipeline.
    # This makes the main House forecast WAR-on by default.
    run_script("Build/update House candidate WAR", "build_house_candidate_war.py")

    # Run the original full House pipeline with the same command-line args
    # such as --import-seed.
    run_script("Run core House full pipeline", "run_house_full_pipeline_core.py", args)

    # Post-forecast WAR diagnostics. These do not feed the same forecast run;
    # they explain where candidate WAR matters most.
    run_script("Build House candidate WAR rankings", "build_house_candidate_war_rankings.py")

    print()
    print("House WAR-on full pipeline complete.")
    print("Main forecast uses candidate WAR generated before the core pipeline.")
    print("For a separate WAR-on/WAR-off diagnostic comparison, run:")
    print("  python3 compare_house_candidate_war_toggle.py")


if __name__ == "__main__":
    main()
