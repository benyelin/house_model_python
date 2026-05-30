import argparse
import subprocess
import sys


def run(cmd):
    print()
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run full House model pipeline.")
    parser.add_argument("--sims", type=int, default=20000)
    parser.add_argument("--import-seed", action="store_true", help="Re-import House Model Data.xlsx before running.")
    args = parser.parse_args()

    py = sys.executable

    if args.import_seed:
        run([py, "import_house_model_seed.py"])

    run([py, "update_house_elasticity.py"])
    run([py, "recalculate_house_fundamentals.py"])
    run([py, "run_house_model.py", "--sims", str(args.sims)])

    print()
    print("House pipeline complete.")
    print("Launch dashboard:")
    print("  streamlit run dashboard_house_app.py")


if __name__ == "__main__":
    main()
