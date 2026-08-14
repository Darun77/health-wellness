"""
main.py
--------
Single entry point to run the whole project end to end, or one stage
at a time.

Usage (from the project root, with venv activated):
    python main.py --stage all
    python main.py --stage data
    python main.py --stage ml
    python main.py --stage dl
"""

import argparse


def run_data():
    from src import data_generation
    data_generation.main()


def run_ml():
    from src import ml_pipeline
    ml_pipeline.main()


def run_dl():
    from src import dl_pipeline
    dl_pipeline.main()


def main():
    parser = argparse.ArgumentParser(description="Health & Wellness Coaching project runner")
    parser.add_argument(
        "--stage",
        choices=["all", "data", "ml", "dl"],
        default="all",
        help="Which stage to run (default: all)",
    )
    args = parser.parse_args()

    if args.stage in ("all", "data"):
        print("\n[1/3] Generating synthetic datasets...")
        run_data()

    if args.stage in ("all", "ml"):
        print("\n[2/3] Running Level 1 - ML pipeline (segmentation, regression, classification)...")
        run_ml()

    if args.stage in ("all", "dl"):
        print("\n[3/3] Running Level 2 - DL pipeline (LSTM + Attention burnout model)...")
        run_dl()

    print("\nDone. Check the 'reports/' folder for metrics/plots and 'models/' for saved artifacts.")


if __name__ == "__main__":
    main()
