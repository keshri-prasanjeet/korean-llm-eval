"""CLI entry point for evaluation runs.

Usage:
    python -m eval.run --help
"""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a model evaluation.")
    parser.add_argument("--config", required=True, help="Path to the run config (YAML/JSON).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise NotImplementedError(f"Evaluation not implemented yet (config: {args.config}).")


if __name__ == "__main__":
    main()
