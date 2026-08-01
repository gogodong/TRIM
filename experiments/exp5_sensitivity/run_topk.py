"""Run the rank-top-K sensitivity matrix with TRIM."""

from __future__ import annotations

import argparse

from .._runner import run_matrix_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp-5 rank-top-K sensitivity.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device")
    parser.add_argument("--run-label")
    args = parser.parse_args()

    output = run_matrix_config(
        args.config,
        expected_kind="exp5_topk",
        device=args.device,
        run_label=args.run_label,
    )
    print(output)


if __name__ == "__main__":
    main()
