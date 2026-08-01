"""Run Exp-3 on the four real-world datasets."""

from __future__ import annotations

import argparse

from .._runner import require_core_parameters, run_matrix_config
from ._metrics import extract_runtime_to_first_feasible


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp-3 real-data timing.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device")
    parser.add_argument("--run-label")
    args = parser.parse_args()

    require_core_parameters("stop_on_utility", "record_iteration_test_metrics")
    output = run_matrix_config(
        args.config,
        expected_kind="exp3_runtime_real",
        device=args.device,
        run_label=args.run_label,
        after_task=extract_runtime_to_first_feasible,
    )
    print(output)


if __name__ == "__main__":
    main()
