"""Plot an Exp-1 privacy--utility panel from raw rows.

TRIM rows come from the Exp-1 ``trajectory.csv``/``trajectory.jsonl`` output.
External methods use the shared long-table baseline schema documented in
``plotting/README.md``.  The command requires the exact point set of every
method and rejects missing, extra, or incomplete points.
"""

try:
    from ._curve import CurveSpec, run_curve_cli
except ImportError:  # Direct ``python plotting/plot_exp1_privacy_utility.py`` use.
    from _curve import CurveSpec, run_curve_cli  # type: ignore


SPEC = CurveSpec(
    experiment="exp1_privacy_utility",
    description="Plot one Exp-1 privacy--utility panel.",
    trim_input_help="Repeat for each selected Exp-1 trajectory file.",
)


if __name__ == "__main__":
    run_curve_cli(SPEC)
