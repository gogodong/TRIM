"""Plot an Exp-2 attack panel from raw rows.

TRIM rows come from the Exp-2 ``results.csv``/``results.jsonl`` output and must
contain both dataset and ``model_name`` fields.  External methods use the
shared long-table baseline schema documented in ``plotting/README.md``.  The
command rejects missing, extra, or incomplete method/point groups.
"""

try:
    from ._curve import CurveSpec, run_curve_cli
except ImportError:  # Direct ``python plotting/plot_exp2_attacks.py`` use.
    from _curve import CurveSpec, run_curve_cli  # type: ignore


SPEC = CurveSpec(
    experiment="exp2_attacks",
    description="Plot one Exp-2 attack panel.",
    trim_input_help="Repeat for each selected Exp-2 result file.",
)


if __name__ == "__main__":
    run_curve_cli(SPEC)
