"""Plot one seed's raw chronological Exp-6b2 selection-rule trajectories."""

try:
    from ._exp6_line import Exp6LineSpec, main
except ImportError:
    from _exp6_line import Exp6LineSpec, main  # type: ignore


if __name__ == "__main__":
    main(Exp6LineSpec(
        experiment_kind="exp6b2_selection",
        description=__doc__,
    ))
