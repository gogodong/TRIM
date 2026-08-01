"""Plot one seed's raw chronological Exp-6a2 structural trajectories."""

try:
    from ._exp6_line import Exp6LineSpec, main
except ImportError:
    from _exp6_line import Exp6LineSpec, main  # type: ignore


if __name__ == "__main__":
    main(Exp6LineSpec(
        experiment_kind="exp6a2_structural",
        description=__doc__,
    ))
