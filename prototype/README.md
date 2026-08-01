# TRIM core

This directory contains the core TRIM implementation. The Python entry point is
`TRIM_prototype_pipeline.run_trim_pipeline(...)`; the configuration-driven
command-line entry point is `run_prototype.py`.

TRIM uses three separately configured model roles:

- `model.estimator` performs first-stage LGA ranking.
- `model.proxy` performs second-stage candidate certification.
- `model.downstream` performs final utility evaluation.

Model specifications are validated against the requirements of each role.

TRIM evaluates every horizontal candidate by default. Set
`reference_model_filtering=True` in the Python interface, or
`pipeline.reference_model_filtering: true` in a configuration file, to use the
fitted proxy for the current release as a reference model. When enabled, a
remaining retention class is enumerated only if it contains a record classified
correctly by the proxy and incorrectly by the current estimator.

## Supported datasets

- Income (ACS Income)
- PubCov (ACS Public Coverage)
- Diabetes (Diabetes 130-US Hospitals)
- BM (Bank Marketing)
- BNG (synthetic credit-g)

The five generalization trees are stored in
`../configs/generalization_trees/`. See `../data/README.md` for data sources
and expected file locations.

## Running TRIM

Run from the project root:

```bash
python -m prototype.run_prototype --config configs/prototype.template.yaml
```

Relative paths in configuration files are resolved from the project root.
Select `cuda`, `cpu`, or another supported device through the configuration
or command line. Results are written only when `results_dir` is set.

`initial_sample_fraction` is applied to the training rows after loading,
cleaning, and the train/validation/test split. Automatically sampled initial
rows are stratified and contain every training label. It is mutually exclusive
with `initial_sample_size` and explicit `initial_row_ids`; explicit rows must
also contain every training label.

## Modules

- `TRIM_prototype_pipeline.py`: search, candidate selection, publication, and metrics.
- `experiment_hooks.py`: ablation controls, scoring context, and observer events.
- `dataset_registry.py` and `dataloader.py`: datasets and generalization-tree encoding.
- `enumeration_horizontal.py`: retention-class enumeration.
- `greedy_selection.py`: search state, logs, and generalization-level utilities.
- `lga.py`: LGA candidate scoring.
- `gpu_*.py` and `model_factory.py`: model implementations.
- `backend_training.py`: training-feature standardization.

## Experiment interfaces

`run_trim_pipeline(...)` provides the following optional experiment controls:

```python
run_trim_pipeline(
    ...,
    ablation_actions=None,
    fixed_row_candidate_builder=None,
    candidate_shortlist=None,
    candidate_state_scorer_factory=None,
    selection_score=None,
    candidate_observer=None,
    reference_model_filtering=False,
    stop_on_utility=True,
    record_iteration_test_metrics=False,
    trim_iteration_observer=None,
)
```

`AblationActions` supports:

- `enabled_action_types`: `retention_class`, `vertical_refinement`, or both.
- `initialization`: `s0` or `all_rows_at_max_generalization`.
- `vertical_postprocessing`: `full_swap`, `none`, or `evict_low_k`.
- `move_out_threshold`: a positive integer used by `evict_low_k`.

Example:

```python
from prototype import AblationActions, run_trim_pipeline

actions = AblationActions(
    enabled_action_types=("vertical_refinement",),
    initialization="all_rows_at_max_generalization",
    vertical_postprocessing="evict_low_k",
    move_out_threshold=3,
)
result = run_trim_pipeline(..., ablation_actions=actions)
```

`fixed_row_candidate_builder(context)` supplies a fixed, non-overlapping
partition of training rows. It may be combined with vertical refinement only
when `vertical_postprocessing="none"`, so refinement cannot move rows across
the fixed candidate groups.

`candidate_shortlist(context)` selects positions from the complete horizontal
and vertical candidate pool. By default, TRIM uses the per-type LGA top-K.

`candidate_state_scorer_factory(context)` constructs a numerical scorer for
the current and candidate states. Its output is available to
`selection_score(context)`. Incompatible scorers and non-finite values raise
an error.

`candidate_observer(observation)` receives a read-only record for each scored
candidate. `trim_iteration_observer(observation)` receives a read-only
record after an action is committed. Their time is reported in
`candidate_observer_time_seconds` and
`trim_iteration_observer_time_seconds` and is excluded from selection
time.

With `stop_on_utility=False`, TRIM continues until `max_iterations` or until no
valid action remains. With `record_iteration_test_metrics=True`, TRIM retrains
the downstream model for each committed snapshot and writes the measurements
to `trim_iterations.json`. This observation cost is reported separately
as `experiment_observation_time_seconds`.
