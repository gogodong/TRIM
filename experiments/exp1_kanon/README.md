# KAnon baseline for Exp-1

The new baseline fits Median Mondrian and InfoGain Mondrian with TRIM's QI
hierarchies. It publishes every training record exactly once, generalizes only
QIs, and preserves labels and non-QI predictors. There is no suppression.
This package is separate from the historical Table IV implementation in
`experiments/exp1_dp/mondrian.py`; it does not reproduce that experiment's
categorical ordering, encoding, optimizer, or evaluation protocol.

## Algorithm and encoding

- Median chooses the legal split on the widest normalized QI. Numeric cuts
  balance record counts without splitting equal values.
- InfoGain examines all legal numeric thresholds and multiway categorical
  splits, maximizing training-label entropy reduction. If no positive gain
  remains, it uses the median rule.
- Categorical splits replace a hierarchy node with its children. Every
  nonempty child, including a missing-value branch, needs at least K records.
  A single occupied child is descended without removing records.
- Every training node stores a region: its categorical hierarchy nodes and
  observed numeric [min, max] ranges, before splitting. Terminal regions are
  exactly the published training partitions. Split thresholds remain fitting
  metadata; evaluation routing checks containment in every QI of a child's
  region. It descends into the unique matching child and stops at the current
  region when no child contains the whole record.
- A separate global root stores every QI at its hierarchy root, including
  numeric QIs. Its only child is the observed training region. This preserves
  the training release even if K=N_train leaves it unsplit, while records
  outside the training range can stop at the global root.
- An empty categorical branch stops at its containing parent. A numeric gap
  such as 35 between children [30, 34] and [36, 40] stops at their observed
  parent [30, 40]. The entire record uses the stopped node's region. All
  training records still reach their own published leaves, and evaluation
  records do not change the fitted tree, publication or privacy metrics.
- Values absent from training but present in the fixed hierarchy are supported.
  Nonmissing values outside that schema raise an error; the runner does not
  extend a hierarchy using validation/test observations.
- Missing categorical values use an explicit internal missing branch with the
  shared all-zero leaf vector. Root domains also admit missing values. A
  missing branch smaller than K prevents that categorical split.
- XGBoost uses the existing shared leaf-column order, uniform weights over
  descendant categorical leaves or overlapping numeric leaves, and one level
  column per QI. Numeric level is the height of the least common ancestor.
  The global fallback explicitly publishes the numeric hierarchy root, so it
  retains that root's actual height even if a unary child has the same leaves.
  Shared TRIM encoding now also records actual node height on irregular trees.
- MLP uses uniform categorical leaf one-hot means. Numeric ranges average
  represented scalar leaves for a scalar-leaf tree, or all represented
  integers in the clipped union for an interval-leaf tree. It does not use the
  empirical training-row mean. A range inside a numeric bin can retain finer
  MLP resolution. Matching published hierarchy values have matching inputs.
  The standardizer is fitted once on shared level-0 training inputs.

Privacy classes use canonical published QI intervals/nodes, not encoded model
vectors. Two distinct ranges can map to the same XGBoost leaf vector without
becoming the same privacy class.

## Pilot, decision, full sweep

Run these commands from the repository root (`submission_code/` in the parent
workspace).

```bash
python -m experiments.exp1_kanon.run \
  --config configs/experiments/exp1_kanon.template.yaml
```

The default pilot uses only Income, both model families, all five configured
seeds, and both variants. It trains on training data and reports validation
loss. It does not encode test model inputs, predict on them, or compute test
utility loss. Original-population privacy references still use D. Review
`raw_results.csv` against achieved min-K. Keep InfoGain unless the validation
pilot gives a clear reason to choose Median; record that reason explicitly.
There is no automated definition of “clearly stronger” and no test-based or
panel-by-panel variant choice.

```bash
python -m experiments.exp1_kanon.select_variant \
  --pilot-dir <pilot-directory> \
  --variant <median-or-infogain> \
  --rationale "<reason based on the validation curves>"

python -m experiments.exp1_kanon.run \
  --config configs/experiments/exp1_kanon.template.yaml \
  --mode full --selection <pilot-directory>/variant_selection.json
```

The full sweep applies the chosen variant to Income, PubCov, Diabetes and BM,
MLP and XGBoost, seeds 42–46. Dataset paths, model settings and stratified
70/15/15 splits come directly from the same matrix YAML as TRIM. Changed
anonymizer/shared-protocol code or changed Income data/tree/model/split
settings invalidate the pilot choice. Use the same device in pilot and full.
Data/model/tree hashes and exact split row IDs are persisted for comparisons.

Default Ks are powers of two from 2 through N_train/2, additional valid values
12 and 235, and N_train. The original model reference is saved separately in
`reference.json`; K=1 is not treated as an anonymization point. Repeat
`--k <value>` or set `sweep.k_values` for a smaller explicit sweep.

The full run writes one `KAnon` curve to `baseline_long.csv`, ready for
`plotting.plot_exp1_privacy_utility --baseline-results <file>`. Use
`test_delta_u` on x and `min_k` with `negative-log-ratio` on y. For tail
risk, use `tail_risk_p99` with the identity transform. Supply the actual K
points through the existing plotter's repeated
`--expected-point KAnon=<K>` arguments.

Each K point also writes `routing_diagnostics.json` separately for validation
and test (test is null in pilot mode). It records mutually exclusive leaf,
internal-node and global-root counts, fallback count/fraction, per-node
`eval_assignment_counts`, stop-depth counts and
`eval_coarsest_release_row_count`. The raw metric rows include the counts,
fractions and review flags. The configurable `routing_review_threshold`
defaults to 0.01: strictly more than 1% stopping above a leaf is flagged and
printed for review at every K. Inspect those flags, especially at small K,
before trusting the curve. Fallback rarity is measured rather than assumed.

## Metrics, artifacts and verification

Original K0 is computed on the declared raw QIs of every loaded row. Tail risk
is P99 over that same original population D: a published training individual
has log risk -log(k_i), and an individual absent from the release has -infinity.
Thus validation/test individuals remain in D and have zero release risk.
The empirical inverse CDF selects rank ceil(0.99*|D|), avoiding interpolation
against -infinity. This C03 correction is shared with the submission TRIM
pipeline. Existing results using the old relative-risk definition need reruns.
Other unresolved TRIM privacy/search checks in the revision plan remain open.

Diabetes now keeps the minimum encounter_id per patient_nbr before feature
selection, splitting and any nrows limit. It preserves source-row IDs and
records the actual patient count rather than assuming 71,518. The readmission
label remains positive for both <30 and >30. All affected Diabetes results
need reruns under C29/R64.

Every K point stores `release.json` (training row IDs, partitions, split tree,
all observed node regions, the global hierarchy root and hashes),
`metrics.json` and routing diagnostics. Fitted trees use
`hierarchy_mondrian.v2`; the previous routing schema is rejected rather than
silently reinterpreted. Each task stores
its split IDs, resolved protocol and original-model reference. Sweep manifests
record completion/failure; `raw_results.jsonl` preserves completed points.

Focused tests are in this repository's `tests/test_kanon_baseline.py`. They
cover K/coverage, label-dependent splits, zero-gain fallback, categorical
legality, missing values, numeric-gap/whole-record/root fallback, routing
counts, training-leaf invariance and saved-tree reconstruction,
shared encodings, model-vector collisions, C03 and Diabetes deduplication.
Run them explicitly from the repository root:

```bash
python -m pytest -q tests/test_kanon_baseline.py
```

Verification status (2026-10-05): all 20 focused tests passed. The Income
validation pilot was started in tmux with both variants, both model families,
seeds 42–46 and the complete K grid (360 points). Its results and the reviewed
variant choice are pending. The full sweep and historical Table IV rerun
have not been executed. Local data paths, logs and generated results are
excluded from the repository.
