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
utility loss. Privacy references use D, the training split. Review
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

Original K0 is computed on the declared raw QIs of the training split. Tail risk
is P99 over that same population D: a published training individual
has log risk -log(k_i), and an individual absent from the release has -infinity.
Validation/test individuals are excluded from D. Only training individuals
absent from the release receive zero risk. Both TRIM and KAnon use this
population, including for K0; model level-0 bins do not merge raw K0 classes.
The empirical inverse CDF selects rank ceil(0.99*|D|), avoiding interpolation
against -infinity. This C03 correction is shared with the submission TRIM
pipeline. Existing results using relative risk or all loaded rows as D need
recomputation.
Other unresolved TRIM privacy/search checks in the revision plan remain open.

Diabetes now keeps the minimum encounter_id per patient_nbr before feature
selection, splitting and any nrows limit. It preserves source-row IDs and
records the actual patient count rather than assuming 71,518. The readmission
label remains positive for both <30 and >30. All affected Diabetes results
need reruns under C29/R64.

Each exact dataset/training split/seed/variant/K fits and saves one complete
tree as compact gzip JSON in `shared_releases/`. MLP and XGBoost share this
artifact while encoding and training their own models. Cache identities include
ordered training features/labels/IDs, hierarchy and implementation hashes;
model settings are deliberately excluded. A different data split or hierarchy
cannot reuse the tree. Cache-hit and fitting/loading timings are recorded.

The corrected first Income point (MLP, seed 42, Median, K=2) was compared
against the archived release: every tree node, partition and training ID is
identical, as is validation delta loss (0.05436071753501892). The complete
artifact shrank from 411,536,290 bytes to 7,027,696 bytes (58.56 times smaller).
Encoding plus model evaluation took 10.60 seconds versus 240.56 seconds on
the earlier implementation. These measurements cover this point, not every
dataset/model/K. The privacy population changed from 112,375 loaded records
to 78,661 training records, as intended.

Every model/K point stores a small `release.json` reference with a relative
path, checksum and model protocol, plus `metrics.json` and routing diagnostics.
The shared compressed file retains all training row IDs, partitions, nodes,
observed regions and provenance. Use `artifacts.load_release_payload(path)`
to read either a reference, compressed tree or legacy plain JSON. Fitted trees use
`hierarchy_mondrian.v2`; the previous routing schema is rejected rather than
silently reinterpreted. Each task stores
its split IDs, resolved protocol and original-model reference. Sweep manifests
record completion/failure; `raw_results.jsonl` preserves completed points.

Focused tests are in this repository's `tests/test_kanon_baseline.py`. They
cover K/coverage, label-dependent splits, zero-gain fallback, categorical
legality, missing values, numeric-gap/whole-record/root fallback, routing
counts, training-leaf invariance and saved-tree reconstruction,
shared encodings, model-vector collisions, training-only raw K0/P99 in both
TRIM and KAnon, compressed cache sharing/invalidation, fallback causes and
Diabetes deduplication. Encoding writes whole attribute blocks and reuses
representations instead of assigning pandas rows for every partition. Routing
skips attributes identical to their parent's region; containment semantics are
unchanged.
Run them explicitly from the repository root:

```bash
python -m pytest -q tests/test_kanon_baseline.py
```

Verification status (2026-10-05): all 24 focused tests passed. The initial
Income pilot was stopped after 72 points; its artifacts remain archived under
the earlier all-loaded-rows definition. The corrected validation pilot uses
both variants, both model families, seeds 42–46 and the complete K grid (360
points). Its results and the reviewed variant choice are pending. The full
sweep and historical Table IV rerun
have not been executed. Local data paths, logs and generated results are
excluded from the repository.

## Investigated fallback at small K

Income seed 42 has 78,661 training and 16,857 validation rows. At K=2,
Median has 31,181 partitions and 10,741 validation fallbacks (63.72%);
InfoGain has 32,554 partitions and 12,446 fallbacks (73.83%). Every training
row still reaches its original leaf. No validation record reaches the global
root. The following causes are mutually exclusive counts of validation stops:

| Cause | Median | InfoGain |
|---|---:|---:|
| Unoccupied categorical child in the local training partition | 4,626 | 5,898 |
| Observed range shrank on a numeric QI other than the split attribute | 4,700 | 5,530 |
| Gap in the split numeric QI's observed ranges | 1,267 | 899 |
| Both numeric causes | 148 | 119 |

Only 447 validation records have an exact ten-QI combination present in
training; 16,410 (97.35%) do not. Globally unseen values account for only 0–2
records per attribute. Most gaps arise inside small local partitions, even
when every individual attribute value exists elsewhere in training. The
categorical value exists elsewhere in training for 4,624 of 4,626 Median
branch stops and all 5,898 InfoGain branch stops. Observed
numeric ranges shrink on **every** numeric QI in a child, including when a
categorical attribute was split. The whole record must stop if any QI fails.
Fallback regions typically remain fairly specific: their median training
population is 7 records for Median and 8 for InfoGain, rather than the entire
training set.

Fitting-domain/threshold routing would send 3,414 Median and 3,009 InfoGain
validation records to leaves whose published regions exclude a true value.
Those counts come from a diagnostic counterfactual, never model evaluation.
It would hide some containment failures rather than solve them. The previously
assumed <1% fallback rate is unsupported at small K. The 1% review flag stays;
the agreed containment/whole-record policy remains unchanged pending review.

Reproduce the diagnosis on validation features only, without predicting test
labels or changing the routing rule:

```bash
python -m experiments.exp1_kanon.diagnose_routing \
  --task-dir <task-directory> \
  --release <task-directory>/median/k_2/release.json \
  --output <diagnostic-output.json>
```
