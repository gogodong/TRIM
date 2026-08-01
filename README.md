# TRIM

This repository provides the implementation for the paper *Privacy–Utility
Optimization for Training Data Minimization*. TRIM jointly applies vertical
and horizontal training-data minimization while preserving model utility
within a specified tolerance.

## Repository structure

| Path | Description |
|---|---|
| `prototype/` | TRIM implementation and command-line entry point |
| `experiments/` | Experiment runners and experiment-specific code |
| `plotting/` | Scripts for generating figures from raw experiment outputs |
| `configs/experiments/` | Configuration files for individual experiments |
| `configs/generalization_trees/` | Generalization trees for the five datasets and a prompt for generating compatible trees |
| `scripts/` | Utilities for preparing generalization-tree inputs |
| `data/` | Dataset download and placement instructions |

Run all commands below from the project root.

## Environment

Create and activate the paper's GPU reference environment:

```bash
conda env create -f environment.yml
conda activate trim
```

This environment fixes Python 3.10.19, PyTorch 2.9.1, and CUDA Toolkit 12.2 as
reported in the paper. PyTorch uses its official CUDA 12.6 GPU wheel. A
compatible NVIDIA driver is required.

```bash
python -c "import sys, torch; print(sys.version); print(torch.__version__); print(torch.version.cuda)"
nvcc --version
nvidia-smi
```

## Data

Raw datasets are not included. Place them at the default locations below, or
set `data_path` in the selected configuration file.

| Dataset | Default input |
|---|---|
| Income | `data/acs/ACSIncome_CA_2014_X.csv` and `ACSIncome_CA_2014_y.csv` |
| PubCov | `data/acs/acs_public_coverage_CA_2014_X.csv` and `acs_public_coverage_CA_2014_y.csv` |
| Diabetes | `data/diabetes_130_us_hospitals/diabetic_data.csv` |
| BM | `data/bank_marketing/bank-full.csv` |
| BNG | `data/bng_credit_g/BNG_credit-g.arff` |

The ACS files for Income and PubCov can be downloaded and prepared through
Folktables. See [`data/README.md`](data/README.md) for sources, licenses, and
detailed placement instructions.

Model training
and utility estimation use every predictor, while retention classes,
generalization, and privacy metrics use only the declared QIs.

## Quick start

The example configuration runs TRIM on Income. The first ACS run may download
the source data if it is not already available.

```bash
python -m prototype.run_prototype \
  --config configs/prototype.template.yaml \
  --device cpu
```

The command prints a JSON summary and writes outputs under
`results/trim_example/`. Copy the configuration to change the dataset, model,
tolerance, device, or output directory. See
[`prototype/README.md`](prototype/README.md) for the Python interface and
advanced options.

## Experiments

Configuration files may be copied and edited; relative paths are resolved from
the project root.

The experiment configuration defines three model roles:

- `model.estimator` is used for first-stage LGA ranking.
- `model.proxy` is used for second-stage candidate certification.
- `model.downstream` is used for final utility evaluation.

The three model roles are configured separately.

TRIM evaluates all horizontal candidates by default. When
`pipeline.reference_model_filtering` is enabled, the proxy fitted for the
selected release is retained as the reference model for the next iteration.
Candidate evaluation is then restricted to retention classes containing a
record that the proxy classifies correctly while the current estimator does
not.

| Experiment | Entry point | Configuration | Main raw output |
|---|---|---|---|
| Exp-1 privacy--utility | `experiments.exp1_privacy_utility.run` | `exp1_privacy_utility.template.yaml` | `trajectory.jsonl` and per-run `trajectory.csv` |
| Exp-1 DP comparison | `experiments.exp1_dp.run_exp1_dp` | `exp1_dp.template.yaml` | `raw_results.csv` and paired summaries |
| Exp-2 attacks | `experiments.exp2_attacks.run_exp2_attacks` | `exp2_attacks.template.yaml` | `results.csv` and per-release details |
| Exp-3 real-data runtime | `experiments.exp3_runtime.run_real` | `exp3_runtime_real.template.yaml` | `run_summaries.csv` |
| Exp-3 BNG scalability | `experiments.exp3_runtime.run_bng` | `exp3_bng_scalability.template.yaml` | `run_summaries.csv` |
| Exp-5 top-K | `experiments.exp5_sensitivity.run_topk` | `exp5_topk.template.yaml` | `run_summaries.csv` |
| Exp-5 initial-sample ratio | `experiments.exp5_sensitivity.run_seed_ratio` | `exp5_seed_ratio.template.yaml` | `run_summaries.csv` and selected-point JSON |
| Exp-5 background knowledge | `experiments.exp5_sensitivity.run_background` | `exp5_background.template.yaml` | per-run `background_knowledge.csv` |
| Exp-6a1 candidate enumeration | `experiments.exp6_ablation.run_enumeration` | `exp6_enumeration.template.yaml` | `run_summaries.csv` and `iterations.csv` |
| Exp-6a2 structural ablation | `experiments.exp6_ablation.run_structural` | `exp6_structural.template.yaml` | `iterations.csv` and `iterations.jsonl` |
| Exp-6b2 selection-rule ablation | `experiments.exp6_ablation.run_selection` | `exp6_selection.template.yaml` | `iterations.csv` and `iterations.jsonl` |
| Exp-32 MLP-LGA ranking | `experiments.exp32_ranking.run` | `exp32_ranking.template.yaml` | per-seed candidate and runtime CSVs |
| Exp-33 Utility-Delta estimation | `experiments.exp33_estimation.run` | `exp33_estimation.template.yaml` | per-seed estimate CSVs |
| Appendix removal view | `experiments.appendix_removal_view.run` | `appendix_removal_view.template.yaml` | `run_summaries.csv` |

### External baselines

The baseline implementations used in the comparison experiments are maintained
in their original repositories:

| Baseline | Repository |
|---|---|
| PAT | [eth-sri/datamin](https://github.com/eth-sri/datamin) |
| FeatSel | [swjz/data-minimization](https://github.com/swjz/data-minimization) |
| Entry-wise DM | [prakharg24/data-minimization-principle](https://github.com/prakharg24/data-minimization-principle) |
| FIDO | [divyashan/learning_to_limit](https://github.com/divyashan/learning_to_limit) |

Run the corresponding upstream implementation to produce each baseline's raw
results, convert those outputs to the plotting input schema, and pass the
converted tables to the plotting commands. The accepted table schemas and
command-line arguments are documented in [`plotting/README.md`](plotting/README.md).

### Exp-1: Privacy and utility

This experiment expands the four real datasets, two downstream models, and
five seeds declared in the configuration. It records downstream test loss at
every TRIM iteration. The privacy axis is computed per seed as `-log(K/K0)`,
where K0 is read from the input data. The search tolerance is `0.01`;
observation-only downstream retraining time is recorded separately from TRIM
execution time.

Each trajectory row also records `tail_risk_p99`, computed by matching release
rows to their original row IDs, evaluating
`log(K_original_i / K_current_i)` for every assigned individual, and then
taking the 99th percentile. Ratios of aggregate K percentiles are retained only
as descriptive compatibility fields and are not used as paper tail risk.

```bash
python -m experiments.exp1_privacy_utility.run \
  --config configs/experiments/exp1_privacy_utility.template.yaml
```

Each sweep writes `run_summaries.csv` and `trajectory.jsonl`; individual TRIM
runs also contain `trajectory.csv` and `trim_iterations.json`.

### Exp-1: Differential privacy

This experiment compares the original Income training data, full-release
Mondrian at K = 235, and the final K = 235 TRIM representation. All three use
matched Poisson minibatch SGD and five random seeds. DP-SGD is evaluated for
`epsilon = ln(x)`, where `x in {5, 25, 100}`.

First run TRIM with `release_target_k=235`, then supply the resulting run
directory. The runner checks that it used 120,000 Income rows, a 0.15/0.15
validation/test split, split seed 42, and target K = 235:

```bash
python -m experiments.exp1_dp.run_exp1_dp \
  --config configs/experiments/exp1_dp.template.yaml \
  --run-dir /absolute/path/to/the/trim/run
```

This experiment requires `opacus` and `scipy`. It writes the raw rows,
seed-paired summaries, training-mode summaries, and the resolved configuration
under `results/exp1_dp/`. The default five-seed summary uses a `0.95`
confidence level.

### Exp-2: Reconstruction and strong-linkage attacks

The reconstruction attack predicts each original QI value from a mixed-level
TRIM release. The strong-linkage attack predicts a raw B attribute from
released A attributes, with A and B required to be disjoint. A configuration
may reference an existing `trim_run_dir` or define a `trim_task` to execute
first.

```bash
python -m experiments.exp2_attacks.run_exp2_attacks \
  --config configs/experiments/exp2_attacks.template.yaml
```

The runner writes `results.jsonl`, `results.csv`, and detailed per-release JSON.
`release_points: all_trim_iterations` evaluates every persisted iteration
and the final release; an explicit list is also accepted. The privacy axis is
computed per seed as `-log(min_k/original_leak_k)`.

### Exp-3: Runtime and scalability

The real-data experiment evaluates Income, PubCov, Diabetes, and BM. The BNG
experiment varies the number of rows. Runtime includes candidate enumeration,
selection, privacy computation, and backend retraining; observation-only test
retraining is reported separately.

```bash
python -m experiments.exp3_runtime.run_real \
  --config configs/experiments/exp3_runtime_real.template.yaml

python -m experiments.exp3_runtime.run_bng \
  --config configs/experiments/exp3_bng_scalability.template.yaml
```

Each sweep writes `run_summaries.csv`. Individual TRIM runs also contain
`paper_runtime.json` with the selected iteration and timing breakdown. Runtime
records require the enumeration, selection, privacy, retraining, initial-class,
and candidate-enumeration timing fields produced by TRIM.

### Exp-5: Sensitivity analyses

The top-K experiment varies `pipeline.rank_top_k`. The initial-sample experiment
varies the S0 fraction and reports the first trajectory point whose test-loss
increase satisfies `report_tolerance`. The background-knowledge experiment
enumerates QI subsets and measures candidate-set and success-probability
statistics from the final TRIM release. The supplied initial-sample
configuration uses `report_tolerance: 0.055`; this post-run filter is distinct
from `pipeline.tolerance`.

```bash
python -m experiments.exp5_sensitivity.run_topk \
  --config configs/experiments/exp5_topk.template.yaml

python -m experiments.exp5_sensitivity.run_seed_ratio \
  --config configs/experiments/exp5_seed_ratio.template.yaml

python -m experiments.exp5_sensitivity.run_background \
  --config configs/experiments/exp5_background.template.yaml
```

Top-K and initial-sample plots use per-seed `-log(K/K0)`. The background plot
uses the recorded success probability directly.

### Exp-6: Ablations

Figure 5h compares TRIM's dynamic retention classes with four alternative
horizontal admission units on all 152,676 PubCov rows. `Sample` uses individual
rows, `KMeans` uses KMeans cluster-distance rings, `IL` groups rows by
generalization-tree information loss, and `RandomSplit` creates seeded
equal-sized partitions.
Each static method enumerates the level-0 training rows once, publishes its
largest group as the initial state at maximum generalization, and keeps that
partition fixed during refinement. The static initial group is not privacy
exempt. The supplied configuration runs five seeds; runtime is normalized to
TRIM within each seed before equal-weight aggregation.

```bash
python -m experiments.exp6_ablation.run_enumeration \
  --config configs/experiments/exp6_enumeration.template.yaml
```

Figure 5k (Exp-6a2) compares structural variants using the complete trajectory
for seed 42 under shared data, model, split, search, and tolerance settings.

| Method | Definition |
|---|---|
| `TRIM` | Dynamic retention-class actions and vertical refinement with S0 initialization and full post-refinement swap |
| `TRIM-H` | Fixed level-0 KMeans cluster-ring row groups and whole-ring admissions |
| `TRIM-V` | Vertical refinement only, starting with all rows at maximum generalization |
| `RefineOnly` | TRIM row and refinement actions without post-refinement swap or eviction |
| `RefineEvict` | TRIM row and refinement actions with threshold-based eviction after refinement |

The supplied `TRIM-H` configuration uses 7 KMeans clusters, 7 percentile rings,
`n_init=10`, and ring seed 42.

Figure 5l (Exp-6b2) compares action-selection rules using the complete
trajectory for seed 42. The non-random methods retain the per-type Logistic-LGA
top-K prefilter.

| Method | Action score |
|---|---|
| `TRIM` | `utility_gain / privacy_cost` |
| `Privacy` | `1 / privacy_cost` |
| `Utility` | `utility_gain` |
| `Random` | Uniform selection from the complete horizontal and vertical action pool |
| `IF` | Fixed-original MLP influence projected onto validation Utility Delta, divided by `privacy_cost` |

Here, `utility_gain` is the current TRIM validation loss minus the candidate
TRIM validation loss, and
`privacy_cost = log(max(current K - candidate K, 2))`. Run both experiments
with an explicit tolerance:

```bash
python -m experiments.exp6_ablation.run_structural \
  --config configs/experiments/exp6_structural.template.yaml \
  --tolerance <value>

python -m experiments.exp6_ablation.run_selection \
  --config configs/experiments/exp6_selection.template.yaml \
  --tolerance <value>
```

All three experiments write per-method summaries and chronological iteration
tables.

### Exp-32: MLP-LGA ranking quality

This experiment uses rolling MLP-LGA for horizontal and vertical candidates.
Each candidate also receives a seeded Random score and a Yang-style
fixed-original-MLP distance score. Ranking precedes top-K selection and TRIM
release evaluation.

```bash
python -m experiments.exp32_ranking.run \
  --config configs/experiments/exp32_ranking.template.yaml
```

Each seed produces `seed_*_candidates.csv` and `seed_*_runtime.csv`. Runtime is
reported separately for TRIM, influence functions, Random, and exact retraining
over their corresponding candidate pools. The supplied configuration runs five
seeds; every seed is summarized over all of its own iterations before the
seed-level metrics are averaged.

### Exp-33: Utility-Delta estimation

This experiment compares the actions selected by TRIM, a Yang-style influence
projection, rolling MLP-LGA, and an approximate retraining oracle. Candidate
actions are ranked by estimated Utility Delta per unit privacy cost. The
approximate oracle retrains the leading 200 candidates of each action type.

```bash
python -m experiments.exp33_estimation.run \
  --config configs/experiments/exp33_estimation.template.yaml
```

Each of the five seeds produces `seed_*_estimates.csv`. The rows contain the
method-selected action's Utility and rho errors relative to the approximate
oracle, together with measured runtime. The plotter uses every iteration
recorded by each seed and then aggregates seed summaries with equal weight.

### Appendix: Removal view

The removal-view variant starts with every level-0 training row. It may remove
a complete active retention class, coarsen one attribute by one level, or
exchange a complete active class for a removed-side class without increasing
row count or reducing min-K. Candidate privacy first maximizes min-K and then
minimizes the number of classes tied at min-K; validation loss is a hard gate.

The supplied configuration uses MLP downstream and proxy models, a logistic
estimator, five seeds, and the following dataset limits:

| Dataset | Maximum rows | Utility tolerance |
|---|---:|---:|
| Income | 120000 | 0.05 |
| PubCov | 92867 | 0.05 |
| Diabetes | 101766 | 0.02 |
| BM | 45211 | 0.005 |

```bash
python -m experiments.appendix_removal_view.run \
  --config configs/experiments/appendix_removal_view.template.yaml
```

The appendix plot reports the per-dataset, per-seed `TRIM-R / TRIM` ratio.
Each invocation writes the resolved task and configuration, candidate and
action records, selected iterations, row identifiers, privacy distribution,
timing components, and final metrics to a new experiment directory.

## Generating figures

Plotters consume explicit CSV, JSON, or JSONL result files. Ratio and log-ratio
metrics are computed from raw rows within each seed before aggregation.

| Experiment | Plotting entry | Metric or comparison |
|---|---|---|
| Exp-1 privacy--utility | `plotting.plot_exp1_privacy_utility` | test Utility Delta against `-log(K/K0)` |
| Exp-1 DP | `plotting.plot_exp1_dp` | DP-SGD logloss or method-relative DP penalty |
| Exp-2 attacks | `plotting.plot_exp2_attacks` | `-log(K/K0)` against reconstruction or linkage performance |
| Exp-3 runtime/scalability | `plotting.plot_exp3_runtime` | method/TRIM runtime ratio or raw BNG runtime |
| Exp-5 sensitivity | `plotting.plot_exp5_sensitivity` | privacy, utility, timing, or background success probability |
| Exp-6a2 / Exp-6b2 | `plotting.plot_exp6a2_line_results` / `plotting.plot_exp6b2_line_results` | chronological utility, privacy, and timing curves |
| Exp-32 ranking | `plotting.plot_exp32_ranking` | ranking quality and candidate-scoring runtime |
| Exp-33 estimation | `plotting.plot_exp33_estimation` | runtime ratio, Utility MAE, and rho MAE |
| Appendix removal view | `plotting.plot_appendix_removal_view` | per-seed TRIM-R/TRIM ratio |

See [`plotting/README.md`](plotting/README.md) for input schemas and complete
plotting commands.

## Outputs

TRIM and experiment configurations write to subdirectories of `results/`.
Experiment outputs include manifests and the raw CSV, JSON, or JSONL records
needed for analysis and plotting. Each invocation creates a new result
directory and does not overwrite prior runs.
