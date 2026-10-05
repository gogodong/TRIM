# Plotting

Run plotting commands from the project root. Each command reads explicit CSV,
JSON, or JSONL inputs and writes one figure. Use `--help` for the complete
argument list shared by a plotting entry.

## Common curve inputs

Exp-1 and Exp-2 external baselines use a long-table schema:

| Field | Description |
|---|---|
| `experiment` | Experiment identifier |
| `method` | Displayed method name |
| `dataset` | Dataset key |
| `model` | Model key |
| `seed` | Replicate seed |
| `point` | Experimental point identifier |
| `x_metric`, `x_value` | Recorded x-axis metric and value |
| `x_reference` | Per-seed reference for an x-axis ratio transform |
| `y_metric`, `y_value` | Recorded y-axis metric and value |
| `y_reference` | Per-seed reference for a y-axis ratio transform |

Use `--expected-point METHOD=POINT` once for every expected point. The
`negative-log-ratio` transform computes `-log(value/reference)` per seed before
aggregation. Aggregation can be `mean` or `median`; error displays can be
`none`, `std`, `sem`, or `minmax`.

## Exp-1 privacy--utility

Input: `trajectory.csv` or `trajectory.jsonl` from
`experiments.exp1_privacy_utility` and optional external-baseline tables.

```bash
python plotting/plot_exp1_privacy_utility.py \
  --trim-results <trajectory.csv-or-jsonl> \
  --baseline-results <external-baseline.csv-or-json> \
  --output <figure.pdf> \
  --dataset income --model <model-key> \
  --trim-method TRIM --methods TRIM <baseline-methods...> \
  --seeds 42 43 44 45 46 \
  --expected-point TRIM=<trim-point> \
  --expected-point <baseline-method>=<baseline-point> \
  --point-column iteration \
  --x-metric test_delta_u --x-transform identity \
  --y-metric min_k --y-transform negative-log-ratio \
  --trim-y-reference-column original_leak_k \
  --aggregate mean --error std \
  --xlabel "Delta U" --ylabel "-log(K/K0)"
```

`test_delta_u` is the per-seed test logloss minus the corresponding full-data
test logloss. Omit `--baseline-results` when plotting TRIM alone.
For the paper's tail-risk panel, use `--y-metric tail_risk_p99` with
`--y-transform identity`. This field is computed per release as
`P99_i_in_D(log(1/k_i) if i is released else -infinity)` over D, the training
split. Validation/test people are excluded. It uses the empirical inverse CDF.
Earlier result files using relative risk or a population that includes
validation/test people require recomputation.

The built-in KAnon runner exports `baseline_long.csv` in this schema, with
one globally selected variant named `KAnon`. It supplies both `min_k` and
`tail_risk_p99`. See [its pilot and sweep workflow](../experiments/exp1_kanon/README.md).

## Exp-1 DP comparison

Input: `raw_results.csv`, JSON, or JSONL from `experiments.exp1_dp`.

```bash
python plotting/plot_exp1_dp.py \
  --input <raw-results.csv-or-json-or-jsonl> \
  --output <figure.pdf> \
  --experiment <experiment-id> --dataset income \
  --target-k <target-k> \
  --methods <all-methods...> --seeds <all-seeds...> \
  --point-field <protocol-point-field> \
  --expected-points <all-protocol-points...> \
  --x-field <recorded-x-field> \
  --y-metric <test_logloss-or-dp_penalty> \
  --aggregation mean --error std \
  --x-scale linear --y-scale linear \
  --x-label <label> --y-label <label>
```

For `dp_penalty`, the plotter pairs DP-SGD and SGD rows with the same method,
seed, and training protocol, then computes the test-logloss difference.

## Exp-2 attacks

Input: `results.csv` or `results.jsonl` from `experiments.exp2_attacks` and
optional external-baseline tables.

```bash
python plotting/plot_exp2_attacks.py \
  --trim-results <attack-results.csv-or-jsonl> \
  --baseline-results <external-baseline.csv-or-json> \
  --output <figure.pdf> \
  --dataset income --model <model-key> \
  --trim-method TRIM --methods TRIM <baseline-methods...> \
  --seeds 42 43 44 45 46 \
  --expected-point TRIM=<trim-point> \
  --expected-point <baseline-method>=<baseline-point> \
  --point-column release_point \
  --x-metric min_k --x-transform negative-log-ratio \
  --trim-x-reference-column original_leak_k \
  --y-metric reconstruction_error --y-transform identity \
  --aggregate mean --error std \
  --xlabel "-log(K/K0)" --ylabel "Reconstruction error"
```

Use a separate invocation for each reconstruction or linkage metric.

## Exp-3 runtime and scalability

Input: `run_summaries.csv` from either Exp-3 runner and optional baseline
runtime tables.

```bash
python plotting/plot_exp3_runtime.py \
  --figure real \
  --input <run_summaries.csv> \
  --baseline-input <external-runtime-baselines.csv-or-json> \
  --output <figure.pdf> \
  --primary-method TRIM --method-field method \
  --x-field dataset --y-field paper_algorithm_time_seconds \
  --seed-field random_state \
  --expected-methods TRIM PAT FIDO Feature-Sel Entry-wise \
  --expected-seeds 42 43 44 45 46 \
  --x-values income pubcov bm diabetes \
  --normalize-to-method TRIM \
  --x-type categorical --aggregation mean --error std \
  --x-label Dataset --y-label "Runtime / TRIM"
```

For BNG scalability, use `--figure bng`, `--x-field nrows`,
`--x-type numeric`, and list the expected row counts with `--x-values`.

## Exp-5 sensitivity

Input: `run_summaries.csv` for Top-K and initial-sample-ratio panels.

```bash
python plotting/plot_exp5_sensitivity.py \
  --figure topk --input <run_summaries.csv> \
  --baseline-input <external-baselines.csv-or-json> \
  --output <figure.pdf> \
  --primary-method TRIM --method-field method \
  --x-field rank_top_k --y-field final_leak_k \
  --y-transform negative-log-ratio \
  --y-reference-field original_leak_k \
  --seed-field random_state \
  --expected-methods TRIM <baseline-methods...> \
  --expected-seeds 42 43 44 45 46 \
  --x-values <all-top-k-values...> \
  --aggregation mean --error std \
  --x-label "Top-K" --y-label "-log(K/K0)"
```

For the initial-sample-ratio panel, use `--figure seed-ratio`,
`initial_sample_fraction` as the x field, and an emitted K field such as
`report_selected_min_k` as the y field.

The background-knowledge panel reads one `background_knowledge.csv` per seed:

```bash
python plotting/plot_exp5_sensitivity.py \
  --figure background \
  --input <seed-42/background_knowledge.csv> \
          <seed-43/background_knowledge.csv> \
          <seed-44/background_knowledge.csv> \
          <seed-45/background_knowledge.csv> \
          <seed-46/background_knowledge.csv> \
  --input-seeds 42 43 44 45 46 \
  --output <figure.pdf> \
  --primary-method TRIM --method-field method \
  --x-field known_attribute_count \
  --y-field strongest_mean_success_probability \
  --seed-field random_state \
  --expected-methods TRIM \
  --expected-seeds 42 43 44 45 46 \
  --x-values <all-known-attribute-counts...> \
  --aggregation mean --error std \
  --x-label "Known attributes" --y-label "Success probability"
```

## Exp-32 ranking comparison

Candidate input schema:

```text
seed,iteration,candidate_type,candidate_id,
mlp_lga_score,yang_if_distance_score,random_score,exact_utility_delta
```

Runtime input schema:

```text
seed,iteration,method,time_seconds,candidate_count,timing_scope
```

```bash
python plotting/plot_exp32_ranking.py \
  --input <seed_42_candidates.csv> <seed_43_candidates.csv> \
          <seed_44_candidates.csv> <seed_45_candidates.csv> \
          <seed_46_candidates.csv> \
  --runtime-input <seed_42_runtime.csv> <seed_43_runtime.csv> \
                  <seed_44_runtime.csv> <seed_45_runtime.csv> \
                  <seed_46_runtime.csv> \
  --output <exp32-ranking.pdf> \
  --top-k <k>
```

The panels report execution time relative to TRIM, HitRate@K, and Recall@K.

## Exp-33 utility-delta estimation

Required input columns:

```text
seed,iteration,method,actual_utility_gap_to_oracle,ratio_abs_error,time_seconds
```

`method` is one of `trim`, `if`, `lga`, or `retrain`. Each row records the
method-selected action's actual Utility Delta and rho gaps to the approximate
oracle, along with the measured iteration runtime.

```bash
python plotting/plot_exp33_estimation.py \
  --input <seed_42_estimates.csv> <seed_43_estimates.csv> \
          <seed_44_estimates.csv> <seed_45_estimates.csv> \
          <seed_46_estimates.csv> \
  --output <exp33-time-mae.pdf>
```

The panels report execution time relative to TRIM, Utility MAE, and rho MAE.
Every seed uses all of its recorded iterations and is summarized before
equal-weight aggregation across seeds.

## Exp-6 ablations

The structural and selection-rule entries read `iterations.csv` or an
equivalent JSON row table and plot one selected seed. Each method supplies its
own chronological iterations. Figure 5h instead reads five-seed final
per-method summaries.

Exp-6a2:

```bash
python plotting/plot_exp6a2_line_results.py \
  --input <exp6a2-experiment-dir/iterations.csv> \
  --output <exp6a2-line.pdf> \
  --dataset pubcov --model xgboost \
  --tolerance <value> --seed 42 \
  --expected-methods TRIM TRIM-H TRIM-V RefineOnly RefineEvict \
  --x-field test_delta_u \
  --top-k-field min_k --top-reference-field original_leak_k \
  --bottom-field tail_risk_p99 \
  --x-label "Delta U" --top-y-label "-log(K/K0)" \
  --bottom-y-label "Tail risk (P99 individual Delta H)"
```

Figure 5h candidate enumeration:

```bash
python plotting/plot_exp6_enumeration.py \
  --input <exp6a1-experiment-dir/run_summaries.csv> \
  --output <figure-5h.pdf> \
  --dataset pubcov --model xgboost \
  --tolerance 0.1 --seeds 42 43 44 45 46
```

For every seed, the renderer requires exactly one utility-feasible row for each
of `TRIM`, `Sample`, `KMeans`, `IL`, and `RandomSplit`. It computes execution
time relative to that seed's TRIM time and computes `leak` directly as
`-log(final_leak_k/original_leak_k)`, then averages the seed-level values with
equal weight.

Exp-6b2:

```bash
python plotting/plot_exp6b2_line_results.py \
  --input <exp6b2-experiment-dir/iterations.csv> \
  --output <exp6b2-line.pdf> \
  --dataset pubcov --model xgboost \
  --tolerance <value> --seed 42 \
  --expected-methods TRIM Privacy Utility Random IF \
  --x-field test_delta_u \
  --top-k-field min_k --top-reference-field original_leak_k \
  --bottom-field tail_risk_p99 \
  --x-label "Delta U" --top-y-label "-log(K/K0)" \
  --bottom-y-label "Tail risk (P99 individual Delta H)"
```

The bottom panel reads the already-computed per-person P99 value directly; it
must not derive tail risk from a ratio of equivalence-class-size percentiles.

## Appendix removal view

```bash
python plotting/plot_appendix_removal_view.py \
  --input <run_summaries.csv> \
  --baseline-input <trim-results.csv-or-json> \
  --output <figure.pdf> \
  --x-field dataset --seed-field random_state \
  --series TRIM-R=final_leak_k \
  --baseline-method-field method --baseline-x-field dataset \
  --baseline-seed-field random_state --baseline-value-field final_leak_k \
  --expected-methods TRIM-R TRIM \
  --expected-seeds 42 43 44 45 46 \
  --x-values income pubcov diabetes bm \
  --normalize-to-method TRIM \
  --aggregation mean --error std \
  --x-label Dataset --y-label "K ratio / TRIM"
```

All plotting entries accept `--title`, `--figsize WIDTH HEIGHT`, `--dpi`, and
`--overwrite` where applicable.
