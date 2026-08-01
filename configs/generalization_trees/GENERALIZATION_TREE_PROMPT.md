# Generalization-tree generation prompt

This prompt is intended for the five datasets used by TRIM.

| Dataset key | QI export | Tree output |
|---|---|---|
| `income` | `acs_income_qi_values.md` | `acs_income_generalization_trees.yaml` |
| `pubcov` | `acs_public_coverage_qi_values.md` | `acs_public_coverage_generalization_tree.yaml` |
| `diabetes` | `diabetes_130_us_hospitals_qi_values.md` | `diabetes_130_us_hospitals_generalization_tree.yaml` |
| `bm` | `bank_marketing_qi_values.md` | `bank_marketing_generalization_tree.yaml` |
| `bng` | `bng_credit_g_qi_values.md` | `bng_credit_g_generalization_tree.yaml` |

Export the observed quasi-identifier domains from the project root:

```bash
python scripts/export_qi_values.py --datasets income pubcov diabetes bm bng
```

Use `--data-path DATASET=PATH` when a dataset is not stored at its default
location. The option may be repeated.

Fill in the placeholders below and send the fenced block to an assistant with
web-search and file-writing tools. Paste the Markdown QI export produced for
the selected dataset into `QI VALUE-DOMAIN EXPORT`.

```text
You are creating a generalization-tree YAML file for one TRIM dataset.
Use web search to verify categorical and coded values against authoritative
dataset documentation, then write the completed YAML file to disk.

DATASET
- key: <income | pubcov | diabetes | bm | bng>
- name: <dataset name and version>
- source family: <publisher or catalog>
- year or version: <year or version>
- output filename: <exact tree output filename from the table above>

QI VALUE-DOMAIN EXPORT
<paste the complete *_qi_values.md content here>

OPTIONAL REFERENCES
<paste relevant official code-list excerpts, or leave blank>

OUTPUT
1. Write the YAML to the exact output filename declared above.
2. Use YAML block style and verify that `yaml.safe_load()` can parse the file.
3. Do not paste the full YAML into the response. Report only:
   - the saved path;
   - the authoritative source used for each coded attribute;
   - any code whose meaning could not be confirmed;
   - the three audit-summary booleans.

SOURCE RESEARCH
1. For every `code` attribute, find the official data dictionary or codebook
   for the declared dataset version. Prefer the original publisher, UCI,
   OpenML, or the U.S. Census Bureau as applicable.
2. Record the source name, publisher, URL, and the meaning of every observed
   code under the tree's `source` metadata.
3. Build categorical groups from documented meaning, not numeric proximity.
4. If a code cannot be confirmed after research, keep it as a standalone leaf
   and record that limitation in `notes`.

REQUIRED TOP-LEVEL KEYS
- `schema_version`
- `generated_at`
- `dataset`
- `audit_policy`
- `attribute_observed_domains`
- `trees`
- `audit_summary`

DATASET METADATA
`dataset` must contain:
- `name`
- `qi_attributes`
- `source_reference` or `code_list_file`

AUDIT POLICY
`audit_policy` must state that:
- leaves cover only values present in the QI export;
- every observed value appears in exactly one leaf;
- parent links and child links are consistent;
- sibling leaf domains are disjoint.

OBSERVED DOMAINS
For every QI attribute, `attribute_observed_domains` must contain:
- `attribute_type`
- `observed_count`
- `observed_values` or `observed_values_compressed`

TREE STRUCTURE
For every QI attribute, `trees` must contain:
- `attribute`
- `attribute_type`
- `root`
- `source`
- `notes`
- `nodes`

Every node must contain:
- `id`: stable and unique within the attribute tree
- `kind`: `root`, `internal`, or `leaf`
- `label`
- `parent`: null only for the root
- `children`: child IDs for root/internal nodes, otherwise an empty list
- `leaf_count`
- `leaf_values_compressed`
- `depth_from_root`
- `height_from_leaf`

Every leaf must also contain `value`:
- continuous attributes use a closed interval string such as `0 ~ 4`;
- numeric codes use unquoted numeric YAML values;
- genuine text codes use quoted strings.

TREE REQUIREMENTS
1. Every observed value appears in exactly one leaf.
2. Do not add unobserved values or omit observed values.
3. Every non-root node has exactly one parent.
4. Every child ID exists in the same attribute tree.
5. Sibling nodes cover disjoint leaf domains.
6. The root covers the full observed domain.
7. Continuous sibling intervals do not overlap.
8. Categorical grouping follows researched semantics.
9. Binary attributes normally use two leaf children.
10. Add useful intermediate levels so TRIM can select different
    generalization granularities.
11. Preserve task-relevant distinctions when they are supported by the
    official code meanings and remain consistent with the coverage rules.

AUDIT SUMMARY
`audit_summary` must contain:
- `all_coverage_ok`
- `all_parent_links_ok`
- `all_sibling_disjointness_ok`
- one entry per attribute containing observed and tree leaf counts, missing,
  extra, and duplicate values, node count, maximum height, and the three
  per-attribute audit booleans.

Before saving, verify coverage, parent/child consistency, sibling
disjointness, connected acyclic root-to-leaf paths, source metadata for every
coded attribute, and successful parsing with `yaml.safe_load()`.
```

After generation, compare the output filename and dataset metadata with
`prototype/dataset_registry.py` before using the tree in an experiment.
