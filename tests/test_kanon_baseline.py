"""Focused baseline checks. Run explicitly with pytest when authorized."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp1_kanon.mondrian import FittedMondrian, HierarchySchema
from experiments.exp1_kanon.run import k_grid
from prototype.dataloader import DiabetesReadmissionDataLoader, load_generalization_rules
from prototype.privacy_metrics import individual_tail_risk_stats, original_equivalence_class_sizes
from prototype.release_artifacts import DatasetSplit


@pytest.fixture
def schema_factory():
    def make(numeric_leaves=None):
        numeric_leaves = list(range(8)) if numeric_leaves is None else numeric_leaves
        loader = SimpleNamespace(
            feature_columns=("age", "kind", "aux"), qi_attributes=("age", "kind"),
            numeric_attributes=("age", "aux"), categorical_attributes=("kind",),
            category_maps={"kind": {"a": 0, "b": 1, "c": 2}},
        )
        trees = {
            "age": {"nodes": [
                {"id": "age_root", "parent": None, "kind": "root", "height_from_leaf": 1},
                *[{"id": f"age_{i}", "value": value, "parent": "age_root", "kind": "leaf", "height_from_leaf": 0}
                  for i, value in enumerate(numeric_leaves)],
            ]},
            "kind": {"nodes": [
                {"id": "kind_root", "parent": None, "kind": "root", "height_from_leaf": 2},
                {"id": "ab", "parent": "kind_root", "kind": "internal", "height_from_leaf": 1},
                {"id": "a", "value": "a", "parent": "ab", "kind": "leaf", "height_from_leaf": 0},
                {"id": "b", "value": "b", "parent": "ab", "kind": "leaf", "height_from_leaf": 0},
                {"id": "c", "value": "c", "parent": "kind_root", "kind": "leaf", "height_from_leaf": 0},
            ]},
        }
        return HierarchySchema(load_generalization_rules(trees, loader, generalization_level=0))
    return make


def test_full_coverage_k_and_original_labels(schema_factory):
    schema = schema_factory()
    X = pd.DataFrame({"age": list(range(8)), "kind": ["a", "b", "c", "a"] * 2, "aux": list(range(100, 108))}, index=range(20, 28))
    y = pd.Series([0, 1] * 4, index=X.index)
    original_X, original_y = X.copy(), y.copy()
    for variant in ("median", "infogain"):
        fitted = FittedMondrian(schema, k=2, variant=variant).fit(X, y)
        ids = [row_id for partition in fitted.partitions for row_id in partition["row_ids"]]
        assert sorted(ids) == X.index.tolist()
        assert len(ids) == len(set(ids))
        assert fitted.per_record_k(X.index).min() >= 2
        encoded = schema.encode(X, fitted.partitions, family="mlp")
        np.testing.assert_array_equal(encoded["aux"], X["aux"])
        training_groups = fitted.transform_groups(X)
        expected_leaf = {
            position: partition["node_id"]
            for partition in fitted.partitions for position in partition["positions"]
        }
        for group in training_groups:
            assert group["stop_kind"] == "leaf"
            assert not group["fallback"]
            assert all(expected_leaf[position] == group["node_id"] for position in group["positions"])
        np.testing.assert_allclose(schema.encode(X, training_groups, family="mlp"), encoded)
        assert fitted.routing_stats(training_groups, len(X))["fallback_count"] == 0
    pd.testing.assert_frame_equal(X, original_X)
    pd.testing.assert_series_equal(y, original_y)


def test_infogain_uses_legal_label_boundary_instead_of_median(schema_factory):
    schema = schema_factory()
    X = pd.DataFrame({"age": range(8), "kind": ["a"] * 8, "aux": range(8)})
    y = pd.Series([0, 0, 1, 1, 1, 1, 1, 1])
    thresholds = {}
    for variant in ("median", "infogain"):
        fitted = FittedMondrian(schema, k=2, variant=variant).fit(X, y)
        thresholds[variant] = next(node["threshold"] for node in fitted.nodes if node["kind"] == "numeric")
    assert thresholds == {"median": 3.5, "infogain": 1.5}


def test_zero_gain_continues_with_median_splits(schema_factory):
    X = pd.DataFrame({"age": range(8), "kind": ["a"] * 8, "aux": range(8)})
    fitted = FittedMondrian(schema_factory(), k=2, variant="infogain").fit(X, pd.Series([0] * 8))
    assert len(fitted.partitions) == 4
    assert fitted.per_record_k(X.index).min() == 2


def test_empty_categorical_branch_uses_containing_ancestor_and_roundtrip(schema_factory):
    schema = schema_factory()
    train = pd.DataFrame({"age": [1] * 4, "kind": ["a"] * 4, "aux": [0] * 4})
    fitted = FittedMondrian(schema, k=2, variant="median").fit(train, pd.Series([0, 1, 0, 1]))
    saved = deepcopy(fitted.to_dict())
    evaluation = pd.DataFrame({"age": [1, 1, 7], "kind": ["b", "c", "a"], "aux": [99] * 3})
    groups = fitted.transform_groups(evaluation)
    by_row = {position: group for group in groups for position in group["positions"]}
    assert by_row[0]["release"]["kind"]["node"] == "ab"
    assert by_row[1]["release"]["kind"]["node"] == "kind_root"
    assert by_row[2]["release"]["age"]["high"] == 7
    assert by_row[2]["release"]["kind"]["node"] == "kind_root"
    assert by_row[2]["node_id"] == 0
    assert by_row[2]["stop_kind"] == "root"
    assert by_row[0]["release"]["age"] == {"low": 1.0, "high": 1.0, "missing": False}
    assert all(group["fallback"] for group in groups)
    assert fitted.to_dict() == saved
    restored = FittedMondrian.from_dict(schema, json.loads(json.dumps(saved)))
    assert restored.transform_groups(evaluation) == groups
    for position, group in by_row.items():
        values = schema.values(evaluation)
        assert all(schema.contains(attribute, domain, values[attribute][[position]])[0]
                   for attribute, domain in group["release"].items())
    stats = fitted.routing_stats(groups, len(evaluation))
    assert (stats["leaf_count"], stats["internal_count"], stats["root_count"]) == (0, 2, 1)
    assert stats["eval_coarsest_release_row_count"] == 1
    assert stats["stop_depth_counts"]["0"] == 1
    assert sum(stats["eval_assignment_counts"].values()) == len(evaluation)


@pytest.mark.parametrize("variant", ["median", "infogain"])
def test_numeric_gap_stops_at_observed_parent_instead_of_threshold_child(schema_factory, variant):
    schema = schema_factory(list(range(30, 41)))
    train = pd.DataFrame({"age": [30, 34, 36, 40], "kind": ["a"] * 4, "aux": [0] * 4})
    fitted = FittedMondrian(schema, k=2, variant=variant).fit(train, pd.Series([0, 1, 0, 1]))
    original_release = deepcopy(fitted.partitions)
    original_k = fitted.per_record_k(train.index).copy()
    evaluation = pd.DataFrame({"age": [31, 35, 39], "kind": ["a"] * 3, "aux": [99] * 3})
    groups = fitted.transform_groups(evaluation)
    by_row = {position: group for group in groups for position in group["positions"]}
    assert by_row[0]["stop_kind"] == by_row[2]["stop_kind"] == "leaf"
    assert by_row[0]["release"]["age"]["high"] == 34
    assert by_row[2]["release"]["age"]["low"] == 36
    gap = by_row[1]
    assert gap["stop_kind"] == "internal"
    assert fitted.nodes[gap["node_id"]]["kind"] == "numeric"
    assert gap["release"]["age"] == {"low": 30.0, "high": 40.0, "missing": False}
    assert gap["release"]["kind"]["node"] == "a"
    assert schema.encode(evaluation, groups, family="mlp").loc[1, "age"] == 35.0
    leaf_width = len(schema.leaves["age"])
    xgboost_input = schema.encode(evaluation, groups, family="xgboost")
    np.testing.assert_allclose(xgboost_input[1, :leaf_width], np.full(leaf_width, 1.0 / leaf_width))
    assert xgboost_input[1, leaf_width] == 1
    pd.testing.assert_series_equal(fitted.per_record_k(train.index), original_k)
    assert fitted.partitions == original_release


def test_every_qi_must_fit_child_and_entire_record_uses_parent(schema_factory):
    original = schema_factory()
    loader = original.loader
    loader.qi_attributes = ("age", "kind", "aux")
    trees = deepcopy(original.generalization.trees)
    trees["aux"] = {"nodes": [
        {"id": "aux_root", "parent": None, "kind": "root", "height_from_leaf": 1},
        *[{"id": f"aux_{value}", "value": value, "parent": "aux_root", "kind": "leaf", "height_from_leaf": 0}
          for value in range(8)],
    ]}
    schema = HierarchySchema(load_generalization_rules(trees, loader, generalization_level=0))
    train = pd.DataFrame({"age": [0, 1, 6, 7], "kind": ["a"] * 4, "aux": [0, 0, 7, 7]})
    fitted = FittedMondrian(schema, k=2, variant="median").fit(train, pd.Series([0, 1, 0, 1]))
    evaluation = pd.DataFrame({"age": [0], "kind": ["a"], "aux": [7]})
    groups = fitted.transform_groups(evaluation)
    assert len(groups) == 1 and groups[0]["stop_kind"] == "internal"
    assert fitted.nodes[groups[0]["node_id"]]["kind"] == "numeric"
    assert groups[0]["release"]["age"]["high"] == 7
    assert groups[0]["release"]["aux"]["low"] == 0
    encoded = schema.encode(evaluation, groups, family="mlp")
    assert encoded.loc[0, "age"] == encoded.loc[0, "aux"] == 3.5


def test_unsplit_training_leaf_remains_distinct_from_global_root(schema_factory):
    schema = schema_factory()
    train = pd.DataFrame({"age": [1, 1, 3, 3], "kind": ["a", "a", "c", "c"], "aux": [0] * 4})
    fitted = FittedMondrian(schema, k=4, variant="median").fit(train, pd.Series([0, 1, 0, 1]))
    assert len(fitted.partitions) == 1
    assert fitted.partitions[0]["node_id"] != 0
    assert fitted.partitions[0]["release"]["age"]["low"] == 1
    assert fitted.partitions[0]["release"]["age"]["high"] == 3
    assert fitted.routing_stats(fitted.transform_groups(train), len(train))["leaf_count"] == 4
    evaluation = pd.DataFrame({"age": [2, 7], "kind": ["c", "a"], "aux": [0, 0]})
    groups = fitted.transform_groups(evaluation)
    by_row = {position: group for group in groups for position in group["positions"]}
    assert by_row[0]["stop_kind"] == "leaf"
    assert by_row[1]["stop_kind"] == "root"
    assert by_row[1]["release"] == schema.root_domain()
    saved = fitted.to_dict()
    assert saved["schema_version"] == "hierarchy_mondrian.v2"
    saved["schema_version"] = "hierarchy_mondrian.v1"
    with pytest.raises(ValueError, match="Unsupported"):
        FittedMondrian.from_dict(schema, saved)


def test_root_fallback_retains_numeric_root_height_in_a_unary_hierarchy(schema_factory):
    original = schema_factory()
    trees = deepcopy(original.generalization.trees)
    trees["age"]["nodes"][0].update({"parent": "age_super_root", "kind": "internal"})
    trees["age"]["nodes"].append({
        "id": "age_super_root", "parent": None, "kind": "root", "height_from_leaf": 2,
    })
    generalization = load_generalization_rules(trees, original.loader, generalization_level=0)
    schema = HierarchySchema(generalization)
    train = pd.DataFrame({"age": [1] * 4, "kind": ["a"] * 4, "aux": [0] * 4})
    fitted = FittedMondrian(schema, k=2, variant="median").fit(train, pd.Series([0, 1, 0, 1]))
    evaluation = pd.DataFrame({"age": [7], "kind": ["a"], "aux": [99]})
    groups = fitted.transform_groups(evaluation)
    assert groups[0]["stop_kind"] == "root"
    np.testing.assert_allclose(
        schema.encode(evaluation, groups, family="xgboost"),
        generalization.change_level({"age": 2, "kind": 2}).encode_xgboost_leaf_space(evaluation).toarray(),
    )


def test_routing_review_flag_is_strictly_above_one_percent(schema_factory):
    train = pd.DataFrame({"age": [1] * 4, "kind": ["a"] * 4, "aux": [0] * 4})
    fitted = FittedMondrian(schema_factory(), k=2, variant="median").fit(train, pd.Series([0, 1, 0, 1]))
    evaluation = pd.DataFrame({"age": [1] * 100, "kind": ["a"] * 99 + ["b"], "aux": [0] * 100})
    stats = fitted.routing_stats(fitted.transform_groups(evaluation), len(evaluation))
    assert stats["fallback_fraction"] == 0.01
    assert not stats["review_needed"]
    evaluation.loc[98, "kind"] = "b"
    stats = fitted.routing_stats(fitted.transform_groups(evaluation), len(evaluation))
    assert stats["fallback_count"] == 2
    assert stats["review_needed"]
    with pytest.raises(AssertionError, match="exactly once"):
        fitted.routing_stats(fitted.transform_groups(evaluation), len(evaluation) - 1)


def test_categorical_split_requires_every_nonempty_child_to_meet_k(schema_factory):
    X = pd.DataFrame({"age": [1] * 5, "kind": ["a"] * 3 + ["c"] * 2, "aux": range(5)})
    fitted = FittedMondrian(schema_factory(), k=3, variant="infogain").fit(X, pd.Series([0, 1, 0, 1, 0]))
    assert len(fitted.partitions) == 1
    assert fitted.partitions[0]["release"]["kind"]["node"] == "kind_root"


def test_missing_categorical_rows_are_published_without_suppression(schema_factory):
    X = pd.DataFrame({"age": [1] * 6, "kind": ["a", "a", "c", "c", pd.NA, pd.NA], "aux": range(6)})
    schema = schema_factory()
    fitted = FittedMondrian(schema, k=2, variant="infogain").fit(X, pd.Series([0, 1] * 3))
    assert len(fitted.per_record_k(X.index)) == 6
    assert fitted.per_record_k(X.index).min() == 2
    encoded = schema.encode(X, fitted.partitions, family="mlp")
    assert not encoded.isna().any().any()
    np.testing.assert_array_equal(encoded.loc[[4, 5], ["kind=a", "kind=b", "kind=c"]], np.zeros((2, 3)))


def test_node_encoding_matches_trim_and_retains_non_qis(schema_factory):
    schema = schema_factory()
    X = pd.DataFrame({"age": [0, 7], "kind": ["a", "c"], "aux": [11, 19]})
    groups = [{"positions": [0, 1], "release": schema.root_domain()}]
    trim = schema.generalization.change_level({"age": 2, "kind": 2})
    np.testing.assert_allclose(schema.encode(X, groups, family="mlp"), trim.encode(X))
    np.testing.assert_allclose(schema.encode(X, groups, family="xgboost"), trim.encode_xgboost_leaf_space(X).toarray())
    irregular = schema.generalization.change_level({"age": 0, "kind": 1}).encode_xgboost_leaf_space(X).toarray()
    # Eight numeric leaves + level, then three categorical leaves + level.
    assert irregular[0, 12] == 1
    assert irregular[1, 12] == 2


def test_numeric_scalar_gaps_and_interval_union(schema_factory):
    sparse = schema_factory([0, 1, 4, 7])
    assert sparse.numeric_representation("age", {"low": 0, "high": 7})[0] == 3.0
    bins = schema_factory(["0 ~ 1", "2 ~ 7"])
    mean, leaves, level = bins.numeric_representation("age", {"low": 0, "high": 7})
    assert (mean, leaves, level) == (3.5, [0, 1], 1)
    mean, leaves, level = bins.numeric_representation("age", {"low": 3, "high": 4})
    assert (mean, leaves, level) == (3.5, [1], 0)


def test_model_leaf_collision_does_not_merge_privacy_classes(schema_factory):
    schema = schema_factory(["0 ~ 3", "4 ~ 7"])
    X = pd.DataFrame({"age": [0, 0, 1, 1], "kind": ["a"] * 4, "aux": [0] * 4})
    fitted = FittedMondrian(schema, k=2, variant="median").fit(X, pd.Series([0, 1, 0, 1]))
    vectors = schema.encode(X, fitted.partitions, family="xgboost")
    np.testing.assert_array_equal(vectors[0], vectors[2])
    assert (fitted.per_record_k(X.index) == 2).all()


def test_tail_risk_includes_removed_individuals_and_handles_all_removed():
    population = pd.Index(range(100))
    retained = pd.Series(np.arange(1, 51), index=range(50), dtype=float)
    stats = individual_tail_risk_stats(population, retained)
    assert stats["tail_risk_population_size"] == 100
    assert stats["tail_risk_released_population_size"] == 50
    assert stats["tail_risk_population"] == "training_split"
    assert stats["tail_risk_p99"] == pytest.approx(-np.log(2))
    assert individual_tail_risk_stats(population, pd.Series(dtype=float))["tail_risk_p99"] == -np.inf
    with pytest.raises(ValueError, match="belong"):
        individual_tail_risk_stats(population, pd.Series([2], index=[100]))


def test_diabetes_deduplicates_before_limit_and_drops_identifiers(tmp_path):
    loader = DiabetesReadmissionDataLoader(csv_path=tmp_path / "diabetic_data.csv")
    rows = []
    for encounter, patient, label in [(30, 10, "NO"), (20, 20, ">30"), (10, 10, "<30")]:
        row = {column: 1 for column in loader.feature_columns}
        row.update({"encounter_id": encounter, "patient_nbr": patient, "readmitted": label})
        rows.append(row)
    pd.DataFrame(rows).to_csv(loader.csv_path, index=False)
    X, y = loader.load(nrows=1)
    # Earliest encounter for patient 10 is source row 2; row 1 remains first
    # in source order after deduplication, and nrows then retains patient 20.
    assert X.index.tolist() == [1]
    assert y.tolist() == [1]
    assert "patient_nbr" not in X and "encounter_id" not in X
    assert loader.preprocessing_metadata["unique_patient_count"] == 2
    X, y = loader.load()
    assert X.index.tolist() == [1, 2]
    assert y.tolist() == [1, 1]


def test_k_grid_keeps_whole_training_endpoint_and_separate_raw_reference():
    assert k_grid(1000) == [2, 4, 8, 12, 16, 32, 64, 128, 235, 256, 1000]
    assert k_grid(1000, explicit=[235, 12, 235]) == [12, 235]
    with pytest.raises(ValueError):
        k_grid(1000, explicit=[1])


def test_pilot_never_scores_test_and_supports_loaders_without_classes(
    schema_factory, tmp_path, monkeypatch,
):
    from experiments.exp1_kanon import run

    schema = schema_factory()
    X = pd.DataFrame({
        "age": list(range(8)) + [2, 3, 100, 100],
        "kind": ["a"] * 12, "aux": range(12),
    })
    # Test-only values and labels deliberately violate the training schema.
    # Any accidental test encoding/scoring during the pilot must fail.
    y = pd.Series([0, 1] * 5 + [9, 9])
    loader = schema.loader
    loader.X_raw, loader.y = X, y
    assert not hasattr(loader, "classes_")
    split = DatasetSplit(X.iloc[:8], X.iloc[8:10], X.iloc[10:], y.iloc[:8], y.iloc[8:10], y.iloc[10:])
    tree_path = tmp_path / "tree.yaml"
    tree_path.write_text("trees: {}\n", encoding="utf-8")
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(yaml.safe_dump({
        "experiment_id": "test", "experiment_kind": "exp1_privacy_utility",
        "base_task": {"dataset": "income", "device": "cpu"},
        "axes": [{"name": "model", "values": [
            {"label": family, "patch": {"model_name": family, "model": {"downstream": {"family": family}}}}
            for family in ("mlp", "xgboost")
        ]}, {"name": "seed", "values": [
            {"label": 42, "patch": {"pipeline": {"random_state": 42}}},
        ]}],
    }), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "experiment_kind": "exp1_kanon", "mode": "pilot",
        "matrix_config": str(matrix_path), "results_root": str(tmp_path / "results"),
        "sweep": {"k_values": [2]},
    }), encoding="utf-8")
    monkeypatch.setattr(run, "implementation_sha256", lambda: "test-code")
    monkeypatch.setattr(run, "build_data_loader", lambda *args, **kwargs: loader)
    monkeypatch.setattr(run, "load_dataset_split", lambda *args, **kwargs: split)
    monkeypatch.setattr(run, "resolve_generalization_tree", lambda *args: tree_path)
    monkeypatch.setattr(run, "load_generalization_rules_from_file", lambda *args, **kwargs: schema.generalization)
    predictions = []

    class FakeModel:
        def set_classes(self, classes):
            np.testing.assert_array_equal(classes, [0, 1])

        def fit(self, features, labels):
            assert len(features) == 8
            pd.testing.assert_series_equal(labels, split.y_train)

        def predict_proba_tensor(self, features):
            assert len(features) == 2
            predictions.append(len(features))
            return torch.full((len(features), 2), 0.5)

    monkeypatch.setattr(run, "build_model_factory", lambda *args, **kwargs: FakeModel)
    assert run.main(["--config", str(config_path)]) == 0
    output_dir = next((tmp_path / "results").iterdir())
    rows = pd.read_csv(output_dir / "raw_results.csv")
    assert set(rows["variant"]) == {"median", "infogain"}
    assert set(rows["model"]) == {"mlp", "xgboost"}
    assert len(rows) == 4
    assert rows["tree_cache_hit"].tolist() == [False, False, True, True]
    assert len(list((output_dir / "shared_releases").glob("*.json.gz"))) == 2
    assert (rows["tail_risk_population_size"] == 8).all()
    assert (rows["tail_risk_population"] == "training_split").all()
    assert rows.groupby("variant")["shared_release_path"].nunique().eq(1).all()
    assert rows["test_loss"].isna().all()
    assert rows["test_delta_u"].isna().all()
    assert (rows["validation_leaf_count"] == 2).all()
    assert (rows["validation_fallback_count"] == 0).all()
    assert rows["test_root_count"].isna().all()
    for release_path in rows["release_path"]:
        diagnostics = json.loads((Path(release_path).parent / "routing_diagnostics.json").read_text(encoding="utf-8"))
        assert diagnostics["test"] is None
        assert diagnostics["validation"]["leaf_count"] == 2
        assert diagnostics["validation"]["eval_coarsest_release_row_count"] == 0
    assert predictions == [2, 2, 2, 2, 2, 2]
    assert not (output_dir / "baseline_long.csv").exists()
    with pytest.raises(ValueError, match="requires --selection"):
        run.main(["--config", str(config_path), "--mode", "full"])


def test_original_privacy_uses_training_raw_qis_not_holdout_or_model_bins(schema_factory):
    training = pd.DataFrame({"age": [0, 0, 1, 1], "kind": ["a"] * 4, "aux": range(4)})
    all_rows = pd.concat([training, training], ignore_index=True)
    schema = schema_factory(["0 ~ 7"])
    sizes = original_equivalence_class_sizes(training, schema.attributes)
    assert sizes.tolist() == [2, 2, 2, 2]
    assert original_equivalence_class_sizes(all_rows, schema.attributes).min() == 4
    # Model level-0 bins intentionally merge numeric values that raw QIs keep separate.
    assert len(schema.generalization.encode(training).drop(columns="aux").drop_duplicates()) == 1
    stats = individual_tail_risk_stats(training.index, sizes)
    assert stats["tail_risk_population_size"] == 4
    assert stats["tail_risk_p99"] == pytest.approx(-np.log(2))


def test_trim_original_and_tail_population_are_the_training_split(schema_factory, tmp_path):
    from prototype.TRIM_prototype_pipeline import run_trim_pipeline
    from prototype.release_artifacts import load_dataset_split

    schema = schema_factory(["0 ~ 3", "4 ~ 7"])
    loader = schema.loader
    X = pd.DataFrame({"age": np.repeat(range(4), 10), "kind": ["a"] * 40, "aux": range(40)})
    y = pd.Series([0, 1] * 20)
    loader.load = lambda nrows=None: (X.copy(), y.copy())
    tree_path = tmp_path / "tree.yaml"
    tree_path.write_text(yaml.safe_dump({"trees": schema.generalization.trees}))
    split = load_dataset_split(loader, {"random_state": 42})
    expected = original_equivalence_class_sizes(split.X_train_raw, loader.qi_attributes)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = run_trim_pipeline(
            loader, tree_path, device="cpu", model_max_iter=2,
            max_iterations=1, initial_sample_size=4, results_dir=tmp_path / "trim",
        )
    finally:
        torch.set_num_threads(previous_threads)
    assert result.original_leak_k == expected.min()
    config = json.loads((Path(result.run_dir) / "config.json").read_text())
    assert config["privacy_population"] == "training_split"
    assert config["privacy_population_size"] == len(split.X_train_raw) == 28
    release = json.loads((Path(result.run_dir) / "trim_release.json").read_text())
    assert release["tail_risk_population"] == "training_split"
    assert release["tail_risk_population_size"] == 28


def test_compressed_shared_tree_roundtrip_and_cache_identity(schema_factory, tmp_path, monkeypatch):
    from experiments.exp1_kanon.artifacts import MondrianReleaseCache, load_release_payload

    schema = schema_factory()
    X = pd.DataFrame({"age": range(8), "kind": ["a"] * 8, "aux": range(8)})
    y = pd.Series([0, 1] * 4)
    identity = {"dataset": "toy", "seed": 42, "variant": "median", "k": 2,
                "training_data_sha256": "train-v1", "tree_sha256": "tree-v1", "implementation_sha256": "code-v1"}
    cache = MondrianReleaseCache(tmp_path / "cache")
    fitted, reference = cache.get_or_fit(schema, X, y, identity=identity)
    assert not reference["cache_hit"]
    assert Path(reference["shared_release_path"]).read_bytes().startswith(b"\x1f\x8b")
    def fail_fit(*args, **kwargs):
        raise AssertionError("A second model must reuse the fitted tree.")
    monkeypatch.setattr(FittedMondrian, "fit", fail_fit)
    reused, second = cache.get_or_fit(schema, X, y, identity=identity)
    assert second["cache_hit"] and second["partition_seconds"] == 0
    assert reused.to_dict() == fitted.to_dict()
    for family in ("mlp", "xgboost"):
        np.testing.assert_allclose(schema.encode(X, reused.partitions, family=family),
                                   schema.encode(X, fitted.partitions, family=family))
    reference_path = tmp_path / "release.json"
    reference_path.write_text(json.dumps(reference))
    assert load_release_payload(reference_path)["training_row_ids"] == X.index.tolist()
    reference["shared_release_sha256"] = "wrong"
    reference_path.write_text(json.dumps(reference))
    with pytest.raises(ValueError, match="checksum"):
        load_release_payload(reference_path)
    for field in ("seed", "training_data_sha256", "tree_sha256", "implementation_sha256"):
        changed = {**identity, field: "changed"}
        with pytest.raises(AssertionError, match="reuse"):
            cache.get_or_fit(schema, X, y, identity=changed)


def test_diagnosis_distinguishes_numeric_gap_from_other_attribute_shrink(schema_factory):
    from experiments.exp1_kanon.diagnose_routing import diagnose_routing

    X = pd.DataFrame({"age": [0, 2, 5, 7], "kind": ["a"] * 4, "aux": range(4)})
    fitted = FittedMondrian(schema_factory(), k=2, variant="median").fit(X, pd.Series([0, 0, 1, 1]))
    evaluation = pd.DataFrame({"age": [1, 3, 6], "kind": ["a"] * 3, "aux": [0] * 3})
    report = diagnose_routing(fitted, X, evaluation)
    assert report["exclusive_fallback_causes"] == {"split_numeric_observed_range_gap": 1}
    assert report["training"]["fallback_count"] == 0
    assert report["counterfactual_fitting_domain_routing"]["leaf_region_exclusion_count"] == 1

    schema = schema_factory()
    schema.loader.qi_attributes = ("age", "kind", "aux")
    trees = deepcopy(schema.generalization.trees)
    trees["aux"] = deepcopy(trees["age"])
    schema = HierarchySchema(load_generalization_rules(trees, schema.loader, generalization_level=0))
    X = pd.DataFrame({"age": range(8), "kind": ["a"] * 8, "aux": range(8)})
    fitted = FittedMondrian(schema, k=2, variant="median").fit(X, pd.Series([0, 1] * 4))
    evaluation = pd.DataFrame({"age": [0], "kind": ["a"], "aux": [7]})
    report = diagnose_routing(fitted, X, evaluation)
    assert report["exclusive_fallback_causes"] == {"non_split_numeric_observed_range_shrink": 1}
    assert report["non_split_numeric_exclusion_counts_nonexclusive"] == {"aux": 1}


def test_diagnosis_identifies_unoccupied_categorical_branch(schema_factory):
    from experiments.exp1_kanon.diagnose_routing import diagnose_routing

    X = pd.DataFrame({"age": [1] * 4, "kind": ["a"] * 4, "aux": [0] * 4})
    fitted = FittedMondrian(schema_factory(), k=2, variant="median").fit(X, pd.Series([0, 1, 0, 1]))
    evaluation = pd.DataFrame({"age": [1], "kind": ["b"], "aux": [0]})
    report = diagnose_routing(fitted, X, evaluation)
    assert report["exclusive_fallback_causes"] == {"unoccupied_categorical_branch": 1}
