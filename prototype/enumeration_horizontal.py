"""Retention-class enumeration used by TRIM."""

from __future__ import annotations

import numpy as np


def _retention_group_columns(generalized_encode, selected_attributes=None):
    if selected_attributes is None:
        return list(generalized_encode.columns)

    group_columns = []
    seen_columns = set()
    missing_attributes = []
    for attribute in selected_attributes:
        if attribute in generalized_encode.columns:
            matching_columns = [attribute]
        else:
            matching_columns = [
                column
                for column in generalized_encode.columns
                if column.startswith(f"{attribute}=")
            ]
            if not matching_columns:
                missing_attributes.append(attribute)
                continue

        for column in matching_columns:
            if column not in seen_columns:
                group_columns.append(column)
                seen_columns.add(column)

    if missing_attributes:
        raise KeyError(
            "Selected attributes are not present in generalized_encode: "
            f"{missing_attributes}"
        )

    return group_columns


def _tuple_group_key(group_key):
    if not isinstance(group_key, tuple):
        return (group_key,)
    return group_key


def select_encoded_attributes(generalized_encode, attributes):
    """Return the encoded columns belonging to the requested raw attributes."""

    columns = _retention_group_columns(
        generalized_encode,
        selected_attributes=attributes,
    )
    return generalized_encode.loc[:, columns]


def build_retention_class_groups(
    generalized_encode,
    selected_attributes=None,
    *,
    sort_keys=True,
):
    """Group rows by generalized values of the selected attributes."""
    group_columns = _retention_group_columns(
        generalized_encode,
        selected_attributes=selected_attributes,
    )

    if not group_columns:
        return {(): generalized_encode.copy()}

    groups = {}
    for group_key, group in generalized_encode.groupby(
        group_columns,
        sort=sort_keys,
        dropna=False,
    ):
        groups[_tuple_group_key(group_key)] = group.copy()

    return groups


build_retention_class_groups.method_name = "qi_level_based"


def build_retention_class_index_groups(
    generalized_encode,
    selected_attributes=None,
    *,
    sort_keys=True,
):
    """Group rows by retention class and return only row indexes."""
    group_columns = _retention_group_columns(
        generalized_encode,
        selected_attributes=selected_attributes,
    )

    if not group_columns:
        return {(): generalized_encode.index.copy()}

    groups = {}
    grouped = generalized_encode.groupby(
        group_columns,
        sort=sort_keys,
        dropna=False,
    )
    for group_key, index in grouped.groups.items():
        groups[_tuple_group_key(group_key)] = index

    return groups


def build_retention_class_position_index(
    generalized_encode,
    selected_attributes=None,
    *,
    sort_keys=True,
):
    """Build a fixed row-position index for retention classes."""
    index_groups = build_retention_class_index_groups(
        generalized_encode,
        selected_attributes=selected_attributes,
        sort_keys=sort_keys,
    )
    row_ids = list(generalized_encode.index)
    row_id_to_position = {
        row_id: position
        for position, row_id in enumerate(row_ids)
    }
    row_class_ids = np.empty(len(row_ids), dtype=np.intp)
    class_keys = []
    class_member_positions = []

    ordered_index_groups = sorted(
        index_groups.items(),
        key=lambda item: row_id_to_position[item[1][0]],
    )

    for class_id, (class_key, class_index) in enumerate(ordered_index_groups):
        positions = np.fromiter(
            (row_id_to_position[row_id] for row_id in class_index),
            dtype=np.intp,
            count=len(class_index),
        )
        class_keys.append(class_key)
        class_member_positions.append(positions)
        row_class_ids[positions] = class_id

    return {
        "row_ids": np.asarray(row_ids, dtype=object),
        "row_id_to_position": row_id_to_position,
        "row_class_ids": row_class_ids,
        "class_keys": class_keys,
        "class_member_positions": class_member_positions,
        "class_total_sizes": np.asarray(
            [len(positions) for positions in class_member_positions],
            dtype=np.intp,
        ),
    }
