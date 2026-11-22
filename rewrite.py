"""Public SQL rewriting function shared by data generation and evaluation.

The mapping itself contains the complete source schema vocabulary, so callers
do not need to pass a separate database path merely to rewrite a query.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sqltranslate.core import Column, Schema, Table
from sqltranslate.sql import rewrite_sql


def _schema_from_mapping(mapping: dict[str, Any]) -> Schema:
    if not isinstance(mapping, dict):
        raise TypeError("mapping must be a dictionary")
    table_mapping = mapping.get("tables")
    column_mapping = mapping.get("columns")
    if not isinstance(table_mapping, dict) or not isinstance(column_mapping, dict):
        raise ValueError("mapping must contain dictionary fields: tables and columns")

    columns_by_table: dict[str, list[str]] = {
        str(table_name): [] for table_name in table_mapping
    }
    for source, target in column_mapping.items():
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError("column mapping keys and values must be strings")
        if source.count(".") != 1 or target.count(".") != 1:
            raise ValueError(
                "column mapping identifiers must use table.column notation"
            )
        source_table, source_column = source.rsplit(".", 1)
        target_table, _ = target.rsplit(".", 1)
        if source_table not in columns_by_table:
            raise ValueError(f"column mapping uses unknown source table: {source}")
        if table_mapping[source_table] != target_table:
            raise ValueError(
                f"column target table disagrees with table mapping: {source} -> {target}"
            )
        if source_column in columns_by_table[source_table]:
            raise ValueError(f"duplicate source column in mapping: {source}")
        columns_by_table[source_table].append(source_column)

    if len(set(table_mapping.values())) != len(table_mapping):
        raise ValueError("table mapping must be one-to-one")
    if len(set(column_mapping.values())) != len(column_mapping):
        raise ValueError("column mapping must be one-to-one")

    # SQLGlot needs names for qualification. Exact SQLite types are unnecessary
    # for identifier rewriting, so a neutral TEXT placeholder is sufficient.
    tables = [
        Table(
            table_name,
            [Column(column_name, "TEXT") for column_name in column_names],
        )
        for table_name, column_names in columns_by_table.items()
    ]
    return Schema(tables=tables, fks=[])


def rewrite(q_A: str, mapping: dict[str, Any]) -> str:
    """Rewrite one source SQL query with a complete schema mapping.

    Data generation calls this with the gold mapping to create ``gold_q_B``.
    Evaluation calls the same function with a model-predicted mapping to create
    the predicted target query.
    """

    if not isinstance(q_A, str) or not q_A.strip():
        raise ValueError("q_A must be a non-empty SQL string")
    source_schema = _schema_from_mapping(mapping)
    return rewrite_sql(q_A, source_schema, mapping)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--sql", required=True)
    args = parser.parse_args()
    with args.mapping.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    print(rewrite(args.sql, mapping))


if __name__ == "__main__":
    main()
