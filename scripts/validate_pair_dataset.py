#!/usr/bin/env python3
"""Validate the generated schema-pair dataset, optionally executing all queries."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPECTED_FILES = {
    "db_A.sqlite",
    "db_B.sqlite",
    "schema_A.json",
    "schema_B.json",
    "gold_mapping.json",
    "queries.jsonl",
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def schema_names(schema: dict[str, Any]) -> tuple[set[str], set[str]]:
    tables: set[str] = set()
    columns: set[str] = set()
    for table in schema["tables"]:
        table_name = table["name"]
        if table_name in tables:
            raise ValueError(f"Duplicate table in schema: {table_name}")
        tables.add(table_name)
        for column in table["columns"]:
            qualified = f"{table_name}.{column['name']}"
            if qualified in columns:
                raise ValueError(f"Duplicate column in schema: {qualified}")
            columns.add(qualified)
    return tables, columns


def sqlite_integrity(path: Path) -> None:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        result = connection.execute("PRAGMA integrity_check").fetchone()
    if result != ("ok",):
        raise ValueError(f"SQLite integrity check failed for {path}: {result}")


def sqlite_schema_names(
    path: Path, expected_internal_tables: set[str]
) -> tuple[set[str], set[str]]:
    tables: set[str] = set()
    columns: set[str] = set()
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        table_rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        for (table_name,) in table_rows:
            if table_name.startswith("sqlite_") and table_name not in expected_internal_tables:
                continue
            tables.add(table_name)
            escaped = table_name.replace('"', '""')
            for row in connection.execute(f'PRAGMA table_info("{escaped}")'):
                columns.add(f"{table_name}.{row[1]}")
    return tables, columns


def canonical_rows(rows: list[tuple[Any, ...]]) -> list[tuple[str, ...]]:
    return sorted(tuple(repr(value) for value in row) for row in rows)


def execute_query(path: Path, query: str) -> list[tuple[str, ...]]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        rows = connection.execute(query).fetchall()
    return canonical_rows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute every q_A/gold_q_B pair and compare result multisets.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/dataset_validation.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    split_document = read_json(data_root / "splits.json")

    pair_to_split: dict[str, str] = {}
    for split in ("train", "dev", "test"):
        for pair_id in split_document[split]:
            if pair_id in pair_to_split:
                raise ValueError(f"Pair appears in multiple splits: {pair_id}")
            pair_to_split[pair_id] = split

    actual_pairs = {
        path.name for path in (data_root / "pairs").iterdir() if path.is_dir()
    }
    if set(pair_to_split) != actual_pairs:
        raise ValueError(
            f"Split/pair mismatch; missing={set(pair_to_split) - actual_pairs}, "
            f"extra={actual_pairs - set(pair_to_split)}"
        )

    base_to_split: dict[str, str] = {}
    all_qids: set[str] = set()
    counts: Counter[str] = Counter()
    split_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    execution_failures: list[dict[str, str]] = []
    equivalence_failures: list[dict[str, str]] = []

    checked_db_inodes: set[tuple[int, int]] = set()
    for pair_id in sorted(actual_pairs):
        pair_dir = data_root / "pairs" / pair_id
        present = {path.name for path in pair_dir.iterdir() if path.is_file()}
        if present != EXPECTED_FILES:
            raise ValueError(
                f"Unexpected files for {pair_id}; "
                f"missing={EXPECTED_FILES - present}, extra={present - EXPECTED_FILES}"
            )

        split = pair_to_split[pair_id]
        metadata = split_document["metadata"]["pairs"][pair_id]
        base_db_id = metadata["base_db_id"]
        previous_split = base_to_split.setdefault(base_db_id, split)
        if previous_split != split:
            raise ValueError(f"Database leakage across splits: {base_db_id}")

        schema_a = read_json(pair_dir / "schema_A.json")
        schema_b = read_json(pair_dir / "schema_B.json")
        mapping = read_json(pair_dir / "gold_mapping.json")
        tables_a, columns_a = schema_names(schema_a)
        tables_b, columns_b = schema_names(schema_b)

        if set(mapping) != {"tables", "columns"}:
            raise ValueError(f"Unexpected mapping keys for {pair_id}")
        if set(mapping["tables"]) != tables_a:
            raise ValueError(f"Source table mapping is incomplete for {pair_id}")
        if set(mapping["tables"].values()) != tables_b:
            raise ValueError(f"Target table mapping is incomplete for {pair_id}")
        if set(mapping["columns"]) != columns_a:
            raise ValueError(f"Source column mapping is incomplete for {pair_id}")
        if set(mapping["columns"].values()) != columns_b:
            raise ValueError(f"Target column mapping is incomplete for {pair_id}")
        if len(set(mapping["tables"].values())) != len(mapping["tables"]):
            raise ValueError(f"Table mapping is not one-to-one for {pair_id}")
        if len(set(mapping["columns"].values())) != len(mapping["columns"]):
            raise ValueError(f"Column mapping is not one-to-one for {pair_id}")

        db_a = pair_dir / "db_A.sqlite"
        db_b = pair_dir / "db_B.sqlite"
        sqlite_tables_a, sqlite_columns_a = sqlite_schema_names(db_a, tables_a)
        sqlite_tables_b, sqlite_columns_b = sqlite_schema_names(db_b, tables_b)
        if sqlite_tables_a != tables_a or sqlite_columns_a != columns_a:
            raise ValueError(f"schema_A.json disagrees with db_A.sqlite for {pair_id}")
        if sqlite_tables_b != tables_b or sqlite_columns_b != columns_b:
            raise ValueError(f"schema_B.json disagrees with db_B.sqlite for {pair_id}")
        for database in (db_a, db_b):
            stat = database.stat()
            inode = (stat.st_dev, stat.st_ino)
            if inode not in checked_db_inodes:
                sqlite_integrity(database)
                checked_db_inodes.add(inode)

        pair_qid_count = 0
        with (pair_dir / "queries.jsonl").open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                if set(row) != {"qid", "q_A", "gold_q_B", "is_join", "level"}:
                    raise ValueError(
                        f"Unexpected query fields in {pair_id}:{line_number}"
                    )
                if row["qid"] in all_qids:
                    raise ValueError(f"Duplicate global qid: {row['qid']}")
                all_qids.add(row["qid"])
                if not isinstance(row["is_join"], bool):
                    raise ValueError(f"is_join is not boolean: {row['qid']}")
                if row["level"] not in {"L1", "L2", "L3"}:
                    raise ValueError(f"Invalid level: {row['qid']}")
                if not row["q_A"].strip() or not row["gold_q_B"].strip():
                    raise ValueError(f"Empty SQL query: {row['qid']}")

                pair_qid_count += 1
                counts["queries"] += 1
                counts[f"level_{row['level']}"] += 1
                counts["join_queries"] += int(row["is_join"])
                split_counts[split]["queries"] += 1
                split_counts[split][f"level_{row['level']}"] += 1
                split_counts[split]["join_queries"] += int(row["is_join"])

                if args.execute:
                    try:
                        rows_a = execute_query(db_a, row["q_A"])
                        rows_b = execute_query(db_b, row["gold_q_B"])
                    except Exception as error:  # report every upstream query defect
                        execution_failures.append(
                            {"pair_id": pair_id, "qid": row["qid"], "error": str(error)}
                        )
                        continue
                    if rows_a != rows_b:
                        equivalence_failures.append(
                            {
                                "pair_id": pair_id,
                                "qid": row["qid"],
                                "rows_A": str(len(rows_a)),
                                "rows_B": str(len(rows_b)),
                            }
                        )

        if pair_qid_count != metadata["query_count"]:
            raise ValueError(f"Query count disagrees with metadata for {pair_id}")
        counts["pairs"] += 1
        split_counts[split]["pairs"] += 1

    report = {
        "status": (
            "passed"
            if not execution_failures and not equivalence_failures
            else "failed"
        ),
        "execute_queries": args.execute,
        "counts": dict(sorted(counts.items())),
        "split_counts": {
            split: dict(sorted(counter.items()))
            for split, counter in sorted(split_counts.items())
        },
        "base_database_count": len(base_to_split),
        "unique_sqlite_files_checked": len(checked_db_inodes),
        "execution_failure_count": len(execution_failures),
        "execution_failures": execution_failures[:100],
        "equivalence_failure_count": len(equivalence_failures),
        "equivalence_failures": equivalence_failures[:100],
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
