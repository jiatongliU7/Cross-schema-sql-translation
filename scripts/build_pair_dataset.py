#!/usr/bin/env python3
"""Build schema-pair data from the Dr.Spider schema perturbation sets.

The output layout is intentionally simple:

    data/
      splits.json
      pairs/<pair_id>/
        db_A.sqlite
        db_B.sqlite
        schema_A.json
        schema_B.json
        gold_mapping.json
        queries.jsonl

Splits are assigned by the original Spider database, never by query or variant.
This prevents renamed versions of the same database from leaking across splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rewrite import rewrite


PERTURBATIONS = {
    "synonym": "DB_schema_synonym",
    "abbreviation": "DB_schema_abbreviation",
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def qualified_column(schema: dict[str, Any], column_index: int) -> str:
    table_index, column_name = schema["column_names_original"][column_index]
    if table_index < 0:
        raise ValueError("The wildcard column cannot be qualified")
    table_name = schema["table_names_original"][table_index]
    return f"{table_name}.{column_name}"


def schema_document(schema: dict[str, Any], *, shuffle: bool, seed: int) -> dict[str, Any]:
    tables: list[dict[str, Any]] = [
        {"name": table_name, "columns": []}
        for table_name in schema["table_names_original"]
    ]

    for column_index, (table_index, column_name) in enumerate(
        schema["column_names_original"]
    ):
        if table_index < 0:
            continue
        tables[table_index]["columns"].append(
            {
                "name": column_name,
                "type": schema["column_types"][column_index],
            }
        )

    primary_keys = [
        qualified_column(schema, column_index)
        for column_index in schema["primary_keys"]
    ]
    foreign_keys = [
        {
            "from": qualified_column(schema, source_index),
            "to": qualified_column(schema, target_index),
        }
        for source_index, target_index in schema["foreign_keys"]
    ]

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(tables)
        for table in tables:
            rng.shuffle(table["columns"])
        rng.shuffle(primary_keys)
        rng.shuffle(foreign_keys)

    return {
        "db_id": schema["db_id"],
        "tables": tables,
        "primary_keys": primary_keys,
        "foreign_keys": foreign_keys,
    }


def derive_base_db_id(post_db_id: str, source_db_ids: Iterable[str]) -> str:
    candidates = [
        source_db_id
        for source_db_id in source_db_ids
        if post_db_id.startswith(f"{source_db_id}_")
        and post_db_id[len(source_db_id) + 1 :].isdigit()
    ]
    if not candidates:
        raise ValueError(f"Cannot derive a source database for {post_db_id!r}")
    return max(candidates, key=len)


def validate_parallel_schemas(source: dict[str, Any], target: dict[str, Any]) -> None:
    source_id = source["db_id"]
    target_id = target["db_id"]
    if len(source["table_names_original"]) != len(target["table_names_original"]):
        raise ValueError(f"Table count changed: {source_id} -> {target_id}")
    if len(source["column_names_original"]) != len(target["column_names_original"]):
        raise ValueError(f"Column count changed: {source_id} -> {target_id}")
    if source["column_types"] != target["column_types"]:
        raise ValueError(f"Column types changed: {source_id} -> {target_id}")
    if source["primary_keys"] != target["primary_keys"]:
        raise ValueError(f"Primary-key positions changed: {source_id} -> {target_id}")
    if source["foreign_keys"] != target["foreign_keys"]:
        raise ValueError(f"Foreign-key positions changed: {source_id} -> {target_id}")

    source_table_positions = [item[0] for item in source["column_names_original"]]
    target_table_positions = [item[0] for item in target["column_names_original"]]
    if source_table_positions != target_table_positions:
        raise ValueError(f"Column-to-table positions changed: {source_id} -> {target_id}")


def gold_mapping(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    validate_parallel_schemas(source, target)

    tables = {
        source_name: target_name
        for source_name, target_name in zip(
            source["table_names_original"], target["table_names_original"]
        )
    }
    columns: dict[str, str] = {}
    for column_index, ((source_table, _), (target_table, _)) in enumerate(
        zip(
            source["column_names_original"],
            target["column_names_original"],
        )
    ):
        if source_table < 0:
            continue
        if source_table != target_table:
            raise ValueError(
                f"Column table index changed at position {column_index}: "
                f"{source['db_id']} -> {target['db_id']}"
            )
        columns[qualified_column(source, column_index)] = qualified_column(
            target, column_index
        )

    return {"tables": tables, "columns": columns}


def collect_referenced_column_ids(value: Any, output: set[int]) -> None:
    """Collect Spider global column IDs from a parsed Spider SQL object."""
    if isinstance(value, dict):
        for child in value.values():
            collect_referenced_column_ids(child, output)
        return
    if not isinstance(value, list):
        return

    # Spider col_unit: [aggregation_id, column_id, is_distinct]
    if (
        len(value) == 3
        and isinstance(value[0], int)
        and not isinstance(value[0], bool)
        and isinstance(value[1], int)
        and not isinstance(value[1], bool)
        and isinstance(value[2], bool)
    ):
        output.add(value[1])
        return

    for child in value:
        collect_referenced_column_ids(child, output)


def collect_referenced_table_ids(value: Any, output: set[int]) -> None:
    if isinstance(value, dict):
        for child in value.values():
            collect_referenced_table_ids(child, output)
        return
    if not isinstance(value, list):
        return
    if len(value) == 2 and value[0] == "table_unit" and isinstance(value[1], int):
        output.add(value[1])
        return
    for child in value:
        collect_referenced_table_ids(child, output)


def sql_has_join(sql: Any) -> bool:
    if isinstance(sql, dict):
        from_clause = sql.get("from")
        if isinstance(from_clause, dict):
            direct_tables = {
                unit[1]
                for unit in from_clause.get("table_units", [])
                if isinstance(unit, list)
                and len(unit) == 2
                and unit[0] == "table_unit"
                and isinstance(unit[1], int)
            }
            if len(direct_tables) > 1:
                return True
        return any(sql_has_join(child) for child in sql.values())
    if isinstance(sql, list):
        return any(sql_has_join(child) for child in sql)
    return False


def rename_exposure_level(
    sql: dict[str, Any], source: dict[str, Any], target: dict[str, Any]
) -> str:
    referenced_columns: set[int] = set()
    referenced_tables: set[int] = set()
    collect_referenced_column_ids(sql, referenced_columns)
    collect_referenced_table_ids(sql, referenced_tables)

    changed_elements = 0
    for table_index in referenced_tables:
        if (
            source["table_names_original"][table_index]
            != target["table_names_original"][table_index]
        ):
            changed_elements += 1
    for column_index in referenced_columns:
        if column_index == 0:
            continue
        if (
            source["column_names_original"][column_index][1]
            != target["column_names_original"][column_index][1]
        ):
            changed_elements += 1

    if changed_elements == 0:
        return "L1"
    if changed_elements == 1:
        return "L2"
    return "L3"


def stable_shuffle_seed(global_seed: int, pair_id: str) -> int:
    digest = hashlib.sha256(f"{global_seed}:{pair_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def materialize_renamed_database(
    source: Path, destination: Path, mapping: dict[str, Any]
) -> None:
    """Copy source data and apply only the gold identifier renames.

    Dr.Spider post-perturbation SQLite files contain a few content differences.
    Building B from A isolates schema naming as the experimental variable.
    Two-phase temporary names avoid collisions during swaps or chained renames.
    """

    if destination.exists():
        raise FileExistsError(destination)
    shutil.copy2(source, destination)
    try:
        with sqlite3.connect(destination) as connection:
            connection.execute("PRAGMA foreign_keys=OFF")
            with connection:
                for table_number, source_table in enumerate(mapping["tables"]):
                    column_changes: list[tuple[str, str]] = []
                    prefix = f"{source_table}."
                    for source_column, target_column in mapping["columns"].items():
                        if not source_column.startswith(prefix):
                            continue
                        old_name = source_column[len(prefix) :]
                        new_name = target_column.rsplit(".", 1)[1]
                        if old_name != new_name:
                            column_changes.append((old_name, new_name))

                    temporary_names: list[tuple[str, str]] = []
                    for column_number, (old_name, new_name) in enumerate(
                        column_changes
                    ):
                        temporary = (
                            f"__cross_schema_tmp_col_{table_number}_{column_number}__"
                        )
                        connection.execute(
                            f"ALTER TABLE {quote_identifier(source_table)} "
                            f"RENAME COLUMN {quote_identifier(old_name)} "
                            f"TO {quote_identifier(temporary)}"
                        )
                        temporary_names.append((temporary, new_name))

                    for temporary, new_name in temporary_names:
                        connection.execute(
                            f"ALTER TABLE {quote_identifier(source_table)} "
                            f"RENAME COLUMN {quote_identifier(temporary)} "
                            f"TO {quote_identifier(new_name)}"
                        )

                table_changes = [
                    (source_table, target_table)
                    for source_table, target_table in mapping["tables"].items()
                    if source_table != target_table
                ]
                temporary_tables: list[tuple[str, str]] = []
                for table_number, (source_table, target_table) in enumerate(
                    table_changes
                ):
                    temporary = f"__cross_schema_tmp_table_{table_number}__"
                    connection.execute(
                        f"ALTER TABLE {quote_identifier(source_table)} "
                        f"RENAME TO {quote_identifier(temporary)}"
                    )
                    temporary_tables.append((temporary, target_table))
                for temporary, target_table in temporary_tables:
                    connection.execute(
                        f"ALTER TABLE {quote_identifier(temporary)} "
                        f"RENAME TO {quote_identifier(target_table)}"
                    )
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def build_split_map(base_db_ids: list[str], seed: int) -> dict[str, list[str]]:
    shuffled = sorted(base_db_ids)
    random.Random(seed).shuffle(shuffled)
    train_end = round(len(shuffled) * 0.70)
    dev_end = train_end + round(len(shuffled) * 0.15)
    return {
        "train": sorted(shuffled[:train_end]),
        "dev": sorted(shuffled[train_end:dev_end]),
        "test": sorted(shuffled[dev_end:]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("work/raw_drspider"),
        help="Directory containing Spider-dev and the two DB_schema_* folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data"),
        help="New dataset directory to create.",
    )
    parser.add_argument("--seed", type=int, default=4705)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {output_root}"
        )

    source_schema_path = raw_root / "Spider-dev" / "tables.json"
    source_schemas = {
        item["db_id"]: item for item in read_json(source_schema_path)
    }
    source_db_ids = sorted(source_schemas)
    split_databases = build_split_map(source_db_ids, args.seed)
    split_for_database = {
        db_id: split
        for split, db_ids in split_databases.items()
        for db_id in db_ids
    }

    pairs_root = output_root / "pairs"
    pairs_root.mkdir(parents=True)
    split_pairs: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    pair_metadata: dict[str, dict[str, Any]] = {}
    link_modes: defaultdict[str, int] = defaultdict(int)

    for perturbation, folder_name in PERTURBATIONS.items():
        perturbation_root = raw_root / folder_name
        target_schemas = read_json(
            perturbation_root / "tables_post_perturbation.json"
        )
        pre_questions = read_json(
            perturbation_root / "questions_pre_perturbation.json"
        )
        post_questions = read_json(
            perturbation_root / "questions_post_perturbation.json"
        )
        if len(pre_questions) != len(post_questions):
            raise ValueError(f"Question count mismatch in {folder_name}")

        paired_questions_by_target: defaultdict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        for index, (pre, post) in enumerate(
            zip(pre_questions, post_questions)
        ):
            if pre.get("q_id_spider_dev") != post.get("q_id_spider_dev"):
                raise ValueError(f"Question ID mismatch in {folder_name} at row {index}")
            if pre.get("question") != post.get("question"):
                raise ValueError(f"Question text mismatch in {folder_name} at row {index}")
            paired_questions_by_target[post["db_id"]].append((pre, post))

        for target_schema in target_schemas:
            target_db_id = target_schema["db_id"]
            base_db_id = derive_base_db_id(target_db_id, source_db_ids)
            source_schema = source_schemas[base_db_id]
            validate_parallel_schemas(source_schema, target_schema)

            variant = target_db_id[len(base_db_id) + 1 :]
            short_name = "syn" if perturbation == "synonym" else "abbr"
            pair_id = f"{base_db_id}__{short_name}_v{variant}"
            pair_dir = pairs_root / pair_id
            pair_dir.mkdir()

            source_db = (
                raw_root
                / "Spider-dev"
                / "databases"
                / base_db_id
                / f"{base_db_id}.sqlite"
            )
            if not source_db.is_file():
                raise FileNotFoundError(source_db)

            link_modes[link_or_copy(source_db, pair_dir / "db_A.sqlite")] += 1

            shuffle_seed = stable_shuffle_seed(args.seed, pair_id)
            write_json(
                pair_dir / "schema_A.json",
                schema_document(source_schema, shuffle=False, seed=shuffle_seed),
            )
            write_json(
                pair_dir / "schema_B.json",
                schema_document(target_schema, shuffle=True, seed=shuffle_seed),
            )
            mapping = gold_mapping(source_schema, target_schema)
            write_json(pair_dir / "gold_mapping.json", mapping)
            materialize_renamed_database(
                source_db, pair_dir / "db_B.sqlite", mapping
            )
            link_modes["generated_renamed_copy"] += 1

            question_pairs = paired_questions_by_target.get(target_db_id, [])
            if not question_pairs:
                raise ValueError(f"No queries found for {target_db_id}")

            level_counts: defaultdict[str, int] = defaultdict(int)
            join_count = 0
            seen_qids: set[str] = set()
            with (pair_dir / "queries.jsonl").open("w", encoding="utf-8") as handle:
                for pre, post in question_pairs:
                    if pre["db_id"] != base_db_id:
                        raise ValueError(
                            f"Query source mismatch for {target_db_id}: {pre['db_id']}"
                        )
                    source_qid = int(pre["q_id_spider_dev"])
                    qid = f"{pair_id}__q{source_qid:04d}"
                    if qid in seen_qids:
                        raise ValueError(f"Duplicate query ID inside {pair_id}: {qid}")
                    seen_qids.add(qid)

                    is_join = sql_has_join(pre["sql"])
                    level = rename_exposure_level(
                        pre["sql"], source_schema, target_schema
                    )
                    join_count += int(is_join)
                    level_counts[level] += 1
                    row = {
                        "qid": qid,
                        "q_A": pre["query"],
                        # The public rewriter is the single source of truth:
                        # gold mapping here, model mapping during evaluation.
                        "gold_q_B": rewrite(pre["query"], mapping),
                        "is_join": is_join,
                        "level": level,
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            split = split_for_database[base_db_id]
            split_pairs[split].append(pair_id)
            pair_metadata[pair_id] = {
                "base_db_id": base_db_id,
                "target_db_id": target_db_id,
                "perturbation": perturbation,
                "split": split,
                "query_count": len(question_pairs),
                "join_query_count": join_count,
                "level_counts": dict(sorted(level_counts.items())),
                "changed_table_count": sum(
                    source_name != target_name
                    for source_name, target_name in mapping["tables"].items()
                ),
                "changed_column_count": sum(
                    source_name != target_name
                    for source_name, target_name in mapping["columns"].items()
                ),
            }

    for pair_ids in split_pairs.values():
        pair_ids.sort()

    split_document = {
        "train": split_pairs["train"],
        "dev": split_pairs["dev"],
        "test": split_pairs["test"],
        "metadata": {
            "seed": args.seed,
            "split_unit": "base_db_id",
            "split_ratio": {"train": 0.70, "dev": 0.15, "test": 0.15},
            "base_databases": split_databases,
            "level_definition": {
                "L1": "query references 0 renamed schema elements",
                "L2": "query references exactly 1 renamed schema element",
                "L3": "query references 2 or more renamed schema elements",
            },
            "schema_B_order": "deterministically shuffled within each pair",
            "gold_q_B_source": "rewrite.rewrite(q_A, gold_mapping)",
            "db_B_source": "copy db_A data, then apply gold identifier renames",
            "sqlite_materialization": dict(sorted(link_modes.items())),
            "pairs": dict(sorted(pair_metadata.items())),
        },
    }
    write_json(output_root / "splits.json", split_document)

    print(f"Created {sum(len(value) for value in split_pairs.values())} pairs")
    for split in ("train", "dev", "test"):
        pair_count = len(split_pairs[split])
        query_count = sum(
            pair_metadata[pair_id]["query_count"] for pair_id in split_pairs[split]
        )
        print(
            f"  {split:5s}: {pair_count:3d} pairs, "
            f"{len(split_databases[split]):2d} base DBs, {query_count:4d} queries"
        )
    print(f"Output: {output_root}")


if __name__ == "__main__":
    main()
