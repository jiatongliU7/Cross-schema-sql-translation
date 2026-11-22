# Toy mapping example

These files are output from the reproducible `demo` command, using hand-created SQLite tables and a generated renaming. They are **not Spider or Dr.Spider data** and the reported metrics are not research results.

- `gold.json`: known mapping recorded during database renaming.
- `predicted.json`: mapping produced by lexical similarity, type scoring, and FK reranking.
- `queries.json`: source SQL queries.
- `gold_report.json` and `predicted_report.json`: results for the three toy queries.

Reproduce locally with `python -m sqltranslate.cli demo --dir demo_output`. The SQLite database files are generated locally and are not committed.
