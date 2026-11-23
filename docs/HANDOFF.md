# Shared handoff format

This contract supports the toy example and the recorded Spider-derived schema pairs. Database names and labels in the toy are artificial. The toy databases are created locally with `tests/fixtures/make_toy.py`; they are not Spider data.

```text
data/
  splits.json
  pairs/<pair_id>/
    metadata.json
    db_A.sqlite
    db_B.sqlite
    schema_A.json
    schema_B.json
    gold_mapping.json
    queries.jsonl
predictions/<method>/<pair_id>.json
```

`splits.json` is `{"train": [...], "dev": [...], "test": [...]}`. A toy pair can provide `metadata.json` with `db_id`, the original database identifier. For committed Spider-derived pairs this file is optional: the handoff loader falls back to `schema_A.json`'s `db_id`. The team batch/Gemini loader instead uses `splits.json.metadata.pairs[*].base_db_id` (or the pair-ID prefix). The loader rejects duplicate pair IDs and a db_id occurring across splits. Verify IDs/provenance manually to detect the same data under different identifiers.

Schemas accept either the committed Spider-style `tables` plus `primary_keys`/`foreign_keys` representation or the output of `python -m sqltranslate.cli schema DATABASE`, with `tables` (name, columns with name/type/pk) and `fks` (each edge contains ordered child/parent column pairs). Sequence order is not a correspondence; shuffle schema_B's tables and columns, without changing the named FK edges. Gold mappings use `{ "tables": {"source": "target"}, "columns": {"source_table.column": "target_table.column"} }`. Dr.Spider positional mappings are **candidates**, not automatically verified ground truth.

Each line of `queries.jsonl` contains `qid`, `q_A`, `gold_q_B`, `is_join` (boolean), and `level` (string). IDs must be unique within a pair. Level labels must be defined and validated for the study; the toy levels merely exercise the interface.

Handoff-format prediction JSON contains `tables` and `columns` mapping source identifiers to ranked `[target, score]` lists; `selected_mapping` contains one chosen mapping in the gold format; `q_B` maps query IDs to rewritten SQL. Optional `rewrite_errors` records exceptions. Candidate scores are similarities rather than probabilities. With the existing implementation, FK reranking is a local search over internal score matrices; `--top-k` bounds column swap candidates but does not constrain table swaps. Thus the final selected mapping need not be limited to the globally displayed top-k lists.

## Prediction formats are distinct

| Consumer | Prediction fields |
| --- | --- |
| `python -m handoff.evaluate` | Ranked candidate lists, `selected_mapping`, and generated `q_B` SQL |
| `evaluate_pipeline.py --methods saved` | Top-level string-to-string `tables` and `columns` mappings |
| Gemini batch | API cache plus checkpoint `mapping`; SQL is rewritten during evaluation |

Do not pass ranked candidate objects to the saved-mapping evaluator. The challenge
experiment provides the compatible simplified format in
`experiments/challenge-run-20261122/predicted_mapping/<method>/<pair_id>.json`.
The richer `predictions/` directory is a different representation. See
[Gemini reproduction](GEMINI_EVALUATION.md) for the LLM workflow and
[dataset groups](DATASET.md) for standard/challenge split counts.

## Run the handoff smoke test

```bash
python -m tests.fixtures.make_toy --data /tmp/handoff-data --pred /tmp/handoff-predictions/mock
python -m handoff.evaluate --data /tmp/handoff-data --pred /tmp/handoff-predictions/mock --outdir /tmp/handoff-results/mock
python -m handoff.predict --data /tmp/handoff-data --pred /tmp/handoff-predictions/lexical --method lexical
python -m handoff.evaluate --data /tmp/handoff-data --pred /tmp/handoff-predictions/lexical --outdir /tmp/handoff-results/lexical
```

For pretrained SBERT use `--method sbert`. For the fine-tuned model add `--model models/schema-matcher`; for explicit type/FK reranking add `--type-weight 0.15 --fk-weight 0.3`. Model downloads may be required. Current model descriptions include type and primary/foreign-key roles, including when explicit reranking weights are zero. A true no-metadata ablation requires different descriptions.

```bash
python -m handoff.predict --data /tmp/handoff-data --pred /tmp/handoff-predictions/pretrained_sbert --method sbert
python -m handoff.evaluate --data /tmp/handoff-data --pred /tmp/handoff-predictions/pretrained_sbert --outdir /tmp/handoff-results/pretrained_sbert
```

The evaluator writes `results.csv`, `error_analysis.json`, and `query_results.json`. The first CSV row is an overall method summary, followed by query-level `level` and `is_join` breakdown rows. Mapping accuracy and column Top-1/Top-3 are aggregated across schema elements, so they are reported in the overall row rather than incorrectly attributing a whole-schema score to individual queries. Run the evaluator once per method; combine the overall rows for the final comparison table.

Gold and predicted SQL are executed on **the same target DB**; duplicate counts are preserved. Outer ORDER BY results are compared in order, and other results as bags. Predicted SQL that is missing or fails counts as incorrect among queries whose gold SQL executes. Gold execution errors are reported separately and excluded from query metric denominators, since correctness cannot be assessed with an invalid reference; include their counts when reporting results. Exact equality can be misleading for floating point, LIMIT, tied orderings, and coincidental equality (particularly empty results). Single-instance execution is not a semantic proof. Inspect `error_analysis.json` and validate queries or use multiple target instances.

## Current boundaries

The toy smoke tests do not establish model quality or validate the full dataset. Recorded dataset builders, audits and training experiments are separate workflows; see the dataset and experiment guides. Their existence is not proof of semantic correspondence or broad schema diversity. `schema_A.json`/`schema_B.json` must reflect the SQLite schemas; check consistency before experiments. Database-dependent query errors and unsupported rewrites are recorded, not silently removed. The toy tests establish interface behavior, not model quality.
