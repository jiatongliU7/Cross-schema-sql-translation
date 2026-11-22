# Cross-schema SQL translation

COMS 4705 research prototype for **SQLite databases that differ only by table and column renaming**. It predicts one reusable mapping per database pair, rewrites SQL with SQLGlot, and evaluates execution and mapping accuracy. Gold mappings are used only for evaluation, training pair creation, or the explicitly named gold oracle.

## Setup and quick start

Requires Python 3.10+ and SQLite 3.26+ (recent Python installations normally include SQLite). Run from this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m sqltranslate.cli demo --dir demo_output
python -m unittest discover -s tests -v
```

The demo creates a new source DB and renamed target DB, saves the gold and predicted mappings, and evaluates three queries. Choose a new `--dir` for another run: existing databases are never overwritten. `gold_report.json` verifies rewriting with the answer; `predicted_report.json` evaluates the lexical matcher. These tiny examples are a smoke test, not research evidence.

To use Sentence-BERT and training, additionally run `pip install -r requirements-neural.txt`. First model use downloads model files from Hugging Face, unless already cached; internet and enough storage are needed. A GPU is optional. The LLM baseline needs API credentials and incurs provider charges. Neither external dataset nor model weights are bundled.

A reproducible toy mapping is included in [`examples/toy`](examples/toy/README.md). For dataset setup and experiments, see the [Spider and Dr.Spider guide](docs/SPIDER_DRSPIDER_GUIDE.md) and the [neural mapping training guide](docs/NEURAL_MAPPING.md).

## Files and data format

* `sqltranslate/core.py`: schema extraction, similarity, type and FK scoring, assignment and reranking.
* `sqltranslate/data.py`: SQLite renaming and training-pair construction.
* `sqltranslate/sql.py`: SQLGlot rewriting and execution comparison.
* `sqltranslate/cli.py`: commands, training, LLM baseline and evaluation.
* `tests/`: small automated regression tests.

The JSON mapping is `{"tables":{"source_table":"target_table"},"columns":{"source_table.source_column":"target_table.target_column"}}`. Both schemas must have the same number of tables and mapped tables the same number of columns. Identifiers containing literal dots are not supported in this representation. A query file is a JSON array of SQL strings. All paths below may be absolute; output parent directories must already exist.

## Build a variant and inspect it

```bash
python -m sqltranslate.cli schema path/source.sqlite --out source_schema.json
python -m sqltranslate.cli generate path/source.sqlite target.sqlite gold.json --difficulty easy --seed 42
python -m sqltranslate.cli schema target.sqlite --out target_schema.json
```

`generate` copies the source database, renames columns and tables via SQLite `ALTER TABLE`, and writes the gold mapping. It refuses to overwrite the target DB. Existing views, triggers, virtual tables, or unusual SQLite DDL may prevent renaming; inspect the databases and validate generated variants. Difficulty `easy` uses a small curated synonym dictionary, `medium` uses synonyms and abbreviation, and `hard` uses opaque random labels. Unknown easy identifiers receive an `alt_` prefix, which leaks surface similarity. Hard identifiers may be impossible to disambiguate from available information. These are starter transformations, **not validated difficulty levels**: replace/curate them, review semantic validity, randomize presentation, audit shortcuts, and report human-verified statistics before making research claims. Schema structure and data are copied, and generated mappings are one-to-one.

Generate reproducible database-level splits **before** generating variants, and never mix variants of the same database across splits:

```bash
# db_ids.json: ["database_1", "database_2", ...]
python -m sqltranslate.cli split db_ids.json --seed 42 --out splits.json
```

The split is a random approximately 70/15/15 split, not stratified. Check split sizes and stratify manually by table/FK counts where needed. For Spider's original train/dev JSON, extract queries for one database:

```bash
python -m sqltranslate.cli spider-queries path/train_spider.json --db-id database_1 --out queries.json
```

Acquire Spider / Dr.Spider separately according to their terms. The tool does not automatically download or convert Dr.Spider; check perturbation artifacts, database overlap and original split provenance yourself. Do not train on databases reserved for testing. TPC-H requires separately obtained and compatible SQLite data and queries; this code does not automatically convert its dialect or generate its data.

## Matching and SQL evaluation

```bash
# lexical only
python -m sqltranslate.cli match source.sqlite target.sqlite --method lexical --type-weight 0 --fk-weight 0 --out lexical.json
# pretrained Sentence-BERT
python -m sqltranslate.cli match source.sqlite target.sqlite --method sbert --type-weight 0 --fk-weight 0 --out sbert.json
# fine-tuned SBERT, types, and optional FK reranking
python -m sqltranslate.cli match source.sqlite target.sqlite --method sbert --model models/schema-sbert --type-weight 0.15 --fk-weight 0 --out sbert_type.json
python -m sqltranslate.cli match source.sqlite target.sqlite --method sbert --model models/schema-sbert --type-weight 0.15 --fk-weight 0.3 --top-k 3 --out sbert_fk.json
python -m sqltranslate.cli rewrite source.sqlite sbert_fk.json --sql 'SELECT customer_name FROM customers'
python -m sqltranslate.cli evaluate source.sqlite target.sqlite sbert_fk.json --queries queries.json --gold gold.json --out report.json
```

The evaluation report records every query, translated SQL or error, execution success, equivalence, mapping accuracy (if `--gold` is supplied), and FK consistency. Failed/unsupported translations count against both query-level rates. Execution compares duplicate-sensitive unordered rows, or ordered lists when the outer query has `ORDER BY`. Values including floats are compared exactly. `LIMIT` without a deterministic order, tied ordering, coincidental equality on a single instance, and floating-point calculations can give misleading results; inspect these cases and use multiple representative database instances for stronger evidence.

SQLite `PRAGMA table_info` and `foreign_key_list` provide schema metadata. Table and column textual descriptions contain their own names, types and key roles. Cosine similarities compare SBERT embeddings; lexical matching compares normalized names. Hungarian assignment enforces a one-to-one table mapping and a one-to-one column mapping within each mapped table. The table-pair score averages table similarity and the mean best column-pair score; impossible table pairs are excluded. Type compatibility is 1 for identical SQLite affinities, 0.5 for different numeric affinities and 0 otherwise. The final objective combines equally weighted mean table and column semantic scores with average column type compatibility, then mixes in the FK fraction when FK edges exist. A composite FK is counted as a single edge and is correct only if every ordered child/parent pair maps to the same target FK. Without source FKs the FK term is omitted. With FK enabled, greedy improving column swaps and table swaps search for a higher objective. Column swaps are constrained by `--top-k`; table swaps are not. Search is local, not globally optimal, and FK scores are not used during initial assignment. Tune weights and k using development databases only; rerun both type-only and FK variants with identical other settings. The `candidates` field contains semantic-only global top-k column candidates, not necessarily the final assignment.

Rewriting uses SQLGlot qualification, scope analysis and star expansion. It handles ordinary SELECTs, aliases, joins, unqualified columns, basic CTEs, and non-correlated subqueries where SQLGlot can resolve the source schema. It rejects correlated/unresolved columns and NATURAL JOINs. SQL dialect differences, DML, arbitrary SQLite extensions, virtual tables, schema-qualified names, ambiguous columns, and all Spider SQL are not guaranteed. Always inspect query-level failures; do not interpret this prototype as support for all Spider SQL. A query executed on the same database content cannot prove semantic equivalence in general.

## Fine-tuning

Create a **training-only** manifest JSON array such as:

```json
[
  {"source":"train_db_a.sqlite", "target":"train_variant_a.sqlite", "mapping":"train_gold_a.json"}
]
```

```bash
python -m sqltranslate.cli pairs train_manifest.json --out train_pairs.json
python -m sqltranslate.cli train train_pairs.json --out models/schema-sbert --epochs 1 --batch-size 8
```

This starter trainer uses cosine similarity regression on positive pairs and reproducibly sampled negative descriptions. Automatically sampled negatives can be false negatives, including semantically equivalent fields from different databases; inspect and curate training examples and establish a proper validation protocol before reporting results. The script does not perform hyperparameter tuning or measure validation performance. Do not include dev/test mappings. Training and SBERT inference have not been exercised without the optional dependencies and model download.

## Few-shot LLM baseline

Prepare `examples.json` as an array of examples **from training databases only**, each containing `source_schema`, `target_schema`, and `mapping` in the same JSON formats produced by `schema` and `generate`. Then:

```bash
export OPENAI_API_KEY='...'
python -m sqltranslate.cli llm-match source.sqlite target.sqlite --examples examples.json --model YOUR_MODEL --out llm.json
python -m sqltranslate.cli evaluate source.sqlite target.sqlite llm.json --queries queries.json --gold gold.json --out llm_report.json
```

The baseline makes one OpenAI-compatible Chat Completions request, asks for a JSON mapping, checks completeness, one-to-one constraints and table/column consistency, and fails explicitly for invalid output. It does not automatically retry or correct invalid outputs: record those failures. `--endpoint` can select another compatible service; never send private schemas to a service without authorization. Record model/version, prompt, examples, token usage and cost externally. This is a full-schema mapping baseline, evaluated through the same rewriter; direct SQL prompting is not implemented.

## Experimental limitations

The code is a starting research implementation, not finished experimental results. It does not download datasets, prove data splits are independent, automatically perform sweeps or statistical significance testing, validate name difficulty, or evaluate all SQL constructs. FK relationships do not disambiguate graph-symmetric columns/tables by themselves. The FK reranker, type scores and one-to-one assumption are specific to structure-preserving renaming and do not model table splits, merges, missing columns or general database migration. Inspect code and generated data before use in a paper.
