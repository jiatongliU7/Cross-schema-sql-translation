# Evaluation workflow

Run commands from the repository root using Python 3.10+ with `requirements.txt`
installed. Neural baselines additionally need `requirements-neural.txt` and the
model weights. Use the complete existing `data/` dataset (including SQLite files).

## 1. Check the pipeline on development data

```bash
python evaluate_pipeline.py --data-root data --split dev \
  --methods oracle lexical --outdir outputs/eval_dev_baselines
```

The oracle passes gold mappings through the same public `rewrite.rewrite()` used
by data generation. Other predictors receive only schema JSON documents and a
pair ID. Evaluation reads gold labels only after prediction. JSON presentation
order, including shuffled target columns, is preserved for model input.

The selected dataset has unchanged table names. Table accuracy is a pipeline
check, not evidence of learning table renaming. Unchanged columns can inflate
overall column accuracy: report **changed_column_accuracy** beside overall
column accuracy. A changed element means `source_identifier != gold_target`.

## 2. Compare neural methods or evaluate your teammate's predictions

```bash
python evaluate_pipeline.py --split dev \
  --methods oracle lexical pretrained finetuned finetuned_type finetuned_fk \
  --finetuned models/schema-matcher --outdir outputs/eval_dev_models
```

Pretrained and fine-tuned baselines use zero explicit type/FK weights. Type-only
and FK variants use the same encoder and type weight; the FK variant adds the
FK weight. Defaults are 0.15 / 0.3, with top-k 3. Tune them only on dev.
Encoder descriptions still contain type/key metadata even with explicit weights
zero; this is not an ablation of metadata in encoder inputs.

Alternatively, the model teammate can provide one complete mapping per pair:

```text
predictions/sbert/<pair_id>.json
```

Each file must contain `tables` and `columns` string dictionaries with complete
one-to-one mappings and table/column consistency. Extra score fields are allowed.

```bash
python evaluate_pipeline.py --split dev --methods oracle saved \
  --mapping-dir predictions/sbert --outdir outputs/eval_dev_saved
```

Use a separate output directory per model checkpoint when using `saved`.
Missing or invalid predictions count as mapping failures and cause every query
for that pair to fail; they are never silently removed from the denominator.
Missing dataset files, malformed gold mappings and empty query sets abort before
inference. This prevents an incomplete dataset from becoming a partial run.

## 3. Prepare and inspect few-shot LLM requests

```bash
python evaluate_pipeline.py --split dev --methods llm --llm-model YOUR_MODEL \
  --shots 3 --seed 4705 --dry-run --outdir outputs/llm_dev_preview
```

This makes **no API calls** and generates no scores. The log directory contains
exact request bodies. Demonstrations are complete schema-pair/mapping examples,
selected reproducibly from train, at most one per original database. Zero-shot
is supported with `--shots 0`. All split assignments are checked for original
DB overlap, including synonym/abbreviation variants.

After inspecting requests, set the API key in the environment and run:

```bash
python evaluate_pipeline.py --split dev --methods llm --llm-model YOUR_MODEL \
  --shots 3 --seed 4705 --outdir outputs/eval_dev_llm
```

The default endpoint is Chat Completions compatible. Override `--endpoint` and
`--api-key-env` for another compatible service. The provider must support the
request parameters (temperature, max_tokens and JSON response format). No API
call is made until this command is explicitly run with a key. Default generation
limit is 4096 tokens; set `--max-tokens` appropriately on dev. Non-stop finish
reasons, invalid JSON and invalid mappings are failures. There are no automatic
retries, repair prompts or selection of the best of multiple answers.

Each request log contains request, raw response when received, response model,
usage, elapsed time, selected example IDs and validation status. Secrets are not
stored. Successful responses are reused by SHA-256 of endpoint + exact request,
so rerunning with a new evaluation outdir and the same `--llm-logdir` avoids
repeat charges. Failed/dry-run logs are not automatically retried: use a new
logdir deliberately. Report any deliberate retries. Compute monetary cost from
recorded token usage and the provider's applicable pricing; no price is assumed.

Freeze model ID, shot count, examples, prompt, and generation parameters on dev,
then run test once with the frozen settings and `--split test`.

## Output and metric definitions

| Output | Contents |
| --- | --- |
| `summary.csv` | Overall micro metrics plus pair/base-database macro Execution Accuracy |
| `summary.json` | Same metrics, error counts and groups by level, join and perturbation |
| `pairs.csv` | Correct/total mapping counts and inference time for every pair/method |
| `queries.csv`, `queries.jsonl` | Source SQL, gold target SQL, predicted SQL, flags and errors |
| `mapping_*.json` | Prediction or failure per pair/method, identified inside the file |
| `config.json`, `run.json` | CLI configuration, split fingerprint and denominator policy |

- **Execution Success Rate:** predicted target query executes / all test queries.
- **Execution Accuracy (EX):** predicted query and gold target query return equal
  results on DB B, and source/gold execution passes the reference audit / all queries.
- **Normalized Exact Match:** SQLGlot parses and reserializes both target SQLs
  with identifier normalization and no comments, then compares strings. This is
  deliberately conservative: it does not remove aliases or reorder predicates,
  and is not proof of semantic equivalence.
- **Mapping accuracy:** correct assignments / all gold elements, micro aggregated.
  Invalid/missing mappings get zero correct. Changed-only metrics are null when
  no element was renamed, rather than reporting an artificial 100%.
- **Pair macro EX:** arithmetic mean of per-pair EX.
- **Base-database macro EX:** group all variants/queries by original database,
  calculate each base DB's EX, then take the arithmetic mean. With only three
  held-out original databases, avoid broad generalization claims.

Unordered results use duplicate-sensitive multisets; outer ORDER BY uses ordered
rows. Column position and duplicate counts matter; floats compare exactly. Read-
only SQLite queries have a default 5-second progress timeout and 100,000-row cap.
Failures stay in denominators. Dataset/reference errors are separately flagged;
inspect `reference_valid_rate` and resolve these errors before publishing scores.
Empty gold results are flagged because accidental equality can inflate EX.
LIMIT without deterministic ordering, ordering ties, volatile functions and exact
float comparison remain limitations. One database instance does not establish
semantic equivalence.

`gold_q_B` is generated by the same rewriter, so Exact Match and oracle success
are consistency checks, not independent proof that the SQL rewriting is correct.
Source-vs-target execution is also recorded as a reference audit. Inspect
representative joins, nesting, aggregation and failure cases manually.

The dataset includes duplicate query text under distinct qids. Official row-level
scores retain every qid; disclose this weighting. Before final reporting, agree
whether to add a separate deduplicated analysis; never silently deduplicate only
one method. L1/L2/L3 are counts of distinct renamed elements referenced by a query,
not validated general SQL difficulty levels.

## Tests

```bash
python -m unittest discover -s tests -v
```

Evaluation tests cover duplicate/order semantics, read-only execution and timeout,
reference mismatch, wrong executable mappings, failure denominators, split
leakage, changed-only metrics and mocked LLM logging/cache/validation. Live LLM
quality, neural accuracy and research results require actual dataset/model runs.

## Verified development run (2026-11-22)

Upstream source/data snapshot: `e46d4667fe001ac68cf20593eaa3e2f52accc764`.
Python 3.12; NumPy 1.26.4; SciPy 1.13.1; SQLGlot 30.0.0.
25 unit/regression tests passed. Actual dev evaluation used all 29 dev pairs
(three original databases), 925 query rows, with no reference audit failures.

| Method | Execution Success | EX | Column Accuracy | Changed Column Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Gold oracle | 100% | 100% | 100% | 100% |
| Lexical (zero type/FK weights) | 100% | 81.62% | 96.24% | 86.21% |

Lexical made 170 result mismatches. These dev scores are preliminary baselines,
not final test results. A 3-shot dry run prepared 29 requests using three distinct
training databases; the model name is a placeholder and no API call was made.
No pretrained/fine-tuned SBERT or live LLM evaluation was performed.
Only development SQLite data and the three selected training demonstration
schema/mapping documents were downloaded into this local source snapshot.
The complete test data and remaining training data must be supplied before
running those splits or changing demonstrations.
