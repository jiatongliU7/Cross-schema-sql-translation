# Gemini 3-shot development results

## Results (query Execution Accuracy)

| Method | Standard dev (925 queries) | Challenge dev (190 queries) |
| --- | ---: | ---: |
| Lexical | 755/925 (81.62%) | 56/190 (29.47%) |
| Pretrained SBERT | 921/925 (99.57%) | 160/190 (84.21%) |
| Fine-tuned SBERT | 921/925 (99.57%) | 152/190 (80.00%) |
| Fine-tuned + types | 921/925 (99.57%) | 152/190 (80.00%) |
| Fine-tuned + types + FK | 921/925 (99.57%) | 184/190 (96.84%) |
| Gemini 3.1 Flash-Lite, 3-shot | 925/925 (100%) | 190/190 (100%) |

SBERT/lexical rows come from upstream commit
`7071cb3da4e17038d8245ee23d077479fc1cc472`:
`experiments/spider-run/summary_dev.csv` and
`experiments/challenge-run-20261122/summary.csv` (dev rows only).
Integer correct-query counts for upstream rows were reconstructed by rounding
reported micro accuracy times query count; standard accuracies were published
to six decimal places. `comparison_dev.csv` contains these counts and ratios.
Gemini rows come from the saved complete local runs copied into this directory.
Standard upstream results use the handoff evaluator and challenge uses the team
batch evaluator; their reference error denominator policies differ, but both have
zero invalid dev references. The comparisons use the same data group/split and
shared SQL rewriting approach. Do not silently combine evaluators if invalid
references are introduced later.

## How Gemini results were obtained

1. Freeze the model, prompt and three training demonstrations in `protocol.json`.
2. For each pair, supply schema A and schema B to Gemini once to predict a complete
   table/column mapping. No evaluated-pair gold mapping or SQL is supplied.
3. Validate completeness, bijection and table/column consistency.
4. Call the shared `rewrite.rewrite(q_A, predicted_mapping)` for each source query.
5. Execute predicted SQL and gold target SQL on the same target SQLite database;
   compare rows with duplicate counts preserved and outer ORDER BY respected.
6. Audit source/reference agreement and count every query occurrence in EX.
   Mapping errors and invalid model outputs remain failures in the denominator.
7. Aggregate per-query results and independently compare predicted mappings to gold.

There are 29 standard dev pairs and 3 challenge dev pairs, all from the same three
base databases: battle_death, car_1, dog_kennels. One mapping is reused across
queries, so the experiment does not make 925 or 190 individual LLM requests.
All 851 standard column assignments and 90 challenge column assignments match
gold. Challenge table mapping is 17/17. The same demonstrations were used in both
groups, and their content hash was checked. There are no empty gold result sets
or invalid references in these development runs. Normalized SQL Exact Match is
also 100% for both Gemini runs.

## Preliminary conclusion

Semantic matching is more robust to schema renaming than lexical similarity.
The standard benchmark nearly saturates for SBERT, while challenge makes errors
more visible. Adding the explicit FK objective improves fine-tuned type-aware
SBERT challenge dev EX from 80.00% to 96.84% (16.84 percentage points). The type
term alone does not alter these results, and fine-tuning alone does not
consistently outperform the pretrained encoder. Encoder inputs already contain
type/key metadata, so this does not isolate a complete removal of structure.

Gemini achieves the highest observed development accuracy on both groups.
These results do not support accuracy superiority of SBERT over the LLM.
A potential efficiency/deployment advantage of local matching remains to be
measured; free API access is not evidence of zero inference resource cost.

The groups reuse only three independent dev databases. Query repetitions and
variants are correlated. Both sets preserve structure/data and only rename
identifiers, so neither establishes general migration performance. Execution on
one database instance does not prove semantic equivalence. This is one fixed
three-example configuration, not an average across prompts or seeds. No Gemini
test-set score is included; upstream test results were previously viewed, so
future test comparisons should be described with that exploratory provenance.

## Files

- `protocol.json`: frozen model/example settings and content hash.
- `comparison_dev.csv`: combined development comparison with upstream provenance.
- `results/standard_dev/`, `results/challenge_dev/`: summaries, per-pair and per-query
  records, checkpoints, configs, API requests/responses and attempt records.
- [Gemini evaluation guide](../../docs/GEMINI_EVALUATION.md): reproduction commands and retry/denominator policy.

Local paths in published records are replaced by placeholders. No API keys,
model weights or SQLite copies are included. Scratch diagnostics and unsuccessful
model-selection smoke runs are not part of the evaluated baseline artifact.
