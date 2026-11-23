# Milestone: Preliminary Evaluation Results

## Task and experimental setup

We study SQL translation between SQLite databases that differ only in table and
column names. The system predicts one reusable schema mapping per database pair
and applies a fixed SQL rewriter. This is schema-renaming translation, not general
schema migration or natural-language-to-SQL generation.

We compare lexical matching, pretrained Sentence-BERT, fine-tuned Sentence-BERT,
type/FK constraints, and a 3-shot Gemini 3.1 Flash-Lite mapping baseline. The LLM
uses the same fixed three training demonstrations across both evaluation groups;
the evaluated pair's gold mapping is not provided as input. Train/dev assignments
are separated by original database. All reported values below are development
results. Model results are from upstream commit
`7071cb3da4e17038d8245ee23d077479fc1cc472`; Gemini results are recorded in
`experiments/gemini-3shot/results/standard_dev/` and
`experiments/gemini-3shot/results/challenge_dev/`. Original local runtime outputs
remain unchanged. See `experiments/gemini-3shot/README.md` for run provenance.

## Preliminary results

Primary metric: query-level Execution Accuracy (EX). Standard and challenge are
reported separately. Standard dev has 29 schema pairs and 925 query instances;
challenge dev has 3 pairs and 190 query instances. Both groups reuse the same
three original dev databases: battle_death, car_1 and dog_kennels.

| Method | Standard dev EX | Challenge dev EX |
| --- | ---: | ---: |
| Lexical | 81.62% | 29.47% |
| Pretrained SBERT | 99.57% | 84.21% |
| Fine-tuned SBERT | 99.57% | 80.00% |
| Fine-tuned SBERT + types | 99.57% | 80.00% |
| Fine-tuned SBERT + types + FK | 99.57% | 96.84% |
| Gemini 3.1 Flash-Lite, 3-shot | 100.00% | 100.00% |

Standard contains synonym/abbreviation column renaming with unchanged table
names. Challenge renames ordinary tables and most columns while preserving
data and structural metadata. On challenge dev, Gemini recovers all 17 table
and 90 column correspondences and translates all 190 query instances correctly.
The 925 standard Gemini query instances also pass execution comparison. All
Gemini mappings are valid, and source/reference execution audits pass.

## Interpretation

Our preliminary results suggest that semantic matching is more robust to schema
renaming than lexical similarity. The challenge set exposes errors that are
largely hidden by the near-saturated standard benchmark. Adding the explicit FK
constraint to the fine-tuned type-aware model improves challenge dev EX from
80.00% to 96.84%, a gain of 16.84 percentage points. The type term alone does not
change EX in these runs. Fine-tuning alone does not improve standard dev EX and
underperforms the pretrained model on challenge dev; these results do not support
a claim that fine-tuning consistently improves downstream translation.

The few-shot LLM achieves the highest observed development accuracy, reaching
100% in both groups. We therefore do not claim that the proposed matcher
outperforms LLMs in accuracy. Instead, the project evaluates the contribution of
semantic representations and structural constraints and investigates whether a
local matcher can provide a useful accuracy/efficiency trade-off. Any efficiency
advantage remains a hypothesis until measured under comparable conditions.

## Limitations

- Only three independent development databases are represented; variants and
  repeated query instances are correlated, not independent samples.
- All tasks preserve schema structure and database content. Results do not
  establish support for table splits/merges, missing fields or dialect conversion.
- Single-instance execution equality can conceal semantic errors. Exact-match
  agreement and gold-mapping checks supplement it but do not prove general SQL
  equivalence; reference SQL is produced with the same rewriter.
- Encoder descriptions already contain type and key-role metadata. Zero explicit
  FK weight does not constitute a complete removal of structural information.
- This is one fixed three-example LLM configuration, not an average across prompts
  or random seeds. It remains uncertain how performance varies across examples.
- Upstream test results have already been inspected, so subsequent test evaluation
  should be described as exploratory; avoid test-informed tuning or calling it
  a completely untouched final evaluation.

## Next steps

1. Keep the current prompt, demonstrations, model and matching weights fixed.
2. Inspect representative SBERT failures and FK corrections on development data.
3. Compare inference time, LLM token usage and local compute requirements under
   documented conditions; distinguish mapping inference from shared SQL execution.
4. Complete a consistently configured test comparison and broaden independent
   schema coverage if feasible. Report all methods on the same group and split.

## Suggested short milestone conclusion

Our preliminary evaluation shows that semantic schema matching substantially
outperforms lexical matching on challenging renaming cases. Explicit foreign-key
constraints improve fine-tuned SBERT's challenge development execution accuracy
from 80.00% to 96.84%, whereas fine-tuning alone does not consistently help. A
3-shot Gemini baseline reaches 100% on both standard and challenge development
sets, indicating that the current benchmark is tractable for a strong LLM. We
therefore frame the remaining work around structural ablations and the
accuracy/efficiency trade-off of a local matcher, rather than claiming accuracy
superiority over LLMs. These findings are preliminary and cover only three
independent development databases.
