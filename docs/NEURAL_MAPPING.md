# Train and evaluate the schema matcher

This workflow encodes schema descriptions using a pretrained Sentence-BERT model, fine-tunes it on **training-only verified pairs**, and compares retrieval on held-out databases. It does not train a Transformer from scratch.

## 1. Prepare datasets and splits

Obtain Spider and Dr.Spider separately and follow [the dataset guide](SPIDER_DRSPIDER_GUIDE.md). Spider does not supply complete cross-schema mappings: create renamed variants of **training** databases and retain the generated gold mapping. Dr.Spider position-aligned mappings are candidates, not verified labels; review them before use. Do not train or tune on databases reserved for final testing. Keep every variant of a database in one split.

A manifest is a JSON array. Use distinct original database IDs across train, development, and test manifests:

```json
[
  {"db_id": "database_a", "source": "data/a.sqlite", "target": "data/a_renamed.sqlite", "mapping": "data/a_gold.json"}
]
```

Paths are resolved relative to the working directory. Check database IDs before training:

```bash
python -m sqltranslate.cli check-splits --train train_manifest.json --dev dev_manifest.json --test test_manifest.json
```

Create positive table and column pairs from training databases only:

```bash
python -m sqltranslate.cli pairs train_manifest.json --out train_pairs.json
```

Each item has `anchor` and `positive` text. These descriptions contain element names, types, and key roles; do not include original target names or the answer in target descriptions. Inspect generated pairs for correct labels. If using Dr.Spider candidate labels, review each mapping first.

## 2. Install dependencies and train

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-neural.txt
python train_sbert.py --train-pairs train_pairs.json --output models/schema-matcher \
  --epochs 3 --batch-size 16 --learning-rate 0.00002
```

The first run needs access to download `all-MiniLM-L6-v2`, unless it is already cached. `--model` accepts a local model directory. Output is saved only to an empty or new directory. The training script uses `MultipleNegativesRankingLoss` and a no-duplicate batch sampler; identical strings in a batch are avoided, but different names can still have the same meaning and become false negatives. Manually review labels and examples and record training settings. For very small datasets results are not meaningful.

The existing `python -m sqltranslate.cli train` command is a separate older cosine regression baseline; use `train_sbert.py` for the contrastive workflow above.

## 3. Get similarity matrices, top-k candidates, and mapping

```bash
python matcher.py --source <source.sqlite> --target <target.sqlite> \
  --model models/schema-matcher --top-k 3 \
  --out similarity.json --mapping-out mapping.json
```

`similarity.json` contains table/column names, matrices, and globally ranked candidates. `mapping.json` contains the complete one-to-one assignment, suitable for the downstream SQL rewriter. Table pairs must have corresponding column counts. The default explicit type and FK weights are zero; use `--type-weight 0.15 --fk-weight 0.3` to invoke the existing type/FK reranker. The encoder inputs still contain type and key role metadata, even with zero explicit weights. Candidate scores are cosine similarities, not probabilities. Mapping calls the encoder a second time; use the same input descriptions and model in all runs.

## 4. Evaluate on held-out databases

```bash
python -m scripts.evaluate_models --test-manifest test_manifest.json \
  --finetuned models/schema-matcher --top-k 3 --outdir results
```

The output `results/evaluation.csv` has one row per database variant and model; `detail_*.json` contains the rankings. Top-1 accuracy and Top-k recall are independently ranked retrieval metrics over all target columns, not mapping accuracy after one-to-one assignment. Report the search space and distinguish macro (mean across databases) from micro (weighted by number of columns) aggregation. Choose hyperparameters on dev data, never test data. Any ambiguous Dr.Spider correspondence requires review before reporting it as ground truth.

## Verification status

The Python data preparation, ranking logic, and training integration have unit tests. In the current workspace, Sentence-Transformers and the official datasets/model weights are unavailable, so a real neural training run and held-out Spider/Dr.Spider evaluation have **not** been performed. Do not treat example outputs as research results.
