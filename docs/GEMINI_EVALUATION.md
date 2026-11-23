# Gemini evaluation: reproduce the development runs

The native Gemini adapter is `sqltranslate/gemini.py`; the batch entry point is
`run_gemini_batch.py`. Use Python 3.10+ with `requirements.txt` installed. No
Sentence-Transformers installation is required for Gemini-only evaluation.

## Authentication

Set `GEMINI_API_KEY` in the shell that runs the command. The key is read from the
environment; request logs omit authentication headers. Do not put keys in code,
config files, commits, or URLs. A free-tier project was used for these runs.
Available models and quotas can change; model listing alone does not guarantee
generation access. The recorded successful model is `gemini-3.1-flash-lite`.

## Fixed protocol and separated groups

`experiments/gemini-3shot/protocol.json` freezes the model, 3 demonstration IDs,
seed 4705, maximum output 8192 tokens, and the hash of full demonstration contents.
The demonstrations are from three distinct training databases, validated against
the split file. They remain identical for standard and challenge. No evaluation
pair gold mapping, SQL reference, or database contents is included in its prompt.
The model receives source/target schema JSON, with target order retained as given.
Temperature is 0; JSON response mode is enabled. Other generation settings are
provider defaults. Exact request bodies and returned model versions are recorded.

With the complete current upstream dataset at `data/`, run from repository root:

```bash
python run_gemini_batch.py --group standard --data-root data --dry-run
python run_gemini_batch.py --group challenge --data-root data --dry-run
python run_gemini_batch.py --group standard --data-root data
python run_gemini_batch.py --group challenge --data-root data
```

`--dry-run` validates all required inputs and frozen demonstrations without API
calls or output creation. Only development data is evaluated. Groups are selected
using declared challenge IDs and conventional `__challenge_` names; they are never
mixed. The completed standard run predates the appended challenge pairs. The
challenge run used an isolated snapshot from upstream commit
`7071cb3da4e17038d8245ee23d077479fc1cc472`. Reproducing on the merged dataset changes
the splits-file fingerprint, but keeps the selected group and demonstrations.

Outputs default to distinct directories `outputs/gemini_standard_dev_3shot/` and
`outputs/gemini_challenge_dev_3shot/`. Existing published results are in
`experiments/gemini-3shot/results/`; these are records of prior runs, not writable
resume directories. The local legacy `outputs/gemini_dev_3shot/` is preserved.

## Failure handling and resume

API request starts are separated by at least 15 seconds. Transient HTTP
429/500/502/503/504 and connection failures permit at most three attempts per
pair per command invocation, with 20/40-second backoff and numeric Retry-After.
Detected daily/zero quotas or Retry-After above 120 seconds pause immediately.
Persistent transient failures pause the run. Rerun the same command to resume;
all attempts are retained, so deliberate repeated invocations are auditable.

Invalid/truncated model output or incomplete/non-bijective mappings count as
failures. They are never repaired or regenerated to select a better answer.
Authentication/configuration failures pause instead of being scored as matching
errors. Successful exact-request caches and per-pair checkpoints are reused.
Interrupted in-flight requests may consume quota even if no response was saved.
Only one process should use an output directory at a time.

`progress.json` and `partial_summary.json` identify incomplete runs. Full
`summary.json` and `summary.csv` are created only after every planned pair has a
record. All failed predictions remain in metric denominators. Config, dataset,
prompt, evaluator and rewriter hashes reject incompatible resumes. Use a new
outdir when these change.

## Metrics and checks

EX compares predicted and reference SQL on the same target database. Outer
ORDER BY uses ordered rows; other results use duplicate-sensitive multisets.
Source/reference results are independently audited. The same public rewriter
is used for all mappings. Mapping metrics compare predictions to gold directly;
changed-only accuracy excludes unchanged identifiers. Normalized SQL Exact Match
is a supplementary syntactic metric, not proof of semantic equivalence.

See `experiments/gemini-3shot/README.md` for results and limitations. The standard
run has one successful smoke response imported by exact request hash, 28 further
successful responses and one transient 503 attempt before success. Challenge
has three successful requests without retries. There was no output-based
resampling or manual correction. API attempts and imported successful responses
are included. Files have been copied for publication with local home paths
replaced by placeholders; original local runtime records remain unchanged.

## Validation

```bash
python -m unittest discover -s tests -v
```

34 tests passed in the preparation environment. The additions test native request
conversion, mapping validation, bounded retries, quota pauses, no retry of invalid
output, exact-request cache reuse, resume denominators, group selection and frozen
example contents. Both recorded dev runs have complete checkpoints and validated
reference execution. No Gemini test-set results are claimed.
