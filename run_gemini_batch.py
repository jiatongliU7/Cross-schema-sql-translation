"""Resumable, rate-limited Gemini evaluation on the development split."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from rewrite import rewrite  # Fail before inference when dependencies are missing.
from sqltranslate.evaluation import (load_splits, load_queries, read_json, validate_mapping,
    evaluate_query, mapping_counts, summarize, write_csv)
from sqltranslate.gemini import GeminiMatcher, TemporaryAPIError, APIConfigurationError, atomic_json


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def group_pair_ids(splits, split, group):
    metadata = splits.get('metadata', {})
    declared = metadata.get('dataset_groups', {}).get('challenge', {}).get('pair_ids', [])
    challenge = set(declared)
    # Older data has no group declaration; conventional IDs retain the same meaning.
    challenge.update(p for name in ('train', 'dev', 'test') for p in splits[name]
                     if '__challenge_' in p)
    return [p for p in splits[split] if (p in challenge) == (group == 'challenge')]


def frozen_examples(root, splits, protocol):
    ids = protocol['example_pair_ids']
    if len(ids) != protocol['shots'] or len(set(ids)) != len(ids):
        raise ValueError('Frozen example count/IDs are invalid')
    train = set(splits['train'])
    metadata = splits.get('metadata', {}).get('pairs', {})
    seen, examples = set(), []
    for pair in ids:
        if pair not in train:
            raise ValueError(f'Frozen example is not in train: {pair}')
        base = metadata.get(pair, {}).get('base_db_id', pair.split('__')[0])
        if base in seen:
            raise ValueError('Frozen examples must use distinct original databases')
        seen.add(base)
        directory = Path(root) / 'pairs' / pair
        source = read_json(directory / 'schema_A.json')
        target = read_json(directory / 'schema_B.json')
        mapping = validate_mapping(read_json(directory / 'gold_mapping.json'), source, target)
        examples.append({'pair_id': pair, 'source_schema': source, 'target_schema': target,
                         'mapping': {key: mapping[key] for key in ('tables', 'columns')}})
    digest = hashlib.sha256(json.dumps(examples, sort_keys=True).encode()).hexdigest()
    if digest != protocol['examples_sha256']:
        raise ValueError('Frozen demonstration contents have changed')
    return examples


def export_results(out, planned_pairs):
    records = [read_json(out / 'checkpoints' / (pair + '.json')) for pair in planned_pairs
               if (out / 'checkpoints' / (pair + '.json')).exists()]
    pairs = [r['pair'] for r in records]
    queries = [q for r in records for q in r['queries']]
    complete = len(records) == len(planned_pairs)
    progress = {'complete': complete, 'completed_pairs': len(records), 'planned_pairs': len(planned_pairs)}
    atomic_json(out / 'progress.json', progress)
    write_csv(out / 'pairs.csv', pairs)
    write_csv(out / 'queries.csv', queries)
    # Partial scores are explicitly labelled and never exported as final scores.
    summary = {**progress, 'metrics_on_completed_pairs': summarize(pairs, queries)} if records else progress
    atomic_json(out / 'partial_summary.json', summary)
    if complete:
        full = summarize(pairs, queries)
        atomic_json(out / 'summary.json', {'method': 'gemini_3shot', 'complete': True, **full})
        write_csv(out / 'summary.csv', [{k: v for k, v in full.items() if not isinstance(v, dict)}])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parent / 'data')
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--group', choices=['standard', 'challenge'], default='standard')
    parser.add_argument('--examples-config', type=Path, default=Path(__file__).resolve().parent / 'experiments/gemini-3shot/protocol.json')
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs and frozen examples without calling the API')
    parser.add_argument('--model', default='gemini-3.1-flash-lite')
    parser.add_argument('--interval', type=float, default=15, help='Minimum seconds between API request starts')
    parser.add_argument('--max-attempts', type=int, default=3, help='Maximum transport attempts per invocation/pair')
    args = parser.parse_args()
    if args.interval < 0 or args.max_attempts < 1:
        parser.error('Invalid interval or max-attempts')
    if not args.dry_run and not os.environ.get('GEMINI_API_KEY'):
        parser.error('Set GEMINI_API_KEY in this Terminal')
    root = args.data_root.resolve()
    out = (args.outdir or (Path(__file__).resolve().parent / 'outputs' / ('gemini_' + args.group + '_dev_3shot'))).resolve()
    splits = load_splits(root)
    ids = group_pair_ids(splits, 'dev', args.group)
    if not ids:
        parser.error('No development pairs')
    protocol = read_json(args.examples_config)
    if args.model != protocol['model']:
        parser.error('Model differs from frozen protocol; use a separate explicit protocol for a new experiment')
    examples = frozen_examples(root, splits, protocol)
    prepared, hashes = {}, {}
    for pair_id in ids:
        directory = root / 'pairs' / pair_id
        files = ('schema_A.json', 'schema_B.json', 'gold_mapping.json', 'queries.jsonl', 'db_A.sqlite', 'db_B.sqlite')
        hashes[pair_id] = {name: fingerprint(directory / name) for name in files}
        source, target = read_json(directory / 'schema_A.json'), read_json(directory / 'schema_B.json')
        gold = validate_mapping(read_json(directory / 'gold_mapping.json'), source, target)
        prepared[pair_id] = (directory, source, target, gold, load_queries(directory / 'queries.jsonl'))
    config = {'group': args.group, 'protocol_sha256': fingerprint(args.examples_config), 'split': 'dev', 'model': args.model, 'shots': 3, 'seed': 4705, 'max_output_tokens': 8192,
              'interval': args.interval, 'max_attempts_per_invocation': args.max_attempts,
              'example_pair_ids': [e['pair_id'] for e in examples],
              'examples_sha256': hashlib.sha256(json.dumps(examples, sort_keys=True).encode()).hexdigest(),
              'splits_sha256': fingerprint(root / 'splits.json'), 'dataset_sha256': hashes,
              'adapter_sha256': fingerprint(Path(__file__).parent / 'sqltranslate' / 'gemini.py'),
              'prompt_sha256': fingerprint(Path(__file__).parent / 'sqltranslate' / 'fewshot.py'),
              'evaluator_sha256': fingerprint(Path(__file__).parent / 'sqltranslate' / 'evaluation.py'),
              'rewriter_sha256': fingerprint(Path(__file__).parent / 'rewrite.py'),
              'sql_rewriter_sha256': fingerprint(Path(__file__).parent / 'sqltranslate' / 'sql.py')}
    if args.dry_run:
        print(f'Ready: {args.group} dev | {len(ids)} pairs | {sum(len(p[4]) for p in prepared.values())} queries')
        print('Frozen examples:', ', '.join(protocol['example_pair_ids']))
        print('No API calls or evaluation output written.')
        return 0
    if (out / 'config.json').exists():
        if read_json(out / 'config.json') != config:
            parser.error('Existing run configuration/data/code differs; choose a new outdir')
    else:
        if out.exists() and any(out.iterdir()):
            parser.error('Nonempty outdir has no config; choose another outdir')
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(out / 'config.json', config)
    (out / 'checkpoints').mkdir(exist_ok=True)
    logdir = out / 'api_logs'
    logdir.mkdir(exist_ok=True)
    # Import only successful exact-request cache records; old errors are not results.
    for path in sorted((Path(__file__).parent / 'outputs').glob('gemini_smoke_*/api_logs/*.json')):
        if '.attempt_' in path.name:
            continue
        record = read_json(path)
        if record.get('status') == 'success' and record.get('model') == args.model:
            digest = hashlib.sha256(record['endpoint'].encode() + json.dumps(record['request'], ensure_ascii=False).encode()).hexdigest()
            if record.get('request_sha256') != digest:
                continue
            destination = logdir / (digest + '.json')
            if not destination.exists():
                record['imported_from'] = str(path)
                atomic_json(destination, record)
    matcher = GeminiMatcher(root, args.model, examples, logdir, max_attempts=args.max_attempts,
                             min_interval=args.interval, notify=lambda msg: print('  ' + msg, flush=True))
    progress = export_results(out, ids)
    print(f'Model: {args.model} | {args.group} dev pairs: {len(ids)} | resumed: {progress["completed_pairs"]}', flush=True)
    try:
        for index, pair_id in enumerate(ids, 1):
            checkpoint = out / 'checkpoints' / (pair_id + '.json')
            if checkpoint.exists():
                print(f'[{index}/{len(ids)}] {pair_id}: 已完成，跳过', flush=True)
                continue
            directory, source, target, gold, rows = prepared[pair_id]
            print(f'[{index}/{len(ids)}] {pair_id}', flush=True)
            mapping, error = None, None
            try:
                mapping = matcher(pair_id, source, target)
            except (TemporaryAPIError, APIConfigurationError) as exc:
                print('暂停:', exc, flush=True)
                print('已保存完成结果。稍后运行相同命令继续；本次未生成完整 dev 分数。', flush=True)
                export_results(out, ids)
                return 2
            except ValueError as exc:
                # Invalid/truncated model responses count as failures. Never repair or resample.
                error = str(exc)
                print('  模型输出无效，记录为失败，不重抽答案', flush=True)
            meta = splits.get('metadata', {}).get('pairs', {}).get(pair_id, {})
            identity = {'method': 'gemini_3shot', 'pair_id': pair_id,
                        'base_db_id': meta.get('base_db_id', pair_id.split('__')[0]),
                        'perturbation': meta.get('perturbation', 'unknown')}
            pair = {**identity, **mapping_counts(mapping, gold), 'mapping_error': error}
            results = [{**identity, **evaluate_query(row, directory / 'db_A.sqlite', directory / 'db_B.sqlite', mapping, error)} for row in rows]
            atomic_json(checkpoint, {'pair': pair, 'mapping': mapping, 'queries': results})
            export_results(out, ids)
            print(f'  EX: {sum(r["execution_correct"] for r in results)}/{len(results)}', flush=True)
    except KeyboardInterrupt:
        export_results(out, ids)
        print('\n已保存进度；运行相同命令可继续。', flush=True)
        return 130
    report = read_json(out / 'summary.json')
    print(f'完成全部 dev：{report["n_queries"]} queries | EX={report["execution_accuracy"]:.2%}', flush=True)
    print('结果目录:', out, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
