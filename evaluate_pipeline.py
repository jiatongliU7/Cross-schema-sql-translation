"""Evaluate schema mappings and translated queries on fixed dataset splits."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from sqltranslate.evaluation import load_splits, read_json, run, write_json
from sqltranslate.fewshot import FewShotMatcher, select_examples


def schema_for_matcher(document):
    from sqltranslate.core import Schema, Table, Column
    primary = set(document.get('primary_keys', []))
    tables = [Table(t['name'], [Column(c['name'], c['type'],
              c.get('pk', t['name'] + '.' + c['name'] in primary)) for c in t['columns']])
              for t in document['tables']]
    if 'fks' in document:
        fks = document['fks']
    else:
        # The dataset's FK format represents single-column edges.
        fks = [[(edge['from'], edge['to'])] for edge in document.get('foreign_keys', [])]
    return Schema(tables, fks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--split', choices=['dev', 'test'], default='dev')
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--methods', nargs='+', choices=['oracle', 'lexical', 'pretrained', 'finetuned', 'finetuned_type', 'finetuned_fk', 'saved', 'llm'], default=['oracle', 'lexical'])
    parser.add_argument('--pretrained', default='all-MiniLM-L6-v2')
    parser.add_argument('--finetuned', help='Local model path or model identifier')
    parser.add_argument('--mapping-dir', type=Path, help='Saved predictions: <pair_id>.json')
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--type-weight', type=float, default=0.15)
    parser.add_argument('--fk-weight', type=float, default=0.3)
    parser.add_argument('--query-timeout', type=float, default=5)
    parser.add_argument('--llm-model')
    parser.add_argument('--shots', type=int, default=3)
    parser.add_argument('--seed', type=int, default=4705)
    parser.add_argument('--llm-logdir', type=Path)
    parser.add_argument('--endpoint', default='https://api.openai.com/v1/chat/completions')
    parser.add_argument('--api-key-env', default='OPENAI_API_KEY')
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--dry-run', action='store_true', help='Write LLM requests without API calls; produces no scores')
    args = parser.parse_args()
    if args.top_k < 1 or args.query_timeout <= 0 or args.max_tokens < 1:
        parser.error('top-k, query-timeout and max-tokens must be positive')
    if not 0 <= args.type_weight <= 1 or not 0 <= args.fk_weight <= 1:
        parser.error('Weights must be between zero and one')
    if len(args.methods) != len(set(args.methods)):
        parser.error('Duplicate methods')
    if any(m.startswith('finetuned') for m in args.methods) and not args.finetuned:
        parser.error('Fine-tuned methods require --finetuned')
    if 'saved' in args.methods and not args.mapping_dir:
        parser.error('saved requires --mapping-dir')
    if 'llm' in args.methods and not args.llm_model:
        parser.error('llm requires --llm-model')
    if 'llm' in args.methods and not args.dry_run and not os.environ.get(args.api_key_env):
        parser.error(f'Set {args.api_key_env} or use --dry-run to preview requests')
    if args.dry_run and args.methods != ['llm']:
        parser.error('--dry-run requires --methods llm')
    if args.outdir.exists():
        parser.error('outdir already exists; choose a new directory')
    methods = []
    matcher = None
    for name in args.methods:
        if name == 'oracle':
            methods.append({'label': name, 'oracle': True})
        elif name == 'saved':
            methods.append({'label': name, 'predictor': lambda pair, a, b: read_json(args.mapping_dir / (pair + '.json'))})
        elif name == 'llm':
            examples = select_examples(args.data_root, args.shots, args.seed)
            matcher = FewShotMatcher(args.data_root, args.llm_model, examples,
                                     args.llm_logdir or args.outdir.with_name(args.outdir.name + '_llm_logs'),
                                     args.endpoint, args.api_key_env, args.max_tokens, args.dry_run)
            methods.append({'label': f'llm_{args.shots}shot', 'predictor': matcher})
        else:
            from sqltranslate.core import predict
            def predictor(pair, source, target, name=name):
                model = args.pretrained if name in ('lexical', 'pretrained') else args.finetuned
                return predict(schema_for_matcher(source), schema_for_matcher(target),
                               method='lexical' if name == 'lexical' else 'sbert', model_name=model,
                               type_weight=args.type_weight if name in ('finetuned_type', 'finetuned_fk') else 0,
                               fk_weight=args.fk_weight if name == 'finetuned_fk' else 0, top_k=args.top_k)
            methods.append({'label': name, 'predictor': predictor})
    if args.dry_run:
        splits = load_splits(args.data_root)
        inputs = []
        for pair in splits[args.split]:
            directory = args.data_root / 'pairs' / pair
            inputs.append((pair, read_json(directory / 'schema_A.json'), read_json(directory / 'schema_B.json')))
        args.outdir.mkdir(parents=True)
        requests = [str(matcher(pair, source, target)) for pair, source, target in inputs]
        write_json(args.outdir / 'dry_run.json', {'requests': requests, 'scores_generated': False})
        print(f'Prepared {len(requests)} requests; no API calls or scores.')
        return
    summaries = run(args.data_root, args.split, methods, args.outdir, args.query_timeout)
    write_json(args.outdir / 'config.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    for label, summary in summaries.items():
        print(f'{label}: EX={summary["execution_accuracy"]:.4f}; queries={summary["n_queries"]}; mapping failures={summary["mapping_failures"]}')


if __name__ == '__main__':
    main()
