"""Evaluate one or more pretrained/fine-tuned models on held-out database pairs."""
import argparse
import csv
import json
from pathlib import Path
from sqltranslate.core import extract_schema, load_json
from sqltranslate.benchmarks import retrieval_metrics


def evaluate(manifest, models, outdir, k):
    entries = load_json(manifest)
    if not isinstance(entries, list) or not entries:
        raise ValueError('Test manifest must be a nonempty JSON array')
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    result = []
    for item in entries:
        for key in ('db_id', 'source', 'target', 'mapping'):
            if not item.get(key): raise ValueError(f'Manifest requires {key}')
        src, tgt, gold = extract_schema(item['source']), extract_schema(item['target']), load_json(item['mapping'])
        for label, name in models:
            report = retrieval_metrics(src, tgt, gold, 'sbert', name, k)
            result.append({'db_id': item['db_id'], 'model_label': label, 'model': name,
                           'n_columns': report['n_columns'], 'top1_accuracy': report['top1_accuracy'],
                           'topk_recall': report['topk_recall'], 'k': k})
            # Indexed names avoid overwriting multiple variants of one database or unsafe IDs.
            (out / f'detail_{len(result):04d}.json').write_text(json.dumps({'db_id': item['db_id'], 'model_label': label, **report}, indent=2) + '\n')
    with (out / 'evaluation.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0]))
        writer.writeheader(); writer.writerows(result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--test-manifest', required=True)
    p.add_argument('--pretrained', default='all-MiniLM-L6-v2')
    p.add_argument('--finetuned', required=True)
    p.add_argument('--top-k', type=int, default=3)
    p.add_argument('--outdir', required=True)
    a = p.parse_args()
    rows = evaluate(a.test_manifest, [('pretrained', a.pretrained), ('finetuned', a.finetuned)], a.outdir, a.top_k)
    print(json.dumps({'output': str(Path(a.outdir) / 'evaluation.csv'), 'rows': len(rows)}, indent=2))


if __name__ == '__main__':
    main()
