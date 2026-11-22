"""Export SBERT similarity matrices, ranked candidates, and a schema mapping."""
import argparse
import json
from pathlib import Path
import numpy as np
from sqltranslate.core import extract_schema, semantic_matrices, predict


def build_report(source, target, table_scores, column_scores, top_k=3):
    if top_k < 1:
        raise ValueError('top-k must be positive')
    tables_a = [t.name for t in source.tables]
    tables_b = [t.name for t in target.tables]
    cols_a = [f'{t.name}.{c.name}' for t in source.tables for c in t.columns]
    cols_b = [f'{t.name}.{c.name}' for t in target.tables for c in t.columns]
    def section(a, b, scores):
        if scores.shape != (len(a), len(b)):
            raise ValueError('Similarity matrix shape does not match schema')
        return {'source': a, 'target': b, 'similarity_matrix': scores.tolist(),
                'top_k': {name: [{'target': b[j], 'score': float(scores[i, j])}
                                 for j in np.argsort(-scores[i], kind='stable')[:top_k]]
                          for i, name in enumerate(a)}}
    return {'tables': section(tables_a, tables_b, table_scores),
            'columns': section(cols_a, cols_b, column_scores)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, help='Source SQLite database')
    p.add_argument('--target', required=True, help='Target SQLite database')
    p.add_argument('--model', default='all-MiniLM-L6-v2')
    p.add_argument('--top-k', type=int, default=3)
    p.add_argument('--out', required=True)
    p.add_argument('--mapping-out', help='Optional complete one-to-one mapping JSON')
    p.add_argument('--type-weight', type=float, default=0)
    p.add_argument('--fk-weight', type=float, default=0)
    args = p.parse_args()
    source, target = extract_schema(args.source), extract_schema(args.target)
    ts, cs = semantic_matrices(source, target, method='sbert', model_name=args.model)
    report = build_report(source, target, ts, cs, args.top_k)
    report['model'] = args.model
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
    if args.mapping_out:
        # Uses the same model and existing assignment/reranking implementation.
        result = predict(source, target, 'sbert', args.model, args.type_weight, args.fk_weight, args.top_k)
        Path(args.mapping_out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'similarity_report': args.out, 'mapping': args.mapping_out}, indent=2))


if __name__ == '__main__':
    main()
