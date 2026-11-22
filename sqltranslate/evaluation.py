"""Batch evaluation for the schema-pair dataset. Gold is never passed to predictors."""
from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def load_splits(root):
    document = read_json(Path(root) / 'splits.json')
    owners, bases = {}, {}
    metadata = document.get('metadata', {}).get('pairs', {})
    for split in ('train', 'dev', 'test'):
        for pair_id in document[split]:
            if not isinstance(pair_id, str) or Path(pair_id).name != pair_id or pair_id in ('.', '..'):
                raise ValueError(f'Invalid pair ID: {pair_id!r}')
            if pair_id in owners:
                raise ValueError(f'Duplicate pair: {pair_id}')
            owners[pair_id] = split
            base = metadata.get(pair_id, {}).get('base_db_id', pair_id.split('__')[0])
            if base in bases and bases[base] != split:
                raise ValueError(f'Base database leakage: {base}')
            bases[base] = split
    return document


def load_queries(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError(f'Empty query set: {path}')
    ids = set()
    for row in rows:
        if not all(isinstance(row.get(k), str) and row[k].strip() for k in ('qid', 'q_A', 'gold_q_B')):
            raise ValueError(f'Expected qid, q_A, gold_q_B strings in {path}')
        if row['qid'] in ids:
            raise ValueError(f'Duplicate qid: {row["qid"]}')
        ids.add(row['qid'])
    return rows


def validate_mapping(mapping, source, target):
    """Validate complete bijections against actual schema names, not the gold answer."""
    if not isinstance(mapping, dict):
        raise ValueError('Mapping must be a JSON object')
    def names(schema):
        return ({t['name'] for t in schema['tables']},
                {t['name'] + '.' + c['name'] for t in schema['tables'] for c in t['columns']})
    st, sc = names(source)
    tt, tc = names(target)
    for key, left, right in (('tables', st, tt), ('columns', sc, tc)):
        values = mapping.get(key)
        if not isinstance(values, dict) or not all(isinstance(a, str) and isinstance(b, str) for a, b in values.items()):
            raise ValueError(f'{key} must be a string-to-string dictionary')
        if set(values) != left or set(values.values()) != right or len(values) != len(set(values.values())):
            raise ValueError(f'{key} must be a complete one-to-one mapping')
    for a, b in mapping['columns'].items():
        if mapping['tables'][a.rsplit('.', 1)[0]] != b.rsplit('.', 1)[0]:
            raise ValueError(f'Column/table disagreement: {a} -> {b}')
    return mapping


def mapping_counts(prediction, gold):
    result = {}
    for field, prefix in (('tables', 'table'), ('columns', 'column')):
        truth = gold[field]
        guessed = (prediction or {}).get(field, {})
        result[prefix + '_correct'] = sum(guessed.get(k) == v for k, v in truth.items())
        result[prefix + '_total'] = len(truth)
        changed = {k: v for k, v in truth.items() if k != v}
        result['changed_' + prefix + '_correct'] = sum(guessed.get(k) == v for k, v in changed.items())
        result['changed_' + prefix + '_total'] = len(changed)
    return result


def execute_readonly(path, sql, timeout=5.0, max_rows=100000):
    """Bound query runtime and result size; never mutate dataset hard links."""
    path = Path(path).resolve(strict=True)
    start = time.monotonic()
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=timeout) as conn:
        conn.execute('PRAGMA query_only=ON')
        conn.set_progress_handler(lambda: int(time.monotonic() - start > timeout), 1000)
        cursor = conn.execute(sql)
        rows = cursor.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            raise ValueError(f'Result exceeds {max_rows} rows')
        return rows


def normalize_sql(sql):
    import sqlglot
    expressions = sqlglot.parse(sql, read='sqlite')
    if len(expressions) != 1 or expressions[0] is None:
        raise ValueError('Expected exactly one SQL statement')
    return expressions[0].sql(dialect='sqlite', normalize=True, comments=False)


def ordered_sql(sql):
    import sqlglot
    return sqlglot.parse_one(sql, read='sqlite').args.get('order') is not None


def compare_rows(expected, actual, ordered):
    return expected == actual if ordered else Counter(expected) == Counter(actual)


def evaluate_query(row, source_db, target_db, mapping, prediction_error=None, timeout=5.0):
    """Target gold comparison is primary; independent source execution audits gold."""
    result = {k: row.get(k) for k in ('qid', 'q_A', 'gold_q_B', 'level', 'is_join')}
    result.update(predicted_sql=None, execution_success=False, execution_correct=False,
                  normalized_exact_match=False, cross_database_correct=False,
                  reference_valid=False, empty_gold=None, error_stage=None, error=None,
                  reference_error=None, prediction_error=prediction_error)
    try:
        expected = execute_readonly(target_db, row['gold_q_B'], timeout)
        original = execute_readonly(source_db, row['q_A'], timeout)
        ordered = ordered_sql(row['gold_q_B'])
        result.update(empty_gold=not expected, ordered=ordered, gold_rows=len(expected), source_rows=len(original),
                      reference_valid=compare_rows(original, expected, ordered))
        if not result['reference_valid']:
            result['reference_error'] = 'Source and target gold results differ'
    except Exception as exc:
        expected = original = None
        result['reference_error'] = f'{type(exc).__name__}: {exc}'
    if prediction_error:
        result.update(error_stage='mapping', error=prediction_error)
        return result
    try:
        from rewrite import rewrite
        result['predicted_sql'] = rewrite(row['q_A'], mapping)
    except Exception as exc:
        result.update(error_stage='rewrite', error=f'{type(exc).__name__}: {exc}')
        return result
    try:
        actual = execute_readonly(target_db, result['predicted_sql'], timeout)
        result['execution_success'] = True
        result['predicted_rows'] = len(actual)
    except Exception as exc:
        result.update(error_stage='execution', error=f'{type(exc).__name__}: {exc}')
        return result
    if expected is None or not result['reference_valid']:
        result.update(error_stage='reference', error=result['reference_error'])
        return result
    result['execution_correct'] = compare_rows(expected, actual, ordered)
    result['cross_database_correct'] = compare_rows(original, actual, ordered)
    try:
        result['normalized_exact_match'] = normalize_sql(result['predicted_sql']) == normalize_sql(row['gold_q_B'])
    except Exception as exc:
        result['normalization_error'] = str(exc)
    if not result['execution_correct']:
        result.update(error_stage='result_mismatch', error='Execution results differ')
    return result


def rates(rows):
    n = len(rows)
    return {'n_queries': n, **{name: sum(bool(r.get(flag)) for r in rows) / n if n else None
            for name, flag in (('execution_success_rate', 'execution_success'),
                               ('execution_accuracy', 'execution_correct'),
                               ('normalized_exact_match', 'normalized_exact_match'),
                               ('reference_valid_rate', 'reference_valid'))}}


def summarize(pair_rows, query_rows):
    """Micro metrics, pair macro and base-database macro are explicitly separate."""
    report = rates(query_rows)
    for prefix in ('table', 'column', 'changed_table', 'changed_column'):
        correct = sum(r[prefix + '_correct'] for r in pair_rows)
        total = sum(r[prefix + '_total'] for r in pair_rows)
        report[prefix + '_accuracy'] = correct / total if total else None
    report['n_pairs'] = len(pair_rows)
    report['mapping_failures'] = sum(bool(r.get('mapping_error')) for r in pair_rows)
    report['error_counts'] = dict(Counter(r['error_stage'] for r in query_rows if r['error_stage']))
    report['empty_gold_queries'] = sum(r['empty_gold'] is True for r in query_rows)
    for field, label in (('pair_id', 'pair'), ('base_db_id', 'base_database')):
        grouped = defaultdict(list)
        for row in query_rows:
            grouped[row[field]].append(row)
        values = [rates(group)['execution_accuracy'] for group in grouped.values()]
        report[label + '_macro_execution_accuracy'] = sum(values) / len(values) if values else None
    report['groups'] = {}
    for field in ('level', 'is_join', 'perturbation'):
        grouped = defaultdict(list)
        for row in query_rows:
            grouped[str(row.get(field, 'unknown'))].append(row)
        report['groups'][field] = {key: rates(value) for key, value in sorted(grouped.items())}
    return report


def write_csv(path, rows):
    if not rows:
        Path(path).write_text('', encoding='utf-8')
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(root, split, methods, outdir, timeout=5.0):
    """methods: [{label, predictor(pair_id, source_json, target_json)->mapping}]."""
    root, out = Path(root), Path(outdir)
    splits = load_splits(root)
    if not splits[split]:
        raise ValueError(f'No pairs in {split}')
    labels = [m['label'] for m in methods]
    if len(set(labels)) != len(labels):
        raise ValueError('Method labels must be unique')
    # Preflight all dataset files before spending inference budget.
    pairs = []
    for pair_id in splits[split]:
        directory = root / 'pairs' / pair_id
        for name in ('db_A.sqlite', 'db_B.sqlite', 'schema_A.json', 'schema_B.json', 'gold_mapping.json', 'queries.jsonl'):
            if not (directory / name).is_file():
                raise FileNotFoundError(directory / name)
        source, target = read_json(directory / 'schema_A.json'), read_json(directory / 'schema_B.json')
        gold = validate_mapping(read_json(directory / 'gold_mapping.json'), source, target)
        pairs.append((pair_id, directory, source, target, gold, load_queries(directory / 'queries.jsonl')))
    out.mkdir(parents=True, exist_ok=False)
    all_pairs, all_queries, summaries = [], [], {}
    for method in methods:
        pair_results, query_results = [], []
        for pair_id, directory, source, target, gold, queries in pairs:
            started = time.monotonic()
            mapping, error = None, None
            try:
                if method.get('oracle'):
                    mapping = gold
                else:
                    # Only schema documents and an opaque pair key are exposed to predictors.
                    mapping = validate_mapping(method['predictor'](pair_id, source, target), source, target)
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
            meta = splits.get('metadata', {}).get('pairs', {}).get(pair_id, {})
            identity = {'method': method['label'], 'pair_id': pair_id,
                        'base_db_id': meta.get('base_db_id', pair_id.split('__')[0]),
                        'perturbation': meta.get('perturbation', 'unknown')}
            p = {**identity, **mapping_counts(mapping, gold), 'mapping_error': error,
                 'inference_seconds': time.monotonic() - started}
            pair_results.append(p)
            index = len(all_pairs) + len(pair_results)
            write_json(out / f'mapping_{index:04d}.json', {**identity, 'mapping': mapping, 'error': error})
            for row in queries:
                query_results.append({**identity, **evaluate_query(row, directory / 'db_A.sqlite', directory / 'db_B.sqlite', mapping, error, timeout)})
        summaries[method['label']] = summarize(pair_results, query_results)
        all_pairs.extend(pair_results)
        all_queries.extend(query_results)
    write_csv(out / 'pairs.csv', all_pairs)
    write_csv(out / 'queries.csv', all_queries)
    with (out / 'queries.jsonl').open('w', encoding='utf-8') as handle:
        for row in all_queries:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    write_json(out / 'summary.json', summaries)
    write_csv(out / 'summary.csv', [{'method': name, **{k: v for k, v in summary.items() if not isinstance(v, dict)}} for name, summary in summaries.items()])
    fingerprint = hashlib.sha256((root / 'splits.json').read_bytes()).hexdigest()
    write_json(out / 'run.json', {'split': split, 'splits_sha256': fingerprint, 'timeout_seconds': timeout,
                                'methods': labels, 'dataset_root': str(root.resolve()),
                                'reference_policy': 'Invalid references count as failures; inspect reference_valid_rate before reporting.'})
    return summaries
