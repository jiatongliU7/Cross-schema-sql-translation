"""Shared data contract for the schema matching pipeline."""
import json
from pathlib import Path
from sqltranslate.core import Schema


def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def read_queries(path):
    result = []
    seen = set()
    with Path(path).open(encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip(): continue
            row = json.loads(line)
            for field in ('qid','q_A','gold_q_B','is_join','level'):
                if field not in row: raise ValueError(f'{path}:{line_no} missing {field}')
            if not isinstance(row['qid'], str) or not row['qid'] or not isinstance(row['is_join'],bool) or not isinstance(row['level'],str) or not all(isinstance(row[k],str) and row[k].strip() for k in ('q_A','gold_q_B')):
                raise ValueError(f'{path}:{line_no} invalid query fields')
            if row['qid'] in seen: raise ValueError(f'Duplicate qid {row["qid"]}')
            seen.add(row['qid']); result.append(row)
    return result


def parse_schema(obj):
    if 'fks' in obj:
        return Schema.from_dict(obj)
    # Spider-style schema metadata stores keys separately from columns.
    from sqltranslate.core import Table, Column
    pk = set(obj.get('primary_keys', []))
    tables = [Table(t['name'], [Column(c['name'], c.get('type', ''),
        f"{t['name']}.{c['name']}" in pk) for c in t['columns']]) for t in obj['tables']]
    edges = [[(fk['from'], fk['to'])] for fk in obj.get('foreign_keys', [])]
    return Schema(tables, edges)


def load_pair(root,pair_id):
    if not isinstance(pair_id,str) or not pair_id or Path(pair_id).name!=pair_id or pair_id in ('.','..'): raise ValueError(f'Invalid pair_id: {pair_id}')
    d=Path(root)/'pairs'/pair_id
    if not d.is_dir(): raise FileNotFoundError(d)
    raw_a=read_json(d/'schema_A.json'); a=parse_schema(raw_a); b=parse_schema(read_json(d/'schema_B.json'))
    gold=read_json(d/'gold_mapping.json'); queries=read_queries(d/'queries.jsonl')
    metadata=read_json(d/'metadata.json') if (d/'metadata.json').exists() else {'db_id':raw_a.get('db_id')}
    if not isinstance(metadata.get('db_id'),str) or not metadata['db_id']: raise ValueError(f'{d}: metadata.json requires db_id')
    for name in ('db_A.sqlite','db_B.sqlite'):
        if not (d/name).is_file(): raise FileNotFoundError(d/name)
    return d,a,b,gold,queries,metadata


def load_splits(root):
    splits=read_json(Path(root)/'splits.json')
    if not {'train','dev','test'}.issubset(splits) or any(not isinstance(splits[k],list) for k in ('train','dev','test')): raise ValueError('splits.json must have train/dev/test arrays')
    seen={}; db_splits={}
    for split in ('train','dev','test'):
        for pair_id in splits[split]:
            if pair_id in seen: raise ValueError(f'Duplicate pair_id across splits: {pair_id}')
            seen[pair_id]=split
            _,_,_,_,_,meta=load_pair(root,pair_id)
            dbid=meta['db_id']
            if dbid in db_splits and db_splits[dbid]!=split: raise ValueError(f'Database {dbid} leaks between splits')
            db_splits[dbid]=split
    return splits
