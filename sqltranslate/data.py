"""Create reproducible renamed database variants and extract training pairs."""
import random, re, sqlite3
from pathlib import Path
from .core import extract_schema, quote, save_json, texts

# Names are intentionally curated; unknown identifiers get randomized opaque names.
SYNONYMS={'customers':'clients','customer':'client','orders':'purchases','order':'purchase','name':'full_name','amount':'total','id':'key','singer':'performer','age':'years','city':'town','student':'pupil','teacher':'instructor'}
def renamed(name,level,rng,used):
    words=re.sub(r'([a-z])([A-Z])',r'\1_\2',name).lower().split('_')
    if level=='easy': base='_'.join(SYNONYMS.get(w,w) for w in words)
    elif level=='medium': base='_'.join(SYNONYMS.get(w,w[:max(2,len(w)//2)]) for w in words)
    else: base='field_'+''.join(rng.choices('abcdefghijklmnopqrstuvwxyz',k=8))
    if base==name.lower(): base='alt_'+base
    candidate=base; i=2
    while candidate.lower() in used: candidate=f'{base}_{i}'; i+=1
    used.add(candidate.lower()); return candidate

def generate(source,target,out_mapping,level='easy',seed=42):
    if level not in ('easy','medium','hard'): raise ValueError('Unknown difficulty')
    source=Path(source).resolve(); target=Path(target).resolve()
    if source==target or target.exists(): raise ValueError('Target must be a new path, different from source')
    schema=extract_schema(source); rng=random.Random(seed)
    # Reserve all existing names so sequential ALTER operations never collide.
    used_tables={t.name.lower() for t in schema.tables}
    tm={t.name:renamed(t.name,level,rng,used_tables) for t in schema.tables}
    cm={}
    for t in schema.tables:
        used={c.name.lower() for c in t.columns}
        for c in t.columns: cm[f'{t.name}.{c.name}']=f'{tm[t.name]}.{renamed(c.name,level,rng,used)}'
    try:
        with sqlite3.connect(f'file:{source}?mode=ro',uri=True) as src, sqlite3.connect(target) as dst:
            src.backup(dst)
            # Keep FK definitions synchronized through ALTER TABLE operations.
            # Renaming with legacy behavior and FK checks disabled can leave a
            # child table referencing the old parent-table name.
            dst.execute('PRAGMA legacy_alter_table=OFF')
            dst.execute('PRAGMA foreign_keys=ON')
            with dst:
                for t in schema.tables:
                    dst.execute(f'ALTER TABLE {quote(t.name)} RENAME TO {quote(tm[t.name])}')
                for t in schema.tables:
                    for c in t.columns:
                        new=cm[f'{t.name}.{c.name}'].rsplit('.',1)[1]
                        dst.execute(f'ALTER TABLE {quote(tm[t.name])} RENAME COLUMN {quote(c.name)} TO {quote(new)}')
        result={'tables':tm,'columns':cm,'difficulty':level,'seed':seed}
        save_json(out_mapping,result)
        return result
    except Exception:
        target.unlink(missing_ok=True)
        raise

def make_pairs(source,target,gold):
    st,sc=texts(source); tt,tc=texts(target)
    tables={t.name:i for i,t in enumerate(target.tables)}
    src_cols=[(t.name,c.name) for t in source.tables for c in t.columns]
    tgt_cols={(t.name,c.name):i for i,t in enumerate(target.tables) for c in t.columns}
    # Rebuild column index globally.
    tgt_cols={(t.name,c.name):i for i,(t,c) in enumerate((t,c) for t in target.tables for c in t.columns)}
    return [{'anchor':st[i],'positive':tt[tables[gold['tables'][t.name]]]} for i,t in enumerate(source.tables)] + [{'anchor':sc[i],'positive':tc[tgt_cols[tuple(gold['columns'][f'{t}.{c}'].rsplit('.',1))]]} for i,(t,c) in enumerate(src_cols)]
