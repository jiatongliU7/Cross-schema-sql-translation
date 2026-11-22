"""Schema extraction, matching and FK-aware assignment for SQLite databases."""
from __future__ import annotations
import json, re, sqlite3
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

@dataclass
class Column:
    name: str
    type: str
    pk: bool = False

@dataclass
class Table:
    name: str
    columns: list[Column]

@dataclass
class Schema:
    tables: list[Table]
    # Each edge groups all column pairs belonging to one (possibly composite) FK.
    fks: list[list[tuple[str, str]]]
    def table(self, name):
        return next(t for t in self.tables if t.name == name)
    def to_dict(self): return asdict(self)
    @classmethod
    def from_dict(cls, d):
        return cls([Table(t['name'], [Column(**c) for c in t['columns']]) for t in d['tables']], d['fks'])

def quote(s): return '"' + s.replace('"', '""') + '"'

def extract_schema(path):
    with sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro', uri=True) as conn:
        names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        tables, edges = [], []
        for name in names:
            rows = conn.execute(f'PRAGMA table_info({quote(name)})').fetchall()
            tables.append(Table(name, [Column(r[1], r[2] or '', bool(r[5])) for r in rows]))
            groups = {}
            for r in conn.execute(f'PRAGMA foreign_key_list({quote(name)})'):
                # SQLite allows omitted referenced columns; resolve their primary-key order.
                parent_col = r[4]
                if parent_col is None:
                    pks = sorted((x for x in conn.execute(f'PRAGMA table_info({quote(r[2])})') if x[5]), key=lambda x:x[5])
                    if r[1] >= len(pks): raise ValueError(f'Cannot resolve FK in {name}')
                    parent_col = pks[r[1]][1]
                groups.setdefault(r[0], []).append((r[1], (f'{name}.{r[3]}', f'{r[2]}.{parent_col}')))
            edges.extend([[pair for _, pair in sorted(g)] for g in groups.values()])
    return Schema(tables, edges)

def affinity(t):
    t = t.upper()
    if 'INT' in t: return 'INTEGER'
    if any(x in t for x in ('CHAR','CLOB','TEXT')): return 'TEXT'
    if 'BLOB' in t or not t: return 'BLOB'
    if any(x in t for x in ('REAL','FLOA','DOUB')): return 'REAL'
    return 'NUMERIC'

def type_score(a,b):
    a,b=affinity(a),affinity(b)
    if a==b: return 1.0
    if a in ('INTEGER','REAL','NUMERIC') and b in ('INTEGER','REAL','NUMERIC'): return .5
    return 0.0

def normalize(s):
    return ' '.join(re.sub(r'([a-z])([A-Z])', r'\1 \2', s).lower().replace('_',' ').split())

def texts(schema):
    fk_cols = {a for edge in schema.fks for a,b in edge}
    tables = [f'Table: {normalize(t.name)}' for t in schema.tables]
    cols = [f'Table: {normalize(t.name)}; column: {normalize(c.name)}; type: {affinity(c.type)}; primary key: {c.pk}; foreign key: {t.name+"."+c.name in fk_cols}' for t in schema.tables for c in t.columns]
    return tables, cols

def semantic_matrices(src,tgt,method='lexical',model_name='all-MiniLM-L6-v2'):
    st,sc=texts(src); tt,tc=texts(tgt)
    if method == 'lexical':
        # Names are scored independently of metadata, preserving a meaningful lexical baseline.
        def sim(a,b): return SequenceMatcher(None, normalize(a), normalize(b)).ratio()
        ts=np.array([[sim(a.name,b.name) for b in tgt.tables] for a in src.tables],dtype=float)
        cs=np.array([[sim(a.name,b.name) for t2 in tgt.tables for b in t2.columns] for t1 in src.tables for a in t1.columns],dtype=float)
        return ts,cs
    if method != 'sbert': raise ValueError('method must be lexical or sbert')
    try: from sentence_transformers import SentenceTransformer
    except ImportError as e: raise RuntimeError('Install optional neural dependencies: pip install -r requirements-neural.txt') from e
    model=SentenceTransformer(model_name)
    def matrix(a,b):
        x=model.encode(a,normalize_embeddings=True,show_progress_bar=False)
        y=model.encode(b,normalize_embeddings=True,show_progress_bar=False)
        return np.asarray(x) @ np.asarray(y).T
    return matrix(st,tt),matrix(sc,tc)

def fk_consistency(mapping,src,tgt):
    if not src.fks: return None
    target={frozenset(tuple(p) for p in edge) for edge in tgt.fks}
    good=0
    for edge in src.fks:
        converted=[(mapping.get(a),mapping.get(b)) for a,b in edge]
        good += all(a is not None and b is not None for a,b in converted) and frozenset(converted) in target
    return good / len(src.fks)

def _assignment(scores):
    if scores.shape[0] != scores.shape[1]: raise ValueError('This project requires corresponding schemas with equal table/column counts')
    rows,cols=linear_sum_assignment(-scores)
    return dict(zip(map(int,rows),map(int,cols)))

def predict(src,tgt,method='lexical',model_name='all-MiniLM-L6-v2',type_weight=.15,fk_weight=.3,top_k=3):
    if not 0 <= type_weight <= 1 or not 0 <= fk_weight <= 1: raise ValueError('Weights must be between 0 and 1')
    if top_k < 1: raise ValueError('top_k must be positive')
    ts,cs=semantic_matrices(src,tgt,method,model_name)
    s_cols=[(t,c) for t in src.tables for c in t.columns]; t_cols=[(t,c) for t in tgt.tables for c in t.columns]
    si={(t.name,c.name):i for i,(t,c) in enumerate(s_cols)}; ti={(t.name,c.name):i for i,(t,c) in enumerate(t_cols)}
    typ=np.array([[type_score(a.type,b.type) for _,b in t_cols] for _,a in s_cols])
    cm=(1-type_weight)*cs+type_weight*typ
    def build(ta):
        tm={src.tables[i].name:tgt.tables[j].name for i,j in ta.items()}
        mapping={}
        for a,b in tm.items():
            ac=src.table(a).columns; bc=tgt.table(b).columns
            if len(ac)!=len(bc): return None
            sub=np.array([[cm[si[a,x.name],ti[b,y.name]] for y in bc] for x in ac])
            for i,j in _assignment(sub).items(): mapping[f'{a}.{ac[i].name}']=f'{b}.{bc[j].name}'
        return tm,mapping
    # Disallow impossible table pairs; account for best column assignment in table selection.
    table_scores=ts.copy()
    for i,a in enumerate(src.tables):
        for j,b in enumerate(tgt.tables):
            if len(a.columns)!=len(b.columns): table_scores[i,j]=-1e6; continue
            sub=np.array([[cm[si[a.name,x.name],ti[b.name,y.name]] for y in b.columns] for x in a.columns])
            ass=_assignment(sub)
            table_scores[i,j]=.5*ts[i,j]+.5*np.mean([sub[x,y] for x,y in ass.items()])
    ta=_assignment(table_scores)
    if any(table_scores[i,j] < -1e5 for i,j in ta.items()): raise ValueError('No one-to-one table assignment with compatible column counts')
    tm,mapping=build(ta)
    def score(tm,m):
        semt=np.mean([ts[next(i for i,t in enumerate(src.tables) if t.name==a),next(i for i,t in enumerate(tgt.tables) if t.name==b)] for a,b in tm.items()])
        semc=np.mean([cs[si[tuple(a.rsplit('.',1))],ti[tuple(b.rsplit('.',1))]] for a,b in m.items()])
        types=np.mean([typ[si[tuple(a.rsplit('.',1))],ti[tuple(b.rsplit('.',1))]] for a,b in m.items()])
        base=(1-type_weight)*(.5*semt+.5*semc)+type_weight*types
        fk=fk_consistency(m,src,tgt)
        return base if fk is None else (1-fk_weight)*base+fk_weight*fk
    # Coordinate ascent over column swaps and table swaps. Retains only strict improvements.
    if fk_weight and src.fks:
        while True:
            best=score(tm,mapping); choice=None
            for table in src.tables:
                keys=[f'{table.name}.{c.name}' for c in table.columns]
                for i,a in enumerate(keys):
                    for b in keys[i+1:]:
                        candidate=mapping.copy(); candidate[a],candidate[b]=candidate[b],candidate[a]
                        # Bound swaps by semantic/type retrieval rank within mapped table.
                        if any(sorted([cm[si[tuple(k.rsplit('.',1))],ti[(tm[table.name],c.name)]] for c in tgt.table(tm[table.name]).columns],reverse=True).index(cm[si[tuple(k.rsplit('.',1))],ti[tuple(candidate[k].rsplit('.',1))]]) >= top_k for k in (a,b)): continue
                        val=score(tm,candidate)
                        if val>best+1e-10: best=val; choice=(tm.copy(),candidate)
            for i in range(len(src.tables)):
                for j in range(i+1,len(src.tables)):
                    trial=ta.copy(); trial[i],trial[j]=trial[j],trial[i]
                    candidate=build(trial)
                    if candidate:
                        val=score(*candidate)
                        if val>best+1e-10: best=val; choice=(*candidate,trial)
            if not choice: break
            tm,mapping=choice[:2]
            if len(choice)==3: ta=choice[2]
    return {'tables':tm,'columns':mapping,'score':float(score(tm,mapping)),'fk_consistency':fk_consistency(mapping,src,tgt), 'candidates':{f'{t.name}.{c.name}':[{'column':f'{t_cols[j][0].name}.{t_cols[j][1].name}','score':float(cs[i,j])} for j in np.argsort(-cs[i])[:top_k]] for i,(t,c) in enumerate(s_cols)}}

def save_json(path,obj): Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def load_json(path): return json.loads(Path(path).read_text())
