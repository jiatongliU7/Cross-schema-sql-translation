"""Spider and Dr.Spider preparation with conservative mapping/data audits."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import sqlite3
import sqlglot
from .core import extract_schema, load_json, save_json, quote, semantic_matrices, predict, fk_consistency
from .data import generate
from .sql import rewrite_sql, evaluate, execute


def _record(path, dbid):
    matches=[x for x in load_json(path) if x['db_id']==dbid]
    if len(matches)!=1: raise ValueError(f'{path}: expected exactly one entry for {dbid}, got {len(matches)}')
    return matches[0]


def _schema_layout(record):
    tables=record['table_names_original']; columns=record['column_names_original']
    if len(set(t.lower() for t in tables))!=len(tables): raise ValueError('Duplicate table names')
    if any('.' in name for name in tables): raise ValueError('Identifiers with dots are unsupported')
    seen=set()
    for ti,name in columns:
        if ti == -1:
            if name != '*': raise ValueError('Unexpected global column')
            continue
        if not isinstance(ti,int) or ti<0 or ti>=len(tables) or '.' in name: raise ValueError('Invalid column or table index')
        key=(ti,name.lower())
        if key in seen: raise ValueError('Duplicate column names')
        seen.add(key)
    return tables,columns


def derive_dr_mapping(pre_record, post_record, source_db, target_db):
    """Propose index-aligned mapping; audits are evidence, not ground truth proof."""
    a,ac=_schema_layout(pre_record); b,bc=_schema_layout(post_record)
    if len(a)!=len(b) or len(ac)!=len(bc): raise ValueError('Schema sizes differ; no index-aligned mapping')
    if any(x[0]!=y[0] for x,y in zip(ac,bc)): raise ValueError('Column table indices differ; cannot align positions')
    for field in ('column_types','primary_keys','foreign_keys'):
        if pre_record.get(field)!=post_record.get(field): raise ValueError(f'Metadata {field} differs; manual review required')
    tables=dict(zip(a,b)); columns={f'{a[i]}.{c}':f'{b[j]}.{d}' for (i,c),(j,d) in zip(ac,bc) if i>=0}
    source=extract_schema(source_db); target=extract_schema(target_db)
    src={t.name:{c.name for c in t.columns} for t in source.tables}; tgt={t.name:{c.name for c in t.columns} for t in target.tables}
    if set(src)!=set(a) or set(tgt)!=set(b): raise ValueError('Metadata table names differ from SQLite schema')
    for name in a:
        expected={c for i,c in ac if i>=0 and a[i]==name}
        postname=tables[name]; expected_post={c for i,c in bc if i>=0 and b[i]==postname}
        if src[name]!=expected or tgt[postname]!=expected_post: raise ValueError(f'Metadata columns differ from SQLite schema for {name}')
    issues=[]; equal_tables=0
    # Compare multisets of complete mapped rows, using bound metadata identifiers, not position order.
    with sqlite3.connect(f'file:{Path(source_db).resolve()}?mode=ro',uri=True) as ca, sqlite3.connect(f'file:{Path(target_db).resolve()}?mode=ro',uri=True) as cb:
        for name in a:
            old=[c for i,c in ac if i>=0 and a[i]==name]
            new=[columns[f'{name}.{c}'].rsplit('.',1)[1] for c in old]
            def rows(conn,table,cols):
                return Counter(conn.execute('SELECT '+','.join(map(quote,cols))+' FROM '+quote(table)))
            if rows(ca,name,old)==rows(cb,tables[name],new): equal_tables+=1
            else: issues.append(f'row contents differ for table {name}')
    fk=fk_consistency(columns,source,target)
    if fk is not None and fk!=1: issues.append('SQLite FK edges differ under candidate mapping')
    # Empty/equal columns or graph symmetries cannot verify positional alignments.
    return {'tables':tables,'columns':columns,'provenance':'index-aligned candidate; requires human review','audit':{'table_contents_equal':equal_tables==len(a),'equal_tables':equal_tables,'total_tables':len(a),'sqlite_fk_consistency':fk,'issues':issues,'warning':'Equal rows and metadata do not prove semantic correspondence; symmetric or empty columns may be ambiguous.'}}


def prepare_spider(db, questions, outdir, difficulty='easy', seed=42):
    db=Path(db).resolve(); out=Path(outdir).resolve(); out.mkdir(parents=True,exist_ok=True)
    source_id=db.stem
    rows=[r['query'] for r in load_json(questions) if r['db_id']==source_id]
    if not rows: raise ValueError(f'No queries found for {source_id}')
    target=out/'target.sqlite'; gold=out/'gold.json'
    generate(db,target,gold,difficulty,seed)
    save_json(out/'queries.json',rows)
    manifest={'dataset':'Spider synthetic renaming','db_id':source_id,'source':str(db),'target':str(target),'mapping':str(gold),'queries':str(out/'queries.json'),'difficulty':difficulty,'seed':seed,'data_correspondence':'SQLite backup plus renaming; validate before evaluation'}
    save_json(out/'manifest.json',manifest)
    return manifest


def prepare_dr(pre_tables, post_tables, source_db, target_db, dbid, outdir, pre_questions=None, post_questions=None):
    out=Path(outdir).resolve(); out.mkdir(parents=True,exist_ok=True)
    mapping=derive_dr_mapping(_record(pre_tables,dbid),_record(post_tables,dbid),source_db,target_db)
    save_json(out/'candidate_mapping.json',mapping)
    manifest={'dataset':'Dr.Spider','db_id':dbid,'source':str(Path(source_db).resolve()),'target':str(Path(target_db).resolve()),'candidate_mapping':str(out/'candidate_mapping.json'),'pre_tables':str(Path(pre_tables).resolve()),'post_tables':str(Path(post_tables).resolve()),'requires_manual_review':True,'audit':mapping['audit']}
    if bool(pre_questions)!=bool(post_questions): raise ValueError('Provide both pre and post question files')
    if pre_questions:
        pre=load_json(pre_questions); post=load_json(post_questions)
        # q_id_spider_dev establishes pairing, avoiding assumptions about array order.
        def by_id(items):
            result={}
            for row in items:
                if row.get('db_id')!=dbid: continue
                k=row.get('q_id_spider_dev')
                if k is None or k in result: raise ValueError('Missing/duplicate q_id_spider_dev; pair manually')
                if not isinstance(row.get('query'),str): raise ValueError('Query field missing; inspect dataset format')
                result[k]=row
            return result
        p,q=by_id(pre),by_id(post)
        if set(p)!=set(q): raise ValueError('Pre/post question IDs differ')
        paired=[{'id':k,'source_sql':p[k]['query'],'target_gold_sql':q[k]['query']} for k in sorted(p)]
        save_json(out/'paired_queries.json',paired); manifest['paired_queries']=str(out/'paired_queries.json'); manifest['query_count']=len(paired)
    save_json(out/'manifest.json',manifest); return manifest


def retrieval_metrics(source,target,gold,method='lexical',model='all-MiniLM-L6-v2',k=3):
    if k<1: raise ValueError('k must be positive')
    ts,cs=semantic_matrices(source,target,method,model)
    targets=[f'{t.name}.{c.name}' for t in target.tables for c in t.columns]
    sources=[f'{t.name}.{c.name}' for t in source.tables for c in t.columns]
    details=[]
    for i,key in enumerate(sources):
        if key not in gold['columns']: raise ValueError(f'Missing gold column {key}')
        candidates=sorted(range(len(targets)),key=lambda j:-cs[i,j]); truth=gold['columns'][key]
        if truth not in targets: raise ValueError(f'Unknown target {truth}')
        details.append({'source':key,'gold':truth,'top_candidates':[{'target':targets[j],'score':float(cs[i,j])} for j in candidates[:k]],'rank':candidates.index(targets.index(truth))+1})
    n=len(details)
    return {'method':method,'model':model if method=='sbert' else None,'k':k,'n_columns':n,'top1_accuracy':sum(x['rank']==1 for x in details)/n if n else None,'topk_recall':sum(x['rank']<=k for x in details)/n if n else None,'details':details}


def evaluate_paired(source_db,target_db,mapping,rows):
    """Compare generated SQL to provided target SQL on target DB, separately from cross-DB comparison."""
    source=extract_schema(source_db); results=[]
    for row in rows:
        item={'id':row['id'],'source_sql':row['source_sql'],'target_gold_sql':row['target_gold_sql']}
        try:
            pred=rewrite_sql(row['source_sql'],source,mapping); item['target_sql']=pred
            item['cross_database']=evaluate(source_db,target_db,row['source_sql'],pred)
            # Same-database comparison remains meaningful even when source/target data differ.
            item['target_gold_comparison']=evaluate(target_db,target_db,row['target_gold_sql'],pred)
        except Exception as e: item.update({'target_sql':None,'error':f'{type(e).__name__}: {e}'})
        results.append(item)
    n=len(results)
    return {'n_queries':n,'target_gold_execution_equivalence':sum(x.get('target_gold_comparison',{}).get('equivalent',False) for x in results)/n if n else None,'cross_database_execution_equivalence':sum(x.get('cross_database',{}).get('equivalent',False) for x in results)/n if n else None,'results':results,'note':'Cross-database comparisons require verified equivalent data; official post queries may contain errors.'}
