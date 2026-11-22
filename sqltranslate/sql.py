"""Scope-aware SQL rewriting and execution evaluation."""
from collections import Counter
import sqlite3
from pathlib import Path
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope, Scope

def rewrite_sql(sql, schema, mapping):
    expressions=sqlglot.parse(sql,read='sqlite')
    if len(expressions)!=1 or not isinstance(expressions[0], exp.Query): raise ValueError('Only a single SELECT query is supported')
    raw=expressions[0]
    if any(str(j.args.get('method') or '').upper() == 'NATURAL' for j in raw.find_all(exp.Join)): raise ValueError('NATURAL JOIN is unsupported')
    # Qualify resolves unqualified columns, expands stars and creates stable output aliases.
    definitions={t.name:{c.name: c.type or 'TEXT' for c in t.columns} for t in schema.tables}
    tree=qualify(raw,dialect='sqlite',schema=definitions,expand_stars=True,validate_qualify_columns=True,quote_identifiers=True)
    for scope in traverse_scope(tree):
        # Correlated references are resolved against parent scopes; fail closed until supported.
        if scope.external_columns: raise ValueError('Correlated or unresolved columns are unsupported')
        sources=scope.sources
        for col in scope.columns:
            source=sources.get(col.table)
            if isinstance(source, exp.Table):
                key=f'{source.name}.{col.name}'
                if key not in mapping['columns']: raise ValueError(f'Missing column mapping: {key}')
                dst=mapping['columns'][key].rsplit('.',1)[-1]
                col.set('this',exp.to_identifier(dst,quoted=True))
        for alias,source in sources.items():
            if isinstance(source,exp.Table):
                if source.name not in mapping['tables']: raise ValueError(f'Missing table mapping: {source.name}')
                source.set('this',exp.to_identifier(mapping['tables'][source.name],quoted=True))
    return tree.sql(dialect='sqlite')

def execute(path,sql):
    with sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro',uri=True) as conn:
        conn.execute('PRAGMA query_only=ON')
        return conn.execute(sql).fetchall()

def evaluate(source_db,target_db,source_sql,target_sql):
    try:
        a=execute(source_db,source_sql); b=execute(target_db,target_sql)
        query=sqlglot.parse_one(source_sql,read='sqlite')
        ordered=query.args.get('order') is not None
        # SQLite result equality is exact; no float tolerance is silently applied.
        equivalent=(a==b if ordered else Counter(a)==Counter(b))
        return {'execution_success':True,'equivalent':equivalent,'ordered':ordered,'source_rows':len(a),'target_rows':len(b),'error':None}
    except (sqlite3.Error,ValueError) as e:
        return {'execution_success':False,'equivalent':False,'error':str(e)}
