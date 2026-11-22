"""Evaluate handoff predictions against gold mapping and target-side SQL."""
import argparse, csv, json, sqlite3
from collections import Counter
from pathlib import Path
import sqlglot
from sqlglot import exp
from handoff.interface import load_pair, load_splits, read_json, write_json
from sqltranslate.sql import execute


def ratio(num,den): return round(num/den,6) if den else None


def compare_target(db,gold_sql,pred_sql):
    try:
        tree=sqlglot.parse(gold_sql,read='sqlite')
        if len(tree)!=1 or not isinstance(tree[0],exp.Query): raise ValueError('gold query must be a single SELECT')
        gold=execute(db,gold_sql)
    except Exception as exc: return {'status':'gold_error','error':f'{type(exc).__name__}: {exc}'}
    if not pred_sql: return {'status':'missing_prediction','error':'No predicted SQL'}
    try:
        trees=sqlglot.parse(pred_sql,read='sqlite')
        if len(trees)!=1 or not isinstance(trees[0],exp.Query): raise ValueError('prediction must be a single SELECT')
        pred=execute(db,pred_sql)
        ordered=bool(tree[0].args.get('order'))
        same=(gold==pred) if ordered else (Counter(gold)==Counter(pred))
        return {'status':'correct' if same else 'incorrect','gold_rows':len(gold),'pred_rows':len(pred),'ordered':ordered}
    except Exception as exc: return {'status':'prediction_error','error':f'{type(exc).__name__}: {exc}'}


def run(data,pred_dir,split,method):
    ids=load_splits(data)[split]; pred_dir=Path(pred_dir); errors=[]; table_total=table_correct=col_total=col_correct=top_total=top1=top3=0; query_results=[]
    for pair_id in ids:
        d,_,_,gold,queries,_=load_pair(data,pair_id); path=pred_dir/(pair_id+'.json')
        prediction=read_json(path) if path.is_file() else {}
        selected=prediction.get('selected_mapping',{})
        for kind in ('tables','columns'):
            for source,target in gold[kind].items():
                guess=selected.get(kind,{}).get(source); candidates=prediction.get(kind,{}).get(source,[])
                if not isinstance(candidates,list): raise ValueError(f'{pair_id}: candidates for {source} must be a list')
                try: names=[x[0] for x in candidates if isinstance(x,list) and len(x)==2 and isinstance(x[0],str) and isinstance(x[1],(float,int))]
                except TypeError as exc: raise ValueError(f'Invalid candidate list for {source}') from exc
                if len(names)!=len(candidates): raise ValueError(f'{pair_id}: candidate items must be [name, score]')
                if kind=='tables': table_total+=1; table_correct+=guess==target
                else:
                    col_total+=1; col_correct+=guess==target; top_total+=1; top1+=target in names[:1]; top3+=target in names[:3]
                if guess!=target: errors.append({'pair_id':pair_id,'type':'mapping','element':source,'gold':target,'predicted':guess,'candidates':candidates})
        for q in queries:
            qid=q['qid']; sql=prediction.get('q_B',{}).get(qid)
            comparison=compare_target(d/'db_B.sqlite',q['gold_q_B'],sql)
            row={'pair_id':pair_id,'qid':qid,'level':q['level'],'is_join':q['is_join'],'source_sql':q['q_A'],'gold_sql':q['gold_q_B'],'pred_sql':sql,**comparison}
            query_results.append(row)
            if comparison['status']!='correct': errors.append({'type':'query',**row})
    mapping={'table_accuracy':ratio(table_correct,table_total),'column_accuracy':ratio(col_correct,col_total),'column_top1_accuracy':ratio(top1,top_total),'column_top3_recall':ratio(top3,top_total),'n_tables':table_total,'n_columns':col_total}
    def group(label,rows,include_mapping=False):
        n=len(rows); counts=Counter(x['status'] for x in rows); eligible=n-counts['gold_error']
        return {'method':method,'group':label,'n_queries':n,'gold_sql_errors':counts['gold_error'],'pred_sql_executable_rate':ratio(counts['correct']+counts['incorrect'],eligible),'execution_equivalence':ratio(counts['correct'],eligible),'missing_sql':counts['missing_prediction'],'prediction_sql_errors':counts['prediction_error'],**(mapping if include_mapping else {k:None for k in mapping})}
    results=[group('overall',query_results,True)]
    for level in sorted({x['level'] for x in query_results}): results.append(group('level='+level,[x for x in query_results if x['level']==level]))
    for is_join in (False,True):
        subset=[x for x in query_results if x['is_join']==is_join]
        if subset: results.append(group('is_join='+str(is_join).lower(),subset))
    return results, errors, query_results


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--data',required=True); p.add_argument('--pred',required=True); p.add_argument('--split',choices=['train','dev','test'],default='test'); p.add_argument('--method'); p.add_argument('--outdir',default='results'); a=p.parse_args()
    rows,errors,queries=run(a.data,a.pred,a.split,a.method or Path(a.pred).name)
    out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    with (out/'results.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(out/'error_analysis.json',errors); write_json(out/'query_results.json',queries)
    print(json.dumps({'results':str(out/'results.csv'),'error_analysis':str(out/'error_analysis.json'),'queries':len(queries),'errors':len(errors)},indent=2))
if __name__=='__main__': main()
