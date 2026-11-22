"""Generate handoff predictions for a split of schema pairs."""
import argparse
from pathlib import Path
from handoff.interface import load_pair, load_splits, write_json
from matcher import build_report
from sqltranslate.core import semantic_matrices, predict
from sqltranslate.sql import rewrite_sql


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--data',required=True); p.add_argument('--pred',required=True); p.add_argument('--split',choices=['train','dev','test'],default='test'); p.add_argument('--method',choices=['lexical','sbert'],default='lexical'); p.add_argument('--model',default='all-MiniLM-L6-v2'); p.add_argument('--top-k',type=int,default=3); p.add_argument('--type-weight',type=float,default=0); p.add_argument('--fk-weight',type=float,default=0); args=p.parse_args()
    for pair_id in load_splits(args.data)[args.split]:
        _,a,b,_,queries,_=load_pair(args.data,pair_id)
        ts,cs=semantic_matrices(a,b,args.method,args.model)
        ranked=build_report(a,b,ts,cs,args.top_k)
        chosen=predict(a,b,args.method,args.model,args.type_weight,args.fk_weight,args.top_k)
        mapping={'tables':chosen['tables'],'columns':chosen['columns']}
        output={kind:{k:[[v['target'],v['score']] for v in vals] for k,vals in ranked[kind]['top_k'].items()} for kind in ('tables','columns')}
        output.update({'selected_mapping':mapping,'q_B':{},'rewrite_errors':{}})
        for q in queries:
            try: output['q_B'][q['qid']]=rewrite_sql(q['q_A'],a,mapping)
            except Exception as exc: output['rewrite_errors'][q['qid']]=f'{type(exc).__name__}: {exc}'
        path=Path(args.pred)/(pair_id+'.json'); write_json(path,output); print(path)
if __name__=='__main__': main()
