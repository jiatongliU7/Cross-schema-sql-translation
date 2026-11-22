"""Run full handoff dataset experiments while caching schema embeddings."""
import argparse, json, traceback
from pathlib import Path
import numpy as np
from handoff.interface import load_pair, load_splits, write_json
from sqltranslate.core import texts
from sqltranslate.sql import rewrite_sql
import sqltranslate.core as core
from matcher import build_report


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--outdir',required=True); p.add_argument('--model',required=True); p.add_argument('--methods',nargs='+',choices=['semantic','types','fk'],default=['semantic','types','fk']); p.add_argument('--label',required=True); p.add_argument('--top-k',type=int,default=3); p.add_argument('--batch-size',type=int,default=64); a=p.parse_args()
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    splits=load_splits(a.data); allpairs=[]; documents=set()
    for split in ('train','dev','test'):
        for pair_id in splits[split]:
            _,source,target,_,queries,_=load_pair(a.data,pair_id)
            src=texts(source); tgt=texts(target)
            for s in (*src,*tgt): documents.update(s)
            allpairs.append((split,pair_id,source,target,queries,src,tgt))
    print('schema pairs:',len(allpairs),'unique descriptions:',len(documents),flush=True)
    model=SentenceTransformer(a.model,device='cpu'); ordered=sorted(documents)
    vec=model.encode(ordered,batch_size=a.batch_size,normalize_embeddings=True,show_progress_bar=True)
    index={s:i for i,s in enumerate(ordered)}
    def matrix(a,b): return vec[[index[x] for x in a]] @ vec[[index[x] for x in b]].T
    errors=[]
    for n,(split,pair_id,source,target,queries,src,tgt) in enumerate(allpairs,1):
        ts,cs=matrix(src[0],tgt[0]),matrix(src[1],tgt[1]); ranked=build_report(source,target,ts,cs,a.top_k)
        core.semantic_matrices=lambda *args,**kwargs:(ts,cs)
        for method in a.methods:
            try:
                type_weight=0 if method=='semantic' else .15; fk_weight=.3 if method=='fk' else 0
                chosen=core.predict(source,target,'sbert',a.model,type_weight,fk_weight,a.top_k)
                mapping={'tables':chosen['tables'],'columns':chosen['columns']}
                result={kind:{key:[[v['target'],v['score']] for v in vals] for key,vals in ranked[kind]['top_k'].items()} for kind in ('tables','columns')}
                result.update({'selected_mapping':mapping,'q_B':{},'rewrite_errors':{},'model':a.model,'method':method})
                for q in queries:
                    try: result['q_B'][q['qid']]=rewrite_sql(q['q_A'],source,mapping)
                    except Exception as exc: result['rewrite_errors'][q['qid']]=f'{type(exc).__name__}: {exc}'
                write_json(Path(a.outdir)/'predictions'/f'{a.label}_{method}'/(pair_id+'.json'),result)
            except Exception as exc:
                errors.append({'pair_id':pair_id,'split':split,'method':method,'error':f'{type(exc).__name__}: {exc}'})
        if n%20==0 or n==len(allpairs): print(f'completed {n}/{len(allpairs)}; failures {len(errors)}',flush=True)
    write_json(Path(a.outdir)/(a.label+'_failures.json'),errors)
    print('done',len(errors),'failures',flush=True)

if __name__=='__main__': main()
