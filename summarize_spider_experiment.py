"""Summarize generated result files and changed-name column accuracy."""
import argparse,csv
from pathlib import Path
from handoff.interface import load_splits,load_pair,read_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--run-dir',required=True);p.add_argument('--split',default='test');a=p.parse_args()
    d=Path(a.run_dir); methods=['lexical','pretrained_semantic','finetuned_semantic','finetuned_types','finetuned_fk'];ids=load_splits(a.data)[a.split];rows=[]
    for method in methods:
        with (d/'results'/f'{method}_{a.split}'/'results.csv').open() as f: groups={r['group']:r for r in csv.DictReader(f)}
        r=groups['overall']; changed=correct=0
        for pid in ids:
            _,_,_,gold,_,_=load_pair(a.data,pid); pred=read_json(d/'predictions'/method/(pid+'.json'))['selected_mapping']['columns']
            for old,new in gold['columns'].items():
                if old.rsplit('.',1)[-1].casefold()!=new.rsplit('.',1)[-1].casefold(): changed+=1;correct+=pred.get(old)==new
        rows.append({'method':method,'split':a.split,'pairs':len(ids),'queries':r['n_queries'],'column_accuracy':r['column_accuracy'],'changed_column_accuracy':round(correct/changed,6) if changed else None,'changed_columns':changed,'column_top1_accuracy':r['column_top1_accuracy'],'column_top3_recall':r['column_top3_recall'],'execution_equivalence':r['execution_equivalence'],'join_execution_equivalence':groups.get('is_join=true',{}).get('execution_equivalence'),'gold_sql_errors':r['gold_sql_errors']})
    with (d/f'summary_{a.split}.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(d/f'summary_{a.split}.csv')
if __name__=='__main__':main()
